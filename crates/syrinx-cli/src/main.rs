//! syrinx-cli — the `syrinx` command-line surface.
//!
//! Real work is behind the `real` feature. Three commands wire the CosyVoice2
//! [`Synthesizer`](syrinx_serve::synth::Synthesizer) into a usable front door:
//!   * `synth`  — render text in a reference voice to a WAV (with optional prosody);
//!   * `serve`  — boot the OpenAI-compatible audio server in that voice;
//!   * `stream` — chunk-streaming synthesis (low first-chunk latency path).
//! The default build is Candle-free and only prints how to enable the real path.

#[cfg(not(feature = "real"))]
fn main() {
    eprintln!(
        "syrinx: built without the `real` feature — the synthesizer is not compiled in.\n\
         \n\
         Rebuild with the model-backed feature to enable the commands:\n\
         \n    cargo build -p syrinx-cli --features real        # CPU (parity device)\
         \n    cargo build -p syrinx-cli --features real,tn     # + text normalization\
         \n    cargo build -p syrinx-cli --features cuda        # GPU (speed)\n\
         \n\
         Commands: synth | serve | stream   (run `syrinx <cmd> --help`)\n"
    );
    std::process::exit(2);
}

#[cfg(feature = "real")]
fn main() {
    std::process::exit(real::run());
}

#[cfg(feature = "real")]
mod real {
    use std::net::SocketAddr;
    use std::path::PathBuf;

    use syrinx_prosody::render_plan::RenderPlan;
    use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};
    use syrinx_serve::synth_cv3::{Cv3SynthConfig, Cv3SynthInputs, Cv3Synthesizer};
    use syrinx_serve::{wavio, Cv3RealSynth, RealSynth};

    const USAGE: &str = "\
syrinx — local CosyVoice2 TTS + zero-shot voice cloning (pure Rust)

USAGE:
    syrinx <COMMAND> [OPTIONS]

COMMANDS:
    synth    Render text in a reference voice to a WAV file.
    serve    Boot the OpenAI-compatible audio server (POST /v1/audio/speech).
    stream   Chunk-streaming synthesis to a WAV (low first-chunk latency path).

Run `syrinx <COMMAND> --help` for command-specific options.

COMMON OPTIONS (all commands):
    --prompt-text <TEXT>   Transcript of the reference clip.            [required]
    --ref-wav <WAV>        Reference voice clip (any rate; resampled).  [required]
    --model-dir <DIR>      Directory of sub-model files (or per-model env vars).
    --max-steps <N>        Cap on live LM generation steps (CPU tractability).
    --cuda                 Run on GPU (requires a --features cuda build).
    --cv3                  Drive the CosyVoice3 synthesizer (synth + serve only)
                           using the SYRINX_CV3_* model files instead of CV2.

MODEL FILES (env var overrides --model-dir/<default>):
    SYRINX_LM_WEIGHTS llm_fp32.safetensors    SYRINX_SPK_WEIGHTS campplus_weights.safetensors
    SYRINX_FLOW_WEIGHTS flow_fp32.safetensors SYRINX_HIFT_WEIGHTS hift_fp32.safetensors
    SYRINX_TOK_JSON tokenizer.json            SYRINX_STOK_ONNX speech_tokenizer_v2.onnx

CV3 MODEL FILES (with --cv3; env var overrides --model-dir/<default>):
    SYRINX_CV3_LM_WEIGHTS llm_fp32.safetensors    SYRINX_CV3_SPK_WEIGHTS campplus_weights.safetensors
    SYRINX_CV3_FLOW_WEIGHTS flow_fp32.safetensors SYRINX_CV3_HIFT_WEIGHTS hift_fp32.safetensors
    SYRINX_CV3_TOK_JSON tokenizer.json            SYRINX_CV3_STOK_ONNX speech_tokenizer_v3.onnx
";

    const SYNTH_USAGE: &str = "\
syrinx synth — render text in a reference voice

    syrinx synth --text <TEXT> --prompt-text <TEXT> --ref-wav <WAV> --out <WAV>
                 [--pitch <SEMITONES>] [--rate <R>] [--plan <FILE.json>]
                 [--model-dir <DIR>] [--max-steps <N>] [--cuda] [--cv3]

    --text <TEXT>          Text to speak.                              [required]
    --out <WAV>            Output 24 kHz mono 16-bit WAV.              [required]
    --rate <R>             Global speech-rate (>1 faster, <1 slower; faithful).
    --pitch <SEMITONES>    Global pitch shift (NOTE: weak training-free lever).
    --plan <FILE.json>     Load a full RenderPlan (overrides --pitch/--rate).
    --cv3                  Use the CosyVoice3 synthesizer (SYRINX_CV3_* files);
                           prosody plan/pitch/rate are CV2-only and are ignored.
";

    const SERVE_USAGE: &str = "\
syrinx serve — boot the OpenAI-compatible audio server in a reference voice

    syrinx serve --prompt-text <TEXT> --ref-wav <WAV> [--port <N>]
                 [--model-dir <DIR>] [--max-steps <N>] [--cuda] [--cv3]

    --port <N>             Listen port (default 8080); binds 127.0.0.1.
    --cv3                  Serve the CosyVoice3 synthesizer (SYRINX_CV3_* files).
    Then POST /v1/audio/speech  {\"model\":\"syrinx\",\"input\":\"<text>\",\"voice\":\"v\"}
";

    const STREAM_USAGE: &str = "\
syrinx stream — chunk-streaming synthesis to a WAV

    syrinx stream --text <TEXT> --prompt-text <TEXT> --ref-wav <WAV> --out <WAV>
                  [--hop <TOKENS>] [--model-dir <DIR>] [--max-steps <N>] [--cuda]

    --hop <TOKENS>         Finalized speech tokens per emitted chunk (default 25).
";

    #[derive(Default)]
    struct Opts {
        text: Option<String>,
        prompt_text: Option<String>,
        ref_wav: Option<PathBuf>,
        out: Option<PathBuf>,
        model_dir: Option<PathBuf>,
        max_steps: Option<usize>,
        cuda: bool,
        pitch: Option<f64>,
        rate: Option<f64>,
        plan: Option<PathBuf>,
        port: Option<u16>,
        hop: Option<usize>,
        cv3: bool,
    }

    pub fn run() -> i32 {
        let mut argv = std::env::args().skip(1);
        let cmd = match argv.next() {
            Some(c) => c,
            None => {
                eprintln!("syrinx: missing command\n\n{USAGE}");
                return 2;
            }
        };
        let (usage, dispatch): (&str, fn(Opts) -> Result<(), String>) = match cmd.as_str() {
            "synth" => (SYNTH_USAGE, cmd_synth),
            "serve" => (SERVE_USAGE, cmd_serve),
            "stream" => (STREAM_USAGE, cmd_stream),
            "-h" | "--help" | "help" => {
                println!("{USAGE}");
                return 0;
            }
            other => {
                eprintln!("syrinx: unknown command `{other}`\n\n{USAGE}");
                return 2;
            }
        };
        let opts = match parse_opts(argv, usage) {
            Ok(o) => o,
            Err(msg) => {
                eprintln!("syrinx {cmd}: {msg}\n\n{usage}");
                return 2;
            }
        };
        match dispatch(opts) {
            Ok(()) => 0,
            Err(msg) => {
                eprintln!("syrinx {cmd}: {msg}");
                1
            }
        }
    }

    fn parse_opts(mut argv: impl Iterator<Item = String>, usage: &str) -> Result<Opts, String> {
        let mut o = Opts::default();
        let need = |argv: &mut dyn Iterator<Item = String>, flag: &str| -> Result<String, String> {
            argv.next().ok_or_else(|| format!("`{flag}` expects a value"))
        };
        let num = |s: String, flag: &str| -> Result<f64, String> {
            s.parse::<f64>()
                .map_err(|_| format!("{flag} expects a number, got `{s}`"))
        };
        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--text" => o.text = Some(need(&mut argv, "--text")?),
                "--prompt-text" => o.prompt_text = Some(need(&mut argv, "--prompt-text")?),
                "--ref-wav" => o.ref_wav = Some(PathBuf::from(need(&mut argv, "--ref-wav")?)),
                "--out" => o.out = Some(PathBuf::from(need(&mut argv, "--out")?)),
                "--model-dir" => o.model_dir = Some(PathBuf::from(need(&mut argv, "--model-dir")?)),
                "--max-steps" => {
                    let v = need(&mut argv, "--max-steps")?;
                    o.max_steps = Some(
                        v.parse()
                            .map_err(|_| format!("--max-steps expects an integer, got `{v}`"))?,
                    );
                }
                "--pitch" => o.pitch = Some(num(need(&mut argv, "--pitch")?, "--pitch")?),
                "--rate" => o.rate = Some(num(need(&mut argv, "--rate")?, "--rate")?),
                "--plan" => o.plan = Some(PathBuf::from(need(&mut argv, "--plan")?)),
                "--port" => {
                    let v = need(&mut argv, "--port")?;
                    o.port = Some(
                        v.parse()
                            .map_err(|_| format!("--port expects a port number, got `{v}`"))?,
                    );
                }
                "--hop" => {
                    let v = need(&mut argv, "--hop")?;
                    o.hop = Some(
                        v.parse()
                            .map_err(|_| format!("--hop expects an integer, got `{v}`"))?,
                    );
                }
                "--cuda" => o.cuda = true,
                "--cv3" => o.cv3 = true,
                "-h" | "--help" => {
                    println!("{usage}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }
        Ok(o)
    }

    /// Resolve a sub-model path: the `env_var` if set, else `<model_dir>/<default>`,
    /// else an error naming both ways to supply it.
    fn resolve_path(
        env_var: &str,
        default_name: &str,
        model_dir: &Option<PathBuf>,
    ) -> Result<String, String> {
        if let Ok(p) = std::env::var(env_var) {
            if !p.is_empty() {
                return Ok(p);
            }
        }
        if let Some(dir) = model_dir {
            return Ok(dir.join(default_name).to_string_lossy().into_owned());
        }
        Err(format!(
            "{env_var} is unset and no --model-dir given (need {default_name})"
        ))
    }

    fn build_config(model_dir: &Option<PathBuf>) -> Result<SynthConfig, String> {
        Ok(SynthConfig {
            lm_weights: resolve_path("SYRINX_LM_WEIGHTS", "llm_fp32.safetensors", model_dir)?,
            spk_weights: resolve_path(
                "SYRINX_SPK_WEIGHTS",
                "campplus_weights.safetensors",
                model_dir,
            )?,
            flow_weights: resolve_path("SYRINX_FLOW_WEIGHTS", "flow_fp32.safetensors", model_dir)?,
            hift_weights: resolve_path("SYRINX_HIFT_WEIGHTS", "hift_fp32.safetensors", model_dir)?,
            tokenizer_json: resolve_path("SYRINX_TOK_JSON", "tokenizer.json", model_dir)?,
            speech_tokenizer_onnx: resolve_path(
                "SYRINX_STOK_ONNX",
                "speech_tokenizer_v2.onnx",
                model_dir,
            )?,
        })
    }

    /// Load every sub-model. `--cuda` picks a GPU device when this binary was built
    /// `--features cuda`; otherwise it transparently runs on CPU.
    fn load_synth(o: &Opts) -> Result<Synthesizer, String> {
        let cfg = build_config(&o.model_dir)?;
        if o.cuda {
            #[cfg(not(feature = "cuda"))]
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
            let dev = syrinx_serve::synth::pick_device(None);
            Synthesizer::load_on_device(&cfg, dev).map_err(|e| e.to_string())
        } else {
            Synthesizer::load(&cfg).map_err(|e| e.to_string())
        }
    }

    /// CV3 model-file config, mirroring [`build_config`] but reading the `SYRINX_CV3_*`
    /// env vars (falling back to `--model-dir/<default>`). The only default-name delta
    /// from CV2 is the v3 speech tokenizer (`speech_tokenizer_v3.onnx`).
    fn build_cv3_config(model_dir: &Option<PathBuf>) -> Result<Cv3SynthConfig, String> {
        Ok(Cv3SynthConfig {
            lm_weights: resolve_path("SYRINX_CV3_LM_WEIGHTS", "llm_fp32.safetensors", model_dir)?,
            spk_weights: resolve_path(
                "SYRINX_CV3_SPK_WEIGHTS",
                "campplus_weights.safetensors",
                model_dir,
            )?,
            flow_weights: resolve_path(
                "SYRINX_CV3_FLOW_WEIGHTS",
                "flow_fp32.safetensors",
                model_dir,
            )?,
            hift_weights: resolve_path(
                "SYRINX_CV3_HIFT_WEIGHTS",
                "hift_fp32.safetensors",
                model_dir,
            )?,
            tokenizer_json: resolve_path("SYRINX_CV3_TOK_JSON", "tokenizer.json", model_dir)?,
            speech_tokenizer_onnx: resolve_path(
                "SYRINX_CV3_STOK_ONNX",
                "speech_tokenizer_v3.onnx",
                model_dir,
            )?,
        })
    }

    /// Load every CV3 sub-model. `--cuda` behaves exactly as in [`load_synth`]: it
    /// picks a GPU device on a `--features cuda` build, else transparently CPU.
    fn load_cv3_synth(o: &Opts) -> Result<Cv3Synthesizer, String> {
        let cfg = build_cv3_config(&o.model_dir)?;
        if o.cuda {
            #[cfg(not(feature = "cuda"))]
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
            let dev = syrinx_serve::synth::pick_device(None);
            Cv3Synthesizer::load_on_device(&cfg, dev).map_err(|e| e.to_string())
        } else {
            Cv3Synthesizer::load(&cfg).map_err(|e| e.to_string())
        }
    }

    /// Read the reference voice: `--prompt-text` + the `--ref-wav` resampled to the
    /// 16 kHz + 24 kHz mono buffers the synthesizer expects.
    fn read_voice(o: &Opts) -> Result<(String, Vec<f32>, Vec<f32>), String> {
        let prompt_text = o.prompt_text.clone().ok_or("missing required --prompt-text")?;
        let ref_wav = o.ref_wav.as_ref().ok_or("missing required --ref-wav")?;
        let (r16, r24) = wavio::read_ref_wav(ref_wav).map_err(|e| e.to_string())?;
        Ok((prompt_text, r16, r24))
    }

    /// Build the editable prosody plan from `--plan` (full JSON) or `--pitch`/`--rate`
    /// (global knobs). `None` means render with no plan (plain `synthesize`).
    fn build_plan(o: &Opts) -> Result<Option<RenderPlan>, String> {
        if let Some(path) = &o.plan {
            let raw = std::fs::read_to_string(path).map_err(|e| format!("read --plan: {e}"))?;
            let plan: RenderPlan =
                serde_json::from_str(&raw).map_err(|e| format!("parse --plan: {e}"))?;
            return Ok(Some(plan));
        }
        if o.pitch.is_some() || o.rate.is_some() {
            let mut plan = RenderPlan::identity();
            if let Some(r) = o.rate {
                plan = plan.with_global_rate(r);
            }
            if let Some(p) = o.pitch {
                plan = plan.with_global_pitch_semitones(p);
            }
            return Ok(Some(plan));
        }
        Ok(None)
    }

    /// CV3 render path for `synth --cv3`: load the CV3 synthesizer and render
    /// `tts_text` in the reference voice to a 24 kHz WAV. The editable prosody plan
    /// (`--pitch/--rate/--plan`) is a CV2-only lever, so it is not applied here.
    fn cmd_synth_cv3(o: &Opts, text: &str, out: &std::path::Path) -> Result<(), String> {
        let (prompt_text, r16, r24) = read_voice(o)?;
        if o.plan.is_some() || o.pitch.is_some() || o.rate.is_some() {
            eprintln!(
                "syrinx synth --cv3: prosody plan/pitch/rate are CV2-only and are ignored \
                 on the CV3 path"
            );
        }
        let mut synth = load_cv3_synth(o)?;
        let inputs = Cv3SynthInputs {
            lm_seed: 0,
            max_gen_steps: o.max_steps,
            ..Default::default()
        };
        let wav = synth
            .synthesize(text, &prompt_text, &r16, &r24, &inputs)
            .map_err(|e| e.to_string())?;
        wavio::write_wav_24k(out, &wav).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({} samples, 24 kHz mono, CV3)",
            out.display(),
            wav.len()
        );
        Ok(())
    }

    fn cmd_synth(o: Opts) -> Result<(), String> {
        let text = o.text.clone().ok_or("missing required --text")?;
        let out = o.out.clone().ok_or("missing required --out")?;
        if o.cv3 {
            return cmd_synth_cv3(&o, &text, &out);
        }
        let (prompt_text, r16, r24) = read_voice(&o)?;
        let plan = build_plan(&o)?;
        let mut synth = load_synth(&o)?;
        let inputs = SynthInputs {
            lm_seed: 0,
            max_gen_steps: o.max_steps,
            ..Default::default()
        };
        let wav = match plan {
            Some(plan) => {
                synth.synthesize_with_plan(&text, &prompt_text, &r16, &r24, &inputs, &plan)
            }
            None => synth.synthesize(&text, &prompt_text, &r16, &r24, &inputs),
        }
        .map_err(|e| e.to_string())?;
        wavio::write_wav_24k(&out, &wav).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({} samples, 24 kHz mono)",
            out.display(),
            wav.len()
        );
        Ok(())
    }

    fn cmd_stream(o: Opts) -> Result<(), String> {
        if o.cv3 {
            return Err(
                "the CV3 synthesizer has no chunk-streaming path; use `synth --cv3` \
                 (buffered) or `serve --cv3`"
                    .to_string(),
            );
        }
        let text = o.text.clone().ok_or("missing required --text")?;
        let out = o.out.clone().ok_or("missing required --out")?;
        let (prompt_text, r16, r24) = read_voice(&o)?;
        let hop = o.hop.unwrap_or(25);
        let mut synth = load_synth(&o)?;
        let inputs = SynthInputs {
            lm_seed: 0,
            max_gen_steps: o.max_steps,
            ..Default::default()
        };
        let mut audio: Vec<f32> = Vec::new();
        let mut nchunks = 0usize;
        synth
            .synthesize_streaming(&text, &prompt_text, &r16, &r24, &inputs, hop, |chunk| {
                nchunks += 1;
                eprintln!("syrinx stream: chunk {nchunks} (+{} samples)", chunk.len());
                audio.extend_from_slice(&chunk);
                Ok(())
            })
            .map_err(|e| e.to_string())?;
        wavio::write_wav_24k(&out, &audio).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({nchunks} chunks, {} samples)",
            out.display(),
            audio.len()
        );
        Ok(())
    }

    fn cmd_serve(o: Opts) -> Result<(), String> {
        let (prompt_text, r16, r24) = read_voice(&o)?;
        let port = o.port.unwrap_or(8080);
        let addr: SocketAddr = ([127, 0, 0, 1], port).into();
        if o.cv3 {
            let synth = load_cv3_synth(&o)?;
            let real =
                Cv3RealSynth::new(synth, prompt_text, r16, r24).with_max_gen_steps(o.max_steps);
            eprintln!("syrinx serve --cv3: listening on http://{addr}  (POST /v1/audio/speech)");
            return syrinx_serve::serve_blocking_cv3(real, addr).map_err(|e| e.to_string());
        }
        let synth = load_synth(&o)?;
        let real = RealSynth::new(synth, prompt_text, r16, r24).with_max_gen_steps(o.max_steps);
        eprintln!("syrinx serve: listening on http://{addr}  (POST /v1/audio/speech)");
        syrinx_serve::serve_blocking(real, addr).map_err(|e| e.to_string())
    }
}
