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

    use syrinx_fish::common::audio as fish_audio;
    use syrinx_fish::common::dualar::DriveParams;
    use syrinx_fish::s1::S1Mini;
    use syrinx_fish::s2::S2Pro;
    use syrinx_fish::FishVariant;
    use syrinx_prosody::render_plan::RenderPlan;
    use syrinx_serve::emotion::{EmotionRegistry, InstructLang};
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
    stt      Transcribe a WAV to text (pure-Rust Whisper; the TTS test oracle).

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
                 [--quality] [--instruct <TEXT>] [--rl <WEIGHTS>] [--quantized]
                 [--emotion-tags] [--emotion-lang <zh|en>] [--list-emotions]
                 [--fish <s1-mini|s2-pro> --fish-dir <DIR>]

    --text <TEXT>          Text to speak.                              [required]
    --out <WAV>            Output 24 kHz mono 16-bit WAV.              [required]
    --rate <R>             Global speech-rate (>1 faster, <1 slower; faithful).
    --pitch <SEMITONES>    Global pitch shift (NOTE: weak training-free lever).
    --plan <FILE.json>     Load a full RenderPlan (overrides --pitch/--rate).
    --cv3                  Use the CosyVoice3 synthesizer (SYRINX_CV3_* files);
                           prosody plan/pitch/rate are CV2-only and are ignored.

  Fish Audio front door (mutually exclusive with --cv3 + all CV3-only flags):
    --fish <VARIANT>       Drive the pure-Rust Fish Audio port instead of CosyVoice:
                           s1-mini (openaudio-s1-mini 0.5B) or s2-pro (5B). Output
                           is the codec's native 44.1 kHz mono 32-bit-float WAV.
    --fish-dir <DIR>       Checkpoint dir holding the model + config.json (and the
                           codec + tokenizer.json). Required with --fish.
                           Reuses --text/--ref-wav/--out; emotion/style tags in
                           --text (e.g. \"[happy]\") are Fish-native PLAIN TEXT.
                           --ref-wav clones the voice (s2-pro: encoded to prompt
                           codes; --prompt-text is the optional ref transcript).
                           s1-mini has no reference-cloning path: --ref-wav is
                           ignored there (text-only synthesis).

  CV3-only feature flags (require --cv3):
    --quality              Render with the real random-phase NSF SineGen source
                           (perceptual-quality path) instead of the deterministic
                           single-harmonic smoke source. Mutually exclusive with
                           --instruct.
    --instruct <TEXT>      Emotion / instruct control: <TEXT> (e.g. \"speak in a
                           sad tone\") takes the LM prompt-text role and the prompt
                           speech tokens are dropped; the cloned voice is kept.
                           Replaces --prompt-text's role for the LM. Mutually
                           exclusive with --quality.
    --rl <WEIGHTS>         Load the RL post-trained LM checkpoint (llm.rl_fp32
                           .safetensors) in place of the base llm_fp32 weights.
    --quantized            Load every CV3 sub-model int4-quantized (the ~488 MB
                           opt-in size footprint). NOTE: dequant-on-fetch is SLOW
                           and lossy — opt-in for size, not the default.

  Inline emotion tagging (require --cv3; mutually exclusive with --instruct):
    --emotion-tags         Treat the text as inline-tagged: each `[tag] …` span is
                           spoken with that emotion and the spans are concatenated
                           (equal-power cross-fade at each seam). Auto-enabled when
                           the text contains a known `[tag]`/`(tag)`.
                           e.g. --text \"[happy] hi there [sad] bye\"
    --emotion-lang <L>     Instruct language for the tags: zh (default, on-box
                           confirmed) or en.
    --list-emotions        Print the tag -> instruct table and exit (no model load).
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

    const STT_USAGE: &str = "\
syrinx stt — transcribe a WAV to text (pure-Rust Whisper, the TTS test oracle)

    syrinx stt --wav <FILE> [--model-dir <DIR>] [--lang <L>] [--cuda]
               [--check-tts \"<expected text>\"]

    --wav <FILE>           Audio clip to transcribe (any rate; resampled to 16 kHz).
    --model-dir <DIR>      Whisper model dir: config.json + tokenizer.json +
                           model.safetensors (HF openai/whisper-* layout).
                           Defaults to $SYRINX_STT_MODEL_DIR.
    --lang <L>             Force the language (e.g. en, zh); default auto-detect.
    --check-tts <TEXT>     Also print the word error rate (WER) of the transcript
                           against <TEXT> — the TTS-intelligibility oracle.
    --cuda                 Run on GPU (requires a --features cuda build).

    Prints the transcript and the detected language to stdout.
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
        // --- CV3 feature front doors (`synth --cv3` only) ---
        quality: bool,
        instruct: Option<String>,
        rl: Option<PathBuf>,
        quantized: bool,
        // --- inline emotion tagging (`synth --cv3` only) ---
        emotion_tags: bool,
        list_emotions: bool,
        emotion_lang: Option<String>,
        // --- Fish Audio front door (`synth --fish` only) ---
        fish: Option<String>,
        fish_dir: Option<PathBuf>,
        // --- speech-to-text (`stt` only) ---
        wav: Option<PathBuf>,
        lang: Option<String>,
        check_tts: Option<String>,
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
            "stt" => (STT_USAGE, cmd_stt),
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
                "--quality" => o.quality = true,
                "--instruct" => o.instruct = Some(need(&mut argv, "--instruct")?),
                "--rl" => o.rl = Some(PathBuf::from(need(&mut argv, "--rl")?)),
                "--quantized" => o.quantized = true,
                "--emotion-tags" => o.emotion_tags = true,
                "--list-emotions" => o.list_emotions = true,
                "--emotion-lang" => o.emotion_lang = Some(need(&mut argv, "--emotion-lang")?),
                "--fish" => o.fish = Some(need(&mut argv, "--fish")?),
                "--fish-dir" => o.fish_dir = Some(PathBuf::from(need(&mut argv, "--fish-dir")?)),
                "--wav" => o.wav = Some(PathBuf::from(need(&mut argv, "--wav")?)),
                "--lang" => o.lang = Some(need(&mut argv, "--lang")?),
                "--check-tts" => o.check_tts = Some(need(&mut argv, "--check-tts")?),
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
    ///
    /// The loader is selected by the CV3 feature flags: `--rl <WEIGHTS>` swaps only the
    /// LM weights for the RL post-trained checkpoint ([`Cv3Synthesizer::load_with_lm`]);
    /// `--quantized` loads every sub-model int4-quantized
    /// ([`Cv3Synthesizer::load_quantized_on_device`]); otherwise the fp32 loader. `--rl`
    /// and `--quantized` are mutually exclusive (there is no quantized RL loader).
    fn load_cv3_synth(o: &Opts) -> Result<Cv3Synthesizer, String> {
        let cfg = build_cv3_config(&o.model_dir)?;
        if o.cuda {
            #[cfg(not(feature = "cuda"))]
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
        }
        // CPU is the parity device; `--cuda` picks a GPU on a `--features cuda` build
        // (and `pick_device` itself falls back to CPU when no GPU is present).
        let dev = if o.cuda {
            syrinx_serve::synth::pick_device(None)
        } else {
            candle_core::Device::Cpu
        };
        match (&o.rl, o.quantized) {
            (Some(_), true) => Err(
                "--rl and --quantized are mutually exclusive (no quantized RL loader)".to_string(),
            ),
            (Some(rl), false) => {
                let rl = rl.to_string_lossy();
                eprintln!("syrinx synth --cv3: loading RL post-trained LM from {rl}");
                Cv3Synthesizer::load_with_lm(&cfg, &rl, dev).map_err(|e| e.to_string())
            }
            (None, true) => {
                eprintln!(
                    "syrinx synth --cv3: loading int4-quantized sub-models \
                     (opt-in size path; dequant-on-fetch is slow + lossy)"
                );
                Cv3Synthesizer::load_quantized_on_device(&cfg, dev).map_err(|e| e.to_string())
            }
            (None, false) => Cv3Synthesizer::load_on_device(&cfg, dev).map_err(|e| e.to_string()),
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
    ///
    /// The CV3 feature flags pick the synthesis path:
    ///   * `--instruct <TEXT>` → [`Cv3Synthesizer::synthesize_instruct`] (the instruct
    ///     text takes the prompt-text role; `--prompt-text` is not needed);
    ///   * `--quality`         → [`Cv3Synthesizer::synthesize_quality`] (real SineGen
    ///     source, seed 0);
    ///   * otherwise the default buffered [`Cv3Synthesizer::synthesize`].
    /// `--quality` and `--instruct` are mutually exclusive. The loader (`--rl` /
    /// `--quantized`) is selected in [`load_cv3_synth`].
    fn cmd_synth_cv3(o: &Opts, text: &str, out: &std::path::Path) -> Result<(), String> {
        if o.quality && o.instruct.is_some() {
            return Err("--quality and --instruct are mutually exclusive".to_string());
        }
        if o.plan.is_some() || o.pitch.is_some() || o.rate.is_some() {
            eprintln!(
                "syrinx synth --cv3: prosody plan/pitch/rate are CV2-only and are ignored \
                 on the CV3 path"
            );
        }

        // Inline emotion tagging: explicit `--emotion-tags`, or auto-detected when the text
        // carries a known `[tag]`/`(tag)`. Mutually exclusive with the single-shot
        // --instruct / --quality paths (those steer the whole utterance).
        let lang = resolve_emotion_lang(o)?;
        let registry = EmotionRegistry::default().with_lang(lang);
        let want_tags = o.emotion_tags || registry.has_emotion_tags(text);
        if want_tags && (o.instruct.is_some() || o.quality) {
            return Err(
                "inline emotion tags are mutually exclusive with --instruct/--quality \
                 (each steers the whole utterance differently)"
                    .to_string(),
            );
        }
        if want_tags {
            return cmd_synth_cv3_tagged(o, text, out, &registry);
        }

        // The reference clip (16 kHz + 24 kHz) is always needed for the cloned voice.
        let ref_wav = o.ref_wav.as_ref().ok_or("missing required --ref-wav")?;
        let (r16, r24) = wavio::read_ref_wav(ref_wav).map_err(|e| e.to_string())?;
        let mut synth = load_cv3_synth(o)?;
        let wav = if let Some(instruct) = &o.instruct {
            // Emotion / instruct: the instruct text replaces the prompt transcript in the
            // LM role; the prompt speech tokens are dropped. `--prompt-text` is ignored.
            eprintln!("syrinx synth --cv3: instruct = {instruct:?}");
            synth
                .synthesize_instruct(text, instruct, &r16, &r24)
                .map_err(|e| e.to_string())?
        } else {
            let prompt_text = o
                .prompt_text
                .clone()
                .ok_or("missing required --prompt-text")?;
            if o.quality {
                synth
                    .synthesize_quality(text, &prompt_text, &r16, &r24, 0)
                    .map_err(|e| e.to_string())?
            } else {
                let inputs = Cv3SynthInputs {
                    lm_seed: 0,
                    max_gen_steps: o.max_steps,
                    ..Default::default()
                };
                synth
                    .synthesize(text, &prompt_text, &r16, &r24, &inputs)
                    .map_err(|e| e.to_string())?
            }
        };
        wavio::write_wav_24k(out, &wav).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({} samples, 24 kHz mono, CV3)",
            out.display(),
            wav.len()
        );
        Ok(())
    }

    /// Inline-tagged CV3 render: parse `[happy] … [sad] …` text into emotion segments and
    /// synthesize each in the reference voice, concatenating with an equal-power cross-fade
    /// ([`Cv3Synthesizer::synthesize_tagged`]). A neutral / unknown-tag span is spoken
    /// plainly; the instruct string for each known tag comes from `registry` (its active
    /// language). `--prompt-text` is required (it conditions the neutral spans).
    fn cmd_synth_cv3_tagged(
        o: &Opts,
        text: &str,
        out: &std::path::Path,
        registry: &EmotionRegistry,
    ) -> Result<(), String> {
        let ref_wav = o.ref_wav.as_ref().ok_or("missing required --ref-wav")?;
        let (r16, r24) = wavio::read_ref_wav(ref_wav).map_err(|e| e.to_string())?;
        let prompt_text = o
            .prompt_text
            .clone()
            .ok_or("missing required --prompt-text")?;
        eprintln!(
            "syrinx synth --cv3: inline emotion tags ({} known tags, lang {:?})",
            registry.len(),
            registry.lang()
        );
        let mut synth = load_cv3_synth(o)?;
        let inputs = Cv3SynthInputs {
            lm_seed: 0,
            max_gen_steps: o.max_steps,
            ..Default::default()
        };
        let wav = synth
            .synthesize_tagged(text, &prompt_text, &r16, &r24, registry, &inputs)
            .map_err(|e| e.to_string())?;
        wavio::write_wav_24k(out, &wav).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({} samples, 24 kHz mono, CV3 emotion-tagged)",
            out.display(),
            wav.len()
        );
        Ok(())
    }

    /// Reject the CV3-only feature flags (`--quality` / `--instruct` / `--rl` /
    /// `--quantized` / `--emotion-tags` / `--emotion-lang`) on any path other than
    /// `synth --cv3`, so they are never silently ignored on a CV2 / serve / stream command.
    fn reject_cv3_feature_flags(o: &Opts) -> Result<(), String> {
        if o.quality
            || o.instruct.is_some()
            || o.rl.is_some()
            || o.quantized
            || o.emotion_tags
            || o.emotion_lang.is_some()
        {
            return Err(
                "--quality/--instruct/--rl/--quantized/--emotion-tags/--emotion-lang are \
                 CV3-only and valid only on `synth --cv3`"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// Reject the Fish front door (`--fish` / `--fish-dir`) on any path other than
    /// `synth --fish`, so they are never silently ignored on a CV2 / CV3 / serve / stream
    /// command (mirrors [`reject_cv3_feature_flags`]).
    fn reject_fish_flags(o: &Opts) -> Result<(), String> {
        if o.fish.is_some() || o.fish_dir.is_some() {
            return Err(
                "--fish/--fish-dir are valid only on `synth --fish` (the pure-Rust Fish \
                 Audio port); they are not valid on the CosyVoice2/CV3 or serve/stream paths"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// `synth --fish <s1-mini|s2-pro>` — drive the pure-Rust Fish Audio port. Loads the
    /// variant from `--fish-dir` (model + `config.json` + codec + tokenizer; the loaders
    /// read `config.json` via [`syrinx_fish::common::config::FishConfig::from_fish_json`]
    /// internally), synthesizes `text`, and writes the codec's native 44.1 kHz mono WAV.
    ///
    /// `--ref-wav` clones the voice: for `s2-pro` the reference is resampled to 44.1 kHz,
    /// encoded to prompt codes by the codec, and fed to [`S2Pro::synthesize_cloned`] (with
    /// `--prompt-text` as the optional reference transcript, defaulting to empty). `s1-mini`
    /// has no reference-conditioned cloning path in the backend, so `--ref-wav` is ignored
    /// there (text-only [`S1Mini::synthesize`]).
    ///
    /// Emotion/style tags in `text` (e.g. `[happy]`, `(whisper)`) are Fish-native PLAIN
    /// TEXT — they are NOT routed through the CV3 emotion module. `--fish` is mutually
    /// exclusive with `--cv3` and every CV3-only flag.
    fn cmd_synth_fish(o: &Opts, text: &str, out: &std::path::Path) -> Result<(), String> {
        if o.cv3 {
            return Err("--fish and --cv3 are mutually exclusive (pick one synthesizer)".to_string());
        }
        // The CV3-only feature flags steer the CosyVoice3 path; reject them on the Fish path
        // so they are never silently ignored (Fish emotion tags are plain text in --text).
        reject_cv3_feature_flags(o)?;
        if o.plan.is_some() || o.pitch.is_some() || o.rate.is_some() {
            eprintln!(
                "syrinx synth --fish: prosody plan/pitch/rate are CosyVoice2-only and are \
                 ignored on the Fish path"
            );
        }

        let variant = FishVariant::from_id(o.fish.as_deref().unwrap_or_default())
            .ok_or("--fish expects `s1-mini` or `s2-pro`")?;
        let dir = o
            .fish_dir
            .as_ref()
            .ok_or("--fish requires --fish-dir <DIR> (the checkpoint directory)")?;

        // CPU is the parity device; `--cuda` picks a GPU on a `--features cuda` build (and
        // `pick_device` itself falls back to CPU when no GPU is present). Mirrors the CV3 path.
        if o.cuda {
            #[cfg(not(feature = "cuda"))]
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
        }
        let dev = if o.cuda {
            syrinx_serve::synth::pick_device(None)
        } else {
            candle_core::Device::Cpu
        };

        // Map `--max-steps` to the dual-AR driver's frame cap; keep the seed pinned (0) so a
        // run is bit-reproducible, mirroring the CV `lm_seed: 0` convention.
        let params = DriveParams {
            seed: 0,
            max_new_frames: o
                .max_steps
                .unwrap_or_else(|| DriveParams::default().max_new_frames),
            ..Default::default()
        };

        let wav = match variant {
            FishVariant::S1Mini => {
                let mut model = S1Mini::load(dir, dev).map_err(|e| e.to_string())?;
                if o.ref_wav.is_some() {
                    eprintln!(
                        "syrinx synth --fish s1-mini: s1-mini has no reference-conditioned \
                         cloning path; --ref-wav is ignored (text-only synthesis)"
                    );
                }
                model.synthesize(text, &params).map_err(|e| e.to_string())?
            }
            FishVariant::S2Pro => {
                // `S2Pro::load` picks the compute dtype from the device: f32 on CPU
                // (parity) and bf16 on CUDA (the 4.4B LM fits a 12 GB GPU in bf16, ~9 GB,
                // where f32 ~18 GB would OOM). So a `--features cuda` + `--cuda` run gets
                // the bf16-fit path automatically; the CPU path is byte-unchanged.
                let mut model = S2Pro::load(dir, dev.clone()).map_err(|e| e.to_string())?;
                match &o.ref_wav {
                    Some(ref_wav) => {
                        // Resample the reference to the codec's 44.1 kHz, encode it to prompt
                        // codes, and clone the voice. The reference transcript (--prompt-text)
                        // is optional; an empty transcript still conditions on the audio codes.
                        let ref_text = o.prompt_text.clone().unwrap_or_default();
                        let samples =
                            fish_audio::read_ref_wav_44k(ref_wav).map_err(|e| e.to_string())?;
                        let n = samples.len();
                        let wav_t = candle_core::Tensor::from_vec(samples, n, &dev)
                            .map_err(|e| e.to_string())?;
                        let ref_codes =
                            model.encode_reference(&wav_t).map_err(|e| e.to_string())?;
                        eprintln!(
                            "syrinx synth --fish s2-pro: cloning from {} ({} samples @44.1k, \
                             ref-transcript {:?})",
                            ref_wav.display(),
                            n,
                            ref_text
                        );
                        if std::env::var("SYRINX_FISH_CODEC_ROUNDTRIP").is_ok() {
                            // DIAGNOSTIC: encode -> decode the reference with NO LM, to isolate
                            // whether garbled output is the codec (round-trip sounds bad) or the
                            // LM (round-trip reconstructs the reference voice => codec is fine).
                            eprintln!("syrinx synth --fish s2-pro: CODEC ROUND-TRIP (no LM)");
                            model.decode_codes(&ref_codes).map_err(|e| e.to_string())?
                        } else {
                            model
                                .synthesize_cloned(&ref_text, &ref_codes, text, &params)
                                .map_err(|e| e.to_string())?
                        }
                    }
                    None => model.synthesize(text, &params).map_err(|e| e.to_string())?,
                }
            }
        };

        fish_audio::write_wav_44k(out, &wav).map_err(|e| e.to_string())?;
        eprintln!(
            "syrinx: wrote {} ({} samples, 44.1 kHz mono f32, Fish {})",
            out.display(),
            wav.len(),
            variant.dir_name()
        );
        Ok(())
    }

    /// Resolve `--emotion-lang` (`zh` default, or `en`) into an [`InstructLang`].
    fn resolve_emotion_lang(o: &Opts) -> Result<InstructLang, String> {
        match o.emotion_lang.as_deref() {
            None | Some("zh") => Ok(InstructLang::Zh),
            Some("en") => Ok(InstructLang::En),
            Some(other) => Err(format!(
                "--emotion-lang expects `zh` or `en`, got `{other}`"
            )),
        }
    }

    /// Print the registry's `tag -> instruct` table (both language variants) and return.
    /// Model-free — `--list-emotions` never loads weights.
    fn list_emotions(o: &Opts) -> Result<(), String> {
        let lang = resolve_emotion_lang(o)?;
        let reg = EmotionRegistry::default().with_lang(lang);
        println!(
            "syrinx emotion tags ({} known; active instruct language: {})",
            reg.len(),
            match lang {
                InstructLang::Zh => "zh",
                InstructLang::En => "en",
            }
        );
        println!("Use inline as `[tag] text …`; (tag) parentheses also accepted.\n");
        for tag in reg.tags() {
            if let Some(pair) = reg.instruct_pair(tag) {
                println!("  {tag:<12}  zh: {}", pair.zh);
                println!("  {:<12}  en: {}", "", pair.en);
            }
        }
        Ok(())
    }

    fn cmd_synth(o: Opts) -> Result<(), String> {
        // `--list-emotions` is a model-free query: print the table and exit before any of
        // the required-arg / model-load machinery.
        if o.list_emotions {
            return list_emotions(&o);
        }
        let text = o.text.clone().ok_or("missing required --text")?;
        let out = o.out.clone().ok_or("missing required --out")?;
        if o.fish.is_some() {
            return cmd_synth_fish(&o, &text, &out);
        }
        reject_fish_flags(&o)?;
        if o.cv3 {
            return cmd_synth_cv3(&o, &text, &out);
        }
        reject_cv3_feature_flags(&o)?;
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
        reject_cv3_feature_flags(&o)?;
        reject_fish_flags(&o)?;
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
        reject_cv3_feature_flags(&o)?;
        reject_fish_flags(&o)?;
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

    /// `stt` — transcribe a WAV with the pure-Rust Whisper engine. The mirror of
    /// the synth path: `audio -> text`, and the native TTS-intelligibility oracle
    /// (`--check-tts` prints the WER of the transcript vs. the expected text).
    fn cmd_stt(o: Opts) -> Result<(), String> {
        let wav = o.wav.clone().ok_or("missing required --wav")?;

        // Model dir: --model-dir, else $SYRINX_STT_MODEL_DIR.
        let model_dir = o
            .model_dir
            .clone()
            .or_else(|| std::env::var("SYRINX_STT_MODEL_DIR").ok().map(PathBuf::from))
            .ok_or(
                "no Whisper model: pass --model-dir <DIR> or set SYRINX_STT_MODEL_DIR \
                 (dir with config.json + tokenizer.json + model.safetensors)",
            )?;

        if o.cuda && !cfg!(feature = "cuda") {
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
        }
        let dev = if o.cuda {
            syrinx_serve::synth::pick_device(None)
        } else {
            candle_core::Device::Cpu
        };

        // Reuse syrinx-serve's WAV reader/resampler — its 16 kHz mono buffer is
        // exactly what Whisper consumes.
        let (samples_16k, _r24) = wavio::read_ref_wav(&wav).map_err(|e| e.to_string())?;

        let stt = syrinx_stt::Stt::load(&model_dir, dev).map_err(|e| e.to_string())?;
        let transcript = stt
            .transcribe_lang(&samples_16k, 16_000, o.lang.as_deref())
            .map_err(|e| e.to_string())?;

        let lang = transcript.language.as_deref().unwrap_or("?");
        eprintln!(
            "syrinx stt: {} ({} segment(s), language: {lang})",
            wav.display(),
            transcript.segments.len()
        );
        println!("language: {lang}");
        println!("{}", transcript.text);

        if let Some(expected) = &o.check_tts {
            let score = syrinx_stt::wer(expected, &transcript.text);
            println!("wer: {score:.4}");
            eprintln!(
                "syrinx stt --check-tts: WER {score:.4} (expected {:?} vs transcript {:?})",
                expected, transcript.text
            );
        }
        Ok(())
    }
}
