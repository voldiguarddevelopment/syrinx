//! syrinx-cli — the `syrinx` command-line surface.
//!
//! The real work lives behind the `real` feature (the `synth` subcommand, which
//! loads the CosyVoice2 [`Synthesizer`] and renders text in a reference voice).
//! The default build is Candle-free and only prints how to enable it.

#[cfg(not(feature = "real"))]
fn main() {
    eprintln!(
        "syrinx: built without the `real` feature — the synthesizer is not compiled in.\n\
         \n\
         Rebuild with the model-backed feature to enable the `synth` command:\n\
         \n    cargo build -p syrinx-cli --features real        # CPU (parity device)\
         \n    cargo build -p syrinx-cli --features cuda        # GPU (speed)\n\
         \n\
         Then:\n\
         \n    syrinx synth --text \"<tts text>\" --prompt-text \"<ref transcript>\" \\\
         \n                 --ref-wav <ref.wav> --out <out.wav> [--model-dir <dir>] \\\
         \n                 [--max-steps N] [--cuda]\n"
    );
    std::process::exit(2);
}

#[cfg(feature = "real")]
fn main() {
    std::process::exit(real::run());
}

#[cfg(feature = "real")]
mod real {
    use std::path::PathBuf;

    use syrinx_serve::synth::{SynthConfig, SynthInputs, Synthesizer};
    use syrinx_serve::wavio;

    /// Parsed `synth` arguments.
    struct Args {
        text: String,
        prompt_text: String,
        ref_wav: PathBuf,
        out: PathBuf,
        model_dir: Option<PathBuf>,
        max_steps: Option<usize>,
        cuda: bool,
    }

    const USAGE: &str = "\
syrinx synth — render text in a reference voice (CosyVoice2)

USAGE:
    syrinx synth --text <TEXT> --prompt-text <TEXT> --ref-wav <WAV> --out <WAV>
                 [--model-dir <DIR>] [--max-steps <N>] [--cuda]

REQUIRED:
    --text <TEXT>          The text to speak.
    --prompt-text <TEXT>   Transcript of the reference clip.
    --ref-wav <WAV>        Reference voice clip (any sample rate/channels; resampled
                           to 16 kHz + 24 kHz mono internally).
    --out <WAV>            Output path for the 24 kHz mono 16-bit PCM WAV.

OPTIONAL:
    --model-dir <DIR>      Directory holding the sub-model files (default filenames
                           below). A per-model env var, when set, overrides this.
    --max-steps <N>        Cap on live LM generation steps (keeps CPU runs tractable;
                           default uses the real (text)*20 ratio).
    --cuda                 Run on GPU (requires a `--features cuda` build).

MODEL FILES (env var overrides --model-dir/<default-filename>):
    SYRINX_LM_WEIGHTS    llm_fp32.safetensors
    SYRINX_SPK_WEIGHTS   campplus_weights.safetensors
    SYRINX_FLOW_WEIGHTS  flow_fp32.safetensors
    SYRINX_HIFT_WEIGHTS  hift_fp32.safetensors
    SYRINX_TOK_JSON      tokenizer.json
    SYRINX_STOK_ONNX     speech_tokenizer_v2.onnx
";

    pub fn run() -> i32 {
        let mut argv = std::env::args().skip(1);
        match argv.next().as_deref() {
            Some("synth") => {}
            Some("-h") | Some("--help") | Some("help") => {
                println!("{USAGE}");
                return 0;
            }
            Some(other) => {
                eprintln!("syrinx: unknown command `{other}`\n\n{USAGE}");
                return 2;
            }
            None => {
                eprintln!("syrinx: missing command\n\n{USAGE}");
                return 2;
            }
        }

        let args = match parse_synth(argv) {
            Ok(a) => a,
            Err(msg) => {
                eprintln!("syrinx synth: {msg}\n\n{USAGE}");
                return 2;
            }
        };

        match synth(args) {
            Ok(out) => {
                eprintln!("syrinx: wrote {}", out.display());
                0
            }
            Err(msg) => {
                eprintln!("syrinx synth: {msg}");
                1
            }
        }
    }

    fn parse_synth(mut argv: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut text = None;
        let mut prompt_text = None;
        let mut ref_wav = None;
        let mut out = None;
        let mut model_dir = None;
        let mut max_steps = None;
        let mut cuda = false;

        // `--flag value` style; `--cuda` is a bare switch.
        let need = |argv: &mut dyn Iterator<Item = String>, flag: &str| -> Result<String, String> {
            argv.next()
                .ok_or_else(|| format!("`{flag}` expects a value"))
        };

        while let Some(arg) = argv.next() {
            match arg.as_str() {
                "--text" => text = Some(need(&mut argv, "--text")?),
                "--prompt-text" => prompt_text = Some(need(&mut argv, "--prompt-text")?),
                "--ref-wav" => ref_wav = Some(PathBuf::from(need(&mut argv, "--ref-wav")?)),
                "--out" => out = Some(PathBuf::from(need(&mut argv, "--out")?)),
                "--model-dir" => model_dir = Some(PathBuf::from(need(&mut argv, "--model-dir")?)),
                "--max-steps" => {
                    let v = need(&mut argv, "--max-steps")?;
                    max_steps = Some(
                        v.parse::<usize>()
                            .map_err(|_| format!("--max-steps expects an integer, got `{v}`"))?,
                    );
                }
                "--cuda" => cuda = true,
                "-h" | "--help" => {
                    println!("{USAGE}");
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument `{other}`")),
            }
        }

        Ok(Args {
            text: text.ok_or("missing required --text")?,
            prompt_text: prompt_text.ok_or("missing required --prompt-text")?,
            ref_wav: ref_wav.ok_or("missing required --ref-wav")?,
            out: out.ok_or("missing required --out")?,
            model_dir,
            max_steps,
            cuda,
        })
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
            spk_weights: resolve_path("SYRINX_SPK_WEIGHTS", "campplus_weights.safetensors", model_dir)?,
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

    fn synth(args: Args) -> Result<PathBuf, String> {
        let cfg = build_config(&args.model_dir)?;

        // Reference voice: read the WAV and resample to the 16 kHz + 24 kHz mono
        // buffers the synthesizer expects (the caller-side resampling contract).
        let (ref_wav_16k, ref_wav_24k) =
            wavio::read_ref_wav(&args.ref_wav).map_err(|e| e.to_string())?;

        // Load every sub-model. `--cuda` picks a GPU device when this binary was
        // built `--features cuda`; otherwise it transparently runs on CPU.
        let mut synth = if args.cuda {
            #[cfg(not(feature = "cuda"))]
            eprintln!(
                "syrinx: --cuda requested but this binary was built without the `cuda` \
                 feature; running on CPU"
            );
            let dev = syrinx_serve::synth::pick_device(None);
            Synthesizer::load_on_device(&cfg, dev).map_err(|e| e.to_string())?
        } else {
            Synthesizer::load(&cfg).map_err(|e| e.to_string())?
        };

        let inputs = SynthInputs {
            lm_seed: 0,
            max_gen_steps: args.max_steps,
            ..Default::default()
        };

        let wav = synth
            .synthesize(
                &args.text,
                &args.prompt_text,
                &ref_wav_16k,
                &ref_wav_24k,
                &inputs,
            )
            .map_err(|e| e.to_string())?;

        wavio::write_wav_24k(&args.out, &wav).map_err(|e| e.to_string())?;
        Ok(args.out)
    }
}
