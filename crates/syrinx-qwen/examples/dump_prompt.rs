//! Print the prompt plan step by step, for comparison against the reference.
//!
//! The reference's geometry is recoverable with a few lines of Python
//! (`Qwen3TTSModel._build_instruct_text` / `_tokenize_texts`, and the
//! `inputs_embeds` shape at `Qwen3TTSTalkerForConditionalGeneration.generate`), so this
//! prints the same facts from the Rust side: the instruct token ids, the total step
//! count, and every step's `(text, codec)` pair. That pairing is where the two
//! implementations can disagree without either one changing length.
//!
//! Usage: dump_prompt <talker-dir> "<text>" [speaker] [language] [instruct]
//!
//! With `SYRINX_QWEN_EMBEDS=<out.safetensors>` it also loads the talker and writes the
//! REALIZED prompt embeddings under key `prompt`, so they can be diffed numerically
//! against the reference's `inputs_embeds`. Identical token ids do not imply identical
//! embeddings — the text and codec addends come from two different tables.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (dir, text) = (&a[0], &a[1]);
    let speaker = a.get(2).map(String::as_str).unwrap_or("serena");
    let language = a.get(3).map(String::as_str).unwrap_or("english");
    let instruct = a.get(4).map(String::as_str).filter(|s| !s.is_empty());

    let cfg_json = std::fs::read_to_string(std::path::Path::new(dir).join("config.json"))?;
    let pcfg = syrinx_qwen::prompt::PromptConfig::from_json(&cfg_json)?;
    let tok = syrinx_qwen::tokenizer::QwenTokenizer::from_dir(dir)?;

    if let Some(ins) = instruct {
        let s = format!(
            "{s}user\n{ins}{e}\n",
            s = syrinx_qwen::tokenizer::IM_START,
            e = syrinx_qwen::tokenizer::IM_END
        );
        println!("instruct text: {s:?}");
        println!("instruct ids : {:?}", tok.encode(&s)?);
    }
    let chat = format!(
        "{s}assistant\n{text}{e}\n{s}assistant\n",
        s = syrinx_qwen::tokenizer::IM_START,
        e = syrinx_qwen::tokenizer::IM_END
    );
    println!("text ids     : {:?}", tok.encode(&chat)?);

    let plan = syrinx_qwen::prompt::build_custom_voice(
        &tok, &pcfg, text, speaker, instruct, language, true,
    )?;
    println!("steps        : {}", plan.len());
    println!("trailing     : {:?}", plan.trailing_text);
    for (i, st) in plan.steps.iter().enumerate() {
        println!("  {i:3}  text={:?}  codec={:?}", st.text, st.codec);
    }

    if let Ok(out) = std::env::var("SYRINX_QWEN_EMBEDS") {
        use candle_core::{DType, Device};
        let dev = match std::env::var("SYRINX_QWEN_DEVICE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            Some(i) => Device::new_cuda(i)?,
            None => Device::Cpu,
        };
        let m = syrinx_qwen::model::Qwen3Tts::load(dir, dev.clone())?;
        let prompt = m.realize_plan(&plan, None, &[])?;
        let t = prompt.inputs_embeds.to_dtype(DType::F32)?.flatten_all()?;
        let n = plan.len();
        let t = t.reshape((n, t.elem_count() / n))?;
        eprintln!("realized embeds {:?} -> {out}", t.shape());
        let mut map = std::collections::HashMap::new();
        map.insert("prompt".to_string(), t);
        candle_core::safetensors::save(&map, &out)?;
    }
    Ok(())
}
