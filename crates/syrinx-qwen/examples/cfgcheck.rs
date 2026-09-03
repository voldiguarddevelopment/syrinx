fn main() {
    for d in std::env::args().skip(1) {
        let p = std::path::Path::new(&d).join("config.json");
        match std::fs::read_to_string(&p) {
            Ok(j) => match syrinx_qwen::Qwen3TtsConfig::from_json(&j) {
                Ok(c) => println!(
                    "  OK  {:<34} {:?}  talker {}L/h{}  cp {}L  groups {}",
                    std::path::Path::new(&d).file_name().unwrap().to_string_lossy(),
                    c.variant, c.talker.num_hidden_layers, c.talker.hidden_size,
                    c.code_predictor.num_hidden_layers, c.num_code_groups),
                Err(e) => println!("  ERR {d}: {e}"),
            },
            Err(e) => println!("  ERR {d}: {e}"),
        }
    }
}
