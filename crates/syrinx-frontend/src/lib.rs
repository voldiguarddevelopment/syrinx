//! syrinx-frontend — deterministic text frontend (T-00.01 scaffold; T-01.01
//! normalize; T-01.02 numeric expansion; T-01.04 G2P phonemization; T-01.05
//! custom pronunciation overrides; T-01.06 heteronym resolution; T-01.07 SSML
//! subset parsing).

pub mod context;
pub mod expand;
pub mod g2p;
pub mod hetero;
pub mod lexicon;
pub mod normalize;
pub mod pacing;
pub mod punct;
pub mod ssml;

/// Dispatch a single named frontend stage over raw `input`, returning the stage's
/// textual output (T-01.11). This is the one entry point the aggregating
/// golden-file suite drives, one stage per fixture sub-tree:
///
///   * `"normalize"` -> [`normalize::normalize`]
///   * `"numbers"`   -> [`expand::expand_numbers`]
///   * `"ssml"`      -> the `Debug` rendering of [`ssml::parse`]
///
/// An unknown stage name is a programming error in the fixture tree and panics.
pub fn render_stage(stage: &str, input: &str) -> String {
    match stage {
        "normalize" => normalize::normalize(input),
        "numbers" => expand::expand_numbers(input),
        "ssml" => format!("{:?}", ssml::parse(input)),
        other => panic!("unknown frontend stage `{other}`"),
    }
}
