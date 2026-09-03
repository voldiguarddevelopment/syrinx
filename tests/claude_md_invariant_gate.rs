//! Doc gate: the constitution must actually CARRY the hard invariant and the D6 crate
//! contract. Both are acceptance criteria, so neither may rest on a reviewer's memory —
//! a doc edit that quietly drops them fails CI here.

const CLAUDE_MD: &str = include_str!("../CLAUDE.md");
const ADR: &str = include_str!("../adr/0001-cue-ir.md");

/// The invariant sentence, taken from ADR-0001 §5 itself rather than retyped here — so the
/// two cannot drift, which is exactly what C5.1a's "byte-for-byte" asks for.
fn adr_invariant_sentence() -> String {
    let sec = ADR
        .split("## 5. The hard invariant")
        .nth(1)
        .expect("ADR-0001 must have section 5");
    let line = sec
        .lines()
        .map(str::trim)
        .find(|l| l.starts_with("> **"))
        .expect("section 5 must state the invariant as a bold blockquote");
    line.trim_start_matches("> **").trim_end_matches("**").to_string()
}

#[test]
fn claude_md_carries_the_adr_invariant_sentence_byte_for_byte() {
    let sentence = adr_invariant_sentence();
    assert!(
        sentence.len() > 40,
        "extracted an implausible invariant sentence: {sentence:?}"
    );
    assert!(
        CLAUDE_MD.contains(&sentence),
        "CLAUDE.md does not contain ADR-0001 §5's sentence byte-for-byte.\n  want: {sentence:?}"
    );
}

#[test]
fn claude_md_states_the_no_literal_leakage_invariant_as_a_hard_rule() {
    let rules = CLAUDE_MD
        .split("## Non-negotiable rules")
        .nth(1)
        .expect("CLAUDE.md must have a Non-negotiable rules section")
        .split("\n## ")
        .next()
        .unwrap();
    assert!(
        rules.contains(&adr_invariant_sentence()),
        "the hard invariant is not listed among the non-negotiable rules"
    );
    // It must name what it covers and where it is enforced, or it is a slogan.
    for needle in ["speaker token", "SSML tag", "release blocker", "invariant_property.rs"] {
        assert!(rules.contains(needle), "invariant rule does not mention {needle:?}");
    }
}

#[test]
fn claude_md_crate_table_reflects_d6() {
    assert!(
        CLAUDE_MD.contains("| `syrinx-cue` |"),
        "syrinx-cue is missing from the crate-contract table"
    );
    let frontend = CLAUDE_MD
        .lines()
        .find(|l| l.starts_with("| `syrinx-frontend` |"))
        .expect("syrinx-frontend row must exist");
    assert!(
        !frontend.contains("SSML"),
        "D6 moved SSML to syrinx-cue, but the frontend row still claims it: {frontend}"
    );
    let cue = CLAUDE_MD
        .lines()
        .find(|l| l.starts_with("| `syrinx-cue` |"))
        .unwrap();
    assert!(cue.contains("SSML"), "syrinx-cue row must own SSML: {cue}");
}

#[test]
fn the_invariants_single_exception_is_named_and_stays_single() {
    // D7 narrowed the invariant for one deprecated parser. That narrowing must stay
    // visible and must not grow: an unnamed or plural exception is how a hard rule rots.
    let rules = CLAUDE_MD
        .split("## Non-negotiable rules")
        .nth(1)
        .unwrap()
        .split("\n## ")
        .next()
        .unwrap();
    assert!(
        rules.contains("legacy_emotion::parse_tagged"),
        "the invariant's exception must name the exact function it covers"
    );
    assert!(rules.contains("One named exception, and only one"));
    assert!(
        rules.contains("ADR-0001 §11") || rules.contains("D7"),
        "the exception must cite the ADR that granted it"
    );
    // Exactly one exception is described.
    assert_eq!(
        rules.matches("exception").count(),
        3,
        "the exception wording changed — re-read ADR-0001 §11 before adjusting this gate"
    );
}


/// C5.1b: the training-data format doc must exist and cover all four §2.6 stages.
#[test]
fn training_data_format_covers_all_four_spec_stages() {
    let doc = std::fs::read_to_string("docs/TRAINING_DATA_FORMAT.md")
        .expect("docs/TRAINING_DATA_FORMAT.md must exist");
    for heading in [
        "## 1. Annotator pipeline",
        "## 2. Cue storage",
        "## 3. Text/audio interleaving parameters",
        "## 4. Reward spec",
    ] {
        assert!(doc.contains(heading), "TRAINING_DATA_FORMAT.md is missing {heading:?}");
    }
    // Each stage must actually say the specific things §2.6 asks for, not just carry the
    // heading — a heading with nothing under it would pass a naive check.
    let lower = doc.to_lowercase();
    for needle in [
        "forced aligner",          // stage 1: bootstrapped rich transcription
        "human-checked",           // stage 1: precision sample
        "<|speaker:N|>",           // stage 2: speaker turns
        "loss-masked",             // stage 2: reference audio prefix
        "interleave_probability",  // stage 3: interleaving parameters
        "without std normalization", // stage 4: GRPO-style advantages
        "missed cues",             // stage 4: heavy penalty term
        "wrong speaker",           // stage 4: heavy penalty term
    ] {
        assert!(
            lower.contains(&needle.to_lowercase()),
            "TRAINING_DATA_FORMAT.md does not specify {needle:?}"
        );
    }
}
