//! C2.1 gate. **AC: the test fails if a backend is added without caps.**
//!
//! Three independent ways a backend can go un-capped, each closed here, because any one
//! alone is escapable:
//!   1. a `BackendId` variant with no `caps.toml` row,
//!   2. a `caps.toml` row that is not a real backend (a typo'd id nothing can select),
//!   3. a whole backend CRATE added without ever touching `BackendId`.
//! (3) is the one that actually happens, and neither (1) nor (2) would catch it.

use std::collections::BTreeSet;
use syrinx_cue::caps::{BackendId, CapsTable, ControlCaps, ExpressiveBackend};
use syrinx_cue::ir::CueKind;
use syrinx_cue::{Granularity, Inline, Support};

fn table() -> CapsTable {
    CapsTable::embedded().expect("caps.toml must parse")
}

#[test]
fn every_backend_has_caps_and_every_caps_row_is_a_backend() {
    let t = table();
    for b in BackendId::ALL {
        assert!(
            t.get(b.as_str()).is_some(),
            "backend `{}` has no caps.toml entry — add one before shipping it",
            b.as_str()
        );
    }
    for c in t.all() {
        assert!(
            BackendId::from_str(&c.id).is_some(),
            "caps.toml row `{}` matches no BackendId variant (typo, or a removed backend)",
            c.id
        );
    }
    assert_eq!(t.len(), BackendId::ALL.len());
}

/// Guards `BackendId::ALL` itself: a variant added to the enum but forgotten in `ALL`
/// would make the check above pass vacuously. Counted from the source text, so the two
/// cannot drift.
#[test]
fn all_covers_every_enum_variant() {
    let src = include_str!("../src/caps.rs");
    let body = src
        .split("pub enum BackendId {")
        .nth(1)
        .expect("BackendId enum must exist")
        .split('}')
        .next()
        .unwrap();
    let variants: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| l.ends_with(',') && !l.starts_with("//") && !l.starts_with('#'))
        .collect();
    assert_eq!(
        variants.len(),
        BackendId::ALL.len(),
        "BackendId has {} variants but ALL lists {} — add the new one to ALL:\n{variants:#?}",
        variants.len(),
        BackendId::ALL.len()
    );
}

/// The gate that catches a NEW BACKEND CRATE. Every crate in the workspace is either
/// declared a non-backend here or must appear as some caps row's `krate`.
#[test]
fn no_backend_crate_exists_without_caps() {
    // Crates that drive no TTS checkpoint. Adding a crate forces a deliberate choice:
    // list it here, or give it caps. Silence is not an option.
    const NON_BACKEND: &[&str] = &[
        "syrinx-cli",       // runner
        "syrinx-cue",       // this crate
        "syrinx-eval",      // metrics harness
        "syrinx-frontend",  // text frontend
        "syrinx-prosody",   // prosody plan model
        "syrinx-speaker",   // speaker embeddings
        "syrinx-stream",    // audio transport
        "syrinx-stt",       // Whisper (recognition, not synthesis)
        // CosyVoice's model crates: they implement CV2/CV3, whose caps rows are attributed
        // to syrinx-serve (the crate that owns the synth entry points).
        "syrinx-acoustic",
        "syrinx-core",
        "syrinx-lm",
        "syrinx-vocoder",
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut on_disk = BTreeSet::new();
    for e in std::fs::read_dir(root).expect("crates/ must be readable") {
        let e = e.unwrap();
        if e.path().join("Cargo.toml").exists() {
            on_disk.insert(e.file_name().to_string_lossy().into_owned());
        }
    }
    assert!(on_disk.len() >= 10, "crate scan found only {on_disk:?} — the gate would be vacuous");

    let capped: BTreeSet<String> = table().all().iter().map(|c| c.krate.clone()).collect();
    for krate in &on_disk {
        assert!(
            capped.contains(krate) || NON_BACKEND.contains(&krate.as_str()),
            "crate `{krate}` is neither listed as a non-backend nor covered by a caps.toml \
             row. If it drives a checkpoint, add its caps; if not, add it to NON_BACKEND."
        );
    }
    // ...and the reverse: a caps row may not name a crate that does not exist.
    for krate in &capped {
        assert!(on_disk.contains(krate), "caps.toml names missing crate `{krate}`");
    }
}

#[test]
fn every_caps_row_cites_a_source() {
    // Same rule the C0.1 survey gate enforces: an uncited capability claim is a guess,
    // and a guess here silently mis-lowers every request.
    for c in table().all() {
        assert!(!c.source.trim().is_empty(), "`{}` has no source citation", c.id);
        assert!(
            c.source.len() > 20 && !c.source.trim().eq_ignore_ascii_case("ibid."),
            "`{}` has a non-substantive source: {:?}",
            c.id,
            c.source
        );
        assert!(!c.notes.trim().is_empty(), "`{}` has no notes", c.id);
        assert!(!c.model.trim().is_empty(), "`{}` names no checkpoint", c.id);
    }
}

/// The distinction the table exists for. If this ever collapses, `Accepted` has become
/// decorative and the 0.6B regression is back.
#[test]
fn accepted_is_not_honored_and_lowers_like_unsupported() {
    assert!(!Support::Accepted.is_effective());
    assert!(!Support::Unsupported.is_effective());
    assert!(Support::Honored.is_effective());

    let t = table();
    let small = t.get("qwen3-0.6b-customvoice").unwrap();
    let big = t.get("qwen3-1.7b-customvoice").unwrap();
    // Same API surface...
    assert_eq!(small.inline, big.inline);
    assert_eq!(small.granularity, big.granularity);
    // ...different truth. This is the real upstream behaviour, pinned.
    assert_eq!(small.instruct, Support::Accepted);
    assert_eq!(big.instruct, Support::Honored);
    let happy = CueKind::Emotion { label: "happy".into(), intensity: 0.6 };
    assert!(!small.can_express(&happy), "0.6B must not be treated as instructable");
    assert!(big.can_express(&happy));

    // At least one row must actually use `Accepted`, or the concept is untested.
    assert!(
        t.all().iter().any(|c| c.instruct == Support::Accepted
            || c.emotion == Support::Accepted
            || c.style == Support::Accepted),
        "no backend uses Accepted — the accepted/honored split would be dead weight"
    );
}

#[test]
fn support_for_maps_every_cue_kind_to_its_axis() {
    let t = table();
    let cv2 = t.get("cosyvoice2").unwrap();
    // CV2: events are inline tokens (honored), emotion only reaches it via instruct.
    assert_eq!(cv2.support_for(&CueKind::Event { label: "laughter".into() }), Support::Honored);
    assert_eq!(
        cv2.support_for(&CueKind::Emotion { label: "sad".into(), intensity: 0.5 }),
        Support::Accepted
    );
    assert_eq!(cv2.support_for(&CueKind::Pause { ms: 300 }), Support::Unsupported);

    // Free text is expressible only where the vocabulary is open.
    let s2 = t.get("fish-s2-pro").unwrap();
    let s1 = t.get("fish-s1-mini").unwrap();
    assert_eq!(s2.inline, Inline::Open);
    assert!(s2.can_express(&CueKind::Free), "open vocabulary must carry Free cues");
    assert_eq!(s1.inline, Inline::Closed);
    assert!(!s1.can_express(&CueKind::Free), "a closed set cannot take free text");

    // A backend with no channel at all expresses nothing.
    let base = t.get("qwen3-0.6b-base").unwrap();
    assert_eq!(base.granularity, Granularity::None);
    assert_eq!(base.inline, Inline::None);
    for k in [
        CueKind::Emotion { label: "happy".into(), intensity: 0.6 },
        CueKind::Style { label: "whisper".into() },
        CueKind::Event { label: "laugh".into() },
        CueKind::Free,
    ] {
        assert!(!base.can_express(&k), "clone-only backend expressed {k:?}");
    }
}

#[test]
fn granularity_and_inline_agree_with_each_other() {
    // A structural invariant, not a style rule: word-level control requires either inline
    // markup or nothing can carry it mid-utterance.
    for c in table().all() {
        if c.granularity == Granularity::Word {
            assert_ne!(c.inline, Inline::None, "`{}` claims word granularity with no inline markup", c.id);
        }
        if c.inline == Inline::None && c.granularity == Granularity::None {
            for s in [c.emotion, c.style, c.event, c.prosody, c.emphasis, c.pause] {
                assert!(!s.is_effective(), "`{}` has no channel yet honors a control", c.id);
            }
        }
    }
}

/// The trait resolves against the table for a concrete backend.
#[test]
fn expressive_backend_trait_resolves_caps() {
    struct Fish(ControlCaps);
    impl ExpressiveBackend for Fish {
        fn caps(&self) -> &ControlCaps {
            &self.0
        }
    }
    let f = Fish(BackendId::FishS2Pro.caps().unwrap());
    assert_eq!(f.backend_id(), "fish-s2-pro");
    assert_eq!(f.caps().krate, "syrinx-fish");
    assert!(f.caps().can_express(&CueKind::Free));
}
