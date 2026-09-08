//! Cue-phrasing tuning: the decision, and only the decision.
//!
//! The loop renders candidate instruct phrasings, measures them, and proposes one. This
//! module is the **proposing** part, and it is pure — measurements in, verdict out — so
//! the guards below are frozen-tested on the model-free board. That is the whole design
//! intent: a future pass that wants to relax a margin has to get past a test to do it,
//! rather than editing a constant in a file nobody runs.
//!
//! # Why the guards are the feature
//!
//! Optimising a phrase against a measurement is a Goodhart machine by default. The
//! measured facts that shape these guards:
//!
//! * the judge is right 91.1% of the time cross-corpus and **66.7% on `fearful`**, so it
//!   can be fooled;
//! * the acoustic test is **direction-blind**, so it cannot tell "sadder" from "louder";
//! * a longer prompt perturbs the AR trajectory on its own, which is why every batch
//!   carries a sham (`renders/2026-09-06-instruct-lang/`).
//!
//! So the criteria are **conjunctive, never a weighted sum**. The moment two of them are
//! tradeable, a search will trade them: a phrase that wrecks intelligibility to gain affect
//! would win on any scalar objective, and here it simply fails.
//!
//! # What this module deliberately cannot do
//!
//! Accept a candidate. [`decide`] returns a [`TuneOutcome`]; a row is inert in
//! `instruct.toml` until a human writes `accepted_by`. Per CLAUDE.md, "intended emotion" is
//! not expressible as a frozen-test gate, and no amount of measurement changes that.

use std::collections::BTreeMap;

/// One phrasing under test.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    /// Canonical `vocab.toml` id this phrasing is for.
    pub label: String,
    /// The instruction text itself.
    pub phrase: String,
}

/// Which arm a measurement came from.
///
/// [`Arm::CounterCue`] is the guard nobody thinks of: the phrase for the **wrong** label
/// (for `[sad]`, the `happy` phrasing). It should lose. If it wins, the judge is not
/// tracking the thing it is supposed to track and the entire batch is void — which turns
/// "the judge might be weak" from a known unknown into a per-run observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TuneArm {
    /// No instruction.
    Plain,
    /// The phrasing currently shipping for this label.
    Incumbent,
    /// The phrasing being proposed.
    Candidate,
    /// Delivery-neutral, length-matched.
    Sham,
    /// The phrasing for a *different* label.
    CounterCue,
}

/// One arm's measured outcome on one split.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneMeasurement {
    pub label: String,
    /// `None` for arms that are not a specific candidate (plain, sham, counter-cue).
    pub phrase: Option<String>,
    pub arm: TuneArm,
    /// Which split this was measured on. Selection reads `Tune`; confirmation reads
    /// `Holdout`, and the two must never be pooled.
    pub split: Split,
    /// Acoustic permutation p-value against the plain arm.
    pub p_vs_plain: f64,
    /// Acoustic permutation p-value against the sham arm — content over perturbation.
    pub p_vs_sham: f64,
    /// Judge delta on the cued class, in the judge's own units (logits).
    pub cued_delta: f64,
    /// Pooled across-seed noise for that delta.
    pub cued_noise: f64,
    /// The label that gained most. A phrase that raises `surprised` while nudging `happy`
    /// is not a win, and comparing this to the target is how that is caught.
    pub largest_gain_label: String,
    /// Word error rate for this arm.
    pub wer: f64,
    /// Cosine to the plain render's speaker embedding.
    pub speaker_similarity: f64,
}

/// Which half of the sentence set a measurement came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Split {
    /// Used to pick a winner.
    Tune,
    /// Used only to confirm one. Never read during selection.
    Holdout,
}

/// The pre-registered decision parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneThresholds {
    /// Family-wise alpha before correction.
    pub alpha: f64,
    /// How many candidates this round measured. Bonferroni divides `alpha` by it — the
    /// only two levers on expected false accepts are shrinking this and shrinking the
    /// per-candidate error rate, and shrinking this is free.
    pub candidates: usize,
    /// How much better than the **re-measured** incumbent a candidate must be, in judge
    /// units. A tie keeps the incumbent.
    pub min_margin: f64,
    /// How much worse than the better of (incumbent, plain) the WER may get.
    pub max_wer_regression: f64,
    /// Floor on cosine to the plain render's speaker.
    ///
    /// Without this, "Speak like a frightened old man" is a legitimate winner: it changes
    /// delivery, moves `fearful`, keeps the words — and destroys the requested voice.
    pub min_speaker_similarity: f64,
    /// How many decisions a holdout partition may serve before it must be replaced.
    ///
    /// A holdout used repeatedly stops being one: every accept/reject leaks a bit about it.
    /// This is the standard way held-out sets rot in an iterated loop, and it is bounded
    /// here rather than left to discipline.
    pub max_holdout_uses: usize,
}

impl Default for TuneThresholds {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            candidates: 1,
            min_margin: 1.0,
            max_wer_regression: 0.02,
            min_speaker_similarity: 0.95,
            max_holdout_uses: 5,
        }
    }
}

impl TuneThresholds {
    pub fn corrected_alpha(&self) -> f64 {
        self.alpha / self.candidates.max(1) as f64
    }
}

/// Why a candidate was not accepted. One variant per criterion, so a rejection explains
/// itself instead of being a bare `false`.
#[derive(Debug, Clone, PartialEq)]
pub enum Reject {
    /// Did not beat plain acoustically.
    NoAcousticChange { p: f64, alpha: f64 },
    /// Beat plain but not the sham — perturbation, not content.
    PerturbationOnly { p_vs_sham: f64, alpha: f64 },
    /// The judge delta is inside its own noise.
    InsideJudgeNoise { delta: f64, noise: f64 },
    /// Beat the incumbent, but by less than the required margin.
    NoMarginOverIncumbent { candidate: f64, incumbent: f64, margin: f64 },
    /// Some other class gained more than the cued one.
    WrongClassGainedMost { gained: String, wanted: String },
    /// Intelligibility regressed.
    WerRegressed { wer: f64, limit: f64 },
    /// The requested voice was not preserved.
    SpeakerDrift { similarity: f64, floor: f64 },
    /// Passed selection but not confirmation.
    FailedOnHoldout { reason: Box<Reject> },
    /// No holdout measurement at all — fails closed.
    NotConfirmedOnHoldout,
    /// The phrase itself is unsafe to ship.
    UnsafePhrase { why: String },
}

/// Why an entire round was thrown away.
#[derive(Debug, Clone, PartialEq)]
pub enum Void {
    /// A delivery-neutral phrase beat plain. Every number in the round is uninterpretable.
    ShamActivated { p: f64, alpha: f64 },
    /// The WRONG label's phrase scored better on the target class than the incumbent. The
    /// judge is not tracking what it is supposed to track.
    CounterCueWon { counter_delta: f64, incumbent_delta: f64 },
    /// The holdout partition has served its budget and must be replaced by a human.
    HoldoutExpired { uses: usize, budget: usize },
    /// The incumbent was not re-measured in this round. Comparing against a stored number
    /// would let driver drift or a GPU change masquerade as an improvement.
    IncumbentNotRemeasured,
}

/// What a round concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct TuneReport {
    pub corrected_alpha: f64,
    /// Set iff the whole round is void. When present, `proposed` is always `None`.
    pub void: Option<Void>,
    /// The single winner, if any. A proposal, never an acceptance.
    pub proposed: Option<Candidate>,
    /// Every candidate that lost, with its reason.
    pub rejected: Vec<(Candidate, Reject)>,
}

impl TuneReport {
    /// A proposal exists. Deliberately not called `passed` — nothing here accepts anything.
    pub fn has_proposal(&self) -> bool {
        self.proposed.is_some()
    }
}

/// Is this phrase safe to put in front of a backend?
///
/// The CLAUDE.md hard invariant applies with full force: an instruct string is prose that
/// reaches the model, so a `[` in it is exactly the cue-markup leak the invariant forbids,
/// and a generated phrase is not reviewed by anyone before it is rendered.
pub fn phrase_is_safe(phrase: &str) -> Result<(), String> {
    let p = phrase.trim();
    if p.is_empty() {
        return Err("empty".into());
    }
    if p != phrase {
        return Err("leading or trailing whitespace".into());
    }
    if p.contains('\n') {
        return Err("multi-line".into());
    }
    if p.chars().count() > 120 {
        return Err(format!("too long ({} chars, limit 120)", p.chars().count()));
    }
    for bad in ['[', ']', '<', '>', '|'] {
        if p.contains(bad) {
            return Err(format!("contains {bad:?}"));
        }
    }
    if p.contains("endofprompt") {
        return Err("carries a prompt delimiter".into());
    }
    Ok(())
}

/// Does this measurement clear every criterion? `incumbent_delta` is the re-measured
/// incumbent's judge delta on the same split.
fn check(m: &TuneMeasurement, incumbent_delta: f64, th: &TuneThresholds) -> Result<(), Reject> {
    let alpha = th.corrected_alpha();
    // (i) it changed the delivery, and by more than a meaningless string of the same size.
    if m.p_vs_plain > alpha {
        return Err(Reject::NoAcousticChange { p: m.p_vs_plain, alpha });
    }
    if m.p_vs_sham > alpha {
        return Err(Reject::PerturbationOnly { p_vs_sham: m.p_vs_sham, alpha });
    }
    // (ii) the judge saw a move larger than its own across-seed noise.
    if m.cued_delta <= m.cued_noise {
        return Err(Reject::InsideJudgeNoise { delta: m.cued_delta, noise: m.cued_noise });
    }
    // (iii) and it is the CUED class that gained most, not merely some class.
    if m.largest_gain_label != m.label {
        return Err(Reject::WrongClassGainedMost {
            gained: m.largest_gain_label.clone(),
            wanted: m.label.clone(),
        });
    }
    // (iv) it beats the incumbent by a margin. Ties keep the incumbent.
    if m.cued_delta < incumbent_delta + th.min_margin {
        return Err(Reject::NoMarginOverIncumbent {
            candidate: m.cued_delta,
            incumbent: incumbent_delta,
            margin: th.min_margin,
        });
    }
    // (v) the words survived.
    if m.wer > th.max_wer_regression {
        return Err(Reject::WerRegressed { wer: m.wer, limit: th.max_wer_regression });
    }
    // (vi) and so did the voice.
    if m.speaker_similarity < th.min_speaker_similarity {
        return Err(Reject::SpeakerDrift {
            similarity: m.speaker_similarity,
            floor: th.min_speaker_similarity,
        });
    }
    Ok(())
}

/// Decide one tuning round.
///
/// `holdout_uses` is how many decisions this holdout partition has already served.
pub fn decide(
    measurements: &[TuneMeasurement],
    holdout_uses: usize,
    th: &TuneThresholds,
) -> TuneReport {
    let alpha = th.corrected_alpha();
    let void = |v: Void| TuneReport {
        corrected_alpha: alpha,
        void: Some(v),
        proposed: None,
        rejected: Vec::new(),
    };

    // ---- round-level voids, checked before anything is scored.
    if holdout_uses >= th.max_holdout_uses {
        return void(Void::HoldoutExpired { uses: holdout_uses, budget: th.max_holdout_uses });
    }
    let on = |arm: TuneArm, split: Split| {
        measurements.iter().find(|m| m.arm == arm && m.split == split)
    };
    if let Some(s) = on(TuneArm::Sham, Split::Tune) {
        if s.p_vs_plain <= alpha {
            return void(Void::ShamActivated { p: s.p_vs_plain, alpha });
        }
    }
    let Some(inc) = on(TuneArm::Incumbent, Split::Tune) else {
        return void(Void::IncumbentNotRemeasured);
    };
    if let Some(cc) = on(TuneArm::CounterCue, Split::Tune) {
        if cc.cued_delta > inc.cued_delta {
            return void(Void::CounterCueWon {
                counter_delta: cc.cued_delta,
                incumbent_delta: inc.cued_delta,
            });
        }
    }

    // ---- score the candidates on the tune split, then confirm survivors on the holdout.
    let holdout: BTreeMap<&str, &TuneMeasurement> = measurements
        .iter()
        .filter(|m| m.arm == TuneArm::Candidate && m.split == Split::Holdout)
        .filter_map(|m| m.phrase.as_deref().map(|p| (p, m)))
        .collect();
    let inc_holdout = on(TuneArm::Incumbent, Split::Holdout).map(|m| m.cued_delta);

    let mut passed: Vec<(&TuneMeasurement, Candidate)> = Vec::new();
    let mut rejected = Vec::new();
    for m in measurements
        .iter()
        .filter(|m| m.arm == TuneArm::Candidate && m.split == Split::Tune)
    {
        let Some(phrase) = m.phrase.clone() else { continue };
        let cand = Candidate { label: m.label.clone(), phrase: phrase.clone() };

        if let Err(why) = phrase_is_safe(&phrase) {
            rejected.push((cand, Reject::UnsafePhrase { why }));
            continue;
        }
        if let Err(r) = check(m, inc.cued_delta, th) {
            rejected.push((cand, r));
            continue;
        }
        // Confirmation. A candidate with no holdout measurement fails CLOSED — an absent
        // measurement must never read as a pass.
        let Some(h) = holdout.get(phrase.as_str()) else {
            rejected.push((cand, Reject::NotConfirmedOnHoldout));
            continue;
        };
        if let Err(r) = check(h, inc_holdout.unwrap_or(f64::NEG_INFINITY), th) {
            rejected.push((cand, Reject::FailedOnHoldout { reason: Box::new(r) }));
            continue;
        }
        passed.push((m, cand));
    }

    // The best surviving candidate by holdout delta — the split that was not optimised on.
    passed.sort_by(|a, b| {
        let (da, db) = (
            holdout.get(a.1.phrase.as_str()).map_or(f64::MIN, |m| m.cued_delta),
            holdout.get(b.1.phrase.as_str()).map_or(f64::MIN, |m| m.cued_delta),
        );
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1))
    });

    TuneReport {
        corrected_alpha: alpha,
        void: None,
        proposed: passed.first().map(|(_, c)| c.clone()),
        rejected,
    }
}

// ---------------------------------------------------------------------------------------
// Per-sentence decisions (added 2026-09-07)
// ---------------------------------------------------------------------------------------
//
// `decide` above evaluates one pooled measurement per arm. That is what the first two
// tuning rounds did, and it does not work: the tuner pooled 3 sentences x 4 seeds into a
// single permutation test, and **the incumbent itself could not clear the acoustic bar**
// (p=0.0166 on the tune split, 0.3580 on the holdout) — while the same phrase, measured
// per sentence at n=8, scores 0.0003 to 0.0057.
//
// More samples (12 pooled vs 8 per sentence) producing WORSE significance is the signature
// of inflated variance, and the three sentence-dependence sweeps say exactly where it comes
// from: sentences differ enough to swamp the cue. Pooling buries the signal it is meant to
// measure, and a gate the incumbent cannot pass can accept nothing — the failure ADR-0003
// named for the 0.85 activation floor, arrived at from the other direction.
//
// So the renders are no longer pooled. Each sentence is tested on its own and the
// DECISIONS are aggregated: a candidate must clear every criterion on at least
// `min_sentences` of them. That also makes "works on some sentences" expressible, which
// the sweeps showed is the actual shape of the phenomenon.
//
// Everything here is additive. `decide`, `TuneMeasurement` and `TuneThresholds` keep their
// exact signatures, so `tests/cue_tune_decision.rs` stays frozen and green.

/// One arm's measurement on one specific sentence.
#[derive(Debug, Clone, PartialEq)]
pub struct SentenceMeasurement {
    /// Sentence id. Grouping key — never pooled across.
    pub sentence: String,
    pub m: TuneMeasurement,
}

/// A per-sentence verdict for one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct PerSentence {
    pub sentence: String,
    pub passed: bool,
    pub reject: Option<Reject>,
}

/// What a per-sentence round concluded.
#[derive(Debug, Clone, PartialEq)]
pub struct PerSentenceReport {
    pub corrected_alpha: f64,
    pub void: Option<Void>,
    pub proposed: Option<Candidate>,
    /// Every candidate, with how many sentences it cleared and why it failed the rest.
    pub detail: Vec<(Candidate, usize, Vec<PerSentence>)>,
    /// How many sentences a candidate had to clear.
    pub required: usize,
}

impl PerSentenceReport {
    pub fn has_proposal(&self) -> bool {
        self.proposed.is_some()
    }
}

fn arm_on<'a>(
    ms: &'a [SentenceMeasurement],
    sentence: &str,
    arm: TuneArm,
    split: Split,
) -> Option<&'a TuneMeasurement> {
    ms.iter()
        .find(|s| s.sentence == sentence && s.m.arm == arm && s.m.split == split)
        .map(|s| &s.m)
}

/// Decide a round from per-sentence measurements.
///
/// `min_sentences` is how many sentences a candidate must clear on **each** split. It is
/// pre-registered like everything else: choosing it after seeing the results would be
/// picking the threshold that admits the answer you liked.
pub fn decide_per_sentence(
    ms: &[SentenceMeasurement],
    holdout_uses: usize,
    th: &TuneThresholds,
    min_sentences: usize,
) -> PerSentenceReport {
    let alpha = th.corrected_alpha();
    let void = |v: Void| PerSentenceReport {
        corrected_alpha: alpha,
        void: Some(v),
        proposed: None,
        detail: Vec::new(),
        required: min_sentences,
    };

    if holdout_uses >= th.max_holdout_uses {
        return void(Void::HoldoutExpired { uses: holdout_uses, budget: th.max_holdout_uses });
    }

    // Tune and holdout use DIFFERENT sentences — that is what a holdout is — so their ids
    // are disjoint and must be partitioned, not unioned. A first version took the union and
    // then demanded a Tune incumbent for every id, which voided every real round on its
    // holdout ids. The synthetic tests missed it because they reused one id set across both
    // splits, modelling a structure the driver does not produce.
    let ids = |split: Split| {
        let mut v: Vec<&str> = ms
            .iter()
            .filter(|s| s.m.split == split)
            .map(|s| s.sentence.as_str())
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    };
    let sentences = ids(Split::Tune);
    let holdout_sentences = ids(Split::Holdout);

    // ---- round-level voids, evaluated per sentence. A sham that activates ANYWHERE at the
    // corrected alpha voids the round: multiplicity is already paid for in the correction,
    // so one hit is one hit too many.
    for s in &sentences {
        if let Some(sh) = arm_on(ms, s, TuneArm::Sham, Split::Tune) {
            if sh.p_vs_plain <= alpha {
                return void(Void::ShamActivated { p: sh.p_vs_plain, alpha });
            }
        }
        if arm_on(ms, s, TuneArm::Incumbent, Split::Tune).is_none() {
            return void(Void::IncumbentNotRemeasured);
        }
    }
    // The counter-cue must lose on a majority. Requiring it to lose everywhere would let a
    // single noisy sentence discard an otherwise sound round.
    let cc_wins = sentences
        .iter()
        .filter(|s| {
            match (
                arm_on(ms, s, TuneArm::CounterCue, Split::Tune),
                arm_on(ms, s, TuneArm::Incumbent, Split::Tune),
            ) {
                (Some(c), Some(i)) => c.cued_delta > i.cued_delta,
                _ => false,
            }
        })
        .count();
    if cc_wins * 2 > sentences.len() {
        return void(Void::CounterCueWon {
            counter_delta: cc_wins as f64,
            incumbent_delta: sentences.len() as f64,
        });
    }

    // ---- score each candidate, per sentence, per split.
    let mut phrases: Vec<&str> = ms
        .iter()
        .filter(|s| s.m.arm == TuneArm::Candidate)
        .filter_map(|s| s.m.phrase.as_deref())
        .collect();
    phrases.sort_unstable();
    phrases.dedup();

    let mut detail = Vec::new();
    let mut winners: Vec<(Candidate, usize, f64)> = Vec::new();
    for phrase in phrases {
        let cand = Candidate {
            label: ms
                .iter()
                .find(|s| s.m.phrase.as_deref() == Some(phrase))
                .map(|s| s.m.label.clone())
                .unwrap_or_default(),
            phrase: phrase.to_string(),
        };
        if let Err(why) = phrase_is_safe(phrase) {
            detail.push((cand, 0, vec![PerSentence {
                sentence: "*".into(),
                passed: false,
                reject: Some(Reject::UnsafePhrase { why }),
            }]));
            continue;
        }

        let mut per = Vec::new();
        let (mut tune_ok, mut hold_ok, mut hold_total) = (0usize, 0usize, 0.0f64);

        // Holdout is scored over ITS OWN sentences, separately.
        for s in &holdout_sentences {
            let hm = ms.iter().find(|x| {
                x.sentence == **s
                    && x.m.arm == TuneArm::Candidate
                    && x.m.split == Split::Holdout
                    && x.m.phrase.as_deref() == Some(phrase)
            });
            let hi = arm_on(ms, s, TuneArm::Incumbent, Split::Holdout).map(|m| m.cued_delta);
            if let (Some(h), Some(i)) = (hm, hi) {
                if check(&h.m, i, th).is_ok() {
                    hold_ok += 1;
                    hold_total += h.m.cued_delta;
                }
            }
        }

        for s in &sentences {
            let tune = arm_on(ms, s, TuneArm::Candidate, Split::Tune)
                .filter(|m| m.phrase.as_deref() == Some(phrase))
                .or_else(|| {
                    ms.iter()
                        .find(|x| {
                            x.sentence == **s
                                && x.m.arm == TuneArm::Candidate
                                && x.m.split == Split::Tune
                                && x.m.phrase.as_deref() == Some(phrase)
                        })
                        .map(|x| &x.m)
                });
            let inc = arm_on(ms, s, TuneArm::Incumbent, Split::Tune).map(|m| m.cued_delta);
            let r = match (tune, inc) {
                (Some(t), Some(i)) => check(t, i, th).err(),
                _ => Some(Reject::NotConfirmedOnHoldout),
            };
            per.push(PerSentence {
                sentence: (*s).to_string(),
                passed: r.is_none(),
                reject: r,
            });
            if per.last().unwrap().passed {
                tune_ok += 1;
            }
        }
        let passed = tune_ok >= min_sentences && hold_ok >= min_sentences;
        detail.push((cand.clone(), tune_ok, per));
        if passed {
            winners.push((cand, hold_ok, hold_total));
        }
    }

    // Rank by holdout breadth first, then holdout magnitude — both from the split that was
    // not selected on.
    winners.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
            .then(a.0.cmp(&b.0))
    });

    PerSentenceReport {
        corrected_alpha: alpha,
        void: None,
        proposed: winners.first().map(|(c, _, _)| c.clone()),
        detail,
        required: min_sentences,
    }
}
