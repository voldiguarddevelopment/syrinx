### T-00.01  Scaffold the Cargo workspace
id: T-00.01
phase: 0
status: done
depends_on: []
stack: rust
criteria:
  - C1: `cargo build` at the workspace root exits 0 with `[workspace]` and `resolver = "2"` in the root Cargo.toml.
  - C2: the workspace declares the eleven member crates syrinx-frontend, syrinx-core, syrinx-lm, syrinx-speaker, syrinx-acoustic, syrinx-vocoder, syrinx-prosody, syrinx-stream, syrinx-serve, syrinx-eval, syrinx-cli.
  - C3: `cargo test` exits 0 across the workspace with zero tests defined.
not_doing:
  - No crate internals beyond an empty lib/bin target each.
  - No dependency wiring beyond what empty crates need to build.
test_files: [tests/workspace_scaffold.rs]
criteria_map:
  C1: [test_root_cargo_declares_workspace_table, test_root_cargo_sets_resolver_two, test_member_crates_have_buildable_target]
  C2: [test_workspace_lists_all_eleven_members, test_all_eleven_member_crates_have_manifests, test_workspace_has_exactly_eleven_members]
  C3: [test_member_crates_define_no_unit_tests, test_member_crates_have_no_integration_test_dirs]
attempts: 2
last_failure: ""
---
The root surface every other task attaches to. Inputs: Cargo manifests only. Outputs: a compiling eleven-crate workspace and a green empty test run. Errors/edges: a manifest that fails to parse is the only failure, surfaced by cargo. Invariant: the workspace compiles from here forward. Done-check: the three cargo-observable criteria.

### T-00.02  Wire the CI gate pipeline
id: T-00.02
phase: 0
status: blocked
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: a pull request with a `cargo fmt --check`, `cargo clippy -D warnings`, `cargo build`, or `cargo test` failure is reported as a failed required check and is non-mergeable.
  - C2: a pull request that edits any frozen-eval-set file without updating the checksum manifest fails the frozen-eval gate job; an unmodified set passes it.
  - C3: a pull request that passes fmt, clippy, build, test, and the frozen-eval gate is reported mergeable with every required check green.
not_doing:
  - No deployment, release, or publishing stages.
  - No self-hosted runner provisioning or secrets management.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The merge gate over the whole workspace. Inputs: a PR against the GitHub repo. Bounds: five required checks (fmt, clippy, build, test, frozen-eval). Outputs: a pass/fail status per check that gates merge. Errors/edges: any one failing check blocks merge; the frozen-eval gate keys off the T-00.04 manifest. Invariant: red anywhere blocks merge. Done-check: PR mergeability flips with check status. BLOCKED: needs the GitHub repository and Actions infrastructure plus human repo-admin configuration of required checks, which the autonomous loop cannot provision.

### T-00.03  Build the eval-harness skeleton
id: T-00.03
phase: 0
status: done
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: `syrinx-eval`'s harness entry function, run against a stub synth input over the default metric set, writes a metrics JSON whose top-level object contains exactly the five keys `sim_o`, `wer`, `mos_proxy`, `ttfb_ms`, `rtf`.
  - C2: every value the stub run records is a finite number — `f64::is_finite` is true for each present metric and no value is NaN or infinite.
  - C3: a metric whose plugged-in implementation yields no result is serialized as JSON `null` for its key, and that key is still present in the object (recorded as null, never omitted).
  - C4: registering a metric set that omits one of the five keys causes the harness to return a typed error naming the missing key rather than writing a partial JSON.
not_doing:
  - No real SIM-o/WER/MOS/latency computation — the stub input and stub metrics only.
  - No audio decoding, model loading, or GPU work.
test_files: [tests/eval_harness.rs]
criteria_map:
  C1: [test_default_run_writes_all_five_keys, test_default_run_writes_no_keys_beyond_the_five, test_required_keys_constant_is_exactly_the_five]
  C2: [test_default_run_values_are_all_finite_numbers]
  C3: [test_metric_yielding_none_is_serialized_as_null, test_metrics_with_results_are_not_null, test_null_metric_run_still_has_exactly_five_keys]
  C4: [test_complete_set_does_not_error, test_missing_key_returns_typed_error_naming_it, test_missing_key_error_names_the_actual_omitted_key, test_missing_key_writes_no_partial_file]
attempts: 3
last_failure: ""
---
The eval substrate that later real metrics plug into. Inputs: a stub synth input and a pluggable metric set. Bounds: the five fixed keys sim_o, wer, mos_proxy, ttfb_ms, rtf. Outputs: a metrics JSON object with all five keys present, finite numbers or explicit null. Errors/edges: an absent metric is null not omitted; a metric set missing a required key is a typed error, not a partial write. Invariant: the JSON schema (five keys, present-or-null) holds for every run. Done-check: the four criteria over the stub run and the missing-key path.

### T-00.04  Checksum the frozen eval set
id: T-00.04
phase: 0
status: done
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: `syrinx-eval`'s manifest builder, given an eval-set directory, writes a checksum manifest mapping each file's relative path to its lowercase-hex SHA-256 digest, with one entry per file in the set.
  - C2: `verify()` returns `Ok(())` when every file's recomputed SHA-256 equals its manifest digest and the set's file membership matches the manifest exactly.
  - C3: editing any byte of a manifested file makes `verify()` return a typed error variant that names the offending file's path; an unchanged set never returns that error.
  - C4: a file present in the manifest but missing from the directory (or present on disk but absent from the manifest) makes `verify()` return a typed membership-mismatch error naming the path, not `Ok`.
not_doing:
  - No encryption, signing, or the eval audio clips/transcripts themselves.
  - No hashing of files outside the declared eval-set directory.
test_files: [tests/eval_manifest.rs]
criteria_map:
  C1: [test_manifest_has_exactly_one_entry_per_file, test_manifest_keys_are_the_relative_paths, test_manifest_digests_are_lowercase_hex_sha256, test_written_manifest_round_trips_through_disk]
  C2: [test_verify_ok_on_unchanged_set, test_verify_ok_after_round_trip_through_disk]
  C3: [test_tampered_byte_yields_digest_mismatch_naming_file, test_unchanged_set_is_not_a_digest_mismatch]
  C4: [test_missing_file_yields_membership_mismatch_naming_path, test_extra_file_yields_membership_mismatch_naming_path, test_membership_drift_is_not_reported_as_ok, test_unchanged_set_is_not_a_membership_mismatch]
attempts: 2
last_failure: ""
---
The immutability mechanism, not the audio set. Inputs: an eval-set directory of files. Bounds: SHA-256 per file, full-membership check. Outputs: a checksum manifest and a verify() verdict. Errors/edges: a tampered byte names the file in a typed error; a missing or extra file is a typed membership-mismatch naming the path; only a byte-identical, membership-identical set returns Ok. Invariant: verify() is Ok iff the set is byte-for-byte and membership-identical to the manifest. Done-check: the four criteria over clean, tampered, and membership-drift cases.

### T-00.05  Screen base-model licenses
id: T-00.05
phase: 0
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a committed matrix document scores each candidate base (Chatterbox, CosyVoice2, F5) across license, parameter count, streaming, cloning, and multilingual columns, with one row per candidate.
  - C2: each candidate's license is classified against the project's redistribution and commercial-use requirements, and any disqualifying license is explicitly flagged in the matrix.
  - C3: the matrix records a recommended shortlist of license-compatible candidates eligible to proceed to the A/B bench.
not_doing:
  - No running, downloading, or benchmarking of the candidate models.
  - No final base-model selection (that is the A/B bench's job).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The license/architecture screen that gates which bases may be benched. Inputs: public license and model-card facts for each candidate. Bounds: the five matrix columns. Outputs: a committed matrix doc with disqualifiers flagged and a compatible shortlist. Errors/edges: an ambiguous or non-redistributable license must be flagged, not assumed permissive. Invariant: only license-compatible candidates advance. Done-check: matrix completeness and flagged disqualifiers. BLOCKED: needs human legal/licensing judgment to classify each license against the project's redistribution and commercial terms, which the loop must not adjudicate.

### T-00.06  Bench candidate base models
id: T-00.06
phase: 0
status: blocked
depends_on: [T-00.03, T-00.05]
stack: rust
criteria:
  - C1: each shortlisted base is run end-to-end over the frozen eval set and produces SIM-o, WER, and latency numbers recorded through the `syrinx-eval` harness.
  - C2: the candidates are ranked by the recorded metrics and the selected base is written into `ARCHITECTURE.md` with its scores.
  - C3: the recorded per-candidate metrics are reproducible to within the harness's stated tolerance on a re-run at a fixed seed.
not_doing:
  - No Rust reimplementation of any candidate's inference.
  - No fine-tuning or quantization of the candidates.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The A/B bench that selects the base. Inputs: the license-screen shortlist and the frozen eval set. Bounds: SIM-o, WER, latency via the harness. Outputs: a ranked result table and a recorded selection in ARCHITECTURE.md. Errors/edges: a candidate that fails to run is recorded as a disqualification, not silently dropped. Invariant: the selection is the metric-ranked winner. Done-check: ranked results plus the recorded decision. BLOCKED: needs the real candidate models running on a GPU to produce SIM-o/WER/latency, which is not expressible as a frozen-test gate and the loop must never attempt.

### T-00.07  Reproduce reference Python inference
id: T-00.07
phase: 0
status: blocked
depends_on: [T-00.06]
stack: rust
criteria:
  - C1: a Python reference script renders audio end-to-end from a fixed text plus reference clip using the selected base model.
  - C2: two runs at the same pinned seed produce bit-identical (or within stated numerical tolerance) output audio.
  - C3: the script pins model revision, seed, and dependency versions such that a third party can reproduce the audio from the committed config alone.
not_doing:
  - No Rust port of the inference path (that is Phase 2).
  - No quality tuning beyond deterministic reproduction.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The deterministic Python reference the Rust port is later checked against. Inputs: fixed text, a reference clip, the selected base, a pinned seed. Bounds: determinism within stated tolerance. Outputs: reproducible reference audio and a pinned config. Errors/edges: nondeterministic output at a fixed seed is a failure. Invariant: same inputs and seed yield the same audio. Done-check: bit/tolerance-stable repeat runs from the committed config. BLOCKED: needs a GPU and the chosen base model's weights to run real inference, which the autonomous loop cannot execute or gate.

### T-00.08  Write ARCHITECTURE.md v0
id: T-00.08
phase: 0
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a committed `ARCHITECTURE.md` documents the eleven-crate map and the per-crate responsibility and boundary for every crate in the workspace.
  - C2: the document records the resolved design decisions (the per-stage paradigm, chunk-aware streaming, separated prosody plan, quantization target) with their rationale.
  - C3: the document states the cross-crate interface contracts and the dataflow from text through frontend, LM, prosody plan, decoder, and vocoder to audio.
not_doing:
  - No CLAUDE.md rewrite (it already exists).
  - No code or interface implementation beyond the documented contracts.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The architecture record the rest of the build conforms to. Inputs: the resolved design decisions and crate map. Bounds: all eleven crates and the end-to-end dataflow. Outputs: a committed ARCHITECTURE.md v0. Errors/edges: a crate left undocumented or a decision left without rationale is incomplete. Invariant: the doc matches the workspace crate set. Done-check: crate-map, decisions, and contracts all present. BLOCKED: needs human architecture decisions (final paradigm and contract choices) that are judgment calls the loop must not invent.

### T-00.09  Define the consent watermarking policy
id: T-00.09
phase: 0
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a committed policy document defines the consent requirement governing any voice-cloning use before such output may ship.
  - C2: the document mandates an audible-or-inaudible watermark on every cloned-voice output and states the detectability requirement that gates release.
  - C3: the document defines the prohibited-use and misuse-reporting policy and ties watermarking and consent to the Phase 2 watermark-embedding and release tasks.
not_doing:
  - No watermark embedding implementation (that is Phase 2).
  - No legal contract drafting beyond the usage-policy statement.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The ethics gate that must precede any cloning ship. Inputs: the project's consent and misuse posture. Bounds: consent, watermark requirement, prohibited use. Outputs: a committed policy document binding later release tasks. Errors/edges: a cloning output without consent or watermark coverage is out of policy. Invariant: no cloned output ships outside this policy. Done-check: consent, watermark, and prohibited-use clauses all present. BLOCKED: needs human and legal judgment to set consent and watermarking policy, which the loop must not author unilaterally.

### T-01.01  Normalize text
id: T-01.01
phase: 1
status: done
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::normalize::normalize("Café\u{0301}")` returns "café" in NFC (the precomposed U+00E9), so the output byte length is 5 not 6 and `is_nfc` holds.
  - C2: `normalize("  Hello\tWorld \r\n")` collapses runs of whitespace to single U+0020 spaces and trims ends, returning exactly "Hello World" (no leading/trailing space, no tab, no CR/LF).
  - C3: a repo-root integration test `tests/normalize_golden.rs` reads (input,expected) pairs from the repo-root `tests/golden/normalize/` and asserts `syrinx_frontend::normalize::normalize` reproduces each expected output byte-for-byte; mutating any single expected file's bytes makes that case fail.
  - C4: `normalize` preserves intra-word casing and does NOT lowercase, so `normalize("iPhone XR")` returns "iPhone XR" (casing folding is a separate, opt-in concern, off by default).
not_doing:
  - No number/date/currency expansion (that is T-01.02).
  - No language-specific transliteration or accent stripping.
test_files: [tests/golden/normalize/casing_preserved.expected, tests/golden/normalize/casing_preserved.in, tests/golden/normalize/nfc_accent.expected, tests/golden/normalize/nfc_accent.in, tests/golden/normalize/nfc_caps_preserved.expected, tests/golden/normalize/nfc_caps_preserved.in, tests/golden/normalize/ws_collapse.expected, tests/golden/normalize/ws_collapse.in, tests/golden/normalize/ws_internal_runs.expected, tests/golden/normalize/ws_internal_runs.in, tests/normalize.rs, tests/normalize_golden.rs]
criteria_map:
  C1: [nfc_composes_combining_acute_to_precomposed, nfc_handles_lone_combining_mark_without_panic]
  C2: [collapses_whitespace_runs_and_trims_ends, collapsed_output_has_no_other_whitespace_or_runs, single_interior_space_is_preserved, all_whitespace_trims_to_empty]
  C3: [golden_cases_match_expected_bytes, golden_corpus_is_non_empty]
  C4: [preserves_intra_word_casing, does_not_lowercase_mixed_case]
attempts: 1
last_failure: ""
---
The deterministic entry point of the frontend. Inputs: an arbitrary `&str` of user text, bounded only by available memory. Outputs: a `String` in Unicode NFC with whitespace runs collapsed to single ASCII spaces and ends trimmed, casing untouched. Errors/edges: empty input returns the empty string; lone combining marks and mixed CR/LF/tab all normalize without panic. Invariant: `normalize` is idempotent — `normalize(normalize(x)) == normalize(x)`. Done-check: the four criteria, the golden suite, and the idempotence property test.

### T-01.02  Expand numbers
id: T-01.02
phase: 1
status: done
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::expand::expand_numbers("$1,200")` returns "one thousand two hundred dollars", and `expand_numbers("$1")` returns "one dollar" (singular), pinning the plural boundary.
  - C2: `expand_numbers("1/2/26")` returns "January second twenty twenty-six" (month/day/year) and `expand_numbers("3.14")` returns "three point one four" (decimal digits read individually), distinguishing date from decimal.
  - C3: `expand_numbers("1st")` returns "first", `expand_numbers("2nd")` returns "second", and `expand_numbers("23rd")` returns "twenty-third", covering the ordinal suffixes st/nd/rd/th.
  - C4: a bare integer `expand_numbers("42")` returns "forty-two" (cardinal, hyphenated) while text with no numeric token, `expand_numbers("hello")`, is returned unchanged as "hello".
not_doing:
  - No currency other than USD `$`; no localized number formats.
  - No Roman-numeral or phone-number expansion.
test_files: [tests/expand_numbers.rs]
criteria_map:
  C1: [currency_thousands_is_plural_dollars, currency_one_is_singular_dollar, currency_two_is_plural_dollars]
  C2: [date_mdy_reads_month_ordinal_day_and_year, decimal_reads_digits_individually, decimal_integer_part_is_cardinal, out_of_range_date_falls_back_to_cardinal_without_panic]
  C3: [ordinal_st_suffix, ordinal_nd_suffix, ordinal_rd_suffix_two_digit_hyphenated, ordinal_th_suffix]
  C4: [bare_integer_is_hyphenated_cardinal, round_ten_cardinal_has_no_hyphen, non_numeric_text_passes_through_unchanged, empty_input_passes_through]
attempts: 1
last_failure: ""
---
Numeric verbalization over already-normalized text. Inputs: a `&str` that may contain currency, dates, decimals, ordinals, and cardinals. Outputs: a `String` with each numeric token replaced by its spoken English form, non-numeric spans passed through verbatim. Errors/edges: singular vs plural currency, date vs decimal disambiguation, and ordinal suffix selection are all pinned on both sides; an out-of-range date component yields the cardinal fallback rather than a panic. Invariant: tokens with no numeric content are byte-identical in the output. Done-check: the four concrete input→output criteria plus a passthrough case.

### T-01.03  Override pronunciations via lexicon
id: T-01.03
phase: 1
status: done
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: with a default lexicon mapping "tomato"→"tom-ah-to" and a user lexicon mapping "tomato"→"tom-ay-to", `Lexicon::with_user(user).lookup("tomato")` returns Some("tom-ay-to"), proving user precedence over default.
  - C2: for a key present ONLY in the default lexicon, e.g. "data"→"day-ta", `lookup("data")` returns Some("day-ta"); for a key in NEITHER, `lookup("zzqx")` returns None.
  - C3: lookup is case-folded on the key so `lookup("Tomato")` and `lookup("tomato")` both resolve to the same Some entry, while the stored replacement value's casing is returned unaltered.
  - C4: an empty user lexicon leaves all default entries reachable — `Lexicon::with_user(empty).lookup("data")` still returns Some("day-ta").
not_doing:
  - No fuzzy/stemmed matching; exact case-folded key only.
  - No IPA validation of replacement strings (that is the G2P layer's concern).
test_files: [tests/lexicon.rs]
criteria_map:
  C1: [user_overrides_default_on_collision, default_value_wins_when_no_user_override]
  C2: [default_only_key_falls_through, missing_key_returns_none, missing_key_with_empty_user_returns_none]
  C3: [lookup_is_case_folded_on_key, default_key_lookup_is_case_folded, stored_value_casing_is_returned_unaltered]
  C4: [empty_user_keeps_defaults_reachable, empty_user_keeps_all_defaults_reachable]
attempts: 1
last_failure: ""
---
A two-tier override table consulted before phonemization. Inputs: a default lexicon and an optional user lexicon, each a map of word→pronunciation, plus a query word. Outputs: `Option<String>` — the winning replacement or None. Errors/edges: user beats default on collision; missing keys return None; case folding applies to keys but never to values. Invariant: precedence is total and deterministic — user ∪ default with user winning every tie. Done-check: the precedence, fallthrough, miss, and case-fold criteria.

### T-01.04  Phonemize to IPA
id: T-01.04
phase: 1
status: done
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: the trait `syrinx_frontend::g2p::Phonemizer` exposes `fn phonemize(&self, word: &str) -> String`, and for the known labeled word "cat" the default implementation returns exactly "kæt".
  - C2: a second known word "the" maps to "ðə"; the full fixed labeled set in `tests/golden/g2p/` round-trips word→IPA with every entry matching exactly.
  - C3: an out-of-vocabulary word "zorptquax" returns a non-empty IPA `String` (every output char is in the defined IPA symbol set) and does NOT panic, so OOV is always covered by the fallback path.
  - C4: `phonemize("")` returns an empty string and does not panic, pinning the empty-input boundary against the OOV path.
not_doing:
  - No stress/syllable-boundary marking beyond bare phoneme symbols.
  - No per-word IPA override (that is T-01.05) and no heteronym disambiguation (T-01.06).
test_files: [tests/g2p.rs, tests/golden/g2p/cat.expected, tests/golden/g2p/cat.in, tests/golden/g2p/fish.expected, tests/golden/g2p/fish.in, tests/golden/g2p/ship.expected, tests/golden/g2p/ship.in, tests/golden/g2p/sun.expected, tests/golden/g2p/sun.in, tests/golden/g2p/the.expected, tests/golden/g2p/the.in, tests/golden/g2p/thin.expected, tests/golden/g2p/thin.in, tests/golden/g2p/van.expected, tests/golden/g2p/van.in]
criteria_map:
  C1: [cat_maps_to_known_ipa]
  C2: [the_maps_to_known_ipa, golden_set_is_non_empty, golden_labeled_set_round_trips]
  C3: [oov_word_is_non_empty, oov_word_chars_all_in_ipa_set]
  C4: [empty_input_maps_to_empty]
attempts: 1
last_failure: ""
---
The grapheme-to-phoneme interface and a deterministic default backend. Inputs: a single word `&str`. Outputs: an IPA `String` drawn from a closed symbol set. Errors/edges: known words hit the labeled table exactly; OOV words take a fallback that always yields valid non-empty IPA; the empty string maps to empty, never panicking. Invariant: `phonemize` is total — every `&str` produces a defined IPA string. Done-check: the known-word golden set, the OOV non-empty/valid-symbol guarantee, and the empty boundary.

### T-01.05  Map custom pronunciations
id: T-01.05
phase: 1
status: done
depends_on: [T-01.04]
stack: rust
criteria:
  - C1: given an override map {"syrinx" → "ˈsɪrɪŋks"}, `OverridingPhonemizer::new(base, map).phonemize("syrinx")` returns exactly "ˈsɪrɪŋks", replacing whatever the base G2P would produce.
  - C2: for a word NOT in the override map, e.g. "cat", `phonemize("cat")` returns the base phonemizer's output "kæt" unchanged, proving the override is consulted only on a hit.
  - C3: override matching is case-folded on the key so an override for "Syrinx" still applies to `phonemize("syrinx")`, returning the mapped IPA exactly.
  - C4: an empty override map makes `OverridingPhonemizer` behave identically to its base for every input, e.g. `phonemize("the")` returns "ðə".
not_doing:
  - No validation that override values are well-formed IPA.
  - No multi-word/phrase overrides; single-word keys only.
test_files: [tests/overrides.rs]
criteria_map:
  C1: [override_hit_returns_mapped_ipa_exactly, override_hit_replaces_base_output]
  C2: [override_miss_delegates_to_base, override_miss_matches_bare_base]
  C3: [override_key_is_case_folded, override_query_is_case_folded]
  C4: [empty_map_passes_known_word_through, empty_map_is_transparent_for_every_input]
attempts: 1
last_failure: ""
---
A decorator over any `Phonemizer` that substitutes per-word IPA. Inputs: a base phonemizer, a word→IPA override map, and a query word. Outputs: the mapped IPA on a hit, else the base output. Errors/edges: hits replace exactly; misses delegate untouched; keys are case-folded; an empty map is a transparent passthrough. Invariant: output equals `map.get(fold(word)).unwrap_or_else(|| base.phonemize(word))`. Done-check: the replace, passthrough, case-fold, and empty-map criteria.

### T-01.06  Resolve heteronyms
id: T-01.06
phase: 1
status: done
depends_on: [T-01.04]
stack: rust
criteria:
  - C1: `syrinx_frontend::hetero::resolve("I read the book yesterday")` selects the past-tense pronunciation "rɛd" for "read", while `resolve("I read books daily")` selects "riːd", disambiguating by tense context.
  - C2: `resolve("lead the way")` selects the verb "liːd" and `resolve("a lead pipe")` selects the noun "lɛd", pinning both sides of the lead heteronym.
  - C3: `resolve("take a bow")` selects "baʊ" and `resolve("a violin bow")` selects "boʊ" on the fixed test set, deterministically (same input always yields the same choice).
  - C4: a sentence with no heteronym, `resolve("the cat sat")`, returns the base phonemization with no substitution, leaving "cat" as "kæt".
not_doing:
  - No statistical/ML POS tagging; rule-based context disambiguation only.
  - No coverage beyond the fixed heteronym test set (read/lead/bow and the listed words).
test_files: [tests/hetero.rs]
criteria_map:
  C1: [read_past_tense_selects_red, read_present_tense_selects_reed, read_two_contexts_differ]
  C2: [lead_verb_selects_liid, lead_noun_selects_led, lead_two_contexts_differ]
  C3: [bow_take_selects_bau, bow_violin_selects_bou, bow_two_contexts_differ, resolution_is_deterministic]
  C4: [no_heteronym_leaves_cat_as_base, no_heteronym_passthrough_full_sequence]
attempts: 1
last_failure: ""
---
Context-sensitive selection among a word's candidate pronunciations. Inputs: a sentence `&str` containing zero or more heteronyms. Outputs: a per-word IPA sequence with each heteronym resolved by surrounding rule/POS context. Errors/edges: both readings of read, lead, and bow are pinned; a non-heteronym sentence passes through unchanged. Invariant: resolution is a pure deterministic function of the sentence — identical input yields identical output every call. Done-check: the three heteronym pairs plus the no-heteronym passthrough.

### T-01.07  Parse the SSML subset
id: T-01.07
phase: 1
status: done
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::ssml::parse("<break time=\"200ms\"/>")` returns `Ok` with one `ControlEvent::Break { ms: 200 }`, pinning the typed break event and its parsed duration.
  - C2: `parse("<emphasis level=\"strong\">hi</emphasis>")` yields events `[Emphasis{level: Strong}, Text("hi"), EmphasisEnd]` in order; `parse("<prosody rate=\"slow\">x</prosody>")`, `say-as`, `phoneme`, and `sub` each map to their typed variant on the fixed subset.
  - C3: malformed input `parse("<break time=\"200ms\">")` (unclosed void tag) returns `Err(SsmlError)` and never panics; an unknown tag `parse("<blink>x</blink>")` also returns `Err(SsmlError)`.
  - C4: plain text with no markup, `parse("hello world")`, returns `Ok(vec![ControlEvent::Text("hello world")])` — a single text event, no error.
not_doing:
  - Only the subset prosody/break/emphasis/say-as/phoneme/sub; any other SSML tag is an error, not silently ignored.
  - No DTD/namespace validation or external entity resolution.
test_files: [tests/ssml.rs]
criteria_map:
  C1: [break_void_tag_parses_to_single_break_event_200ms, break_void_tag_parses_distinct_duration_375ms]
  C2: [emphasis_strong_yields_open_text_end_in_order, prosody_rate_slow_maps_to_prosody_variant, say_as_maps_to_say_as_variant, phoneme_maps_to_phoneme_variant, sub_maps_to_sub_variant]
  C3: [unclosed_void_break_tag_is_error, unknown_tag_is_error]
  C4: [plain_text_becomes_single_text_event]
attempts: 1
last_failure: ""
---
A recursive-descent parser for the documented SSML subset into typed control events. Inputs: a `&str` of SSML or plain text. Outputs: `Result<Vec<ControlEvent>, SsmlError>`. Errors/edges: well-formed subset tags produce typed events in source order; malformed and out-of-subset tags return a typed `SsmlError`; plain text becomes a single `Text` event. Invariant: parsing is total — every input yields either `Ok(events)` or `Err`, never a panic. Done-check: the typed-event, multi-tag, malformed, and plain-text criteria.

### T-01.08  Map punctuation to prosody
id: T-01.08
phase: 1
status: done
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::punct::hints("Stop. Go")` emits a `ProsodyHint::Boundary { tone: Falling, strength: Full }` marker at the period, distinct from any comma marker.
  - C2: `hints("Wait, now")` emits `ProsodyHint::Break { kind: Short }` at the comma, and the period in C1 maps to a Full boundary not a Short break, pinning the comma↔period distinction.
  - C3: `hints("Really?")` emits `ProsodyHint::Boundary { tone: Rising }` for the question mark, while `hints("Stop!")` emits a Falling/exclamatory boundary, distinguishing rising from falling terminal tone.
  - C4: text with no punctuation, `hints("hello world")`, emits zero prosody markers (an empty marker list).
not_doing:
  - No semicolon/colon/dash handling beyond period, comma, question, exclamation.
  - No acoustic realization of the hints; markers are typed metadata only.
test_files: [tests/punct.rs]
criteria_map:
  C1: [test_period_is_full_falling_boundary, test_period_count_invariant, test_period_marker_is_not_a_comma_break]
  C2: [test_comma_is_short_break, test_comma_count_invariant, test_comma_period_distinction, test_mixed_punctuation_ordered_and_counted]
  C3: [test_question_is_rising_boundary, test_exclamation_is_falling_exclamatory_boundary, test_question_count_invariant, test_exclamation_count_invariant, test_question_rising_vs_exclamation_falling, test_exclamation_distinct_from_period]
  C4: [test_no_punctuation_yields_no_markers]
attempts: 1
last_failure: ""
---
A mapping from terminal/internal punctuation to typed prosody markers. Inputs: a normalized `&str`. Outputs: an ordered list of `ProsodyHint` markers keyed to punctuation positions. Errors/edges: period→full falling boundary, comma→short break, `?`→rising, `!`→falling exclamatory are each pinned against one another; unpunctuated text yields no markers. Invariant: the marker count equals the count of recognized punctuation marks. Done-check: the period, comma, question, exclamation, and empty criteria.

### T-01.09  Window cross-sentence context
id: T-01.09
phase: 1
status: done
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::context::window(&sentences, 2, radius=1)` for `["a","b","c","d"]` returns a `ContextWindow` whose `current` is "c", `before` is ["b"], and `after` is ["d"], pinning a radius-1 window.
  - C2: at the first index, `window(&s, 0, 1)` yields empty `before` and `after == ["b"]`; at the last index `window(&s, 3, 1)` yields `after` empty and `before == ["c"]`, pinning both clamp boundaries.
  - C3: with `radius=2` on the same input, `window(&s, 1, 2)` returns `before == ["a"]` (clamped, not 2) and `after == ["c","d"]` (exactly 2), so the window never exceeds the bounded radius and never reads out of range.
  - C4: `window(&s, 1, 0)` returns empty `before` and `after` with `current == "b"`, pinning the zero-radius boundary.
not_doing:
  - No tokenization or sentence splitting (input is a pre-split slice).
  - No semantic relevance weighting; positional window only.
test_files: [tests/context_window.rs]
criteria_map:
  C1: [test_interior_radius1_current_is_c, test_interior_radius1_before_is_b, test_interior_radius1_after_is_d, test_interior_radius1_lengths_within_radius]
  C2: [test_first_index_before_empty, test_first_index_after_is_b, test_last_index_after_empty, test_last_index_before_is_c]
  C3: [test_over_radius_before_clamped_to_one, test_over_radius_after_is_two, test_over_radius_current_is_b]
  C4: [test_zero_radius_current_only]
attempts: 1
last_failure: ""
---
Assembly of a bounded conditioning window around a target sentence. Inputs: a slice of sentence strings, a current index, and a radius. Outputs: a typed `ContextWindow { before, current, after }` with `before`/`after` length ≤ radius. Errors/edges: window is clamped at both ends so it never indexes out of range; radius 0 yields only the current sentence. Invariant: `before.len() ≤ radius` and `after.len() ≤ radius` always hold. Done-check: the interior, both-end-clamp, over-radius-clamp, and zero-radius criteria.

### T-01.10  Compute paragraph pacing
id: T-01.10
phase: 1
status: pending
depends_on: [T-01.01]
stack: rust
criteria:
  - C1: `syrinx_frontend::pacing::breath_markers(text, interval_words=10)` on a 25-word single paragraph inserts breath markers after word 10 and word 20 (2 markers), and the marker positions are exactly those word indices.
  - C2: a paragraph of exactly 10 words yields zero breath markers (interval reached but not exceeded), while 11 words yields one marker — pinning the `> interval` boundary not `>=`.
  - C3: `breath_markers` is deterministic — calling it twice on identical input returns identical marker positions.
  - C4: a paragraph boundary always forces a breath marker, so two paragraphs of 3 words each (below the interval) still yield exactly one marker, at the paragraph break.
not_doing:
  - No prosodic duration assignment to breaths (markers are positional only).
  - No language-specific breathing models; uniform word-interval policy.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Deterministic insertion of breath markers by word interval and paragraph structure. Inputs: paragraph text and a words-per-breath interval. Outputs: an ordered list of breath-marker positions. Errors/edges: markers fall strictly after each completed interval (boundary pinned at `>` not `>=`); every paragraph break forces a marker regardless of length. Invariant: identical input yields identical marker positions on every call. Done-check: the interior-interval, off-by-one boundary, determinism, and paragraph-break criteria.

### T-01.11  Assemble the frontend test suite
id: T-01.11
phase: 1
status: pending
depends_on: [T-01.01, T-01.02, T-01.07]
stack: rust
criteria:
  - C1: `cargo test -p syrinx-frontend` runs the golden-file suite covering normalize, number-expansion, and SSML and exits 0 with all golden cases passing.
  - C2: the suite is driven by golden files under the repo-root `tests/golden/`; mutating any single golden INPUT file changes the produced output so its paired case fails, proving the goldens actually gate behaviour.
  - C3: the suite enumerates every golden fixture directory automatically (a newly added (input,expected) pair is picked up without editing the test harness), and an input with no matching expected file fails the run rather than silently skipping.
not_doing:
  - No CI/workflow YAML wiring (that is a Phase 0 concern).
  - No coverage of crates other than syrinx-frontend.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The aggregating golden-file harness for the deterministic frontend. Inputs: the golden fixture tree of (input,expected) pairs. Outputs: a pass/fail test run over every fixture. Errors/edges: a changed input perturbs output and fails its case; a missing expected file fails rather than skips. Invariant: every fixture directory is auto-discovered, so adding a pair needs no harness edit. Done-check: the green run, the changed-input-fails property, and the auto-enumeration criterion.

### T-01.12  Version the frontend-LM contract
id: T-01.12
phase: 1
status: pending
depends_on: [T-01.04, T-01.07]
stack: rust
criteria:
  - C1: the struct `syrinx_frontend::contract::FrontendOutput` carries an explicit `schema_version: u32` field set to the current version constant `SCHEMA_VERSION`, and a constructed value exposes that exact integer.
  - C2: `FrontendOutput` holds typed token/phoneme entries and control events, and `serde_json::to_string` then `from_str` round-trips a populated value to an equal struct (`PartialEq` holds before and after).
  - C3: deserializing a JSON payload whose `schema_version` differs from `SCHEMA_VERSION` (e.g. an older integer) returns a typed `ContractError::VersionMismatch`, not a silent accept and not a panic.
  - C4: a payload missing the `schema_version` field fails deserialization with a typed error rather than defaulting the version.
not_doing:
  - No backward-compatibility migration between schema versions; mismatch is rejected.
  - No wire format other than JSON for this contract.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The typed, versioned hand-off struct from the frontend to `syrinx-lm`. Inputs: phoneme/token entries plus control events produced upstream. Outputs: a serializable `FrontendOutput` with an explicit `schema_version`. Errors/edges: serialize→deserialize round-trips to an equal value; a version mismatch or absent version field yields a typed `ContractError`, never a silent accept or panic. Invariant: the schema version is an explicit, checked field on every payload. Done-check: the version-field, round-trip, mismatch-rejection, and missing-field criteria.

### T-02.01  Port base weights to Rust tensors
id: T-02.01
phase: 2
status: blocked
depends_on: [T-00.07]
stack: rust
criteria:
  - C1: `syrinx-core`'s weight loader reads every tensor named in the base checkpoint and the loaded `HashMap<String, Tensor>` keys are exactly the reference parameter names, with none missing and none extra.
  - C2: for every loaded tensor the Rust shape equals the Python reference shape element-for-element, and a single transposed or off-by-one axis fails the check.
  - C3: every loaded tensor's dtype matches the reference dtype (fp16 stays fp16, fp32 stays fp32) and a silent fp32→fp16 cast is rejected.
  - C4: max-abs elementwise difference between each loaded Rust tensor and the reference array is ≤ 1e-6 on the fixed checkpoint, and 1e-5 corruption in any element is caught.
not_doing:
  - No forward-pass execution; loading and shape/dtype/value verification only.
  - No quantization or device placement beyond host-memory load.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the chosen base checkpoint plus its Python-reference tensor dump. Bounds: keys, shapes, dtypes, and values pinned at both equality and just-past corruption. Outputs: an in-memory `syrinx-core` tensor map. Errors/edges: missing key, extra key, transposed axis, dtype cast, 1e-5 value drift all fail. Invariant: the Rust map is bit-faithful to the reference within 1e-6. Done-check: the four parity criteria against the reference dump. BLOCKED: needs the real base model weights and a Python reference tensor dump to compare against, neither of which exists until a human ports the chosen base.

### T-02.02  Run the semantic LM forward pass
id: T-02.02
phase: 2
status: blocked
depends_on: [T-02.01]
stack: rust
criteria:
  - C1: `syrinx-lm`'s forward pass on the fixed tokenized input produces output logits whose tensor shape equals the Python reference logits shape exactly.
  - C2: the Rust LM logits match the Python reference logits within 1e-3 max-abs on the fixed input, and a 2e-3 perturbation in any logit fails the check.
  - C3: greedy argmax token IDs over the reference prompt are identical between the Rust and Python LM for every position in the fixed sequence.
  - C4: the paralinguistic-token logits occupy the reference vocabulary index range and a one-slot vocabulary offset is caught.
not_doing:
  - No sampling, beam search, or KV-cache optimization beyond a single deterministic forward.
  - No acoustic or vocoder stages; LM logits only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the loaded base weights and a fixed tokenized prompt with paralinguistic tokens. Bounds: logits pinned at 1e-3 and rejected at 2e-3; argmax pinned per position. Outputs: per-position logits over the LM vocabulary. Errors/edges: shape mismatch, >1e-3 drift, argmax divergence, vocab offset all fail. Invariant: Rust logits equal the Python reference within tolerance. Done-check: the four parity criteria. BLOCKED: needs the ported weights, a GPU, and a Python reference forward pass to produce comparison logits — none gateable without the real model.

### T-02.03  Run the speaker encoder forward
id: T-02.03
phase: 2
status: blocked
depends_on: [T-02.01]
stack: rust
criteria:
  - C1: `syrinx-speaker`'s encoder on the fixed reference clip emits an embedding whose dimensionality equals the reference embedding length exactly.
  - C2: the Rust speaker embedding matches the Python reference embedding within 1e-3 max-abs, and a 2e-3 perturbation fails the check.
  - C3: cosine similarity between the Rust embedding and the reference embedding is ≥ 0.9999 on the fixed clip, and a 0.999 result fails.
  - C4: two distinct reference clips yield embeddings whose pairwise cosine ordering matches the reference ordering, and a swapped pair is caught.
not_doing:
  - No enrollment store, blending, or morphing — single forward only.
  - No attribute conditioning or disentanglement.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the loaded encoder weights and fixed reference audio clips. Bounds: embedding pinned at 1e-3/0.9999 and rejected just past. Outputs: a fixed-dimension speaker embedding. Errors/edges: dim mismatch, >1e-3 drift, low cosine, swapped ordering all fail. Invariant: Rust embedding equals the reference within tolerance. Done-check: the four parity criteria. BLOCKED: needs the ported encoder weights, a GPU, and a Python reference embedding for the fixed clip — perceptual/numerical parity unverifiable without the real model.

### T-02.04  Run the flow-matching acoustic decoder
id: T-02.04
phase: 2
status: blocked
depends_on: [T-02.01]
stack: rust
criteria:
  - C1: `syrinx-acoustic`'s DiT + ODE solver on the fixed seed and step count emits a mel tensor whose shape equals the Python reference mel shape exactly.
  - C2: the Rust mel matches the Python reference mel within 1e-2 max-abs at the fixed seed and solver step count, and a 2e-2 perturbation fails the check.
  - C3: changing the ODE solver step count from the fixed value to one fewer step changes the mel beyond 1e-2, confirming the solver is genuinely integrating.
  - C4: the chunk-aware causal path produces, for the first chunk, mel identical to the corresponding prefix of the whole-utterance mel within 1e-3.
not_doing:
  - No vocoder waveform synthesis — mel output only.
  - No streaming buffer management beyond first-chunk causal equivalence.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: LM/plan conditioning, speaker embedding, fixed seed, fixed step count. Bounds: mel pinned at 1e-2, rejected at 2e-2; chunk prefix pinned at 1e-3. Outputs: a reference-shaped mel spectrogram. Errors/edges: shape mismatch, >1e-2 drift, step-count insensitivity, chunk/whole divergence all fail. Invariant: Rust mel equals the reference within tolerance at fixed seed/steps. Done-check: the four parity criteria. BLOCKED: needs the ported decoder weights, a GPU, and a seed-pinned Python reference mel — flow-matching parity is not gateable without the real model.

### T-02.05  Run the vocoder waveform synthesis
id: T-02.05
phase: 2
status: blocked
depends_on: [T-02.01]
stack: rust
criteria:
  - C1: `syrinx-vocoder` on the fixed reference mel emits a 48kHz waveform whose sample count equals the reference waveform length exactly.
  - C2: the Rust waveform matches the Python reference waveform within 1e-3 max-abs per sample on the fixed mel, and a 2e-3 perturbation fails the check.
  - C3: the reconstructed waveform's log-mel re-analysis matches the input mel within the reference spectral tolerance, and a half-band frequency shift is caught.
  - C4: the 8kHz telephony path resamples the same mel to a waveform whose sample count equals the reference 8kHz length and whose band-limit cutoff matches the reference within tolerance.
not_doing:
  - No perceptual MOS scoring — numerical waveform parity only.
  - No streaming packetization or playback.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the fixed reference mel and the loaded vocoder weights. Bounds: waveform pinned at 1e-3, rejected at 2e-3; 8kHz length and cutoff pinned. Outputs: 48kHz and 8kHz waveforms. Errors/edges: length mismatch, >1e-3 drift, spectral shift, wrong band-limit all fail. Invariant: Rust waveform equals the reference within tolerance. Done-check: the four parity criteria. BLOCKED: needs the ported vocoder weights, a GPU, and a Python reference waveform — "no audible artifacts" is a perceptual judgment requiring the real model and ears.

### T-02.06  Wire the end-to-end inference pipeline
id: T-02.06
phase: 2
status: blocked
depends_on: [T-01.12, T-02.02, T-02.03, T-02.04, T-02.05]
stack: rust
criteria:
  - C1: the pipeline accepts a frontend token/control stream plus a reference clip and returns a 48kHz waveform with no Python process invoked anywhere in the call path.
  - C2: the end-to-end Rust waveform for the fixed text+reference matches the Python reference end-to-end waveform within 1e-2 max-abs per sample, and a 2e-2 perturbation fails.
  - C3: the same input run twice at the fixed seed yields byte-identical output, confirming determinism, and any nondeterministic stage is caught.
  - C4: the stage hand-off types (frontend→LM→plan→decoder→vocoder) connect through the versioned interfaces with no intermediate stage skipped, verified by a stage-trace assertion.
not_doing:
  - No quantization, watermarking, or streaming — full-precision batch path only.
  - No latency or footprint measurement.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: a fixed text + reference clip through the frontend contract. Bounds: end-to-end waveform pinned at 1e-2, rejected at 2e-2; determinism byte-exact. Outputs: a single 48kHz utterance with no Python in path. Errors/edges: Python invocation, >1e-2 drift, nondeterminism, skipped stage all fail. Invariant: pure-Rust deterministic end-to-end synthesis. Done-check: the four criteria. BLOCKED: needs every ported stage (T-02.02–T-02.05) plus a GPU and a Python reference end-to-end run — not gateable until the full model exists in Rust.

### T-02.07  Build the numerical parity harness
id: T-02.07
phase: 2
status: blocked
depends_on: [T-02.06]
stack: rust
criteria:
  - C1: `syrinx-eval`'s parity harness reports a per-stage max-abs difference (LM logits, speaker embedding, mel, waveform) against the Python reference for every stage in the pipeline.
  - C2: the harness passes only when every stage is within its declared tolerance (1e-3 LM, 1e-3 speaker, 1e-2 mel, 1e-3 waveform) and fails if any single stage exceeds its tolerance by injecting a known over-tolerance perturbation.
  - C3: the harness emits a machine-readable JSON report with one entry per stage carrying stage name, measured max-abs diff, tolerance, and pass/fail, and a missing stage entry is caught.
  - C4: a deliberately corrupted reference fixture forces the corresponding stage to FAIL, proving the harness is not vacuously green.
not_doing:
  - No tuning of model weights to reach tolerance — measurement and gating only.
  - No perceptual metrics; numerical per-stage parity only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the Rust pipeline stages and the Python per-stage reference dumps. Bounds: each stage pinned at its tolerance and rejected just past. Outputs: a per-stage JSON parity report with pass/fail. Errors/edges: over-tolerance stage, missing stage entry, corrupted fixture all fail. Invariant: green iff every stage is within tolerance. Done-check: the four criteria including a planted over-tolerance failure. BLOCKED: needs the running Rust pipeline (T-02.06) and per-stage Python reference dumps on a GPU — the harness cannot measure parity without the real model on both sides.

### T-02.08  Build the 4-bit quantization path
id: T-02.08
phase: 2
status: blocked
depends_on: [T-02.06]
stack: rust
criteria:
  - C1: `syrinx-core`'s ISQ-style 4-bit quantizer produces packed weights whose resident size is ≤ 30% of the fp16 weight size, and a 4.5-bit packing that exceeds the budget fails.
  - C2: dequantize(quantize(W)) reconstructs each weight within the declared 4-bit reconstruction error bound, and an error one ULP past the bound is caught.
  - C3: SIM-o and WER on the frozen eval set at 4-bit degrade by no more than the declared budget versus fp16, and a degradation just past budget fails.
  - C4: the fp16 fallback is selectable at load time and, when selected, the served weights are bit-identical to the unquantized fp16 weights.
not_doing:
  - No sub-4-bit or mixed-precision schemes beyond 4-bit and the fp16 fallback.
  - No footprint measurement of the full running process (that is T-02.12).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the fp16 base weights and the frozen eval set. Bounds: size ≤30%, reconstruction at the bound, SIM-o/WER at budget, all rejected just past. Outputs: packed 4-bit weights plus a selectable fp16 fallback. Errors/edges: over-budget packing, over-bound reconstruction, over-budget quality loss, non-identical fallback all fail. Invariant: 4-bit stays within quality and size budget; fallback is exact. Done-check: the four criteria. BLOCKED: needs the real weights, a GPU, and SIM-o/WER eval over the frozen set to measure quality degradation — perceptual/eval metrics not gateable without the model.

### T-02.09  Validate zero-shot cloning quality
id: T-02.09
phase: 2
status: blocked
depends_on: [T-02.06]
stack: rust
criteria:
  - C1: `syrinx-eval`'s cloning run synthesizes from each frozen reference clip and computes SIM-o between the synthesized output and the held-out target speaker for every clip in the frozen set.
  - C2: mean SIM-o over the frozen set is ≥ the declared baseline threshold, and a result one point below the threshold fails.
  - C3: per-clip SIM-o is reported individually so a single regressing speaker is caught, not masked by the mean.
  - C4: the SIM-o computation reproduces the reference scorer's value within 1e-3 on a fixed (synth, target) pair, proving the metric itself is correct.
not_doing:
  - No cross-lingual or accent evaluation (that is T-02.10).
  - No model retraining to lift SIM-o — measurement and gating only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the frozen reference clips, held-out targets, and the running pipeline. Bounds: mean SIM-o at the baseline, per-clip surfaced, scorer pinned at 1e-3. Outputs: per-clip and mean SIM-o with pass/fail. Errors/edges: below-threshold mean, regressing clip, wrong scorer value all fail. Invariant: green iff cloning meets the SIM-o baseline. Done-check: the four criteria. BLOCKED: needs the running model on a GPU and SIM-o perceptual-similarity scoring against held-out targets — a perceptual eval metric not expressible as a frozen-test gate.

### T-02.10  Validate cross-lingual and accent transfer
id: T-02.10
phase: 2
status: blocked
depends_on: [T-02.09]
stack: rust
criteria:
  - C1: `syrinx-eval` synthesizes each frozen cross-lingual prompt with a source-language reference and computes ASR-based WER against the target-language transcript for every prompt.
  - C2: mean WER over the cross-lingual set is ≤ the declared target, and a result one point above the target fails.
  - C3: accent retention is scored against the reference accent classifier and the mean accent-match is ≥ the declared threshold, with a just-below result failing.
  - C4: per-language WER and per-accent retention are reported individually so a single failing language or accent is caught, not averaged away.
not_doing:
  - No new-language model training — evaluation of transfer only.
  - No SIM-o re-validation (covered by T-02.09).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the frozen cross-lingual prompts, source references, target transcripts, and accent classifier. Bounds: WER at target, accent-match at threshold, both rejected just past; per-item surfaced. Outputs: per-language WER and per-accent retention with pass/fail. Errors/edges: over-target WER, low accent retention, masked per-item regression all fail. Invariant: green iff cross-lingual transfer meets WER and accent targets. Done-check: the four criteria. BLOCKED: needs the running model on a GPU, an ASR system for WER, and an accent classifier over the frozen set — perceptual/eval metrics not gateable without the model and corpus.

### T-02.11  Embed the output watermark
id: T-02.11
phase: 2
status: blocked
depends_on: [T-00.09, T-02.06]
stack: rust
criteria:
  - C1: `syrinx-serve` embeds a PerTh-style watermark on every synthesized output and the detector recovers the watermark from the unmodified waveform with detection confidence ≥ the declared threshold.
  - C2: the watermark survives MP3 transcode and a bounded waveform edit, with post-distortion detection rate ≥ near-100% over the frozen set and a just-below rate failing.
  - C3: the detector's false-positive rate on non-watermarked reference audio is ≤ the declared bound, and a rate just past the bound fails.
  - C4: watermarking changes the output perceptually within the declared SNR budget (watermark is inaudible), and an embed that exceeds the SNR budget is caught.
not_doing:
  - No watermark key management or rotation policy (lives in the ethics/policy doc).
  - No detection of third-party watermarks — own-watermark embed and detect only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: synthesized waveforms and the watermark key from the policy doc. Bounds: detection at threshold, post-distortion near-100%, FPR at bound, SNR at budget, all rejected just past. Outputs: watermarked audio plus a detection verdict. Errors/edges: low detection, distortion failure, high FPR, audible embed all fail. Invariant: every output carries a robust, inaudible, detectable watermark. Done-check: the four criteria. BLOCKED: needs the running pipeline on a GPU plus real synthesized audio and a perceptual SNR/detection eval through MP3 and edit distortion — watermark robustness is not gateable without the model output.

### T-02.12  Check the 4-bit memory footprint
id: T-02.12
phase: 2
status: blocked
depends_on: [T-02.08]
stack: rust
criteria:
  - C1: the resident memory of the loaded 4-bit model measured by `syrinx-core` is ≤ ~300MB, and a load that reaches 320MB fails the check.
  - C2: peak VRAM during a single inference on one RTX 4090-class GPU stays within the declared budget, and a run that exceeds the budget fails.
  - C3: the footprint report is emitted as JSON carrying resident bytes, peak VRAM, and the budget, and a missing field is caught.
  - C4: switching to the fp16 fallback raises the reported footprint above the 4-bit figure by the expected ratio, proving the measurement tracks the actual loaded precision.
not_doing:
  - No concurrency or stress testing (that is Phase 7).
  - No latency measurement; footprint only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the loaded 4-bit (and fp16-fallback) model on one 4090-class GPU. Bounds: resident ≤300MB rejected at 320MB; VRAM at budget; ratio pinned. Outputs: a JSON footprint report. Errors/edges: over-budget resident, over-budget VRAM, missing field, wrong precision ratio all fail. Invariant: the 4-bit model fits the ~300MB footprint on one 4090. Done-check: the four criteria. BLOCKED: needs the quantized model loaded on a real RTX 4090-class GPU to measure resident memory and peak VRAM — a hardware footprint measurement not expressible without the weights and GPU.

### T-03.01  Define the prosody plan model
id: T-03.01
phase: 3
status: pending
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: in `syrinx-prosody`, `serde_json::to_vec` then `from_slice` on a `ProsodyPlan` round-trips byte-identically (re-serialized bytes equal the original bytes) for a plan with at least one phoneme.
  - C2: a `ProsodyPlan` constructed for N phonemes has `durations_ms.len() == N` and `pitch_hz.len() == N` for N == 0 and for N == 3; a constructor given mismatched array lengths returns `Err(PlanError::LengthMismatch)`, not a panic.
  - C3: `ProsodyPlan::phoneme(i)` returns `Ok` for i == N-1 and `Err(PlanError::IndexOutOfRange)` for i == N (one past the last), and never panics on any usize index.
  - C4: a deserialized `ProsodyPlan` exposes `schema_version` equal to the crate's `PLAN_SCHEMA_VERSION` constant, and JSON missing the `schema_version` field fails to deserialize with an error.
not_doing:
  - No prosody prediction, defaults, or model inference — values are caller-supplied.
  - No audio rendering or DSP — this is the data model only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The typed editable plan every control task edits. Inputs: a phoneme count N plus equal-length `durations_ms: Vec<f32>` and `pitch_hz: Vec<f32>` arrays and a `schema_version`. Outputs: a `ProsodyPlan` that serializes to JSON and round-trips byte-identically. Errors/edges: mismatched array lengths → `PlanError::LengthMismatch`; index access at i == N → `PlanError::IndexOutOfRange` (boundary at i == N-1 still `Ok`); JSON without `schema_version` → deserialize error; N == 0 is a valid empty plan. Invariant: `durations_ms.len() == pitch_hz.len() == N` always holds, and no index access ever panics. Done-check: the four frozen criteria over serialize/round-trip, length agreement, index boundary, and schema presence.

### T-03.02  Expose the duration predictor override
id: T-03.02
phase: 3
status: blocked
depends_on: [T-03.01]
stack: rust
criteria:
  - C1: with the trained duration predictor loaded, predicting durations for a fixed phoneme sequence at a pinned seed yields per-phoneme `durations_ms` matching the reference predictor within tolerance, and the count equals the phoneme count.
  - C2: overriding phoneme i's duration to a value V and re-rendering produces a segment whose measured duration equals V within tolerance, while phonemes other than i are unchanged within tolerance.
  - C3: a duration override at index i == N returns `Err(PlanError::IndexOutOfRange)` and renders nothing, while i == N-1 applies.
not_doing:
  - No pitch or volume control — duration timing only.
  - No re-training of the predictor — exposure and override only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Surfaces predicted per-phoneme durations on the plan and lets a caller override any entry. Inputs: a phoneme sequence and the trained duration predictor; optional per-index duration overrides. Outputs: a `ProsodyPlan` whose `durations_ms` reflect predictions plus overrides, and rendered timing honoring them. Errors/edges: override index at N → `IndexOutOfRange` (N-1 applies); non-overridden phonemes must stay fixed. Invariant: only the overridden indices change the rendered timing. Done-check: prediction parity, override-changes-timing-predictably, and the index boundary, measured on rendered audio. BLOCKED: requires the trained duration predictor and the acoustic renderer (human-and-GPU work, DESIGN §12 / CLAUDE THE BUILD SCOPE) before predicted durations or rendered-timing tolerances exist to gate against.

### T-03.03  Expose the pitch predictor override
id: T-03.03
phase: 3
status: blocked
depends_on: [T-03.01]
stack: rust
criteria:
  - C1: with the trained F0 predictor loaded, predicting the pitch contour for a fixed phoneme sequence at a pinned seed yields `pitch_hz` matching the reference within tolerance, one entry per phoneme.
  - C2: a per-phoneme pitch override at index i changes the measured F0 of segment i toward the target within tolerance while other phonemes' F0 stay unchanged within tolerance.
  - C3: a per-word pitch edit spanning the word's phoneme span shifts the measured F0 across exactly that span and leaves phonemes outside the span unchanged within tolerance.
not_doing:
  - No duration or volume control — pitch/F0 only.
  - No intonation presets — those build on this in T-03.07.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Surfaces the predicted F0 contour and lets a caller override pitch at word and phoneme granularity. Inputs: a phoneme sequence, the trained F0 predictor, and per-word or per-phoneme pitch edits. Outputs: a `ProsodyPlan` whose `pitch_hz` reflects predictions plus edits, audibly applied on render. Errors/edges: edits outside a word's phoneme span must not bleed; out-of-range indices error per the plan model. Invariant: only edited phonemes/spans change measured F0. Done-check: contour parity plus per-phoneme and per-word edits verified on rendered audio. BLOCKED: requires the trained F0 predictor and the acoustic renderer to measure F0 against; until a human ports the model these tolerances and contours do not exist (CLAUDE THE BUILD SCOPE).

### T-03.04  Stretch the speech rate
id: T-03.04
phase: 3
status: blocked
depends_on: [T-03.01]
stack: rust
criteria:
  - C1: applying a rate factor R to rendered model output yields total duration equal to the original divided by R within tolerance (R == 2.0 halves, R == 0.5 doubles).
  - C2: the measured fundamental frequency of the rate-stretched audio equals the original F0 within tolerance — rate scaling does not shift pitch.
  - C3: a rate factor R == 1.0 returns audio equal to the input within tolerance, and a non-positive R returns a typed error.
not_doing:
  - No pitch shifting — rate only, pitch preserved.
  - No per-phoneme rate; this is an utterance-level stretch.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Time-stretches rendered speech without pitch change. Inputs: a rendered sample buffer (model output) and a positive rate factor R. Outputs: a buffer whose duration scales by 1/R with F0 preserved. Errors/edges: R == 1.0 is identity; R <= 0 is a typed error; boundary behavior pinned at R and just past it. Invariant: pitch is invariant under rate change. Done-check: duration-scales, pitch-unchanged, and identity/error edges, measured on stretched audio. BLOCKED: the audio DSP operates on the acoustic decoder's model output, which does not exist until a human ports/trains the renderer (CLAUDE THE BUILD SCOPE) — there is nothing to stretch or to measure F0 on yet.

### T-03.05  Apply volume automation curves
id: T-03.05
phase: 3
status: pending
depends_on: [T-03.01]
stack: rust
criteria:
  - C1: in `syrinx-prosody`, applying an envelope of all 1.0 to an f32 buffer returns each sample bit-identical to the input.
  - C2: applying an envelope of all 0.5 returns each output sample equal to exactly 0.5 times the corresponding input (e.g. input 1.0 → 0.5, input -0.4 → -0.2), asserted at the threshold and for a non-0.5 gain to kill scale mutants.
  - C3: across a segment boundary from gain A to gain B the applied gain interpolates per spec — the first sample uses A, the last uses B, and the midpoint sample uses (A+B)/2 within tolerance.
  - C4: an envelope whose length differs from the buffer (one longer and one shorter) returns `Err(EnvelopeError::LengthMismatch)`, while an exactly-equal length applies.
not_doing:
  - No pitch or duration changes — amplitude/gain only.
  - No model inference — pure deterministic DSP on a given buffer.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
A deterministic per-segment gain envelope over an f32 sample buffer. Inputs: an f32 sample buffer and a segment gain envelope. Outputs: a new buffer with per-sample gain applied. Errors/edges: flat 1.0 is identity; 0.5 halves exactly; segment boundaries interpolate (endpoints A and B, midpoint (A+B)/2); envelope length ≠ buffer length (both longer and shorter) → `EnvelopeError::LengthMismatch`, equal length applies. Invariant: output length equals input length and gain is applied sample-exact. Done-check: the four frozen criteria pinning identity, exact halving, boundary interpolation, and the length boundary.

### T-03.06  Steer the emotion
id: T-03.06
phase: 3
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: for a fixed text and a target emotion prompt, a blind A/B perceptual check identifies the intended emotion at a rate above the agreed threshold against the neutral baseline.
  - C2: increasing the intensity scale across its range produces monotonically increasing rated intensity of the target emotion on the perceptual panel within the agreed margin.
  - C3: intensity 0 renders audio perceptually equivalent to the neutral baseline within the agreed margin.
not_doing:
  - No sarcasm/irony composition — that is T-03.08.
  - No new emotion taxonomy beyond the agreed prompt set.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Text-prompted emotion with a monotonic intensity scale. Inputs: text, an emotion prompt, and an intensity in [0,1]. Outputs: rendered audio carrying the intended emotion scaled by intensity. Errors/edges: intensity 0 collapses to neutral; intensity is monotonic. Invariant: intended emotion identifiable and intensity ordering preserved. Done-check: A/B emotion identification and monotonic-intensity panel results. BLOCKED: the gate is a perceptual A/B + intensity-monotonicity judgment requiring the trained model and human listeners (CLAUDE THE BUILD SCOPE) — it cannot be expressed as a frozen-test + mutation gate.

### T-03.07  Manipulate the intonation contour
id: T-03.07
phase: 3
status: blocked
depends_on: [T-03.03]
stack: rust
criteria:
  - C1: applying a named intonation preset (e.g. rising question contour) to a fixed utterance produces the preset's specified F0 shape, with terminal F0 rising versus the neutral baseline within tolerance.
  - C2: a manually supplied F0 curve is honored point-for-point — measured F0 tracks the supplied curve within tolerance across the utterance.
  - C3: the falling-contour preset produces a terminal F0 below the neutral baseline within tolerance, distinguishing it from the rising preset to pin contour direction on both sides.
not_doing:
  - No emotion semantics — contour shape only.
  - No per-phoneme pitch API — that is T-03.03.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Contour presets plus manual F0 curves over an utterance. Inputs: an utterance plan and either a named preset or a manual F0 curve. Outputs: rendered audio whose measured F0 follows the chosen contour. Errors/edges: rising vs falling presets pinned in opposite directions; manual curve tracked point-for-point. Invariant: the applied contour governs measured F0. Done-check: preset-direction and manual-curve tracking on rendered audio. BLOCKED: depends on the F0 predictor exposure (T-03.03) and the renderer to measure contours against, both human-and-GPU prerequisites (CLAUDE THE BUILD SCOPE); no measurable F0 exists until then.

### T-03.08  Control the sarcasm inflection
id: T-03.08
phase: 3
status: blocked
depends_on: [T-03.06, T-03.07]
stack: rust
criteria:
  - C1: toggling the sarcasm/irony inflection on for a fixed utterance produces the expected contour shift (the agreed drawl/flattened-terminal signature) versus the toggle-off baseline, measurable in the eval harness above threshold.
  - C2: with the inflection off, the rendered contour equals the non-sarcastic baseline within tolerance — the toggle has no effect when disabled.
  - C3: a blind listening panel rates the inflected rendering as sarcastic above the agreed rate versus the sincere baseline.
not_doing:
  - No new emotion axes — composes existing emotion + intonation.
  - No automatic sarcasm detection from text.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Composes emotion steering and intonation into a sarcasm/irony toggle. Inputs: an utterance plus a sarcasm toggle/level. Outputs: rendered audio with the irony contour signature when on. Errors/edges: off equals the sincere baseline; on shifts the contour as specified. Invariant: the toggle's effect is present only when enabled. Done-check: eval-measured contour shift plus a blind perceptual rating. BLOCKED: builds on emotion steering (T-03.06) and intonation (T-03.07) and is judged perceptually with the trained model and human listeners (CLAUDE THE BUILD SCOPE) — not frozen-test gateable.

### T-03.09  Edit the phoneme-level plan
id: T-03.09
phase: 3
status: blocked
depends_on: [T-03.02, T-03.03]
stack: rust
criteria:
  - C1: editing phoneme i's duration in the plan and rendering produces segment i with the edited duration within tolerance, with neighboring phonemes' durations unchanged within tolerance.
  - C2: editing phoneme i's pitch in the plan and rendering produces segment i with the edited F0 within tolerance, with neighboring phonemes' F0 unchanged within tolerance.
  - C3: the renderer honors a simultaneous duration and pitch edit on the same phoneme, with both edits reflected within tolerance in the rendered segment.
not_doing:
  - No volume editing in this API — duration and pitch only.
  - No batch/scripted edit language — single-phoneme edits.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
The plan editor: edit any phoneme's duration and pitch and have the renderer honor it. Inputs: a `ProsodyPlan` and per-phoneme duration/pitch edits. Outputs: rendered audio in which the edited phoneme reflects the edit and neighbors do not change. Errors/edges: edits are local to the targeted phoneme; combined dur+pitch edits both apply. Invariant: only the edited phoneme's rendered segment changes. Done-check: rendered duration, pitch, and combined edits verified per phoneme. BLOCKED: requires the duration (T-03.02) and pitch (T-03.03) predictor exposure plus the acoustic renderer to honor edits, all human-and-GPU prerequisites (CLAUDE THE BUILD SCOPE).

### T-03.10  Round-trip the edited plan
id: T-03.10
phase: 3
status: blocked
depends_on: [T-03.09]
stack: rust
criteria:
  - C1: serializing an edited plan, deserializing it, and rendering at a pinned seed yields audio bit-equivalent (within the deterministic tolerance) to rendering the in-memory edited plan.
  - C2: the rendered audio reflects the applied edit — the edited phoneme's measured duration/pitch matches the edit, distinguishing it from the unedited-plan render.
  - C3: re-rendering the same serialized edited plan twice at the same seed produces identical audio, confirming determinism.
not_doing:
  - No new edit operations — exercises T-03.09's edits end to end.
  - No perceptual quality scoring — determinism and edit-fidelity only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
End-to-end determinism: an edited plan survives serialize→deserialize→render and the audio matches the edit. Inputs: an edited `ProsodyPlan`. Outputs: rendered audio matching both the edit and a repeat render. Errors/edges: round-tripped render must equal the in-memory render; repeat renders must be identical. Invariant: rendering is a deterministic function of the (serialized) edited plan and seed. Done-check: round-trip equivalence, edit-fidelity, and render determinism. BLOCKED: needs the renderer and the phoneme-edit API (T-03.09) to produce audio to compare; both require the ported model and GPU (CLAUDE THE BUILD SCOPE).

### T-03.11  Evaluate the prosody prediction quality
id: T-03.11
phase: 3
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: running the default (un-edited) prosody predictions over the frozen eval set yields a MOS-proxy score at or above the agreed target threshold, and below threshold for a deliberately degraded plan to pin the gate.
  - C2: the eval is reproducible — the same model and eval set at a pinned seed produce the same MOS-proxy score across runs.
  - C3: the report attributes scores per utterance so regressions localize to specific items.
not_doing:
  - No model re-training — measures the existing predictor.
  - No control-edit eval — default prediction quality only.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Automated quality gate on default prosody predictions. Inputs: the trained predictor and the frozen eval set. Outputs: a MOS-proxy score and per-utterance report. Errors/edges: degraded plans must score below target; runs must be reproducible at a pinned seed. Invariant: the score is a deterministic function of model + eval set + seed. Done-check: threshold pass on defaults, sub-threshold on degraded, and run-to-run reproducibility. BLOCKED: the MOS-proxy is a perceptual-quality metric requiring the trained model and the frozen perceptual eval set (CLAUDE THE BUILD SCOPE) — not expressible as a frozen-test + mutation gate.

### T-04.01  Audit the speaker-embedding space
id: T-04.01
phase: 4
status: blocked
depends_on: [T-02.03]
stack: rust
criteria:
  - C1: a structural report over the speaker-encoder embedding store documents pairwise distance distribution and cluster structure, with same-speaker distances below cross-speaker distances above the agreed margin.
  - C2: an interpolability check confirms that midpoints between two enrolled embeddings remain within the populated region (no collapse/NaN), documented with the measured caveats.
  - C3: the audit is reproducible — the same embedding set yields the same distance and clustering statistics at a pinned seed.
not_doing:
  - No blending or morphing yet — structure analysis only.
  - No attribute disentanglement — that is Phase 6.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Characterizes the embedding space so blend/morph rest on real structure. Inputs: the speaker encoder and a set of enrolled embeddings. Outputs: a report on distances, clustering, and interpolability with caveats. Errors/edges: degenerate/collapsed regions must be flagged; stats reproducible at a pinned seed. Invariant: same-speaker tighter than cross-speaker by the agreed margin. Done-check: distance/cluster report, interpolability finding, and reproducibility. BLOCKED: requires the trained speaker encoder (T-02.03) to produce embeddings; that encoder needs ported weights + GPU (CLAUDE THE BUILD SCOPE), so there is no space to audit until a human builds it.

### T-04.02  Blend multiple speaker profiles
id: T-04.02
phase: 4
status: blocked
depends_on: [T-04.01]
stack: rust
criteria:
  - C1: a weighted interpolation of two or more enrolled embeddings produces a blended embedding whose weights sum to 1, and rendering it yields coherent speech rated above the agreed threshold in eval.
  - C2: weight 1.0 on a single speaker reproduces that speaker's rendering within tolerance (boundary of the blend), and an even 0.5/0.5 blend sits perceptually between the two.
  - C3: the blended voice is stable — re-rendering the same blend weights yields the same audio at a pinned seed.
not_doing:
  - No real-time/cross-chunk morphing — that is T-04.03.
  - No more than the enrolled embeddings as inputs.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Weighted interpolation of enrolled speaker embeddings into a coherent blend. Inputs: two or more enrolled embeddings and blend weights summing to 1. Outputs: a blended embedding and coherent rendered speech. Errors/edges: single-speaker weight reproduces that speaker; even blend sits between; renders stable at a seed. Invariant: weights normalize and the blend is deterministic. Done-check: coherence eval, single-speaker boundary, and render stability. BLOCKED: depends on the embedding-space audit (T-04.01) and needs the trained speaker encoder plus perceptual coherence judging (CLAUDE THE BUILD SCOPE) — not frozen-test gateable.

### T-04.03  Morph the voice in real time
id: T-04.03
phase: 4
status: blocked
depends_on: [T-04.02, T-07.01]
stack: rust
criteria:
  - C1: interpolating speaker embedding live across streamed chunks produces a morph whose chunk-boundary transitions are artifact-free above the agreed perceptual threshold.
  - C2: a zero-rate morph (start embedding equals end embedding) yields audio equal to the static-speaker stream within tolerance (boundary case).
  - C3: the perceived speaker identity at the morph endpoints matches the start and end speakers respectively within the agreed margin.
not_doing:
  - No offline blending — that is T-04.02.
  - No more than a two-endpoint morph trajectory.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Live cross-chunk speaker interpolation during streaming. Inputs: start/end embeddings, a morph trajectory, and the streaming chunk path. Outputs: a continuously morphing stream with clean boundaries. Errors/edges: zero-rate equals static; endpoints match start/end speakers. Invariant: no audible discontinuity at chunk boundaries. Done-check: artifact-free transitions and endpoint identity, perceptually judged. BLOCKED: needs speaker blending (T-04.02) and the streaming packet path (T-07.01) plus perceptual artifact judging with the trained model (CLAUDE THE BUILD SCOPE) — not frozen-test gateable.

### T-04.04  Switch languages bilingually
id: T-04.04
phase: 4
status: blocked
depends_on: [T-02.10]
stack: rust
criteria:
  - C1: a mid-utterance language switch renders with the speaker's timbre held constant across the switch — measured speaker similarity before and after the boundary stays above the agreed threshold.
  - C2: both language spans are intelligible — WER on each span is at or below target on the eval set.
  - C3: the switch boundary is free of timbre break above the agreed perceptual threshold.
not_doing:
  - No more than two languages per utterance.
  - No accent-morphing — identity held fixed across the switch.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Seamless mid-utterance language change with stable timbre. Inputs: text spanning two languages and a fixed speaker. Outputs: rendered audio that flips language without changing voice. Errors/edges: similarity must hold across the boundary; each span intelligible. Invariant: speaker identity is invariant under language switch. Done-check: cross-boundary similarity, per-span WER, and a clean-boundary perceptual check. BLOCKED: depends on cross-lingual/multi-accent validation (T-02.10) and needs the trained model plus SIM-o/WER perceptual gates (CLAUDE THE BUILD SCOPE) — not frozen-test gateable.

### T-04.05  Enroll speaker profiles
id: T-04.05
phase: 4
status: blocked
depends_on: [T-02.03]
stack: rust
criteria:
  - C1: enrolling from a reference clip produces a persisted embedding, and recalling it returns the same embedding bytes that were stored (storage round-trip is exact).
  - C2: re-enrolling the same clip yields an embedding equal to the first within tolerance — enrollment is stable for identical input.
  - C3: rendering from a recalled embedding reproduces the enrolled speaker's identity above the agreed similarity threshold versus rendering from the live embedding.
not_doing:
  - No blending/morphing — single-profile enrollment and recall.
  - No noisy-clip robustness — that is T-07.08.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Store and recall speaker profiles from reference clips. Inputs: a reference clip and a profile store. Outputs: a persisted embedding and stable recall. Errors/edges: storage round-trips exactly; identical clips enroll stably; recall renders the same identity. Invariant: a recalled profile equals what was enrolled. Done-check: storage round-trip, enrollment stability, and recall-identity similarity. BLOCKED: the embedding comes from the trained speaker encoder (T-02.03), which needs ported weights + GPU (CLAUDE THE BUILD SCOPE) — there is no real embedding to persist or recall until a human builds it.

### T-04.06  Evaluate the blend and morph quality
id: T-04.06
phase: 4
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a blind listening panel rates blended voices as coherent at or above the agreed pass rate, and below it for a deliberately incoherent blend to pin the gate.
  - C2: morph transitions are rated artifact-free at or above the agreed threshold on the held-out morph set.
  - C3: the eval is reproducible — the same renders presented to the protocol yield the same aggregate scores under the fixed scoring rubric.
not_doing:
  - No new blend/morph features — measures T-04.02/T-04.03 output.
  - No automated proxy substituting for the listening panel.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Perceptual acceptance gate for blend and morph. Inputs: rendered blends/morphs and a blind listening protocol. Outputs: coherence and artifact-free pass/fail scores. Errors/edges: incoherent blends must fail; scoring rubric fixed for reproducibility. Invariant: scores reflect the agreed perceptual protocol. Done-check: blend coherence, morph artifact ratings, and rubric reproducibility. BLOCKED: this is a blind perceptual listening eval requiring the trained model and human listeners (CLAUDE THE BUILD SCOPE) — it cannot be a frozen-test + mutation gate.

### T-05.01  Define the paralinguistic taxonomy
id: T-05.01
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the label schema enumerates every paralinguistic class (breath, laugh, sigh, throat-clear, hesitation) and every phonation mode (whisper, shout/projection, vocal fry, vocal fatigue) with a stable string identifier per class.
  - C2: the annotation guide gives each class a falsifiable inclusion/exclusion rule such that two trained annotators independently label the same span with inter-annotator agreement ≥ 0.8 (Cohen's kappa) on a curated pilot set.
  - C3: the schema versions itself and is consumed without ambiguity by `syrinx-lm` token vocabulary construction (every taxonomy id maps to exactly one token slot, no collisions).
not_doing:
  - No corpus collection or actual annotation against the schema.
  - No LM token-emission wiring (that is T-05.10).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: prior paralinguistic literature, NovaFox label conventions, legal review of class boundaries. Bounds: a closed, versioned class set covering five artifact classes plus four phonation modes. Outputs: a schema doc plus annotation guide with per-class rules and an agreement target. Errors/edges: ambiguous overlapping classes (e.g. sigh vs breath) must be disambiguated by explicit rule. Invariant: every downstream label references a defined, versioned id. Done-check: the schema/guide review plus a pilot agreement ≥ 0.8. BLOCKED: requires human curation of the taxonomy and consent/licensing judgment on which classes are ethically and legally annotatable, neither expressible as a frozen-test gate.

### T-05.02  Author the corpus sourcing manifest
id: T-05.02
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the manifest records, per source, a provenance entry (origin, license identifier, acquisition date) and a consent record with a verifiable status field (granted / pending / refused).
  - C2: no source enters the collectable set unless its license permits TTS training redistribution and its consent status is exactly "granted"; any other status excludes the source from the pipeline manifest.
  - C3: the manifest's planned aggregate spans ≥ the target hours per phonation mode and per artifact class required by T-05.05's labeling target, with the shortfall per class computed and flagged.
not_doing:
  - No ingestion or normalization of any clip (that is T-05.03).
  - No annotation of collected material.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: candidate corpora, license texts, consent agreements, the T-05.01 class set. Bounds: only granted-consent, training-permissive sources are eligible. Outputs: a per-source provenance + consent manifest with coverage/shortfall accounting. Errors/edges: a source with ambiguous license or missing consent is excluded, not assumed. Invariant: provenance and consent are tracked for every byte that may reach training. Done-check: manifest review against license and consent records. BLOCKED: requires human curation of sources and legal consent/licensing judgment per source, which cannot be adjudicated by a frozen-test gate.

### T-05.03  Build the collection pipeline
id: T-05.03
phase: 5
status: blocked
depends_on: [T-05.02]
stack: rust
criteria:
  - C1: the ingest pipeline converts every consented source clip to the normalized 48kHz mono target format and rejects any clip whose source is absent from the T-05.02 granted manifest.
  - C2: each ingested clip carries metadata linking it back to its provenance/consent manifest entry, and an ingest run emits a coverage report of accumulated hours per class.
  - C3: ingested clips pass a quality floor (clipping, SNR, duration bounds) and material below the floor is quarantined rather than admitted.
not_doing:
  - No forced alignment of ingested clips (that is T-05.04).
  - No paralinguistic annotation of clips.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the granted manifest, raw source recordings. Bounds: only manifest-listed sources; 48kHz mono normalized output. Outputs: normalized clips with provenance-linked metadata and a coverage report. Errors/edges: a clip from an unlisted source, or below the quality floor, is rejected/quarantined. Invariant: every admitted clip traces to a consented source. Done-check: ingest run produces normalized clips plus coverage report. BLOCKED: requires the human-curated consented corpus to exist and human QC of audio admission, neither available to the autonomous loop.

### T-05.04  Build the forced-alignment tooling
id: T-05.04
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the aligner emits per-phoneme start/end timestamps for each clip referenced to the `syrinx-frontend` phoneme inventory, with monotonically non-decreasing boundaries covering the clip with no gaps or overlaps.
  - C2: alignment accuracy on a hand-labeled held-out set meets the boundary tolerance target (median absolute boundary error ≤ the configured threshold, and just past the threshold flagged as failing).
  - C3: clips that fail to align above a confidence floor are flagged for manual review rather than emitted as accepted alignments.
not_doing:
  - No paralinguistic-event labeling (that is T-05.05).
  - No model training using the alignments.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: normalized clips, transcripts, the frontend phoneme inventory. Bounds: phoneme-timestamped output with monotonic, gapless boundaries. Outputs: per-clip phoneme alignment plus per-clip confidence. Errors/edges: low-confidence alignments are flagged, not silently accepted. Invariant: timestamps tile the clip without overlap. Done-check: median boundary error ≤ threshold on the labeled held-out set. BLOCKED: requires an acoustic alignment model and a human-labeled boundary reference set to measure accuracy, neither expressible as a frozen-test gate.

### T-05.05  Annotate the paralinguistic events
id: T-05.05
phase: 5
status: blocked
depends_on: [T-05.03, T-05.04]
stack: rust
criteria:
  - C1: the labeled set spans ≥ the target hours and labels every artifact class (breath, laugh, sigh, throat-clear, hesitation) with span-level boundaries aligned to the T-05.04 phoneme timeline.
  - C2: inter-annotator agreement across independent annotators is ≥ 0.8 (Cohen's kappa) per class, with any class below 0.8 sent back for guideline revision.
  - C3: every label references a defined T-05.01 class id and a provenance-linked T-05.03 clip; orphan labels (unknown class or unknown clip) are zero.
not_doing:
  - No model or LoRA training on the labels (that is T-05.11).
  - No phonation-mode collection (T-05.06 through T-05.09).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: normalized aligned clips, the T-05.01 schema, trained annotators. Bounds: ≥ target hours, agreement ≥ 0.8 per class. Outputs: span-level paralinguistic labels keyed to phoneme timestamps. Errors/edges: a class below agreement is revised, not shipped; orphan labels are rejected. Invariant: every label references a known class and a consented clip. Done-check: hours ≥ target and kappa ≥ 0.8 across classes. BLOCKED: requires human annotators and multi-pass annotation to reach the agreement and hours targets, which no automated gate can produce.

### T-05.06  Collect the whisper-mode set
id: T-05.06
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the whispered set spans ≥ the target hours of consented, normalized clips, each carrying the whisper phonation-mode control label from T-05.01.
  - C2: a phonation-mode classifier (or human audit) confirms ≥ the target fraction of clips are genuinely whispered (low periodicity / absent voicing), and clips below the confidence floor are excluded.
  - C3: every clip traces to a granted-consent T-05.02 source and is excluded otherwise.
not_doing:
  - No transition modeling between whisper and modal speech (that is T-05.12).
  - No collection of other phonation modes.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: granted whisper-mode sources. Bounds: ≥ target hours, whisper-labeled. Outputs: labeled whispered clip set. Errors/edges: non-whispered or unconsented clips excluded. Invariant: every clip is consented and mode-labeled. Done-check: hours ≥ target and whisper-purity ≥ target. BLOCKED: requires human collection and perceptual/auditory confirmation of whisper phonation, not gateable by frozen tests.

### T-05.07  Collect the shout-projection set
id: T-05.07
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the shout/projection set spans ≥ the target hours of consented, normalized clips, each carrying the projection phonation-mode label from T-05.01.
  - C2: a level/spectral audit confirms ≥ the target fraction exhibit projected phonation (elevated SPL and spectral tilt within the projected-voice band), excluding clips below the floor.
  - C3: every clip traces to a granted-consent T-05.02 source; non-consented clips are excluded.
not_doing:
  - No loudness-normalization training tricks (collection only).
  - No other phonation modes.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: granted projection-mode sources. Bounds: ≥ target hours, projection-labeled. Outputs: labeled shout/projection clip set. Errors/edges: under-projected or unconsented clips excluded. Invariant: every clip consented and mode-labeled. Done-check: hours ≥ target and projection-purity ≥ target. BLOCKED: requires human collection and perceptual confirmation of projected phonation, not expressible as a frozen-test gate.

### T-05.08  Collect the vocal-fry set
id: T-05.08
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the vocal-fry set spans ≥ the target hours of consented, normalized clips, each carrying the vocal-fry phonation-mode label from T-05.01.
  - C2: an audit confirms ≥ the target fraction exhibit creaky/fry phonation (sub-harmonic, irregular low-F0 pulses), excluding clips below the confidence floor.
  - C3: every clip traces to a granted-consent T-05.02 source; otherwise excluded.
not_doing:
  - No fry-level controllability modeling (that is T-05.13).
  - No other phonation modes.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: granted vocal-fry sources. Bounds: ≥ target hours, fry-labeled. Outputs: labeled vocal-fry clip set. Errors/edges: non-fry or unconsented clips excluded. Invariant: every clip consented and mode-labeled. Done-check: hours ≥ target and fry-purity ≥ target. BLOCKED: requires human collection and perceptual confirmation of creaky/fry phonation, not gateable by frozen tests.

### T-05.09  Collect the vocal-fatigue set
id: T-05.09
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the vocal-fatigue set spans ≥ the target hours of consented, normalized clips, each carrying the fatigue phonation-mode label from T-05.01.
  - C2: an audit confirms ≥ the target fraction exhibit fatigue markers (reduced range, increased jitter/shimmer beyond the fatigue threshold) and excludes clips below the floor.
  - C3: every clip traces to a granted-consent T-05.02 source; otherwise excluded.
not_doing:
  - No fatigue-progression synthesis modeling.
  - No other phonation modes.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: granted vocal-fatigue sources. Bounds: ≥ target hours, fatigue-labeled. Outputs: labeled vocal-fatigue clip set. Errors/edges: non-fatigued or unconsented clips excluded. Invariant: every clip consented and mode-labeled. Done-check: hours ≥ target and fatigue-marker purity ≥ target. BLOCKED: requires human collection and perceptual confirmation of fatigue phonation, not expressible as a frozen-test gate.

### T-05.10  Extend the LM paralinguistic vocabulary
id: T-05.10
phase: 5
status: blocked
depends_on: [T-02.02, T-05.05]
stack: rust
criteria:
  - C1: the `syrinx-lm` token vocabulary adds one token per T-05.01 paralinguistic class and phonation mode, with no collision against existing semantic tokens and a stable id-to-token mapping.
  - C2: each new token round-trips end-to-end — emitted by the LM forward pass, decoded through the pipeline, and recoverable from the output stream with zero loss.
  - C3: a sequence containing paralinguistic tokens parses and renders without corrupting adjacent semantic tokens (boundary token before and after the artifact preserved).
not_doing:
  - No LoRA fine-tuning for controllable triggering (that is T-05.11).
  - No taxonomy redefinition (owned by T-05.01).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the T-05.02-trained LM forward pass, the T-05.05 labels, the T-05.01 schema. Bounds: one token per class/mode, collision-free. Outputs: an extended token vocabulary that round-trips through the pipeline. Errors/edges: a colliding or unmapped token id fails construction. Invariant: existing semantic tokens are unperturbed. Done-check: every new token round-trips and neighbors survive. BLOCKED: requires the trained Rust LM forward pass (T-02.02) and the human-built labeled corpus (T-05.05) to exist on a GPU before tokens can emit, so it cannot be gated autonomously.

### T-05.11  Fine-tune paralinguistic insertion control
id: T-05.11
phase: 5
status: blocked
depends_on: [T-05.10]
stack: rust
criteria:
  - C1: a LoRA/fine-tune over the extended vocabulary triggers each paralinguistic artifact on demand with ≥ the target precision/recall against held-out annotations.
  - C2: enabling a control token produces the artifact and disabling it suppresses the artifact in matched renders (both sides of the toggle verified, not just the on-state).
  - C3: insertion preserves baseline intelligibility — WER on control utterances stays within the configured budget of the un-triggered baseline.
not_doing:
  - No mode-transition modeling (that is T-05.12).
  - No contextual/dynamic auto-triggering (that is T-05.13).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the extended-vocabulary LM, labeled artifact data, GPUs. Bounds: controllable per-artifact triggering within a WER budget. Outputs: a fine-tuned adapter that inserts artifacts on command. Errors/edges: triggering that breaks intelligibility beyond budget fails. Invariant: the toggle is bidirectional and intelligibility-preserving. Done-check: per-artifact precision/recall ≥ target and WER within budget. BLOCKED: requires GPU fine-tuning over the human-built corpus and held-out perceptual/ASR evaluation, none of which a frozen-test gate can supply.

### T-05.12  Model whisper-to-spoken transitions
id: T-05.12
phase: 5
status: blocked
depends_on: [T-05.06, T-05.11]
stack: rust
criteria:
  - C1: a mid-utterance whisper↔modal switch renders without a discontinuity artifact — boundary energy/F0 continuity stays within the configured smoothness threshold (and just past it flagged as a failure).
  - C2: both transition directions (whisper→spoken and spoken→whisper) are verified, not only one.
  - C3: blind listeners rate the transitions clean at ≥ the target preference rate versus a hard-cut baseline.
not_doing:
  - No new whisper data collection (consumes T-05.06).
  - No other phonation-mode transitions beyond whisper↔modal.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the whisper set, the fine-tuned insertion adapter. Bounds: continuous, artifact-free mid-utterance mode switches in both directions. Outputs: clean whisper↔spoken transition rendering. Errors/edges: a boundary discontinuity beyond threshold fails. Invariant: both directions hold. Done-check: smoothness within threshold and blind preference ≥ target. BLOCKED: requires the trained insertion model and human blind-listening evaluation of transition cleanliness, not expressible as a frozen-test gate.

### T-05.13  Add contextual paralinguistic triggering
id: T-05.13
phase: 5
status: blocked
depends_on: [T-05.11]
stack: rust
criteria:
  - C1: context-driven injection (laughter, hesitation, organic throat-clear) fires from textual/contextual cues at ≥ the target appropriateness rate judged on a held-out eval, with false-trigger rate below the configured ceiling.
  - C2: vocal-fry level is a continuous control where increasing the level monotonically increases measured creak density across the range (monotonicity checked at the endpoints and an interior point).
  - C3: dynamic triggering preserves intelligibility within the WER budget relative to the un-triggered baseline.
not_doing:
  - No new corpus collection (consumes T-05.11 outputs).
  - No mode-transition modeling (that is T-05.12).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the fine-tuned adapter, contextual cue features. Bounds: context-appropriate injection plus a monotonic fry-level knob. Outputs: dynamic, level-adjustable paralinguistic triggering. Errors/edges: over-triggering beyond the false-positive ceiling fails. Invariant: level control is monotonic and intelligibility-preserving. Done-check: appropriateness ≥ target, monotonic fry level, WER within budget. BLOCKED: requires the trained control model and perceptual evaluation of contextual appropriateness, neither gateable by frozen tests.

### T-05.14  Evaluate paralinguistic organic-ness
id: T-05.14
phase: 5
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: in a blind A/B against human recordings, synthesized paralinguistic artifacts are rated natural at ≥ the target naturalness threshold with the human reference as the upper anchor.
  - C2: the eval covers every artifact class and phonation mode, with per-class naturalness reported and any class below threshold flagged.
  - C3: the listening panel meets the minimum rater count and the result reaches the configured statistical-significance bound.
not_doing:
  - No model changes (evaluation only).
  - No automated proxy substituting for human ratings.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: rendered artifacts, human reference clips, a blind listening panel. Bounds: naturalness ≥ target per class at significance. Outputs: a per-class organic-ness report. Errors/edges: a class below threshold is flagged for rework. Invariant: human reference anchors the scale. Done-check: per-class naturalness ≥ target at the required significance. BLOCKED: requires a human blind-listening panel to rate naturalness, an inherently perceptual judgment no frozen-test gate can render.

### T-06.01  Tag the attribute label set
id: T-06.01
phase: 6
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a tagged subset assigns each clip age, gender, and accent labels from the defined attribute schema, with every value drawn from the closed enumerated set (no free-text leakage).
  - C2: attribute inter-annotator agreement is ≥ the target per axis, with any axis below target sent back for guideline revision.
  - C3: every tagged clip traces to a granted-consent source and references a defined attribute id; orphan or unconsented tags are zero.
not_doing:
  - No conditioning wiring into the model (that is T-06.02).
  - No disentanglement training.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: consented clips, the attribute schema, annotators. Bounds: closed-enum age/gender/accent tags at the agreement target. Outputs: an attribute-tagged subset ready for conditioning. Errors/edges: out-of-enum or unconsented tags rejected. Invariant: every tag references a known attribute and a consented clip. Done-check: per-axis agreement ≥ target and zero orphan tags. BLOCKED: requires human annotation of sensitive demographic attributes plus consent judgment, which no frozen-test gate can adjudicate.

### T-06.02  Wire attribute conditioning inputs
id: T-06.02
phase: 6
status: blocked
depends_on: [T-06.01, T-02.04]
stack: rust
criteria:
  - C1: age, gender, and accent enter the model as a separate conditioning input distinct from the timbre embedding, with a typed conditioning vector whose dimensions map one-to-one to the attribute axes.
  - C2: changing a conditioning value while holding the reference embedding fixed changes the conditioned output (the path is live), and a null/default conditioning reproduces the unconditioned baseline.
  - C3: the conditioning interface is versioned and consumed by the `syrinx-acoustic` decoder without altering the timbre-embedding contract.
not_doing:
  - No adversarial disentanglement loss (that is T-06.03).
  - No per-axis perceptual eval (T-06.04 through T-06.06).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the tagged subset, the flow-matching decoder (T-02.04). Bounds: attributes as a separate, versioned conditioning vector. Outputs: an attribute-conditioned decoder path. Errors/edges: null conditioning must equal the baseline. Invariant: the timbre-embedding contract is unchanged. Done-check: conditioning is live and the default reproduces baseline. BLOCKED: requires the trained acoustic decoder (T-02.04) on a GPU and the human-tagged subset before conditioning can be exercised, so it is not autonomously gateable.

### T-06.03  Add adversarial disentanglement loss
id: T-06.03
phase: 6
status: blocked
depends_on: [T-06.02]
stack: rust
criteria:
  - C1: with the adversarial loss trained, a classifier predicting age/gender/accent from the timbre embedding drops toward chance accuracy (within the configured margin of the per-axis chance rate).
  - C2: stripping attributes from timbre does not collapse cloning fidelity — SIM-o stays within the configured budget of the pre-disentanglement baseline.
  - C3: the conditioning path still steers the stripped attribute (the attribute is controllable via conditioning even though it is absent from timbre).
not_doing:
  - No per-axis perceptual eval (T-06.04 through T-06.06).
  - No final metrics writeup (that is T-06.07).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the conditioned model, attribute classifiers, GPUs. Bounds: classifier-on-timbre accuracy → chance within budget while preserving SIM-o. Outputs: a disentangled timbre embedding plus a controllable conditioning path. Errors/edges: disentanglement that tanks SIM-o beyond budget fails. Invariant: stripped attributes remain controllable via conditioning. Done-check: classifier accuracy near chance and SIM-o within budget. BLOCKED: requires adversarial GPU training and SIM-o evaluation against a trained model, none of which a frozen-test gate can produce.

### T-06.04  Evaluate the age-progression axis
id: T-06.04
phase: 6
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: sweeping the age knob monotonically shifts listener-perceived age across the range (monotonicity verified at both endpoints and an interior point).
  - C2: the shift is measurably independent — gender and accent perception stay within the configured tolerance while age moves (independence quantified, not asserted).
  - C3: blind raters meet the minimum panel size and the age-shift effect reaches the configured significance bound.
not_doing:
  - No model retraining (evaluation only).
  - No gender or dialect axis evaluation.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the disentangled model, a blind rating panel. Bounds: monotonic perceived-age shift with bounded cross-axis leakage at significance. Outputs: an age-axis independence report. Errors/edges: non-monotonic or leaky shifts fail. Invariant: only the age percept moves materially. Done-check: monotonic age shift with bounded leakage at significance. BLOCKED: requires human perceptual rating of perceived age and cross-axis independence, an inherently auditory judgment outside any frozen-test gate.

### T-06.05  Evaluate gender-neutral synthesis
id: T-06.05
phase: 6
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: a gender-neutral target is reachable — blind raters classify the neutral setting as ambiguous at ≥ the target ambiguity rate (neither clearly masculine nor feminine).
  - C2: timbre stays stable across the gender sweep — speaker identity (SIM-o against the reference) stays within the configured budget while gender is neutralized.
  - C3: the rating panel meets the minimum size and the neutrality result reaches the configured significance bound.
not_doing:
  - No model retraining (evaluation only).
  - No age or dialect axis evaluation.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the disentangled model, a blind rating panel. Bounds: a reachable neutral point with stable timbre at significance. Outputs: a gender-neutrality eval report. Errors/edges: timbre drift beyond budget fails. Invariant: identity holds while gender neutralizes. Done-check: ambiguity ≥ target and SIM-o within budget at significance. BLOCKED: requires human perceptual rating of gender ambiguity, a judgment by ear that no frozen-test gate can render.

### T-06.06  Evaluate the dialect-shifting axis
id: T-06.06
phase: 6
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: sweeping the dialect knob shifts listener-perceived accent toward the target dialect at ≥ the target recognition rate by raters familiar with the dialect.
  - C2: the shift shows partial independence — perceived age and gender stay within the configured tolerance while accent moves (leakage quantified, partial-result tolerant).
  - C3: the rating panel meets the minimum size and the dialect-shift effect reaches the configured significance bound.
not_doing:
  - No model retraining (evaluation only).
  - No age or gender axis evaluation.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the disentangled model, dialect-familiar raters. Bounds: target-dialect recognition with partial cross-axis independence at significance. Outputs: a dialect-axis independence report. Errors/edges: unrecognized shifts or heavy leakage fail. Invariant: accent moves while other axes stay bounded. Done-check: recognition ≥ target with bounded leakage at significance. BLOCKED: requires human perceptual rating of accent shift by dialect-familiar listeners, an auditory judgment outside any frozen-test gate.

### T-06.07  Report the disentanglement metrics
id: T-06.07
phase: 6
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the report records per-axis independence scores (classifier-on-timbre accuracy versus chance, cross-axis leakage) for age, gender, and accent with the source eval each derives from.
  - C2: partial results are reported honestly — every axis below its independence target is labeled partial with the residual leakage stated, not omitted or rounded up.
  - C3: each reported number traces to a named eval run (T-06.03 through T-06.06) with no orphan or unsourced metric.
not_doing:
  - No model changes (reporting only).
  - No new evaluation runs beyond aggregating existing ones.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the disentanglement and per-axis eval results. Bounds: an honest, fully-sourced per-axis independence writeup. Outputs: a disentanglement metrics report with partial-result labeling. Errors/edges: an unsourced or optimistically rounded metric fails review. Invariant: every number traces to a named run. Done-check: per-axis scores sourced and partials labeled. BLOCKED: requires the upstream perceptual and training evals (T-06.03 through T-06.06) to have produced real human/GPU results before they can be aggregated, so it is not autonomously gateable.

### T-07.01  Stream the packet path
id: T-07.01
phase: 7
status: blocked
depends_on: [T-02.06]
stack: rust
criteria:
  - C1: `syrinx-stream` emits chunked audio packets through a ring buffer to a `cpal` output with no buffer underruns across a sustained synthesis run.
  - C2: the ring buffer never reports an underrun event while the producer keeps pace, and reports exactly one underrun event the first time the consumer outpaces the producer.
  - C3: packet ordering is monotonic and lossless: every emitted packet index is delivered to the sink exactly once, in order.
not_doing:
  - No TTFB latency tuning (that is T-07.02).
  - No telephony or 8kHz resampling path.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: a live stream of decoded audio chunks from the chunk-aware decoder. Bounds: producer/consumer rates set by real-time playback. Outputs: a continuous `cpal` audio stream with a ring buffer between produce and consume. Errors/edges: consumer outpacing producer must surface an underrun event, not silently corrupt. Invariant: packets are delivered in order, exactly once, with no underruns under nominal rate. Done-check: a live no-underrun playback run on the running model. BLOCKED: requires the trained model running on a GPU to produce a real-time decoded-chunk stream; no-underrun behaviour is only observable against live synthesis, so it is not frozen-test gateable.

### T-07.02  Tune the time-to-first-byte path
id: T-07.02
phase: 7
status: blocked
depends_on: [T-07.01]
stack: rust
criteria:
  - C1: end-to-end streaming TTFB measured at the `syrinx-stream` sink is under 200ms at the p50 over a representative request set.
  - C2: at the chosen chunk size, p50 TTFB is below the 200ms threshold and reducing chunk size further does not push p50 to or above 200ms.
  - C3: the chunk-size/quality trade chosen keeps p50 TTFB under 200ms without dropping output quality below the established acoustic bar.
not_doing:
  - No overall RTF optimization (that is T-07.03).
  - No change to the streaming packet transport itself.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: streaming requests through the live decode+stream path with a tunable first-chunk size. Bounds: first-chunk latency budget of 200ms p50. Outputs: a chunk-size setting meeting the TTFB target. Errors/edges: chunk too small degrades quality, too large blows the budget. Invariant: p50 TTFB < 200ms while quality holds. Done-check: a measured p50 latency under load. BLOCKED: requires the trained model running on a GPU; TTFB is a wall-clock latency of live inference and cannot be gated by frozen tests.

### T-07.03  Optimize the real-time factor
id: T-07.03
phase: 7
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: the real-time factor of full synthesis on a single RTX 4090 is at or below the project RTF target.
  - C2: kernel fusion and batching reduce measured RTF below the target threshold, and disabling them regresses RTF to at or above the target.
  - C3: optimized output is numerically equivalent to the unoptimized path within the established tolerance, so speed is not bought with quality.
not_doing:
  - No TTFB-specific first-chunk tuning.
  - No multi-GPU or distributed inference.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the full inference pipeline on one 4090 with fusion/batching toggles. Bounds: the RTF target on a single 4090. Outputs: a fused/batched path meeting RTF. Errors/edges: fusion that changes numerics beyond tolerance is a regression. Invariant: RTF <= target with output parity preserved. Done-check: a measured RTF under target on a 4090. BLOCKED: requires the trained model and a physical RTX 4090; RTF is a wall-clock throughput measurement of GPU inference and is not frozen-test gateable.

### T-07.04  Synthesize the telephony path
id: T-07.04
phase: 7
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: `syrinx-vocoder` produces a validated 8kHz narrowband output via resample plus band-limit plus codec from the full-band synthesis.
  - C2: the band-limited output retains energy only within the narrowband passband and is attenuated above the cutoff, verified against a measured spectral bound.
  - C3: the 8kHz output remains intelligible over a narrowband channel as judged against the perceptual intelligibility bar.
not_doing:
  - No 48kHz full-band path changes.
  - No echo cancellation or network jitter handling.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: full-band synthesized waveform from the running vocoder. Bounds: 8kHz narrowband telephony band. Outputs: a resampled, band-limited, codec-encoded 8kHz stream. Errors/edges: aliasing on downsample or out-of-band energy fails the path. Invariant: output is narrowband-valid and intelligible. Done-check: validated 8kHz output judged intelligible. BLOCKED: requires the trained model producing real waveform output, and narrowband intelligibility is a perceptual judgment; neither is frozen-test gateable.

### T-07.05  Harden noise robustness
id: T-07.05
phase: 7
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: voice cloning from a noisy reference clip stays within the cloning-quality budget after reference denoise plus augmentation.
  - C2: speaker similarity from a noisy reference is at or above the SIM-o budget threshold, and a reference below the noise floor still does not drop similarity past the cliff.
  - C3: denoise plus augmentation improves cloned-output quality on noisy references versus the no-denoise baseline by the documented margin.
not_doing:
  - No enrollment-time pipeline changes (that is T-07.08).
  - No clean-reference path regression work.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: noisy reference clips into the speaker/cloning path. Bounds: a SIM-o / quality budget under noise. Outputs: stable cloning within budget. Errors/edges: noise beyond the floor must degrade gracefully, not cliff. Invariant: cloned quality stays within budget on noisy input. Done-check: measured SIM-o within budget on noisy references. BLOCKED: requires the trained speaker model on a GPU and SIM-o/perceptual judgment of cloned output; not frozen-test gateable.

### T-07.06  Export the lip-sync timeline
id: T-07.06
phase: 7
status: pending
depends_on: [T-01.04]
stack: rust
criteria:
  - C1: `syrinx-prosody` maps a list of (phoneme, start_ms, end_ms) entries to a viseme timeline where each phoneme resolves to its viseme class per the fixed phoneme-to-viseme table.
  - C2: the output timeline covers the full input span contiguously with no gaps and no overlaps: each segment's start equals the previous segment's end, and the last end equals the input's last end.
  - C3: an empty input list yields an empty timeline, and an unknown phoneme maps to the neutral/rest viseme without panicking.
not_doing:
  - No audio alignment from model output (input is a typed timestamp list).
  - No interpolation or smoothing between visemes.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: a typed list of (phoneme, start_ms, end_ms) entries. Bounds: timestamps ordered and contiguous across the input span. Outputs: a viseme timeline of (viseme_class, start_ms, end_ms) segments. Errors/edges: empty input yields empty output; an unknown phoneme yields the neutral/rest viseme and never panics. Invariant: the timeline covers the input span with no gaps or overlaps and every phoneme maps via the fixed table. Done-check: the deterministic table-mapping, span-coverage, empty-input, and unknown-phoneme criteria.

### T-07.07  Measure the footprint under stress
id: T-07.07
phase: 7
status: blocked
depends_on: [T-02.12]
stack: rust
criteria:
  - C1: the resident memory footprint at 4-bit quantization is at or below 300MB during synthesis on a single RTX 4090.
  - C2: under the concurrency target the engine sustains all concurrent requests without an out-of-memory failure, and at one request beyond the target it still does not OOM within the documented headroom.
  - C3: footprint at 4-bit stays at or below 300MB while full-precision exceeds it, confirming the quantized path is what meets the budget.
not_doing:
  - No throughput/RTF tuning.
  - No multi-GPU scaling.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the quantized engine under concurrent load on one 4090. Bounds: <=300MB at 4-bit and a fixed concurrency target. Outputs: a stress report meeting footprint and concurrency. Errors/edges: OOM at or below the concurrency target fails the gate. Invariant: footprint <= 300MB and no OOM at the concurrency target. Done-check: measured resident memory and a no-OOM concurrency run on a 4090. BLOCKED: requires the quantized trained model and a physical RTX 4090; memory footprint and OOM behaviour under concurrency are live-hardware measurements, not frozen-test gateable.

### T-07.08  Enroll from noisy references
id: T-07.08
phase: 7
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: speaker enrollment from a background-noise reference clip produces an embedding whose cloned output stays at or above the quality bar with no quality cliff.
  - C2: a noisy enrollment clip yields cloned-output quality at or above the threshold, and a clip just past the supported noise floor degrades gracefully rather than cliffing.
  - C3: enrollment from a noisy clip reaches quality within the documented margin of enrollment from the equivalent clean clip.
not_doing:
  - No inference-time noise robustness (that is T-07.05).
  - No corpus-side annotation work.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: noisy reference clips at enrollment time into the speaker encoder. Bounds: a quality bar relative to clean enrollment. Outputs: a robust speaker embedding from noisy input. Errors/edges: noise past the floor must degrade gracefully, not cliff. Invariant: noisy enrollment quality stays within margin of clean. Done-check: measured cloned-output quality from noisy enrollment. BLOCKED: requires the trained speaker encoder on a GPU and perceptual quality judgment of cloned output; not frozen-test gateable.

### T-08.01  Scaffold the audio server
id: T-08.01
phase: 8
status: pending
depends_on: [T-00.01]
stack: rust
criteria:
  - C1: `syrinx-serve` exposes an Axum `/v1/audio/speech` route whose handler accepts a typed request (model, input, voice, response_format) and calls a pluggable synth trait whose default stub returns a fixed silent buffer.
  - C2: a well-formed POST to `/v1/audio/speech` returns 200 with the expected audio content-type, and a request missing a required field returns 422 with a typed error body.
  - C3: when response_format selects streaming, the handler returns the streaming response shape rather than a single buffered body.
  - C4: the router registers the `/v1/audio/speech` route exactly once.
not_doing:
  - No real synthesis behind the trait (default stub returns a silent buffer).
  - No health or version endpoint (that is T-08.02).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: typed OpenAI-style speech requests over HTTP into the Axum router. Bounds: required fields model/input/voice/response_format. Outputs: 200 with audio content-type from the stub synth, or the streaming response shape. Errors/edges: a missing required field returns 422 with a typed error body. Invariant: the route is registered exactly once and the synth is a pluggable trait defaulting to a silent-buffer stub. Done-check: the 200-content-type, 422-typed-error, streaming-shape, and single-registration criteria.

### T-08.02  Add the parity health endpoint
id: T-08.02
phase: 8
status: pending
depends_on: [T-08.01]
stack: rust
criteria:
  - C1: `syrinx-serve` exposes a `/v1/health` route that returns 200 with the documented typed JSON body.
  - C2: a GET to `/v1/health` returns 200 with the documented JSON shape, and a non-GET method to the same path returns 405 method-not-allowed.
  - C3: the `/v1/health` route is registered exactly once and the returned JSON deserializes into the documented typed health struct.
not_doing:
  - No liveness/readiness orchestration or dependency probing.
  - No metrics or telemetry endpoint.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: HTTP requests to the health/version path on the Axum router. Bounds: GET-only. Outputs: 200 with a typed JSON health body. Errors/edges: a non-GET method returns 405. Invariant: the route is registered once and the body matches the documented typed shape. Done-check: the 200-JSON-shape, 405-on-non-GET, and single-registration criteria.

### T-08.03  Wire NovaFox onto Syrinx
id: T-08.03
phase: 8
status: blocked
depends_on: [T-08.01]
stack: rust
criteria:
  - C1: NovaFox v2 runs end-to-end on Syrinx with OpenVoice/Piper fully replaced, producing speech through the `/v1/audio` server path.
  - C2: the end-to-end NovaFox run produces audio output indistinguishable in quality from the prior stack at or above the acceptance bar, and no request path still routes to OpenVoice/Piper.
  - C3: NovaFox v2 exercises the real synth trait implementation rather than the silent-buffer stub.
not_doing:
  - No new API surface beyond what NovaFox consumes.
  - No release packaging (that is T-08.05).
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: NovaFox v2 driving the live Syrinx server with the real synth implementation. Bounds: end-to-end parity with the replaced stack. Outputs: NovaFox running entirely on Syrinx. Errors/edges: any residual OpenVoice/Piper path fails the swap. Invariant: NovaFox produces acceptable speech end-to-end on Syrinx only. Done-check: a live end-to-end NovaFox run at the quality bar. BLOCKED: requires the whole trained engine wired behind the server (real synth, not the stub) plus perceptual quality acceptance; not frozen-test gateable.

### T-08.04  Author the project documentation
id: T-08.04
phase: 8
status: blocked
depends_on: []
stack: rust
criteria:
  - C1: documentation covers the API surface, the control surface, deployment, and ethics, each section complete with worked examples.
  - C2: every documented API and control-surface example is accurate against the shipped server and runs as written.
  - C3: the ethics section states the consent, disclosure, and misuse-prevention stance required for a voice-cloning release.
not_doing:
  - No release packaging or model card (that is T-08.05).
  - No auto-generated API reference replacing human-authored prose.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the finished server and control surfaces to be documented. Bounds: API, control, deployment, and ethics sections. Outputs: complete docs with runnable examples. Errors/edges: an example that does not match the shipped surface fails review. Invariant: docs are complete, accurate, and ethics-bearing. Done-check: human review of completeness and example accuracy. BLOCKED: docs are human-authored against the whole finished engine and require human judgment of completeness, accuracy, and the ethics stance; not frozen-test gateable.

### T-08.05  Package the release
id: T-08.05
phase: 8
status: blocked
depends_on: [T-00.09]
stack: rust
criteria:
  - C1: a versioned release artifact is produced and published alongside a model card and an ethics statement.
  - C2: the model card documents training data provenance, capabilities, limitations, and the consent/misuse policy to the publication bar.
  - C3: the release version, model card, and ethics statement are mutually consistent and legally cleared for publication.
not_doing:
  - No new engine features in the release.
  - No post-release operations or update channel.
test_files: []
criteria_map: {}
attempts: 0
last_failure: ""
---
Inputs: the finished, documented engine plus the license-screen matrix. Bounds: one versioned release with a published model card. Outputs: a release artifact, model card, and ethics statement. Errors/edges: missing provenance or unresolved license clearance blocks publication. Invariant: the release ships with a complete, legally cleared model card and ethics statement. Done-check: human/legal sign-off on the published release. BLOCKED: requires the trained model to be card-able plus human and legal judgment on provenance, the model card, and the ethics statement; not frozen-test gateable.
