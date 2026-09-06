#!/usr/bin/env bash
# =============================================================================
# test-groups.sh — the SINGLE source of truth for what the suite is made of.
#
# Sourced by both scripts/test-all.sh and scripts/verify.sh. It defines nothing
# executable: only the group -> tests map, the family -> groups map, and the
# selector resolver they share.
#
# Why it exists: the two runners each carried their own copy of every GROUP_*
# line, so adding one test meant editing both, and a drifted copy would silently
# run a different suite in each. One definition, two consumers.
#
# Vocabulary:
#   test    one repo-root tests/<name>.rs binary
#   group   a named set of tests that share a prerequisite (weights, fixtures)
#   family  a named set of groups — usually one model family, plus a few
#           meta-families for "everything model-free" and the like
#
# A family name is never also a group name, so a bare selector is unambiguous.
# =============================================================================

# ---- group -> test files ----------------------------------------------------
# (model-free groups run everywhere; the rest self-skip without their env vars)

# Every member crate's #[cfg(test)] modules, in one row. `crate_unit_tests` is a
# PSEUDO-TEST, not a tests/<name>.rs file — see the pseudo-test section below for
# why the row exists and how the runners execute it.
GROUP_unit="crate_unit_tests"

GROUP_modelfree="voice_lib emotion_tags watermark audio_server health_endpoint server_hardening real_cv3_quality_source real_cv3_multinomial"

GROUP_cv2="real_lm_parity real_lm_gen_parity real_lm_kvcache real_lm_quant real_feat_parity real_tokenizer_parity real_textnorm_parity real_speech_token_parity real_flow_parity real_flow_stream_consistency real_vocoder_parity real_speaker_parity real_token2wav_parity real_quant_footprint"
GROUP_cv2e2e="real_synth_e2e real_eval_metrics real_eval_multilingual real_lm_hammer"
GROUP_cv3="real_cv3_lm_parity real_cv3_flow_parity real_cv3_flow_stream_consistency real_cv3_hift_parity real_cv3_stok_parity real_cv3_ras real_cv3_quant_parity real_cv3_quant_footprint"
GROUP_cv3e2e="real_cv3_e2e_parity real_cv3_eval_metrics real_cv3_voice real_cv3_emotion"

# Fish groups — the runner tolerates absent files (reports MISSING) so this stays
# stable as the port lands.
GROUP_fish_s1="real_fish_s1_parity real_fish_s1_e2e"
GROUP_fish_s2="real_fish_s2_parity real_fish_s2_e2e real_fish_s2_codec_clamp"

# STT (pure-Rust Whisper) — the audio->text reverse path + the native TTS oracle.
# Env-gated on a Whisper model dir + a test clip; self-skips off-box. Download:
#   hf download openai/whisper-base --local-dir "$SYRINX_STT_MODEL_DIR"
GROUP_stt="real_stt"

# Expressive control (the cue layer: syrinx-cue + its wiring). Model-free and
# deterministic — these run everywhere and must never SKIP.
GROUP_cue="control_survey_gate claude_md_invariant_gate cue_token_alignment prosody_cue_overrides expressive_api cue_activation_gate cue_activation_measure instruct_phrases cue_contrast_gate cue_tune_decision"

# Qwen3-TTS port (syrinx-qwen). Model-FREE half only: the geometry contract, the
# loader's tensor manifest against the published safetensors HEADERS (checked in
# under tests/golden/qwen/, no weight data), and the pure-Rust sampling stack. The
# weight-backed half is NOT here — see docs/backends/QWEN_PORT_STATUS.md.
GROUP_qwen="qwen_config_contract qwen_tensor_manifest qwen_sampling_contract qwen_server"

# Weight-backed Qwen tests. Kept OUT of GROUP_qwen deliberately: that group is model-free
# and must never SKIP, while these self-skip without a checkpoint and the reference dump
# from scripts/gen-qwen-ref.py.
# real_qwen_greedy_parity is NOT here — it is an opt-in test, see OPT_IN_TESTS below.
# Measured on NovaBox (CPU/f32, --features real --release, warm build): these four take
# 23 s together, where the five-test group took 21.6 min.
GROUP_qwen_ckpt="real_qwen_prompt_parity real_qwen_stack_parity real_qwen_encode_parity real_qwen_speaker_parity"

ALL_GROUPS="unit modelfree cue qwen cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt qwen_ckpt"

# ---- opt-in tests (deliberately in NO group) ---------------------------------
# A test named here is reachable only by naming it:
#
#     ./scripts/test-all.sh --test <name>          (verify.sh takes the same selector)
#
# It belongs to no group and no family, so neither a bare `test-all.sh`, nor a bare
# `verify.sh`, nor any family selector ever fires it. That is reserved for gates whose
# cost is measured in tens of minutes: real gates, run deliberately, never on a routine
# board. Putting one in a group of its own would NOT achieve that — both runners expand
# ALL_GROUPS when given no selector, so a group is by definition part of the full board;
# and a group left out of ALL_GROUPS would vanish from `--list` and from the grouped run
# output, which is worse than being honestly ungrouped.
#
# They ARE listed by `--list` (list_groups prints this table), because a gate nobody can
# find is a gate nobody runs. Opt-in, not hidden.
OPT_IN_TESTS="real_cue_activation real_qwen_greedy_parity real_fish_s2_batch_parity real_qwen_serve real_qwen_eval real_qwen_seed real_qwen_affect real_emotion2vec real_cue_activation_qwen"

optin_why() {
  case "$1" in
    real_cue_activation)
      echo "C4.2 cue-activation certification — full render sweep + Whisper scoring" ;;
    real_qwen_greedy_parity)
      echo "Qwen3-TTS multi-frame greedy AR loop vs reference — ~21 min, 1.7B CPU/f32" ;;
    real_cue_activation_qwen)
      echo "C4.2' certification on the Qwen shipping path (ADR-0003) — ~26 min GPU, set SYRINX_C42_OUT" ;;
    real_emotion2vec)
      echo "emotion2vec+ judge vs its funasr reference — needs --features affect + SYRINX_EMOTION2VEC_{ONNX,REF}" ;;
    real_qwen_affect)
      echo "affect judge (SER) vs its python reference + cue direction report — needs --features affect" ;;
    real_qwen_seed)
      echo "per-render seed control: reproduces, differs, unclobberable — ~6 min, 0.6B CPU/f32" ;;
    real_qwen_eval)
      echo "syrinx-eval Qwen hookup: native WER + RTF on real weights — ~37 min, 1.7B CPU/f32" ;;
    real_fish_s2_batch_parity)
      echo "Fish s2 batched vs single prefill (SYRINX_FISH_BATCH_PARITY=1) — ~19 GB CPU/f32" ;;
    real_qwen_serve)
      echo "Qwen3-TTS over /v1/audio/speech, three real renders + WER — ~12 min, 0.6B CPU/f32" ;;
    *) echo "" ;;
  esac
}

# ---- pseudo-tests (a board row that is NOT a repo-root tests/<name>.rs) ------
#
# A row on the board is normally one repo-root integration binary, run as
# `cargo test --test <name>`. That shape is exactly why 133 passing member-crate
# unit tests were invisible to every board: `--test <name>` reaches integration
# targets only, so no crate's #[cfg(test)] module has ever been run by
# test-all.sh or verify.sh — not syrinx-qwen's 108, not syrinx-fish's 13, not
# syrinx-cue's 8, not syrinx-stt's 4.
#
# `crate_unit_tests` closes that. It is a row whose cargo invocation is
# `--workspace --lib` instead of `--test <name>`. Rather than teaching each
# runner about it (two places to drift, which is the bug this file exists to
# prevent), the four things a runner needs to know about ANY row are answered
# here, once, and both runners' run_one only calls these:
#
#   test_present    <name>            -> is it runnable, or MISSING?
#   test_cargo_args <name>            -> the cargo argv, one word per line
#   test_can_skip   <name>            -> does a SKIP marker mean the ROW skipped?
#   test_detail     <name> <logfile>  -> an annotation for the board line, or ""
#
# `test_can_skip` is the subtle one. SKIP means "this row's prerequisite is not
# configured" — weights, a parity fixture. `crate_unit_tests` has no
# prerequisite: it always runs, so it is PASS or FAIL and never SKIP. Individual
# unit tests inside it DO self-skip (on this box exactly one does:
# syrinx-qwen's Mimi `matches_the_python_reference`, gated on
# SYRINX_QWEN_ENCODER_REF), and the plain SKIP grep would have reported the whole
# 133-test row as SKIP on account of that one line — a strictly worse board than
# no row at all. So the row keeps its own verdict and reports the self-skips as a
# COUNT in its detail annotation instead of hiding behind them.
#
# The row is model-free in the sense that matters: with no env set every
# weight-backed unit test self-skips and it still passes. When test-all.env IS
# sourced it opportunistically gains coverage (syrinx-qwen's tokenizer goldens and
# real_checkpoint_parity run against the real checkpoints). Either way it is
# PASS/FAIL, which is why it belongs in FAMILY_free.
#
# Cost, measured on NovaBox 2026-09-05 (--features real --release, warm build,
# 32 cores): 13.6 s with no env / 14.5 s with scripts/test-all.env sourced. Of
# that, syrinx-fish's 13 tests are ~11.9 s and everything else is ~2 s.

PSEUDO_TESTS="crate_unit_tests"

is_pseudo_test() { case " $PSEUDO_TESTS " in *" $1 "*) return 0 ;; esac; return 1; }

# The cargo argv for one row, ONE WORD PER LINE so the caller can read it into an
# array without word-splitting surprises.
test_cargo_args() {
  case "$1" in
    real_qwen_affect) printf '%s\n' --features affect --test real_qwen_affect ;;
    # --no-fail-fast because this row is 13 separate binaries: without it cargo
    # stops at the first crate that fails and the crates after it are never run,
    # so the board would say "128 passed in 3 crates" and quietly omit the rest.
    # It does NOT soften the verdict — cargo still exits non-zero.
    crate_unit_tests) printf '%s\n' --workspace --lib --no-fail-fast ;;
    *)                printf '%s\n' --test "$1" ;;
  esac
}

# MISSING is "the test file has not been written yet". A pseudo-test has no file.
test_present() {
  is_pseudo_test "$1" && return 0
  [ -f "$SYRINX_ROOT/tests/$1.rs" ]
}

# Whether a `SKIP `/`skipping ` marker in this row's output makes the ROW a SKIP.
test_can_skip() { ! is_pseudo_test "$1"; }

# A short annotation appended to the board line, or "" for rows that need none.
# For crate_unit_tests the counts are the whole point: a row that just says PASS
# cannot distinguish 133 tests from 0, and a --workspace --lib run that silently
# stopped compiling one crate's tests would look identical.
test_detail() {
  local t="$1" log="$2"
  case "$t" in
    crate_unit_tests)
      [ -f "$log" ] || return 0
      local sk; sk="$(grep -cE 'SKIP |skipping ' "$log")"
      awk -v sk="$sk" '
        /^test result:/ { p += $4; f += $6; g += $8; if ($4 + $6 + $8 > 0) c++ }
        END { printf "%d passed, %d failed, %d self-skipped, %d ignored in %d crates", p, f, sk, g, c }
      ' "$log"
      ;;
    *) : ;;
  esac
}

# ---- family -> groups --------------------------------------------------------
# One model family per line, plus meta-families for the two cuts that matter most
# in practice: what runs anywhere, and what needs the box.

FAMILY_fish="fish_s1 fish_s2"
FAMILY_qwen3="qwen qwen_ckpt"
FAMILY_cosyvoice2="cv2 cv2e2e"
FAMILY_cosyvoice3="cv3 cv3e2e"
FAMILY_cosyvoice="cv2 cv2e2e cv3 cv3e2e"
FAMILY_whisper="stt"

# Everything that needs no weights, no fixtures and no GPU. These must never SKIP:
# a SKIP here is a bug in the runner or the test, not a partial box.
FAMILY_free="unit modelfree cue qwen"
# The complement: everything gated on weights or parity fixtures.
FAMILY_weights="cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt qwen_ckpt"
FAMILY_all="$ALL_GROUPS"

ALL_FAMILIES="fish qwen3 cosyvoice cosyvoice2 cosyvoice3 whisper free weights all"

# ---- accessors ---------------------------------------------------------------

group_tests()  { local v="GROUP_$1";  echo "${!v:-}"; }
family_groups() { local v="FAMILY_$1"; echo "${!v:-}"; }

is_group()  { [ -n "$(group_tests  "$1")" ]; }
is_family() { [ -n "$(family_groups "$1")" ]; }
# A selector names a test if there is a repo-root file for it OR it is a
# pseudo-test (see below), so `--test crate_unit_tests` resolves like any other.
is_test()   { test_present "$1"; }

list_optin() {
  for t in $OPT_IN_TESTS; do printf '  %-24s %s\n' "$t" "$(optin_why "$t")"; done
}

# Prints the groups, then the opt-in tests that deliberately belong to none of them.
# Both runners' `--list` calls this one function, so the second table cannot drift.
list_groups() {
  for g in $ALL_GROUPS; do printf '  %-12s %s\n' "$g" "$(group_tests "$g")"; done
  echo "opt-in tests (in no group — run with: --test <name>):"
  list_optin
}

list_families() {
  for f in $ALL_FAMILIES; do printf '  %-12s %s\n' "$f" "$(family_groups "$f")"; done
}

# Resolve one selector to a list of test names on stdout, or fail with a message
# on stderr and a non-zero status. Resolution order is group, then family, then a
# bare test name — the three namespaces do not overlap.
#
# Unknown selectors are a HARD ERROR by design. The previous runner treated an
# unmatched --group as "run nothing" and then printed "all configured groups
# green" with exit 0, so a typo produced a passing board that had tested nothing.
resolve_selector() {
  local sel="$1"
  if is_group "$sel"; then
    group_tests "$sel"; return 0
  fi
  if is_family "$sel"; then
    local out=""
    for g in $(family_groups "$sel"); do out="$out $(group_tests "$g")"; done
    echo "$out"; return 0
  fi
  if is_test "$sel"; then
    echo "$sel"; return 0
  fi
  {
    echo "unknown selector '$sel'."
    echo "  groups:   $ALL_GROUPS"
    echo "  families: $ALL_FAMILIES"
    echo "  or any repo-root test name (see tests/*.rs)"
  } >&2
  return 1
}

# De-duplicate a test list while preserving first-seen order, so overlapping
# selectors (--family fish --group fish_s2) never run a test twice.
dedupe() {
  local seen=" " out="" t
  for t in $1; do
    case "$seen" in *" $t "*) continue ;; esac
    seen="$seen$t "; out="$out $t"
  done
  echo "$out"
}
