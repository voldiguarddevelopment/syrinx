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
GROUP_cue="control_survey_gate claude_md_invariant_gate cue_token_alignment prosody_cue_overrides expressive_api cue_activation_gate cue_activation_measure"

# Qwen3-TTS port (syrinx-qwen). Model-FREE half only: the geometry contract, the
# loader's tensor manifest against the published safetensors HEADERS (checked in
# under tests/golden/qwen/, no weight data), and the pure-Rust sampling stack. The
# weight-backed half is NOT here — see docs/backends/QWEN_PORT_STATUS.md.
GROUP_qwen="qwen_config_contract qwen_tensor_manifest qwen_sampling_contract"

# Weight-backed Qwen tests. Kept OUT of GROUP_qwen deliberately: that group is model-free
# and must never SKIP, while these self-skip without a checkpoint and the reference dump
# from scripts/gen-qwen-ref.py.
GROUP_qwen_ckpt="real_qwen_prompt_parity real_qwen_stack_parity real_qwen_encode_parity real_qwen_speaker_parity real_qwen_greedy_parity"

ALL_GROUPS="modelfree cue qwen cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt qwen_ckpt"

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
FAMILY_free="modelfree cue qwen"
# The complement: everything gated on weights or parity fixtures.
FAMILY_weights="cv2 cv2e2e cv3 cv3e2e fish_s1 fish_s2 stt qwen_ckpt"
FAMILY_all="$ALL_GROUPS"

ALL_FAMILIES="fish qwen3 cosyvoice cosyvoice2 cosyvoice3 whisper free weights all"

# ---- accessors ---------------------------------------------------------------

group_tests()  { local v="GROUP_$1";  echo "${!v:-}"; }
family_groups() { local v="FAMILY_$1"; echo "${!v:-}"; }

is_group()  { [ -n "$(group_tests  "$1")" ]; }
is_family() { [ -n "$(family_groups "$1")" ]; }
is_test()   { [ -f "$SYRINX_ROOT/tests/$1.rs" ]; }

list_groups() {
  for g in $ALL_GROUPS; do printf '  %-12s %s\n' "$g" "$(group_tests "$g")"; done
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
