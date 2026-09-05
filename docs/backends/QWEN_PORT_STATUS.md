# `syrinx-qwen` — port status and verification boundary

**Crate:** `crates/syrinx-qwen/` (workspace member; `default = ["real"]`, so a plain
`cargo build -p syrinx-qwen` builds the Candle path).
**Upstream:** Alibaba Qwen3-TTS, five published checkpoints, **Apache-2.0** — which is the
reason this port exists next to `syrinx-fish`, whose weights are non-commercial
(`crates/syrinx-qwen/src/lib.rs:1-8`).
**Landed in:** `73c7d6e` (`feat(fish,qwen): … the Qwen3-TTS port`), ~8 400 lines of Rust.

This file states, per component, **what is implemented**, **what has actually been
verified and how**, and **what each unverified item still needs** — weights, CPU, or GPU.
That boundary is the point of the document: everything above the line is gateable off-box
and now runs on the board; everything below it cannot be turned green without hardware,
and must not be faked into looking green.

Last audited 2026-09-03; amended 2026-09-05 (the `unit` board row, and the 0.6B-Base
speaker anchor — see the last two sections).

---

## 1. What is on the board today

`GROUP_qwen` in `scripts/test-groups.sh`, shared by `scripts/test-all.sh` and
`scripts/verify.sh` (also in `verify.sh --quick`):

| repo-root test | what it certifies |
|----------------|-------------------|
| `tests/qwen_config_contract.rs` (14 tests) | `Qwen3TtsConfig::from_json` against all five **real published** `config.json` files: geometry, derived widths, `code_predictor_heads`, every silent default, every rejection path |
| `tests/qwen_tensor_manifest.rs` (8 tests) | `load::expected_tensors` against the real **safetensors headers** of all five checkpoints: 402/404 tensors, zero missing, zero shape mismatches, the only unaccounted tensors being the 76 `speaker_encoder.*` of the two `-Base` checkpoints |
| `tests/qwen_sampling_contract.rs` (19 tests) | the pure-Rust sampling stack: HF warper order, the `<` / `<=` boundary conventions, the repetition-penalty sign rule, the suppress/min-new-tokens masks, SplitMix64 |

Fixtures: `tests/golden/qwen/config/` (published `config.json`, byte-for-byte) and
`tests/golden/qwen/index/` (`{tensor name: {shape, dtype}}` from the safetensors header —
**no weight data**). Regenerate with `python3 scripts/gen-qwen-index.py`; see
`tests/golden/qwen/README.md`.

These are model-free: no weights, no Candle, no GPU, no Python. They must never SKIP.

> **The crate's own unit tests are ON the board since 2026-09-05** — 108 of them for
> `syrinx-qwen`, 133 workspace-wide. They used to be invisible: `scripts/test-all.sh`
> invoked `cargo test --test <name>`, which only ever runs repo-root integration binaries,
> so no member crate's `#[cfg(test)]` module was exercised by `verify.sh`, for
> `syrinx-qwen` or for any other crate. The `unit` group now carries one row,
> `crate_unit_tests`, that runs `cargo test --workspace --lib`. See §5 gap 2.

---

## 2. Component inventory

### 2.1 Implemented and verified off-box (no weights, no GPU)

| component | where | evidence |
|-----------|-------|----------|
| Model geometry parser | `src/config.rs:120` (`from_json`), `:182` (`code_predictor_heads`) | `tests/qwen_config_contract.rs`, against the five published files |
| Loader tensor manifest | `src/load.rs:52` (`expected_tensors`) | `tests/qwen_tensor_manifest.rs`, against the five published safetensors headers |
| Stack split | `src/load.rs:206` (`CODE_PREDICTOR_PREFIX`), `:213` (`split_stacks`) | `tests/qwen_tensor_manifest.rs::the_prefix_splits_the_manifest_into_two_disjoint_stacks` |
| Sampling / warpers / PRNG | `src/sampling.rs` | `tests/qwen_sampling_contract.rs` |
| Transformer primitives | `src/nn.rs` (RMSNorm `:73`, RoPE `:95`/`:134`, GQA attention `:250`, `KvCache` `:173`, SwiGLU `:299`) | 6 in-crate tests, synthetic tensors (§5 gap: not on the board) |
| RVQ decode | `src/codec/rvq.rs:38` (`load`), `:64` (`decode`) | 3 in-crate tests on a hand-computed 2-layer fixture; the EMA quotient `embed_sum / cluster_usage.clamp(1e-5)` is pinned |
| Prompt assembly (3 modes) | `src/prompt.rs:521` / `:548` / `:570`, `PromptConfig::from_json:204` | 13 in-crate tests; the module docs record a position-by-position diff of 13 cases against the reference `generate()` run as an unbound function with marker embeddings (no weights, no GPU) |
| Dual-AR driver loop | `src/model.rs:260` (`generate`), `:328` (`predict_frame`), `:372` (`predictor_prefill`), `:393` (`realize_plan`), `:485` (`suppressed_ids`) | 17 in-crate tests on a synthetic checkpoint |
| Codec decode stack | `src/codec/decoder.rs:517`/`:534`/`:563`/`:589` | 13 in-crate tests; `matches_the_pytorch_reference` agrees with a rebuilt PyTorch model stage-for-stage to 2e-7 (CPU, f32, name-seeded weights on both sides) |
| Speaker encoder (ECAPA-TDNN) + log-mel front end | `src/speaker.rs:173` (`tensor_manifest`), `:296` (`log_mel_spectrogram`), `:538` (`forward`), `:559` (`embed`) | 13 in-crate tests; the manifest reproduces all 76 checkpoint tensors, and the front end matches a torch dump on a reduced-width fixture |
| Codec encode stack (Mimi) | `src/codec/encoder.rs:186` (config), `:710` (`encode`), `:720` (`encode_with`) | 14 in-crate tests; chunked == one-shot |

### 2.2 Implemented, verified only **with the real weights on CPU** (no GPU needed)

These are the tests that already exist and already pass on this box, but only when their
env var points at a checkpoint. None of them is on the board, because each SKIPs without
the weights and `GROUP_qwen` is defined as never-SKIP.

| what | where | gate | status on this box |
|------|-------|------|--------------------|
| Text tokenizer: golden ids vs `AutoTokenizer` | `src/tokenizer.rs:408`, `:416`, `:431`, `:445`, `:471`, `:486` | `SYRINX_QWEN_DIR`, else `~/models/Qwen3-TTS-12Hz-*` | **passing** (`cargo test -p syrinx-qwen`) — 6 golden id sequences, round trips, the two pre-tokenizer regexes |
| Speaker encoder vs the reference on real weights | `src/speaker.rs` (`real_checkpoint_parity`, in-crate) | `SYRINX_QWEN_BASE_DIR` | superseded on the board by `tests/real_qwen_speaker_parity.rs`, which since 2026-09-05 covers **both** `-Base` widths (see the parity table below). Still useful as a hand-copied-numbers cross-check, and it now runs on the board too, inside the `crate_unit_tests` row |
| Mimi encoder vs HuggingFace `MimiModel` | `src/codec/encoder.rs:1239` (`matches_the_python_reference`) | `SYRINX_QWEN_TOKENIZER_DIR` + `SYRINX_QWEN_ENCODER_REF` | SKIPs (env unset). The module header records an earlier manual run: 16×5 and 16×32 codes identical, latent max-abs 1.2e-4 / 2.5e-4 |
| Whole-checkpoint load + shape verify | `src/load.rs:137` (`verify_checkpoint`), `examples/verify.rs` | needs `model.safetensors` (1.8–3.9 GB) | never run in CI; `tests/qwen_tensor_manifest.rs` now proves the *manifest* it checks against is right |
| Weight materialisation | `src/load.rs:182` (`load_tensors`), `examples/loadcheck.rs` | needs the weights | never automated |

**How to promote these to the board:** they need an env-gated group that is *allowed* to
SKIP (the `fish_s1` / `stt` pattern), plus repo-root test files, plus `SYRINX_QWEN_*`
entries in `scripts/test-all.env.example`. Deliberately not done here: `GROUP_qwen` was
scoped to the never-SKIP half, and inventing a second group whose tests cannot be run in
this pass would put unverified rows on the board.

### 2.3 Not verified at all — needs weights, and in most cases a GPU

| what | why it is unverified | what it needs |
|------|----------------------|---------------|
| End-to-end synthesis (`examples/synth.rs`) | never executed | talker (1.8/3.9 GB) + tokenizer (682 MB) checkpoints; CPU/f32 will work but is slow, CUDA/bf16 is the intended path |
| Numerical parity of the **talker + code predictor** against `Qwen3TTSModel` | there is no Python reference dump for this crate at all — nothing like `scripts/gen-fish-ref.py` exists for Qwen | a reference-dump script + `~/.venvs/qwen`; then a `real_qwen_*_parity.rs` |
| Audio quality / intelligibility of the port's output | perceptual + WER; blocked-on-human per `CLAUDE.md` | rendered audio + the `syrinx-stt` WER oracle |
| ~~The **voice-clone path** end to end (`-Base`)~~ | **CLOSED 2026-09-03.** `crates/syrinx-qwen/examples/clone.rs` drives it: WAV → `speaker::resample` → `SpeakerEncoder::embed` → (`MimiEncoder::encode` for ICL) → `build_voice_clone` → `realize_plan` → generate → codec → WAV. Both reference modes render at WER 0.000 on CPU (renders under `renders/2026-09-03-qwen-base-clone/`) | — |
| Any CUDA execution | nothing in this crate has ever run on a GPU | a Blackwell-prepared box (`scripts/setup-cuda-blackwell.sh`) |
| Any bf16 execution | `Qwen3Tts::load` (`src/model.rs:164`) picks bf16 on CUDA; only the f32 CPU path has been exercised | GPU |
| Memory figures in the doc comments (283 MB encode chunk, ~700 MB wave chunk, 3.8 GiB one-shot) | derived from candle's `im2col` sizing, never measured | a run with RSS instrumentation |

### 2.4 Explicit `// PARITY:` markers (the crate's own "unconfirmed off-box" flags)

Only three, all narrow:

- `src/sampling.rs:266` — `multinomial` samples the same categorical distribution as
  `torch.multinomial`, but not the same *stream*. Unfixable by design; needs a
  distribution-level check against the reference, not a bit-exact one.
- `src/codec/encoder.rs:355` — `MimiEuclideanCodebook.quantize` casts to f32 before
  `cdist`. Confirmed by the (env-gated) encoder reference test when it runs.
- `src/codec/decoder.rs:688` — the mean/variance reduction runs in f32 for bf16 stability;
  needs the GPU/bf16 path to confirm the normalised output still matches.

No `TODO`, `FIXME`, `unimplemented!()` or `todo!()` anywhere in `src/` or `examples/`.
There are no stubs: every module implements the real algorithm.

---

## 3. Two mutants the new tests cannot kill (stated, not hidden)

Mutation-checking `tests/qwen_sampling_contract.rs` against hand-applied operator flips in
`crates/syrinx-qwen/src/sampling.rs` killed the flips on `top_k`'s `<`, `top_p`'s `<=`,
`top_p`'s `>= 1.0` guard, its `0..n-1` bound, `block_ids`' `i < len`, and the
temperature/nucleus ordering. Two survive, and both are **equivalent mutants**:

- `sampling.rs:136` `if l < 0.0` → `if l <= 0.0`. The branches differ only at `l == 0.0`,
  where `0.0 / p == 0.0 * p == 0.0`. No input distinguishes them.
- `sampling.rs:278` `if u < acc` → `if u <= acc`. Differs only when the uniform draw lands
  exactly on a cumulative boundary, which requires `next_f64() == 0.0` — reachable only if
  the top 53 bits of a SplitMix64 output are all zero.

Writing a test that "kills" either would mean asserting something the code does not
promise. They are documented instead.

---

## 4. Exact commands for the model box

None of these were run in the pass that produced this file (the GPU and all heavy runs are
serialised by the main session). Run them one at a time, never two `real`-feature suites
concurrently.

**1. Compile gate — the new tests under the feature set the board actually uses.**
The repo-root Qwen tests are written against the model-free half of the crate and were
verified with `cargo test --no-default-features`; the board runs every group with
`--features real --release`, so confirm they compile and pass there too:

```bash
cargo test --features real --release --test qwen_config_contract  -- --nocapture
cargo test --features real --release --test qwen_tensor_manifest  -- --nocapture
cargo test --features real --release --test qwen_sampling_contract -- --nocapture
# or the whole group at once:
./scripts/test-all.sh --group qwen
```

Expected: `PASS` on all three, 41 assertions-bearing tests, no SKIP. (No GPU, no weights,
cheap — but it does pull in the full `real` dependency graph, which is why it is here.)

**2. The crate's own 108 unit tests** (CPU, ~2 s, no GPU) — or the whole workspace's 133,
which is exactly what the board's `unit` group runs:

```bash
cargo test -p syrinx-qwen
./scripts/test-all.sh unit     # cargo test --workspace --lib, 13.6 s warm
```

**3. Un-SKIP the weight-backed checkpoints** (CPU only; reads 1.8–3.9 GB from disk):

```bash
export SYRINX_QWEN_DIR=~/models/Qwen3-TTS-12Hz-0.6B-Base            # text tokenizer goldens
export SYRINX_QWEN_BASE_DIR=~/models/Qwen3-TTS-12Hz-0.6B-Base       # speaker-encoder parity
export SYRINX_QWEN_TOKENIZER_DIR=~/models/Qwen3-TTS-Tokenizer-12Hz  # Mimi encoder parity
export SYRINX_QWEN_ENCODER_REF=...                                  # the Python dump dir
cargo test -p syrinx-qwen
```

**4. Loader verification against the real safetensors** (CPU, reads every checkpoint):

```bash
MEMMAX=24G scripts/run-isolated.sh \
  cargo run -p syrinx-qwen --example verify -- ~/models/Qwen3-TTS-12Hz-*
```

Expected, from `tests/qwen_tensor_manifest.rs`: `checked 402` (0.6B) / `404` (1.7B),
`missing 0`, `mismatched 0`, `unaccounted 76` on the two `-Base` (all `speaker_encoder.*`)
and `0` on the other three. Anything else is a real finding.

**5. First end-to-end synthesis — the step nothing has ever done.** CPU first (parity
dtype), then CUDA:

```bash
MEMMAX=24G scripts/run-isolated.sh \
  cargo run -p syrinx-qwen --release --example synth -- \
    ~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice ~/models/Qwen3-TTS-Tokenizer-12Hz \
    /tmp/qwen-cpu.wav "The quick brown fox jumps over the lazy dog." serena english

SYRINX_QWEN_DEVICE=0 cargo run -p syrinx-qwen --release --features cuda --example synth -- \
    ~/models/Qwen3-TTS-12Hz-0.6B-CustomVoice ~/models/Qwen3-TTS-Tokenizer-12Hz \
    /tmp/qwen-cuda.wav "The quick brown fox jumps over the lazy dog." serena english
```

Both are unverified paths. The most likely first failures, in order: the codec decode
stack in bf16 (`src/codec/decoder.rs:688`), the talker prompt's speaker slot width, and
peak memory during the wave stage (`SYRINX_QWEN_CODEC_CHUNK_FRAMES` bounds it).

---

## 5. Open gaps, in priority order

1. ~~**No repo-root, env-gated group for the weight-backed tests.**~~ **CLOSED
   2026-09-03.** `GROUP_qwen_ckpt` exists in `scripts/test-groups.sh` with five
   `tests/real_qwen_*.rs` members (prompt, stack, encode, speaker, greedy) and the
   `SYRINX_QWEN_*` paths in `test-all.env`. It self-skips off-box; the `qwen3` family
   selector runs it together with the model-free `GROUP_qwen`.
2. ~~**No member-crate unit tests reach the board, for any crate.**~~ **CLOSED
   2026-09-05.** `GROUP_unit="crate_unit_tests"` in `scripts/test-groups.sh` is a
   **pseudo-test**: a board row whose cargo argv is `--workspace --lib --no-fail-fast`
   instead of `--test <name>`. Both runners were changed to ask `test-groups.sh` what a
   row *is* — `test_present` / `test_cargo_args` / `test_can_skip` / `test_detail` — so the
   row is defined once and neither runner special-cases it. 133 unit tests now report on
   every board (108 `syrinx-qwen`, 13 `syrinx-fish`, 8 `syrinx-cue`, 4 `syrinx-stt`), in
   13.6 s warm / 14.5 s with `test-all.env` sourced, and the row prints its counts:
   `PASS (133 passed, 0 failed, 1 self-skipped, 0 ignored in 4 crates)`.
   Two decisions worth keeping:
   - The row can never be **SKIP**. SKIP means "this row's prerequisite is unconfigured",
     and a workspace-lib run has none. Individual unit tests inside it do self-skip — on
     this box exactly one, the Mimi `matches_the_python_reference` — and the runners'
     generic `SKIP ` grep would otherwise have painted all 133 tests yellow because of that
     one line. The self-skips are reported as a count instead.
   - `--no-fail-fast`, because the row is 13 binaries: without it cargo stops at the first
     failing crate and the board says "128 passed in 3 crates", silently omitting the rest.
     It does not soften the verdict — cargo still exits non-zero and the row goes FAIL.
   Verified red-on-real-failure, not assumed: flipping `speaker.rs:1034`'s
   `assert_eq!(cfg.enc_dim, 1024)` to `1025` gave
   `crate_unit_tests FAIL (132 passed, 1 failed, 1 self-skipped, 0 ignored in 4 crates)`
   with `PASS 0 … FAIL 1` and exit 1 from `test-all.sh`, and `VERIFICATION FAILED` / exit 1
   from `verify.sh`. Restored afterwards.
3. ~~**No Python reference dump for Qwen.**~~ **CLOSED 2026-09-03.** Four generators now
   exist — `gen-qwen-ref.py` (prompt, talker, predictor, codec decode) plus
   `gen-qwen-ref-encoder.py`, `gen-qwen-ref-speaker.py` and `gen-qwen-ref-greedy.py`.
   All capture the reference's own tensors and refuse to run if `qwen_tts` cannot be
   imported. This was the gap that let the `text_projection` `silu` bug ship.
4. ~~**The voice-clone path has no driver.**~~ **CLOSED 2026-09-03** by
   `examples/clone.rs` plus `tests/real_qwen_speaker_parity.rs`. See §2.3.
5. **`suppressed_ids`, `PromptConfig`, `MimiEncoderConfig::from_json` and
   `DecoderConfig::from_json` are all pure functions of a config file but sit behind the
   `real` feature gate** (`src/lib.rs:37-53`), because their modules import Candle. Moving
   the parsers to a Candle-free module would let them join `GROUP_qwen` as never-SKIP
   tests. Worth doing; not done here, because it is a production refactor and this pass
   was scoped to getting the crate honestly onto the board.
6. ~~**Not wired into anything.**~~ **CLOSED 2026-09-05.** `syrinx qwen` renders through
   the port from the CLI, and `syrinx-serve` gained a Qwen backend (`src/qwen.rs` planning
   layer + `src/synth_qwen.rs` Candle engine) reachable over `/v1/audio/speech`. Both route
   cues through `syrinx-cue` rather than parsing any syntax themselves, and both respect
   the per-checkpoint capability rows: Base drops every cue, 0.6B-CustomVoice reports
   accepted-and-discarded, 1.7B-CustomVoice and VoiceDesign honour the instruction, and
   VoiceDesign refuses to split (its instruct describes the VOICE, so a mid-line split
   would change who is speaking). Conflicting cues on an instructable checkpoint become N
   requests, rendered separately and concatenated with the server's crossfade — verified
   end to end: `[happy] ... [sad] ...` produced 2 segments transcribing as
   "What a wonderful morning, but then the letter arrived."
   Still unexecuted: `synth_qwen.rs` is compile-verified but has never loaded a checkpoint
   (needs a `real_qwen_serve_*` test on the box), and there is no `syrinx-eval` hookup.


## First end-to-end renders — 2026-09-03

The port synthesizes. `renders/2026-09-03-qwen-first/FINDINGS.md` has the full table;
the short version:

- **Plain synthesis works on both sizes**, WER 0.000 against the native Whisper oracle
  (0.6B-CustomVoice and 1.7B-CustomVoice, `Come closer, I have something to tell you.`).
  ~1.3 s talker load, ~4 s for a short utterance on the 1.7B at bf16 on one RTX 5070.
- **The per-checkpoint capability rows are confirmed against the running model.** The
  0.6B-CustomVoice plain and cued renders are bit-identical (md5
  `24f7866572c0082b3c2ba123d0feffcf`), so its `instruct = accepted` row — accepts the
  instruction and silently discards it — is proven rather than asserted.
- **DEFECT: any instruct makes the 1.7B talker repeat the target text**, two to five
  times, scaling with frame count. The plain path is clean, so it is specific to the
  instruct block. Sampler defaults (already the published generation_config), prompt-side
  text duplication (+7 steps, exactly the instruct block) and ASR hallucination are all
  ruled out in FINDINGS.md §3. The open question is whether `assemble_text_mode` places
  the instruct turn where the reference does — which is precisely what the missing Qwen
  reference dump would answer.
- ~~`-Base` still cannot render at all~~ — **closed 2026-09-03**: `examples/clone.rs`
  builds the x-vector from a WAV, and both reference clone modes render at WER 0.000
  through the Whisper oracle (`renders/2026-09-03-qwen-base-clone/`, CPU/f32, seed 0):
  x-vector-only 32 frames / 2.56 s from a 10-step prompt; in-context 37 frames / 2.96 s
  from a 135-step prompt carrying 125 reference frames, with the reference's own
  decode-`cat(ref_code, generated)`-then-cut-`125/162` behaviour reproduced.

## Parity status — 2026-09-03 (after the reference landed)

The port is now anchored to the reference at **every stage of the chain**, which it was
not when the `silu` bug shipped. `scripts/gen-qwen-ref.py` dumps the anchors from the
reference's own modules; `tests/real_qwen_prompt_parity.rs` and
`tests/real_qwen_stack_parity.rs` check them, in `GROUP_qwen_ckpt` / the `qwen3` family.

| stage | anchor | measured |
|---|---|---|
| prompt | `prompt.{plain,instruct}.inputs_embeds` | 0.00000 |
| talker | `talker.prefill_logits` | 0.00003 |
| code predictor | `predictor.logits` (fed the reference's own input) | 0.00003 |
| codec (decode) | `codec.wav`, `codec_edge.wav` | 0.000022 / 0.000005 |
| codec (encode) | `{full,ragged}.{wav,latent,codes}` from `scripts/gen-qwen-ref-encoder.py`, checked by `tests/real_qwen_encode_parity.rs` | **codes bit-exact** (2000 + 512, zero differences); latent 3.206e-4 / 2.351e-4 against max abs 37.492 |
| speaker encoder (`-Base` x-vector), **1.7B** | `mel`, `xvector` from `scripts/gen-qwen-ref-speaker.py`, checked by `tests/real_qwen_speaker_parity.rs` | mel 0.00025; **x-vector 0.0000010** per component (2048 wide, L2 norm 17.028715 on both sides), from the reference's own mel and end to end alike |
| speaker encoder (`-Base` x-vector), **0.6B** | the same generator + gate, second fixture (`SYRINX_QWEN_REF_SPEAKER_0_6B`) | mel 0.00025 (bit-identical fixture, same clip); **x-vector 0.0000006** per component (1024 wide, L2 norm 10.409607 vs 10.409612), from the reference's own mel and end to end alike |

Fixtures are generated on **CPU/float32** deliberately: the reference decoding identical
codes on CUDA vs CPU disagrees with itself by 0.031 on a [-1,1] waveform, which would
consume the whole error budget. See renders/2026-09-03-qwen-first/FINDINGS.md §5.

The **encode** anchor is exact rather than tolerance-bounded, because a code is an index:
`tests/real_qwen_encode_parity.rs` asserts equality on all 16 x 125 and 16 x 32 codes, for
the one-shot cascade and for every chunk length (1, 2, 7, 64, 128 steps) including the
shipped default that `encode()` actually takes. It stores and replays the reference's own
post-resample `input_values`, so librosa's resampler is not smuggled into the measurement,
and it pins CPU rather than honouring `SYRINX_QWEN_DEVICE` — a device-level drift that
flips one `argmin` would turn an exact gate into a false alarm.

One correction to `renders/2026-09-03-qwen-first/FINDINGS.md` §5, which said the config's
`semantic_codebook_size: 4096` "belongs to the encoder path": it does not. Read from the
checkpoint header, the ENCODE-side tables are 2048 too —
`encoder.quantizer.{semantic,acoustic}_residual_vector_quantizer.layers.*.codebook.embed_sum`
are all `[2048, 256]`, matching the 2048 the decode side already showed. The 4096 matches
no table in the checkpoint; it is a dead field in `decoder_config`. The conclusion §5 drew
from it still stands (2047 is the true edge everywhere) — only the attribution was wrong.

### The generation loop, gated by greedy decoding

The anchors above cover **one step each**, which left the autoregressive loop itself
ungated: the talker's KV cache and position advance across frames, the code predictor's
per-frame reset and RoPE walk, the talker->predictor handoff, the 16-embedding feedback
sum, the trailing-text schedule, the `min_new_tokens` EOS guard and `suppress_tokens` were
exercised once or not at all, so a drift first appearing at frame 5 was invisible.

Greedy decoding closes it, and it is the only thing that can: sampling draws from two
different PRNGs, while `do_sample=False` on both heads makes the run a deterministic
function of `(weights, prompt)` on both sides. `scripts/gen-qwen-ref-greedy.py` dumps the
reference's greedy code matrix and `tests/real_qwen_greedy_parity.rs` compares it as
**integers, with no tolerance to loosen**. Both sides CPU/f32.

| case | prompt | trailing rows | frames | result |
|---|---|---|---|---|
| `plain` | 21 | 1 | 41 | 41/41 frames, 656/656 codes identical, EOS at the same frame |
| `instruct` | 28 | 1 | 44 | 44/44 frames, 704/704 codes identical, EOS at the same frame |
| `streaming` | 10 | 10 | 44 | 44/44 frames, 704/704 codes identical, EOS at the same frame |

2064 of 2064 codes agree exactly, `stopped_on_eos` agrees in all three, and no bug was
found. The realized prompts measure 1e-6 max abs against the reference's; the `streaming`
prompt is new coverage, since no other test builds a multi-row `trailing_text_hidden`.

Three details that keep this a real gate rather than a re-run of the sampler:

- **What greedy does NOT switch off.** `_get_logits_processor` installs the temperature /
  top-k / top-p warpers only under `if generation_config.do_sample:`; the repetition
  penalty, the min-new-tokens EOS guard and `suppress_tokens` sit above that line. So a
  greedy talker still runs `repetition_penalty = 1.05` while the code predictor runs `1.0`
  (from `code_predictor_config`, not from a flag) — the whole processor list stays in the
  path, asymmetry included.
- **Greedy is not `top_k = 1`.** `top_k = 1` keeps every tied maximum and lets the PRNG
  choose between them; `torch.argmax` always takes the first. `DriveParams::greedy` selects
  `syrinx_qwen::sampling::Sampler::greedy`, a real first-index argmax that applies no
  warper and consumes no randomness.
- **`streaming` earns its place.** In non-streaming mode the reference sets
  `trailing_text_hidden = tts_pad_embed` — one row, equal to the pad — so a loop with an
  off-by-one in the per-frame schedule emits byte-identical codes. Only
  `non_streaming_mode=False` (10 rows consumed over 44 frames) can see it.

Because the gate came out green on the first run, its teeth were measured rather than
assumed, with two negative controls on the `streaming` case (run once, then deleted):

| injected defect | caught at |
|---|---|
| control (no defect) | — identical for all frames, so the comparison is well-formed |
| trailing schedule shifted one row (`trailing_text_hidden[step + 1]`) | frame 1, group 5 (got 964, reference 1498) |
| talker `repetition_penalty` dropped to 1.0 | frame 13, group 0 (got 1667, reference 1107) |

The second is the reason the fixture runs the **whole utterance** and not a handful of
frames: group-0 code 342 first repeats at frame 7, and the penalty does not flip an argmax
until frame 13. Truncated to 12 frames the same gate sees nothing at all.

Still not gated: end-to-end generated **audio** under the shipped sampling settings. Greedy
is the substitute for it, and it deliberately leaves the three warpers and the multinomial
out of the path — `tests/qwen_sampling_contract.rs` gates those against HF's semantics
model-free, but nothing compares a *sampled* run to the reference, and nothing can.

The `-Base` clone path is now covered on both halves — the reference codes by
`real_qwen_encode_parity`, the x-vector by `real_qwen_speaker_parity` — and driven end to
end by `examples/clone.rs`. One deliberate hole remains inside it: **resampling**. The
reference reaches its encoder through `librosa.resample` (soxr `HQ`) and this workspace has
no soxr, so the fixture stores the reference's *post-resample* clip and every numeric anchor
starts from it. The driver's own `speaker::resample` is gated separately, and by effect
rather than by value: its x-vector must stay at cosine >= 0.9995 of the reference's
(measured 0.999977). That assertion has already earned its place — it caught the first
version of the resampler (Lanczos-16 at cutoff 1.0, copied from `syrinx_serve::wavio`) at
cosine 0.996886, which turned out to be real image leakage above the input Nyquist landing
in the top mel bands.

## Both `-Base` widths anchored — 2026-09-05

`renders/2026-09-03-qwen-base-clone/FINDINGS.md` §5 left the 0.6B-Base "unanchored against
a real clip": the speaker fixture was dumped from the 1.7B, and the two checkpoints differ
in exactly one thing, `speaker_encoder_config.enc_dim` (1024 vs 2048). That is a small
difference and precisely the kind a port can get wrong in one direction only, so it was a
real hole rather than a formality.

Closed. `scripts/gen-qwen-ref-speaker.py --ckpt …-0.6B-Base` dumped a second fixture
(`/home/floofy/parity-qwen/speaker-0.6b.safetensors`, CPU/float32, same
`voice_en_10s.wav`), and `tests/real_qwen_speaker_parity.rs` now iterates over **every
configured checkpoint** instead of the first one it finds.

| anchor | 1.7B-Base (2048) | 0.6B-Base (1024) | bound |
|---|---|---|---|
| mel `[937, 128]` | 0.0002508 | 0.0002508 | 5e-3 |
| x-vector from the reference's own mel | 0.0000010 | **0.0000006** | 1e-4 |
| x-vector end to end (`embed`) | 0.0000010 | **0.0000006** | 1e-4 |
| driver-path resample, cosine | 0.999977 | **0.999980** | >= 0.9995 |
| x-vector L2 norm | 17.028715 / 17.028715 | 10.409607 / 10.409612 | — |

**No bug.** `SpeakerEncoderConfig::from_model_config` already read `enc_dim` from
`config.json` rather than hardcoding it, and the 1024-wide stack agrees with the reference
as well as the 2048-wide one does. The tolerances were NOT touched: the 0.6B lands inside
the bounds the 1.7B set, with more margin, not less.

**The 6e-7 is the reference's own noise floor, and that was measured rather than
asserted.** Re-dumping the same 0.6B fixture from the same reference, same CPU, same f32,
changing only `OMP_NUM_THREADS` (32 -> 1) moves its x-vector by **4.8e-7**, while `wav24`
and `mel` come back bit-identical — so all of that drift is the encoder's own reduction
order. The port's disagreement is the same magnitude as the reference's disagreement with
itself; there is no residue left for a porting fault to hide in. (This is the same
discipline that made the Fish RoPE diagnosis trustworthy: establish what the reference does
to itself before attributing anything to the port.)

Two negative controls, run once and then reverted:

| injected defect | result |
|---|---|
| the `tdnn` ReLU dropped (`speaker.rs:589`) — the `text_projection` bug's exact shape | encoder / end-to-end / driver all FAIL on the first case at 556.68 and cosine 0.2678; the mel anchor stays green on both, so the localization claim still holds |
| the 0.6B **fixture** nudged by 2e-5 relative (1.5e-4 absolute) | the 1.7B case passes at 1.0e-6 and the **0.6B case fails** at 0.0001516 against the 1e-4 bound — i.e. the second checkpoint is genuinely asserted, not merely loaded |

### Configuration

One new variable, `SYRINX_QWEN_REF_SPEAKER_0_6B`, pointed at the second fixture. There is
deliberately no required second checkpoint-dir variable: `gen-qwen-ref-speaker.py` records
the absolute `--ckpt` in the fixture's safetensors `__metadata__`, and the test reads it
from there, so a fixture can never be silently paired with the wrong checkpoint.
`SYRINX_QWEN_BASE_DIR_0_6B` exists only as an override for a checkpoint that has moved
since the dump. A fixture whose checkpoint is reachable by neither route is a **hard
failure**, not a skip — a half-configured anchor is a hole. With no fixture variable set at
all the file SKIPs as before, and a box with only the 1.7B configured still reports PASS
(the unconfigured checkpoint prints a plain `not configured` note, deliberately without the
`SKIP ` token the runners grep for, so one missing fixture cannot paint a row that really
ran).

Still open on this path, unchanged: CUDA/bf16 has never been run (the CUDA tolerances in
the test are labelled slack, not evidence), clone *quality* is perceptual and
blocked-on-human, and resampling is bounded by effect rather than anchored tensor-for-tensor.
