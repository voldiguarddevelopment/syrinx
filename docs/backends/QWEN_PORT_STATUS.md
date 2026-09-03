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

Last audited 2026-09-03.

---

## 1. What is on the board today

`GROUP_qwen` in `scripts/test-all.sh` and `scripts/verify.sh` (also in `verify.sh --quick`):

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

> **Also on disk but NOT on the board: 105 unit tests inside the crate**
> (`cargo test -p syrinx-qwen`, all passing). `scripts/test-all.sh` invokes
> `cargo test --test <name>`, which only ever runs repo-root integration binaries, so no
> member crate's `#[cfg(test)]` module is exercised by `verify.sh` — for `syrinx-qwen` or
> for any other crate. That is a harness-wide gap, not a Qwen one; it is recorded in §5.

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
| Speaker encoder vs the reference on real weights | `src/speaker.rs:934` (`real_checkpoint_parity`) | `SYRINX_QWEN_BASE_DIR` | SKIPs (env unset). The module header claims < 1e-4 on both `-Base` checkpoints from an earlier manual run |
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
| The **voice-clone path** end to end (`-Base`) | no example or CLI drives it. `realize_plan` (`src/model.rs:393`) takes the x-vector and the reference frames as *arguments*; nothing in-tree builds them from a WAV | a WAV reader → `speaker::SpeakerEncoder::embed` → `codec::encoder::MimiEncoder::encode` → `prompt::build_voice_clone(CloneRef::InContext)`. CPU-feasible; currently a missing integration, not a missing algorithm |
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

**2. The crate's own 105 unit tests** (CPU, ~8 s, no GPU):

```bash
cargo test -p syrinx-qwen
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

1. **No repo-root, env-gated group for the weight-backed tests.** The tokenizer goldens
   already pass on this box and are invisible to `verify.sh`. Wants `tests/real_qwen_*.rs`
   plus a SKIP-allowed `GROUP_qwen_ckpt` and `SYRINX_QWEN_*` in `test-all.env.example`.
2. **No member-crate unit tests reach the board, for any crate.** 105 passing Qwen tests
   (and every other crate's) are outside `verify.sh` because `run_one` calls
   `cargo test --test <name>`. A `cargo test --workspace --lib` row would close it.
3. **No Python reference dump for Qwen.** `scripts/gen-fish-ref.py` has no Qwen sibling,
   so there is no talker/code-predictor parity fixture and no way to gate one.
4. **The voice-clone path has no driver.** See §2.3; this is integration work, not
   research, and it is CPU-feasible.
5. **`suppressed_ids`, `PromptConfig`, `MimiEncoderConfig::from_json` and
   `DecoderConfig::from_json` are all pure functions of a config file but sit behind the
   `real` feature gate** (`src/lib.rs:37-53`), because their modules import Candle. Moving
   the parsers to a Candle-free module would let them join `GROUP_qwen` as never-SKIP
   tests. Worth doing; not done here, because it is a production refactor and this pass
   was scoped to getting the crate honestly onto the board.
6. **Not wired into anything.** `syrinx-qwen` has no `syrinx-cli` subcommand, no
   `syrinx-serve` backend, and no `syrinx-eval` hookup. `crates/syrinx-cue/caps.toml`
   already carries the five Qwen `[[backend]]` entries, so the cue layer knows about it;
   nothing else does. (`scripts/render-qwen.py` drives the **Python** reference, not this
   crate.)
