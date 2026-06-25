//! Real CosyVoice2 LM forward via Candle (the DESIGN T2.1 "port base weights to a
//! Rust tensor format" path — the real model behind the toy reference parity).
//!
//! Loads the base model's **Qwen2-0.5B** LM backbone plus CosyVoice2's `llm_decoder`
//! head (converted to fp32 safetensors offline — too large to vendor) and reproduces
//! the reference per-position logits. This is gated behind the `real` cargo feature
//! and a model path on disk; the parity test skips cleanly when the weights are absent
//! (mirroring the device-bound task recipe) and runs for real where they exist.
//!
//! Architecture (from the checkpoint manifest): 24 decoder layers, hidden 896,
//! GQA with 14 query heads / 2 KV heads (head_dim 64, q/k/v carry bias, o_proj does
//! not), SwiGLU MLP (intermediate 4864), RoPE θ=1e6, RMSNorm eps 1e-6. The CosyVoice2
//! head is `llm_decoder: Linear(896 -> 6564)`.

use candle_core::{safetensors, DType, Device, Result, Tensor, D};
use std::collections::HashMap;

const HIDDEN: usize = 896;
const N_LAYERS: usize = 24;
const N_HEADS: usize = 14;
const N_KV: usize = 2;
const HEAD_DIM: usize = 64;
const EPS: f64 = 1e-6;
const ROPE_THETA: f32 = 1_000_000.0;

/// The real Qwen2-0.5B LM + CosyVoice2 `llm_decoder`, loaded from fp32 safetensors.
pub struct Qwen2Lm {
    w: HashMap<String, Tensor>,
    dev: Device,
}

impl Qwen2Lm {
    /// Load the converted fp32 checkpoint (`llm_fp32.safetensors`) onto `dev`.
    pub fn load(path: &str, dev: Device) -> Result<Self> {
        let raw = safetensors::load(path, &dev)?;
        // Normalise to f32 so the forward is a clean fp32 reference match.
        let mut w = HashMap::with_capacity(raw.len());
        for (k, v) in raw {
            w.insert(k, v.to_dtype(DType::F32)?);
        }
        Ok(Self { w, dev })
    }

    fn g(&self, name: &str) -> Result<Tensor> {
        self.w
            .get(name)
            .ok_or_else(|| candle_core::Error::Msg(format!("missing weight: {name}")))
            .cloned()
    }

    /// `x * rsqrt(mean(x^2, -1) + eps) * weight`, computed in f32 (Qwen2 RMSNorm).
    fn rms_norm(&self, x: &Tensor, wname: &str) -> Result<Tensor> {
        let w = self.g(wname)?; // [HIDDEN]
        let var = x.sqr()?.mean_keepdim(D::Minus1)?; // [.., 1]
        let xn = x.broadcast_div(&(var + EPS)?.sqrt()?)?;
        xn.broadcast_mul(&w)
    }

    /// `x @ W^T (+ b)` for a `[.., in]` input and a `[out, in]` weight.
    fn linear(&self, x: &Tensor, wname: &str, bias: Option<&str>) -> Result<Tensor> {
        let w = self.g(wname)?;
        let y = x.broadcast_matmul(&w.t()?)?;
        match bias {
            Some(b) => y.broadcast_add(&self.g(b)?),
            None => Ok(y),
        }
    }

    fn attn(&self, x: &Tensor, layer: usize, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.self_attn");
        let (b, t, _) = x.dims3()?;
        let q = self.linear(x, &format!("{p}.q_proj.weight"), Some(&format!("{p}.q_proj.bias")))?;
        let k = self.linear(x, &format!("{p}.k_proj.weight"), Some(&format!("{p}.k_proj.bias")))?;
        let v = self.linear(x, &format!("{p}.v_proj.weight"), Some(&format!("{p}.v_proj.bias")))?;
        let q = q.reshape((b, t, N_HEADS, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,14,t,64]
        let k = k.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let v = v.reshape((b, t, N_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b,2,t,64]
        let q = rope(&q, cos, sin)?;
        let k = rope(&k, cos, sin)?;
        let k = repeat_kv(&k, N_HEADS / N_KV)?; // [b,14,t,64]
        let v = repeat_kv(&v, N_HEADS / N_KV)?;
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?; // [b,14,t,t]
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let ctx = probs.matmul(&v)?; // [b,14,t,64]
        let ctx = ctx.transpose(1, 2)?.contiguous()?.reshape((b, t, HIDDEN))?;
        self.linear(&ctx, &format!("{p}.o_proj.weight"), None)
    }

    fn mlp(&self, x: &Tensor, layer: usize) -> Result<Tensor> {
        let p = format!("llm.model.model.layers.{layer}.mlp");
        let gate = self.linear(x, &format!("{p}.gate_proj.weight"), None)?;
        let up = self.linear(x, &format!("{p}.up_proj.weight"), None)?;
        let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.linear(&act, &format!("{p}.down_proj.weight"), None)
    }

    /// Run the 24 decoder layers + final RMSNorm over an input embedding sequence
    /// `[b, t, 896]`, returning the last hidden state `[b, t, 896]`.
    pub fn forward_hidden(&self, embeds: &Tensor) -> Result<Tensor> {
        let (_b, t, _) = embeds.dims3()?;
        let (cos, sin) = rope_cos_sin(t, &self.dev)?;
        let mask = causal_mask(t, &self.dev)?;
        let mut h = embeds.clone();
        for l in 0..N_LAYERS {
            let pre = format!("llm.model.model.layers.{l}");
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.input_layernorm.weight"))?;
            h = (r + self.attn(&hn, l, &cos, &sin, &mask)?)?;
            let r = h.clone();
            let hn = self.rms_norm(&h, &format!("{pre}.post_attention_layernorm.weight"))?;
            h = (r + self.mlp(&hn, l)?)?;
        }
        self.rms_norm(&h, "llm.model.model.norm.weight")
    }

    /// Full LM forward: hidden state -> CosyVoice2 `llm_decoder` -> logits `[b, t, 6564]`.
    pub fn forward_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let h = self.forward_hidden(embeds)?;
        self.linear(&h, "llm_decoder.weight", Some("llm_decoder.bias"))
    }

    // ---------------------------------------------------------------------
    // Autoregressive speech-token generation (CosyVoice2 `Qwen2LM.inference`)
    // ---------------------------------------------------------------------

    /// Gather rows `ids` from a `[V, HIDDEN]` embedding table, returning `[1, n, HIDDEN]`.
    ///
    /// `ids` are u32 token ids; this is a plain row lookup (the `nn.Embedding` op).
    fn embed_rows(&self, table: &str, ids: &[u32]) -> Result<Tensor> {
        let w = self.g(table)?; // [V, HIDDEN]
        let idx = Tensor::from_vec(ids.to_vec(), (ids.len(),), &self.dev)?;
        let rows = w.index_select(&idx, 0)?; // [n, HIDDEN]
        rows.unsqueeze(0) // [1, n, HIDDEN]
    }

    /// The Qwen2 text embedding for `text_token` ids (`embed_tokens`), `[1, n, HIDDEN]`.
    pub fn text_embed(&self, text_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("llm.model.model.embed_tokens.weight", text_token)
    }

    /// One `llm_embedding` row (`sos`=0 / `task_id`=1), shaped `[1, 1, HIDDEN]`.
    fn llm_embed_row(&self, id: u32) -> Result<Tensor> {
        self.embed_rows("llm_embedding.weight", &[id])
    }

    /// `speech_embedding` rows for the given speech-token ids, `[1, n, HIDDEN]`.
    pub fn speech_embed(&self, speech_token: &[u32]) -> Result<Tensor> {
        self.embed_rows("speech_embedding.weight", speech_token)
    }

    /// Assemble the step-0 LM input exactly as `Qwen2LM.inference`:
    /// `[sos_emb, text_emb(text_token), task_id_emb, prompt_speech_emb]` -> `[1, T0, HIDDEN]`.
    ///
    /// `text_token` here is already the concatenation of `prompt_text` and `text`
    /// (the reference concatenates them before embedding). `prompt_speech_token` may be
    /// empty, in which case the prompt-speech segment is omitted.
    pub fn build_lm_input(&self, text_token: &[u32], prompt_speech_token: &[u32]) -> Result<Tensor> {
        let sos = self.llm_embed_row(SOS)?; // [1,1,H]
        let task = self.llm_embed_row(TASK_ID)?; // [1,1,H]
        let text = self.text_embed(text_token)?; // [1,Tt,H]
        let mut parts: Vec<Tensor> = vec![sos, text, task];
        if !prompt_speech_token.is_empty() {
            parts.push(self.speech_embed(prompt_speech_token)?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Tensor::cat(&refs, 1)
    }

    /// Last-position raw `llm_decoder` logits for an input embedding sequence
    /// `[1, T, HIDDEN]`, returning `[V]` (`V = SPEECH_VOCAB = 6564`).
    ///
    /// Recomputes the full transformer each call (O(n²) but logit-identical to a KV
    /// cache — same positions, same causal mask), which is what we want for parity.
    pub fn step_logits(&self, embeds: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(embeds)?; // [1, T, V]
        let t = logits.dim(1)?;
        logits.narrow(1, t - 1, 1)?.reshape((SPEECH_VOCAB,))
    }

    /// Autoregressively generate speech tokens, mirroring `Qwen2LM.inference`.
    ///
    /// Starts from `build_lm_input`, then per step: full forward -> last-position logits
    /// -> `log_softmax` -> `ras_sampling` (with `seed`-pinned multinomial draws) -> stop
    /// if the chosen id is a stop token, else append its `speech_embedding` row and
    /// continue. EOS is masked while `step < min_len`. Returns the generated token ids
    /// (stop token excluded), matching the reference's `out_tokens`.
    pub fn generate(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        min_len: usize,
        max_len: usize,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let mut embeds = self.build_lm_input(text_token, prompt_speech_token)?;
        let mut rng = SplitMix64::new(seed);
        let mut out: Vec<u32> = Vec::new();
        for i in 0..max_len {
            let logits = self.step_logits(&embeds)?;
            let logp = log_softmax_vec(&logits)?;
            let ignore_eos = i < min_len;
            let top = ras_sampling(&logp, &out, ignore_eos, &mut rng);
            if STOP_TOKENS.contains(&top) {
                break;
            }
            out.push(top);
            let row = self.speech_embed(&[top])?; // [1,1,H]
            embeds = Tensor::cat(&[&embeds, &row], 1)?;
        }
        Ok(out)
    }

    /// Teacher-forced per-step logits: given the reference's chosen token sequence,
    /// rebuild the full embedding sequence and return every step's last-position logits
    /// as `[N, V]` (step `k`'s logits at row `k`). This proves the AR forward reproduces
    /// the reference logit-for-logit independent of the (stochastic) sampler, and is the
    /// real correctness signal for the generation loop.
    pub fn teacher_forced_logits(
        &self,
        text_token: &[u32],
        prompt_speech_token: &[u32],
        gen_tokens: &[u32],
    ) -> Result<Tensor> {
        let lm_input0 = self.build_lm_input(text_token, prompt_speech_token)?;
        let t0 = lm_input0.dim(1)?;
        // Append speech embeddings for all but the last generated token: step k's
        // last-position logit lives at absolute position (t0 - 1 + k).
        let embeds = if gen_tokens.len() > 1 {
            let tail = self.speech_embed(&gen_tokens[..gen_tokens.len() - 1])?;
            Tensor::cat(&[&lm_input0, &tail], 1)?
        } else {
            lm_input0
        };
        let logits = self.forward_logits(&embeds)?; // [1, T, V]
        let n = gen_tokens.len();
        // rows [t0-1 .. t0-1+n) are the n per-step last-position logits.
        logits.narrow(1, t0 - 1, n)?.reshape((n, SPEECH_VOCAB))
    }
}

// --- generation constants (from the CosyVoice2 Qwen2LM definition) -----------

/// `sos` row index into `llm_embedding`.
const SOS: u32 = 0;
/// `task_id` row index into `llm_embedding`.
const TASK_ID: u32 = 1;
/// `llm_decoder` output width (`speech_token_size + 3` = 6561 + 3).
const SPEECH_VOCAB: usize = 6564;
/// `speech_token_size`; `eos_token = speech_token_size`, and the stop set is the three
/// ids `[speech_token_size + i for i in range(3)]`.
const SPEECH_TOKEN_SIZE: u32 = 6561;
/// The decode-stop token ids (`stop_token_ids`).
const STOP_TOKENS: [u32; 3] = [SPEECH_TOKEN_SIZE, SPEECH_TOKEN_SIZE + 1, SPEECH_TOKEN_SIZE + 2];

/// `log_softmax` over a 1-D logit vector `[V]`, returned as a host `Vec<f32>`.
fn log_softmax_vec(logits: &Tensor) -> Result<Vec<f32>> {
    let v: Vec<f32> = logits.to_vec1()?;
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f64;
    for &x in &v {
        sum += ((x - m) as f64).exp();
    }
    let lse = m as f64 + sum.ln();
    Ok(v.iter().map(|&x| (x as f64 - lse) as f32).collect())
}

/// Deterministic SplitMix64 PRNG — pins the otherwise-stochastic multinomial draws so a
/// `generate` run is bit-reproducible from a seed (the reference pins torch's RNG; we
/// pin ours). `next_f64` yields a uniform in `[0, 1)`.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn next_f64(&mut self) -> f64 {
        // 53-bit mantissa uniform in [0,1)
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// Sample one index from a categorical distribution given by `probs` (need not be
/// normalised) using inverse-CDF on a single uniform draw — the deterministic analogue
/// of `torch.multinomial(probs, 1)`.
fn multinomial1(probs: &[f32], rng: &mut SplitMix64) -> usize {
    let total: f64 = probs.iter().map(|&p| p as f64).sum();
    let u = rng.next_f64() * total;
    let mut acc = 0f64;
    for (i, &p) in probs.iter().enumerate() {
        acc += p as f64;
        if u < acc {
            return i;
        }
    }
    probs.len() - 1
}

/// `nucleus_sampling`: softmax(logp) is `exp(logp)`; sort descending (stable), take the
/// leading tokens while `cum_prob < top_p` AND `count < top_k`, then sample one of those
/// by `multinomial`. Returns the chosen vocab id. `logp` is a log-probability vector.
fn nucleus_sampling(logp: &[f32], top_p: f32, top_k: usize, rng: &mut SplitMix64) -> u32 {
    // probabilities = exp(log_softmax)
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    // stable descending sort by probability (ties keep ascending index, like torch stable)
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    let mut cum = 0f32;
    let mut cand_idx: Vec<u32> = Vec::new();
    let mut cand_prob: Vec<f32> = Vec::new();
    for &i in &order {
        if cum < top_p && cand_prob.len() < top_k {
            cum += probs[i];
            cand_prob.push(probs[i]);
            cand_idx.push(i as u32);
        } else {
            break;
        }
    }
    let pick = multinomial1(&cand_prob, rng);
    cand_idx[pick]
}

/// `random_sampling`: full-softmax multinomial over the whole vocab (used by `ras` after
/// it masks a repeated token).
fn random_sampling(logp: &[f32], rng: &mut SplitMix64) -> u32 {
    let probs: Vec<f32> = logp.iter().map(|&x| x.exp()).collect();
    multinomial1(&probs, rng) as u32
}

/// `ras_sampling` (Repetition-Aware Sampling): nucleus-sample a candidate; if it has
/// repeated `>= win_size * tau_r` times in the last `win_size` decoded tokens, mask it
/// and fall back to `random_sampling`. EOS (`speech_token_size`) is `-inf`-masked first
/// when `ignore_eos`. Mirrors `cosyvoice.utils.common.ras_sampling` with the pinned
/// `top_p=0.8, top_k=25, win_size=10, tau_r=0.1`.
fn ras_sampling(logp: &[f32], decoded: &[u32], ignore_eos: bool, rng: &mut SplitMix64) -> u32 {
    const TOP_P: f32 = 0.8;
    const TOP_K: usize = 25;
    const WIN: usize = 10;
    const TAU_R: f32 = 0.1;
    let mut logp = logp.to_vec();
    if ignore_eos {
        logp[SPEECH_TOKEN_SIZE as usize] = f32::NEG_INFINITY;
    }
    let top = nucleus_sampling(&logp, TOP_P, TOP_K, rng);
    let start = decoded.len().saturating_sub(WIN);
    let rep = decoded[start..].iter().filter(|&&t| t == top).count();
    if (rep as f32) >= WIN as f32 * TAU_R {
        logp[top as usize] = f32::NEG_INFINITY;
        return random_sampling(&logp, rng);
    }
    top
}

/// Apply rotary position embedding (HF `rotate_half` convention) to `[b, h, t, d]`.
fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
    x.broadcast_mul(cos)?.add(&rot.broadcast_mul(sin)?)
}

/// GQA: expand `[b, kv, t, d]` KV heads so each serves `n` query heads -> `[b, kv*n, t, d]`.
fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, kv, t, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, kv, n, t, d))?.reshape((b, kv * n, t, d))
}

/// Build RoPE cos/sin tables `[t, head_dim]` for positions `0..t`.
fn rope_cos_sin(t: usize, dev: &Device) -> Result<(Tensor, Tensor)> {
    let half = HEAD_DIM / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / ROPE_THETA.powf(2.0 * i as f32 / HEAD_DIM as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, (half,), dev)?;
    let pos: Vec<f32> = (0..t).map(|i| i as f32).collect();
    let pos = Tensor::from_vec(pos, (t,), dev)?;
    let freqs = pos.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [t, half]
    let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1)?; // [t, head_dim]
    Ok((emb.cos()?, emb.sin()?))
}

/// Additive causal mask `[t, t]`: 0 on/below the diagonal, -inf above.
fn causal_mask(t: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (t, t), dev)
}
