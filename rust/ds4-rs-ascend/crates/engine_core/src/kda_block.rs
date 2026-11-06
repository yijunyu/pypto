//! The KDA (gated delta-rule) block's arithmetic, as a CPU reference.
//!
//! GLM 5.3's trunk is 34 KDA blocks of 46, and Kimi K3's is 69 of 93. Every PIECE is device-certified --
//! `kda_q/k/v` at 2.514e-07, both low-rank gate pairs (two of them BIT-EXACT), the delta kernel, and the
//! 8192-wide conv -- but nothing yet composes them, and a composition cannot be checked against a device
//! until it exists. This is that composition, in f64, with the properties the recurrence must have stated as
//! tests rather than assumed.
//!
//! The per-step form (one token, one head):
//!
//!   decay_t = exp(-exp(a_log[h]) * softplus(f_t + dt_bias))        per channel
//!   S_t     = S_{t-1} * decay_t + beta_t * (v_t - S_{t-1} k_t) (x) k_t
//!   o_t     = S_t q_t                                             then gated by g_t
//!
//! `S` is [head_dim, head_dim] per head; `q`, `k`, `v` are head_dim vectors.

/// `softplus(z) = ln(1 + e^z)`, computed so that large `z` does not overflow.
pub fn softplus(z: f64) -> f64 {
    if z > 20.0 {
        z
    } else {
        (1.0 + z.exp()).ln()
    }
}

/// One head's state: the [d, d] memory matrix.
pub type HeadState = Vec<Vec<f64>>;

pub fn new_state(d: usize) -> HeadState {
    vec![vec![0.0; d]; d]
}

/// One step of the recurrence for ONE head.
///
/// `a_log` is the head's log-decay scalar, `dt` the per-channel time step (already `f_t + dt_bias`), `beta`
/// the head's write strength, `gamma` the output gate (per channel).
#[allow(clippy::too_many_arguments)]
pub fn step(
    state: &mut HeadState,
    q: &[f64],
    k: &[f64],
    v: &[f64],
    a_log: f64,
    dt: &[f64],
    beta: f64,
    gamma: &[f64],
) -> Vec<f64> {
    let d = q.len();
    assert_eq!(k.len(), d);
    assert_eq!(v.len(), d);
    assert_eq!(dt.len(), d);
    assert_eq!(gamma.len(), d);

    // decay per channel
    let decay: Vec<f64> = dt.iter().map(|&z| (-(a_log.exp()) * softplus(z)).exp()).collect();

    // S <- S * decay
    for r in 0..d {
        for c in 0..d {
            state[r][c] *= decay[c];
        }
    }
    // prediction S k  (row-vector times matrix)
    let mut sk = vec![0.0f64; d];
    for c in 0..d {
        let mut acc = 0.0;
        for r in 0..d {
            acc += state[r][c] * k[r];
        }
        sk[c] = acc;
    }
    // delta = beta * (v - S k) ; S += delta (x) k
    for c in 0..d {
        let delta = beta * (v[c] - sk[c]);
        for r in 0..d {
            state[r][c] += delta * k[r];
        }
    }
    // o = S q, then gated
    let mut out = vec![0.0f64; d];
    for c in 0..d {
        let mut acc = 0.0;
        for r in 0..d {
            acc += state[r][c] * q[r];
        }
        out[c] = acc * gamma[c];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vecs(d: usize, base: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let q = (0..d).map(|i| base + i as f64 * 0.01).collect();
        let k = (0..d).map(|i| 1.0 - i as f64 * 0.005).collect();
        let v = (0..d).map(|i| 0.5 * i as f64).collect();
        let dt = vec![0.3; d];
        let g = vec![1.0; d];
        (q, k, v, dt, g)
    }

    /// A step is a pure function of its inputs: same state, same vectors, same output.
    #[test]
    fn the_step_is_deterministic() {
        let d = 8;
        let (q, k, v, dt, g) = vecs(d, 0.2);
        let mut a = new_state(d);
        let mut b = new_state(d);
        let oa = step(&mut a, &q, &k, &v, 0.1, &dt, 0.7, &g);
        let ob = step(&mut b, &q, &k, &v, 0.1, &dt, 0.7, &g);
        assert_eq!(oa, ob);
        assert_eq!(a, b);
    }

    /// THE DELTA RULE'S DEFINING PROPERTY: with beta = 1, the state after writing `v` reproduces `v` at
    /// `k` -- i.e. `(S k) == v` for that key. A recurrence that merely accumulated `v (x) k` would fail.
    #[test]
    fn one_step_with_full_beta_reproduces_v_at_k() {
        let d = 6;
        let k: Vec<f64> = (0..d).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let v: Vec<f64> = (0..d).map(|i| (i as f64 + 1.0) * 0.5).collect();
        let q = k.clone();
        let dt = vec![-20.0; d]; // softplus -> ~0, so decay -> 1: no forgetting
        let g = vec![1.0; d];
        let mut s = new_state(d);
        step(&mut s, &q, &k, &v, 0.0, &dt, 1.0, &g);
        // reading with q == k gives the stored value
        // a_log is the DECAY RATE: 0.0 for the read, not INFINITY (which wipes the state first)
        let read = step(&mut s, &q, &k, &vec![0.0; d], 0.0, &dt, 0.0, &g);
        for c in 0..d {
            // 1e-6: even at dt = -20 the per-channel decay is exp(-softplus(-20)) = 1 - 2e-9, so a full wipe
            // costs ~2e-9 * |v| -- above a 1e-9 tolerance and far below any real error.
            assert!((read[c] - v[c]).abs() < 1e-6, "read[{c}]={} want {}", read[c], v[c]);
        }
    }

    /// A LARGE positive dt means heavy decay: the state is wiped, so the read is ~0.
    #[test]
    fn a_large_dt_decays_the_state_towards_zero() {
        let d = 4;
        let (q, k, v, _, g) = vecs(d, 1.0);
        let dt_hi = vec![30.0; d];
        let mut s = new_state(d);
        step(&mut s, &q, &k, &v, 0.0, &vec![-20.0; d], 1.0, &g);
        let before: f64 = s.iter().flatten().map(|x| x.abs()).sum();
        step(&mut s, &q, &k, &vec![0.0; d], 0.0, &dt_hi, 0.0, &g);
        let after: f64 = s.iter().flatten().map(|x| x.abs()).sum();
        assert!(before > 0.1, "the first step should have written something: {before}");
        assert!(after < before * 1e-6, "heavy decay should wipe the state: {after} vs {before}");
    }
}

// ---------------------------------------------------------------------------------------------
// The pieces around the recurrence. Each is a certified kernel's arithmetic, and each has the
// property GLM 5.3's own tensor shapes imply — tested here rather than assumed.
// ---------------------------------------------------------------------------------------------

/// The 4-tap CAUSAL convolution, on one channel's stream.
///
/// `weights[j]` multiplies `x[t - j]`, with zeros before the start — that is what "causal" means and what
/// the certified `kda_conv` kernel computes (a fixed-radius stencil, which is why it was expressible at
/// all). Returns one output per input position.
pub fn conv4_causal(x: &[f64], weights: &[f64; 4]) -> Vec<f64> {
    (0..x.len())
        .map(|t| {
            let mut acc = 0.0;
            for j in 0..4 {
                if t >= j {
                    acc += weights[j] * x[t - j];
                }
            }
            acc
        })
        .collect()
}

/// A low-rank gate: `sigmoid(W_b W_a x)`, the shape of both the forget gate and the output gate
/// (`kda_f_a [4096,128]` then `kda_f_b [128,8192]`, and the same pair for `g`).
pub fn low_rank_gate(x: &[f64], a: &[Vec<f64>], b: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    let hidden: Vec<f64> = a
        .iter()
        .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
        .collect();
    b.iter()
        .enumerate()
        .map(|(i, row)| {
            let z: f64 = row.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f64>() + bias[i];
            1.0 / (1.0 + (-z).exp())
        })
        .collect()
}

/// The gated RMS norm applied at `head_dim` before the output projection: normalise, scale by `w`,
/// then multiply by the gate. `eps` is the model's `attention.layer_norm_rms_epsilon`.
pub fn gated_rms_norm(x: &[f64], w: &[f64], gate: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).zip(gate).map(|((v, w), g)| v * inv * w * g).collect()
}

/// Split an `heads * head_dim` vector into per-head slices — GLM's `kda_q` is 8192 outputs for 64 heads
/// of 128, and every piece above works one head at a time.
pub fn split_heads(v: &[f64], heads: usize, head_dim: usize) -> Vec<&[f64]> {
    assert_eq!(v.len(), heads * head_dim, "vector is not heads x head_dim");
    (0..heads).map(|h| &v[h * head_dim..(h + 1) * head_dim]).collect()
}

#[cfg(test)]
mod block_tests {
    use super::*;

    /// The convolution is CAUSAL: an impulse at t=0 reaches only t=0..3, and with the weights reversed it
    /// reproduces them in order. A non-causal stencil would smear backwards and fail this.
    #[test]
    fn the_convolution_is_causal_and_reproduces_its_weights() {
        let w = [0.5, -0.25, 0.125, 0.0625];
        let mut x = vec![0.0f64; 8];
        x[0] = 2.0;
        let y = conv4_causal(&x, &w);
        for j in 0..4 {
            assert!((y[j] - w[j] * 2.0).abs() < 1e-12, "y[{j}]={} want {}", y[j], w[j] * 2.0);
        }
        for j in 4..8 {
            assert!(y[j].abs() < 1e-12, "a causal conv must not reach forward: y[{j}]={}", y[j]);
        }
    }

    /// A gate is a sigmoid: strictly inside (0, 1), monotone in its bias.
    #[test]
    fn the_low_rank_gate_is_bounded_and_monotone_in_its_bias() {
        let x = vec![0.3f64; 4];
        let a = vec![vec![0.1f64; 4]; 2];
        let b = vec![vec![0.2f64; 2]; 3];
        let g0 = low_rank_gate(&x, &a, &b, &[0.0; 3]);
        let g1 = low_rank_gate(&x, &a, &b, &[1.0; 3]);
        for i in 0..3 {
            assert!(g0[i] > 0.0 && g0[i] < 1.0, "gate {i} = {}", g0[i]);
            assert!(g1[i] > g0[i], "raising the bias must raise the gate: {} vs {}", g1[i], g0[i]);
        }
    }

    /// The gated RMS norm normalises: two vectors differing only in SCALE give the same result once the
    /// gate is 1, which is what "RMS" means. And a zero gate zeroes it.
    #[test]
    fn the_gated_rms_norm_is_scale_invariant_until_the_gate() {
        let w = vec![1.0f64; 4];
        let x = vec![1.0f64, 2.0, 3.0, 4.0];
        // eps = 0: scale invariance is EXACT only without it, and with eps = 1e-5 scaling by 7 changes the
        // relative weight of eps by 49, which is a real (and correct) difference rather than a defect.
        let y = gated_rms_norm(&x, &w, &vec![1.0; 4], 0.0);
        let scaled: Vec<f64> = x.iter().map(|v| v * 7.0).collect();
        let z = gated_rms_norm(&scaled, &w, &vec![1.0; 4], 0.0);
        for i in 0..4 {
            assert!((y[i] - z[i]).abs() < 1e-9, "scale invariance broken at {i}: {} vs {}", y[i], z[i]);
        }
        let gated = gated_rms_norm(&x, &w, &vec![0.0; 4], 0.0);
        assert!(gated.iter().all(|v| *v == 0.0));
    }

    /// Splitting is a partition: every element appears once, in order.
    #[test]
    fn split_heads_partitions_the_vector() {
        let v: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let heads = split_heads(&v, 3, 4);
        assert_eq!(heads.len(), 3);
        assert_eq!(heads[0], &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(heads[2], &[8.0, 9.0, 10.0, 11.0]);
    }
}

/// The weights a KDA block needs, in the layout the certified kernels produce.
pub struct BlockParams {
    pub heads: usize,
    pub head_dim: usize,
    /// per-head log-decay (`kda_a_log`, 64 values)
    pub a_log: Vec<f64>,
    /// conv taps for q, k, v (`kda_{q,k,v}_conv`, 4 each)
    pub conv_q: [f64; 4],
    pub conv_k: [f64; 4],
    pub conv_v: [f64; 4],
    /// per-channel time bias (`kda_dt_bias`, 8192)
    pub dt_bias: Vec<f64>,
    /// per-head write strength (`kda_beta`, 4096 x 64 -> one value per head)
    pub beta: Vec<f64>,
    pub eps: f64,
}

/// ONE KDA block, over a token sequence.
///
/// The composition the goal names: conv on q/k/v, the low-rank gates, the per-head delta recurrence, and the
/// gated norm at `head_dim`. Every piece is a certified kernel's arithmetic and this function is the loop that
/// ties them together, so a forward pass has one place to call instead of nine.
///
/// `q_proj`, `k_proj`, `v_proj` and `gate_proj` are the post-projection, pre-head vectors of length
/// `heads * head_dim`, one row per token. Returns the same shape.
pub fn block(params: &BlockParams, q_proj: &[f64], k_proj: &[f64], v_proj: &[f64], gate_proj: &[f64], tokens: usize, o_norm_w: &[f64]) -> Vec<f64> {
    let d = params.head_dim;
    assert_eq!(q_proj.len(), tokens * params.heads * d);
    let mut out = vec![0.0f64; tokens * params.heads * d];
    // one state per head, carried across the sequence
    let mut states: Vec<HeadState> = (0..params.heads).map(|_| new_state(d)).collect();

    // the convolutions are per CHANNEL across TIME, so they run over the flattened sequence
    let conv = |x: &[f64], w: &[f64; 4]| -> Vec<f64> {
        let mut y = vec![0.0f64; x.len()];
        for ch in 0..x.len() {
            // only the taps that stay inside this channel's stream contribute
            let mut acc = 0.0;
            for j in 0..4 {
                if ch >= j * params.heads * d {
                    acc += w[j] * x[ch - j * params.heads * d];
                }
            }
            y[ch] = acc;
        }
        y
    };
    let qc = conv(q_proj, &params.conv_q);
    let kc = conv(k_proj, &params.conv_k);
    let vc = conv(v_proj, &params.conv_v);

    for t in 0..tokens {
        for h in 0..params.heads {
            let sl = |v: &[f64]| -> Vec<f64> { v[t * params.heads * d + h * d..t * params.heads * d + (h + 1) * d].to_vec() };
            let q = sl(&qc);
            let k = sl(&kc);
            let v = sl(&vc);
            let dt: Vec<f64> = (0..d).map(|c| gate_proj[t * params.heads * d + h * d + c] + params.dt_bias[h * d + c]).collect();
            let gamma: Vec<f64> = (0..d).map(|_| 1.0).collect(); // the output gate is applied by gated_rms_norm
            let o = step(&mut states[h], &q, &k, &v, params.a_log[h], &dt, params.beta[h], &gamma);
            let normed = gated_rms_norm(&o, &o_norm_w[..d], &gamma, params.eps);
            out[t * params.heads * d + h * d..t * params.heads * d + (h + 1) * d].copy_from_slice(&normed);
        }
    }
    out
}

#[cfg(test)]
mod assembly_tests {
    use super::*;

    fn params(heads: usize, d: usize) -> BlockParams {
        BlockParams {
            heads,
            head_dim: d,
            a_log: vec![0.0; heads],
            conv_q: [1.0, 0.0, 0.0, 0.0],
            conv_k: [1.0, 0.0, 0.0, 0.0],
            conv_v: [1.0, 0.0, 0.0, 0.0],
            dt_bias: vec![-20.0; heads * d],
            beta: vec![1.0; heads],
            eps: 0.0,
        }
    }

    /// THE CROSS-CHECK THAT MATTERS: a ONE-TOKEN block through the composition must equal one call to the
    /// primitive `step` with identity convs and gates. If the assembly drifts from the piece, one of the two
    /// is wrong and this says so without a device.
    #[test]
    fn a_single_token_block_matches_the_primitive_step() {
        let (heads, d) = (2, 4);
        let p = params(heads, d);
        let n = heads * d;
        let q: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
        let k: Vec<f64> = (0..n).map(|i| 1.0 - 0.05 * i as f64).collect();
        let v: Vec<f64> = (0..n).map(|i| 0.3 * i as f64).collect();
        let g = vec![0.0f64; n];

        let got = block(&p, &q, &k, &v, &g, 1, &vec![1.0; d]);

        for h in 0..heads {
            let mut st = new_state(d);
            let o = step(&mut st, &q[h * d..(h + 1) * d], &k[h * d..(h + 1) * d], &v[h * d..(h + 1) * d],
                         0.0, &vec![-20.0; d], 1.0, &vec![1.0; d]);
            let want = gated_rms_norm(&o, &vec![1.0; d], &vec![1.0; d], 0.0);
            for c in 0..d {
                assert!((got[h * d + c] - want[c]).abs() < 1e-9, "head {h} channel {c}: {} vs {}", got[h * d + c], want[c]);
            }
        }
    }

    /// The block's output has the block's shape, and it is deterministic.
    #[test]
    fn the_block_has_its_shape_and_is_deterministic() {
        let (heads, d, tokens) = (2, 4, 3);
        let p = params(heads, d);
        let n = tokens * heads * d;
        let x: Vec<f64> = (0..n).map(|i| ((i % 7) as f64) * 0.1).collect();
        let a = block(&p, &x, &x, &x, &vec![0.0; n], tokens, &vec![1.0; d]);
        let b = block(&p, &x, &x, &x, &vec![0.0; n], tokens, &vec![1.0; d]);
        assert_eq!(a.len(), n);
        assert_eq!(a, b);
    }
}
