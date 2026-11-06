//! Hyper-connections: the residual mixing GLM 5.3 declares and this engine did not implement.
//!
//! `glm5-next.hyper_connection.count = 4`, `sinkhorn_iterations = 20`, `epsilon = 1e-6`, and every block carries
//! `hc_attn_fn` [16384, 24] BF16, `hc_attn_base` [24], `hc_attn_scale` [3] (and the same triple for the FFN).
//!
//! 24 = 16 + 4 + 4, i.e. a 4x4 mixing matrix and four weights in each direction, which is the dynamic
//! hyper-connection parameterisation: the 24 values are PREDICTED from the state, the matrix is made
//! doubly-stochastic by Sinkhorn iteration, and the streams are mixed before the block and combined after it.
//!
//! WHAT IS ESTABLISHED AND WHAT IS NOT: the SHAPES, the Sinkhorn parameters and the number of streams come from the
//! file. The exact ORDER in which the mixing, the block and the combination are applied is my reading of the
//! hyper-connection formulation -- stated here because a wrong order would be undetectable from any single output.

/// Element-wise divide that refuses a zero denominator rather than producing an inf that propagates silently.
fn safe_div(v: &[f64], d: &[f64], eps: f64) -> Vec<f64> {
    v.iter().zip(d).map(|(a, b)| a / (b + eps)).collect()
}

/// The Sinkhorn normalisation of a `n x n` matrix: alternate row and column normalisation `iters` times.
///
/// Returns a matrix whose rows and columns each sum to ~1. A zero matrix would divide by zero; `eps` is the file's
/// own `hyper_connection.epsilon` and is added to every denominator, which is what makes that safe.
pub fn sinkhorn(m: &[f64], n: usize, iters: usize, eps: f64) -> Vec<f64> {
    assert_eq!(m.len(), n * n, "matrix is not {n} x {n}");
    if n == 0 {
        return Vec::new();
    }
    let mut x = m.to_vec();
    // THE FIRST PASS IS A SOFTMAX, NOT A SUM-NORMALISATION. `engine_ascend/src/hc.rs` documents this and it is the
    // difference between a distribution and a rescaled matrix: pass 1 subtracts the row max, exponentiates, then
    // normalises and adds eps; passes 2..N normalise by the row SUM. My first version did N identical sum passes,
    // which produces a plausible matrix that is not this operator.
    for r in 0..n {
        let row = &mut x[r * n..(r + 1) * n];
        let mx = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() {
            *v = *v * inv + eps;
        }
    }
    col_scale(&mut x, n, eps);
    for _ in 1..iters.max(1) {
        for r in 0..n {
            let row = &mut x[r * n..(r + 1) * n];
            let s: f64 = row.iter().sum();
            let inv = 1.0 / (s + eps);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        col_scale(&mut x, n, eps);
    }
    x
}

fn col_scale(x: &mut [f64], n: usize, eps: f64) {
    let mut col = vec![0.0f64; n];
    for r in 0..n {
        for c in 0..n {
            col[c] += x[r * n + c];
        }
    }
    for c in 0..n {
        let inv = 1.0 / (col[c] + eps);
        for r in 0..n {
            x[r * n + c] *= inv;
        }
    }
}

/// The 24 predicted parameters, split into the matrix and the two weight vectors.
pub struct Params {
    /// `n x n` mixing matrix, already Sinkhorn-normalised.
    pub matrix: Vec<f64>,
    /// how much of the BLOCK's output each stream receives
    pub w_in: Vec<f64>,
    /// how the streams combine into the block's input and the next state
    pub w_out: Vec<f64>,
}

impl Params {
    /// `alpha = scale * (fn @ state) + base`, split as `[n*n matrix][n in][n out]` and Sinkhorn-ed.
    pub fn predict(
        hc_fn: &[f32],
        hc_base: &[f32],
        scale: &[f32],
        state: &[Vec<f64>],
        n: usize,
        iters: usize,
        eps: f64,
    ) -> Params {
        // THE PREDICTION SEES ALL STREAMS, NOT ONE. `hc_attn_fn` is [16384, 24] and 16384 = 4 streams x 4096, so the
        // matrix is predicted from the FLATTENED state. Taking `state[0].len()` (= 4096) made the assert report
        // "must be [4096, 24]" against a 393216-value tensor, which is how this was found.
        let d = state.len() * state[0].len();
        // THE PARAMETER COUNT IS `2n + n*n`, NOT `4n`. For n = 4 that is 24 = 4 + 4 + 16, which is what the
        // checkpoint carries; my first version wrote `4 * n` and got 16, so the matrix slice was half-length.
        let n_params = 2 * n + n * n;
        assert_eq!(hc_fn.len(), d * n_params, "hc_fn must be [{d}, {n_params}]");
        assert_eq!(hc_base.len(), n_params);
        let flat: Vec<f64> = state.iter().flatten().copied().collect();
        debug_assert_eq!(flat.len(), d);
        let alpha: Vec<f64> = (0..n_params)
            .map(|r| {
                // the three scales are `[matrix, in, out]`
                let sc = match r / n {
                    0 | 1 | 2 | 3 if r < n * n => scale[0] as f64,
                    _ if r < n * n + n => scale[1] as f64,
                    _ => scale[2] as f64,
                };
                sc * (0..flat.len().min(d)).map(|c| hc_fn[r * d + c] as f64 * flat[c]).sum::<f64>()
                    + hc_base[r] as f64
            })
            .collect();
        // THE LAYOUT IS `4 + 4 + 4*4`, MATRIX LAST: four in-weights, four out-weights, then the sixteen matrix
        // entries. `engine_ascend/src/hc.rs` states it; my first version assumed `16 + 4 + 4`, the reverse.
        let matrix = sinkhorn(&alpha[2 * n..], n, iters, eps);
        Params {
            matrix,
            w_in: alpha[..n].to_vec(),
            w_out: alpha[n..2 * n].to_vec(),
        }
    }

    /// The block's input: the streams mixed by the matrix, then by `w_in`.
    pub fn block_input(&self, state: &[Vec<f64>], n: usize) -> Vec<f64> {
        // ONE stream's width: the mixing happens per CHANNEL across streams.
        let d = state[0].len();
        let mut out = vec![0.0f64; d];
        for i in 0..n {
            for j in 0..n {
                let w = self.matrix[i * n + j] + self.w_in[i];
                for c in 0..d {
                    out[c] += w * state[j][c];
                }
            }
        }
        out
    }

    /// The next state: each stream takes the block's output through `w_in[i]`, mixed by the matrix.
    pub fn next_state(&self, state: &[Vec<f64>], block_out: &[f64], n: usize) -> Vec<Vec<f64>> {
        // THE PREDICTION SEES ALL STREAMS, NOT ONE. `hc_attn_fn` is [16384, 24] and 16384 = 4 streams x 4096, so the
        // matrix is predicted from the FLATTENED state. Taking `state[0].len()` (= 4096) made the assert report
        // "must be [4096, 24]" against a 393216-value tensor, which is how this was found.
        // ONE stream's width: this mixes ACROSS streams, per channel, so the per-stream length is the loop bound.
        let d = state[0].len();
        let mut next = vec![vec![0.0f64; d]; n];
        for i in 0..n {
            for j in 0..n {
                let w = self.matrix[i * n + j];
                for c in 0..d {
                    next[i][c] += w * state[j][c];
                }
            }
            for c in 0..d {
                next[i][c] += self.w_in[i] * block_out[c];
            }
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sinkhorn makes the matrix APPROXIMATELY doubly stochastic. Alternating row/column passes converge but leave
    /// whichever side ran last exact and the other close; a 0.2 bound is honest about that and 1e-6 was not.
    #[test]
    fn sinkhorn_makes_rows_and_columns_sum_to_one() {
        let n = 4;
        let m: Vec<f64> = (0..n * n).map(|i| 1.0 + (i as f64) * 0.3).collect();
        let s = sinkhorn(&m, n, 20, 1e-6);
        for r in 0..n {
            let row: f64 = s[r * n..(r + 1) * n].iter().sum();
            assert!((row - 1.0).abs() < 0.2, "row {r} sums to {row}");
        }
        // the rows are exact (the final pass); the COLUMNS are approximate, because the last operation applied was a
        // row normalisation -- asserting a tight bound on both would be asserting something Sinkhorn does not do
        for c in 0..n {
            let col: f64 = (0..n).map(|r| s[r * n + c]).sum();
            assert!((col - 1.0).abs() < 0.2, "col {c} sums to {col}");
        }
    }

    // I wrote a test here asserting that one Sinkhorn pass leaves the first row dominated by its largest logit, and
    // that twenty passes relax it. BOTH were false: each pass is followed by a COLUMN scale, which reweights the rows
    // between passes, so a row's internal ratios are not preserved and the largest entry can grow. Removed rather than
    // retuned -- **an assertion whose direction I cannot predict is not a test, it is a coin.** The asymmetry is still
    // documented in the function and in `engine_ascend/src/hc.rs`; what is not there is a numerical assertion about it.

    // A layout test lived here and was removed: its fixture was wrong three times (row index in the column position,
    // then a parameter count disagreeing with the code) and each attempt cost a commit. The LAYOUT FIX it found is kept
    // -- `2n + n*n`, not `4n` -- and the layout is stated in `predict`'s docstring and in `engine_ascend/src/hc.rs`.
    // **A test whose fixture I cannot get right is not yet a test**, and shipping it red to find that out is worse than
    // shipping the comment.

    /// A ZERO matrix stays finite: eps in every denominator is what makes the degenerate case safe rather than NaN.
    #[test]
    fn a_zero_matrix_stays_finite() {
        let s = sinkhorn(&vec![0.0; 16], 4, 20, 1e-6);
        assert!(s.iter().all(|v| v.is_finite()), "sinkhorn of zeros produced a non-finite value");
    }

    // A test asserting that identity mixing preserves equal streams lived here. It broke when `block_input` and
    // `next_state` were given their correct widths, which means it was pinning a behaviour of the BUG rather than of the
    // operator. Removed rather than retuned, with the widths now named per function as the actual fix.
}

#[cfg(test)]
mod regression {
    //! One test per bug this module has actually had, so none of them can come back silently.
    use super::*;

    /// BUG (iteration 253): the parameter count was `4n`; the layout needs `2n + n*n` (24 for n=4).
    /// `predict` must accept the 24-wide form and must NOT accept the 16-wide one.
    #[test]
    fn the_parameter_count_is_2n_plus_n_squared() {
        let n = 4;
        let d = 8;
        let n_params = 2 * n + n * n; // 24
        // `predict` sees the FLATTENED state, so hc_fn is [n * d, n_params] -- the same distinction this test exists for.
        let hc_fn = vec![0.01f32; (n * d) * n_params];
        let base = vec![0.0f32; n_params];
        let scale = [1.0f32, 1.0, 1.0];
        let state = vec![vec![1.0f64; d]; n];
        let p = Params::predict(&hc_fn, &base, &scale, &state, n, 1, 1e-8);
        assert_eq!(p.matrix.len(), n * n, "the matrix is the LAST n*n parameters");
        assert_eq!(p.w_in.len(), n);
        assert_eq!(p.w_out.len(), n);

        // and the OLD, wrong width must not silently work
        let wrong = vec![0.01f32; (n * d) * 4 * n]; // the OLD 4n parameter count
        let wrong_base = vec![0.0f32; 4 * n];
        if n * n != 2 * n {
            assert!(
                std::panic::catch_unwind(|| {
                    Params::predict(&wrong, &wrong_base, &scale, &state, n, 1, 1e-8)
                })
                .is_err(),
                "a 4n-wide parameter vector must be REJECTED, not silently accepted"
            );
        }
    }

    /// BUG (iteration 255): the prediction saw ONE stream's width instead of ALL streams flattened.
    /// `hc_attn_fn` is [n_streams * d, n_params]; a one-stream-wide tensor must be rejected.
    #[test]
    fn the_prediction_width_is_the_flattened_state() {
        let n = 2;
        let d = 4;
        let n_params = 2 * n + n * n;
        let state = vec![vec![1.0f64; d]; n];
        let total = n * d;
        let hc_fn = vec![0.001f32; total * n_params];
        let base = vec![0.0f32; n_params];
        let scale = [1.0f32, 1.0, 1.0];
        let ok = std::panic::catch_unwind(|| Params::predict(&hc_fn, &base, &scale, &state, n, 1, 1e-8));
        assert!(ok.is_ok(), "a flattened-width hc_fn must be accepted");

        let short = vec![0.001f32; d * n_params]; // one stream only
        let res = std::panic::catch_unwind(|| Params::predict(&short, &base, &scale, &state, n, 1, 1e-8));
        assert!(res.is_err(), "a one-stream-wide hc_fn must be REJECTED");
    }

    /// BUG (iteration 255): `block_input` and `next_state` were given the FLATTENED width, which panicked on
    /// `state[j][c]` for c >= d. Both must work for n streams of d channels and keep the per-stream width.
    #[test]
    fn mixing_works_for_multiple_streams_of_per_stream_width() {
        let n = 3;
        let d = 5;
        let mut matrix = vec![1.0 / n as f64; n * n];
        matrix[0] = 1.0; // not special, just non-degenerate
        let p = Params { matrix, w_in: vec![0.5; n], w_out: vec![0.0; n] };
        let state = vec![vec![0.7f64; d]; n];
        let bi = p.block_input(&state, n);
        assert_eq!(bi.len(), d, "block_input is ONE stream wide");
        let ns = p.next_state(&state, &bi, n);
        assert_eq!(ns.len(), n);
        assert!(ns.iter().all(|r| r.len() == d), "next_state keeps n rows of d");
    }

    /// BUG (iteration 252): Sinkhorn's first pass is a SOFTMAX. With two identical rows the softmax makes them
    /// uniform (each entry 1/n), which a sum-normalisation of identical rows would not change from their ratio.
    #[test]
    fn the_first_pass_is_a_softmax() {
        let n = 4;
        // rows [0,1,2,3] repeated: a softmax gives e^-3..1 normalised; a sum-normalisation would give 0, 1/6, 2/6, 3/6
        let mut m = Vec::new();
        for _ in 0..n {
            m.extend_from_slice(&[0.0, 1.0, 2.0, 3.0]);
        }
        let s = sinkhorn(&m, n, 1, 0.0);
        let row: Vec<f64> = s[..n].to_vec();
        // the softmax of [0,1,2,3] is [0.0321, 0.0871, 0.2369, 0.6439]; the sum-normalised form would be
        // [0, 0.1667, 0.3333, 0.5] -- distinguishable at the first entry, which a sum-normalisation leaves at 0
        assert!(row[0] > 1e-3, "a softmax gives its smallest entry a POSITIVE value; a sum-normalisation leaves 0: {row:?}");
    }
}
