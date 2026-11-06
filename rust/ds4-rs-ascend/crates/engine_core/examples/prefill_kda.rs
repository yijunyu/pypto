//! THE CAUSALITY INVARIANT, on GLM 5.3's real KDA blocks.
//!
//! With two tokens, POSITION 0's residual must be IDENTICAL to the single-token run of token 0, because attention
//! and the delta recurrence are causal. A prefill that leaks future tokens -- the classic off-by-one in a conv's
//! history window or in the recurrence's ordering -- fails this, and nothing else would notice.
//!
//! usage: prefill_kda <gguf>

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::schedule;
use engine_core::glm5_ref::DefaultsGlm5;
use engine_core::glm5_stream::ShardedGguf;
use engine_core::kda_block::{block, gated_rms_norm, BlockParams};
use engine_core::tokenizer::Vocab;
use std::path::PathBuf;

struct Ctx {
    d: usize,
    heads: usize,
    head_dim: usize,
    eps: f64,
}

fn embed(g: &ShardedGguf, v: &Vocab, tok: u32, d: usize) -> anyhow::Result<Vec<f64>> {
    let emb = g.dequant_name("token_embd.weight")?;
    let cols = emb.len() / d;
    let id = tok as usize;
    anyhow::ensure!(id < cols, "token {id} outside the {cols}-wide embedding");
    Ok((0..d).map(|c| emb[c * cols + id] as f64).collect())
}

/// Run `blocks` blocks over `xs` (one row per token). Returns one row per token.
fn run(g: &ShardedGguf, c: &Ctx, blocks: usize, xs: Vec<Vec<f64>>) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = xs.len();
    let mut x = xs;
    for b in 0..blocks {
        let norm_w = g.dequant_name(&format!("blk.{b}.attn_norm.weight"))?;
        let nw: Vec<f64> = norm_w.iter().map(|v| *v as f64).collect();
        let mut q = Vec::with_capacity(n);
        let mut k = Vec::with_capacity(n);
        let mut v = Vec::with_capacity(n);
        let mut gate = Vec::with_capacity(n);
        for t in 0..n {
            let xin = gated_rms_norm(&x[t], &nw, &vec![1.0; c.d], c.eps);
            for (dst, name) in [(&mut q, "kda_q"), (&mut k, "kda_k"), (&mut v, "kda_v")] {
                let w = g.dequant_name(&format!("blk.{b}.{name}.weight"))?;
                let out: Vec<f64> = (0..c.heads * c.head_dim)
                    .map(|r| (0..c.d).map(|cc| w[r * c.d + cc] as f64 * xin[cc]).sum())
                    .collect();
                dst.push(out);
            }
            gate.push(vec![0.0f64; c.heads * c.head_dim]);
        }
        let flat = |rows: &Vec<Vec<f64>>| -> Vec<f64> { rows.iter().flatten().copied().collect() };
        let params = BlockParams {
            heads: c.heads,
            head_dim: c.head_dim,
            a_log: vec![0.0; c.heads],
            conv_q: [1.0, 0.0, 0.0, 0.0],
            conv_k: [1.0, 0.0, 0.0, 0.0],
            conv_v: [1.0, 0.0, 0.0, 0.0],
            dt_bias: vec![0.0; c.heads * c.head_dim],
            beta: vec![0.5; c.heads],
            eps: c.eps,
        };
        let onw = vec![1.0f64; c.head_dim];
        let out = block(&params, &flat(&q), &flat(&k), &flat(&v), &flat(&gate), n, &onw);
        let wo = g.dequant_name(&format!("blk.{b}.kda_output.weight"))?;
        for t in 0..n {
            let o = &out[t * c.heads * c.head_dim..(t + 1) * c.heads * c.head_dim];
            for r in 0..c.d {
                let mut acc = 0.0;
                for cc in 0..o.len() {
                    acc += wo[r * o.len() + cc] as f64 * o[cc];
                }
                x[t][r] += acc;
            }
        }
    }
    Ok(x)
}

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: prefill_kda <gguf>"))?);
    let g0 = GgufFile::open(&path)?;
    let cfg = DefaultsGlm5::from_gguf(&g0)?;
    let plan = schedule(&g0)?;
    let v = Vocab::from_gguf(&g0)?;
    let g = ShardedGguf::open(&path)?;
    let c = Ctx { d: cfg.hidden_size, heads: 64, head_dim: 128, eps: cfg.rms_norm_eps as f64 };
    anyhow::ensure!(plan[0].attn == engine_core::glm53_plan::Attn::Kda, "block 0 must be KDA");

    let ids = v.encode("Hello world");
    println!("prompt -> {ids:?}");
    anyhow::ensure!(ids.len() == 2, "this example wants exactly two tokens, got {}", ids.len());

    // single-token run of token 0
    let one = run(&g, &c, 1, vec![embed(&g, &v, ids[0], c.d)?])?;
    // two-token run of the same token followed by another
    let two = run(&g, &c, 1, vec![embed(&g, &v, ids[0], c.d)?, embed(&g, &v, ids[1], c.d)?])?;

    let diff = one[0].iter().zip(&two[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    println!("position 0 after block 0: |1-token - 2-token| = {diff:.3e}");
    anyhow::ensure!(diff < 1e-12, "the two-token prefill CHANGED position 0 by {diff:.3e} -- attention is not causal");
    println!("PASS: position 0 is identical whether or not a later token is present -- the recurrence is causal");
    Ok(())
}
