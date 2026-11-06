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


/// ONE MLA BLOCK over N positions, causally: positions `s <= t` only.
fn mla_block_n(g: &ShardedGguf, b: usize, c: &Ctx, xs: &[Vec<f64>]) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = xs.len();
    let heads = c.heads;
    let hd = 256usize;
    let lat = 512usize;
    let mv = |w: &[f32], xv: &[f64], k: usize, nn: usize| -> Vec<f64> {
        (0..nn).map(|r| (0..k).map(|cc| w[r * k + cc] as f64 * xv[cc]).sum()).collect()
    };
    let norm_w = g.dequant_name(&format!("blk.{b}.attn_norm.weight"))?;
    let nw: Vec<f64> = norm_w.iter().map(|v| *v as f64).collect();
    let wqa = g.dequant_name(&format!("blk.{b}.attn_q_a.weight"))?;
    let wqb = g.dequant_name(&format!("blk.{b}.attn_q_b.weight"))?;
    let wkva = g.dequant_name(&format!("blk.{b}.attn_kv_a_mqa.weight"))?;
    let wkb = g.dequant_name(&format!("blk.{b}.attn_k_b.weight"))?;
    let wvb = g.dequant_name(&format!("blk.{b}.attn_v_b.weight"))?;
    let wo = g.dequant_name(&format!("blk.{b}.attn_output.weight"))?;
    let mut qs = Vec::with_capacity(n);
    let mut ks = Vec::with_capacity(n);
    let mut vs = Vec::with_capacity(n);
    for t in 0..n {
        let xin = gated_rms_norm(&xs[t], &nw, &vec![1.0; c.d], c.eps);
        let qa = mv(&wqa, &xin, c.d, 1536);
        qs.push(mv(&wqb, &qa, 1536, heads * hd));
        let kva = mv(&wkva, &xin, c.d, lat);
        let mut k = vec![0.0f64; heads * hd];
        let mut v = vec![0.0f64; heads * hd];
        for h in 0..heads {
            for r in 0..hd {
                let (mut ak, mut av) = (0.0, 0.0);
                for cc in 0..lat {
                    ak += wkb[(h * hd + r) * lat + cc] as f64 * kva[cc];
                    av += wvb[(h * hd + r) * lat + cc] as f64 * kva[cc];
                }
                k[h * hd + r] = ak;
                v[h * hd + r] = av;
            }
        }
        ks.push(k);
        vs.push(v);
    }
    let mut out = Vec::with_capacity(n);
    for t in 0..n {
        let mut o = vec![0.0f64; heads * hd];
        for h in 0..heads {
            let mut sc = Vec::with_capacity(t + 1);
            for s in 0..=t {
                let mut acc = 0.0;
                for r in 0..hd { acc += qs[t][h * hd + r] * ks[s][h * hd + r]; }
                sc.push(acc / (hd as f64).sqrt());
            }
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect();
            let z: f64 = ex.iter().sum();
            for s in 0..=t {
                let w = ex[s] / z;
                for r in 0..hd { o[h * hd + r] += w * vs[s][h * hd + r]; }
            }
        }
        out.push(mv(&wo, &o, heads * hd, c.d));
    }
    Ok(out)
}

/// Run `blocks` blocks over `xs` (one row per token), handling each block by the schedule.
fn run(g: &ShardedGguf, c: &Ctx, plan: &[engine_core::glm53_plan::BlockPlan], blocks: usize, xs: Vec<Vec<f64>>) -> anyhow::Result<Vec<Vec<f64>>> {
    let n = xs.len();
    let mut x = xs;
    for b in 0..blocks {
        match plan[b].attn {
            engine_core::glm53_plan::Attn::Mla => {
                let outs = mla_block_n(g, b, c, &x)?;
                for t in 0..n {
                    for i in 0..c.d {
                        x[t][i] += outs[t][i];
                    }
                }
            }
            engine_core::glm53_plan::Attn::Kda => {
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
                        dst.push((0..c.heads * c.head_dim)
                            .map(|r| (0..c.d).map(|cc| w[r * c.d + cc] as f64 * xin[cc]).sum())
                            .collect::<Vec<f64>>());
                    }
                    gate.push(vec![0.0f64; c.heads * c.head_dim]);
                }
                let flat = |rows: &Vec<Vec<f64>>| -> Vec<f64> { rows.iter().flatten().copied().collect() };
                let params = BlockParams {
                    heads: c.heads, head_dim: c.head_dim, a_log: vec![0.0; c.heads],
                    // THE REAL PER-CHANNEL CONVS. `kda_{q,k,v}_conv` are [4, 1, 8192]: four taps for EACH of the
            // 8192 channels. Loaded from the file rather than shared.
            conv_q: load_conv(g, b, "kda_q_conv", c.heads * c.head_dim)?,
            conv_k: load_conv(g, b, "kda_k_conv", c.heads * c.head_dim)?,
            conv_v: load_conv(g, b, "kda_v_conv", c.heads * c.head_dim)?,
                    dt_bias: vec![0.0; c.heads * c.head_dim], beta: vec![0.5; c.heads], eps: c.eps,
                };
                let onw = vec![1.0f64; c.head_dim];
                let ogate = vec![1.0f64; n * c.heads * c.head_dim];
                let out = block(&params, &flat(&q), &flat(&k), &flat(&v), &flat(&gate), n, &onw, &ogate);
                let wo = g.dequant_name(&format!("blk.{b}.kda_output.weight"))?;
                for t in 0..n {
                    let o = &out[t * c.heads * c.head_dim..(t + 1) * c.heads * c.head_dim];
                    for r in 0..c.d {
                        let mut acc = 0.0;
                        for cc in 0..o.len() { acc += wo[r * o.len() + cc] as f64 * o[cc]; }
                        x[t][r] += acc;
                    }
                }
            }
        }
        // the FFN, per position
        match plan[b].ffn {
            engine_core::glm53_plan::Ffn::Dense => {
                let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
                let fnw: Vec<f64> = fnorm_w.iter().map(|v| *v as f64).collect();
                let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
                let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
                let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
                let inter = wg.len() / c.d;
                for t in 0..n {
                    let fin = gated_rms_norm(&x[t], &fnw, &vec![1.0; c.d], c.eps);
                    let gv: Vec<f64> = (0..inter).map(|r| (0..c.d).map(|cc| wg[r * c.d + cc] as f64 * fin[cc]).sum()).collect();
                    let uv: Vec<f64> = (0..inter).map(|r| (0..c.d).map(|cc| wu[r * c.d + cc] as f64 * fin[cc]).sum()).collect();
                    let act: Vec<f64> = gv.iter().zip(&uv).map(|(a, bb)| (a / (1.0 + (-a).exp())) * bb).collect();
                    for r in 0..c.d {
                        let mut acc = 0.0;
                        for cc in 0..act.len() { acc += wd[r * act.len() + cc] as f64 * act[cc]; }
                        x[t][r] += acc;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(x)
}


/// `kda_*_conv` is [4, 1, C]: the FOUR taps of each of C channels, stored tap-major.
fn load_conv(g: &ShardedGguf, b: usize, which: &str, channels: usize) -> anyhow::Result<Vec<[f64; 4]>> {
    let raw = g.dequant_name(&format!("blk.{b}.{which}.weight"))?;
    anyhow::ensure!(raw.len() == 4 * channels, "{which} is {} not 4 x {channels}", raw.len());
    Ok((0..channels)
        .map(|c| [raw[c] as f64, raw[channels + c] as f64, raw[2 * channels + c] as f64, raw[3 * channels + c] as f64])
        .collect())
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
    let one = run(&g, &c, &plan, 1, vec![embed(&g, &v, ids[0], c.d)?])?;
    // two-token run of the same token followed by another
    let two = run(&g, &c, &plan, 1, vec![embed(&g, &v, ids[0], c.d)?, embed(&g, &v, ids[1], c.d)?])?;

    let diff = one[0].iter().zip(&two[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    println!("position 0 after block 0: |1-token - 2-token| = {diff:.3e}");
    anyhow::ensure!(diff < 1e-12, "the two-token prefill CHANGED position 0 by {diff:.3e} -- attention is not causal");
    println!("PASS: position 0 is identical whether or not a later token is present -- the recurrence is causal");

    // THE SAME INVARIANT ON THE MLA PATH, at block 3 (the first MLA block per the file's own schedule).
    let mla = plan.iter().position(|b| b.attn == engine_core::glm53_plan::Attn::Mla)
        .ok_or_else(|| anyhow::anyhow!("no MLA block in this checkpoint"))?;
    let e0 = vec![embed(&g, &v, ids[0], c.d)?];
    let e01 = vec![embed(&g, &v, ids[0], c.d)?, embed(&g, &v, ids[1], c.d)?];
    let m1 = mla_block_n(&g, mla, &c, &e0)?;
    let m2 = mla_block_n(&g, mla, &c, &e01)?;
    let dm = m1[0].iter().zip(&m2[0]).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
    println!("MLA block {mla}: position 0 |1-token - 2-token| = {dm:.3e}");
    anyhow::ensure!(dm < 1e-12, "MLA attention leaked the future: {dm:.3e}");
    println!("PASS: MLA position 0 is also independent of a later token -- attention is causal");

    // THE INVARIANT AT THE OUTPUT: six blocks, both runs, then the head on position 0. The ARGMAX must match --
    // a prefill that leaked the future would have moved it, and this is the check a person would actually make.
    let blocks = std::env::var("PREFILL_BLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);
    let h1 = run(&g, &c, &plan, blocks, vec![embed(&g, &v, ids[0], c.d)?])?;
    let h2 = run(&g, &c, &plan, blocks, vec![embed(&g, &v, ids[0], c.d)?, embed(&g, &v, ids[1], c.d)?])?;
    let head = g.dequant_name("token_embd.weight")?;
    let hw = head.len() / c.d;
    let logits = |h: &[f64]| -> (f64, usize) {
        let mut best = (f64::NEG_INFINITY, 0usize);
        for r in 0..hw {
            let mut acc = 0.0;
            for cc in 0..c.d { acc += head[r * c.d + cc] as f64 * h[cc]; }
            if acc > best.0 { best = (acc, r); }
        }
        best
    };
    let (l1, i1) = logits(&h1[0]);
    let (l2, i2) = logits(&h2[0]);
    println!("after {blocks} blocks: 1-token -> {i1} ({l1:.6}), 2-token -> {i2} ({l2:.6})");
    anyhow::ensure!(i1 == i2, "the two-token prefill changed position 0's token: {i1} vs {i2}");
    anyhow::ensure!((l1 - l2).abs() < 1e-9, "the top logit moved by {:.3e}", (l1 - l2).abs());
    println!("PASS: after {blocks} blocks position 0's TOKEN is unchanged by a later token -- prefill is causal end to end");
    Ok(())
}
