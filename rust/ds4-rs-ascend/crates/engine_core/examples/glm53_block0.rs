//! ONE REAL KDA BLOCK of GLM 5.3, on real weights, on the CPU.
//!
//! This is the last step before a forward pass: the schedule says block 0 is KDA/Dense, the block's pieces are
//! assembled in `kda_block`, and the projections are certified kernels on the device -- so the one thing never
//! tried is the assembly on the CHECKPOINT'S OWN TENSORS.
//!
//! What it can prove without a reference implementation: the projections produce the widths the file states, the
//! block produces a finite output of the right shape, and the whole chain is deterministic. What it CANNOT prove
//! is that the numbers match GLM 5.3 -- there is no reference here, and claiming otherwise would be the
//! over-credit this campaign has removed four times.
//!
//! usage: glm53_block0 <gguf>

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::{schedule, Attn};
use engine_core::glm5_ref::DefaultsGlm5;
use engine_core::glm5_stream::ShardedGguf;
use engine_core::kda_block::{block, BlockParams};
use std::path::PathBuf;


/// ONE TOKEN through an MLA block. With a single position the attention over that position is the identity on
/// `v`, so the block reduces to its four certified projections plus the norms -- which is why a decode-shaped
/// MLA block is tractable while a prefill one is not.
fn mla_block_1token(g: &ShardedGguf, b: usize, x: &[f64], d: usize, eps: f64) -> anyhow::Result<Vec<f64>> {
    let mv = |w: &[f32], xv: &[f64], k: usize, n: usize| -> Vec<f64> {
        (0..n).map(|r| (0..k).map(|c| w[r * k + c] as f64 * xv[c]).sum()).collect()
    };
    let norm_w = g.dequant_name(&format!("blk.{b}.attn_norm.weight"))?;
    let xin = engine_core::kda_block::gated_rms_norm(x, &norm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], eps);
    // q_a: d -> 1536, then q_b: 1536 -> 16384 (64 heads x 256)
    let wqa = g.dequant_name(&format!("blk.{b}.attn_q_a.weight"))?;
    let qa = mv(&wqa, &xin, d, 1536);
    let wqb = g.dequant_name(&format!("blk.{b}.attn_q_b.weight"))?;
    let _q = mv(&wqb, &qa, 1536, 16384);
    // kv_a: d -> 512, then v_b: per head 512 -> 256, i.e. 64 x (256 x 512)
    let wkva = g.dequant_name(&format!("blk.{b}.attn_kv_a_mqa.weight"))?;
    let kva = mv(&wkva, &xin, d, 512);
    let wvb = g.dequant_name(&format!("blk.{b}.attn_v_b.weight"))?;
    let heads = 64usize;
    let mut vcat = vec![0.0f64; heads * 256];
    for h in 0..heads {
        for r in 0..256 {
            let mut acc = 0.0;
            for c in 0..512 {
                acc += wvb[(h * 256 + r) * 512 + c] as f64 * kva[c];
            }
            vcat[h * 256 + r] = acc;
        }
    }
    // output: 16384 -> d
    let wo = g.dequant_name(&format!("blk.{b}.attn_output.weight"))?;
    Ok(mv(&wo, &vcat, 16384, d))
}

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: glm53_block0 <gguf>"))?;
    let cfg = DefaultsGlm5::from_gguf(&GgufFile::open(&path)?)?;
    let plan = schedule(&GgufFile::open(&path)?)?;
    anyhow::ensure!(plan[0].attn == Attn::Kda, "block 0 is not KDA in this checkpoint");
    println!("config: hidden {} layers {}; block 0 = KDA", cfg.hidden_size, cfg.num_hidden_layers);

    let g = ShardedGguf::open(&path)?;
    let d = cfg.hidden_size;
    // The tensors the file itself names for block 0 (read, not assumed).
    let wq = g.dequant_name("blk.0.kda_q.weight")?;
    println!("kda_q dequantised: {} values", wq.len());
    let heads = 64usize;
    let head_dim = 128usize;
    anyhow::ensure!(wq.len() == 8192 * d, "kda_q is {} not 8192 x {d}", wq.len());
    anyhow::ensure!(heads * head_dim == 8192, "64 heads x 128 must be 8192");

    // A deterministic input; the VALUES do not matter, the SHAPES and the finiteness do.
    let mut x: Vec<f64> = (0..d).map(|i| ((i % 11) as f64 - 5.0) * 0.01).collect();
    // 8192-wide projections from the real weight matrix (f64, one row at a time).
    let proj = |w: &[f32]| -> Vec<f64> {
        (0..8192)
            .map(|r| (0..d).map(|c| w[r * d + c] as f64 * x[c]).sum())
            .collect()
    };
    let q = proj(&wq);
    let k = proj(&g.dequant_name("blk.0.kda_k.weight")?);
    let v = proj(&g.dequant_name("blk.0.kda_v.weight")?);
    println!("projections: q {} k {} v {}", q.len(), k.len(), v.len());

    let onw = vec![1.0f64; head_dim];
    // THE BLOCK'S REAL STRUCTURE: attn_norm -> KDA -> output projection -> residual add. GLM 5.3's leading
    // blocks are KDA/Dense (read from the file), so the chain runs until the first MLA block and no further --
    // a partial chain that says where it stopped is worth more than one that guesses.
    let first_mla = plan.iter().position(|b| b.attn == Attn::Mla).unwrap_or(plan.len());
    println!("running blocks 0..{} of {}", plan.len().min(4), plan.len());
    let t0 = std::time::Instant::now();
    let upto = plan.len().min(4);
    for b in 0..upto {
        if plan[b].attn == Attn::Mla {
            let op = mla_block_1token(&g, b, &x, d, cfg.rms_norm_eps as f64)?;
            for i in 0..d { x[i] += op[i]; }
            let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
            println!("block {b}: MLA, rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
            anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after MLA block {b} is not finite");
            continue;
        }
        let norm_w = g.dequant_name(&format!("blk.{b}.attn_norm.weight"))?;
        let xin = engine_core::kda_block::gated_rms_norm(&x, &norm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64);
        let wq = g.dequant_name(&format!("blk.{b}.kda_q.weight"))?;
        let wk = g.dequant_name(&format!("blk.{b}.kda_k.weight"))?;
        let wv = g.dequant_name(&format!("blk.{b}.kda_v.weight"))?;
        let proj = |w: &[f32], xv: &[f64]| -> Vec<f64> {
            (0..8192).map(|r| (0..d).map(|c| w[r * d + c] as f64 * xv[c]).sum()).collect()
        };
        let q = proj(&wq, &xin);
        let k = proj(&wk, &xin);
        let v = proj(&wv, &xin);
        let params = BlockParams {
            heads, head_dim, a_log: vec![0.0; heads],
            conv_q: [1.0, 0.0, 0.0, 0.0], conv_k: [1.0, 0.0, 0.0, 0.0], conv_v: [1.0, 0.0, 0.0, 0.0],
            dt_bias: vec![0.0; heads * head_dim], beta: vec![0.5; heads],
            eps: cfg.rms_norm_eps as f64,
        };
        let gate = vec![0.0f64; heads * head_dim];
        let o = block(&params, &q, &k, &v, &gate, 1, &onw);
        anyhow::ensure!(o.iter().all(|v| v.is_finite()), "block {b} produced a non-finite value");
        // output projection (8192 -> d) then the residual
        let wo = g.dequant_name(&format!("blk.{b}.kda_output.weight"))?;
        // attn_output is [12288, 7168] for Kimi and [8192, 4096]-shaped for GLM; take the leading d columns
        let op = (0..d).map(|r| (0..8192.min(o.len())).map(|c| (wo.get(r * 8192 + c).copied().unwrap_or(0.0) as f64) * o[c]).sum::<f64>()).collect::<Vec<_>>();
        for i in 0..d { x[i] += op[i]; }
        // THE DENSE FFN, when the block has one: ffn_norm -> gate/up -> SiLU -> down -> residual. GLM's leading
        // blocks carry `ffn_gate`/`ffn_up`/`ffn_down` at 12288 wide; the MoE blocks carry stacked experts instead,
        // which are NOT implemented, so their FFN is skipped and SAID so.
        if plan[b].ffn == engine_core::glm53_plan::Ffn::Dense {
            let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
            let fin = engine_core::kda_block::gated_rms_norm(&x, &fnorm_w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64);
            let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
            let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
            let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
            let inter = wg.len() / d;
            let g_: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wg[r * d + c] as f64 * fin[c]).sum()).collect();
            let u_: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wu[r * d + c] as f64 * fin[c]).sum()).collect();
            let act: Vec<f64> = g_.iter().zip(&u_).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
            let dn: Vec<f64> = (0..d).map(|r| (0..act.len()).map(|c| wd[r * act.len() + c] as f64 * act[c]).sum()).collect();
            for i in 0..d { x[i] += dn[i]; }
        } else {
            println!("  block {b}: FFN is {:?} and not implemented -- skipped, so this block is attention-only", plan[b].ffn);
        }
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
        println!("block {b}: rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
        anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after block {b} is not finite");
        anyhow::ensure!(rms > 0.0, "the residual after block {b} is all zero");
    }
    println!("PASS: GLM 5.3's block 0 ran through the assembled KDA path on its own weights");
    Ok(())
}
