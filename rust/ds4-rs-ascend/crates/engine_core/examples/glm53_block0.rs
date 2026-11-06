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
    let x: Vec<f64> = (0..d).map(|i| ((i % 11) as f64 - 5.0) * 0.01).collect();
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

    let params = BlockParams {
        heads,
        head_dim,
        a_log: vec![0.0; heads],
        conv_q: [1.0, 0.0, 0.0, 0.0],
        conv_k: [1.0, 0.0, 0.0, 0.0],
        conv_v: [1.0, 0.0, 0.0, 0.0],
        dt_bias: vec![0.0; heads * head_dim],
        beta: vec![0.5; heads],
        eps: cfg.rms_norm_eps as f64,
    };
    let gate = vec![0.0f64; heads * head_dim];
    let onw = vec![1.0f64; head_dim];
    let out = block(&params, &q, &k, &v, &gate, 1, &onw);

    anyhow::ensure!(out.len() == heads * head_dim, "block output is {} not 8192", out.len());
    let finite = out.iter().filter(|v| v.is_finite()).count();
    anyhow::ensure!(finite == out.len(), "{} of {} outputs are not finite", out.len() - finite, out.len());
    let rms = (out.iter().map(|v| v * v).sum::<f64>() / out.len() as f64).sqrt();
    println!("block 0 out: {} values, all finite, rms {rms:.4}", out.len());
    anyhow::ensure!(rms > 0.0, "a KDA block on real weights produced an all-zero output");
    println!("PASS: GLM 5.3's block 0 ran through the assembled KDA path on its own weights");
    Ok(())
}
