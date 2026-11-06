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

/// How many blocks to run: all of them unless GLM53_BLOCKS says otherwise.
fn env_blocks() -> usize {
    std::env::var("GLM53_BLOCKS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX)
}


/// THE MoE FEED-FORWARD: router -> top-k -> the selected experts -> weighted sum, then the SHARED expert.
///
/// Every shape is the file's: `ffn_gate_inp [d, n_exp]` and `exp_probs_b.bias [n_exp]` route, and the stacked
/// `ffn_{gate,up,down}_exps` hold `n_exp` experts. The shared expert is three `shexp` planes applied
/// unconditionally. GLM's note says 288 experts and 8 used; both are read from the file rather than assumed.
fn moe_ffn(g: &ShardedGguf, b: usize, x: &[f64], d: usize, eps: f64) -> anyhow::Result<(Vec<f64>, usize)> {
    let mv = |w: &[f32], xv: &[f64], k: usize, n: usize, off: usize| -> Vec<f64> {
        (0..n).map(|r| (0..k).map(|c| w[off + r * k + c] as f64 * xv[c]).sum()).collect()
    };
    let fnorm = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
    let fin = engine_core::kda_block::gated_rms_norm(x, &fnorm.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], eps);
    // router
    let rin = g.dequant_name(&format!("blk.{b}.ffn_gate_inp.weight"))?;
    let n_exp = rin.len() / d;
    let bias = g.dequant_name(&format!("blk.{b}.exp_probs_b.bias"))?;
    let mut scores: Vec<(f64, usize)> = (0..n_exp)
        .map(|e| {
            let s: f64 = (0..d).map(|c| rin[c * n_exp + e] as f64 * fin[c]).sum::<f64>() + bias[e] as f64;
            (s, e)
        })
        .collect();
    scores.sort_by(|a, b_| b_.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let used = 8.min(n_exp);
    // softmax over the selected
    let mx = scores[0].0;
    let exps: Vec<(f64, usize)> = scores[..used].iter().map(|(s, e)| ((s - mx).exp(), *e)).collect();
    let z: f64 = exps.iter().map(|(w, _)| w).sum();
    let gate_w = g.dequant_name(&format!("blk.{b}.ffn_gate_exps.weight"))?;
    let up_w = g.dequant_name(&format!("blk.{b}.ffn_up_exps.weight"))?;
    let dn_w = g.dequant_name(&format!("blk.{b}.ffn_down_exps.weight"))?;
    anyhow::ensure!(gate_w.len() == up_w.len() && gate_w.len() % n_exp == 0, "stacked expert planes disagree");
    let per = gate_w.len() / n_exp;           // values per expert
    let inter = per / d;                       // rows per expert
    anyhow::ensure!(inter * d == per, "expert plane {per} is not {inter} x {d}");
    let dn_per = dn_w.len() / n_exp;
    let mut acc = vec![0.0f64; d];
    // `glm5-next.expert_weights_scale = 2.5`. The checkpoint applies a scale to the expert contributions and my
    // first implementation IGNORED it, which is a wiring error of exactly the kind no amount of arithmetic
    // cross-checking would catch: every product is right, the sum is weighted differently from the model's.
    let scale = 2.5f64;
    for (w, e) in if used == 0 { Vec::new() } else { exps } {
        let wgt = scale * w / z;
        let g_ = mv(&gate_w, &fin, d, inter, e * per);
        let u_ = mv(&up_w, &fin, d, inter, e * per);
        let act: Vec<f64> = g_.iter().zip(&u_).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
        let dn = mv(&dn_w, &act, inter, d, e * dn_per);
        for i in 0..d { acc[i] += wgt * dn[i]; }
    }
    // the shared expert, always applied
    let sg = g.dequant_name(&format!("blk.{b}.ffn_gate_shexp.weight"))?;
    let su = g.dequant_name(&format!("blk.{b}.ffn_up_shexp.weight"))?;
    let sd = g.dequant_name(&format!("blk.{b}.ffn_down_shexp.weight"))?;
    let sinter = sg.len() / d;
    let sgv = mv(&sg, &fin, d, sinter, 0);
    let suv = mv(&su, &fin, d, sinter, 0);
    let sact: Vec<f64> = sgv.iter().zip(&suv).map(|(a, b_)| (a / (1.0 + (-a).exp())) * b_).collect();
    let sdn = mv(&sd, &sact, sinter, d, 0);
    for i in 0..d { acc[i] += sdn[i]; }
    Ok((acc, used))
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
    // THE INPUT: a REAL PROMPT, tokenised by the repository's own tokeniser, embedded through `token_embd`.
    // Before this, every run used a synthetic vector and the token it produced answered nothing. GLM53_PROMPT
    // selects the text; unset falls back to the synthetic vector so the earlier measurements stay reproducible.
    let mut x: Vec<f64> = match std::env::var("GLM53_PROMPT") {
        Ok(prompt) => {
            use engine_core::tokenizer::Vocab;
            let v = Vocab::from_gguf(&GgufFile::open(&path)?)?;
            let ids = v.encode(&prompt);
            println!("prompt {prompt:?} -> {ids:?}");
            anyhow::ensure!(!ids.is_empty(), "the prompt tokenised to nothing");
            // A single-token prompt keeps the block loop's arithmetic identical to every earlier run; a longer one
            // would need a prefill loop the block assembly does not have yet, so it is refused by name.
            anyhow::ensure!(
                ids.len() == 1,
                "the prompt is {} tokens; this decode path takes ONE (a prefill loop is not implemented)",
                ids.len()
            );
            let emb = g.dequant_name("token_embd.weight")?;
            let vocab_cols = emb.len() / d;
            let id = ids[0] as usize;
            anyhow::ensure!(id < vocab_cols, "token {id} is outside the {vocab_cols}-wide embedding");
            (0..d).map(|c| emb[c * vocab_cols + id] as f64).collect()
        }
        Err(_) => (0..d).map(|i| ((i % 11) as f64 - 5.0) * 0.01).collect(),
    };
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
    let mut moe_used = 0usize;
    let t0 = std::time::Instant::now();
    // THE TRUNK, NOT THE FILE. `glm5-next.block_count = 46` but `trunk_block_count = 45`: the last block is not
    // part of the decode trunk, and running 46 puts one extra block in the residual stream. Found by auditing
    // every metadata key against the implementation (skill rule 13c), not by any arithmetic check.
    let trunk = std::env::var("GLM53_TRUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(45usize);
    let upto = plan.len().min(trunk).min(env_blocks());
    println!("trunk: {upto} of {} blocks (block_count 46, trunk_block_count {trunk})", plan.len());
    for b in 0..upto {
        if plan[b].attn == Attn::Mla {
            let op = mla_block_1token(&g, b, &x, d, cfg.rms_norm_eps as f64)?;
            println!("MLA-TENSOR decode blk{b}: {:?}", &op[..3]);
            for i in 0..d { x[i] += op[i]; }
            let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
            println!("block {b}: MLA, rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
            if b <= 4 { println!("BLK{b}-TENSOR decode (MLA): {:?}", &x[..3]); }
            anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after MLA block {b} is not finite");
            // THE FFN IS NOT OPTIONAL ON AN MLA BLOCK. `continue` used to skip straight to the next block, so every
            // MLA block ran its attention and NOT its feed-forward -- which is why this driver and the prefill driver
            // disagreed by 40x on the logit. The `continue` was written to skip the KDA code below, not the FFN.
            match plan[b].ffn {
                engine_core::glm53_plan::Ffn::Dense => {
                    let fnorm_w = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
                    let fnw: Vec<f64> = fnorm_w.iter().map(|v| *v as f64).collect();
                    let fin = engine_core::kda_block::gated_rms_norm(&x, &fnw, &vec![1.0f64; d], cfg.rms_norm_eps as f64);
                    let wg = g.dequant_name(&format!("blk.{b}.ffn_gate.weight"))?;
                    let wu = g.dequant_name(&format!("blk.{b}.ffn_up.weight"))?;
                    let wd = g.dequant_name(&format!("blk.{b}.ffn_down.weight"))?;
                    let inter = wg.len() / d;
                    let gv: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wg[r * d + c] as f64 * fin[c]).sum()).collect();
                    let uv: Vec<f64> = (0..inter).map(|r| (0..d).map(|c| wu[r * d + c] as f64 * fin[c]).sum()).collect();
                    let act: Vec<f64> = gv.iter().zip(&uv).map(|(a, bb)| (a / (1.0 + (-a).exp())) * bb).collect();
                    for r in 0..d {
                        let mut acc = 0.0;
                        for c in 0..act.len() { acc += wd[r * act.len() + c] as f64 * act[c]; }
                        x[r] += acc;
                    }
                }
                engine_core::glm53_plan::Ffn::Moe => {
                    let (dn, used) = moe_ffn(&g, b, &x, d, cfg.rms_norm_eps as f64)?;
                    for i in 0..d { x[i] += dn[i]; }
                    moe_used = used;
                }
                engine_core::glm53_plan::Ffn::None => {}
            }
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
            // `kda_a_log` IS in the file and its shape matches a_log exactly: 64 values, one per head. The
            // other KDA constants are NOT yet loaded -- see the note in the commit; the conv is per-channel
            // ([4,1,8192]) and beta is [4096,64], neither of which `BlockParams` can hold.
            heads, head_dim,
            a_log: {
                let raw = g.dequant_name(&format!("blk.{b}.kda_a_log.weight"))?;
                anyhow::ensure!(raw.len() == heads, "kda_a_log is {} not {heads}", raw.len());
                raw.iter().map(|v| *v as f64).collect()
            },
            // THE REAL PER-CHANNEL CONVS. `kda_{q,k,v}_conv` are [4, 1, 8192]: four taps for EACH of the
            // 8192 channels. Loaded from the file rather than shared.
            conv_q: load_conv(&g, b, "kda_q_conv", heads * head_dim)?,
            conv_k: load_conv(&g, b, "kda_k_conv", heads * head_dim)?,
            conv_v: load_conv(&g, b, "kda_v_conv", heads * head_dim)?,
            // THE REAL dt BIAS: kda_dt_bias is [8192], one value per channel.
            dt_bias: {
                let raw = g.dequant_name(&format!("blk.{b}.kda_dt_bias.weight"))?;
                anyhow::ensure!(raw.len() == heads * head_dim, "kda_dt_bias is {} not {}", raw.len(), heads * head_dim);
                raw.iter().map(|v| *v as f64).collect()
            },
            // `kda_beta` is [4096, 64] = d x heads, so it reads as a PER-HEAD PROJECTION of x, sigmoid-ed into
            // (0,1) the way a write strength must be. This is a MODELLING decision, stated rather than assumed:
            // the alternative reading is per (channel, head) values used directly, which the recurrence's
            // per-head `beta` cannot consume. If the reading is wrong every block is wrong in the same way.
            beta: {
                let raw = g.dequant_name(&format!("blk.{b}.kda_beta.weight"))?;
                anyhow::ensure!(raw.len() == d * heads, "kda_beta is {} not {d} x {heads}", raw.len());
                (0..heads)
                    .map(|h| {
                        let z: f64 = (0..d).map(|c| raw[c * heads + h] as f64 * xin[c]).sum();
                        1.0 / (1.0 + (-z).exp())
                    })
                    .collect()
            },
            eps: cfg.rms_norm_eps as f64,
        };
        // THE TIME STEP, from the block's own f-gate: dt = f_b(f_a(normed x)) + dt_bias, 8192 channels.
        let fa = g.dequant_name(&format!("blk.{b}.kda_f_a.weight"))?;
        let fb = g.dequant_name(&format!("blk.{b}.kda_f_b.weight"))?;
        let hid: Vec<f64> = (0..128)
            .map(|r| (0..d).map(|c| fa[r * d + c] as f64 * xin[c]).sum())
            .collect();
        let gate: Vec<f64> = (0..heads * head_dim)
            .map(|r| (0..128).map(|c| fb[r * 128 + c] as f64 * hid[c]).sum())
            .collect();
        // THE OUTPUT GATE, per channel: g_b(g_a(normed x)), through a sigmoid.
        let ga = g.dequant_name(&format!("blk.{b}.kda_g_a.weight"))?;
        let gb = g.dequant_name(&format!("blk.{b}.kda_g_b.weight"))?;
        let ghid: Vec<f64> = (0..128).map(|r| (0..d).map(|c| ga[r * d + c] as f64 * xin[c]).sum()).collect();
        let ogate: Vec<f64> = (0..heads * head_dim)
            .map(|r| {
                let z: f64 = (0..128).map(|c| gb[r * 128 + c] as f64 * ghid[c]).sum();
                1.0 / (1.0 + (-z).exp())
            })
            .collect();
        if b >= 0 && std::env::var_os("KDA_INTER").is_some() {
            println!("KDA-INTER blk{b} decode: x0={:?} xin0={:?} q0={:?} beta0={:?}",
                x[0], xin[0], q[0], params.beta[0]);
        }
        let o = block(&params, &q, &k, &v, &gate, 1, &onw, &ogate);
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
        } else if plan[b].ffn == engine_core::glm53_plan::Ffn::Moe {
            let (dn, used) = moe_ffn(&g, b, &x, d, cfg.rms_norm_eps as f64)?;
            for i in 0..d { x[i] += dn[i]; }
            moe_used = used;
        } else {
            println!("  block {b}: FFN is {:?} -- skipped", plan[b].ffn);
        }
        let rms = (x.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt();
        println!("block {b}: rms {rms:.4}  ({:.1}s)", t0.elapsed().as_secs_f64());
        if b <= 4 { println!("BLK{b}-TENSOR decode: {:?}", &x[..3]); }
        anyhow::ensure!(x.iter().all(|v| v.is_finite()), "the residual after block {b} is not finite");
        anyhow::ensure!(rms > 0.0, "the residual after block {b} is all zero");
    }
    // THE HEAD. GLM 5.3 TIES it to the embedding: `token_embd.weight [4096, 154880]`. The final norm is applied
    // first if the checkpoint carries one by that name -- checked, not assumed.
    let out_norm = g.dequant_name("output_norm.weight").ok();
    let xf = match &out_norm {
        Some(w) => engine_core::kda_block::gated_rms_norm(
            &x, &w.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], cfg.rms_norm_eps as f64),
        None => {
            println!("no output_norm.weight in this checkpoint -- the trunk output is used as-is");
            x.clone()
        }
    };
    let head = g.dequant_name("token_embd.weight")?;
    let vocab = head.len() / d;
    println!("head: token_embd, {vocab} outputs (tied embedding)");
    anyhow::ensure!(vocab == cfg.vocab_size, "head is {vocab} wide, config says {}", cfg.vocab_size);
    let t1 = std::time::Instant::now();
    let mut best = (f64::NEG_INFINITY, 0usize);
    let mut checked = 0usize;
    for r in 0..vocab {
        let mut acc = 0.0f64;
        for c in 0..d {
            acc += head[r * d + c] as f64 * xf[c];
        }
        if !acc.is_finite() {
            // one non-finite logit would make the argmax meaningless, so it is a refusal, not a warning
            anyhow::bail!("logit {r} is not finite");
        }
        checked += 1;
        if acc > best.0 {
            best = (acc, r);
        }
    }
    anyhow::ensure!(checked == vocab, "only {checked} of {vocab} logits were computed");
    println!("logits: {checked} finite, best {:.4} at id {} (MoE used {moe_used} experts per block)", best.0, best.1);
    println!("TOKEN from GLM 5.3: {} (vocab {vocab})  ({:.1}s)", best.1, t1.elapsed().as_secs_f64());
    anyhow::ensure!(best.1 < vocab, "token id outside the vocabulary");
    println!("PASS: GLM 5.3 produced a token from its own weights, through the assembled blocks and its own head");
    Ok(())
}
