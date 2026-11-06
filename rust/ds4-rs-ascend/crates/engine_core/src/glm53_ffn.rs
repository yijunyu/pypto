//! The MoE feed-forward, in the LIBRARY so both drivers call one implementation.
//!
//! It lived in an example until iteration 213, which is how the prefill driver came to have NO MoE at all while the
//! decode driver had one: two drivers, one author, and the shared piece was copied rather than shared. `moe_route` was
//! extracted for the same reason and it is the same lesson twice.

use crate::gguf::GgufFile;
use crate::glm5_stream::ShardedGguf;
use crate::kda_block::gated_rms_norm;

/// Read `expert_weights_scale` from the checkpoint, REFUSING rather than defaulting.
///
/// It was hardcoded 2.5 here until iteration 220, transcribed from a `grep` of the metadata. A transcription is a copy;
/// this is a read, and a checkpoint without the key is a loud error rather than a silently-1.0 expert sum.
pub fn expert_weights_scale(g: &GgufFile) -> anyhow::Result<f64> {
    for (k, v) in &g.meta {
        if k.ends_with("expert_weights_scale") {
            return match v {
                crate::gguf::MetaValue::F32(x) => Ok(*x as f64),
                crate::gguf::MetaValue::F64(x) => Ok(*x),
                other => anyhow::bail!("expert_weights_scale is {other:?}, not a float"),
            };
        }
    }
    anyhow::bail!("no expert_weights_scale in this checkpoint: the MoE sum would be weighted wrongly and silently")
}

pub fn moe_ffn(g: &ShardedGguf, scale: f64, b: usize, x: &[f64], d: usize, eps: f64) -> anyhow::Result<(Vec<f64>, usize)> {
    // `scale` now comes from `expert_weights_scale(&GgufFile)` at the driver, so the 2.5 transcription is gone.
    let mv = |w: &[f32], xv: &[f64], k: usize, n: usize, off: usize| -> Vec<f64> {
        (0..n).map(|r| (0..k).map(|c| w[off + r * k + c] as f64 * xv[c]).sum()).collect()
    };
    let fnorm = g.dequant_name(&format!("blk.{b}.ffn_norm.weight"))?;
    let fin = gated_rms_norm(x, &fnorm.iter().map(|v| *v as f64).collect::<Vec<_>>(), &vec![1.0f64; d], eps);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, MetaValue};

    fn file_with(meta: Vec<(String, MetaValue)>) -> GgufFile {
        GgufFile {
            version: 3,
            n_tensors: 0,
            n_meta_kv: meta.len() as u64,
            meta,
            tensors: Vec::new(),
            tensor_table_offset: 0,
            tensor_data_offset: 0,
            file_size: 0,
        }
    }

    /// BUG (iteration 180): the expert scale was a transcribed 2.5. The reader must FIND it by suffix.
    #[test]
    fn the_expert_scale_is_read_by_suffix() {
        let g = file_with(vec![(
            "glm5-next.expert_weights_scale".to_string(),
            MetaValue::F32(2.5),
        )]);
        assert!((expert_weights_scale(&g).unwrap() - 2.5).abs() < 1e-6);
    }

    /// And a checkpoint WITHOUT the key must REFUSE rather than default, which was the actual defect: a missing
    /// scale weights the expert sum wrongly and silently.
    #[test]
    fn a_missing_expert_scale_is_refused() {
        let g = file_with(vec![(
            "glm5-next.expert_count".to_string(),
            MetaValue::U32(288),
        )]);
        let err = expert_weights_scale(&g).expect_err("a missing scale must not default");
        assert!(format!("{err}").contains("expert_weights_scale"), "got: {err}");
    }

    /// A non-float value is refused rather than cast.
    #[test]
    fn a_non_float_expert_scale_is_refused() {
        let g = file_with(vec![(
            "glm5-next.expert_weights_scale".to_string(),
            MetaValue::U32(2),
        )]);
        let err = expert_weights_scale(&g).expect_err("an integer scale must not be cast");
        assert!(format!("{err}").contains("not a float"), "got: {err}");
    }
}
