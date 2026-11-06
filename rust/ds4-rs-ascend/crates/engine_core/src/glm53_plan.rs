//! Which attention — and which feed-forward — each GLM 5.3 / Kimi K3 block uses, read FROM THE FILE.
//!
//! The registry's note says GLM 5.3 is a hybrid of 34 delta-net blocks and 12 MLA blocks, and Kimi K3 of 69 and
//! 24. Those numbers came from somewhere; a forward pass cannot use them unless the source is the checkpoint,
//! because a schedule inferred from a note is a guess with a citation.
//!
//! The checkpoint states it TWICE and both are readable without the weights:
//!   * `blk.<n>.kda_*` tensors exist iff block `n` is a gated delta net;
//!   * `blk.<n>.attn_q_a` exists iff it is MLA.
//! So the schedule is a scan of the tensor table, which this module does.

use crate::gguf::GgufFile;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attn {
    /// The gated delta-net / KDA path (`kda_q`, `kda_k`, `kda_v`, convs, gates).
    Kda,
    /// Multi-head latent attention (`attn_q_a`, `attn_kv_a`, ...).
    Mla,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ffn {
    Dense,
    Moe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockPlan {
    pub attn: Attn,
    pub ffn: Ffn,
}

/// One entry per block, in index order.
pub fn schedule(g: &GgufFile) -> anyhow::Result<Vec<BlockPlan>> {
    let n = g
        .tensors
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|b| b.parse::<usize>().ok())
        })
        .max()
        .ok_or_else(|| anyhow::anyhow!("no blk.<n>.* tensors: is this a block-stacked model?"))?
        + 1;

    let mut out = Vec::with_capacity(n);
    for b in 0..n {
        let has = |suffix: &str| g.tensors.iter().any(|t| t.name == format!("blk.{b}.{suffix}"));
        let attn = if has("kda_q.weight") {
            Attn::Kda
        } else if has("attn_q_a.weight") {
            Attn::Mla
        } else {
            anyhow::bail!("block {b} has neither kda_q nor attn_q_a: the schedule cannot be read from names");
        };
        // MoE iff the block carries an expert table; the leading dense blocks do not.
        // THE STACKED-EXPERT NAMES, read from the file rather than guessed: GLM 5.3 spells them
        // `blk.<n>.ffn_gate_exps.weight`, `ffn_up_exps`, `ffn_down_exps` (plus `ffn_*_shexp` for the shared
        // one and `exp_probs_b.bias` for the router). My first version looked for `ffn.experts.` and found
        // ZERO MoE blocks in a model whose registry note says 43 of them.
        let ffn = if g.tensors.iter().any(|t| t.name == format!("blk.{b}.ffn_gate_exps.weight")) {
            Ffn::Moe
        } else {
            Ffn::Dense
        };
        out.push(BlockPlan { attn, ffn });
    }
    Ok(out)
}

/// A compact census, for reporting and for comparing two checkpoints.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub blocks: usize,
    pub kda: usize,
    pub mla: usize,
    pub moe: usize,
    pub dense: usize,
}

impl Census {
    pub fn of(s: &[BlockPlan]) -> Census {
        let mut c = Census { blocks: s.len(), ..Default::default() };
        for b in s {
            match b.attn {
                Attn::Kda => c.kda += 1,
                Attn::Mla => c.mla += 1,
            }
            match b.ffn {
                Ffn::Moe => c.moe += 1,
                Ffn::Dense => c.dense += 1,
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{GgmlType, TensorInfo};

    fn t(name: &str) -> TensorInfo {
        TensorInfo { name: name.to_string(), dims: vec![1], ttype: GgmlType::F32, offset: 0 }
    }

    /// The scan is a function of the NAMES, so a synthetic file proves it without a checkpoint.
    #[test]
    fn the_schedule_is_read_from_tensor_names() {
        let g = GgufFile {
            version: 3,
            n_tensors: 0,
            n_meta_kv: 0,
            meta: vec![],
            tensors: vec![
                t("blk.0.kda_q.weight"),
                t("blk.0.ffn_gate_exps.weight"),
                t("blk.1.attn_q_a.weight"),
                t("blk.1.ffn.dense.w1.weight"),
            ],
            tensor_table_offset: 0,
            tensor_data_offset: 0,
            file_size: 0,
        };
        let s = schedule(&g).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], BlockPlan { attn: Attn::Kda, ffn: Ffn::Moe });
        assert_eq!(s[1], BlockPlan { attn: Attn::Mla, ffn: Ffn::Dense });
        assert_eq!(Census::of(&s), Census { blocks: 2, kda: 1, mla: 1, moe: 1, dense: 1 });
    }

    /// A block with neither attention is a REFUSAL, not a default: guessing would silently produce a schedule
    /// that disagrees with the checkpoint.
    #[test]
    fn a_block_with_no_attention_is_refused() {
        let g = GgufFile {
            version: 3, n_tensors: 0, n_meta_kv: 0, meta: vec![],
            tensors: vec![t("blk.0.ffn.dense.w1.weight")],
            tensor_table_offset: 0, tensor_data_offset: 0, file_size: 0,
        };
        let err = schedule(&g).expect_err("no attention tensor");
        assert!(format!("{err}").contains("neither kda_q nor attn_q_a"), "got: {err}");
    }
}
