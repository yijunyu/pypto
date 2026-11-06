//! THE ENGINE'S OWN layer path, driven from the w8a8 package.
//!
//! Everything here is the engine's code, not an example's: `AttnLayerWeights::from_layer_view` builds
//! the layer's weights from a `WeightSource` over the package, `LayerParams` comes from the checkpoint's
//! config (see the campaign's parameter table), and `decode_attn_layer_with` runs the prefix (hc
//! collapse, q/kv norm, rope, attention, GROUPED output projection, hc expand) and the suffix.
//!
//! That is what the hand-rolled `w8a8_layer0_token` example could not do: the `hc_*` mechanism, which
//! the engine implements and the example skipped.
//!
//! usage: w8a8_real_layer <package dir> <token id>

use engine_core::attn_dispatch::{
    CpuAttentionDispatcher,
    decode_attn_layer_with, AttnLayerWeights, AttnStepInputs, KvCacheView, LayerParams,
};
use engine_core::safetensors::Package;
use engine_core::w8a8_roles::{PackageViews, WeightSource};  // WeightSource supplies `layer()`
use std::path::PathBuf;

fn matvec(w: &[f32], x: &[f32], n: usize) -> Vec<f32> {
    let k = x.len();
    (0..n).map(|i| (0..k).map(|j| w[i * k + j] * x[j]).sum()).collect()
}
fn rms_norm(x: &[f32], g: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum();
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    x.iter().zip(g).map(|(v, g)| v * inv * g).collect()
}

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_real_layer <package dir> <token id>"))?;
    let token: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let pkg = Package::open(&dir)?;
    let views = PackageViews::new(&pkg);

    let d = 4096usize; let n_hc = 4usize; let heads = 64usize; let hd = 512usize; let lora_q = 1024usize;

    let params = LayerParams {
        layer_idx: 0,
        d_embd: d as u32,
        n_hc: n_hc as u32,
        n_head: heads as u32,
        head_dim: hd as u32,
        n_rot: 64,
        n_lora_q: lora_q as u32,
        n_lora_kv: hd as u32,
        hc_sinkhorn_iter: 20,
        hc_eps: 1e-6,
        rms_eps: 1e-6,
        rope_orig_ctx: 65536,
        rope_freq_base: 10000.0,
        rope_freq_scale: 1.0 / 16.0,
        rope_ext_factor: 0.0,
        rope_attn_factor: 1.0,
        compress_ratio: 0,
        n_out_group: 8,
    };

    let view = views.layer(0)?.into_owned();
    let layer = AttnLayerWeights::from_layer_view(&views, &view, params, 2048, f32::INFINITY, false)?;
    println!("AttnLayerWeights built from the package: layer 0");

    let emb = pkg.dequant_tensor("embed.weight").or_else(|_| pkg.f32_tensor("embed.weight"))?;
    let vocab = emb.len() / d;
    anyhow::ensure!(token < vocab, "token {token} past the vocabulary");
    let x = &emb[token * d..(token + 1) * d];

    // The two inputs the caller computes: the q lora-up product and the kv-down row.
    let gamma = pkg.f32_tensor("layers.0.attn_norm.weight")?;
    let n = rms_norm(x, &gamma, 1e-6);
    let wq_a = pkg.dequant_tensor("layers.0.attn.wq_a.weight")?;
    let qn = pkg.f32_tensor("layers.0.attn.q_norm.weight")?;
    let q_ln = rms_norm(&matvec(&wq_a, &n, lora_q), &qn, 1e-6);
    let wq_b = pkg.dequant_tensor("layers.0.attn.wq_b.weight")?;
    let q_heads = matvec(&wq_b, &q_ln, heads * hd);
    let wkv = pkg.dequant_tensor("layers.0.attn.wkv.weight")?;
    let kv_raw_row = matvec(&wkv, &n, hd);

    // HC slots: each holds a full-width copy of the embedding, so the residual starts as n_hc copies.
    let mut cur_hc = vec![0.0f32; n_hc * d];
    for h in 0..n_hc { cur_hc[h * d..(h + 1) * d].copy_from_slice(x); }

    let mut kvc = vec![0.0f32; hd];          // cap 1 row at pos 0
    let routed = vec![0.0f32; d];
    let step = AttnStepInputs {
        cur_hc: &cur_hc,
        kv_raw_row: &kv_raw_row,
        q_heads: &q_heads,
        pos: 0,
        kv_view: KvCacheView { raw: &mut kvc, raw_cap: 1, pos: 0 },
        n_raw: 0,
        raw_start: 0,
        routed_out: &routed,
        kv_comp_rows: None,
        n_comp: 0,
        comp_selected: None,
        n_selected: 0,
    };

    let out = decode_attn_layer_with(&CpuAttentionDispatcher, &layer, step);
    let rms = (out.after_ffn_hc.iter().map(|v| v * v).sum::<f32>() / out.after_ffn_hc.len() as f32).sqrt();
    println!("after_ffn_hc: {} = {} hc x {} d, finite={}, rms={:.4}",
             out.after_ffn_hc.len(), n_hc, d,
             out.after_ffn_hc.iter().all(|v| v.is_finite()), rms);
    println!("PASS: the engine's own layer path ran on the package (hc, MLA, grouped output)");
    Ok(())
}
