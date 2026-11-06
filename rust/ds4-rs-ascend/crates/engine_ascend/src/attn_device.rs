//! The attention dispatcher that holds a KERNEL dispatcher -- the seam the MLA path needs.
//!
//! `engine_core`'s `AttentionDispatcher` is, in its own words, "intentionally NOT a supertrait of
//! `KernelDispatcher`"; dispatchers compose by HOLDING both. Its only implementation is
//! `CpuAttentionDispatcher`, a unit struct, so the real MLA attention path has had no device hook.
//!
//! This is that hook, and it is deliberately INCREMENTAL: methods are routed to the kernel dispatcher
//! one at a time, each with a bit-comparison against the CPU reference, because each one carries its
//! own semantics (eps, gamma, clamps) that a bulk rewrite would get subtly wrong.
//!
//! ROUTED SO FAR: `qkv_rms_norm_rows` -- the two RMS norms over the q-lora row and the kv row now go
//! through `KernelDispatcher::rms_norm`, which on the device is `ds4_rmsmul_*`. Everything else still
//! delegates to `CpuAttentionDispatcher`.

use engine_core::attn_dispatch::{
    AttnHeadsOut, AttnPrefixBatchedOut, AttentionDispatcher, CpuAttentionDispatcher, HcKind,
    KvCacheView, LayerParams,
};
use engine_core::dispatch::KernelDispatcher;

pub struct AscendAttentionDispatcher<D: KernelDispatcher> {
    /// The reference implementation, for every method not yet routed.
    inner: CpuAttentionDispatcher,
    /// The kernel dispatcher: on device this is `AscendDispatcher`, which forwards to the emitted
    /// kernels; in tests it is `CpuDispatcher`, which makes the comparison exact.
    kernels: D,
}

impl<D: KernelDispatcher> AscendAttentionDispatcher<D> {
    pub fn new(kernels: D) -> Self {
        Self { inner: CpuAttentionDispatcher, kernels }
    }

    pub fn kernels(&self) -> &D {
        &self.kernels
    }
}

impl<D: KernelDispatcher> AttentionDispatcher for AscendAttentionDispatcher<D> {
    fn qkv_rms_norm_rows(
        &self,
        params: &LayerParams,
        qr: &[f32],
        kv_raw: &[f32],
        gamma_q: &[f32],
        gamma_kv: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        (
            self.kernels.rms_norm(qr, gamma_q, params.rms_eps),
            self.kernels.rms_norm(kv_raw, gamma_kv, params.rms_eps),
        )
    }

    fn hc_collapse_norm(
        &self,
        params: &LayerParams,
        kind: HcKind,
        hc_fn: &[f32],
        hc_scale: &[f32],
        hc_base: &[f32],
        prev_hc: &[f32],
        use_gamma: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        self.inner.hc_collapse_norm(params, kind, hc_fn, hc_scale, hc_base, prev_hc, use_gamma)
    }

    fn rope_tail(&self, params: &LayerParams, x: &mut [f32], pos: u32, backward: bool) {
        self.inner.rope_tail(params, x, pos, backward)
    }

    fn kv_fp8_store(&self, params: &LayerParams, kv_row_f32: &[f32], view: &mut KvCacheView<'_>) {
        self.inner.kv_fp8_store(params, kv_row_f32, view)
    }

    fn flash_attn_decode(
        &self,
        params: &LayerParams,
        q: &[f32],
        kv_raw: &[f32],
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        kv_comp: Option<&[f32]>,
        n_comp: u32,
        comp_selected: Option<&[u32]>,
        n_selected: u32,
        attn_sinks: &[f32],
    ) -> AttnHeadsOut {
        self.inner.flash_attn_decode(
            params, q, kv_raw, n_raw, raw_cap, raw_start, kv_comp, n_comp, comp_selected,
            n_selected, attn_sinks,
        )
    }

    fn attn_output_proj(
        &self,
        params: &LayerParams,
        heads: &[f32],
        w_o_a: &[f32],
        w_o_b: &[f32],
        cur_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        self.inner.attn_output_proj(params, heads, w_o_a, w_o_b, cur_hc, hc_split_post, hc_split_comb)
    }

    fn shared_expert(
        &self,
        params: &LayerParams,
        ffn_norm: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        shared_dim: u32,
        clamp: f32,
    ) -> Vec<f32> {
        self.inner.shared_expert(params, ffn_norm, w_gate, w_up, shared_dim, clamp)
    }

    fn shared_down_hc_expand_add(
        &self,
        params: &LayerParams,
        shared_mid: &[f32],
        w_down: &[f32],
        routed_out: &[f32],
        after_attn_hc: &[f32],
        hc_split_post: &[f32],
        hc_split_comb: &[f32],
    ) -> Vec<f32> {
        self.inner.shared_down_hc_expand_add(
            params, shared_mid, w_down, routed_out, after_attn_hc, hc_split_post, hc_split_comb,
        )
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::dispatch::CpuDispatcher;

    fn params() -> LayerParams {
        LayerParams {
            layer_idx: 0, d_embd: 8, n_hc: 2, n_head: 2, head_dim: 4, n_rot: 2,
            n_lora_q: 6, n_lora_kv: 4, hc_sinkhorn_iter: 1, hc_eps: 1e-6, rms_eps: 1e-6,
            rope_orig_ctx: 4096, rope_freq_base: 10000.0, rope_freq_scale: 1.0,
            rope_ext_factor: 0.0, rope_attn_factor: 1.0, compress_ratio: 0, n_out_group: 1,
        }
    }

    /// The routed method must agree with the CPU reference EXACTLY when both sit on the CPU
    /// dispatcher. That is the property that makes routing one method at a time safe: a mismatch is a
    /// difference in SEMANTICS, not in hardware.
    #[test]
    fn the_routed_method_matches_the_cpu_reference() {
        let p = params();
        let qr: Vec<f32> = (0..6).map(|i| (i as f32 - 2.5) * 0.3).collect();
        let kv: Vec<f32> = (0..4).map(|i| (i as f32 - 1.5) * 0.7).collect();
        let gq: Vec<f32> = (0..6).map(|i| 1.0 + i as f32 * 0.05).collect();
        let gk: Vec<f32> = (0..4).map(|i| 1.0 - i as f32 * 0.03).collect();

        let reference = CpuAttentionDispatcher.qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);
        let ours = AscendAttentionDispatcher::new(CpuDispatcher).qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);

        assert_eq!(ours.0.len(), reference.0.len());
        for (a, b) in ours.0.iter().zip(&reference.0) {
            assert!((a - b).abs() <= 1e-6, "q row disagrees: {a} vs {b}");
        }
        for (a, b) in ours.1.iter().zip(&reference.1) {
            assert!((a - b).abs() <= 1e-6, "kv row disagrees: {a} vs {b}");
        }
    }

    /// And the values must actually be an RMS norm, not a pass-through: a test that would pass on a
    /// function returning its input unchanged would not be a test.
    #[test]
    fn the_routed_method_normalises_rather_than_passing_through() {
        let p = params();
        let qr = vec![3.0f32; 6];
        let kv = vec![3.0f32; 4];
        let gq = vec![1.0f32; 6];
        let gk = vec![1.0f32; 4];
        let (q, k) = AscendAttentionDispatcher::new(CpuDispatcher)
            .qkv_rms_norm_rows(&p, &qr, &kv, &gq, &gk);
        // A constant row normalises to ~1 per element, not to 3.
        for v in q.iter().chain(k.iter()) {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0 after RMS norm, got {v}");
        }
    }
}
