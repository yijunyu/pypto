//! Map the engine's weight ROLES onto a W8A8 package's tensor names.
//!
//! The decoder asks for `blk.N.attn_q_a.weight`; the deployment package stores
//! `layers.N.attn.wq_a.weight`. Both are DS4's own vocabulary, so this is a rename rather than an
//! architecture translation -- but a rename that has to be written down and tested, because a
//! wrong entry would load a real tensor with the wrong shape and fail three layers later with a
//! dimension error that looks like a kernel bug.
//!
//! The mapping is derived from the package itself (`examples/w8a8_model` walks it), not from a
//! document: every role below was checked to exist in the 0731 package.

use crate::safetensors::{Dtype, Package, TensorRef};
use anyhow::{anyhow, bail, Result};

/// A role the decoder needs, with the package name it corresponds to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoleTensor {
    /// The engine's role name, e.g. `attn_q_b`.
    pub role: &'static str,
    /// The package tensor that carries it, e.g. `layers.7.attn.wq_b.weight`.
    pub name: String,
}

/// One layer's roles, in the order the decoder asks for them.
pub fn layer_roles(layer: usize) -> Vec<RoleTensor> {
    let l = layer;
    let mut v = vec![
        RoleTensor { role: "attn_norm", name: format!("layers.{l}.attn_norm.weight") },
        RoleTensor { role: "attn_q_a", name: format!("layers.{l}.attn.wq_a.weight") },
        RoleTensor { role: "attn_q_b", name: format!("layers.{l}.attn.wq_b.weight") },
        RoleTensor { role: "attn_kv", name: format!("layers.{l}.attn.wkv.weight") },
        RoleTensor { role: "attn_output_a", name: format!("layers.{l}.attn.wo_a.weight") },
        RoleTensor { role: "attn_output_b", name: format!("layers.{l}.attn.wo_b.weight") },
        RoleTensor { role: "ffn_norm", name: format!("layers.{l}.ffn_norm.weight") },
        RoleTensor { role: "ffn_gate_inp", name: format!("layers.{l}.ffn.gate.weight") },
        RoleTensor { role: "ffn_gate_shexp", name: format!("layers.{l}.ffn.shared_experts.w1.weight") },
        RoleTensor { role: "ffn_down_shexp", name: format!("layers.{l}.ffn.shared_experts.w2.weight") },
        RoleTensor { role: "ffn_up_shexp", name: format!("layers.{l}.ffn.shared_experts.w3.weight") },
    ];
    // The model-wide roles are carried on layer 0 so a single call site can resolve everything.
    if layer == 0 {
        v.push(RoleTensor { role: "token_embd", name: "embed.weight".to_string() });
        v.push(RoleTensor { role: "output", name: "head.weight".to_string() });
    }
    v
}

/// A routed expert's three planes. The package stores 256 SEPARATE tensors per layer, not one
/// stacked plane, so the engine's slice arithmetic (`e * stride`) does not carry over.
pub fn expert_roles(layer: usize, expert: usize) -> Vec<RoleTensor> {
    ["w1", "w2", "w3"]
        .iter()
        .zip(["ffn_gate_exps", "ffn_down_exps", "ffn_up_exps"])
        .map(|(w, role)| RoleTensor {
            role,
            name: format!("layers.{layer}.ffn.experts.{expert}.{w}.weight"),
        })
        .collect()
}

/// Resolve every role of one layer, checking it exists and reporting its dtype and shape.
///
/// A missing role is an ERROR here rather than a zero-filled placeholder: a silently absent
/// weight is a model that runs and produces nonsense.
pub fn resolve_layer(pkg: &Package, layer: usize) -> Result<Vec<(RoleTensor, TensorRef)>> {
    let mut out = Vec::new();
    for r in layer_roles(layer) {
        let t = pkg.tensor(&r.name).map_err(|_| {
            anyhow!("layer {layer}: role `{}` has no tensor `{}` in the package", r.role, r.name)
        })?;
        out.push((r, t));
    }
    Ok(out)
}

/// The roles that arrive quantised, and those that do not.
///
/// Recorded as data because it is a property of the EXPORT, not of a role: the same role could be
/// I8 in one package and BF16 in another, and a loader that hardcodes either will misread the other.
pub fn describe(t: &TensorRef) -> &'static str {
    match t.dtype {
        Dtype::I8 => "w8a8: (q - offset[i]) * scale[i] per output channel",
        Dtype::BF16 => "bf16: widened to f32",
        Dtype::F32 => "f32: read directly",
        Dtype::F16 => "f16: converted to f32",
        _ => "unsupported dtype",
    }
}


/// A package as a SOURCE of weight data, addressed by the decoder's roles.
///
/// This is the shape the converted `from_views_impl` will call: `dequant(layer, role)` returns
/// owned f32 (I8 dequantised per output channel, BF16 widened, F32 read), and `raw(layer, role)`
/// returns the tensor's bytes for the planes the engine keeps quantised on the host.
///
/// Both return OWNED data deliberately. `LayerViews::bytes_for` lends a slice into its single
/// mmap; a package spans 74 files, so lending is not available and pretending otherwise would
/// force a lifetime through a hot loop for no benefit.
pub struct PackageSource<'a> {
    pkg: &'a Package,
}

impl<'a> PackageSource<'a> {
    pub fn new(pkg: &'a Package) -> Self {
        Self { pkg }
    }

    fn name_for(&self, layer: usize, role: &str) -> Result<String> {
        layer_roles(layer)
            .into_iter()
            .find(|r| r.role == role)
            .map(|r| r.name)
            .ok_or_else(|| anyhow!("layer {layer}: no role named {role}"))
    }

    /// A role as f32, whatever its on-disk dtype.
    pub fn dequant(&self, layer: usize, role: &str) -> Result<Vec<f32>> {
        let name = self.name_for(layer, role)?;
        let t = self.pkg.tensor(&name)?;
        match t.dtype {
            Dtype::I8 => self.pkg.dequant_tensor(&name),
            _ => self.pkg.f32_tensor(&name),
        }
    }

    /// A role's raw bytes, for a caller that will dequantise on the device.
    pub fn raw(&self, layer: usize, role: &str) -> Result<Vec<u8>> {
        let name = self.name_for(layer, role)?;
        Ok(self.pkg.raw(&name)?.1)
    }

    /// One routed expert plane: `plane` is 1, 2 or 3 (`w1` gate, `w2` down, `w3` up).
    pub fn dequant_expert(&self, layer: usize, expert: usize, plane: u8) -> Result<Vec<f32>> {
        let role = match plane {
            1 => "ffn_gate_exps",
            2 => "ffn_down_exps",
            3 => "ffn_up_exps",
            _ => bail!("expert plane must be 1, 2 or 3"),
        };
        let e = expert_roles(layer, expert)
            .into_iter()
            .find(|r| r.role == role)
            .ok_or_else(|| anyhow!("layer {layer}: no expert role {role}"))?;
        let t = self.pkg.tensor(&e.name)?;
        match t.dtype {
            Dtype::I8 => self.pkg.dequant_tensor(&e.name),
            _ => self.pkg.f32_tensor(&e.name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_layer_role_has_a_distinct_package_name() {
        let roles = layer_roles(3);
        let mut names: Vec<&str> = roles.iter().map(|r| r.name.as_str()).collect();
        let n = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), n, "two roles map to the same tensor: {roles:?}");
        for r in &roles {
            assert!(r.name.starts_with("layers.3."), "layer 3 role escaped its layer: {r:?}");
        }
    }

    #[test]
    fn the_model_wide_roles_are_carried_once() {
        // They must appear when asked for layer 0 and NOT repeated for other layers, or a
        // caller summing per-layer bytes would count the 2 GB head 43 times.
        let z = layer_roles(0);
        assert!(z.iter().any(|r| r.role == "token_embd"));
        assert!(z.iter().any(|r| r.role == "output"));
        let other = layer_roles(1);
        assert!(!other.iter().any(|r| r.role == "token_embd"));
    }

    #[test]
    fn expert_roles_are_per_expert_not_stacked() {
        let e = expert_roles(2, 17);
        assert_eq!(e.len(), 3);
        assert!(e.iter().all(|r| r.name.contains("experts.17.")));
        assert_eq!(e[0].name, "layers.2.ffn.experts.17.w1.weight");
        assert_eq!(e[1].name, "layers.2.ffn.experts.17.w2.weight");
        assert_eq!(e[2].name, "layers.2.ffn.experts.17.w3.weight");
    }
}
