//! One REAL routed expert's down projection, device vs CPU, on the package's own int8 weights.
//!
//! The package stores each expert as `I8` codes plus a per-output-channel `F32` scale and offset --
//! exactly the w8a8 layout -- so nothing is dequantised before the launch. The CPU side dequantises
//! through the same (q - offset) * scale formula and computes the same dot in f64.
//!
//! usage: w8a8_expert_down <package dir> <layer> <expert> <elf>

use engine_ascend::dispatcher::AscendDispatcher;
use engine_core::dispatch::CpuDispatcher;
use engine_core::safetensors::Package;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: w8a8_expert_down <pkg> <layer> <expert> <elf>"))?;
    let layer: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let expert: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let elf = std::env::args().nth(4).ok_or_else(|| anyhow::anyhow!("need the device ELF"))?;
    let pkg = Package::open(&dir)?;
    let dev = AscendDispatcher::new(CpuDispatcher)
        .with_devices(std::path::Path::new(&elf), &[0])?;

    let p = format!("layers.{layer}.ffn.experts.{expert}.w2");
    let (codes, scale, offset) = {
        let raw = pkg.raw(&format!("{p}.weight"))?.1;
        let codes: Vec<i8> = raw.iter().map(|&b| b as i8).collect();
        (codes, pkg.f32_tensor(&format!("{p}.weight_scale"))?, pkg.f32_tensor(&format!("{p}.weight_offset"))?)
    };
    let d_out = scale.len();                    // one scale per output channel
    let d_in = codes.len() / d_out;
    println!("w2: {} codes, {} outputs, k={}", codes.len(), d_out, d_in);
    anyhow::ensure!(offset.len() == d_out, "offset {} != outputs {}", offset.len(), d_out);

    // A deterministic activation; the VALUES do not matter, only that both sides see them.
    let x: Vec<f32> = (0..d_in).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();

    let dev_out = dev.matvec_w8a8_experts(
        if d_in == 4096 { "w8a8_experts_4096_2048" } else { "w8a8_experts_2048_4096" },
        &codes, &scale, &offset, &x, d_out)?;

    // CPU reference in f64, through the package's own formula.
    let mut want = vec![0.0f64; d_out];
    for r in 0..d_out {
        let mut acc = 0.0f64;
        for c in 0..d_in {
            let q = codes[r * d_in + c] as f64;
            acc += (q - offset[r] as f64) * scale[r] as f64 * x[c] as f64;
        }
        want[r] = acc;
    }
    let mut abs_sum = 0.0f64;
    let mut sum_ref = 0.0f64;
    let mut max_abs = 0.0f64;
    for r in 0..d_out {
        let d = (dev_out[r] as f64 - want[r]).abs();
        abs_sum += d;
        sum_ref += want[r].abs();
        if d > max_abs { max_abs = d; }
    }
    let backward = abs_sum / sum_ref.max(1e-30);
    println!("device vs CPU: max_abs {max_abs:.3e}  backward_rel {backward:.3e}");
    anyhow::ensure!(backward <= 1e-5, "device and CPU disagree: {backward}");
    println!("PASS: a real expert's down projection ran through the w8a8 kernel");
    Ok(())
}
