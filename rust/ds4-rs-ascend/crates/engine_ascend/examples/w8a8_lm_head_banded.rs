//! A vocabulary-wide projection as BANDS of a certified kernel.
//!
//! `lm_head` is 154,880 outputs. That is 620 KiB of output buffer against a 188 KiB unified buffer, so it
//! cannot be one kernel at any k-chunking -- the output-width bound is separate from the k bound. The caller
//! decomposes it over the OUTPUT dimension: N bands of 4096 weight rows, each through a certified 5-argument
//! w8a8 matvec at (4096 out, k=4096).
//!
//! This example proves the mechanism at 3 bands (12,288 outputs) with deterministic synthetic planes, and
//! notes the full size: at 154,880 outputs that is 38 bands.
//!
//! usage: w8a8_lm_head_banded <elf> [bands]

use engine_ascend::device::{DeviceLaunch, Partitioning};
use engine_ascend::launch::KernelLaunch;
use std::path::PathBuf;

const K: usize = 4096;        // lm_head's input width (GLM 5.3's D)
const BAND: usize = 4096;     // outputs per band: the certified kernel's width

fn main() -> anyhow::Result<()> {
    let elf = std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: w8a8_lm_head_banded <elf> [bands]"))?;
    let bands: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let dev = DeviceLaunch::open(std::path::Path::new(&elf), 0)?;

    // Deterministic synthetic w8a8 planes for `bands` bands; real weights are 154,880 x 4096 = 634 MB.
    let d_out = bands * BAND;
    let mut codes = vec![0i8; d_out * K];
    let mut scale = vec![0.0f32; d_out];
    let mut offset = vec![0.0f32; d_out];
    for r in 0..d_out {
        for c in 0..K { codes[r * K + c] = (((r * 7 + c * 13) % 255) as i32 - 127) as i8; }
        scale[r] = 0.5 / ((r % 7) + 1) as f32;
        offset[r] = ((r % 5) as f32) - 2.0;
    }
    let x: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) * 0.01).collect();
    println!("{d_out} outputs as {bands} bands of {BAND}, k={K}");

    let mut got = Vec::with_capacity(d_out);
    for b in 0..bands {
        let lo = b * BAND;
        let hi = lo + BAND;
        let w = &codes[lo * K..hi * K];
        let sc = &scale[lo..hi];
        let of = &offset[lo..hi];
        let mut band = dev.run_by_symbol(
            "w8a8_lm_head_band",
            "glm53_proj_output_mix_aiv",
            Partitioning::ByRow,
            &[
                (unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 4) }, false),
                (unsafe { std::slice::from_raw_parts(w.as_ptr() as *const u8, w.len()) }, false),
                (unsafe { std::slice::from_raw_parts(sc.as_ptr() as *const u8, sc.len() * 4) }, false),
                (unsafe { std::slice::from_raw_parts(of.as_ptr() as *const u8, of.len() * 4) }, false),
            ],
            BAND,
        )?;
        anyhow::ensure!(band.len() == BAND, "band {b} returned {}", band.len());
        got.append(&mut band);
    }

    // f64 reference over the same bands.
    let (mut abs_sum, mut sum_ref, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
    for r in 0..d_out {
        let mut want = 0.0f64;
        for c in 0..K {
            want += (codes[r * K + c] as f64 - offset[r] as f64) * scale[r] as f64 * x[c] as f64;
        }
        let d = (got[r] as f64 - want).abs();
        abs_sum += d; sum_ref += want.abs();
        if d > max_abs { max_abs = d; }
    }
    let backward = abs_sum / sum_ref.max(1e-30);
    println!("banded device vs f64: max_abs {max_abs:.3e}  backward_rel {backward:.3e}");
    anyhow::ensure!(backward <= 1e-5, "banded projection disagrees: {backward}");
    println!("PASS: a vocabulary-wide projection ran as {bands} bands of a certified kernel");
    Ok(())
}
