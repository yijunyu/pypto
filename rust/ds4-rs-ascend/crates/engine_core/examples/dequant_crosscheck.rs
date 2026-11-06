//! The same three numbers an independent Q8_0 decoder prints, from the engine's own path.
//!
//! usage: dequant_crosscheck <gguf> <tensor>

use engine_core::glm5_stream::ShardedGguf;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: <gguf> <tensor>"))?);
    let name = std::env::args().nth(2).ok_or_else(|| anyhow::anyhow!("tensor name"))?;
    let g = ShardedGguf::open(&path)?;
    let v = g.dequant_name(&name)?;
    anyhow::ensure!(v.len() >= 4096, "tensor has only {} values", v.len());
    let head: Vec<f64> = v[..3].iter().map(|x| *x as f64).collect();
    let sum: f64 = v[..4096].iter().map(|x| *x as f64).sum();
    let rms = (v[..4096].iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / 4096.0).sqrt();
    println!("first 3: {:.9e} {:.9e} {:.9e}", head[0], head[1], head[2]);
    println!("sum of first 4096: {sum:.9e}");
    println!("rms of first 4096: {rms:.9e}");
    Ok(())
}
