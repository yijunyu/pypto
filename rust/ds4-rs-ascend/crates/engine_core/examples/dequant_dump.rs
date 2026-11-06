//! Dump the RAW bytes and the DEQUANTISED values of the first few Q8_0 blocks of a tensor.
//!
//! The dequantiser is the piece most able to be wrong invisibly: a wrong block scale or a misaligned nibble
//! produces plausible magnitudes everywhere. This prints what it read so an independent decoder can check it.
//!
//! usage: dequant_dump <gguf> <tensor-name> [blocks]

use engine_core::gguf::GgufFile;
use engine_core::glm5_stream::ShardedGguf;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: dequant_dump <gguf> <tensor>"))?);
    let name = std::env::args().nth(2).ok_or_else(|| anyhow::anyhow!("tensor name"))?;
    let blocks: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let g = GgufFile::open(&path)?;
    let t = g.tensors.iter().find(|t| t.name == name)
        .ok_or_else(|| anyhow::anyhow!("{name} not in the tensor table"))?;
    println!("tensor {name}: dims {:?} type {:?}", t.dims, t.ttype);
    let bytes = g.tensor_bytes(t)?;
    let per = 32usize;
    anyhow::ensure!(bytes.len() >= blocks * (2 + per), "tensor is too small for {blocks} blocks");
    for b in 0..blocks {
        let off = b * (2 + per);
        let raw = &bytes[off..off + 2 + per];
        println!("block {b} raw {}", raw.iter().map(|x| format!("{x:02x}")).collect::<String>());
    }
    let sh = ShardedGguf::open(&path)?;
    let vals = sh.dequant_name(&name)?;
    let head: Vec<String> = vals[..blocks * per].iter().map(|v| format!("{v:.9e}")).collect();
    println!("dequant first {}", head.join(" "));
    Ok(())
}
