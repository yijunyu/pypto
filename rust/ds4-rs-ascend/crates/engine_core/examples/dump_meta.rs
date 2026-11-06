//! Dump a GGUF's metadata keys, so the engine's config reader can learn the file's own vocabulary.
use engine_core::gguf::GgufFile;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let p = std::env::args().nth(1).map(PathBuf::from).ok_or_else(|| anyhow::anyhow!("usage: dump_meta <gguf>"))?;
    let g = GgufFile::open(&p)?;
    println!("version {}  tensors {}  meta {}", g.version, g.n_tensors, g.n_meta_kv);
    for (k, v) in &g.meta {
        let s = format!("{v:?}");
        if k.contains("block_count") || k.contains("head") || k.contains("hidden")
            || k.contains("expert") || k.contains("vocab") || k.contains("arch")
            || k.contains("lora") || k.contains("rope") || k.contains("norm") {
            println!("{k} = {}", &s[..s.len().min(90)]);
        }
    }
    println!("--- every tensor of block 0 ---");
    for t in g.tensors.iter().filter(|t| t.name.starts_with("blk.0.")) {
        println!("{} {:?} {:?}", t.name, t.dims, t.ttype);
    }
    println!("--- first 24 tensor names ---");
    for t in g.tensors.iter().take(24) {
        println!("{} {:?} {:?}", t.name, t.dims, t.ttype);
    }
    Ok(())
}
