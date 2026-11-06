//! Read GLM 5.3's (or Kimi K3's) per-block schedule FROM THE CHECKPOINT and print the census.
//!
//! usage: glm53_schedule <gguf>

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::{schedule, Census};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("usage: glm53_schedule <gguf>"))?;
    let g = GgufFile::open(&path)?;
    let s = schedule(&g)?;
    let c = Census::of(&s);
    println!("{path:?}");
    println!(
        "  {} blocks: {} KDA + {} MLA, {} MoE + {} dense",
        c.blocks, c.kda, c.mla, c.moe, c.dense
    );
    // the first few, so a wrong scan is visible rather than merely counted
    for (i, b) in s.iter().take(4).enumerate() {
        println!("  blk {i}: {:?} / {:?}", b.attn, b.ffn);
    }
    anyhow::ensure!(c.blocks > 10, "only {} blocks -- is this a complete checkpoint?", c.blocks);
    anyhow::ensure!(c.kda + c.mla == c.blocks, "every block must have exactly one attention kind");
    println!("PASS: the schedule was read from the checkpoint's own tensor names");
    Ok(())
}
