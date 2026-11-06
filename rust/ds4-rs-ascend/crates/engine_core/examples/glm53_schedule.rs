//! Read GLM 5.3's / Kimi K3's per-block schedule FROM THE CHECKPOINT and print the census.
//!
//! Accepts MULTIPLE gguf paths, because a sharded checkpoint distributes the block tensors across its shards
//! (Kimi K3 has 14, and shard 1 carries metadata only). Merging the tensor tables is what makes the schedule a
//! property of the model rather than of whichever shard happened to be opened.
//!
//! usage: glm53_schedule <gguf> [gguf ...]

use engine_core::gguf::GgufFile;
use engine_core::glm53_plan::{schedule_of, Census};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let paths: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    anyhow::ensure!(!paths.is_empty(), "usage: glm53_schedule <gguf> [gguf ...]");
    let mut tensors = Vec::new();
    for p in &paths {
        let g = GgufFile::open(p)?;
        tensors.extend(g.tensors);
    }
    println!("{} shard(s), {} tensors", paths.len(), tensors.len());
    let s = schedule_of(&tensors)?;
    let c = Census::of(&s);
    println!(
        "  {} blocks: {} KDA + {} MLA, {} MoE + {} dense",
        c.blocks, c.kda, c.mla, c.moe, c.dense
    );
    for (i, b) in s.iter().take(4).enumerate() {
        println!("  blk {i}: {:?} / {:?}", b.attn, b.ffn);
    }
    anyhow::ensure!(c.blocks > 10, "only {} blocks -- is this a complete checkpoint?", c.blocks);
    anyhow::ensure!(c.kda + c.mla == c.blocks, "every block must have exactly one attention kind");
    println!("PASS: the schedule was read from the checkpoint's own tensor names");
    Ok(())
}
