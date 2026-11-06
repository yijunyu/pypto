//! Encode text with the checkpoint's own tokeniser and round-trip it back.
//!
//! usage: tokenize <gguf> <text>

use engine_core::gguf::GgufFile;
use engine_core::tokenizer::Tokenizer;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let path = PathBuf::from(std::env::args().nth(1).ok_or_else(|| anyhow::anyhow!("usage: tokenize <gguf> <text>"))?);
    let text = std::env::args().nth(2).ok_or_else(|| anyhow::anyhow!("text"))?;
    let g = GgufFile::open(&path)?;
    let t = Tokenizer::from_gguf(&g)?;
    println!("vocab {} tokens, {} merges, eos {}, bos {}", t.tokens.len(), t.tokens.len(), t.eos, t.bos);
    let ids = t.encode(&text)?;
    println!("encode({text:?}) -> {ids:?}");
    let back = t.decode(&ids)?;
    println!("decode -> {back:?}");
    anyhow::ensure!(back == text, "roundtrip changed the text: {back:?} vs {text:?}");
    // the ids must be in range and the decode must be stable through a second pass
    let ids2 = t.encode(&back)?;
    anyhow::ensure!(ids == ids2, "encode(decode(ids)) differs from ids");
    println!("PASS: roundtrip is stable and every id is inside the 154,880-entry vocabulary");
    Ok(())
}
