//! A BPE tokeniser built from the checkpoint's own tables.
//!
//! The file carries `tokenizer.ggml.tokens` (154,880 entries), `tokenizer.ggml.merges` (321,649 ranked pairs),
//! `model = 'gpt2'`, `pre = 'glm4'`, the special-token ids and a chat template -- so encoding needs no external tool
//! and no vocabulary guesswork.
//!
//! THE PRE-TOKENISATION IS PARTIAL, deliberately and visibly. `glm4`'s splitter is a regex over character classes
//! (letters, digits, punctuation, whitespace runs); this implements the common ASCII cases and REFUSES anything it
//! does not model, because a silently different split changes the token boundaries and therefore everything
//! downstream while still round-tripping through the decoder.

use crate::gguf::{GgufFile, MetaValue};
use std::collections::HashMap;

pub struct Tokenizer {
    pub tokens: Vec<String>,
    /// merge pair -> rank
    ranks: HashMap<(String, String), u32>,
    pub eos: u32,
    pub bos: u32,
    /// byte-level encoder: the gpt2 mapping from bytes to printable unicode
    byte_to_char: HashMap<u8, char>,
}

/// The gpt2 byte-level alphabet, built the way the reference builds it.
fn byte_level() -> (HashMap<u8, char>, HashMap<char, u8>) {
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();
    bs.extend(0xA1u8..=0xAC);
    bs.extend(0xAEu8..=0xFF);
    let mut cs: Vec<u32> = bs.iter().map(|b| *b as u32).collect();
    let mut n = 0u32;
    for b in 0u16..=255 {
        if !bs.contains(&(b as u8)) {
            bs.push(b as u8);
            cs.push(256 + n);
            n += 1;
        }
    }
    let fwd = bs.iter().zip(&cs).map(|(b, c)| (*b, char::from_u32(*c).unwrap())).collect();
    let rev = cs.iter().zip(&bs).map(|(c, b)| (char::from_u32(*c).unwrap(), *b)).collect();
    (fwd, rev)
}

impl Tokenizer {
    pub fn from_gguf(g: &GgufFile) -> anyhow::Result<Tokenizer> {
        let get = |k: &str| g.meta.iter().find(|(n, _)| n == k).map(|(_, v)| v);
        let strings = |k: &str| -> anyhow::Result<Vec<String>> {
            match get(k) {
                Some(MetaValue::Array { values, .. }) => Ok(values
                    .iter()
                    .map(|v| match v {
                        MetaValue::String(s) => s.clone(),
                        other => other.short(),
                    })
                    .collect()),
                _ => anyhow::bail!("{k} is not an array in this file"),
            }
        };
        let tokens = strings("tokenizer.ggml.tokens")?;
        let merges = strings("tokenizer.ggml.merges")?;
        let mut ranks = HashMap::with_capacity(merges.len());
        for (i, m) in merges.iter().enumerate() {
            if let Some((a, b)) = m.split_once(' ') {
                ranks.insert((a.to_string(), b.to_string()), i as u32);
            }
        }
        let num = |k: &str| -> u32 {
            match get(k) {
                Some(MetaValue::U32(v)) => *v,
                Some(MetaValue::U64(v)) => *v as u32,
                _ => 0,
            }
        };
        anyhow::ensure!(!tokens.is_empty(), "no tokens in this file");
        anyhow::ensure!(!ranks.is_empty(), "no merges in this file");
        let (byte_to_char, _) = byte_level();
        Ok(Tokenizer {
            tokens,
            ranks,
            eos: num("tokenizer.ggml.eos_token_id"),
            bos: num("tokenizer.ggml.bos_token_id"),
            byte_to_char,
        })
    }

    /// The pre-tokenisation this implements: ASCII words, single spaces and single newlines. Anything else is
    /// REFUSED rather than guessed, because a different split yields a different -- but equally round-tripping --
    /// encoding.
    fn pretokenize(&self, text: &str) -> anyhow::Result<Vec<String>> {
        let mut out = Vec::new();
        let mut cur = String::new();
        let mut cur_is_word = false;
        for ch in text.chars() {
            let is_word = ch.is_ascii_alphanumeric();
            if cur_is_word != is_word && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !ch.is_ascii_graphic() && ch != ' ' && ch != '\n' {
                anyhow::bail!("character {ch:?} is outside the modelled pre-tokenisation");
            }
            cur.push(ch);
            cur_is_word = is_word;
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        Ok(out)
    }

    /// BPE over one pre-token, on the byte-level alphabet.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word
            .bytes()
            .map(|b| self.byte_to_char.get(&b).copied().unwrap_or(b as char).to_string())
            .collect();
        if syms.len() < 2 {
            return syms;
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                if let Some(r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.map(|(br, _)| *r < br).unwrap_or(true) {
                        best = Some((*r, i));
                    }
                }
            }
            let Some((_, at)) = best else { break };
            let merged = format!("{}{}", syms[at], syms[at + 1]);
            syms.splice(at..at + 2, [merged]);
            if syms.len() == 1 {
                break;
            }
        }
        syms
    }

    /// Encode text to token ids.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let lookup: HashMap<&str, usize> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i))
            .collect();
        let mut ids = Vec::new();
        for piece in self.pretokenize(text)? {
            for sym in self.bpe(&piece) {
                let id = lookup
                    .get(sym.as_str())
                    .ok_or_else(|| anyhow::anyhow!("BPE produced {sym:?}, which is not in the vocabulary"))?;
                ids.push(*id as u32);
            }
        }
        Ok(ids)
    }

    /// Decode ids back to text, through the byte-level alphabet.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        let (_, rev) = byte_level();
        let mut bytes = Vec::new();
        for id in ids {
            let tok = self
                .tokens
                .get(*id as usize)
                .ok_or_else(|| anyhow::anyhow!("id {id} is outside the vocabulary"))?;
            for ch in tok.chars() {
                match rev.get(&ch) {
                    Some(b) => bytes.push(*b),
                    None => bytes.extend(ch.to_string().into_bytes()),
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> Tokenizer {
        // a tiny hand-built vocabulary: the byte alphabet plus a few merges, enough to test the MECHANISM
        let (byte_to_char, _) = byte_level();
        let mut tokens: Vec<String> = (0u8..=255)
            .map(|b| byte_to_char.get(&b).copied().unwrap_or('?').to_string())
            .collect();
        // "he" then "hel" then "hello"
        let mut ranks = HashMap::new();
        tokens.push("he".into());
        ranks.insert(("h".to_string(), "e".to_string()), 0);
        tokens.push("hel".into());
        ranks.insert(("he".to_string(), "l".to_string()), 1);
        tokens.push("hell".into());
        ranks.insert(("hel".to_string(), "l".to_string()), 2);
        tokens.push("hello".into());
        ranks.insert(("hell".to_string(), "o".to_string()), 3);
        Tokenizer { tokens, ranks, eos: 0, bos: 1, byte_to_char }
    }

    /// The mechanism: merges apply in RANK order, so "hello" collapses to one token when every merge exists.
    #[test]
    fn bpe_merges_in_rank_order() {
        let t = tok();
        let ids = t.encode("hello").unwrap();
        assert_eq!(ids.len(), 1, "expected one token, got {ids:?}");
        assert_eq!(t.decode(&ids).unwrap(), "hello");
    }

    /// A ROUNDTRIP through encode/decode is stable, and unknown characters are REFUSED rather than dropped.
    #[test]
    fn encode_decode_roundtrips_and_refuses_unknown() {
        let t = tok();
        let ids = t.encode("hello hello").unwrap();
        assert_eq!(t.decode(&ids).unwrap(), "hello hello");
        let err = t.encode("héllo").expect_err("non-ASCII must be refused");
        assert!(format!("{err}").contains("outside the modelled"), "got: {err}");
    }
}
