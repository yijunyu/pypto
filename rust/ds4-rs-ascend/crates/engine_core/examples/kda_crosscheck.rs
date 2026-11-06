//! Cross-check `kda_block::step` against an INDEPENDENT numpy implementation (see ~/kda/ref.py).
//!
//! The numpy version is written from the formula with an outer-product state update and `S.T @ k` reads; this one
//! uses explicit loops. Agreement to 1e-12 across three shapes is what makes the recurrence's arithmetic a
//! CORRECTNESS check rather than a consistency check -- the only discharge available on a host where llama.cpp
//! refuses GLM 5.3's architecture.
//!
//! usage: kda_crosscheck <case-file>

use engine_core::kda_block::{new_state, step};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/kda_case.txt".into());
    let text = std::fs::read_to_string(&path)?;
    let mut lines = text.lines();
    let mut cases = 0usize;
    let mut worst = 0.0f64;
    while let Some(head) = lines.next() {
        if !head.starts_with("CASE ") {
            continue;
        }
        let h: Vec<&str> = head.split_whitespace().collect();
        let d: usize = h[1].parse()?;
        let a_log: f64 = h[2].parse()?;
        let beta: f64 = h[3].parse()?;
        let mut get = |want: &str| -> anyhow::Result<Vec<f64>> {
            let line = lines.next().ok_or_else(|| anyhow::anyhow!("truncated at {want}"))?;
            let mut it = line.split_whitespace();
            let name = it.next().unwrap_or("");
            anyhow::ensure!(name == want, "expected {want}, got {name}");
            Ok(it.map(|v| v.parse::<f64>().unwrap_or(f64::NAN)).collect())
        };
        let q = get("q")?;
        let k = get("k")?;
        let v = get("v")?;
        let dt = get("dt")?;
        let g = get("g")?;
        let want = get("o")?;
        anyhow::ensure!(q.len() == d && want.len() == d, "case {cases}: expected width {d}");
        let mut s = new_state(d);
        let got = step(&mut s, &q, &k, &v, a_log, &dt, beta, &g);
        let err = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
        worst = worst.max(err);
        cases += 1;
        println!("case {cases} (d={d}): worst |rust - numpy| = {err:.3e}");
        anyhow::ensure!(err < 1e-12, "case {cases} disagrees by {err:.3e} -- one implementation is wrong");
    }
    anyhow::ensure!(cases > 0, "no cases parsed from {path}");
    println!("PASS: {cases} cases agree with the independent numpy recurrence, worst {worst:.3e}");
    Ok(())
}
