// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The V4.1 corpus, held to the same rule as the playground one: every
//! generated artefact is re-derived, not trusted.
//!
//! A separate test from the playground gate rather than a generalisation of it,
//! because the two corpora make different claims and their counts are the claim.
//! Folding them together would mean one number covering both, and a number that
//! covers two corpora tells you nothing when it moves.
//!
//! This corpus is mostly REFUSED, and that is its point: these are a real
//! model's shapes rather than teaching sizes, so they hit the part's bounds.

use std::path::{Path, PathBuf};
use std::process::Command;

fn corpus() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../bench/deepseek_v41")
        .canonicalize()
        .expect("the V4.1 corpus must be reachable")
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("v41_{}_{}", std::process::id(), name));
    p
}

fn emit(src: &Path, dst: &Path) -> bool {
    Command::new(env!("CARGO_BIN_EXE_rspto"))
        .arg(src)
        .arg("-o")
        .arg(dst)
        .output()
        .expect("rspto runs")
        .status
        .success()
}

fn tiles() -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(corpus().join("tile"))
        .expect("tile/")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "tile"))
        .collect();
    v.sort();
    v
}

#[test]
fn the_corpus_is_the_size_it_claims() {
    assert_eq!(tiles().len(), 17, "the corpus changed size; update this deliberately");
}

#[test]
fn every_checked_in_pto_is_byte_identical_to_a_fresh_emit() {
    let root = corpus();
    let (mut compared, mut refused) = (0, 0);
    for src in tiles() {
        let stem = src.file_stem().unwrap().to_string_lossy().to_string();
        let expected = root.join(format!("pto/{stem}.pto"));
        let out = tmp(&format!("{stem}.pto"));
        if !emit(&src, &out) {
            assert!(
                !expected.exists(),
                "{stem}: rspto refuses it, but a .pto is checked in"
            );
            refused += 1;
            continue;
        }
        assert!(expected.exists(), "{stem}: emits, but no .pto is recorded");
        assert_eq!(
            std::fs::read(&out).unwrap(),
            std::fs::read(&expected).unwrap(),
            "{stem}: the checked-in PTO differs from what this emitter produces"
        );
        compared += 1;
        let _ = std::fs::remove_file(&out);
    }
    // The split is the corpus's claim about these shapes, so it is pinned.
    assert_eq!(compared, 8, "expected 8 reproducible PTO artefacts");
    assert_eq!(refused, 9, "expected 9 refusals");
}

/// What makes this corpus worth having: most of it is refused, and the refusals
/// are not one bound hit over and over.
#[test]
fn the_refusals_name_several_different_bounds() {
    let mut reasons = std::collections::BTreeSet::new();
    for src in tiles() {
        let out = tmp("probe.pto");
        let run = Command::new(env!("CARGO_BIN_EXE_rspto"))
            .arg(&src)
            .arg("-o")
            .arg(&out)
            .output()
            .unwrap();
        if run.status.success() {
            let _ = std::fs::remove_file(&out);
            continue;
        }
        let msg = String::from_utf8_lossy(&run.stderr);
        for bound in [
            "NPU tile bound",
            "NPU UB budget",
            "must be a multiple of Nb",
            "must be deferred",
            "pair count",
        ] {
            if msg.contains(bound) {
                reasons.insert(bound);
            }
        }
    }
    assert!(
        reasons.len() >= 4,
        "only {} distinct bound(s) hit: {reasons:?}. If this corpus has collapsed \
         to one failure mode it has stopped being a useful spread of shapes.",
        reasons.len()
    );
}
