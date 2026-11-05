// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Softmax tests. S2 fuses the 5-op chain into a single `tile_softmax_fNN`;
//! the raw (un-fused) path stays available for debugging.

use ptoas_rs::{convert_pto_to_rust, convert_pto_to_rust_raw};

const SOFTMAX_PTO: &str = include_str!("fixtures/softmax.pto");

#[test]
fn fused_softmax_collapses_to_one_call() {
    let rust = convert_pto_to_rust(SOFTMAX_PTO).expect("PTO -> Rust conversion failed");

    assert!(
        rust.contains("pub unsafe extern \"C\" fn tile_softmax("),
        "missing function header\n--- output ---\n{rust}"
    );

    // After fusion, exactly one softmax call; no stray pieces of the chain.
    assert_eq!(
        rust.matches("tile_softmax_f32::<1, 1024>").count(),
        1,
        "expected fused softmax\n{rust}"
    );
    for needle in [
        "tile_reduce_max",
        "tile_reduce_sum",
        "tile_row_expand_sub",
        "tile_row_expand_div",
        "tile_exp",
    ] {
        assert!(
            !rust.contains(needle),
            "leaked decomposed op `{needle}`:\n{rust}"
        );
    }

    // Load + store still bracket the kernel.
    assert!(rust.contains("tile_load_f32::<1, 1024>"));
    assert!(rust.contains("tile_store_f32::<1, 1024>"));
}

/// With fusion off, the chain stays decomposed -- and a decomposed chain has
/// no lift, so raw mode refuses.
///
/// This test used to assert the opposite: that raw mode emitted
/// `tile_row_expand_sub_f32::<1, 1024>` and `tile_row_expand_div_f32::<1,
/// 1024>`. Neither name exists in `tile_std` at any dtype, so what it
/// certified was Rust that cannot compile. See `tests/fuse_soundness.rs` for
/// the full account; the short version is that the row-expansions are liftable
/// only inside a fused whole.
#[test]
fn raw_mode_refuses_because_the_decomposed_chain_has_no_lift() {
    let err = convert_pto_to_rust_raw(SOFTMAX_PTO)
        .expect_err("raw mode leaves the chain decomposed, which has no lift");
    // Naming the row-expansion is what proves fusion was OFF: had it fired,
    // no such op would remain to refuse. The same input lifts cleanly in
    // `fused_softmax_collapses_to_one_call` above, so the flag demonstrably
    // changes the outcome.
    assert!(
        err.contains("pto.trowexpandsub") || err.contains("pto.trowexpanddiv"),
        "raw mode either fused anyway or refused for another reason: {err}"
    );
}

#[test]
fn empty_module_is_ok() {
    let rust = convert_pto_to_rust("module {\n}\n").expect("empty module should parse");
    assert!(rust.contains("use tile_std::tile"));
}

#[test]
fn unknown_type_surfaces_error() {
    let bad = r#"
module {
  func.func @bad(%a: !pto.ptr<f99>) {
    return
  }
}
"#;
    let err = convert_pto_to_rust(bad).unwrap_err();
    assert!(
        err.contains("f99"),
        "expected error to mention bad dtype, got: {err}"
    );
}
