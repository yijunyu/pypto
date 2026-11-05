// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! The five files the README called "one op each" — measured by bypass.
//!
//! After the whole-kernel and cube/vector tiers were closed, five artifacts
//! were left, and the README said four of them "need one missing `tile_std`
//! wrapper each (`tile_load_i32`/`u32`, an i8→f16 cast, a column-strided view,
//! a sub-view)". That is the FIRST GATE talking, for the eighth time in this
//! crate. Relaxing each gate as a throwaway probe and recounting, 2026-11-01:
//!
//! | file | granting the named wrapper moves it to |
//! |---|---|
//! | `g1.pto` | `pto.tgatherb`, whose offsets count 32-byte BLOCKS on a2/a3 and ELEMENTS on a5 — no wrapper fixes that |
//! | `e8m0.pto` | `pto.tadds`, which needs a whole scalar-operand arm, not a cast |
//! | `sstore.pto` | the partition view, which has no address to fold offsets into over a non-contiguous parent |
//! | `sv.pto` | nothing — it lifts, and the lift is WRONG; rustc catches it |
//!
//! So one of the four would move behind one change, and that one produces
//! code rustc rejects. The tier is closed, and the corpus is fully accounted
//! for: 13 of 45 distinct artifacts lift, 32 are refused for reasons that are
//! each measured rather than pending.
//!
//! Four of the five are also not kernels. `gprobe`, `e8m0probe`, `sstore` and
//! `svprobe` are hardware capability probes: their job is to be handed to
//! `ptoas` so that an acceptance or a rejection answers a question, and
//! `e8m0.pto` says so in its own header — "WHAT IS UNPROVEN, named so a
//! rejection says which". Lifting a question into Rust would need the forward
//! emitter to re-ask it, which is the round trip this crate cannot run.

mod shared;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Resolve a fixture path against this crate.
///
/// Two shapes arrive here: a bare probe name, and a path this crate's own tree
/// spells out. The ported version resolved both against the repository root of
/// the tree these files came from, which this workspace is not.
fn read(rel: &str) -> String {
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = match rel.strip_prefix("crates/ptoas_rs/") {
        Some(inside) => crate_dir.join(inside),
        // `crates/tile_std/...` is the sibling DSL crate, resolved against
        // the workspace root. Named exactly, not by a `crates/` prefix: the
        // fixtures below are stored under their original repo-relative paths
        // (`crates/mlir_to_pto_tests/l0/e.pto`), so a prefix test captures 13
        // of them and sends them to the wrong directory.
        None if rel.starts_with("crates/tile_std") => crate_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("crate dir has a workspace root")
            .join(rel),
        None => crate_dir.join("tests/fixtures/probes").join(rel),
    };
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()))
}

fn refusal(rel: &str) -> String {
    ptoas_rs::convert_pto_to_rust(&read(rel))
        .map(|_| String::new())
        .unwrap_or_else(|e| e)
}

/// `g1.pto`: the i32 load is the first gate; the wall is part-dependent.
///
/// `pto.tgatherb`'s offset unit is 32-byte blocks on a2/a3 and elements on a5,
/// and nothing in the module names the part — so a lift would pick one and be
/// wrong on the other half of the fleet, at exit 0. No `tile_std` wrapper
/// changes that, which is why the README's "needs `tile_load_i32`" was the
/// wrong thing to put in front of a reader.
///
/// And the tile API's gathers are a different OPERATION, not a missing dtype:
/// every one of them takes the indices as a GM pointer or a 4-bit lane mask.
/// None takes an index TILE, which is what `pto.tgatherb` consumes, so lifting
/// it would also have to drop the index tile's own load.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL's declarations, and
/// that crate is not in this tree yet. Restore it with the Rust input mode.
#[test]
fn the_gather_probes_wall_is_the_part_not_the_i32_load() {
    // The refusal a reader sees today names the first gate, which is honest
    // about what the emitter hit -- but the measurement behind it belongs in
    // the ranked remainder, not in the reader's head.
    let err = refusal("g1.pto");
    assert!(
        err.contains("i32"),
        "g1.pto no longer refuses on the i32 dtype: {err}"
    );

    // The wall, read off the emitter's own source rather than reached, because
    // reaching it needs the gate bypassed.
    let emitter = read("crates/ptoas_rs/src/emitter.rs");
    assert!(
        emitter.contains("32-byte BLOCKS on a2/a3 and ELEMENTS on a5"),
        "the `pto.tgatherb` refusal no longer names the part-dependent offset unit; if the \
         AST now carries a target part, re-measure this file"
    );

    // And the shape mismatch in the API: pointers and masks, never a tile.
    let tile_std = read("crates/tile_std/src/tile.rs");
    let mut gathers = 0usize;
    for (i, _) in tile_std.match_indices("pub fn __tile_gather") {
        gathers += 1;
        let decl = &tile_std[i..(i + 400).min(tile_std.len())];
        let params = &decl[..decl.find(')').unwrap_or(decl.len())];
        assert!(
            params.contains("indices: *const u32") || params.contains("mask_pattern: u32"),
            "a `__tile_gather*` now takes its indices some other way:\n{params}\nIf one takes \
             a TILE of indices, `pto.tgatherb` has gained a counterpart and this file's \
             measurement is stale"
        );
    }
    assert!(
        gathers >= 2,
        "only {gathers} `__tile_gather*` declarations found; the scan is broken, not the crate"
    );
}

/// `e8m0.pto`: a cast is not enough, and the file's own questions are open.
///
/// Granting every cast moves it to `pto.tadds` — a scalar-operand arm the tile
/// API does not have at all (the `__tile_adds_*` that exist are buffer-API,
/// over `UbBuf`). Behind that sit three steps the artifact itself declares
/// unproven, including whether `tmov` between differently-typed tiles converts
/// or reinterprets. Both compile; they compute different things. A lift would
/// have to choose, which is the silent substitution this crate refuses.
/// DEFERRED, NOT RETIRED -- it reads the kernel-side DSL's declarations, and
/// that crate is not in this tree yet. Restore it with the Rust input mode.
#[test]
fn the_e8m0_probe_needs_more_than_a_cast_and_its_own_questions_are_open() {
    let err = refusal("e8m0.pto");
    assert!(
        err.contains("pto.tcvt") && err.contains("i8->f16"),
        "e8m0.pto no longer refuses on the cast: {err}"
    );

    // The second gate, from the emitter's source. It USED to assert that the
    // scalar-operand refusal still existed and that `tile_std` had no
    // `tile_adds_*`; on 2026-11-04 both flipped, which is what this test said
    // to re-measure. It did: `pto.tadds` now lifts, and e8m0's own chain stops
    // at the cast above, so the assertion moves to the fact that matters -- the
    // wrapper exists and the tier that named its absence is gone.
    let tile_std_now = read("crates/tile_std/src/tile.rs");
    assert!(
        tile_std_now.contains("pub fn tile_adds_f32"),
        "`tile_adds_f32` is gone again; the scalar-add tier should reopen"
    );
    let ast = read("crates/ptoas_rs/src/ast.rs");
    assert!(
        ast.contains("tile_adds_f32"),
        "the lift no longer names the tile wrapper it should emit"
    );
    // ... and the tier that used to sit here is CLOSED, not reopened. It fired
    // on 2026-11-04 exactly as written -- "tile_std::tile now exports a
    // scalar-operand add wrapper, so pto.tadds may be liftable" -- and the
    // answer was yes: the wrapper, the PTO arm and the lift arm all landed, the
    // tier went empty and its census row was removed. What e8m0.pto hits now is
    // the cast above, so the reopen did not reach it.

    // The artifact's own open questions. These are why lifting it would be
    // guessing, and they are stated in the file.
    let src = read("e8m0.pto");
    for claim in [
        "WHAT IS UNPROVEN",
        "`pto.tcast` DOES NOT EXIST",
        "a REINTERPRET returns 2^(e-127). Both",
    ] {
        assert!(
            src.contains(claim),
            "e8m0.pto no longer states `{claim}`. If its probes have been ANSWERED, the \
             answer decides what a lift would emit and this file needs re-measuring"
        );
    }
}

/// `sstore.pto`: the strided view is the first of two gates.
///
/// Accepting any stride as contiguous moves the refusal to the partition view
/// taken of it — the window model folds a constant offset into an address, and
/// over a non-contiguous parent there is no address to fold into. So this is
/// the view model, not a wrapper.
#[test]
fn the_strided_store_probe_has_a_second_gate_in_the_view_model() {
    let err = refusal("sstore.pto");
    assert!(
        err.contains("strided [1, 2]"),
        "sstore.pto no longer refuses on the stride: {err}"
    );
    let emitter = read("crates/ptoas_rs/src/emitter.rs");
    assert!(
        emitter.contains("a contiguous rank-2 tensor view, so there is no address to fold the"),
        "the partition-view refusal behind the stride gate is gone; re-measure what \
         sstore.pto hits once the stride is accepted"
    );
    // And the probe's own question, which is what the file is for.
    let src = read("sstore.pto");
    assert!(
        src.contains("If TSTORE has looser layout rules than TLOAD"),
        "sstore.pto no longer states the question it was written to ask"
    );
}

/// `sv.pto`: the one that WOULD move, and the lift would be wrong.
///
/// Aliasing the sub-view to its parent makes the file lift — 20 lines — and
/// the emitted Rust drops the row offset, loading 4x32 and storing 2x32.
/// `tests/lifted_kernels_compile.rs` compiles that exact shape and gets
/// `E0308: expected 2, found 4`, so the type system is the second line of
/// defence.
///
/// THIS TEST USED TO PIN A DISPUTED HARDWARE CLAIM AS A REQUIRED SUBSTRING --
/// `err.contains("910B2") && err.contains("mis-computes vector ops")` -- on the
/// grounds that it was "the half the type system cannot check". Two problems,
/// and the second is why the assertion is inverted rather than deleted.
///
/// The claim is disputed: the per-chip 910 knowledge base holds our undated
/// rule, the vendor's contrary recommendation of `tensor[offset]` as a
/// zero-copy view, and OUR OWN device run on real 910B2 / CANN 8.5.2 finding
/// the vectorised offset form bit-identical for one shape -- where the blanket
/// ban had been blocking an 85.7x corpus win.
///
/// And pinning it as a substring is what let a correction land on ONE of the
/// two refusal sites and stay green: the message in `emitter.rs` was rewritten
/// to say the claim is disputed, the message in `emitter_tile.rs` still
/// asserted it flat, and this test passed either way because the rewritten text
/// still contained both words.
///
/// So the assertion now requires the UNDISPUTED reason and forbids the disputed
/// one being stated as settled -- which fails on any site that regresses.
#[test]
fn the_subview_probe_refuses_on_shape_without_asserting_a_disputed_erratum() {
    let err = refusal("sv.pto");
    assert!(
        err.contains("has no sub-view form"),
        "sv.pto no longer refuses on the sub-view: {err}"
    );
    assert!(
        !err.contains("mis-computes vector ops") || err.contains("DISPUTED"),
        "the sub-view refusal states the 910B2 sub-view erratum as settled fact. It is \
         disputed -- a device run on real 910B2 found the vectorised offset form \
         bit-identical for one shape. Refuse on the missing tile-API form, which is \
         sufficient and undisputed, or mark the erratum DISPUTED: {err}"
    );
    // The shape half is pinned where the compiler is available.
    let gate = read("crates/ptoas_rs/tests/lifted_kernels_compile.rs");
    assert!(
        gate.contains("a sub-view dropped to its parent must not compile"),
        "the compile gate no longer exhibits the wrong lift this refusal prevents"
    );
}

/// The deliberately malformed fixture must keep refusing, on the shape.
#[test]
fn the_bad_shape_fixture_still_refuses_for_its_own_reason() {
    let err = refusal("crates/ptoas_rs/tests/fixtures/matmul_bad_shape.pto");
    assert!(
        err.contains("K disagrees") && err.contains("32x128") && err.contains("64x32"),
        "the bad-shape fixture is refused for something other than its shape, or not at \
         all: {err}"
    );
}

/// Four of the five are PROBES, not kernels.
///
/// Asserted because it is the part that decides whether the tier is worth
/// reopening at all: a probe exists to be handed to `ptoas` so an acceptance
/// or a rejection answers a question about the hardware. Lifting it to Rust
/// would need the forward emitter to re-ask the question.
#[test]
fn four_of_the_five_are_capability_probes_not_kernels() {
    for (f, func) in [
        ("g1.pto", "gprobe"),
        ("e8m0.pto", "e8m0probe"),
        ("sv.pto", "svprobe"),
    ] {
        let src = read(f);
        assert!(
            src.contains(&format!("func.func @{func}(")),
            "{f} no longer declares `@{func}`; if it was renamed to something that is not a \
             probe, re-read what it is for"
        );
    }
    // `sstore` is not named `*probe` but says what it is asking, which is the
    // criterion that matters; its question is pinned in its own test above.
    let src = read("sstore.pto");
    assert!(src.contains("func.func @sstore("), "sstore.pto was renamed");
}

/// NO RESIDUE: every distinct artifact is either lifting or refused by a tier
/// this crate has measured.
///
/// The point of the whole exercise. Each refusal is matched to the tier it
/// belongs to by a token the refusal itself carries, and an artifact that
/// matches none fails this test naming its own diagnostic — so a NEW artifact,
/// or an old one that starts refusing for a different reason, surfaces as an
/// unexamined case instead of disappearing into a remainder count.
#[test]
fn every_refused_artifact_belongs_to_a_measured_tier() {
    // (tier, a token only that tier's refusal carries)
    // `arith.index_cast` is deliberately NOT a row: every corpus artifact that
    // holds one also holds a `pto.get_block_idx` that refuses first, so a row
    // for it would be a tier with no members -- which is what the per-tier
    // floor below exists to catch.
    //
    // The last six tiers arrived when the walk was repointed from `rust/` to
    // the repository root. They are not new defects: they are the bench
    // corpus's refusals, which this test could not see while its denominator
    // was this crate's own fixtures. Each is a separate work item with its own
    // fix shape, and collapsing them into one "lift coverage" number is what
    // hides that four are cheap and one must not be fixed.
    const TIERS: [(&str, &str); 14] = [
        // Surfaced on 2026-11-04 when `pto.tadds` started lifting: rms_norm.pto
        // used to stop at the scalar add and now reaches the square root.
        ("unmodelled op reached once the scalar add lifted", "pto.trsqrt"),
        // Measured 2026-11-05 when rspto gained a chunked path: the first chunked
        // artefact assembles but the lift refuses it, because the loop's body
        // addresses the parent tensor at an offset derived from the induction
        // variable while this lift aliases the base pointer.
        ("whole-kernel call, geometry from an emit-time knob", "pto.get_block_idx"),
        ("cube/vector pair across a c2v FIFO", "cube and vector halves of ONE kernel"),
        ("gather with a part-dependent offset unit", "dtype `i32`"),
        ("cast chain whose own probes are unproven", "pto.tcvt"),
        ("strided tensor view", "is strided"),
        ("sub-view: no API form, and wrong on 910B2 silicon", "has no sub-view form"),
        ("the deliberately malformed fixture", "K disagrees"),
        // Upstream's own vPTO fixture, not ours: an AIV function attribute this
        // crate does not model. Dropping it silently would make a lossy
        // re-emission look lossless, so it refuses instead.
        ("upstream vPTO fixture: an unmodelled function attribute", "pto.aiv"),
        // The scalar-operand MULTIPLY tier is gone: `pto.tmuls` now emits
        // `tile_scale_f32` with the literal the artifact carried, so that tier
        // has no members. Removing its row exposed the case behind it, which is
        // why there is still a scalar tier -- `pto.tadds` has no tile-API arm at
        // all (the `__tile_adds_*` are buffer-API over `UbBuf`), and
        // `rms_norm.pto` refused on it only after the multiply lifted. The
        // per-tier floor is what caught the empty row instead of leaving one
        // that describes nothing.
        // NEEDS A NEW INTRINSIC. `tile_std` has no tile-to-tile copy at all --
        // `__tile_duplicate_*` fills from a scalar and `__tile_copy_l1_to_l0*`
        // is cube staging DMA -- so there is nothing to lower a `pto.tmov` to.
        ("tile-to-tile copy, which the tile API does not have", "no tile-to-tile copy wrapper"),
        // NEEDS AN OP MODEL. `pto.tci` (iota / constant-index tile) has no AST
        // meaning here, so top-k and RoPE both stop at it.
        ("constant-index tile, not modelled", "pto.tci"),
        // NEEDS AN OP MODEL. Row-wise argmax, same shape of gap as `pto.tci`.
        ("row-wise argmax, not modelled", "pto.trowargmax"),
        // A LANGUAGE CONSTRAINT, not a missing name: `Tile` is not `Copy`, and
        // the tile a compute op produced cannot be re-loaded the way one read
        // from memory can. Lifting it anyway would move the value twice.
        ("computed tile read twice, and `Tile` is not `Copy`", "is not `Copy`"),
        // The tier the compile gate created. All 9 members are matvec or
        // projection kernels where a row-reduce feeds a store, and in every
        // one it is the reduce output: `rows=8, cols=1` allocated against the
        // 512-byte fractal, `v_row=1, v_col=1` logically. The default path
        // used to emit these, taking the const generics from the ALLOCATED
        // extent -- `tile_store_f32::<8, 1>` for a `Tile<1, 1>` value -- and
        // all 9 failed with E0308 the first time anything compiled the output.
        // They are the single largest refusal class, and the one with the
        // clearest fix: thread the valid extent through, then prove on
        // hardware that the store writes one element and not eight.
        ("reduce output padded to the fractal, so the logical extent is smaller", "partial or dynamic v_row"),
    ];

    // The same root the walk uses, so a path prints relative instead of
    // absolute. These were two different roots until the walk was widened, and
    // the mismatch showed up as absolute paths in this very message.
    let root = shared::corpus_root();
    let mut files = Vec::new();
    shared::collect_pto_files(&shared::corpus_root(), &mut files);
    let mut by: std::collections::BTreeMap<Vec<u8>, Vec<std::path::PathBuf>> = Default::default();
    for f in files {
        if let Ok(b) = std::fs::read(&f) {
            by.entry(b).or_default().push(f);
        }
    }
    assert!(!by.is_empty(), "no .pto artifacts found");

    let mut lifted = 0usize;
    let mut per_tier = [0usize; TIERS.len()];
    let mut unexplained: Vec<String> = Vec::new();
    for (bytes, paths) in &by {
        let Ok(src) = std::str::from_utf8(bytes) else {
            continue;
        };
        let name = paths[0]
            .strip_prefix(&root)
            .unwrap_or(&paths[0])
            .to_string_lossy()
            .to_string();
        match ptoas_rs::convert_pto_to_rust(src) {
            Ok(_) => lifted += 1,
            Err(e) => match TIERS.iter().position(|(_, tok)| e.contains(tok)) {
                Some(i) => per_tier[i] += 1,
                None => unexplained.push(format!("{name}: {e}")),
            },
        }
    }

    assert!(
        unexplained.is_empty(),
        "{} artifact(s) are refused for a reason no measured tier covers. Either the \
         refusal changed and a tier token is stale, or there is a new case to measure:\n{}",
        unexplained.len(),
        unexplained.join("\n")
    );
    // Every tier must still have members: a tier that empties out means its
    // artifacts moved, and the partition above would then be certifying a
    // token nothing produces.
    for (i, (tier, tok)) in TIERS.iter().enumerate() {
        assert!(
            per_tier[i] > 0,
            "tier `{tier}` (token `{tok}`) now has no artifacts. If they lift now, say so \
             in the README; if they refuse differently, the token is stale"
        );
    }
    let refused: usize = per_tier.iter().sum();
    assert_eq!(
        lifted + refused,
        by.len(),
        "{lifted} lift + {refused} refused != {} distinct artifacts",
        by.len()
    );
    eprintln!(
        "{lifted} of {} distinct artifacts lift; {refused} refused across {} measured tiers",
        by.len(),
        TIERS.len()
    );
    for (i, (tier, _)) in TIERS.iter().enumerate() {
        eprintln!("    {:>2}  {tier}", per_tier[i]);
    }
}
