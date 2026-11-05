// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
//! Where each view lands in the buffer the lift aliases.
//!
//! Its own module because three passes need the SAME answer: the
//! `PartitionView` emitter arm, which folds a constant offset into the
//! address; `emitter::barrier_blocker`, which can only tell two stores apart
//! by their addresses; and `linearize`, which may only re-load a tile when no
//! store can have touched the window it came from. Three copies of this
//! arithmetic would be three chances to disagree about whether two accesses
//! collide, and the one that disagreed would be the one that emitted a wrong
//! kernel at exit 0.

use crate::ast::*;
use std::collections::BTreeMap;

/// The span of the aliased buffer a `pto.partition_view` addresses.
///
/// The tile API has no view abstraction: the lift aliases the base pointer and
/// every `tile_load_*`/`tile_store_*` walks `rows * cols` CONTIGUOUS elements
/// from it. So a view is fully described, for this lift's purposes, by which
/// pointer it is rooted at and which elements of it the call will touch --
/// which is also the only thing that can say whether two stores collide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Window {
    /// The PTO SSA of the pointer the view chain is rooted at, which is the
    /// `%ptr` of the `pto.make_tensor_view`. Two windows are only comparable
    /// when this matches: two distinct pointer ARGUMENTS may alias at the call
    /// site, and nothing in the module says they do not.
    pub base: Ssa,
    /// Elements from `base` to the first element this view addresses.
    pub off: i64,
    /// Elements this view spans.
    pub len: i64,
}

/// Resolve every rank-2 view in `ops` to a [`Window`], or to the reason it
/// cannot be one.
///
/// `Err` is carried rather than dropped so the `PartitionView` arm can report
/// the specific reason instead of a generic "not resolvable": a loop-variant
/// offset, a window that leaves the parent, and a multi-row window narrower
/// than the parent are three different problems with three different answers.
///
/// A view missing from the map entirely is one whose PARENT this lift never
/// modelled -- `make_tensor_view` refuses a non-contiguous or rank-N parent
/// before any partition of it is reached, so in practice the absent case is a
/// partition of something that is not a tensor view at all.
///
/// Deliberately does NOT recurse into `scf.for` / `scf.if` bodies. A view
/// inside a loop would be recorded once while the loop writes through it on
/// every iteration, so a store at a constant offset in a loop body would look
/// like a single store that conflicts with nothing. Both region arms refuse
/// before emitting a body today, so the omission is unreachable; leaving it out
/// means the first change that starts emitting regions gets a refusal (every
/// lookup misses, and [`provably_disjoint`] answers `false`) rather than a
/// quietly wrong answer.
pub fn view_windows(
    ops: &[PtoOp],
    consts: &BTreeMap<Ssa, i64>,
) -> BTreeMap<Ssa, Result<Window, String>> {
    // (base pointer, rows, cols) per tensor view. `row_stride` is not carried
    // because `MakeTensorView` refuses any view whose row stride is not `cols`,
    // so it is `cols` by construction -- and reading it off `cols` here keeps
    // the two arms from drifting apart about which one the flat offset uses.
    let mut parents: BTreeMap<Ssa, (Ssa, i64, i64)> = BTreeMap::new();
    let mut out: BTreeMap<Ssa, Result<Window, String>> = BTreeMap::new();
    for op in ops {
        match op {
            PtoOp::MakeTensorView {
                dst,
                ptr,
                shape,
                strides,
                layout,
                ..
            } => {
                // Mirror of the `MakeTensorView` arm's own guards: a rank-N or
                // fractal view has no rows/cols, and a non-contiguous one makes
                // the flat offset below meaningless. Both are refused there, so
                // recording nothing here is enough.
                if layout.is_some() || shape.len() != 2 || strides.len() != 2 {
                    continue;
                }
                let (Some(r), Some(c), Some(rs), Some(cs)) = (
                    crate::emitter::resolve(consts, &shape[0]),
                    crate::emitter::resolve(consts, &shape[1]),
                    crate::emitter::resolve(consts, &strides[0]),
                    crate::emitter::resolve(consts, &strides[1]),
                ) else {
                    continue;
                };
                if rs != c || cs != 1 {
                    continue;
                }
                parents.insert(dst.clone(), (ptr.clone(), r, c));
                out.insert(
                    dst.clone(),
                    Ok(Window {
                        base: ptr.clone(),
                        off: 0,
                        len: r.saturating_mul(c),
                    }),
                );
            }
            PtoOp::PartitionView {
                dst,
                src,
                offsets,
                sizes,
                ..
            } => {
                let Some((base, prows, pcols)) = parents.get(src).cloned() else {
                    continue;
                };
                let resolved = [
                    crate::emitter::resolve(consts, &offsets[0]),
                    crate::emitter::resolve(consts, &offsets[1]),
                    crate::emitter::resolve(consts, &sizes[0]),
                    crate::emitter::resolve(consts, &sizes[1]),
                ];
                let [Some(r), Some(c), Some(sr), Some(sc)] = resolved else {
                    // The common case by far: `offsets = [%r, %c0]` inside an
                    // `scf.for`, where `%r` is the induction variable. No
                    // constant map can resolve it, and the window it names is a
                    // different one each iteration.
                    out.insert(
                        dst.clone(),
                        Err(format!(
                            "partition view `{}` has an operand this lift cannot resolve to a \
                             constant (offsets [{}, {}], sizes [{}, {}]); a loop-variant window \
                             is a different span each iteration and the lift has one address",
                            dst.0, offsets[0], offsets[1], sizes[0], sizes[1]
                        )),
                    );
                    continue;
                };
                // A window narrower than its parent and taller than one row is
                // STRIDED in the parent, and the tile call reads contiguously.
                // That is wrong at the origin too -- this is not a consequence
                // of admitting an offset -- but admitting offsets is what makes
                // it reachable, so it is checked here rather than left for the
                // next reader. 26 corpus instances, all in the GMM cube files.
                if sr > 1 && sc != pcols {
                    out.insert(
                        dst.clone(),
                        Err(format!(
                            "partition view `{}` is a {sr}x{sc} window of a {prows}x{pcols} \
                             parent; its rows are {pcols} elements apart in memory while the \
                             tile call reads {} contiguous ones, so the lift would read past \
                             each row",
                            dst.0,
                            sr.saturating_mul(sc),
                        )),
                    );
                    continue;
                }
                if r < 0 || c < 0 || r + sr > prows || c + sc > pcols {
                    out.insert(
                        dst.clone(),
                        Err(format!(
                            "partition view `{}` is a {sr}x{sc} window at [{r}, {c}] of a \
                             {prows}x{pcols} parent, which leaves the parent; the lift would \
                             emit an out-of-bounds address",
                            dst.0
                        )),
                    );
                    continue;
                }
                out.insert(
                    dst.clone(),
                    Ok(Window {
                        base,
                        off: r.saturating_mul(pcols).saturating_add(c),
                        len: sr.saturating_mul(sc),
                    }),
                );
            }
            _ => {}
        }
    }
    out
}

/// Whether two windows are known not to touch the same element.
///
/// `false` whenever anything is unknown -- a missing window, an unresolved one,
/// or two different base pointers, which may alias at the call site.
pub fn provably_disjoint(
    windows: &BTreeMap<Ssa, Result<Window, String>>,
    a: &Ssa,
    b: &Ssa,
) -> bool {
    let (Some(Ok(x)), Some(Ok(y))) = (windows.get(a), windows.get(b)) else {
        return false;
    };
    x.base == y.base && (x.off + x.len <= y.off || y.off + y.len <= x.off)
}
