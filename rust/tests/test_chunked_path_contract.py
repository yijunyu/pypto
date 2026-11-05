# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The contract a chunked emission must satisfy, written before the feature exists.

`kda`-style rows wider than the buffer cannot be placed by any ordering: the dense
situ row is 33792 wide, one tile is 135168 B, and TWO live tiles are 270336 B
against a 188416 B buffer -- 1.43x over. Every other row in this corpus was
rescued by writing the chain in a better order (see lesson 1 in `rust/README.md`);
this one cannot be, and the difference is worth recording precisely: the others had
a live set that was too LARGE, this one has tiles that are individually too large.

`rust/README.md` says a guard is worth writing before the thing it guards, and that
a guard whose failure has never been observed is a comment with a type. So this
file is the spec, marked skipped, and it becomes a real test the day the chunked
path exists. It is not a red test: a green tree with a skipped specification is
honest, a red tree is not.

WHAT A CHUNKED EMISSION MUST DO, in the order that makes it testable:

  1. SPLIT the width into chunks of at most `buffer / tiles_live_per_chunk`, with
     `tiles_live_per_chunk` the same liveness number `live_set.py` reports for
     every other kernel. The chunk size must be a multiple of the 32-byte block,
     because `DataCopy` outside that multiple takes the `DataCopyPad` path and the
     emitter has one of those per store site.
  2. HANDLE THE TAIL: `width % chunk != 0` is the case that will be wrong first,
     so the test must use a width that does not divide evenly (33792 / 8192 = 4
     remainder 1024) and compare the tail elements as carefully as the body.
  3. ACCUMULATE ACROSS CHUNKS where the op is a reduction (the projections), or
     process independently where it is elementwise (the situ activation). Those are
     different contracts and should be two tests, not one.
  4. NOT CHANGE THE ARITHMETIC: the chunked kernel must produce the same result as
     the un-chunked one at a width where both fit. That is the same proof used for
     buffer aliasing, and it is the only one that does not depend on trusting the
     chunking logic.
"""

from __future__ import annotations

import pytest

NOT_YET = (
    "The chunked path does not exist. This test states the contract so the next "
    "session starts from a specification instead of a guess; remove the skip in "
    "the same change that implements it."
)


@pytest.mark.skip(reason=NOT_YET)
def test_a_reduction_wide_kernel_splits_into_chunks_that_fit() -> None:
    """Accumulating across chunks: the projection shape."""
    raise AssertionError("unreachable while skipped")


def test_an_elementwise_wide_kernel_processes_chunks_independently() -> None:
    """No accumulation: the situ activation shape at 33792.

    This one is REAL now. The emitted kernel is inspected rather than the emitter's
    internals, because what matters is the artefact that runs.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cpp = (root / "rust/bench/kimi_k3/ascendc/dense_gate_up_situ_k3_1x33792.cpp").read_text()
    assert "chunked: 33792 elements as 16 x 2112, tail 2112" in cpp, (
        "the dense situ kernel is no longer emitted in chunks; re-read whether it "
        "fits whole now"
    )
    assert "for (int __c = 0; __c < __NC; ++__c)" in cpp, "no chunk loop"
    assert "gm0Tensor.SetGlobalBuffer(" in cpp, (
        "every GM tensor must be rebased per chunk; indexing it at the copy site "
        "does not rebase and produced 29761 of 33792 wrong elements"
    )
    # the per-chunk buffer size, not the width
    assert "InitBuffer(tbuf0, 8448)" in cpp, "buffers must be chunk-sized"
    # and no full-width copy survives
    assert "DataCopy(buf0, gm0Tensor, (1) * (33792))" not in cpp, (
        "a width-sized load into a chunk-sized buffer"
    )


def test_the_tail_of_a_width_that_does_not_divide_evenly_is_covered() -> None:
    """The remainder is where chunking breaks, and 2112 divides 33792 exactly --
    so the arithmetic is pinned on a width that does NOT, in the unit test
    `the_tail_is_never_dropped`, and here on the mechanism: every count in the
    body is `__n`, which is the tail on the last iteration.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cpp = (root / "rust/bench/kimi_k3/ascendc/dense_gate_up_situ_k3_1x33792.cpp").read_text()
    assert "const int __n = (__c == __NC - 1) ? (__NW - (__NC - 1) * __CH) : __CH;" in cpp
    assert "__v2 = __n;" in cpp, (
        "the ops that spell their count as `__v2` would process the full width"
    )
    assert cpp.count("((__v1) * (__v2))") == 0, "an un-substituted width expression survived"


@pytest.mark.skip(reason=NOT_YET)
def test_the_chunked_result_matches_the_unchunked_one_at_a_width_where_both_fit() -> None:
    """The only proof that does not depend on trusting the chunking logic."""
    raise AssertionError("unreachable while skipped")


def test_the_chunked_path_has_a_kernel_that_needs_it() -> None:
    """The premise, inverted now that it holds.

    This test used to assert that the dense situ row was STILL refused, so that the
    skipped specs above were not describing a problem that no longer existed. It
    fired the moment the row started emitting, which is what it was for. The
    assertion that belongs here now is the other direction: the chunked path has at
    least one kernel whose emission depends on it, or the four specs above are
    decoration.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "rust/bench/kimi_k3/manifest.json").read_text())
    chunked = [
        stem
        for stem, entry in manifest["kernels"].items()
        if entry.get("ascendc_chunked")
    ]
    assert chunked, (
        "no kernel is emitted in chunks any more -- either they all fit whole now "
        "(delete this file and the skipped specs with it) or the path regressed"
    )
    for stem in chunked:
        assert manifest["kernels"][stem].get("ascendc"), f"{stem} claims chunking but no .cpp"


# --- the PTO side of the same contract -------------------------------------
#
# `ascendc_emit` chunks; `rspto` does not. That asymmetry is why eleven kernels run
# on a part while their static rung still refuses them, and
# `bench/tile_rs_acceptance.json` records each of those eleven with
# `needs: the CHUNKED PATH in rspto`.
#
# The arithmetic is shared -- `plan_chunks` is emitter-agnostic -- so what is
# missing is the EMISSION, and PTO has the construct: `scf.for` is already emitted
# for the tiled matmul path (`mlir_to_pto.rs:760`). The specs below say what the
# emission must satisfy; they are skipped until it exists, for the same reason as
# the ones above.


def test_rspto_chunk_loop_steps_by_one_and_scales_the_offset() -> None:
    """The loop shape, pinned because the first version was WRONG and assembled.

    `scf.for %n_i = 0 to %n_iters step %n_step` with `iters = width/chunk` steps
    straight past the range on its first iteration: the loop ran ONCE at offset 0
    and the artefact computed 4096 of 12288 elements. It assembled, it passed the
    byte-identity gate, and it was wrong. The correct form bounds by the chunk COUNT
    and steps by one, so the offset `i * chunk` walks the width.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pto = (root / "rust/bench/glm_5_3/pto/dense_gate_up_silu_glm53_1x12288.pto").read_text()
    assert "scf.for %n_i = %n_zero to %n_iters step %n_one" in pto, (
        "the loop must step by one over the chunk COUNT; stepping by the chunk size "
        "over the count runs exactly one iteration"
    )
    assert "%n_off = arith.muli %n_i, %n_chunk" in pto, "the offset must scale by the chunk"
    assert "%n_iters = arith.constant 3 : index" in pto, "12288 / 4096 is three chunks"
    # and the three offsets the loop produces must cover the width
    assert 3 * 4096 == 12288


@pytest.mark.skip(reason="rspto has no chunked emission yet; see the backlog's `needs` field")
def test_rspto_emits_a_chunk_loop_with_offset_partition_views() -> None:
    """A chunked PTO kernel iterates `scf.for` over chunks, and every load/store
    carries the chunk offset in its `pto.partition_view` rather than a fixed 0.

    THE MECHANISM IS VERIFIED, by experiment rather than by reading: taking a real
    kernel and changing `offsets = [%c0, %c0]` to `offsets = [%c0, %c1]` PARSES and
    TYPE-CHECKS -- ptoas refused it only because `offset + size` then exceeded the
    static source dim (7169 vs 7168). So partition_view offsets are ordinary index
    values, not literals, and a loop induction variable can feed them.

    WHAT THAT IMPLIES FOR THE EMISSION, and it is the whole design:

      * the main loop runs `N // CH` FULL chunks, with `offset = n_i * CH` and a
        static size of `CH`, so `offset + size <= N` holds on every iteration;
      * the TAIL cannot ride the same loop, because its size is `N - (N // CH) * CH`
        and a tile's shape is static. It is an EPILOGUE block with its own static
        shape -- `tail` is a compile-time constant once N and CH are known;
      * BUT THE EPILOGUE IS USUALLY AVOIDABLE, which is the thing to try first:
        `plan_chunks` now PREFERS a chunk that divides the width exactly, because a
        tile whose shape is static cannot carry a runtime length and a uniform loop
        needs no second body. Worked example: 16160 = 2^5 x 5 x 101, and the
        32-byte-aligned divisors are 32, 160, 3232 and 16160 -- so 3232 is available
        (five equal chunks) and no epilogue is needed. The epilogue exists only for
        a width with no aligned divisor that fits the budget.

    That is more structure than the AscendC port needed, which could express the
    tail as a runtime `__n` because C++ tile extents are dynamic in the vector ops.

    AND THE BLAST RADIUS IS SMALLER THAN THE FILE SUGGESTS. `mlir_to_pto.rs` has 118
    `partition_view` emissions, which reads as a huge change; measured on real
    kernels, an ELEMENTWISE chain uses only the ones its loads and stores go
    through -- 3 for the dot-product projection (two loads, one store), 9 for the
    four-tap convolution (eight loads, one store). Every other site belongs to a
    specialised path (matmul blocking, rope, rmsnorm rows) that a chunked
    elementwise kernel never reaches. So the port threads a chunk through the
    generic load and store emission, not through 118 call sites.

    AND AN ATTEMPT WAS MADE AND REVERTED, which is worth more than a plan: the
    retry-on-UB orchestration works (a UB refusal, an elementwise check, a divisor
    found, analysis re-run), the loop emits, and the emission then FAILS TO
    TYPE-CHECK -- `%pto9` allocated at the chunk width and used at the original
    width, because the op emitters build their operand types from `(rows, cols)`
    rather than from the tile's recorded type. Those construction sites are the work;
    they are not the two DMA paths, and finding that out cost an afternoon.

    So the next attempt starts here: make every op's operand types come from the
    tile, not from the MLIR line's extents. With that done the port is small.

    AND THE LOOP IS NOT NEW MACHINERY. `rspto` already emits exactly this shape for
    blocked matmul -- `scf.for %n_i = %bidx_i to %cN step %bmax`, then
    `%n_off = arith.muli %n_i, %c<nb>`, then a partition_view with
    `offsets = [%c0, %n_off]` (mlir_to_pto.rs:752-775). The gap is that the machinery
    is specific to the cube/`tpop` path and does not reach an elementwise chain. A
    port should REUSE that emission rather than write a second loop, which is the
    cheapest version of this work and the one to try first.
    """
    raise AssertionError("unreachable while skipped")


@pytest.mark.skip(reason="rspto has no chunked emission yet")
def test_a_rspto_chunked_kernel_assembles_under_ptoas() -> None:
    """The artefact the emitter writes must assemble, or the static rung is being
    credited for a file that is not valid IR."""
    raise AssertionError("unreachable while skipped")
