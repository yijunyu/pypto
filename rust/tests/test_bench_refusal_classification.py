# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every refusal is classified, and an unknown shape fails rather than hiding.

A coverage ratio says "7 of 20 reach PTO" and a reviewer cannot act on it,
because the thirteen absences are not one thing: some are this repository's
emitters being unfinished, some are the part refusing a tile that does not fit,
and one is the downstream shadow of another. These tests pin that the partition
stays computable, stays total, and keeps pointing at the right owner.

What they do NOT assert is that any particular count is good. The counts move as
the corpus and the emitters move. What must not move is that every refusal lands
in exactly one declared kind, that a derived refusal resolves to a real cause,
and that an asymmetry is confirmed against the other emitter's source rather
than assumed from its silence.
"""

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
sys.path.insert(0, str(BENCH / "shapes"))

import classify_refusals as C  # noqa: E402
from classify_refusals import (  # noqa: E402
    CHECKED_LIFT,
    DERIVED,
    EMITTER_DEFECT,
    EMITTER_GAP,
    EMITTER_SOURCE,
    HARDWARE_BOUND,
    NOT_A_PROBLEM,
    NOT_OURS,
    OURS,
    SHAPE_LIMIT,
    THE_CORPUS,
    asymmetries,
    classify,
    corpora,
    manifest,
    missing_arms,
    reconcile,
    refusals,
    supports,
)

KINDS = {EMITTER_GAP, EMITTER_DEFECT, SHAPE_LIMIT, HARDWARE_BOUND, CHECKED_LIFT, DERIVED}


def test_there_are_refusals_to_classify():
    """Guard against the suite passing because it found nothing.

    A classifier over an empty list satisfies every other test in this file.
    """
    rows = refusals()
    assert len(rows) > 20, f"only {len(rows)} refusals found; the corpus or the reader broke"
    assert len(corpora()) == 4


def test_every_refusal_lands_in_exactly_one_declared_kind():
    """Totality. The classifier has no catch-all, so this is also a smoke test.

    `classify()` raises on an unrecognised shape, which means a new refusal
    wording in any emitter fails here instead of being silently bucketed. That
    is deliberate: a classifier with an `other` bucket absorbs exactly the case
    nobody has looked at yet, and its totals keep looking complete.
    """
    for corpus, name, rep, kind, _, text in refusals():
        assert kind in KINDS, f"{corpus}/{name} [{rep}]: {kind} is not a declared kind"
        assert text.strip(), f"{corpus}/{name} [{rep}]: empty refusal text"


def test_an_unknown_refusal_shape_raises():
    """The no-catch-all property, asserted rather than trusted.

    This caught a real omission the first time it ran: the bench carries an
    `argmax over N rows` refusal, which is neither an emitter gap nor a
    hardware bound, and an earlier ad-hoc pass had put it in an `other` bucket
    where nobody would have looked at it.
    """
    with pytest.raises(ValueError, match="unclassified refusal"):
        classify("the emitter declined, for reasons of its own")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("mlir_to_pto has no arm for `__tile_embedding_f32`. Emitting it would", EMITTER_GAP),
        ("mlir_to_cpp has no arm for `__tile_topk_f32`. Emitting it would", EMITTER_GAP),
        ("matmul_i8: both operand loads must be deferred for blocked emit", EMITTER_DEFECT),
        ("blocked matmul: N=16160 must be a multiple of Nb=64", SHAPE_LIMIT),
        ("mul_mv_id_mxfp4_pk: pair count (80) must be a multiple of the 64 chunk", SHAPE_LIMIT),
        ("mlir_to_cpp: __tile_argmax_f32 over 8 rows -- the scan carries", SHAPE_LIMIT),
        ("UB overflow in kernel 'glm53_argmax_154880': buffer 0 needs", HARDWARE_BOUND),
        ("NPU UB budget (arch a2a3): tile of 35840B", HARDWARE_BOUND),
        ("NPU tile bound (arch a2a3): row_major data tile", HARDWARE_BOUND),
        ("no PTO to lift", DERIVED),
        ("refused bench/kimi_k3/pto/rope_k3_1x64.pto", CHECKED_LIFT),
    ],
)
def test_each_kind_has_a_worked_example(text, expected):
    """One sample per kind, so a pattern that stops matching is caught here.

    A screen with no sample of its own hits cannot tell you it went blind -- it
    reports zero and reads as clean.
    """
    kind, _ = classify(text)
    assert kind is expected


def test_the_defect_pattern_wins_over_the_shape_patterns():
    """Order dependence, pinned where it bites.

    The `matmul_i8` refusal is also about a blocked matmul, so a `blocked
    matmul` pattern placed first would swallow it and file a defect in our own
    pass as a shape that needs padding -- sending someone to change a corpus
    extent that is already correct.
    """
    kind, _ = classify("matmul_i8: both operand loads must be deferred for blocked emit")
    assert kind is EMITTER_DEFECT
    kind, _ = classify("blocked matmul: N=16160 must be a multiple of Nb=64")
    assert kind is SHAPE_LIMIT


def test_the_kinds_partition_into_owners_with_nothing_left_over():
    """The four owner buckets must cover every kind exactly once.

    The whole point of the partition is routing. A kind that belongs to no
    bucket, or to two, means a refusal has no owner or two owners, and the
    summary that reads "work in this repository: N" becomes unfalsifiable.
    """
    buckets = list(OURS) + list(THE_CORPUS) + list(NOT_OURS) + list(NOT_A_PROBLEM)
    assert sorted(buckets) == sorted(KINDS)
    assert len(buckets) == len(set(buckets)), "a kind appears in two owner buckets"


def test_every_derived_refusal_resolves_to_a_recorded_cause():
    """A consequence with no cause is a manifest asserting something it cannot show.

    `no PTO to lift` is only honest if the kernel truly has no `.pto` AND the
    PTO stage recorded why. Either half failing means the refusal is unmoored.
    """
    problems = reconcile()
    assert not problems, "\n".join(problems)


def test_a_derived_refusal_whose_cause_is_missing_is_caught(tmp_path, monkeypatch):
    """Prove the reconciliation can fire.

    Seeded by making one kernel claim there was no PTO to lift while carrying a
    `.pto` -- which is the self-contradiction the check exists for.
    """
    real = C.manifest

    def fake(corpus):
        m = real(corpus)
        if corpus == "glm_5_3":
            name = sorted(m["kernels"])[0]
            m["kernels"][name]["tile_rs_refused"] = "no PTO to lift"
            m["kernels"][name]["pto"] = "pto/somewhere.pto"
        return m

    monkeypatch.setattr(C, "manifest", fake)
    problems = C.reconcile()
    assert problems, "a kernel claiming no PTO while holding one was not caught"
    assert "no PTO to lift" in problems[0]


def test_missing_arms_name_an_operation_and_cost_at_least_one_kernel():
    """The actionable list is specific, not a count."""
    arms = missing_arms()
    assert arms, "no emitter gaps found at all, which would be a first"
    for op, where in arms.items():
        assert op.startswith("__tile_"), f"{op} is not an intrinsic name"
        assert where, f"{op} is listed as missing but costs no kernel"
        for corpus, name, rep in where:
            assert corpus in corpora()
            assert name in manifest(corpus)["kernels"]


def test_an_asymmetry_is_confirmed_against_the_other_emitters_source():
    """Support must be READ, not inferred from the absence of a refusal.

    Two defects this pins, both of which the first version of `asymmetries()`
    had. It reported the emitter named in the refusal text as the one that HAS
    the arm, when that is the one MISSING it -- every row backwards. And it took
    the other emitter's silence as support, when an operation no corpus kernel
    exercises draws no refusal from either emitter.
    """
    for op, (missing, present) in asymmetries().items():
        assert missing and present, f"{op}: an asymmetry needs both sides"
        assert not (set(missing) & set(present)), f"{op}: an emitter cannot be both"
        for emitter in missing:
            assert not supports(emitter, op), (
                f"{op} is reported missing from {emitter}, but {emitter}'s source "
                "dispatches on it -- the sides are the wrong way round"
            )
        for emitter in present:
            assert supports(emitter, op), (
                f"{op} is reported present in {emitter} without a dispatch site "
                "in its source. Silence is not support."
            )


def test_silence_is_not_support_shown_on_a_case_the_corpus_does_not_have():
    """The property the corpus cannot currently test, tested on a built case.

    `asymmetries()` must confirm the other emitter's support by READING it. A
    version that instead assumed silence meant support produces the IDENTICAL
    report on today's bench, because every operation refused by exactly one
    emitter happens to be genuinely supported by the other, and the one
    operation neither supports is refused by both -- so it never reaches the
    inference. The mutant was undetectable from the corpus.

    That is the trap: the property would have been pinned by the implementation
    and by nothing else, and the next person to simplify the function would get
    a green suite. So the discriminating case is constructed rather than waited
    for: an operation refused by ONE emitter which the OTHER does not support
    either. The honest answer is that this is not an asymmetry at all.
    """
    op = "__tile_mul_mv_id_q8_0"  # dispatched by neither emitter
    assert supports("mlir_to_pto", op) is False
    assert supports("mlir_to_cpp", op) is False

    rows = [
        (
            "glm_5_3",
            "synthetic",
            "pto",
            EMITTER_GAP,
            op,
            f"mlir_to_pto has no arm for `{op}`. Emitting it would drop it.",
        )
    ]
    assert asymmetries(rows) == {}, (
        f"{op} was reported as an asymmetry, but the AscendC emitter does not "
        "support it either. Only one emitter refused it, and that silence was "
        "read as support."
    )


def test_support_requires_a_dispatch_site_not_just_the_name():
    """The name-list trap, asserted against a MEASURED instance.

    Both emitters carry a list of intrinsic names, and in both the list is
    WIDER than the dispatch: measured here, 6 names in the PTO emitter's list
    and 16 in the AscendC emitter's have no dispatch site at all. So a check for
    the bare name would report support for arms that lower nothing, which is the
    documented failure of an emitter claiming an operation because only the
    intrinsic existed one layer down.

    `__tile_mul_mv_id_q8_0` is one such name: present in the PTO emitter's list,
    dispatched by neither emitter. If an arm is ever written for it this test
    fails and wants a different example, which is the correct outcome -- an
    example that stops being an example is not a silent pass.
    """
    listed_only = "__tile_mul_mv_id_q8_0"
    pto = EMITTER_SOURCE["mlir_to_pto"]
    assert f'"{listed_only}",' in pto.read_text(encoding="utf-8"), (
        f"{listed_only} is no longer in the PTO emitter's name list, so this "
        "test has lost the instance that makes the trap real"
    )
    assert supports("mlir_to_pto", listed_only) is False, (
        f"{listed_only} now has a dispatch site; pick another list-only name"
    )

    # The bare `__tile_` prefix guard must not be mistaken for support either.
    # The AscendC emitter routes into its dispatch through `starts_with("__tile_")`,
    # which is a prefix of EVERY intrinsic, so a `starts_with`-aware check would
    # report that emitter as supporting all of them.
    assert supports("mlir_to_cpp", "__tile_attention_decode_f32") is False
    assert supports("mlir_to_pto", "__tile_attention_decode_f32") is False


def test_the_embedding_is_refused_by_both_emitters_for_different_reasons():
    """`__tile_embedding_f32` is not an asymmetry, and the reason is measured.

    It used to be the one operation that ran the other way: present in the
    AscendC emitter and absent from PTO, so the PTO column was short for a
    reason that looked like an unfinished arm. Reading what the AscendC arm
    actually emits shows the "support" was a dispatch site and nothing more: it
    gathers rows WITHIN the weight tile it was handed, while a real vocabulary
    is 2.5-4.7 GB against a 192 KiB unified buffer, and every corpus fixture
    loads one row and indexes it by the runtime row id. So the arm now refuses
    that shape, and the operation is refused by BOTH emitters -- PTO for a
    missing GM-driven gather, AscendC for the table not fitting any buffer.

    Asserts the partition rather than a count, so it keeps holding when either
    side gains a real implementation: if PTO grows the arm, or AscendC grows a
    GM-driven row copy, this test fails and the asymmetry list is where the new
    fact belongs.
    """
    op = "__tile_embedding_f32"
    assert op not in asymmetries(), (
        f"{op} is reported as an asymmetry, but the emitter said to support it "
        f"refuses every corpus vocabulary"
    )
    rows = [r for r in refusals() if r[4] == op or (r[4] is None and op in (r[5] or ""))]
    kinds = {(r[2], r[3]) for r in rows if r[1].startswith("embedding")}
    assert ("pto", EMITTER_GAP) in kinds, f"PTO should refuse it for a missing arm: {kinds}"
    assert ("ascendc", HARDWARE_BOUND) in kinds, (
        f"AscendC should refuse it as a hardware bound, not emit a gather that reads past its tile: {kinds}"
    )


def test_the_decode_attention_gap_is_a_missing_extent_not_a_missing_arm():
    """Four of the thirteen EMITTER_GAP rows are one operation, and writing an
    arm would not close them.

    `__tile_attention_decode_f32` is refused by BOTH emitters, so it costs four
    rows -- two kernels times two columns -- and it reads as the largest single
    item on the list. It is not an arm: the call site cannot say how many keys to
    attend over.

    The declared signature is `(dst, q, k, v, rows, cols)`, where `rows`/`cols`
    are the head count and head width; the caches are `(NKV, max_seq, DH)` at
    RUNTIME, and `max_seq` is not a parameter. And every fixture that exercises
    it loads K and V at Q's own extent, so the tiles hold one key, not a cache.
    An emitter could only emit "out = v" (the softmax over one score is 1),
    which is correct for that shape and wrong for every real decode step.

    So the row belongs to the operation's signature, not to either emitter. This
    test holds that still: if a fixture starts carrying a cache extent, or the
    declaration grows a cache-length parameter, the assumption has changed and
    the row becomes real work on the emitter side.
    """
    op = "__tile_attention_decode_f32"
    rows = [r for r in refusals() if r[4] == op]
    assert rows, f"{op} is no longer refused by either emitter; the gap list is stale"
    assert {r[3] for r in rows} == {EMITTER_GAP}, (
        f"{op} should be a plain missing arm in both columns, got {sorted({r[3] for r in rows})}"
    )

    # The declaration takes no cache length. Parameter names, read from the DSL.
    decl = (ROOT / "rust/crates/tile_std/src/tile.rs").read_text(encoding="utf-8")
    start = decl.index(f"pub fn {op}(")
    sig = decl[start : decl.index("-> u32;", start)]
    params = [p.split(":")[0].strip() for p in sig[sig.index("(") + 1 :].split(",") if ":" in p]
    assert params == ["dst", "q", "k", "v", "rows", "cols"], (
        f"{op}'s signature changed to {params}; if it now carries a cache extent, the four "
        f"EMITTER_GAP rows are real work and this test is the thing to delete"
    )

    # And the fixtures pass K/V at Q's extent: one key, no cache rows.
    fixtures = sorted(BENCH.glob("*/tile/attention_decode_*.tile"))
    assert fixtures, "no fixture exercises the operation; the rows above came from nowhere"
    for path in fixtures:
        text = path.read_text(encoding="utf-8")
        loads = re.findall(r"%\w+ = llvm\.call @__tile_load_f32\(%\w+, (%\w+), (%\w+)\)", text)
        assert len(loads) == 3, f"{path.name}: expected three loads, got {loads}"
        extents = {tuple(e) for e in loads}
        assert len(extents) == 1, (
            f"{path.name}: K/V now carry a different extent from Q ({extents}), so the fixture "
            f"models a cache and the emitter gap is real work"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
