# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Check every corpus call against the intrinsic the DSL declares.

The corpus `.tile` files are hand-captured MLIR, and nothing was comparing their
calls to `tile_std`'s declarations. Three were wrong, and each cost something
different:

`playground16/matmul.tile`
    Passed 5 operands where `__tile_matmul_f16(dst, a, b, m, k, n)` declares 6,
    omitting the leading `dst`. The PTO emitter has a SECOND declared contract
    at 5 operands -- a 5-arg cuTile form `(a, b, acc, m, n)` -- so the call did
    not fail; it was silently reinterpreted under the other contract, which
    parsed `m` as 64 where the kernel means 1. The emitted kernel was correct
    anyway, because the accumulator's shape comes from the A tile's own rows and
    not from that parsed `m`. Right by luck. The cost was two dead full-operand
    UB tiles and their DMA loads: with the call corrected the emitter defers
    both operand loads and the emission drops four operations, with a
    byte-identical `tmatmul`/`tstore` core.

`playground16/attn_gqa.tile`
    Passed `(tq, tk, tv, seq, dim, nh, nkv, causal)` where
    `__tile_attention_gqa_f32(dst, q, k, v, heads_q, heads_kv, seq_len,
    head_dim)` is declared. **The arity matches**, 8 against 8, so an
    arity-only check reports it clean -- which is why this file checks more than
    arity. Every operand was shifted by one and a `causal` operand was appended
    that the surviving declaration does not carry at all. The AscendC emitter
    refused it ("operand 3 (v) is %seq, which is not a tile"); the PTO emitter
    accepted it and emitted attention over the wrong operands, with the K/V
    tiles' real 32x64 shape appearing nowhere in the output.

`deepseek_v41/gemm_i8_v41_1x5120x2304.tile`
    Passed 6 operands where the i8 dequant matmul declares 7, dropping the
    `scale: *const f32` and passing the A tile as `dst`. Its refusal blamed one
    of our own passes -- "detect_blocked_matmul_loads must mark i8 matmul
    inputs" -- and that was a misdiagnosis caused by the malformed input. With
    the call corrected the refusal becomes the true one: on the 910B2 cube, M
    must be a multiple of 16, so a decode-shape M=1 i8 GEMM needs padding. The
    corpus already carries the padded M=16 form.

# The three rules, and why each is derived rather than listed

1. **Arity** must equal the declaration's parameter count.
2. For an intrinsic whose first declared parameter is named `dst`, **argument 0
   must be a constant**, never a tile-producing SSA. `dst` is first in 89 of the
   142 declarations, so this is read per-intrinsic from the DSL rather than
   assumed universal. This is the rule that catches an operand shift at correct
   arity.
3. A parameter declared as a **pointer** (`*const`/`*mut`) must receive a
   pointer-typed argument, judged by the call's own MLIR type signature.

Nothing here is hand-listed per intrinsic: all three rules read `tile_std`'s
declarations, so an intrinsic whose signature changes is re-checked against the
new one rather than against a copy that has to be remembered.

# What it deliberately does NOT check

Tile-versus-extent for the operands AFTER `dst`. In the DSL a tile handle and an
extent are both `u32`, so the declaration cannot distinguish them and neither
can this. Rule 2 catches the shift at the only position where the DSL's own
naming makes the intent unambiguous. A stronger check would need the DSL to
carry the distinction in a type, which is the real fix and is not this file's to
make.
"""

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
# bench lives at rust/bench, so the REPOSITORY root is one more level up.
ROOT = BENCH.parent.parent
DSL = ROOT / "rust/crates/tile_std/src/tile.rs"

CALL = re.compile(r"llvm\.call @(__tile_\w+)\((?P<args>.*)\)\s*:\s*\((?P<types>.*?)\)")


def declarations(text=None):
    """`{intrinsic: [(param_name, param_type)]}` read from the DSL."""
    text = DSL.read_text(encoding="utf-8") if text is None else text
    out = {}
    for m in re.finditer(r"pub fn (__tile_\w+)\(([^)]*)\)", text):
        params = []
        for raw in m.group(2).split(","):
            raw = raw.strip()
            if not raw:
                continue
            name, _, ty = raw.partition(":")
            params.append((name.strip(), ty.strip()))
        out[m.group(1)] = params
    return out


def split_top_level(s):
    """Split on commas not nested inside brackets."""
    depth, cur, out = 0, "", []
    for ch in s:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


def constants_and_tiles(lines):
    """`(constants, tiles)` -- the SSA names each category defines in one function."""
    constants, tiles = set(), set()
    for line in lines:
        m = re.match(r"\s*(%[\w.]+)\s*=\s*llvm\.mlir\.constant", line)
        if m:
            constants.add(m.group(1))
            continue
        m = re.match(r"\s*(%[\w.]+)\s*=\s*llvm\.call @__tile_", line)
        if m:
            tiles.add(m.group(1))
    return constants, tiles


def check_file(path, decls):
    """Return one message per non-conforming call in `path`."""
    rel = path.relative_to(ROOT) if ROOT in path.parents else path
    lines = path.read_text(encoding="utf-8").splitlines()
    constants, tiles = constants_and_tiles(lines)
    findings = []

    for number, line in enumerate(lines, start=1):
        m = CALL.search(line)
        if not m:
            continue
        name = m.group(1)
        params = decls.get(name)
        if params is None:
            findings.append(f"{rel}:{number}: `{name}` is not declared in tile_std")
            continue
        args = split_top_level(m.group("args"))
        types = split_top_level(m.group("types"))

        # 1. arity
        if len(args) != len(params):
            findings.append(
                f"{rel}:{number}: `{name}` takes {len(params)} operand(s) "
                f"({', '.join(p for p, _ in params)}) and the call passes "
                f"{len(args)} -- every operand after the first missing one is shifted"
            )
            continue

        # 2. a declared `dst` is either a constant (allocate a fresh tile) or the
        #    SAME tile as the next operand (destination-passing / in-place). A
        #    tile that is neither means the operands are shifted by one.
        if params and params[0][0] == "dst" and len(args) >= 2:
            dst, nxt = args[0], args[1]
            if dst in tiles and dst != nxt:
                findings.append(
                    f"{rel}:{number}: `{name}` declares operand 0 as `dst`, and the "
                    f"call passes the tile `{dst}` while operand 1 "
                    f"(`{params[1][0]}`) is `{nxt}`. A `dst` is either a constant "
                    "or the same tile as the operand after it; a different tile "
                    "means every operand is shifted by one -- which arity alone "
                    "cannot see"
                )

        # 3. a pointer parameter needs a pointer argument
        for i, (pname, ptype) in enumerate(params):
            if "*const" not in ptype and "*mut" not in ptype:
                continue
            if i >= len(types):
                continue
            if "ptr" not in types[i]:
                findings.append(
                    f"{rel}:{number}: `{name}` declares `{pname}: {ptype}` at "
                    f"operand {i} and the call passes `{args[i]}` typed "
                    f"`{types[i]}`, which is not a pointer"
                )

    return findings


def captures():
    """The top-level `tile/` files: FAITHFUL captures of a real lowering."""
    return sorted(p for p in BENCH.glob("*/tile/*.tile") if p.parent.name == "tile")


def fixed_variants():
    """The `tile/fixed/` files: corrected variants, which MUST conform."""
    return sorted(BENCH.glob("*/tile/fixed/*.tile"))


def tile_files():
    """Every corpus `.tile`, captures and fixed variants alike."""
    return sorted(BENCH.glob("*/tile/**/*.tile"))


def partition(decls=None):
    """`(answered, unanswered, defects)` over the whole corpus.

    `answered`   non-conforming captures that have a conforming fixed sibling.
    `unanswered` non-conforming captures with no sibling, or none that conforms.
    `defects`    findings in a `tile/fixed/` file, which exists to be correct.

    Split out of `main` so the specification is callable from a test rather than
    re-derived there, and so neither copy can drift from the other.
    """
    decls = declarations() if decls is None else decls
    defects, answered, unanswered = [], [], []

    for path in fixed_variants():
        defects.extend(check_file(path, decls))

    for path in captures():
        findings = check_file(path, decls)
        if not findings:
            continue
        sibling = path.parent / "fixed" / path.name
        if not sibling.is_file():
            unanswered.append((path, findings, "no tile/fixed/ sibling"))
            continue
        sib = check_file(sibling, decls)
        if sib:
            unanswered.append((path, findings, "its fixed sibling does not conform either"))
        else:
            answered.append((path, sibling))

    return answered, unanswered, defects


def _rel(path):
    return path.relative_to(ROOT) if ROOT in path.parents else path


def main() -> int:
    decls = declarations()
    if len(decls) < 100:
        print(f"Only {len(decls)} declarations parsed from the DSL; the reader broke")
        return 1

    caps, fixed = captures(), fixed_variants()
    if not caps:
        print("No corpus captures to check")
        return 1

    print(
        f"Checking {len(caps)} capture(s) and {len(fixed)} fixed variant(s) "
        f"against {len(decls)} declarations...\n"
    )

    answered, unanswered, defects = partition(decls)

    if answered:
        print(f"{len(answered)} capture(s) are non-conforming and ANSWERED by a fixed variant:")
        for cap, sib in answered:
            print(f"  {_rel(cap)}\n      answered by {_rel(sib)}")
        print()

    if unanswered:
        print(f"{len(unanswered)} capture(s) are non-conforming and UNANSWERED:")
        for cap, findings, why in unanswered:
            print(f"  {_rel(cap)}  ({why})")
            for m in findings:
                print(f"      {m.split(': ', 1)[1]}")
            print(f"      -> add {_rel(cap).parent}/fixed/{cap.name} with the call corrected")
        print()

    if defects:
        print("Findings in a fixed variant, which are defects outright:")
        for m in defects:
            print(f"  {m}")
        print()

    bad = len(unanswered) + len(defects)
    if bad:
        print(f"Results: {bad} problem(s)")
        return 1
    print(
        f"Results: every fixed variant conforms, and each of the {len(answered)} "
        "non-conforming capture(s) has a conforming sibling"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
