# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Emit the DeepSeek V4.1 Flash rows the corpus was missing.

The V4.1 corpus arrived as `ptobench` fixtures -- quantised GEMMs, `mx4pk`, a
`swiglu` -- which are real kernels but are not a projection of the inventory in
`registry.json`. Three of that inventory's rows had no kernel at all:

    embedding / get_rows at D
    residual add at D
    attention core (decode shape)

This generator adds exactly those, from the shape set the registry records
(`D`, `VOCAB`, `HEAD_DIM`), and nothing else. It refuses to run if the registry
stops carrying a shape set, for the same reason the GLM and K3 generators do:
an invented middle configuration measures nothing and looks exactly like a real
one afterwards.

# What it deliberately does not add

The inventory also names five MLA projections. Two of them are reachable from
sourced values here (`q_a` needs `Q_LORA`, `output` needs `D`), but the
up-projections need an output width the registry does not record -- the same
missing `KV_B_OUT` that blocks GLM 5.3 and Kimi K3. Adding them would mean
choosing a width, so they are left out and named here rather than guessed.

The routed-expert rows are also left alone: the corpus already holds their
quantised indexed forms (`gemm_i8_v41_*`, `mx4pk_v41_*`), which is the form the
engine actually runs, and an f32 corpus cannot hold the raw GGUF block buffer
that form takes.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "registry.json"
OUT = HERE.parent / "deepseek_v41" / "tile"

T = 1  # decode: one token, which is where the latency KPI lives


def c(sym, v, ty="i32"):
    return f"    %{sym} = llvm.mlir.constant({v} : {ty}) : {ty}\n"


def mod(name, nptr, body):
    args = ", ".join(f"%arg{i}: !llvm.ptr<1>" for i in range(nptr))
    return (
        "// DeepSeek V4.1 Flash (deepseek_v41), shapes read from the model's own GGUF.\n"
        "// Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n"
        f"module {{\n  llvm.func @{name}({args}) "
        f"attributes {{hacc.entry}} {{\n{body}\n    llvm.return\n  }}\n}}\n"
    )


def kernels(s):
    """{stem: mlir} for the three rows the corpus was missing."""
    D = s["D"]
    VOCAB = s["VOCAB"]
    HD = s["HEAD_DIM"]
    out = {}

    # --- residual add at the model width ------------------------------------
    b = (
        c("r", T)
        + c("cl", D)
        + c("z", 0)
        + """    %a = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %b = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_add_f32(%z, %a, %b, %r, %cl) : (i32, i32, i32, i32, i32) -> i32
    llvm.call @__tile_store_f32(%arg2, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"residual_add_v41_{T}x{D}"] = mod("v41_residual_add", 3, b)

    # --- the embedding, VOCAB x D, indexed by a token id --------------------
    b = (
        c("z", 0)
        + c("v", VOCAB)
        + c("d", D)
        + c("n", T)
        + """    %w = llvm.call @__tile_load_f32(%arg0, %n, %d) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_embedding_f32(%z, %w, %arg1, %v, %d, %n) : """
        + "(i32, i32, !llvm.ptr<1>, i32, i32, i32) -> i32\n"
        + """    llvm.call @__tile_store_f32(%arg2, %o, %n, %d) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"embedding_v41_{VOCAB}x{D}"] = mod(f"v41_embedding_{D}", 3, b)

    # --- attention core at decode shape -------------------------------------
    b = (
        c("z", 0)
        + c("r", T)
        + c("cl", HD)
        + """    %q = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %k = llvm.call @__tile_load_f32(%arg1, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %v = llvm.call @__tile_load_f32(%arg2, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32
    %o = llvm.call @__tile_attention_decode_f32(%z, %q, %k, %v, %r, %cl) : """
        + "(i32, i32, i32, i32, i32, i32) -> i32\n"
        + """    llvm.call @__tile_store_f32(%arg3, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()"""
    )
    out[f"attention_decode_v41_{T}x{HD}"] = mod("v41_attn_decode", 4, b)

    return out


def main() -> int:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entry = reg["sets"]["v41"]
    shapes = entry.get("values")
    if not shapes or entry.get("status") != "sourced":
        print("v41 has no sourced shape set in the registry; refusing to invent any")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    written = kernels(shapes)
    for stem, text in written.items():
        (OUT / f"{stem}.tile").write_text(text, encoding="utf-8")
    print(f"wrote {len(written)} kernel(s) into {OUT}")
    for stem in written:
        print(f"  {stem}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
