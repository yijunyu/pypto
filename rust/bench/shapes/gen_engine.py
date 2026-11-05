# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernels the Rust SERVING ENGINE launches, at the shapes it launches them.

Why this corpus exists. The engine's NPU kernels come from one hand-written
`layer0.cpp` ELF (`engine_ascend::device::LAYER0_KERNELS`). The goal is to run
the three models on kernels the Rust emitter produced instead, so each engine
kernel needs an artefact with ITS shape -- and several shapes appear nowhere in
the model corpora, because a model row is named after the block it came from
rather than after the engine slot it fills.

This generator covers only the engine's own slots. A row here is not a model
row: it exists because `rms_norm_512` is launched, not because GLM-5.3 has a
512-wide normalisation.

Nothing beyond the shapes is invented: the body is the same `__tile_rms_norm_f32`
chain every model corpus uses for its normalisation.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "engine" / "tile"
MANIFEST = HERE.parent / "engine" / "manifest.json"

# The engine's rms_norm slots with no model-corpus counterpart. `eps` is the
# engine's own (1e-6, `ds4_rmsmul_*`), not a model's.
SLOTS = [
    ("rms_norm_512", 512),
    ("rms_norm_1024", 1024),
]
EPS = 1e-6


def c(sym, v, ty="i32"):
    return f"    %{sym} = llvm.mlir.constant({v} : {ty}) : {ty}\n"


def mod(name, body):
    return (
        "// Engine slot (ds4-rs-ascend), shape from LAYER0_KERNELS.\n"
        "// Form: llvm.func + hacc.entry + __tile_* -- the shape the emitters consume.\n"
        "module {\n  llvm.func @" + name + "(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>) "
        "attributes {hacc.entry} {\n" + body + "\n    llvm.return\n  }\n}\n"
    )


def kernels():
    out = {}
    for slot, n in SLOTS:
        body = (
            c("r", 1)
            + c("cl", n)
            + f'    %eps = llvm.mlir.constant({EPS} : f32) : f32\n'
            f'    %x = llvm.call @__tile_load_f32(%arg0, %r, %cl) : (!llvm.ptr<1>, i32, i32) -> i32\n'
            f'    %o = llvm.call @__tile_rms_norm_f32(%x, %x, %eps, %r, %cl) : (i32, i32, f32, i32, i32) -> i32\n'
            f'    llvm.call @__tile_store_f32(%arg1, %o, %r, %cl) : (!llvm.ptr<1>, i32, i32, i32) -> ()'
        )
        out[f"rmsnorm_engine_1x{n}"] = mod(f"engine_rmsnorm_1_{n}", body)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, mlir in kernels().items():
        p = OUT / f"{stem}.tile"
        p.write_text(mlir, encoding="utf-8")
        written.append(stem)
    print(f"wrote {len(written)} kernel(s) to {OUT.relative_to(HERE.parent.parent)}:")
    for s in written:
        print("  ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
