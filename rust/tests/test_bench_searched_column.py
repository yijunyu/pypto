# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""A file in `searched/` must be a tiling that was FOUND.

`tile-search` writes its chosen kernel whether or not the choice differs from
the captured value — "chose %cl = 7168, which is the captured value: nothing to
change" still produces a file. Copying that into the corpus claims a rescue
that did not happen, and the claim is invisible: the file is well-formed, it is
in the right directory, and the manifest total goes up by one.

Two such files were committed to `kimi_k3/searched/` before this test existed,
both byte-identical to their input. Both were kernels refused by
`pto-to-ascendc`, and tile-search decides candidates by running the *PTO*
emitter — so a refusal from another emitter is structurally invisible to it. It
answered its own question correctly and the wrong question for those kernels.
"""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"


def corpora():
    return sorted(d for d in BENCH.iterdir() if d.is_dir() and (d / "manifest.json").is_file())


def test_there_are_corpora_to_check():
    assert corpora(), f"no corpus with a manifest under {BENCH}"


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_no_searched_file_is_a_copy_of_its_input(corpus):
    searched = BENCH / corpus / "searched"
    if not searched.is_dir():
        return
    noops = []
    for f in sorted(searched.glob("*.tile")):
        src = BENCH / corpus / "tile" / f.name
        if src.is_file() and src.read_bytes() == f.read_bytes():
            noops.append(f.name)
    assert not noops, (
        f"{len(noops)} file(s) in {corpus}/searched/ are byte-identical to their input, "
        f"so they claim a tiling that was not found: {noops}\n\n"
        "tile-search writes its choice even when the choice is the captured value. "
        "Check whether the kernel was refused by an emitter tile-search does not "
        "consult -- it decides candidates with the PTO emitter, so an AscendC-side "
        "refusal is invisible to it -- and delete the file rather than keep a copy "
        "of the input."
    )


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_the_rescue_total_matches_the_directory(corpus):
    man = json.loads((BENCH / corpus / "manifest.json").read_text(encoding="utf-8"))
    claimed = man.get("totals", {}).get("rescued_by_search")
    if claimed is None:
        return
    searched = BENCH / corpus / "searched"
    on_disk = len(list(searched.glob("*.tile"))) if searched.is_dir() else 0
    assert claimed == on_disk, (
        f"{corpus}/manifest.json claims rescued_by_search={claimed} and "
        f"{corpus}/searched/ holds {on_disk} file(s). A total nobody recomputes is "
        "how a manifest comes to describe coverage it does not have."
    )


@pytest.mark.parametrize("corpus", [d.name for d in corpora()])
def test_every_searched_file_names_a_kernel_in_the_corpus(corpus):
    searched = BENCH / corpus / "searched"
    if not searched.is_dir():
        return
    orphans = [
        f.name for f in sorted(searched.glob("*.tile")) if not (BENCH / corpus / "tile" / f.name).is_file()
    ]
    assert not orphans, (
        f"{corpus}/searched/ holds file(s) with no kernel of that name in "
        f"{corpus}/tile/: {orphans}. A searched tiling is a tiling OF something."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
