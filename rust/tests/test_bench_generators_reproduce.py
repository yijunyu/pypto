# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Every committed `.tile` must be what its generator produces today.

This exists because the same mistake was made TWICE in one session: a change to a
generator was committed without the files it produces, so the tree carried a
generator and artefacts that disagreed. Locally nothing looked wrong (the working
tree had the new files); on the remote it was the other way round, and
`check_call_conformance` failed on the pushed commit -- a gate that belongs to
another owner saw it first.

A corpus has three gate owners -- pytest, cargo, the bench scripts -- and a
generator has two outputs, the script and the files it writes. This test is the
one that says so, in the cheapest possible way: regenerate into a scratch tree
and diff.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "rust" / "bench"
SHAPES = BENCH / "shapes"

GENERATORS = {
    "deepseek_v41": "gen_deepseek_v41.py",
    "glm_5_3": "gen_glm_5_3.py",
    "kimi_k3": "gen_kimi_k3.py",
    # The engine-slots corpus: kernels the Rust serving engine launches, at the
    # shapes it launches them. A row here exists for an ENGINE SLOT rather than a
    # model block, which is why it is a separate corpus and not a model row.
    "engine": "gen_engine.py",
}


def regenerate(tmp: Path, corpus: str, script: str) -> Path:
    """Run one generator in an isolated copy of its own layout.

    The generators write to `<shapes>/../<corpus>/tile`, so the scratch tree
    mirrors that two-level shape rather than passing an output path -- a path
    argument would test a code path the real run never takes.
    """
    shapes = tmp / "bench" / "shapes"
    shapes.mkdir(parents=True)
    shutil.copy(SHAPES / script, shapes / script)
    shutil.copy(SHAPES / "registry.json", shapes / "registry.json")
    (tmp / "bench" / corpus / "tile").mkdir(parents=True)
    proc = subprocess.run(
        [sys.executable, str(shapes / script)],
        check=False,
        cwd=shapes,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"{script} refused to run against the committed registry:\n{proc.stderr[-2000:]}"
    )
    return tmp / "bench" / corpus / "tile"


@pytest.mark.parametrize("corpus", sorted(GENERATORS))
def test_the_committed_tiles_are_what_the_generator_produces(corpus: str, tmp_path: Path) -> None:
    produced = regenerate(tmp_path, corpus, GENERATORS[corpus])
    fresh = {p.stem: p.read_text(encoding="utf-8") for p in sorted(produced.glob("*.tile"))}
    assert fresh, f"{GENERATORS[corpus]} produced no tiles"

    committed_dir = BENCH / corpus / "tile"
    committed = {p.stem: p.read_text(encoding="utf-8") for p in sorted(committed_dir.glob("*.tile"))}

    missing = sorted(set(fresh) - set(committed))
    assert not missing, (
        f"{corpus}: {len(missing)} tile(s) the generator writes are NOT committed: "
        f"{missing[:5]}. A generator and its artefacts must land in the same commit."
    )

    differing = [stem for stem in fresh if stem in committed and fresh[stem] != committed[stem]]
    assert not differing, (
        f"{corpus}: {len(differing)} committed tile(s) differ from what the generator "
        f"produces now: {differing[:5]}. Re-run the generator and commit its output."
    )


def test_the_generator_list_covers_every_corpus_with_one() -> None:
    """A new corpus with a generator must be added here, not silently skipped."""
    on_disk = {p.name for p in SHAPES.glob("gen_*.py")}
    listed = set(GENERATORS.values())
    assert on_disk == listed, (
        f"generators on disk {sorted(on_disk)} do not match the ones this test runs "
        f"{sorted(listed)}; add the new corpus rather than letting it go unchecked"
    )


def test_no_corpus_declares_a_generator_it_does_not_have() -> None:
    """And the reverse: a corpus claiming a generator must have one."""
    manifest_names = set()
    for corpus in GENERATORS:
        manifest = BENCH / corpus / "manifest.json"
        if not manifest.is_file():
            continue
        text = json.loads(manifest.read_text(encoding="utf-8"))
        if text.get("corpus"):
            manifest_names.add(corpus)
    assert manifest_names <= set(GENERATORS), (
        f"corpora with a manifest but no generator under test: {sorted(manifest_names - set(GENERATORS))}"
    )
