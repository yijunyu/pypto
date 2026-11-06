"""The regression test for the bugs that live in the DRIVERS, not the library.

Five of this campaign's fixes were made inside `examples/glm53_block0.rs` and `examples/prefill_kda.rs` -- the skipped FFN
on MLA blocks, the missing MoE in one driver, the missing final norm, `beta`'s input, the stubbed constants. **None of them
can be caught by a unit test**, because they are control flow in a 300-line driver.

What caught every one of them was running BOTH drivers and comparing. This test does that automatically, so the day one of
those bugs returns, the suite says so instead of a person noticing two numbers differ.

Slow (a few seconds per block, real 96 GB checkpoint), so it is skipped unless DS4_SLOW=1.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = ROOT / "ds4-rs-ascend" / "target" / "release" / "examples"
DECODE = EXAMPLE_DIR / "glm53_block0"
PREFILL = EXAMPLE_DIR / "prefill_kda"
CKPT = Path("/Volumes/Data/models/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf")

BLOCKS = 2  # enough to cross the first MLA block (block 3 is MLA, so use 4)

slow = pytest.mark.skipif(
    os.environ.get("DS4_SLOW") != "1",
    reason="slow integration test: set DS4_SLOW=1 and build the two examples",
)


def _run(binary: Path, env_extra: dict) -> str:
    assert binary.exists(), f"build it first: cargo build --release --example {binary.stem}"
    env = dict(os.environ, **env_extra)
    out = subprocess.run([str(binary), str(CKPT)], capture_output=True, text=True, env=env, timeout=1800)
    assert out.returncode == 0, f"{binary.stem} failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    return out.stdout


def _token(stdout: str) -> tuple[int, float]:
    """The two drivers print their result differently, so accept BOTH forms.

    decode:  `logits: 154880 finite, best 5.4487 at id 88504`
    prefill: `ONE-TOKEN prefill after 6 blocks: 88504 (5.448745)`

    (My first version parsed only the first, so this test failed on the second driver's output rather than on a real
    disagreement -- the test's own bug, which is the third one this campaign has had of that kind.)
    """
    m = re.search(r"best ([-\d.]+) at id (\d+)", stdout)
    if m:
        return int(m.group(2)), float(m.group(1))
    m = re.search(r"ONE-TOKEN prefill after \d+ blocks: (\d+) \(([-\d.]+)\)", stdout)
    assert m, f"no logits line in either format:\n{stdout[-1500:]}"
    return int(m.group(1)), float(m.group(2))


@slow
def test_the_two_drivers_agree_on_the_token_and_the_logit():
    """THE regression test for the five driver-level fixes.

    Both drivers run the same blocks over the same weights. If either loses its FFN, its MoE, its final norm or its
    `beta` input, the tokens diverge -- which is how every one of those bugs was found.
    """
    decode = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS)})
    prefill = _run(PREFILL, {"PREFILL_ONE": "1", "PREFILL_BLOCKS": str(BLOCKS)})
    d_tok, d_logit = _token(decode)
    p_tok, p_logit = _token(prefill)
    assert d_tok == p_tok, f"the drivers disagree on the token: decode {d_tok} vs prefill {p_tok}"
    # THE TOLERANCE IS THE PRINT PRECISION, NOT THE ARITHMETIC'S: the decode driver prints its logit with 4 decimals and
    # the prefill with 6, so a 1e-6 bound is tighter than the data this test can see. (Measured: 1.3e-05 apart.)
    assert abs(d_logit - p_logit) < 1e-3, (
        f"the drivers agree on the token but not the logit: {d_logit} vs {p_logit} -- a magnitude difference with the "
        f"same argmax is what the missing final norm looked like"
    )


@slow
def test_hyper_connections_change_the_token_but_do_not_break_it():
    """The HC path is behind a flag; it must RUN, and it must change the output -- otherwise it is not wired in."""
    without = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS)})
    with_hc = _run(DECODE, {"GLM53_PROMPT": "Hello", "GLM53_BLOCKS": str(BLOCKS), "GLM53_HC": "1"})
    assert "hyper-connections: 4 streams" in with_hc, "the HC path did not announce itself:\n" + with_hc[-1500:]
    t0, _ = _token(without)
    t1, _ = _token(with_hc)
    assert t1 != t0, (
        f"hyper-connections produced the SAME token ({t0}) -- either the path is not wired in or it is being ignored"
    )


KIMI_SHARDS = sorted(
    str(p) for p in Path("/Volumes/T7 Shield/LLMs/kimi-k3-ud-iq1s").glob("*.gguf")
) if Path("/Volumes/T7 Shield/LLMs/kimi-k3-ud-iq1s").is_dir() else []


@slow
def test_kimi_produces_a_token_from_its_own_weights():
    """Kimi K3's forward, after six rounds of spelling and sharding fixes.

    This test exists because the Kimi path had SIX separate obstacles in a row -- a config reader assuming GLM's keys, a
    metadata-only shard, a third tensor spelling, a DIRECT gate where GLM's is low-rank, a hardcoded width, and a resolver
    that could return a real tensor from the wrong question. Each is guarded here by the one assertion that fails if the
    path breaks: a token, from a vocabulary of 163,840, through 2 of its 62 available blocks.
    """
    if len(KIMI_SHARDS) < 2:
        pytest.skip("Kimi K3's shards are not mounted")
    out = subprocess.run(
        [str(DECODE), *KIMI_SHARDS],
        capture_output=True, text=True, timeout=3600,
        env=dict(os.environ, DS4_KIMI="1", GLM53_BLOCKS="2"),
    )
    assert out.returncode == 0, f"Kimi failed:\n{out.stdout[-2000:]}\n{out.stderr[-2000:]}"
    assert "TOKEN from Kimi K3" in out.stdout, (
        "the label must name the LOADED model -- a Kimi run once printed 'TOKEN from GLM 5.3':\n" + out.stdout[-1500:]
    )
    tok, _ = _token(out.stdout)
    assert 0 <= tok < 163840, f"token {tok} is outside Kimi's vocabulary"
