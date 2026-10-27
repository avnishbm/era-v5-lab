#!/usr/bin/env python3
"""Run the stitched KE2 experiment for P = 32 and P = 64.

Steps: build the decode and raw byte tables (build_tables.py), check Lemma 1, Lemma 2 and
Theorem 1 on the real tokenizer (proofs/proof_checks.py), check CPU decoding determinism in
three fresh processes (determinism/tokenizer_decode_cpu.py) and, when MPS is available, the
MPS forward-codec experiment (determinism/codec_mps.py). Outputs go to teams/04-ke2/results/.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
RESULTS = SRC.parent / "results"
TOKENIZER_SHA256 = "01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff"


def step(name: str, cmd: list[str]) -> None:
    print(f"== {name}: {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer-dir", required=True, type=Path,
                    help="folder with tokenizer.json and special_tokens_map.json")
    ap.add_argument("--pos-dims", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--skip-mps", action="store_true")
    a = ap.parse_args(argv)
    py, tok = sys.executable, a.tokenizer_dir.resolve()

    step("tables", [py, str(SRC / "build_tables.py"), "--tokenizer", str(tok / "tokenizer.json"),
                    "--out", str(RESULTS / "tables"), "--pos-dims", *map(str, a.pos_dims)])
    step("proof checks", [py, str(SRC / "proofs" / "proof_checks.py"), str(tok),
                          str(RESULTS / "proof_checks.json")])
    step("cpu decode determinism", [py, str(SRC / "determinism" / "tokenizer_decode_cpu.py"),
                                    "--tokenizer", str(tok / "tokenizer.json"),
                                    "--expected-sha256", TOKENIZER_SHA256, "--out", str(RESULTS)])
    if not a.skip_mps:
        import torch

        if torch.backends.mps.is_available():
            import kronecker_embeddings

            codec_src = Path(kronecker_embeddings.__file__).resolve().parents[1]
            step("mps codec", [py, str(SRC / "determinism" / "codec_mps.py"), "--tables", str(RESULTS / "tables"),
                               "--codec-src", str(codec_src), "--out", str(RESULTS)])
        else:
            print("== mps codec: skipped (MPS not available)")
    print("ALL STEPS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
