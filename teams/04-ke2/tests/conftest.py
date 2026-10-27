"""Make teams/04-ke2/src/proofs importable for the proof tests (test_lemma1, test_lemma2, test_theorem1, test_proof_checks).

The proof tests use no decorators at the start of a line: in the diff such a line begins with a plus
sign, then the at sign and a dotted name, and pr-guard's email check fails the pull request on it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "proofs"))
