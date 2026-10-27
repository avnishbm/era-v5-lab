"""Shared test setup for teams/04-ke2.

* Puts src/, src/proofs and src/determinism on sys.path so every test can import its module.
* Finds the pinned BrahmicTokenizer-131K offline (KE2_TOKENIZER, BRAHMIC_TOKENIZER_DIR, or the
  Hugging Face cache) and exports BRAHMIC_TOKENIZER_DIR for the proof tests.
* Provides the table fixtures used by test_tables.py. Tests that need the real tokenizer are
  skipped when it cannot be found.
"""

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
for p in (SRC, SRC / "proofs", SRC / "determinism"):
    sys.path.insert(0, str(p))

HF_ID = "theschoolofai/BrahmicTokenizer-131K"
HF_REVISION = "93df154cbc9dbf038a222c010d9b43906a8a72c3"


def _find_tokenizer_dir():
    for env in ("KE2_TOKENIZER", "BRAHMIC_TOKENIZER_DIR"):
        value = os.environ.get(env)
        if value:
            p = Path(value).expanduser()
            p = p if p.is_dir() else p.parent
            if (p / "tokenizer.json").is_file():
                return p
    try:
        from huggingface_hub import hf_hub_download

        for name in ("special_tokens_map.json", "tokenizer.json"):
            path = hf_hub_download(HF_ID, name, revision=HF_REVISION, local_files_only=True)
        return Path(path).parent
    except Exception:
        return None


TOKENIZER_DIR = _find_tokenizer_dir()
if TOKENIZER_DIR is not None:
    os.environ.setdefault("BRAHMIC_TOKENIZER_DIR", str(TOKENIZER_DIR))


@pytest.fixture(scope="session")
def tb():
    if TOKENIZER_DIR is None:
        pytest.skip("BrahmicTokenizer-131K not found; set KE2_TOKENIZER to its tokenizer.json")
    import build_tables

    return build_tables.load_tokenizer(str(TOKENIZER_DIR / "tokenizer.json"))


@pytest.fixture(scope="session")
def tables(tb):
    import build_tables

    return {
        (kind, P): builder(tb, P)
        for P in build_tables.POS_DIMS
        for kind, builder in (("decode", build_tables.build_decode_table), ("raw", build_tables.build_raw_table))
    }


@pytest.fixture(scope="session")
def dec32(tables):
    return tables["decode", 32]


@pytest.fixture(scope="session")
def raw32(tables):
    return tables["raw", 32]
