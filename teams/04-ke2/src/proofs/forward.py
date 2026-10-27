"""Forward-encode helpers for KE2 week 1.

The codec itself is the one in the public kronecker-embeddings repository
(kronecker_embeddings.codec.kronecker_codec). This module adds what the proof in
proof.md needs on top of it:

  truncate        the cut rule from Lemma 2: the repository rule by default, our proposed variant with rule="proposed"
  codes           codes for a list of byte strings, using the repository codec
  read_support    reads the bytes back from a code (the argument of Lemma 1)
  expected_values closed-form high and low value of a code (Lemma 1)
  pair_cause      names why two tokens share a code (corollary of Theorem 1)

The kronecker-embeddings package is found in this order: already installed, the KRONECKER_SRC
environment variable (path to the repository's src folder), a sibling folder named kronecker-embeddings
next to the lab repository.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

try:
    import kronecker_embeddings  # noqa: F401
except ImportError:
    candidates = [os.environ.get("KRONECKER_SRC"), Path(__file__).resolve().parents[4].parent / "kronecker-embeddings" / "src"]
    for c in candidates:
        if c and (Path(c) / "kronecker_embeddings").is_dir():
            sys.path.insert(0, str(c))
            break
    else:
        raise ImportError(
            "kronecker_embeddings not found. Clone https://github.com/theschoolofai/kronecker-embeddings "
            "and either pip install it or set KRONECKER_SRC to its src folder."
        )

import torch
from kronecker_embeddings.codec import kronecker_codec
from kronecker_embeddings.tokenizer_utils import utf8_safe_truncate

EPS = 1e-6  # the repository default for the z-norm denominator
CAUSES = ("empty", "cut_to_empty", "same_raw", "bad_bytes_in", "truncation")


def repo_truncate(raw: bytes, P: int) -> bytes:
    """The repository rule (utf8_safe_truncate): the first of the prefixes of length P, P-1, P-2, P-3
    that is valid UTF-8, else b"". A token longer than P that has no such prefix (for example one
    that starts with a stray tail byte) becomes b"" and gets the all-zero code."""
    return utf8_safe_truncate(raw, P)


def proposed_truncate(raw: bytes, P: int) -> bytes:
    """Our proposed fix: cut to at most P bytes without splitting a UTF-8 letter, but keep the bytes.

    Try the cut positions P, P-1, P-2, P-3 in that order. Take the first one where the
    byte at the cut is not a continuation byte (0x80-0xBF), and keep everything before it.
    If none of the four works (a run of stray continuation bytes), cut at exactly P.
    It never asks whether the prefix is valid UTF-8, so a non-empty token is never cut to nothing.
    Equal to the repository rule on valid UTF-8.
    """
    if len(raw) <= P:
        return raw
    for c in range(P, max(P - 4, 0), -1):
        if (raw[c] & 0xC0) != 0x80:
            return raw[:c]
    return raw[:P]


RULES = {"repo": repo_truncate, "proposed": proposed_truncate}


def truncate(raw: bytes, P: int, rule: str = "repo") -> bytes:
    """The cut used by the proof. The default is the repository rule; rule="proposed" is our variant."""
    return RULES[rule](raw, P)


def codes(strings: list[bytes], P: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Codes for byte strings that are already at most P bytes. Shape (len(strings), 256 * P)."""
    buf = torch.zeros(len(strings), P, dtype=torch.uint8)
    lens = torch.zeros(len(strings), dtype=torch.long)
    for i, s in enumerate(strings):
        if len(s) > P:
            raise ValueError(f"string {i} has {len(s)} bytes, more than P={P}")
        if s:
            buf[i, : len(s)] = torch.frombuffer(bytearray(s), dtype=torch.uint8)
        lens[i] = len(s)
    return kronecker_codec(buf, lens, pos_dim=P, out_dtype=dtype)


def code_digest(code: torch.Tensor) -> bytes:
    """A hash of one code vector, to compare many codes without keeping them all."""
    return hashlib.sha1(code.float().contiguous().numpy().tobytes()).digest()


def read_support(code: torch.Tensor, P: int) -> bytes:
    """Read the byte string back from a code: the high cells are the lit cells.

    Cell i of the flattened (256, P) grid is byte i // P at position i % P.
    Raises ValueError if the lit cells are not exactly one per position 0..L-1.
    """
    z = code.float()
    hi, lo = z.max(), z.min()
    if hi == lo:  # the all-zero code of the empty string
        return b""
    lit = (z > (hi + lo) / 2).nonzero().flatten().tolist()
    by_pos: dict[int, int] = {}
    for i in lit:
        p = i % P
        if p in by_pos:
            raise ValueError(f"two lit cells at position {p}")
        by_pos[p] = i // P
    if sorted(by_pos) != list(range(len(by_pos))):
        raise ValueError("lit positions are not 0..L-1")
    return bytes(by_pos[p] for p in range(len(by_pos)))


def expected_values(L: int, P: int, eps: float = EPS) -> tuple[float, float]:
    """(high, low): the two values of the code of a string with L bytes (1 <= L <= P)."""
    D = 256 * P
    m = L**0.5 / D
    sd = (((1 - L / D) / (D - 1)) ** 0.5) + eps
    return (1 / L**0.5 - m) / sd, -m / sd


def decode_style_bytes(raw: bytes) -> bytes:
    """A model of tokenizer.decode([id]).encode(): invalid bytes become U+FFFD.

    A stand-in for tests. On BrahmicTokenizer-131K it equals the real `tokenizers` decoder for every id
    (proof.md section 4); other tokenizers may differ.
    """
    return raw.decode("utf-8", errors="replace").encode("utf-8")


def pair_cause(raw_a: bytes, raw_b: bytes, fed_a: bytes, fed_b: bytes, P: int, rule: str = "repo") -> str | None:
    """Why do two tokens share a code? None if they do not.

    raw_*  the token's real vocabulary bytes
    fed_*  the bytes the table feeds to the codec (equal to raw_* for the raw byte table)
    rule   the cut rule, "repo" or "proposed"
    """
    cut_a, cut_b = truncate(fed_a, P, rule), truncate(fed_b, P, rule)
    if cut_a != cut_b:
        return None
    if not cut_a:
        return "empty" if not fed_a and not fed_b else "cut_to_empty"
    if raw_a == raw_b:
        return "same_raw"
    if fed_a == fed_b:
        return "bad_bytes_in"
    return "truncation"
