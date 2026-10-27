#!/usr/bin/env python3
"""
KE2 table builder — BrahmicTokenizer-131K x Kronecker codec.

Builds the two per-token byte tables that feed ``kronecker_codec``:

* **decode table** — what the upstream repo does today
  (``tokenizer_utils.token_id_to_bytes``): ``tokenizer.decode([id])`` then
  ``.encode("utf-8")``.  Any token whose bytes are not a complete UTF-8
  sequence (a BPE piece holding *part* of a multi-byte character) decodes to
  U+FFFD, so every such fragment is written as ``EF BF BD`` and shares one
  codec vector with every other fragment.  Kept here as the baseline.

* **raw byte table** — the fix.  The tokenizer is GPT-2 ``ByteLevel`` BPE:
  every vocab string is a sequence of "byte-characters" from a fixed
  256-symbol alphabet that is a *bijection* with byte values.  Inverting that
  bijection symbol-by-symbol recovers the exact bytes of every token, whether
  or not they form valid UTF-8.  Added (special) tokens are stored as the
  UTF-8 of their literal content, exactly as upstream does.

Both tables are built for P (``pos_dim``) = 32 and 64 and saved as ``.npz``
(``bytes`` uint8 ``(V, P)``, ``lengths`` int16 ``(V,)``) plus a JSON
manifest with SHA-256 fingerprints.

Usage
-----
    python build_tables.py [--tokenizer PATH_OR_HF_ID] [--out DIR] [--pos-dims 32 64]

``--tokenizer`` resolution order when omitted: ``$KE2_TOKENIZER``, then
``teams/04-ke2/data/tokenizer.json``, then the HF Hub id
``theschoolofai/BrahmicTokenizer-131K``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Locate the upstream package (works both installed and from a source checkout)
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent            # teams/04-ke2/src
TEAM_DIR = HERE.parent                            # teams/04-ke2
REPO_ROOT = TEAM_DIR.parent.parent                # repo root
_SRC = REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

HF_ID = "theschoolofai/BrahmicTokenizer-131K"
DEFAULT_LOCAL_TOKENIZER = TEAM_DIR / "data" / "tokenizer.json"
DEFAULT_OUT = TEAM_DIR / "results" / "tables"
CHAR_DIM = 256
POS_DIMS = (32, 64)
U_FFFD = "�".encode("utf-8")  # b'\xef\xbf\xbd'


# --------------------------------------------------------------------------- #
# Tokenizer loading
# --------------------------------------------------------------------------- #
def resolve_tokenizer_path(arg: Optional[str] = None) -> Path:
    """Return a local path to ``tokenizer.json`` (downloading from HF if needed)."""
    candidates = []
    if arg:
        candidates.append(arg)
    if os.environ.get("KE2_TOKENIZER"):
        candidates.append(os.environ["KE2_TOKENIZER"])
    candidates.append(str(DEFAULT_LOCAL_TOKENIZER))

    for c in candidates:
        p = Path(c).expanduser()
        if p.is_dir():
            p = p / "tokenizer.json"
        if p.is_file():
            return p

    # Fall back to the HF Hub (either the explicit arg or the canonical id).
    repo_id = arg if (arg and "/" in arg and not Path(arg).exists()) else HF_ID
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise FileNotFoundError(
            f"tokenizer.json not found locally ({candidates}) and huggingface_hub "
            f"is not installed to fetch {repo_id}"
        ) from e
    return Path(hf_hub_download(repo_id, "tokenizer.json"))


@dataclass
class TokenizerBundle:
    """Everything we need from a tokenizer.json, in one place."""

    path: Path
    tok: object                                    # tokenizers.Tokenizer
    json: dict
    vocab_size: int
    id_to_piece: Dict[int, str]                    # model vocab (ByteLevel strings)
    added: Dict[int, dict]                         # added_tokens by id
    special_ids: set = field(default_factory=set)
    _sha256: str = field(default="", repr=False)

    @property
    def sha256(self) -> str:
        """SHA-256 of the tokenizer.json file (the identity every table is bound to)."""
        if not self._sha256:
            self._sha256 = sha256_file(self.path)
        return self._sha256


def load_tokenizer(path_or_id: Optional[str] = None) -> TokenizerBundle:
    from tokenizers import Tokenizer  # local import: keeps numpy-only helpers importable

    path = resolve_tokenizer_path(path_or_id)
    tok = Tokenizer.from_file(str(path))
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    vocab = d["model"]["vocab"]
    added = {a["id"]: a for a in d.get("added_tokens", [])}
    vocab_size = max(max(vocab.values()), *(added.keys() or [0])) + 1
    hf_size = tok.get_vocab_size(with_added_tokens=True)
    if hf_size != vocab_size:
        raise ValueError(f"vocab size mismatch: json={vocab_size} tokenizers={hf_size}")
    return TokenizerBundle(
        path=path,
        tok=tok,
        json=d,
        vocab_size=vocab_size,
        id_to_piece={v: k for k, v in vocab.items()},
        added=added,
        special_ids={i for i, a in added.items() if a.get("special", False)},
    )


# --------------------------------------------------------------------------- #
# GPT-2 ByteLevel alphabet  (byte value <-> printable unicode "byte-character")
# --------------------------------------------------------------------------- #
def gpt2_bytes_to_unicode() -> Dict[int, str]:
    """The canonical GPT-2 mapping; identical to HF ``bytes_to_unicode``."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_UNI: Dict[int, str] = gpt2_bytes_to_unicode()
UNI_TO_BYTE: Dict[str, int] = {u: b for b, u in BYTE_TO_UNI.items()}
assert len(UNI_TO_BYTE) == 256, "ByteLevel alphabet must be a bijection on 256 symbols"


def piece_to_raw_bytes(piece: str) -> bytes:
    """Invert the ByteLevel bijection symbol by symbol. Never lossy."""
    try:
        return bytes(UNI_TO_BYTE[ch] for ch in piece)
    except KeyError as e:
        raise ValueError(f"piece {piece!r} contains non-ByteLevel symbol {e}") from e


# --------------------------------------------------------------------------- #
# Per-token byte sources
# --------------------------------------------------------------------------- #
class TableContractError(RuntimeError):
    """A byte table failed an integrity / compatibility check (see ByteTable.load)."""


class VocabHoleError(ValueError):
    """A token id has no vocabulary entry (or an empty byte string).

    Such ids would all be encoded as the empty string, whose codec vector is
    exactly 0 — so every hole collides with every other hole (a third source
    of collisions besides truncation and lossy decode; see proof.md, Lemma 2
    remark).  They are refused in strict mode.
    """


def raw_bytes_for_id(tb: TokenizerBundle, tid: int, *, missing_ok: bool = False) -> bytes:
    """Exact bytes of token ``tid``: inverted ByteLevel piece, or literal content for added tokens.

    Raises ``VocabHoleError`` for an id with no vocabulary entry unless
    ``missing_ok=True`` (then returns ``b""``).
    """
    if tid in tb.added:
        return tb.added[tid]["content"].encode("utf-8")
    piece = tb.id_to_piece.get(tid)
    if piece is None:
        if missing_ok:
            return b""
        raise VocabHoleError(f"token id {tid} has no vocabulary entry (hole in id space)")
    return piece_to_raw_bytes(piece)


def decode_bytes_for_id(tb: TokenizerBundle, tid: int) -> bytes:
    """Upstream behaviour: ``tokenizer.decode([id]).encode('utf-8')``.

    Mirrors ``kronecker_embeddings.tokenizer_utils.token_id_to_bytes`` for a
    GPT-2 ByteLevel tokenizer with no ``<0xNN>`` byte-fallback pieces.
    """
    if tid in tb.special_ids:
        return tb.added[tid]["content"].encode("utf-8")
    text = tb.tok.decode([tid], skip_special_tokens=False)
    return (text or "").encode("utf-8")


def utf8_safe_truncate(raw: bytes, max_bytes: int) -> bytes:
    """Cut at a UTF-8 character boundary (same contract as upstream)."""
    if len(raw) <= max_bytes:
        return raw
    cut = max_bytes
    while cut > 0 and (raw[cut] & 0xC0) == 0x80:  # don't split a continuation byte off its lead
        cut -= 1
    return raw[:cut]


# --------------------------------------------------------------------------- #
# Table construction
# --------------------------------------------------------------------------- #
TABLE_FORMAT_VERSION = 2   # bump whenever the on-disk layout or hashing rule changes


@dataclass
class ByteTable:
    """A (V, P) byte table plus the provenance needed to trust it.

    Contract
    --------
    * ``tokenizer_sha256`` — SHA-256 of the ``tokenizer.json`` the table was
      built from.  A table is only valid together with that exact tokenizer.
    * ``sha256()`` — content hash over (kind, pos_dim, bytes, lengths).  Stored
      in the ``.npz`` on save and re-verified on load, so a corrupted or
      hand-edited file is refused.
    * ``load()`` refuses to return a table unless the caller proves it knows
      which tokenizer it is for (``tokenizer=`` or ``tokenizer_sha256=``) or
      explicitly opts out with ``allow_unverified=True``.
    """

    kind: str                 # "decode" | "raw"
    pos_dim: int
    bytes: np.ndarray         # uint8 (V, P)
    lengths: np.ndarray       # int16 (V,)
    n_truncated: int
    n_fffd_rows: int          # rows containing EF BF BD
    n_empty_rows: int = 0     # rows with L == 0 (all collide on the zero vector)
    tokenizer_sha256: str = ""

    @property
    def vocab_size(self) -> int:
        return int(self.bytes.shape[0])

    def row(self, tid: int) -> bytes:
        return self.bytes[tid, : int(self.lengths[tid])].tobytes()

    def sha256(self) -> str:
        """Content hash: kind | pos_dim | bytes | lengths (independent of file layout)."""
        h = hashlib.sha256()
        h.update(f"ke2-table-v{TABLE_FORMAT_VERSION}|{self.kind}|{self.pos_dim}|".encode())
        h.update(np.ascontiguousarray(self.bytes, dtype=np.uint8).tobytes())
        h.update(np.ascontiguousarray(self.lengths, dtype=np.int16).tobytes())
        return h.hexdigest()

    # ---- contract helpers -------------------------------------------------
    def check_tokenizer(self, tokenizer_sha256: str) -> None:
        if not self.tokenizer_sha256:
            raise TableContractError("table carries no tokenizer_sha256; rebuild it with build_tables.py")
        if tokenizer_sha256 != self.tokenizer_sha256:
            raise TableContractError(
                f"table was built for tokenizer {self.tokenizer_sha256[:12]}… but caller has "
                f"{tokenizer_sha256[:12]}… — rebuild the table for this tokenizer")

    def check_integrity(self, expected_sha256: str) -> None:
        actual = self.sha256()
        if actual != expected_sha256:
            raise TableContractError(
                f"table content hash {actual[:12]}… != recorded {expected_sha256[:12]}… (file corrupted?)")

    # ---- persistence ---------------------------------------------------------
    def save(self, path: Path) -> str:
        """Write the table + contract to ``path`` (.npz); return the content hash."""
        if not self.tokenizer_sha256:
            raise TableContractError("refusing to save a table without tokenizer_sha256")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        digest = self.sha256()
        np.savez_compressed(path, bytes=self.bytes, lengths=self.lengths,
                 pos_dim=np.int64(self.pos_dim), kind=np.array(self.kind),
                 format_version=np.int64(TABLE_FORMAT_VERSION),
                 tokenizer_sha256=np.array(self.tokenizer_sha256),
                 table_sha256=np.array(digest),
                 n_truncated=np.int64(self.n_truncated), n_empty_rows=np.int64(self.n_empty_rows))
        return digest

    @classmethod
    def load(cls, path: Path, *, tokenizer: "TokenizerBundle | None" = None,
             tokenizer_sha256: Optional[str] = None, allow_unverified: bool = False) -> "ByteTable":
        """Load and verify a table.

        Exactly one of ``tokenizer`` / ``tokenizer_sha256`` should be given so the
        table can be checked against the tokenizer it will be used with.  Pass
        ``allow_unverified=True`` only for forensic inspection of a stray file.
        """
        z = np.load(path, allow_pickle=False)
        fmt = int(z["format_version"]) if "format_version" in z else 1
        if fmt != TABLE_FORMAT_VERSION:
            raise TableContractError(
                f"{path}: table format v{fmt}, expected v{TABLE_FORMAT_VERSION}; rebuild with build_tables.py")
        t = cls(kind=str(z["kind"]), pos_dim=int(z["pos_dim"]), bytes=z["bytes"], lengths=z["lengths"],
                n_truncated=int(z["n_truncated"]), n_fffd_rows=-1, n_empty_rows=int(z["n_empty_rows"]),
                tokenizer_sha256=str(z["tokenizer_sha256"]))
        t.check_integrity(str(z["table_sha256"]))
        if tokenizer is not None:
            tokenizer_sha256 = tokenizer.sha256
        if tokenizer_sha256 is not None:
            t.check_tokenizer(tokenizer_sha256)
        elif not allow_unverified:
            raise TableContractError(
                "ByteTable.load needs tokenizer= or tokenizer_sha256= to verify the table, "
                "or allow_unverified=True")
        t.n_fffd_rows = sum(U_FFFD in t.row(i) for i in range(t.vocab_size))
        return t


def _fill(rows: Iterable[bytes], vocab_size: int, pos_dim: int, kind: str,
          truncate, *, strict: bool, tokenizer_sha256: str) -> ByteTable:
    buf = np.zeros((vocab_size, pos_dim), dtype=np.uint8)
    lens = np.zeros((vocab_size,), dtype=np.int16)
    n_trunc = n_fffd = n_empty = 0
    for tid, raw in enumerate(rows):
        if len(raw) > pos_dim:
            raw = truncate(raw, pos_dim)
            n_trunc += 1
        L = len(raw)
        if L:
            buf[tid, :L] = np.frombuffer(raw, dtype=np.uint8, count=L)
        else:
            n_empty += 1
            if strict:
                raise VocabHoleError(
                    f"token id {tid} yields an empty byte string in the {kind} table; "
                    f"all empty rows collide on the zero vector")
        lens[tid] = L
        if U_FFFD in raw:
            n_fffd += 1
    return ByteTable(kind, pos_dim, buf, lens, n_trunc, n_fffd, n_empty, tokenizer_sha256)


def build_decode_table(tb: TokenizerBundle, pos_dim: int, *, strict: bool = False) -> ByteTable:
    """Baseline table (upstream semantics): decode([id]) -> utf-8 bytes, UTF-8-safe truncation.

    Not strict by default: this is the *diagnostic* baseline, so empty rows are
    counted (``n_empty_rows``) rather than refused.
    """
    rows = (decode_bytes_for_id(tb, i) for i in range(tb.vocab_size))
    return _fill(rows, tb.vocab_size, pos_dim, "decode", utf8_safe_truncate,
                 strict=strict, tokenizer_sha256=tb.sha256)


def build_raw_table(tb: TokenizerBundle, pos_dim: int, *, strict: bool = True) -> ByteTable:
    """Fixed table: exact token bytes via the inverted ByteLevel alphabet; plain byte truncation.

    Strict by default: a hole in the id space or an empty token is a hard
    error (``VocabHoleError``) because it silently creates collisions.
    """
    rows = (raw_bytes_for_id(tb, i, missing_ok=not strict) for i in range(tb.vocab_size))
    return _fill(rows, tb.vocab_size, pos_dim, "raw", lambda r, n: r[:n],
                 strict=strict, tokenizer_sha256=tb.sha256)


# --------------------------------------------------------------------------- #
# Script classes (see README "How script classes were chosen")
# --------------------------------------------------------------------------- #
BRAHMIC_BLOCKS: Dict[str, Tuple[int, int]] = {
    "Devanagari": (0x0900, 0x097F),
    "Bengali":    (0x0980, 0x09FF),
    "Gurmukhi":   (0x0A00, 0x0A7F),
    "Gujarati":   (0x0A80, 0x0AFF),
    "Oriya":      (0x0B00, 0x0B7F),
    "Tamil":      (0x0B80, 0x0BFF),
    "Telugu":     (0x0C00, 0x0C7F),
    "Kannada":    (0x0C80, 0x0CFF),
    "Malayalam":  (0x0D00, 0x0D7F),
}
# Vedic Extensions / Devanagari Extended are used by Devanagari text too.
BRAHMIC_EXTRA = {"Devanagari": [(0x1CD0, 0x1CFF), (0xA8E0, 0xA8FF)]}

SPECIAL_CLASS = "Special/Control"
PARTIAL_CLASS = "Partial-UTF8"
CODE_CLASS = "Code/Structure"
DIGIT_CLASS = "Digits"
LATIN_CLASS = "Latin"
MIXED_CLASS = "Mixed"
OTHER_CLASS = "Other"


def _char_class(ch: str) -> Optional[str]:
    """Class of one code point, or None for ignorable (space/newline/ZW*)."""
    cp = ord(ch)
    if ch in " \t\r\n\x0b\x0c" or cp in (0x200B, 0x200C, 0x200D, 0xFEFF):
        return None  # layout / joiner: ignored for classification
    for name, (lo, hi) in BRAHMIC_BLOCKS.items():
        if lo <= cp <= hi:
            return name
    for name, ranges in BRAHMIC_EXTRA.items():
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    cat = unicodedata.category(ch)
    if cat.startswith("N"):
        return DIGIT_CLASS
    if cp < 0x80:
        return CODE_CLASS if not ch.isalpha() else LATIN_CLASS
    if cat.startswith("L") or cat.startswith("M"):
        try:
            if unicodedata.name(ch).startswith("LATIN"):
                return LATIN_CLASS
        except ValueError:
            pass
        return OTHER_CLASS
    # Non-ASCII punctuation/symbols (curly quotes, U+FFFD, box drawing, ...)
    return OTHER_CLASS


def _lead_byte_hint(raw: bytes) -> str:
    """Which script a UTF-8 fragment most likely belongs to, from its lead byte(s)."""
    # 3-byte lead E0 + second byte selects the Brahmic block (U+0900..U+0DFF).
    for i, b in enumerate(raw):
        if 0xE0 <= b <= 0xEF:
            if b == 0xE0 and i + 1 < len(raw):
                b2 = raw[i + 1]
                cp_hi = ((b & 0x0F) << 12) | ((b2 & 0x3F) << 6)
                for name, (lo, hi) in BRAHMIC_BLOCKS.items():
                    if lo <= cp_hi <= hi or lo <= cp_hi + 63 <= hi:
                        return name
                return "3-byte(E0: Sinhala/Thai/Tibetan remnant)"
            if b == 0xE0:
                return "3-byte(E0: Brahmic, block unknown)"
            if b == 0xE1:
                return "3-byte(E1: Georgian/Cherokee/ext)"
            if b == 0xE2:
                return "3-byte(E2: punctuation/math symbols)"
            if b == 0xE3:
                return "3-byte(E3: Kana/CJK punct)"
            if 0xE4 <= b <= 0xE9:
                return "3-byte(CJK remnant)"
            if 0xEA <= b <= 0xED:
                return "3-byte(Hangul remnant)"
            if b == 0xEE:
                return "3-byte(EE: private use)"
            return "3-byte(EF: fullwidth/specials)"
        if 0xC2 <= b <= 0xDF:
            return "2-byte(Latin-ext/Greek/Cyrillic)"
        if 0xF0 <= b <= 0xF4:
            return "4-byte(emoji/supp)"
    return "continuation-only"


def script_class(raw: bytes, *, is_special: bool = False, detail: bool = False) -> str:
    """Map a token's exact bytes to one script class (string).

    With ``detail=True`` partial-UTF-8 tokens get a sub-tag naming the script
    block their lead bytes point at, e.g. ``Partial-UTF8[Devanagari]``.
    """
    if is_special:
        return SPECIAL_CLASS
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return f"{PARTIAL_CLASS}[{_lead_byte_hint(raw)}]" if detail else PARTIAL_CLASS
    classes = {c for ch in text if (c := _char_class(ch)) is not None}
    if not classes:
        return CODE_CLASS  # pure whitespace / newline runs: structure tokens
    if len(classes) == 1:
        return classes.pop()
    # Latin + punctuation (e.g. "'s", "(x") is still "Latin"; digits + punct is Digits.
    if classes == {LATIN_CLASS, CODE_CLASS}:
        return LATIN_CLASS
    if classes == {DIGIT_CLASS, CODE_CLASS}:
        return DIGIT_CLASS
    brahmic = classes & set(BRAHMIC_BLOCKS)
    if len(brahmic) == 1 and classes - brahmic <= {CODE_CLASS, DIGIT_CLASS}:
        return brahmic.pop()  # Indic word with a trailing danda/punct/digit
    return MIXED_CLASS


def classify_vocab(tb: TokenizerBundle, detail: bool = False) -> List[str]:
    return [script_class(raw_bytes_for_id(tb, i), is_special=i in tb.special_ids, detail=detail)
            for i in range(tb.vocab_size)]


# --------------------------------------------------------------------------- #
# KE2 forward: table rows -> Kronecker codec (upstream math, unchanged)
# --------------------------------------------------------------------------- #
def ke2_encode(input_ids, table: ByteTable, *, char_dim: int = CHAR_DIM, **codec_kwargs):
    """Embed ``input_ids`` (any shape) with the upstream ``kronecker_codec``.

    Returns a float32 tensor of shape ``input_ids.shape + (char_dim * table.pos_dim,)``.
    """
    import torch
    from kronecker_embeddings.codec import kronecker_codec

    ids = torch.as_tensor(input_ids, dtype=torch.long).cpu()
    flat = ids.reshape(-1)
    byte_buf = torch.from_numpy(table.bytes)[flat]        # (N, P) uint8
    len_buf = torch.from_numpy(table.lengths.astype(np.int64))[flat]
    out = kronecker_codec(byte_buf, len_buf, char_dim=char_dim, pos_dim=table.pos_dim, **codec_kwargs)
    return out.reshape(*ids.shape, char_dim * table.pos_dim)


def ke2_output_dim(pos_dim: int, char_dim: int = CHAR_DIM) -> int:
    return char_dim * pos_dim


def ke2_decode_vector(vec, pos_dim: int, char_dim: int = CHAR_DIM) -> bytes:
    """Reference inverse of the codec (Lemma 1): recover the bytes from one KE2 vector.

    After z-normalisation every coordinate takes one of two values; the larger
    one marks the support ``{b_p * P + p}``.  Decoding the support gives L, and
    for each index ``i``: position ``p = i % P``, byte value ``b_p = i // P``.
    Works for the length-normalised *and* z-normalised output (both are
    strictly increasing affine maps of the one-hot sum), and for the raw sum.
    """
    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.shape[0] != char_dim * pos_dim:
        raise ValueError(f"expected D={char_dim*pos_dim}, got {v.shape[0]}")
    hi = v.max()
    if hi <= v.min():          # all-equal vector <=> empty byte string (L = 0)
        return b""
    support = np.flatnonzero(np.isclose(v, hi, rtol=0, atol=1e-6 * max(1.0, abs(hi))))
    pos = support % pos_dim
    vals = support // pos_dim
    if sorted(pos.tolist()) != list(range(len(pos))):
        raise ValueError("support positions are not 0..L-1 — not a valid codec vector")
    out = bytearray(len(pos))
    for p, b in zip(pos, vals):
        out[int(p)] = int(b)
    return bytes(out)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def table_path(out_dir: Path, kind: str, pos_dim: int) -> Path:
    return out_dir / f"{kind}_table_P{pos_dim}.npz"


def load_table(out_dir: Path, kind: str, pos_dim: int, tb: TokenizerBundle) -> ByteTable:
    """Load a saved table *verified against* ``tb`` (the only sanctioned way to load one)."""
    return ByteTable.load(table_path(Path(out_dir), kind, pos_dim), tokenizer=tb)


def build_all(tb: TokenizerBundle, out_dir: Path, pos_dims: Sequence[int] = POS_DIMS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": TABLE_FORMAT_VERSION,
        "tokenizer": {"file": tb.path.name, "sha256": tb.sha256, "vocab_size": tb.vocab_size,
                      "added_tokens": len(tb.added), "special_ids": len(tb.special_ids),
                      "hf_id": HF_ID},
        "char_dim": CHAR_DIM,
        "tables": {},
    }
    raw_lens = [len(raw_bytes_for_id(tb, i)) for i in range(tb.vocab_size)]   # raises on holes
    manifest["max_raw_byte_length"] = int(max(raw_lens))
    for P in pos_dims:
        for builder in (build_decode_table, build_raw_table):
            t = builder(tb, P)
            p = table_path(out_dir, t.kind, P)
            digest = t.save(p)
            uniq = len({t.row(i) for i in range(t.vocab_size)})
            manifest["tables"][p.name] = {
                "kind": t.kind, "pos_dim": P, "shape": list(t.bytes.shape),
                "ke2_output_dim": ke2_output_dim(P),
                "n_truncated": t.n_truncated, "n_empty_rows": t.n_empty_rows,
                "n_rows_containing_U+FFFD": t.n_fffd_rows,
                "n_unique_rows": uniq, "n_colliding_tokens": t.vocab_size - uniq,
                "tokenizer_sha256": t.tokenizer_sha256, "sha256": digest,
            }
            print(f"[{t.kind:6s} P={P:2d}] shape={t.bytes.shape} truncated={t.n_truncated} "
                  f"empty={t.n_empty_rows} U+FFFD rows={t.n_fffd_rows} "
                  f"unique rows={uniq}/{t.vocab_size} sha256={digest[:12]}…")
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default=None, help="path to tokenizer.json / dir, or HF id")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output directory for .npz tables")
    ap.add_argument("--pos-dims", type=int, nargs="+", default=list(POS_DIMS))
    a = ap.parse_args(argv)
    tb = load_tokenizer(a.tokenizer)
    print(f"tokenizer: {tb.path}  (V={tb.vocab_size}, sha256={tb.sha256[:12]}...)")
    build_all(tb, Path(a.out), a.pos_dims)
    return 0


if __name__ == "__main__":
    sys.exit(main())
