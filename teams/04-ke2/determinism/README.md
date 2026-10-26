# CPU decoding determinism

This checks that decoding every token ID gives identical results in three fresh
CPU processes. It uses the Rust `tokenizers` engine through its Python API; no
model, PyTorch, GPU, or random seed is needed.

Run from the lab repository root with a local tokenizer artifact:

```sh
python3 -m pip install tokenizers==0.22.2 pytest
python3 teams/04-ke2/determinism/decode_cpu.py \
  --tokenizer /path/to/tokenizer.json \
  --expected-sha256 01d968a7970b029c071b29f060abf6104b66168dedf2ff3b1a0290a545739fff
python3 -m pytest teams/04-ke2/tests/test_determinism.py -q
```

Keep the tokenizer outside this repository. The script never downloads artifacts.
Obtain `tokenizer.json` from https://huggingface.co/theschoolofai/BrahmicTokenizer-131K/tree/main
and verify its hash with the command above.
The SHA above identifies the BrahmicTokenizer-131K artifact reviewed for week 1.
When G3 supplies a new tokenizer, verify its identity and rerun with its SHA.
The metadata records the exact Python and tokenizers versions used.

## What each pass does

1. Read the tokenizer file, compute its SHA-256 and load that same file content.
2. Require a contiguous vocabulary and visit every ID in ascending order.
3. Call `decode([id], skip_special_tokens=False)` for every ID, including specials.
   Require each special to decode to its literal content, matching Swati's decode
   table policy. Revisit this policy with G10's final special-token list.
4. Encode each decoded string as UTF-8. Preserve U+FFFD, whitespace and all other
   returned characters; apply no extra normalization or cleanup.
5. Hash the framed records using SHA-256, then report the hash and environment.

The parent starts three separate Python processes and requires their hashes,
tokenizer identities, coverage and environment records to agree. A failed child
or disagreement produces a nonzero exit status without writing passing results.

## Hash format and outputs

The hash begins with ASCII `ke2-decode-cpu-v1` followed by LF. Each record is the
token ID as four big-endian bytes, the UTF-8 byte length as four big-endian bytes,
then the UTF-8 bytes. Explicit lengths distinguish different record boundaries.

- `teams/04-ke2/results/decode_cpu_sha256.txt`: three hashes, one per pass.
- `teams/04-ke2/results/decode_cpu_metadata.json`: artifact identity, versions,
  coverage, settings and matching hashes. No local file paths are recorded.

This hashes decoded token records, rather than `.npz` files, padded table arrays,
or Kronecker vectors. Its hashes need only match each other; other serialization
formats have different hashes. P=32 and P=64 are downstream table capacities and
do not enter single-token decoding, so decoding determinism is checked once.

The test covers repeatability, including repeatability of replacement characters.
It does not prove raw-byte preservation, collision freedom, codec determinism,
or determinism across library versions and different machines. Those are separate
checks. The raw-byte table fixes information loss that this decode baseline retains.

## References

- Table PR: https://github.com/theschoolofai/era-v5-lab/pull/7
- Reviewed builder commit: `aac72afd5806b289c515486034abcc0445fe3a43`.
- Proof: https://github.com/subramanyanaik/era-v5-lab/blob/ke2-forward-proof/teams/04-ke2/theorem/theorem1/proof.md

The decoder output was compared against the reviewed table builder for every
token ID of the pinned tokenizer. All 131,072 inputs matched. All three CPU runs
produced `3bbeb19318f233b76c3a5d725a7034327681837ef60d9aa2367518db1ef08e57`.
The generated metadata is the record of these runs; rerun after changing the
tokenizer, decoder or token policy.
