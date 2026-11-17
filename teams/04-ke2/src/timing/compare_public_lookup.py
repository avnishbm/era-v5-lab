"""Compare the public projected byte embedding with GPT-style table lookup.

Run one arm per process so peak memory is attributable to that arm. The three
arms share vocabulary, width, batch, architecture, input IDs and optimizer:
``table-tied`` (common GPT choice), ``table-untied`` (isolates the input-side
change), and ``public-ke`` (Team 6's embedding=ke2 implementation). The code
is the public V1 byte code at P=32, with the frozen p1536 raw table.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch

from bench_public_ke import ByteTable, FROZEN_TABLES_DIR, FROZEN_TOKENIZER_SHA256
from bench_public_ke import GPT, ModelConfig, StepTimer, stats, sync

P1536_P32_TABLE_SHA256 = "21072027b19ef6f05d57ead7a2fc9fb4c52129e491186bf38d0ba079e8d5168a"


def timed(fn, device: torch.device, warmup: int, steps: int) -> dict:
    """Measure ``fn`` with Team 6's event/wall timer.

    Args:
        fn: Zero-argument tensor operation or complete training step.
        device: Device performing the work.
        warmup: Calls made before timing.
        steps: Number of measured calls; CUDA event reads happen afterward.
    Returns:
        Millisecond distribution from Team 6's ``stats`` function.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    timer = StepTimer(device, expected_steps=steps)
    for _ in range(steps):
        timer.start()
        fn()
        timer.stop()
    return stats(timer.times_ms())


def tensor_bytes(tensors) -> int:
    """Sum storage bytes of a tensor iterable without materializing storage.

    Args:
        tensors: Parameters, buffers, or optimizer-state tensors.
    Returns:
        Logical tensor bytes. Shared storage may be counted twice unless the
        caller deduplicates tensor objects.
    """
    return sum(t.numel() * t.element_size() for t in tensors)


def mps_phase_allocations(model, opt, x: torch.Tensor, y: torch.Tensor) -> dict:
    """Sample live MPS tensor bytes at training phase boundaries, outside timing.

    Args:
        model: The already warmed up model.
        opt: Its initialized optimizer.
        x: Input token IDs for one training step.
        y: Shifted target token IDs.
    Returns:
        Phase-boundary allocations in bytes; transient within-op peaks are
        not measured.
    """
    def sample() -> int:
        torch.mps.synchronize()
        return torch.mps.current_allocated_memory()

    opt.zero_grad(set_to_none=True)
    phases = {"before_forward": sample()}
    logits, loss = model(x, y)
    if loss is None:
        raise AssertionError("memory probe returned no loss")
    phases["after_forward"] = sample()
    loss.backward()
    phases["after_backward"] = sample()
    opt.step()
    phases["after_optimizer"] = sample()
    del logits, loss
    opt.zero_grad(set_to_none=True)
    phases["after_cleanup"] = sample()
    return phases


def run(args: argparse.Namespace) -> dict:
    """Build one arm, check the full step, then measure lookup and training.

    Args:
        args: Arm, workload, device, timer and output settings from the CLI.
    Returns:
        Serializable measurements and resource accounting for this arm.
    """
    if min(args.batch_size, args.seq_len, args.lookup_tokens, args.width, args.steps, args.threads,
           args.layers, args.heads) < 1:
        raise ValueError("workload sizes, repeats and thread count must be positive")
    if args.width % args.heads:
        raise ValueError("width must be divisible by heads")
    if args.generation_context < 0 or args.generation_context > args.seq_len:
        raise ValueError("generation context must be in 0..seq_len")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")

    vocab = 131_072
    token_bytes = None
    table_hash = None
    if args.arm == "public-ke":
        table = ByteTable.load(
            FROZEN_TABLES_DIR / "raw_table_P32.npz", tokenizer_sha256=FROZEN_TOKENIZER_SHA256
        )
        if table.vocab_size != vocab or table.n_truncated or table.n_empty_rows:
            raise ValueError("frozen raw table failed the vocabulary/length gate")
        table_hash = table.sha256()
        if table_hash != P1536_P32_TABLE_SHA256:
            raise ValueError("frozen raw table SHA-256 mismatch")
        token_bytes = (
            torch.from_numpy(table.bytes.copy()),
            torch.from_numpy(table.lengths.astype(np.int64)),
        )
    cfg = ModelConfig(
        vocab_size=vocab, embedding="ke2" if args.arm == "public-ke" else "table",
        tie_embeddings=(args.arm == "table-tied"), block_size=args.seq_len,
        n_layer=args.layers, n_head=args.heads, n_embd=args.width, ffn="mlp", reversible=False,
        attn_impl="sdpa", ce_impl="torch",
    )
    model = GPT(cfg, token_bytes=token_bytes).to(device)
    lookup_ids = torch.arange(args.lookup_tokens, dtype=torch.long, device=device) % vocab
    x = torch.arange(args.batch_size * args.seq_len, dtype=torch.long, device=device)
    x = x.reshape(args.batch_size, args.seq_len) % vocab
    y = (x + 1) % vocab
    model.eval()
    with torch.no_grad():
        assert model.tok(lookup_ids).shape == (args.lookup_tokens, args.width)
        lookup_ms = timed(lambda: model.tok(lookup_ids), device, args.warmup, args.steps)
        generation_ms = None
        generated_id = None
        if args.generation_context:
            context = x[:1, :args.generation_context]

            def generate_one() -> torch.Tensor:
                logits, _ = model(context)
                return logits[:, -1, :].argmax(dim=-1)

            generated = generate_one()
            if generated.shape != (1,):
                raise AssertionError("generation output shape mismatch")
            generation_ms = timed(generate_one, device, args.warmup, args.steps)
            sync(device)
            generated_id = int(generated.item())

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def train_step() -> None:
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None:
            raise AssertionError("training step returned no loss")
        loss.backward()
        opt.step()

    train_step()  # end-to-end gate and optimizer allocation, outside timing
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step_ms = timed(train_step, device, args.warmup, args.steps)
    sync(device)
    mps_phases = mps_phase_allocations(model, opt, x, y) if device.type == "mps" else None

    # A tied GPT uses one tensor for token lookup and output logits. Count it
    # once, as ``model.parameters()`` does, and report the optimizer separately.
    params = list(model.parameters())
    buffers = list(model.buffers())
    opt_tensors = [v for state in opt.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = peak if platform.system() == "Darwin" else peak * 1024
    return {
        "arm": args.arm,
        "contract": "matched GPT step; public-ke uses Team 6's projected public V1 byte code",
        "tokenizer_sha256": FROZEN_TOKENIZER_SHA256,
        "table_sha256": table_hash,
        "vocab_size": vocab, "width": args.width, "layers": args.layers, "heads": args.heads,
        "batch_size": args.batch_size, "sequence_length": args.seq_len,
        "lookup_tokens": args.lookup_tokens, "generation_context": args.generation_context,
        "generation_next_id": generated_id,
        "device": str(device), "torch": torch.__version__, "platform": platform.platform(),
        "threads": torch.get_num_threads(), "warmup": args.warmup, "steps": args.steps,
        "parameter_count": sum(p.numel() for p in params),
        "parameter_bytes": tensor_bytes(params),
        "buffer_bytes": tensor_bytes(buffers),
        "optimizer_state_bytes": tensor_bytes(opt_tensors),
        "process_peak_rss_bytes": peak_bytes,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory() if device.type == "mps" else None,
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory() if device.type == "mps" else None,
        "mps_phase_allocated_bytes": mps_phases,
        "mps_phase_max_observed_allocated_bytes": max(mps_phases.values()) if mps_phases else None,
        "timings_ms": {"input_lookup_forward": lookup_ms, "full_training_step": step_ms,
                       "one_token_generation_no_kv_cache": generation_ms},
    }


def main() -> None:
    """Parse the matched workload and print/write one arm's result."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=("table-tied", "table-untied", "public-ke"))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=16)
    ap.add_argument("--lookup-tokens", type=int, default=4096)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--generation-context", type=int, default=0,
                    help="time one next token with full-context recomputation; 0 skips")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    start = time.perf_counter()
    result = run(args)
    result["wall_seconds"] = round(time.perf_counter() - start, 3)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"arm={args.arm} V={result['vocab_size']} width={args.width} tokens={args.lookup_tokens}")
    print(f"input_lookup_forward: median {result['timings_ms']['input_lookup_forward']['median']:.3f} ms")
    print(f"full_training_step: median {result['timings_ms']['full_training_step']['median']:.3f} ms")
    if result["timings_ms"]["one_token_generation_no_kv_cache"] is not None:
        print(f"one_token_generation_no_kv_cache: median {result['timings_ms']['one_token_generation_no_kv_cache']['median']:.3f} ms")
    print(f"parameter_bytes={result['parameter_bytes']} optimizer_state_bytes={result['optimizer_state_bytes']}")
    print(f"process_peak_rss_bytes={result['process_peak_rss_bytes']} wall_seconds={result['wall_seconds']:.3f}")


if __name__ == "__main__":
    main()
