# Public byte-code input versus learned lookup (CPU)

This compares the public Team 6 `embedding=ke2` projected byte-code input
with `nn.Embedding` on the frozen p1536 vocabulary. This public path uses the
V1 byte code; the name of its Team 6 switch does not identify a different
codec. The same Team 6 GPT, one transformer layer, 131,072-token vocabulary,
AdamW, input IDs, and loss are used in all arms. `table-tied` is the usual
lookup baseline; `table-untied` isolates replacement of the input embedding
while retaining a dense output head. The public byte-code arm also has a
dense, untied output head.

Source tokenizer SHA-256:
`e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b`.
P=32 raw table SHA-256:
`21072027b19ef6f05d57ead7a2fc9fb4c52129e491186bf38d0ba079e8d5168a`.
Runs used CPU, float32, four PyTorch threads, two warmups, ten measured steps,
and a fresh process per arm. The lookup probe uses 4,096 token IDs; the full
step includes forward, cross entropy, backward, and AdamW. Times are medians.
The script uses Team 6's `StepTimer` and checks the full training step before
timing it.

| Width; B×T | Arm | Input lookup ms | Full step ms | Parameters MiB | AdamW state MiB | Buffers MiB | Peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64; 2×16 | table-tied | 0.027 | 22.121 | 32.2 | 64.4 | 0.0 | 542.1 |
| 64; 2×16 | table-untied | 0.041 | 28.831 | 64.2 | 128.4 | 0.0 | 682.6 |
| 64; 2×16 | public byte code | 7.052 | 19.758 | 34.2 | 68.4 | 17.0 | 592.9 |
| 128; 2×64 | table-tied | 0.055 | 56.909 | 64.8 | 129.6 | 0.0 | 715.2 |
| 128; 2×64 | table-untied | 0.055 | 72.217 | 128.8 | 257.6 | 0.0 | 1005.0 |
| 128; 2×64 | public byte code | 11.556 | 55.055 | 68.8 | 137.6 | 17.0 | 871.2 |

These results do **not** show a memory advantage over a tied lookup at these
widths. The public byte-code input is much slower on the isolated lookup
probe, while full-step CPU times are similar in this short run because other
operations dominate. Its parameter and optimizer memory is below an untied
table, but above a tied table; its peak RSS lies between the two. This is an
implementation comparison, not evidence of equal model quality or GPU speed.
The 4,096-ID probe repeats vocabulary IDs and measures a batched embedding
call, not autoregressive generation latency.

Reproduce from the lab repository root with Python/PyTorch dependencies
installed. Run each arm as a separate command; substitute `ARM` and `OUT`:

```sh
PYTHONDONTWRITEBYTECODE=1 python teams/04-ke2/src/timing/compare_public_lookup.py \
  --arm ARM --device cpu --batch-size 2 --seq-len 64 --lookup-tokens 4096 \
  --width 128 --warmup 2 --steps 10 --out teams/04-ke2/results/timing/OUT
```

For the width-64 rows, use `--seq-len 16 --width 64`. Raw measurements are in
the six `lookup_*_cpu*.json` files beside this report. The width-128 run ended:

```text
table-tied: input_lookup_forward 0.055 ms; full_training_step 56.909 ms; wall_seconds 1.204
table-untied: input_lookup_forward 0.055 ms; full_training_step 72.217 ms; wall_seconds 1.576
public-ke: input_lookup_forward 11.556 ms; full_training_step 55.055 ms; wall_seconds 1.219
```

Before claiming practical speed or memory parity for an open-source model,
repeat at the target model width, context length, batch size, dtype and GPU,
and compare validation bits per byte at equal training compute. Report both
prefill and one-token generation throughput and accelerator peak allocation.
