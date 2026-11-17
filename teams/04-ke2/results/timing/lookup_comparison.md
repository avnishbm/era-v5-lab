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

## Apple M2 MPS follow-up

The same p1536 workload completed on this Mac's MPS device with PyTorch
2.14.0. Team 6's MPS timer synchronizes at each step boundary and reports
wall time, not GPU kernel-only time. The first six runs use the CPU settings
above; the width-128 tied and public arms were repeated with five warmups and
50 measured steps. Each arm ran in a fresh process.

| Width; B×T | Arm | Input lookup ms | Full step ms | Observed MPS phase max MiB |
| --- | --- | ---: | ---: | ---: |
| 64; 2×16 | table-tied | 0.213 | 27.750 | 144.8 |
| 64; 2×16 | table-untied | 0.252 | 35.117 | 272.8 |
| 64; 2×16 | public byte code | 1.644 | 27.259 | 169.8 |
| 128; 2×64 | table-tied | 0.279 | 74.926 | 323.7 |
| 128; 2×64 | table-untied | 0.278 | 89.693 | 579.2 |
| 128; 2×64 | public byte code | 2.308 | 76.239 | 356.2 |
| 128; 2×64, 50 steps | table-tied | 0.244 | 75.072 | 323.7 |
| 128; 2×64, 50 steps | public byte code | 2.206 | 76.656 | 356.2 |

The width-128 50-step full-step median is about 2% slower than tied lookup,
while its input lookup is about nine times slower. The largest sampled
training-phase allocation is 33 MiB higher than tied lookup. This is a lower
bound on true peak: transient allocations inside operations can be missed.
The JSON also records end-of-run and driver allocation; neither is a peak
live tensor measure. Mac process RSS must not be interpreted as a comparable
GPU peak. These measurements show no public byte-code memory advantage over
tied lookup on MPS.

Reproduce by replacing `--device cpu` with `--device mps` in the command
above. For the repeat, use `--warmup 5 --steps 50`. MPS must be accessible to
the process; a sandboxed process may report it unavailable even on this Mac.
The eight width-64/128 `lookup_*_mps*.json` files contain the raw readings.

## Six-layer MPS workload

I also ran a 384-wide, six-layer, six-head model with batch 1 and context
64 on the same 131,072-token vocabulary. This is a larger shape that fits
the 8 GB Mac; it is still smaller than a production model. The run includes
a one-token next-ID probe. Neither model path has a KV cache, so that probe
recomputes the full context and all logits; it must not be described as a
cached generation benchmark. Both arms use the same input IDs and random
seed. Each was run once for five measured steps and twice for 20 steps; the
table gives the two 20-step medians as ranges to show run variation.

| Arm | Input lookup ms | Full step ms | No-cache next token ms | Observed MPS phase max MiB |
| --- | ---: | ---: | ---: | ---: |
| table-tied | 0.37–0.38 | 135.7–139.3 | 12.4–12.7 | 963.4 |
| public byte code | 7.95–8.03 | 131.5–132.1 | 13.0–13.4 | 1027.9 |

The full-step medians differ by only a few percent, so they support roughly
similar training speed at this workload. Input lookup remains over 20 times
slower.
The public byte-code arm has about 65 MiB more live MPS allocation at the
largest *sampled training phase boundary*. That value is a lower bound on
the true peak because a transient allocation inside an operation can be
missed. The script samples after forward, backward and optimizer, outside
timed steps. Its JSON also records end-of-run allocation and driver
allocation. The result still shows no memory win over tied lookup.

Exact repeat command from the lab repository root, with `ARM` set to
`table-tied` or `public-ke` and `OUT` to the matching JSON filename:

```sh
PYTHONDONTWRITEBYTECODE=1 python teams/04-ke2/src/timing/compare_public_lookup.py \
  --arm ARM --device mps --batch-size 1 --seq-len 64 --lookup-tokens 4096 \
  --width 384 --layers 6 --heads 6 --generation-context 64 \
  --warmup 5 --steps 20 --out teams/04-ke2/results/timing/OUT
```

The last repeat printed `full_training_step: median 139.336 ms` and
`one_token_generation_no_kv_cache: median 12.381 ms` for table-tied;
`132.058 ms` and `13.395 ms` for public byte code. The six
`lookup_*_mps_w384_l6_t64*.json` files hold all raw values.

Before claiming practical speed or memory parity for an open-source model,
repeat at the target model width, context length, batch size and dtype on
the target GPU, and compare validation bits per byte at equal training
compute. Report both prefill and one-token generation throughput and
accelerator peak allocation.
