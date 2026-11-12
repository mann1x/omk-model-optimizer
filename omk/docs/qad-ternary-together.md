# QAD and ternary together: one teacher, several students, one or more GPUs

An IQ2 student ([QAD](qad.md)) and a ternary student ([ternary](ternary.md)) of the same model need the same
expensive inputs: the layout probe, the teacher's top-K over the training stream, the eval rows and the imatrix pass
(both the model-basis and the folded statistics). `omk_qad.multi` computes those **once** and then trains each
student in its **own** `omk_qad.train` process, so each one pauses, resumes, fails and is relaunched on its own.

```bash
ENV=$(conda info --base)/envs/omk-modelopt/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID        # GPU indices in the spec = nvidia-smi's indices
$ENV -m omk_qad.multi --spec SPEC.yaml all # plan -> cache -> preflight -> train -> export -> report
```

| phase | what | GPU |
|---|---|---|
| `plan` | one probe; one `layout.pt` per student (an existing one is never replaced); the Hadamard-block check | CPU |
| `cache` | teacher top-K + eval rows + imatrix (model-basis and folded), once, in `<out>/cache` | `cache.gpu` |
| `preflight` | each student **alone** for `preflight_steps` steps: measures its peak memory (`PEAK_MEM`) | the student's `gpu` |
| `train` | every unfinished student at once, one process each, all resuming from their own `last.pt` | each student's `gpu` |
| `export` | one GGUF per finished student, at its own layout and ternarizer | each student's `gpu` |
| `report` | one table (klk, jaccard, top-1, ppl per eval family) over the same eval rows: `report.md` / `report.json` | — |

`--students NAME[,NAME]` acts on some students only (e.g. relaunch one after a failure). Signals sent to the `multi`
driver are forwarded to every child (SIGTERM / SIGINT / SIGUSR1 = checkpoint and exit at the step boundary, SIGUSR2
= pause toggle). Children run in their own sessions: a driver killed hard leaves them training, and their run locks
refuse a second copy.

Layout of `<out>`: `cache/`, `<student>/layout.pt`, `<student>/stage1/{last,final,ckpt_step*}.pt`,
`<student>/<student>.gguf`, `logs/{cache,<student>.train,<student>.export}.out`, `multi.json` (state, preflight peaks).

## Same GPU

Both students on GPU 0. `train` refuses to co-schedule them unless the preflight peaks fit in that GPU's free memory
now (sum of each student's reserved peak + 1 GiB context each); otherwise it names the GPU and asks you to move a
student.

```yaml
model: /models/Qwen3.5-4B
out: /runs/qwen35_4b
llama_cpp: /opt/llama.cpp
layout: {rest: IQ2_M}                     # defaults for every student's plan
cache: {gpu: 0, data: "mix/a.jsonl=0.6,mix/b.parquet=0.4", eval_data: "mix/eval_code.jsonl,mix/eval_chat.jsonl",
        n_seq: 19500, n_eval: 16, seq: 1024, topk: 64}
preflight_steps: 5
students:
  - name: iq2
    gpu: 0
    layout: {fixed: {dense_mlp: iq2_xxs}}
    train: {lr: 1.0e-5, loss: jaccard, steps: 2400, bs: 8, warmup: 50, eval_every: 300}
  - name: tern
    gpu: 0
    layout: {fixed: {dense_mlp: pq2_0}, override: {attention: pq2_0, linear_attn: pq2_0, lm_head: pq2_0}}
    train: {lr: 6.0e-5, loss: kl, steps: 2400, bs: 8, warmup: 50, eval_every: 300, ternary_method: "top:86"}
```

This is the real Qwen3.5-4B M6 recipe shape ([`recipe_m6.yaml`](../models/qwen3.5-4b/recipe_m6.yaml)): both students
on one RTX PRO 6000 (96 GB), peaks 24.3 GiB (iq2) and 29.7 GiB (tern), 8.6 h and 6.0 h wall time side by side.

## Different GPUs

Change only the `gpu` keys. The cache can run on either GPU (or a third one); no fit check is needed when no GPU is
shared, so `preflight` is optional (`plan`, `cache`, `train` in turn).

```yaml
cache: {gpu: 0, data: "...", eval_data: "...", n_seq: 19500, seq: 1024}
students:
  - name: iq2
    gpu: 0
    layout: {fixed: {dense_mlp: iq2_xxs}}
    train: {lr: 1.0e-5, loss: jaccard, steps: 2400, bs: 8}
  - name: tern
    gpu: 1
    layout: {fixed: {dense_mlp: pq2_0}, override: {attention: pq2_0, linear_attn: pq2_0, lm_head: pq2_0}}
    train: {lr: 6.0e-5, loss: kl, steps: 2400, bs: 8, ternary_method: "top:86"}
```

```bash
$ENV -m omk_qad.multi --spec SPEC.yaml plan
$ENV -m omk_qad.multi --spec SPEC.yaml cache
$ENV -m omk_qad.multi --spec SPEC.yaml train     # both start together, iq2 on GPU 0, tern on GPU 1
$ENV -m omk_qad.multi --spec SPEC.yaml export
$ENV -m omk_qad.multi --spec SPEC.yaml report
```

Mixed: three students on two GPUs (`gpu: 0`, `gpu: 0`, `gpu: 1`) work the same way; only GPU 0 needs a preflight.
`gpu: null` runs a student on the CPU (tests and tiny models only).

## Without `multi`: two processes by hand

The spec is a convenience; the same thing by hand (this is what `multi` runs):

```bash
M=/models/Qwen3.5-4B; O=/runs/qwen35_4b
$ENV -m omk_qad.layout --model $M --out $O/iq2/layout.pt  --rest IQ2_M --fixed dense_mlp=iq2_xxs
$ENV -m omk_qad.layout --model $M --out $O/tern/layout.pt --rest IQ2_M --fixed dense_mlp=pq2_0 \
    --override attention=pq2_0,linear_attn=pq2_0,lm_head=pq2_0
# ONE cache (given the first student's layout; any plan of the model maps the same names)
CUDA_VISIBLE_DEVICES=0 $ENV -m omk_qad.train --phase cache --model $M --layout $O/iq2/layout.pt --cache $O/cache \
    --data ... --eval-data ... --n-seq 19500 --seq 1024
# two trainers on the same cache: different GPUs here; the same index for one GPU
CUDA_VISIBLE_DEVICES=0 nohup $ENV -m omk_qad.train --phase train --model $M --layout $O/iq2/layout.pt \
    --cache $O/cache --out $O/iq2 --loss jaccard --lr 1e-5 --steps 2400 --bs 8 > $O/iq2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup $ENV -m omk_qad.train --phase train --model $M --layout $O/tern/layout.pt \
    --cache $O/cache --out $O/tern --loss kl --lr 6e-5 --steps 2400 --bs 8 --ternary-method top:86 > $O/tern.log 2>&1 &
```

By hand there is no fit check: on a shared GPU, run each trainer alone for a few steps first and read `PEAK_MEM` in
its log.

## Rules that hold either way

- **One Hadamard block.** The folded imatrix is computed once with one block size; every ternary student must fold
  with that block (`plan` and `cache` check it). Ternary students with different ternary class sets can need
  different blocks: train such a student from its own cache.
- **Layouts are fixed per run.** Changing a student's layout in the spec after `plan` is refused; use a new student
  name.
- **The teacher is never resident while students train**, so a student's memory is its own model + latents +
  activations. For a model whose two students cannot share one GPU, use two GPUs.
- Each student's run follows the single-student rules: resumable bit for bit, numbered snapshots, immutable finished
  stages ([qad.md](qad.md#3-train)).

BF16 recovery (`--mode bf16`) is not a `multi` student (it has no layout). A typical chain is recovery first, then
QAD/ternary students of the recovered checkpoint: [bf16-recovery.md](bf16-recovery.md) → export HF → `model:` of the
spec above.
