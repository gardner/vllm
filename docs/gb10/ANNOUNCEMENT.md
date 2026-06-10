# Native NVFP4 (W4A16) MoE serving on DGX Spark / GB10 — a working proof, and exactly what's missing

**TL;DR.** `nvidia/Qwen3.6-35B-A3B-NVFP4` now serves **natively** on a single
DGX Spark (GB10, `sm_121a`) in vLLM — the MoE runs on a real FP4 NVFP4
tensor-core kernel, FP8 attention is native, and there are **no silent
fallbacks**. This is a **proof-of-concept on one model, text-only**, validated
on one host. It is **not** a release: nothing is `pip install`-able yet, and the
honest list of less-than-native accommodations is in
[`GAPS.md`](./GAPS.md). Please read that before getting excited.

We're sharing it *because* the gap between "NVFP4 checkpoint loads" and "NVFP4
actually runs on the FP4 tensor cores" is where most GB10 attempts quietly fall
back to dequant-to-BF16 — and this one doesn't (for the MoE). Here's what's
real, what isn't, and the numbers.

## What was actually blocking it

`nvidia/Qwen3.6-35B-A3B-NVFP4` is **not** a plain NVFP4 model. It's a ModelOpt
**MIXED_PRECISION** checkpoint:
- **FP8** on attention + the Qwen3-Next gated-delta-net (`linear_attn`) — 130 layers
- **W4A16-NVFP4** on the MoE experts + `shared_expert` + `lm_head` — 161 layers
- a multimodal **vision tower**

The MoE is **W4A16** (FP4 weights × BF16 activations), and W4A16-NVFP4 had **no
native GB10 backend** in vLLM — it only knew Marlin, which the GB10 fork rejects
(and whose `gptq_marlin_repack` op isn't even in the `sm_121a` build). So the
model couldn't load. The fix is real kernel wiring, not a flag.

## What's genuinely native now

- **MoE (256 routed experts + shared-expert routing):** FlashInfer **b12x
  `quant_mode="w4a16"`** NVFP4 kernel — FP4 weights stay packed, BF16
  activations, real `sm_121a` tensor cores. Kernel numerically validated at the
  exact model shape (hidden 2048, intermediate 512, 256 experts, top-k 8).
- **FP8 attention / linear-attn projections:** native FlashInfer FP8 scaled-MM.
- **Main attention:** native FlashInfer FA2, FP8 E4M3 KV cache.
- **No silent fallback:** every backend selection is logged/recorded; Marlin and
  emulation stay hard-rejected on SM12x.

## Numbers (single GB10, validated CUDA-graph path)

- **Correctness (GSM8K, 5-shot, completions):** **89.5%** exact-match on 200
  questions, **0% invalid** responses (answer extraction never failed). That's
  a strong absolute score for a 35B-A3B model and good evidence the NVFP4
  quantization isn't badly degraded. *Caveat:* we do **not** yet have a BF16/FP8
  reference run on the same prompts to isolate the NVFP4 quantization delta, and
  this is plain completions (not the model's chat/think template) — so it's a
  solid floor, not the model's headline number.
- **Throughput:** **2,906 total tok/s** (323 output tok/s, 2.52 req/s) at batch
  (random input 1024 / output 256, 256 prompts, `gpu_mem_util=0.6`, CUDA-graph).
  Single-stream interactive is ~26–30 tok/s. *Caveat:* no comparison baseline
  yet (vs FP8/BF16 on the same Spark, or vs concurrency sweeps), and the FP8
  GEMM hits an untuned runner for some shapes (see GAPS.md #3).
- Generation is coherent and correct on spot checks ("Paris … Seine River",
  "17+25 = 42", multi-step reasoning).

## What this is NOT (read before you `git pull`)

The honest gaps live in [`GAPS.md`](./GAPS.md). The headline ones:

1. **W4A16 *dense* layers (`lm_head`, `shared_expert`) dequant FP4→BF16 at
   load** and run a BF16 GEMM — correct W4A16 math, but not a fused FP4 kernel,
   and they lose the FP4 weight-memory win. (The 256 MoE experts — the bulk —
   stay FP4.)
2. **Gated-delta-net** runs on Triton/FLA, not the flashinfer-native GDN path.
3. **No published artifact.** It's a fork branch + a local `docker build`
   overlay (Python over the published `sm_121a` `_C`). No wheel, no GHCR image,
   no release tag, no from-source build through the `inspect-wheels` SM121A gate.
4. **One model, text-only.** The model is multimodal — image/video are rejected.
   No multi-model coverage, no NVFP4 KV cache (FP8 KV only), no concurrency
   benchmark.
5. **The support matrix is mostly rejections** (15 native / 142 not-supported).
   Your favorite GPTQ/AWQ/other-NVFP4 model will likely hit a `not_supported`
   wall — by design (fail-fast over silent fallback), but it means "supports
   almost nothing" today.

## Try it (proof-of-concept, not a release)

- vLLM branch: `gardner/vllm@gb10-native-nvfp4` (commit `86ceeddcc`)
- FlashInfer branch: `gardner/flashinfer@gb10-release-discipline` (`7008ec93`)
- Build a local image from the vLLM fork root (Python overlay; no `_C`
  recompile): `docker build -f docs/gb10/Dockerfile.gb10-qwen36 -t
  vllm-gb10-qwen36-native:local .`
- Serve text-only: `vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4
  --limit-mm-per-prompt '{"image":0,"video":0}'`

## What a real release needs

1. A published wheel + image **built from source** through the SM121A
   `inspect-wheels` gate, with manifest/checksum/evidence provenance.
2. An **accuracy eval vs a BF16/FP8 reference** (does NVFP4 degrade quality?),
   plus a proper throughput/concurrency benchmark.
3. **≥3 models** across the W4A16/mixed-precision space.
4. The **fused FP4-weight × BF16-activation dense kernel** (or this dequant
   disclosed up front).
5. Lead with [`GAPS.md`](./GAPS.md), not the demo.

Skepticism is warranted and welcome — file it against the gaps above.
