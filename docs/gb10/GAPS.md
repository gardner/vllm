# GAPS — native NVFP4 Qwen3.6-35B-A3B on GB10

Scope: where the `nvidia/Qwen3.6-35B-A3B-NVFP4` enablement (2026-06-10) uses
less-than-fully-native code, or where coverage/validation is thinner than the
"native NVFP4 on GB10" headline suggests. This is the honest list. None of
these are *silent* fallbacks — every path is logged/recorded — but several are
"works, but not the ideal native kernel" accommodations, and the validation
breadth is narrow.

Legend: **[accommodation]** = correct but not the ideal native kernel ·
**[coverage]** = validated narrowly · **[packaging]** = release-process gap.

---

## 1. W4A16 dense layers dequantize FP4→bf16 (no fused FP4-weight kernel) [accommodation]

- **Where:** `ModelOptNvFp4W4A16LinearMethod` on GB10 (`shared_expert.{gate,up,
  down}_proj` on all 40 layers, and `lm_head`).
- **What it does:** dequantizes the packed FP4 weights to bf16 **once at load**
  and runs a plain bf16 tensor-core GEMM at runtime (`torch.matmul`).
- **Why it's not ideal:** this is the *correct* W4A16 math (bf16 activations,
  exact FP4 weight reconstruction, real bf16 tensor cores) — but it does **not**
  keep those weights in FP4 at runtime, so it loses the ~4× weight-memory
  benefit for those layers, and it does not use a fused FP4-weight × bf16-act
  MMA (which does not exist for SM121 today). `lm_head` is the big one:
  128256×2048 → ~525 MB bf16 vs ~128 MB FP4. `shared_expert` projections are
  small. **The 256 routed MoE experts (the bulk of the weights) DO stay FP4**
  via the native b12x w4a16 kernel — only the dense W4A16 layers are dequanted.
- **Recorded as:** `record_nvfp4_backend_selection("linear_w4a16",
  "GB10W4A16DequantBf16LinearKernel", is_fallback=False)` — i.e. it is treated
  as native (it is the only sensible native compute for W4A16 dense on SM121),
  **but a reviewer should know it is dequant-based, not a fused FP4 kernel.**
- **Native path forward:** a fused FP4-weight × bf16-activation dense GEMM
  (CuTe/CUTLASS) for SM121 — real kernel work.
- **Note:** off-GB10 this method still uses Marlin; only the SM12x branch
  dequants. On GB10, Marlin is not even an option — `gptq_marlin_repack` is
  absent from the `sm_121a` `_C` build.

## 2. Gated-delta-net linear attention runs on Triton/FLA, not flashinfer-native [accommodation]

- **Where:** the Qwen3-Next `linear_attn` layers.
- **What it does:** vLLM auto-selects the Triton/FLA GDN prefill kernel +
  `fused_recurrent_gated_delta_rule` decode (log: *"Using Triton/FLA GDN
  prefill kernel (requested=auto)"*). Correct output.
- **Why it's not ideal:** the support matrix lists `flashinfer_gdn_prefill:
  supported_native`, but that flashinfer-native GDN path was **not** the one
  auto-selected here. The Triton kernels also JIT-compile on first inference
  (latency spikes in the logs). This is a working, non-flashinfer-native
  linear-attention path — not a masking fallback, but not the "validated native
  GDN" either. Whether to force the flashinfer GDN path is an open perf/
  consistency follow-up.

## 3. FP8 dense GEMM falls back to an untuned cuBLAS/CUTLASS runner for unseen shapes [accommodation]

- **Where:** FP8 attention / `linear_attn` projections via
  `FlashInferFP8ScaledMMLinearKernel` (native FP8 — this part is fine).
- **The gap:** the autotuner warned *"No tuned config covers fp8_gemm
  input_shapes=… falling back to runner=CutlassFp8GemmRunner tactic=-1"* for
  shapes outside the tuning buckets — a **perf cliff**, not a correctness
  problem. The FP8 path is native but untuned for this model's exact shapes.

## 4. `modelopt_fp8_quantization` matrix entry stays `not_supported` while FP8 dense is actually native [coverage]

- The support-matrix entry for ModelOpt FP8 is kept `not_supported` because FP8
  **MoE / PerChannelPerToken / PB-WO** are still rejected — but FP8 **dense**
  W8A8 is in fact native now. The binary, family-level entry **under-claims**:
  it does not record that dense FP8 is supported. A reviewer auditing the
  matrix would see "FP8 not supported" and miss that the model's FP8 attention
  runs natively. (Splitting into dense/MoE entries is a contract follow-up.)

## 5. Multimodal: text-only only; the vision tower is dead weight [coverage]

- The model is genuinely multimodal (vision tower, image/video tokens). We
  serve it **text-only**: media inputs are rejected (`limit_mm_per_prompt`
  image/video must be 0; `multimodal_runtime` stays `not_supported`). The
  vision-tower modules still **construct** (and consume memory) but never run.
  Image/video understanding is a hard gap, not a fallback. The vision encoder's
  own attention paths (`mm_encoder_*`) remain `not_supported`.

## 6. W4A16 MoE ignores the checkpoint's stored activation scales [accommodation]

- The MoE experts ship `input_scale` tensors on disk (the checkpoint is
  W4A4-capable), but the declared deployment is `W4A16_NVFP4`, so we run bf16
  activations and **ignore those static activation scales** (the b12x kernel
  does its own thing for w4a16). This matches the checkpoint's declared, higher-
  quality W4A16 intent — but it means the stored scales are unused, and we are
  *not* exercising the W4A4 (FP4-activation) path for this model.

## 7. Validation is narrow: one model, no *reference* correctness, no comparison baseline [coverage]

- **Correctness** now has real numbers, including a reference delta that
  surfaced a genuine quality cost:
  - Qwen3.6-35B-A3B-NVFP4 (W4A16): **GSM8K 5-shot 89.5%**, 0% invalid — strong
    absolute, but no 70 GB BF16 reference to delta.
  - Llama-3.1-8B (same harness): BF16 **75.5%** vs NVFP4 W4A4 **68.0%** =
    **−7.5 pt**. So **NVFP4 is not free** — W4A4 in particular costs real
    accuracy, and the headline 89.5% (W4A16) hid this until a reference was run.
  - Still owed: larger n (200q ≈ ±3 pt), more tasks (MMLU/etc.), a Qwen3.6
    W4A16 reference delta, and the chat/think template (not just completions).
- **Throughput** now has a real number: **2,906 total tok/s / 323 output tok/s**
  (`vllm bench throughput`, random 1024→256, 256 prompts, CUDA-graph). But
  **no comparison baseline** (vs FP8/BF16 on the same Spark, vs other engines),
  no concurrency sweep, no prefill/decode split, no power/latency curve.
- **Multi-model** is now 4 native (nvidia Qwen3.6 W4A16; RedHatAI Llama-8B W4A4
  dense; RedHatAI Qwen3.6 W4A4 MoE multimodal-text-only; Nemotron-3-Nano hybrid
  Mamba+MoE W4A4) — but accuracy was measured on only 2, there's no systematic
  per-model accuracy/throughput table, and the exact per-model native backend
  wasn't re-captured with INFO logging for the evidence bundle.

## 8. Durable image is a Python overlay, not a from-source release build [packaging]

- `vllm-gb10-qwen36-native:local` is built `FROM` the published
  `vllm-gb10:…92d4d7aae27e` image with the changed **Python** files `COPY`ed
  over the existing `sm_121a` `_C` (all changes are Python, so this is
  functionally equivalent and needs no recompile). But the **canonical**
  release path builds the wheel from source and runs the `inspect-wheels`
  SM121A-only gate — that has not been done for this enablement. The image is
  also **not pushed to GHCR**, not tagged as a release, no manifest/checksum/
  evidence-bundle provenance.

## 9. b12x w4a16 kernel provenance pinned to the published flashinfer, not a fresh release cut [packaging]

- The native MoE runs on the published image's flashinfer (`0.6.12+cu130gb10`,
  commit `1c80efb3`), which already contains the w4a16 path. The new accuracy
  test (`test_qwen3_moe_shape_w4a16_accuracy`) lives on the flashinfer fork head
  (`gb10-release-discipline`). A canonical release would cut a flashinfer build
  that pins the validated w4a16 + Qwen3.6 coverage and AOT/cubin-bakes those
  shapes (today the w4a16 kernel JIT-compiles on first use).

## 10. Nothing is published; broader project release gaps still stand [packaging]

- Branch changes are committed but **not pushed**; no release tag/artifact.
- The pre-existing, project-wide "glossed over" items from `[[gb10-progress-
  glossed-over]]` still apply and are orthogonal to this work: the official CI
  release gate has never passed end-to-end (no GPU in CI), NVFP4 **KV cache** is
  numerically broken, there is no throughput baseline for any NVFP4 model, and
  the one previously-claimed "native" model (Llama-3.1-8B-NVFP4) still has no
  saved smoke evidence.

---

## Things that ARE genuinely native (so the list above is in context)

- **MoE (256 routed experts + shared-expert routing):** native FlashInfer b12x
  `quant_mode="w4a16"` NVFP4 kernel, FP4 weights, bf16 activations, on real
  `sm_121a` tensor cores. Numerically validated at the model shape (10/10).
- **FP8 attention / linear_attn projections:** native FlashInfer FP8 scaled-MM.
- **Main self-attention:** native FlashInfer FA2, FP8 E4M3 KV cache.
- **No silent fallback:** every selection is logged/recorded; Marlin/emulation
  remain hard-rejected on SM12x; the W4A16 dense dequant is the correct W4A16
  compute (not a W4A4-masking dequant of an NVFP4 checkpoint).
