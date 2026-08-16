# repros

Minimal reproductions for upstream bugs found while running one speech-recognition
model (NVIDIA `parakeet realtime_eou_120m`, cache-aware FastConformer + RNNT,
ONNX export) through the [burn](https://github.com/tracel-ai/burn) ML stack and
onnxruntime, validated by character-exact differential testing against an
onnxruntime baseline over ~5,500 LibriSpeech clips plus a 30-clip reviewed corpus.

Every bug below was isolated with recorded evidence; each directory carries the
repro (code where minimal code exists, otherwise the exact recipe + captured
output). Ordered by evidence strength — strongest first.

| # | Directory | Bug | File against | Repro form |
|---|---|---|---|---|
| 1 | [`burn-tch-permute-alias`](burn-tch-permute-alias/) | `permute`/`swap_dims`/`flip` lose buffer-alias tracking → libtorch aborts | tracel-ai/burn | runnable crate |
| 2 | [`radv-cubecl-fused-miscompile`](radv-cubecl-fused-miscompile/) | RADV miscompiles a fused CubeCL SPIR-V kernel — bit-identical garbage on 2 RDNA archs | Mesa (RADV), cross-ref tracel-ai/cubecl | recipe + triangulation evidence |
| 3 | [`burn-ir-bf16-dtype-mismatch`](burn-ir-bf16-dtype-mismatch/) | burn-onnx bakes 227 f32-tagged constructions → whole bf16 CubeCL lane panics | tracel-ai/burn | recipe + panic capture |
| 4 | [`burn-onnx-reversed-slice-i32`](burn-onnx-reversed-slice-i32/) | reversed slice emits i64 sentinel as untyped literal → `literal out of range for i32` | tracel-ai/burn | recipe (public Zipformer model) |
| 5 | [`ort-migraphx-empty-cache-path`](ort-migraphx-empty-cache-path/) | empty `ORT_MIGRAPHX_MODEL_CACHE_PATH` crashes masquerading as a compile failure | microsoft/onnxruntime | recipe + error capture |
| 6 | [`cubecl-cpu-cache-config-ignored`](cubecl-cpu-cache-config-ignored/) | `CompilationConfig.cache` never read by the CPU backend — no kernel cache exists | tracel-ai/cubecl | source pointers + measurements |
| 7 | [`cubecl-cpu-mlir-arith-muli`](cubecl-cpu-mlir-arith-muli/) | MLIR `arith.muli` codegen error on i64/generated graphs → silent garbage output | tracel-ai/cubecl | recipe + parity evidence |
| 8 | [`burn-onnx-reshape-drop`](burn-onnx-reshape-drop/) | `Reshape` with runtime-computed shape silently dropped from generated code | tracel-ai/burn | recipe |
| 9 | [`cubecl-cpu-batch32-stack-overflow`](cubecl-cpu-batch32-stack-overflow/) | batch size 32 on real shapes → reproducible stack overflow | tracel-ai/cubecl | recipe |
| 10 | [`cubecl-cpu-bf16-software`](cubecl-cpu-bf16-software/) | bf16 path ignores hardware `avx512_bf16` — 4–5× slower than the same backend's f32 | tracel-ai/cubecl | measurements |

## Common environment

- burn 0.21.0 (burn-tch / burn-wgpu / burn-rocm / burn-cpu / burn-store 0.21.0), cubecl 0.10.0
- tch 0.22.0 / libtorch 2.9
- Mesa 26.2 (RADV), ROCm 6.4.3 and 7.2.3, onnxruntime 1.27.1 (`ort` crate 2.0-rc)
- NixOS, Linux 6.18; CPUs with `avx512_bf16` (Zen 4/5)
- GPUs exercised: RX 7900 XTX (gfx1100), Radeon 890M (gfx1150), Radeon 760M, Radeon 840M, Raphael iGPU (gfx1036), GTX 1060

## Method

One trick, applied relentlessly: run the same model under the reference engine
(onnxruntime) and under burn; require character-exact transcript parity per clip
and per-layer numeric agreement against a recorded oracle capture. Any
divergence bisects to a layer, then to an op, then to a backend line.
