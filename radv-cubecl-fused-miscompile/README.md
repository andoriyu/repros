# RADV miscompiles a fused CubeCL SPIR-V kernel — bit-identical garbage on two RDNA architectures

**File against:** [Mesa](https://gitlab.freedesktop.org/mesa/mesa/-/issues) (RADV/ACO), cross-reference [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) · **Severity:** silently wrong results

## Summary

A CubeCL kernel produced by burn-fusion (tracing-JIT op fusion → single SPIR-V
kernel, executed via wgpu/Vulkan) computes garbage under RADV. The kernel is the
fused subsampling front-end of a FastConformer speech encoder — ordinary conv2d
math. Everything about the failure points at the (RADV × this fused kernel)
pair:

| Configuration | Result |
|---|---|
| RX 7900 XTX (gfx1100, RDNA3), RADV, **fused** SPIR-V | **GARBAGE** — first-layer max_abs error 1.679e3 / 1.571e3 vs oracle |
| Radeon 890M (gfx1150, RDNA3.5), RADV, **fused** SPIR-V | **GARBAGE** — **numerically identical** wrong values (1.679e3 / 1.571e3) |
| Radeon 840M (gfx1152, RDNA3.5), RADV, fused | **GARBAGE** — end-to-end: 8/30 transcripts, WER 1.0 (all deletions) |
| Radeon AI PRO R9700 (gfx1201, RDNA4), RADV, fused | **PASS** — worst max_rel 2.117e-4, full gate clean |
| GTX 1060, NVIDIA ICD, same fused SPIR-V | PASS — outputs ≤ 1.6e-6 vs oracle |
| Same fused op trace, MLIR CPU JIT (cubecl-cpu) | PASS |
| Same fused op trace, HIP RTC (cubecl-hip) on the same AMD silicon | PASS |
| Same kernel **unfused** (fusion disabled), RADV, same GPUs | PASS |

Two RDNA3-family GPUs producing bit-identical wrong values — while RDNA4 and
NVIDIA run the same workload correctly and the unfused equivalent is correct on
the failing GPUs — is the signature of a deterministic miscompile in the RDNA3
path of the shared shader compiler (ACO/NIR), or of CubeCL emitting UB-carrying
SPIR-V that only that path is sensitive to. Either way the wrongness is
deterministic and machine-independent.

## Captured evidence in this directory

- `890m-kernels.log` — **full per-kernel SPIR-V disassembly** of every kernel
  the failing run compiles (59 kernels, `CUBECL_DEBUG_LOG` capture on the 890M,
  Mesa 26.2.0). The miscompiled kernel is among the fusion-generated set
  (`ElemwiseFuse` / `ReduceKernelFused` / fused matmul entries).
- `890m-gate-full.log` — the failing run's gate output on the 890M: per-chunk
  pre-encode diffs vs oracle, max_abs up to 2.238e3 (six chunks, all FAIL,
  same signature values as the original triangulation).
- `r9700-kernels.log` / `r9700-gate-full.log` — the same gate on RDNA4
  (gfx1201): full pass. Kernel sets differ slightly (autotune picks per
  device), which is why the 890M capture is the authoritative one.

## Reproduce

The one-line reproducer in the original harness is a backend type-alias flip:

```rust
// garbage on RADV:
type B = burn_wgpu::Vulkan<f32, i64>;          // Fusion<CubeBackend<WgpuRuntime>>
// same code, same GPU, passes:
type B = burn_wgpu::CubeBackend<burn_wgpu::WgpuRuntime, f32, i64, u8>;  // unfused
```

i.e. run any conv2d-heavy `burn` model on `burn-wgpu` 0.21 with fusion on vs
off under RADV and diff the outputs against any reference (CPU backend works).
The SPIR-V of every kernel the failing run compiles is captured in
`890m-kernels.log` (via `CUBECL_DEBUG_LOG=<file>`, which logs each compiled
kernel's disassembly).

## Versions

Mesa 26.2.0 (RADV), vulkan-loader 1.4.350.0, burn / burn-wgpu 0.21.0,
cubecl 0.10.0, Linux 6.18 (NixOS). Garbage on gfx1100 (RX 7900 XTX) and
gfx1150 (Radeon 890M); clean on gfx1201 (Radeon AI PRO R9700) and NVIDIA
GTX 1060 with the same workload.

## Impact

The Vulkan lane is quarantined in the downstream production crate until this is
resolved — fused Vulkan was otherwise the fastest correct GPU configuration
class in the campaign.
