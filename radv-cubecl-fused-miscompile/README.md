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
| RX 7900 XTX (gfx1100), RADV, **fused** SPIR-V | **GARBAGE** — first-layer max_abs error 1.679e3 / 1.571e3 vs oracle |
| Radeon 890M (gfx1150), RADV, **fused** SPIR-V | **GARBAGE** — **numerically identical** wrong values (1.679e3 / 1.571e3) |
| GTX 1060, NVIDIA ICD, same fused SPIR-V | PASS — outputs ≤ 1.6e-6 vs oracle |
| Same fused op trace, MLIR CPU JIT (cubecl-cpu) | PASS |
| Same fused op trace, HIP RTC (cubecl-hip) on the same AMD silicon | PASS |
| Same kernel **unfused** (fusion disabled), RADV, same GPUs | PASS |

Two different RDNA generations producing bit-identical wrong values is the
signature of a deterministic compiler-side miscompile in the shared shader
compiler (ACO/NIR) — or of CubeCL emitting UB-carrying SPIR-V that the NVIDIA
stack happens to execute as intended. Either way the wrongness is deterministic
and machine-independent.

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
The failing kernel's SPIR-V dump (via CubeCL's kernel-dump env or RenderDoc)
should accompany the Mesa filing.

## Versions

Mesa 26.2 (RADV), burn / burn-wgpu 0.21.0, cubecl 0.10.0, Linux 6.18 (NixOS).
Confirmed on gfx1100 (RX 7900 XTX) and gfx1150 (Radeon 890M); NVIDIA GTX 1060
passes with the identical SPIR-V.

## Impact

The Vulkan lane is quarantined in the downstream production crate until this is
resolved — fused Vulkan was otherwise the fastest correct GPU configuration
class in the campaign.
