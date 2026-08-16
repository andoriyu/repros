# burn-onnx bakes f32 dtypes into generated code → whole bf16 CubeCL lane dies with `IrError::DTypeMismatch`

**File against:** [tracel-ai/burn](https://github.com/tracel-ai/burn) (burn-onnx × burn-ir composability) · **Severity:** crash — an entire backend configuration is unusable

## Summary

Two components, individually defensible, compose into a dead lane:

1. **burn-onnx 0.21** turns folded ONNX `Constant` nodes into Rust literals with
   the float dtype baked in: the generated encoder for a FastConformer model
   carries **227** `DType::F32`-tagged tensor constructions inside **127**
   `Param::uninitialized` closures. These constants are not in the weights file,
   so no `burn-store` `ModuleAdapter` can retarget them — they stay f32 on every
   backend, including bf16 ones.

2. **burn-ir**'s `output_dtype` requires a binary op's operand dtypes to be
   **exactly** equal and returns `Err(IrError::DTypeMismatch)` otherwise —
   unwrapped, so a panic (`crates/burn-ir/src/.../builder.rs:212` in 0.21).

Run the generated model on a bf16 CubeCL backend and the first op that mixes a
baked f32 literal with a bf16 activation panics. The `cubecl-rocm` bf16
configuration was dead on arrival — confirmed identically on two GPUs
(RX 7900 XTX gfx1100 and Radeon 890M gfx1150; same panic, same site), so it is
the codegen constant, not the architecture.

The same strict-equality check also bites f32 lanes: cubecl's default int
element is i32 while burn-onnx emits ONNX int64 constants as `DType::I64`, and
the two meet in an `add`. Workaround there is choosing `Rocm<f32, i64>` /
`Vulkan<f32, i64>` backend types.

## Reproduce

1. Import any ONNX model with folded float constants via burn-onnx 0.21 (any
   graph whose generated `.rs` contains `DType::F32` literals — a FastConformer
   or any transformer encoder will do).
2. Instantiate the generated model on a bf16 CubeCL backend, e.g.
   `Rocm<bf16, i64>`.
3. First forward call panics with `IrError::DTypeMismatch`.

A hand-written equivalent of the same architecture (no baked dtype literals)
runs the bf16 lane fine — which localizes the defect to the generated code, not
the backend math.

## Versions

burn / burn-onnx / burn-ir 0.21.0, cubecl 0.10.0, ROCm 6.4.3. Confirmed on
gfx1100 and gfx1150.

## Suggested fix

Generate dtype-generic constant construction (follow the backend's float
element) instead of hardcoding `DType::F32`, or route folded constants through
the weights file where a `ModuleAdapter` can reach them; independently,
`burn-ir` could promote rather than panic on mixed-width float pairs.
