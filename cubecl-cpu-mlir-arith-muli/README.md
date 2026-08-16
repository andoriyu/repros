# cubecl-cpu: MLIR `arith.muli` codegen error on i64 / generated graphs → silent garbage output

**File against:** [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) · **Severity:** silently wrong results

## Summary

Running a burn-onnx-generated FastConformer encoder on the cubecl-cpu (MLIR
JIT) backend with `i64` int element produces degenerate output — the model
"runs" but transcribes garbage:

- 0/5 transcript parity on a 5-clip smoke set that every other backend gets
  exactly right (tch-cpu/tch-rocm/cubecl-rocm/cubecl-vulkan/onnxruntime all
  agree character-for-character)
- outputs degenerate to repeated tokens: `a a a a`,
  `how how assetetetetet li li f f f...`
- accompanied by continuous MLIR `arith.muli` diagnostics and autotune-fallback
  churn, and a 40.9× slowdown vs tch-cpu on the same clips (3243.84 vs
  79.32 ms/chunk)

**Scoping evidence:** a hand-written burn-nn BERT at i32 on the same backend
passes its numerics gate. The failure is specific to i64 / burn-onnx-generated
graphs — the ones that carry ONNX int64 constants as `DType::I64`.

## Reproduce

1. Import any ONNX transformer-class model via burn-onnx 0.21 (int64 shape
   arithmetic is what matters — any real ONNX export has it).
2. Run on `burn-cpu` 0.21 (`Cpu<f32, i64>`).
3. Compare output against any other backend (tch-cpu is the easy reference).
4. Expected: identical results. Actual: `arith.muli` diagnostics during JIT +
   numerically garbage output.

## Versions

cubecl 0.10.0 (MLIR via tracel-llvm 20.1.4-7), burn / burn-cpu 0.21.0,
Linux x86_64 (Zen 4/Zen 5).
