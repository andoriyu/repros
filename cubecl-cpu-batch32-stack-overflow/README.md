# cubecl-cpu: reproducible stack overflow at batch size 32 on real model shapes

**File against:** [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) · **Severity:** crash

## Summary

A burn-onnx-generated FastConformer encoder that runs correctly at batch 1 on
cubecl-cpu dies by stack overflow at batch size 32 — every run, deterministic.
`RUST_MIN_STACK` does not help, which suggests the overflow is on a thread the
env var doesn't reach or that stack growth is proportional to per-element work
(recursion where iteration belonged) rather than a fixed-size shortfall.

Batch 1 (the streaming inference case) is unaffected; the crash caps any
batched/offline use of the backend.

## Reproduce

1. Any large burn model on `burn-cpu` 0.21 (`Cpu<f32, i64>`); the observed case
   is a 17-block cache-aware conformer encoder, input `[32, 128, 25]`
   (batch × mel bins × frames) chunks.
2. Run a forward pass at batch 32.
3. Expected: same results as batch 1 × 32. Actual: SIGSEGV, stack overflow;
   `RUST_MIN_STACK=…` (tested up to large values) changes nothing.

## Versions

cubecl 0.10.0 (MLIR via tracel-llvm 20.1.4-7), burn / burn-cpu 0.21.0,
Linux x86_64.
