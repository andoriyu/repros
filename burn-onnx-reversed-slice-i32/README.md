# burn-onnx: reversed slice emits the i64 sentinel as an untyped literal → `literal out of range for i32`

**File against:** [tracel-ai/burn](https://github.com/tracel-ai/burn) · **Severity:** generated code fails to compile (would be silently wrong if the value fit)

## Summary

A reversed full-length ONNX `Slice` (step −1, end = i64 min sentinel) makes
burn-onnx 0.21 emit `-9223372036854775807` as an **untyped** integer literal in
the generated Rust. Rust infers integer literals as `i32` by default, so the
generated file fails with:

```
error: literal out of range for i32
```

One-line fix: emit the literal with an `i64` suffix. The compile error is the
lucky outcome — a smaller sentinel that happened to fit in i32 would truncate
silently and produce a wrong slice bound instead.

## Reproduce

Found on a public model: the streaming Zipformer from sherpa-onnx
(k2-fsa). Import it with burn-onnx 0.21 (after constant-folding the graph so it
imports; `onnxsim --skip-optimization`, 2 passes, `ONNXSIM_FIXED_POINT_ITERS=200`
gets its 544 `Shape` nodes to 0) and `cargo check` the generated crate — the
reversed-slice node produces the out-of-range literal.

Any minimal ONNX graph with a `Slice` node using `starts=[-1], ends=[INT64_MIN],
steps=[-1]` should generate the same pattern.

## Versions

burn / burn-onnx 0.21.0. Model: sherpa-onnx streaming Zipformer (public);
the same node shape also appears in a FastConformer EOU graph after static
folding, masked there by other errors.
