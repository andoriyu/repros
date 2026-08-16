# burn-tch: `permute`/`swap_dims`/`flip` lose buffer-alias tracking → libtorch aborts

**File against:** [tracel-ai/burn](https://github.com/tracel-ai/burn) · **Severity:** crash (process abort)

## Summary

burn-tch tracks whether a tensor's buffer may be mutated in place with a
per-buffer Arc (`Storage::can_mut`, `src/tensor.rs`). Ops that produce a *view*
are supposed to inherit the parent's storage handle via
`TchTensor::from_existing` so the refcount reflects the sharing. `permute`,
`swap_dims` and `flip` do not — they call `TchTensor::new`
(`src/ops/base.rs:652-668` in burn-tch 0.21.0), which mints a fresh Arc for a
buffer that is still shared with the parent.

Consequence: `can_mut()` says yes for a shared buffer, a binary op takes
libtorch's in-place path (`logical_and_`) over overlapping memory, and
libtorch's overlap assert kills the op — a panic through tch's unwrap (an
outright process abort in larger generated-model contexts):

```
unsupported operation: some elements of the input tensor and the written-to
tensor refer to a single memory location. Please clone() the tensor before
performing the operation.
(assert_no_partial_overlap at ATen/MemoryOverlap.cpp:97)
```

## Reproduce

```
LIBTORCH=<path to libtorch 2.9> cargo run
```

`src/main.rs` is 20 lines: build a `[1,72,72]` boolean mask, `permute` it,
`bool_and` it with itself. Expected: the AND, out of place. Actual (verified,
burn-tch 0.21.0 / tch 0.22.0 / libtorch 2.9.0): panic at
`logical_and_` with the overlap message above.

The pattern is not exotic — it is exactly what burn-onnx generates for a
FastConformer attention mask (`mask & mask^T`), which is how it was found:
the generated encoder aborted deterministically at the first chunk.

## Versions

burn / burn-tch 0.21.0, tch 0.22.0, libtorch 2.9 (CPU build), Linux x86_64.

## Workaround carried in production

`production-workaround.patch` — a `Bool → Int → Bool` round trip on the permuted
operand (`to_kind` allocates, so the alias is broken; identity on values). With
the patch applied, the model passes 30/30 exact-transcript parity against an
onnxruntime baseline, which is the evidence the workaround — and the rest of the
op's semantics — are value-correct.

## Suggested fix

Build the view results of `permute`/`swap_dims`/`flip` with
`TchTensor::from_existing(parent.storage, ..)` like the other view ops, so
`can_mut()` sees the sharing.
