# burn-onnx: `Reshape` with a runtime-computed shape operand is silently dropped from the generated code

**File against:** [tracel-ai/burn](https://github.com/tracel-ai/burn) · **Severity:** silent miscompile — imports without error, wrong graph

## Summary

When an ONNX `Reshape` node's shape operand is computed at run time (the output
of shape-arithmetic nodes, not an initializer), burn-onnx 0.21 **silently drops
the node**: code generation succeeds, no error, no diagnostic. The only tell is
an `unused variable` warning in the generated Rust — the reshape's input is
bound and never consumed.

The resulting model compiles and runs with the reshape missing — wrong
shapes/semantics downstream.

## Reproduce

1. Take any ONNX graph where a `Reshape`'s `shape` input is produced by e.g.
   `Shape → Gather → Concat` (standard dynamic-batch export pattern; any
   non-constant-folded transformer export has these).
2. Import with burn-onnx 0.21.
3. Generation succeeds. Inspect the generated `.rs`: the `Reshape` is absent,
   with the corresponding `unused variable` warning as the only trace.

Found on a FastConformer encoder export: the un-folded graph imports "cleanly"
but produces a wrong model; after constant-folding the graph so every `Reshape`
shape is static, the same import produces a model that matches the onnxruntime
reference bit-for-bit on transcripts (30/30 clips, then ~5,500 LibriSpeech
clips). The fold is what localizes the drop to runtime-shape `Reshape` nodes.

## Suggested fix

If runtime-shape `Reshape` is unsupported, make import return an error naming
the node.

## Versions

burn / burn-onnx 0.21.0.
