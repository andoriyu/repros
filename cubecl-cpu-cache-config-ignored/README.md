# cubecl-cpu: `CompilationConfig.cache` is never read — no persistent kernel cache exists for the CPU backend

**File against:** [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) · **Severity:** structural performance defect + config silently ignored

## Summary

CubeCL's config file documents a kernel-compilation cache:

```toml
[compilation]
cache = { file = "<dir>" }
```

The CPU (MLIR JIT) backend parses this config and never consults it.
Source-confirmed in cubecl 0.10.0: only the **cuda**, **hip** and **wgpu**
runtimes read `CompilationConfig.cache`; the CPU runtime does not — **no
persistent kernel cache exists for this backend at all**.

Consequence measured on a real model (FastConformer speech encoder, generated
by burn-onnx): the first encoder chunk carries the whole MLIR compile bill —
**269 s** (`warmup_first_encoder_chunk_ms = 269,247.8`) — and it is paid again
on **every process start**. Across 8 runs on two hosts, run2/run1 total-time
ratio is 0.98–1.05: a "warm" cache changes nothing because there is none.

## Reproduce

1. Set the config above for any cubecl-cpu workload (e.g. any burn model on
   `burn-cpu` 0.21).
2. Run the process twice; time the first kernel execution of each run and check
   the configured cache dir.
3. Expected: second run skips compilation, cache dir is populated.
   Actual: identical compile bill both runs, cache dir untouched.

Or by source inspection: grep the CPU runtime for uses of
`CompilationConfig.cache` — there are none.

## Footnote: the config syntax itself is a trap

`cache = "file"` (the obvious reading of the docs) panics at config load in
cubecl-runtime 0.10.0 — `CacheConfig`'s serde external tagging makes unit
variants bare strings (`"local"`, `"target"`, `"global"`) while `File` carries a
path payload and must be a table: `cache = { file = "<dir>" }`. The panic fires
before the process touches any device. Worth fixing alongside (better serde
shape or a load-time error message naming the accepted forms).

## Versions

cubecl / cubecl-runtime 0.10.0, burn-cpu 0.21.0, Linux x86_64 (Zen 4 and Zen 5
hosts).
