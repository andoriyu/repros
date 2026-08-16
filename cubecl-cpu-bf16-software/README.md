# cubecl-cpu: bf16 is a software path — `avx512_bf16` hardware never used

**File against:** [tracel-ai/cubecl](https://github.com/tracel-ai/cubecl) · **Severity:** structural performance defect / benchmarking trap

## Summary

On CPUs with native bf16 vector instructions (`avx512_bf16`, present on Zen 4
and Zen 5), the cubecl-cpu backend executes bf16 workloads without using them.
Measured on a FastConformer encoder:

- cubecl-cpu **bf16 is 4–5× slower than the same backend's own f32** — the
  opposite of what the hardware delivers
- ~100–170× behind burn-tch's bf16 on the same machine (libtorch's CPU bf16
  path uses the hardware; tch-bf16 runs ~1.5–2× faster than tch-f32 there)

Results are numerically correct; the defect is performance only.

## Reproduce

1. Any compute-heavy burn model on `burn-cpu` 0.21, f32 vs bf16 element type,
   on a Zen 4/Zen 5 (or Cooper Lake+) machine.
2. Compare wall time; optionally `perf stat`/disassemble for `vdpbf16ps` /
   `vcvtne2ps2bf16` — absent.
3. Reference point: the same model via burn-tch bf16 shows the expected
   hardware speedup on the same machine.

## Versions

cubecl 0.10.0, burn / burn-cpu 0.21.0, AMD Zen 4 (7000-series) and Zen 5
hosts with `avx512_bf16` in /proc/cpuinfo.
