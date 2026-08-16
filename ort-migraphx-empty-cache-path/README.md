# onnxruntime MIGraphX EP: unset/empty `ORT_MIGRAPHX_MODEL_CACHE_PATH` crashes at first inference, masquerading as a compile failure

**File against:** [microsoft/onnxruntime](https://github.com/microsoft/onnxruntime) · **Severity:** crash + misleading diagnostics

## Summary

onnxruntime 1.27's MIGraphX EP **always** saves each compiled subgraph to
`<cache>/<arch>-<hash>-<hash>-<n>.mxr`. When the cache path is unset (or set to
the empty string) the directory is `""`, and the first inference dies inside
libmigraphx:

```
migraphx_save: Error: src/file_buffer.cpp:77: write_buffer: Failure opening file:
  ""/20f00-<hash>-<hash>-0.mxr
[E:onnxruntime:...] Non-zero status code returned while running
  MGXKernel_graph_main_graph_<id>_0 node ... Status Message: Failed to call function
```

That reads like a MIGraphX compilation failure. It is a missing output
directory. Setting `ORT_MIGRAPHX_MODEL_CACHE_PATH` to any writable directory is
necessary and sufficient — with it set, the same model ran clean on five
different AMD GPUs.

## Reproduce

1. Build/run any model through the MIGraphX EP via the `ort` crate (2.0-rc) or
   the C API with `ORT_MIGRAPHX_MODEL_CACHE_PATH` unset.
2. First inference crashes with the output above.
3. `export ORT_MIGRAPHX_MODEL_CACHE_PATH=$(mktemp -d)` — same run passes.

## Versions

onnxruntime 1.27.1, MIGraphX (ROCm 6.4.3), gfx1100/gfx1150/gfx1151/gfx1152/gfx1036.

## Suggested fix

Guard the empty path (default to a temp dir or make caching opt-in), and let the
error name the real cause (missing/unwritable cache directory) instead of
surfacing as a kernel-execution failure.
