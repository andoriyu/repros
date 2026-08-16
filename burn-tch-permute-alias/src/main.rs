//! burn-tch 0.21: `permute`/`swap_dims`/`flip` build their result view with
//! `TchTensor::new` (src/ops/base.rs:652-668) instead of
//! `TchTensor::from_existing`, minting a fresh storage Arc for a buffer that is
//! still shared with the parent. burn-tch's in-place safety check
//! (`Storage::can_mut`) then wrongly approves in-place execution, `bool_and`
//! takes libtorch's `logical_and_` over overlapping memory, and libtorch's own
//! overlap assert aborts the process:
//!
//!   unsupported operation: some elements of the input tensor and the
//!   written-to tensor refer to a single memory location. Please clone()
//!   the tensor before performing the operation.
//!   (assert_no_partial_overlap at ATen/MemoryOverlap.cpp:97)
//!
//! Expected: `mask & mask^T` computed out-of-place (any square boolean mask
//! AND-ed with a transposed view of itself is a perfectly ordinary op — this
//! one is the attention-mask pattern burn-onnx generates for a FastConformer
//! encoder). Actual: process abort.
//!
//! Workaround: break the alias with an allocating round trip on the view,
//! e.g. `permuted.int().bool()` — identity on values.

use burn::tensor::{Bool, Distribution, Tensor};
use burn_tch::LibTorch;

type B = LibTorch<f32>;

fn main() {
    let device = Default::default();
    // [1, 72, 72] matches the original attention mask; any square mask works.
    let mask: Tensor<B, 3, Bool> =
        Tensor::<B, 3>::random([1, 72, 72], Distribution::Uniform(0.0, 1.0), &device)
            .greater_elem(0.5);

    // A view sharing the parent's buffer — but with a fresh storage Arc.
    let permuted = mask.clone().permute([0, 2, 1]);

    // Aborts here on burn-tch 0.21.
    let out = mask.bool_and(permuted);

    println!(
        "no abort — bug fixed (dims {:?}, true count {})",
        out.dims(),
        out.clone().int().sum().into_scalar()
    );
}
