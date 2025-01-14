/*
 * // Copyright (c) Radzivon Bartoshyk. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::too_many_arguments)]
#![cfg_attr(not(feature = "unsafe"), forbid(unsafe_code))]
#![deny(unreachable_pub)]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
mod avx;
mod flip;
mod float_32;
mod flop;
#[cfg(all(target_arch = "aarch64", feature = "unsafe"))]
mod neon;
mod rgba8;
mod rotate180;
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "unsafe"))]
mod sse;
mod transpose_arbitrary;
mod transpose_arbitrary_group;
mod unsigned_16;
mod unsigned_8;
mod utils;

pub use flip::{
    flip_arbitrary, flip_plane, flip_plane16, flip_plane16_with_alpha, flip_plane_f32,
    flip_plane_f32_with_alpha, flip_plane_with_alpha, flip_rgb, flip_rgb16, flip_rgb_f32,
    flip_rgba, flip_rgba16, flip_rgba_f32,
};
pub use float_32::{
    transpose_plane_f32, transpose_plane_f32_with_alpha, transpose_rgb_f32, transpose_rgba_f32,
};
pub use flop::{
    flop_arbitrary, flop_plane, flop_plane16, flop_plane16_with_alpha, flop_plane_f32,
    flop_plane_f32_with_alpha, flop_plane_with_alpha, flop_rgb, flop_rgb16, flop_rgb_f32,
    flop_rgba, flop_rgba16, flop_rgba_f32,
};
pub use rgba8::transpose_rgba8_chunked;
pub use rotate180::{
    rotate180_arbitrary, rotate180_plane, rotate180_plane16, rotate180_plane16_with_alpha,
    rotate180_plane_f32, rotate180_plane_f32_with_alpha, rotate180_plane_with_alpha, rotate180_rgb,
    rotate180_rgb16, rotate180_rgb_f32, rotate180_rgba, rotate180_rgba16, rotate180_rgba_f32,
};
pub use transpose_arbitrary::transpose_arbitrary;
pub use unsigned_16::{
    transpose_plane16, transpose_plane16_with_alpha, transpose_rgb16, transpose_rgba16,
};
pub use unsigned_8::{transpose_plane, transpose_plane_with_alpha, transpose_rgb, transpose_rgba};
pub use utils::{FlipMode, FlopMode, TransposeError};
