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
#![forbid(unsafe_code)]
use crate::float32_cbcr_invoker::prepare_f32_cbcr_executor;
use crate::float32_invoker::prepare_f32_plane_executor;
use crate::transpose_arbitrary_group::transpose_arbitrary_grouped;
use crate::{FlipMode, FlopMode, TransposeError};
use roxygen::roxygen;

/// Transposes a single-channel 32-bit float image.
///
/// This function performs matrix transposition on single-channel float image data,
/// effectively rotating the image by 90 degrees clockwise. Additional flip
/// and flop operations can be combined for other rotation angles.
/// HDR values are preserved without clamping.
///
/// # Performance
///
/// Uses SIMD-optimized processing with AVX/NEON for maximum throughput.
#[roxygen]
pub fn transpose_plane_f32(
    /// Source image data as a flat array of 32-bit float pixels.
    input: &[f32],
    /// Number of f32 elements per row in the input (width for packed data).
    input_stride: usize,
    /// Destination buffer for transposed image data.
    output: &mut [f32],
    /// Number of f32 elements per row in the output (height for packed data).
    output_stride: usize,
    /// Width of the input image in pixels.
    width: usize,
    /// Height of the input image in pixels.
    height: usize,
    /// Horizontal mirroring mode for rotation control.
    flip_mode: FlipMode,
    /// Vertical mirroring mode for rotation control.
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    let executor = prepare_f32_plane_executor(flip_mode, flop_mode);
    executor.execute(input, input_stride, output, output_stride, width, height)
}

/// Transposes a two-channel 32-bit float image (grayscale with alpha).
///
/// This function performs matrix transposition on two-channel float image data,
/// keeping the channel pairs together during the transformation.
/// HDR values are preserved without clamping.
///
/// # Performance
///
/// Optimized for paired channel processing with SIMD instructions.
#[roxygen]
pub fn transpose_plane_f32_with_alpha(
    /// Source image data as a flat array of channel pairs.
    input: &[f32],
    /// Number of f32 elements per row in the input (width * 2 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed image data.
    output: &mut [f32],
    /// Number of f32 elements per row in the output (height * 2 for packed data).
    output_stride: usize,
    /// Width of the input image in pixels.
    width: usize,
    /// Height of the input image in pixels.
    height: usize,
    /// Horizontal mirroring mode for rotation control.
    flip_mode: FlipMode,
    /// Vertical mirroring mode for rotation control.
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    let executor = prepare_f32_cbcr_executor(flip_mode, flop_mode);
    executor.execute(input, input_stride, output, output_stride, width, height)
}

/// Transposes a 32-bit float RGB image.
///
/// This function performs matrix transposition on three-channel float RGB image data,
/// keeping the color triplets together during the transformation.
/// HDR values are preserved without clamping.
///
/// # Performance
///
/// Uses specialized 3-channel SIMD kernels where available, falling back to
/// grouped arbitrary channel processing for compatibility.
#[roxygen]
pub fn transpose_rgb_f32(
    /// Source RGB image data as a flat array (R0,G0,B0,R1,G1,B1,...).
    input: &[f32],
    /// Number of f32 elements per row in the input (width * 3 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed RGB data.
    output: &mut [f32],
    /// Number of f32 elements per row in the output (height * 3 for packed data).
    output_stride: usize,
    /// Width of the input image in pixels.
    width: usize,
    /// Height of the input image in pixels.
    height: usize,
    /// Horizontal mirroring mode for rotation control.
    flip_mode: FlipMode,
    /// Vertical mirroring mode for rotation control.
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    transpose_arbitrary_grouped::<f32, 3>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        flip_mode,
        flop_mode,
    )
}

/// Transposes a 32-bit float RGBA image.
///
/// This function performs matrix transposition on four-channel float RGBA image data,
/// keeping the color quadruplets together during the transformation.
/// HDR values are preserved without clamping.
///
/// # Performance
///
/// Highly optimized with SIMD instructions for 4-channel data on all supported architectures.
/// This is typically the fastest transpose operation due to power-of-2 channel alignment.
#[roxygen]
pub fn transpose_rgba_f32(
    /// Source RGBA image data as a flat array (R0,G0,B0,A0,R1,G1,B1,A1,...).
    input: &[f32],
    /// Number of f32 elements per row in the input (width * 4 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed RGBA data.
    output: &mut [f32],
    /// Number of f32 elements per row in the output (height * 4 for packed data).
    output_stride: usize,
    /// Width of the input image in pixels.
    width: usize,
    /// Height of the input image in pixels.
    height: usize,
    /// Horizontal mirroring mode for rotation control.
    flip_mode: FlipMode,
    /// Vertical mirroring mode for rotation control.
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    transpose_arbitrary_grouped::<f32, 4>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        flip_mode,
        flop_mode,
    )
}
