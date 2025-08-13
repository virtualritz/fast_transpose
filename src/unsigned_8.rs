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
use crate::cbcr8::transpose_cbcr8_chunked;
use crate::plane8::transpose_plane8_chunked;
use crate::rgba8::transpose_rgba8_chunked;
use crate::transpose_arbitrary_group::transpose_arbitrary_grouped;
use crate::utils::FlopMode;
use crate::{FlipMode, TransposeError};
use roxygen::roxygen;

/// Transposes a single-channel (grayscale) image.
///
/// This function performs matrix transposition on single-channel image data,
/// effectively rotating the image by 90 degrees clockwise. Additional flip
/// and flop operations can be combined for other rotation angles.
///
/// # Performance
///
/// Uses SIMD-optimized chunked processing for maximum throughput on supported architectures.
#[roxygen]
pub fn transpose_plane(
    /// Source image data as a flat array of pixels.
    input: &[u8],
    /// Number of bytes per row in the input (width for packed data).
    input_stride: usize,
    /// Destination buffer for transposed image data.
    output: &mut [u8],
    /// Number of bytes per row in the output (height for packed data).
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
    transpose_plane8_chunked(
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

/// Transposes a two-channel image (grayscale with alpha).
///
/// This function performs matrix transposition on two-channel image data,
/// keeping the channel pairs together during the transformation.
///
/// # Performance
///
/// Optimized for paired channel processing with SIMD instructions.
#[roxygen]
pub fn transpose_plane_with_alpha(
    /// Source image data as a flat array of channel pairs.
    input: &[u8],
    /// Number of bytes per row in the input (width * 2 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed image data.
    output: &mut [u8],
    /// Number of bytes per row in the output (height * 2 for packed data).
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
    transpose_cbcr8_chunked(
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

/// Transposes an RGB image.
///
/// This function performs matrix transposition on three-channel RGB image data,
/// keeping the color triplets together during the transformation.
///
/// # Performance
///
/// Uses specialized 3-channel SIMD kernels where available, falling back to
/// grouped arbitrary channel processing for compatibility.
#[roxygen]
pub fn transpose_rgb(
    /// Source RGB image data as a flat array (R0,G0,B0,R1,G1,B1,...).
    input: &[u8],
    /// Number of bytes per row in the input (width * 3 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed RGB data.
    output: &mut [u8],
    /// Number of bytes per row in the output (height * 3 for packed data).
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
    transpose_arbitrary_grouped::<u8, 3>(
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

/// Transposes an RGBA image.
///
/// This function performs matrix transposition on four-channel RGBA image data,
/// keeping the color quadruplets together during the transformation.
///
/// # Performance
///
/// Highly optimized with SIMD instructions for 4-channel data on all supported architectures.
/// This is typically the fastest transpose operation due to power-of-2 channel alignment.
#[roxygen]
pub fn transpose_rgba(
    /// Source RGBA image data as a flat array (R0,G0,B0,A0,R1,G1,B1,A1,...).
    input: &[u8],
    /// Number of bytes per row in the input (width * 4 for packed data).
    input_stride: usize,
    /// Destination buffer for transposed RGBA data.
    output: &mut [u8],
    /// Number of bytes per row in the output (height * 4 for packed data).
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
    transpose_rgba8_chunked(
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
