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
use crate::common::common_process;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_transpose_16x16, neon_transpose_16x16_intl_2, neon_transpose_16x16_intl_3,
    neon_transpose_16x16_intl_4, neon_transpose_8x8, neon_transpose_8x8_intl_2,
    neon_transpose_8x8_intl_3, neon_transpose_8x8_intl_4,
};
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use crate::sse::{
    sse_transpose_8x8, sse_transpose_8x8_intl2, sse_transpose_8x8_intl3, sse_transpose_8x8_intl4,
};
use crate::utils::FlopMode;
use crate::{FlipMode, TransposeError};

pub fn transpose_plane(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, false, 1>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, false, 1>(matrix, target, width, height),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, true, 1>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, true, 1>(matrix, target, width, height),
        },
    }
}

pub fn transpose_plane_with_alpha(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, false, 2>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, false, 2>(matrix, target, width, height),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, true, 2>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, true, 2>(matrix, target, width, height),
        },
    }
}

pub fn transpose_rgb(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, false, 3>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, false, 3>(matrix, target, width, height),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, true, 3>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, true, 3>(matrix, target, width, height),
        },
    }
}

pub fn transpose_rgba(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, false, 4>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, false, 4>(matrix, target, width, height),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_u8_impl::<false, true, 4>(matrix, target, width, height),
            FlopMode::Flop => transpose_u8_impl::<true, true, 4>(matrix, target, width, height),
        },
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn transpose_u8_impl<const FLOP: bool, const FLIP: bool, const PIXEL_STRIDE: usize>(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    assert!(PIXEL_STRIDE >= 1 && PIXEL_STRIDE <= 4);
    if matrix.len() != target.len() {
        return Err(TransposeError::MismatchDimensions);
    }

    if matrix.len() != width * height * PIXEL_STRIDE {
        return Err(TransposeError::MismatchDimensions);
    }

    let row_size = width * PIXEL_STRIDE;

    let mut y = 0usize;

    while y + 16 < height {
        let source_row = if FLIP {
            &matrix[(height - 16 - y) * row_size..((height - y) * row_size)]
        } else {
            &matrix[y * row_size..((y + 16) * row_size)]
        };
        let start_y = y * PIXEL_STRIDE;

        let dst_stride = height * PIXEL_STRIDE;

        let mut x = 0usize;

        while x + 16 < width {
            let dst = &mut target[(start_y + dst_stride * if FLOP { x } else { width - 16 - x })..];
            if PIXEL_STRIDE == 4 {
                neon_transpose_16x16_intl_4::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 3 {
                neon_transpose_16x16_intl_3::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 2 {
                neon_transpose_16x16_intl_2::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 1 {
                neon_transpose_16x16::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            }
            x += 16;
        }

        if x < width {
            common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
                matrix, row_size, target, width, height, x, y, 16,
            );
        }

        y += 16;
    }

    while y + 8 < height {
        let source_row = if FLIP {
            &matrix[(height - 8 - y) * row_size..((height - y) * row_size)]
        } else {
            &matrix[y * row_size..((y + 8) * row_size)]
        };
        let start_y = y * PIXEL_STRIDE;
        let mut x = 0usize;

        while x + 8 < width {
            let dst_stride = height * PIXEL_STRIDE;
            let dst = &mut target[(start_y + dst_stride * if FLOP { x } else { width - 8 - x })..];
            if PIXEL_STRIDE == 4 {
                neon_transpose_8x8_intl_4::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 3 {
                neon_transpose_8x8_intl_3::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 2 {
                neon_transpose_8x8_intl_2::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            } else if PIXEL_STRIDE == 1 {
                neon_transpose_8x8::<FLOP, FLIP>(
                    &source_row[(x * PIXEL_STRIDE)..],
                    row_size,
                    dst,
                    dst_stride,
                );
            }
            x += 8;
        }

        if x < width {
            common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
                matrix, row_size, target, width, height, x, y, 8,
            );
        }

        y += 8;
    }

    common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
        matrix,
        row_size,
        target,
        width,
        height,
        0,
        y,
        height - y,
    );

    Ok(())
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
fn transpose_u8_impl<const FLOP: bool, const FLIP: bool, const PIXEL_STRIDE: usize>(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    assert!(PIXEL_STRIDE >= 1 && PIXEL_STRIDE <= 4);
    if matrix.len() != target.len() {
        return Err(TransposeError::MismatchDimensions);
    }

    if matrix.len() != width * height * PIXEL_STRIDE {
        return Err(TransposeError::MismatchDimensions);
    }

    let row_size = width * PIXEL_STRIDE;

    let ssse3_available = std::arch::is_x86_feature_detected!("ssse3");
    let sse4_available = std::arch::is_x86_feature_detected!("sse4.1");

    let mut y = 0usize;

    if (ssse3_available && PIXEL_STRIDE == 1) || sse4_available {
        while y + 8 < height {
            let source_row = if FLIP {
                &matrix[(height - 8 - y) * row_size..((height - y) * row_size)]
            } else {
                &matrix[y * row_size..((y + 8) * row_size)]
            };
            let start_y = y * PIXEL_STRIDE;
            let mut x = 0usize;

            while x + 8 < width {
                let dst_stride = height * PIXEL_STRIDE;
                let dst =
                    &mut target[(start_y + dst_stride * if FLOP { x } else { width - 8 - x })..];
                if PIXEL_STRIDE == 4 {
                    sse_transpose_8x8_intl4::<FLOP, FLIP>(
                        &source_row[(x * PIXEL_STRIDE)..],
                        row_size,
                        dst,
                        dst_stride,
                    );
                } else if PIXEL_STRIDE == 3 {
                    sse_transpose_8x8_intl3::<FLOP, FLIP>(
                        &source_row[(x * PIXEL_STRIDE)..],
                        row_size,
                        dst,
                        dst_stride,
                    );
                } else if PIXEL_STRIDE == 2 {
                    sse_transpose_8x8_intl2::<FLOP, FLIP>(
                        &source_row[(x * PIXEL_STRIDE)..],
                        row_size,
                        dst,
                        dst_stride,
                    );
                } else if PIXEL_STRIDE == 1 {
                    sse_transpose_8x8::<FLOP, FLIP>(
                        &source_row[(x * PIXEL_STRIDE)..],
                        row_size,
                        dst,
                        dst_stride,
                    );
                }
                x += 8;
            }

            if x < width {
                common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
                    matrix, row_size, target, width, height, x, y, 8,
                );
            }

            y += 8;
        }
    }

    common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
        matrix,
        row_size,
        target,
        width,
        height,
        0,
        y,
        height - y,
    );

    Ok(())
}

#[cfg(not(any(
    all(target_arch = "aarch64", target_feature = "neon"),
    any(target_arch = "x86_64", target_arch = "x86")
)))]
fn transpose_u8_impl<const FLOP: bool, const FLIP: bool, const PIXEL_STRIDE: usize>(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    assert!(PIXEL_STRIDE >= 1 && PIXEL_STRIDE <= 4);
    if matrix.len() != target.len() {
        return Err(TransposeError::MismatchDimensions);
    }

    if matrix.len() != width * height * PIXEL_STRIDE {
        return Err(TransposeError::MismatchDimensions);
    }

    let row_size = width * PIXEL_STRIDE;

    common_process::<u8, FLOP, FLIP, PIXEL_STRIDE>(
        matrix, row_size, target, width, height, 0, 0, height,
    );

    Ok(())
}
