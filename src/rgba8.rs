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
use crate::neon::{neon_transpose_4x4_u8x4, neon_transpose_4x4_u8x4x8};
use crate::{FlipMode, FlopMode, TransposeError};

trait TransposeBlock {
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize);
}

fn transpose_section<const CN: usize, const FLOP: bool, const FLIP: bool>(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
) {
    for y in start_y..height {
        let input_y = if FLIP { height - 1 - y } else { y };

        for x in 0..width {
            let output_x = if FLOP { x } else { width - 1 - x };

            let input_index = x * CN + input_y * input_stride;
            let output_index = y * CN + output_x * output_stride;

            #[cfg(feature = "unsafe")]
            {
                unsafe {
                    for i in 0..CN {
                        *output.get_unchecked_mut(output_index + i) =
                            *input.get_unchecked(input_index + i);
                    }
                }
            }
            #[cfg(not(feature = "unsafe"))]
            {
                for i in 0..N {
                    output[output_index + i] = input[input_index + i];
                }
            }
        }
    }
}

struct TransposeBlockNeon4x4<const FLOP: bool, const FLIP: bool> {}

impl<const FLOP: bool, const FLIP: bool> TransposeBlock for TransposeBlockNeon4x4<FLOP, FLIP> {
    #[inline(always)]
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize) {
        neon_transpose_4x4_u8x4::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

struct TransposeBlockNeon8x8<const FLOP: bool, const FLIP: bool> {}

impl<const FLOP: bool, const FLIP: bool> TransposeBlock for TransposeBlockNeon8x8<FLOP, FLIP> {
    #[inline(always)]
    fn transpose_block(&self, src: &[u8], src_stride: usize, dst: &mut [u8], dst_stride: usize) {
        neon_transpose_4x4_u8x4x8::<FLOP, FLIP>(src, src_stride, dst, dst_stride);
    }
}

#[inline(always)]
fn transpose_executor<
    const BLOCK_SIZE: usize,
    const CN: usize,
    const FLOP: bool,
    const FLIP: bool,
>(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
    start_y: usize,
    exec: impl TransposeBlock,
) -> usize {
    let mut y = start_y;

    let mut src_buffer = vec![0; BLOCK_SIZE * BLOCK_SIZE * CN];
    let mut dst_buffer = vec![0; BLOCK_SIZE * BLOCK_SIZE * CN];

    unsafe {
        while y + BLOCK_SIZE < height {
            let input_y = if FLIP { height - BLOCK_SIZE - y } else { y };

            let src = input.get_unchecked(input_stride * input_y..);

            let mut x = 0usize;

            while x + BLOCK_SIZE < width {
                let output_x = if FLOP { x } else { width - BLOCK_SIZE - x };

                let src = src.get_unchecked(x * CN..);
                let dst = output.get_unchecked_mut(y * CN + output_stride * output_x..);

                exec.transpose_block(src, input_stride, dst, output_stride);

                x += BLOCK_SIZE;
            }

            if x < width {
                let rem_x = width - x;
                assert!(rem_x < CN * BLOCK_SIZE);

                let output_x = if FLOP { x } else { 0 };
                let src = src.get_unchecked(x * CN..);

                for j in 0..BLOCK_SIZE {
                    std::ptr::copy_nonoverlapping(
                        src.get_unchecked(j * input_stride..).as_ptr(),
                        src_buffer.get_unchecked_mut(j * (BLOCK_SIZE * CN)..).as_mut_ptr(),
                        rem_x * CN,
                    );
                }

                exec.transpose_block(
                    src_buffer.as_slice(),
                    BLOCK_SIZE * CN,
                    dst_buffer.as_mut_slice(),
                    BLOCK_SIZE * CN,
                );

                let dst = output.get_unchecked_mut(y * CN + output_stride * output_x..);

                for j in 0..rem_x {
                    if FLOP {
                        std::ptr::copy_nonoverlapping(
                            dst_buffer.get_unchecked_mut(j * (BLOCK_SIZE * CN)..).as_mut_ptr(),
                            dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                            BLOCK_SIZE * CN,
                        );
                    } else {
                        std::ptr::copy_nonoverlapping(
                            dst_buffer.get_unchecked_mut((BLOCK_SIZE - j - 1) * (BLOCK_SIZE * CN)..).as_mut_ptr(),
                            dst.get_unchecked_mut(j * output_stride..).as_mut_ptr(),
                            BLOCK_SIZE * CN,
                        );
                    }
                }
            }

            y += BLOCK_SIZE;
        }
    }

    y
}

fn transpose_rgba8_impl<const FLOP: bool, const FLIP: bool>(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
) {
    const CN: usize = 4;

    let mut y = 0usize;

    y = transpose_executor::<8, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockNeon8x8::<FLOP, FLIP> {},
    );


    y = transpose_executor::<4, 4, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
        TransposeBlockNeon4x4::<FLOP, FLIP> {},
    );

    transpose_section::<CN, FLOP, FLIP>(
        input,
        input_stride,
        output,
        output_stride,
        width,
        height,
        y,
    )
}

pub fn transpose_rgba8_chunked(
    input: &[u8],
    input_stride: usize,
    output: &mut [u8],
    output_stride: usize,
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    if input.len() != input_stride * height {
        return Err(TransposeError::MismatchDimensions);
    }
    if output.len() != output_stride * width {
        return Err(TransposeError::MismatchDimensions);
    }
    if input_stride < width * 4 {
        return Err(TransposeError::MismatchDimensions);
    }
    if output_stride < height * 4 {
        return Err(TransposeError::MismatchDimensions);
    }

    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_rgba8_impl::<false, false>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
            ),
            FlopMode::Flop => transpose_rgba8_impl::<true, false>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
            ),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_rgba8_impl::<false, true>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
            ),
            FlopMode::Flop => transpose_rgba8_impl::<true, true>(
                input,
                input_stride,
                output,
                output_stride,
                width,
                height,
            ),
        },
    }
    Ok(())
}
