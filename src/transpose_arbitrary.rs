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
use crate::{FlipMode, FlopMode, TransposeError};

#[inline(always)]
fn transpose_block<V: Copy, const FLOP: bool, const FLIP: bool>(
    input: &[V],
    output: &mut [V],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    block_width: usize,
    block_height: usize,
) {
    for inner_x in 0..block_width {
        for inner_y in 0..block_height {
            let x = start_x + inner_x;
            let y = start_y + inner_y;

            let input_y = if FLIP { height - 1 - y } else { y };
            let output_x = if FLOP { x } else { width - 1 - x };

            let input_index = x + input_y * width;
            let output_index = y + output_x * height;

            #[cfg(feature = "unsafe")]
            {
                unsafe {
                    *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
                }
            }
            #[cfg(not(feature = "unsafe"))]
            {
                output[output_index] = input[input_index];
            }
        }
    }
}

#[inline(always)]
fn transpose_block_segmented<
    T: Copy,
    const FLOP: bool,
    const FLIP: bool,
    const BLOCK_SIZE_X: usize,
    const BLOCK_SIZE_Y: usize,
>(
    input: &[T],
    output: &mut [T],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
) {
    let height_per_div = BLOCK_SIZE_Y / 4;
    for subblock in 0..4 {
        for inner_x in 0..BLOCK_SIZE_X {
            for inner_y in 0..height_per_div {
                let x = start_x + inner_x;
                let y = start_y + inner_y + subblock * height_per_div;

                let input_y = if FLIP { height - 1 - y } else { y };
                let output_x = if FLOP { x } else { width - 1 - x };

                let input_index = x + input_y * width;
                let output_index = y + output_x * height;

                #[cfg(feature = "unsafe")]
                {
                    unsafe {
                        *output.get_unchecked_mut(output_index) = *input.get_unchecked(input_index);
                    }
                }
                #[cfg(not(feature = "unsafe"))]
                {
                    output[output_index] = input[input_index];
                }
            }
        }
    }
}

fn transpose_arbitrary_impl<V: Copy, const FLOP: bool, const FLIP: bool>(
    input: &[V],
    output: &mut [V],
    row_start: usize,
    row_end: usize,
    col_start: usize,
    col_end: usize,
    total_columns: usize,
    total_rows: usize,
) {
    let nbr_rows = row_end - row_start;
    let nbr_cols = col_end - col_start;
    const LIMIT: usize = 128;
    const BLOCK_SIZE: usize = 16;
    if (nbr_rows <= LIMIT && nbr_cols <= LIMIT) || nbr_rows <= 2 || nbr_cols <= 2 {
        let x_block_count = nbr_cols / BLOCK_SIZE;
        let y_block_count = nbr_rows / BLOCK_SIZE;

        let remainder_x = nbr_cols - x_block_count * BLOCK_SIZE;
        let remainder_y = nbr_rows - y_block_count * BLOCK_SIZE;

        for y_block in 0..y_block_count {
            for x_block in 0..x_block_count {
                transpose_block_segmented::<V, FLOP, FLIP, BLOCK_SIZE, BLOCK_SIZE>(
                    input,
                    output,
                    total_columns,
                    total_rows,
                    col_start + x_block * BLOCK_SIZE,
                    row_start + y_block * BLOCK_SIZE,
                );
            }

            if remainder_x > 0 {
                transpose_block::<V, FLOP, FLIP>(
                    input,
                    output,
                    total_columns,
                    total_rows,
                    col_start + x_block_count * BLOCK_SIZE,
                    row_start + y_block * BLOCK_SIZE,
                    remainder_x,
                    BLOCK_SIZE,
                );
            }
        }

        if remainder_y > 0 {
            for x_block in 0..x_block_count {
                transpose_block::<V, FLOP, FLIP>(
                    input,
                    output,
                    total_columns,
                    total_rows,
                    col_start + x_block * BLOCK_SIZE,
                    row_start + y_block_count * BLOCK_SIZE,
                    BLOCK_SIZE,
                    remainder_y,
                );
            }

            if remainder_x > 0 {
                transpose_block::<V, FLOP, FLIP>(
                    input,
                    output,
                    total_columns,
                    total_rows,
                    col_start + x_block_count * BLOCK_SIZE,
                    row_start + y_block_count * BLOCK_SIZE,
                    remainder_x,
                    remainder_y,
                );
            }
        }
    } else if nbr_rows >= nbr_cols {
        transpose_arbitrary_impl::<V, FLOP, FLIP>(
            input,
            output,
            row_start,
            row_start + (nbr_rows / 2),
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
        transpose_arbitrary_impl::<V, FLOP, FLIP>(
            input,
            output,
            row_start + (nbr_rows / 2),
            row_end,
            col_start,
            col_end,
            total_columns,
            total_rows,
        );
    } else {
        transpose_arbitrary_impl::<V, FLOP, FLIP>(
            input,
            output,
            row_start,
            row_end,
            col_start,
            col_start + (nbr_cols / 2),
            total_columns,
            total_rows,
        );
        transpose_arbitrary_impl::<V, FLOP, FLIP>(
            input,
            output,
            row_start,
            row_end,
            col_start + (nbr_cols / 2),
            col_end,
            total_columns,
            total_rows,
        );
    }
}

/// Performs arbitrary transposition
///
/// # Arguments
///
/// * `input`: Input date
/// * `output`: Output data
/// * `width`: Array width
/// * `height`: Array height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn transpose_arbitrary<V: Copy>(
    input: &[V],
    output: &mut [V],
    width: usize,
    height: usize,
    flip_mode: FlipMode,
    flop_mode: FlopMode,
) -> Result<(), TransposeError> {
    if input.len() != output.len() {
        return Err(TransposeError::MismatchDimensions);
    }

    if input.len() != width * height {
        return Err(TransposeError::MismatchDimensions);
    }

    match flip_mode {
        FlipMode::NoFlip => match flop_mode {
            FlopMode::NoFlop => transpose_arbitrary_impl::<V, false, false>(
                input, output, 0, height, 0, width, width, height,
            ),
            FlopMode::Flop => transpose_arbitrary_impl::<V, true, false>(
                input, output, 0, height, 0, width, width, height,
            ),
        },
        FlipMode::Flip => match flop_mode {
            FlopMode::NoFlop => transpose_arbitrary_impl::<V, false, true>(
                input, output, 0, height, 0, width, width, height,
            ),
            FlopMode::Flop => transpose_arbitrary_impl::<V, true, true>(
                input, output, 0, height, 0, width, width, height,
            ),
        },
    }

    Ok(())
}
