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
use crate::TransposeError;

/// Performs arbitrary flipping
///
/// # Arguments
///
/// * `input`: Input date
/// * `output`: Output data
/// * `width`: Array width
/// * `height`: Array height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_arbitrary<V: Copy>(
    input: &[V],
    output: &mut [V],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    if input.len() != output.len() {
        return Err(TransposeError::MismatchDimensions);
    }

    if input.len() != width * height {
        return Err(TransposeError::MismatchDimensions);
    }

    for (dst, src) in output
        .chunks_exact_mut(width)
        .zip(input.chunks_exact(width))
    {
        for (dst, src) in dst.iter_mut().rev().zip(src.iter()) {
            *dst = *src;
        }
    }

    Ok(())
}

/// Performs plane image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(matrix, target, width, height)
}

/// Performs plane with alpha flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane_with_alpha(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 2]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 2]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGB image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgb(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 3]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 3]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGBA image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgba(
    matrix: &[u8],
    target: &mut [u8],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u8; 4]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u8; 4]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs plane image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane16(
    matrix: &[u16],
    target: &mut [u16],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(matrix, target, width, height)
}

/// Performs plane with alpha image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane16_with_alpha(
    matrix: &[u16],
    target: &mut [u16],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u16; 2]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u16; 2]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGB image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgb16(
    matrix: &[u16],
    target: &mut [u16],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u16; 3]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u16; 3]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGBA image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgba16(
    matrix: &[u16],
    target: &mut [u16],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[u16; 4]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[u16; 4]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs plane image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane_f32(
    matrix: &[f32],
    target: &mut [f32],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    flip_arbitrary(matrix, target, width, height)
}

/// Performs plane with alpha image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
/// * `flip_mode`: see [FlipMode]
/// * `flop_mode`: see [FlopMode]
///
/// returns: Result<(), TransposeError>
///
pub fn flip_plane_f32_with_alpha(
    matrix: &[f32],
    target: &mut [f32],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[f32; 2]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[f32; 2]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGB image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgb_f32(
    matrix: &[f32],
    target: &mut [f32],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[f32; 3]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[f32; 3]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}

/// Performs RGBA image flipping
///
/// # Arguments
///
/// * `input`: Input data
/// * `output`: Output data
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), TransposeError>
///
pub fn flip_rgba_f32(
    matrix: &[f32],
    target: &mut [f32],
    width: usize,
    height: usize,
) -> Result<(), TransposeError> {
    let casted_source: &[[f32; 4]] = match bytemuck::try_cast_slice(matrix) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    let casted_target: &mut [[f32; 4]] = match bytemuck::try_cast_slice_mut(target) {
        Err(_) => return Err(TransposeError::MismatchDimensions),
        Ok(casted_source) => casted_source,
    };
    flip_arbitrary(casted_source, casted_target, width, height)
}
