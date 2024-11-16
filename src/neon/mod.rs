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
#![deny(unreachable_pub)]
mod transpose_16x16_u8;
mod transpose_4x4_f32;
mod transpose_8x8_u16;
mod transpose_8x8_u8;
mod utils;

pub(crate) use transpose_16x16_u8::{
    neon_transpose_16x16, neon_transpose_16x16_intl_2, neon_transpose_16x16_intl_3,
    neon_transpose_16x16_intl_4,
};
pub(crate) use transpose_4x4_f32::{
    neon_transpose_4x4_f32, neon_transpose_4x4_f32_intl_2, neon_transpose_4x4_f32_intl_3,
    neon_transpose_4x4_f32_intl_4,
};
pub(crate) use transpose_8x8_u16::{
    neon_transpose_8x8_u16, neon_transpose_8x8_u16_intl_2, neon_transpose_8x8_u16_intl_3,
    neon_transpose_8x8_u16_intl_4,
};
pub(crate) use transpose_8x8_u8::{
    neon_transpose_8x8, neon_transpose_8x8_intl_2, neon_transpose_8x8_intl_3,
    neon_transpose_8x8_intl_4,
};
