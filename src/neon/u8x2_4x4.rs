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
use crate::neon::utils::{xvld1_u8_u16, xvst1_u8_u16};
use std::arch::aarch64::uint16x4x4_t;

#[inline]
pub(crate) fn neon_transpose_u8x2_4x4<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = xvld1_u8_u16(src.get_unchecked(0..).as_ptr());
        let row1 = xvld1_u8_u16(src.get_unchecked(src_stride..).as_ptr());
        let row2 = xvld1_u8_u16(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = xvld1_u8_u16(src.get_unchecked(3 * src_stride..).as_ptr());

        let v0 = crate::neon::u16_4x4::neon_transpose_u16_4x4_impl::<FLIP>(uint16x4x4_t(
            row0, row1, row2, row3,
        ));

        if FLOP {
            xvst1_u8_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            xvst1_u8_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            xvst1_u8_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            xvst1_u8_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
        } else {
            xvst1_u8_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.0);
            xvst1_u8_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.1);
            xvst1_u8_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.2);
            xvst1_u8_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.3);
        }
    }
}
