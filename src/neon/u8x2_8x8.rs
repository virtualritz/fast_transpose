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
use crate::neon::utils::{xvld1q_u8_u16, xvst1q_u8_u16};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) fn neon_transpose_u8x2_8x8<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = xvld1q_u8_u16(src.get_unchecked(0..).as_ptr());
        let row1 = xvld1q_u8_u16(src.get_unchecked(src_stride..).as_ptr());
        let row2 = xvld1q_u8_u16(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = xvld1q_u8_u16(src.get_unchecked(3 * src_stride..).as_ptr());
        let row4 = xvld1q_u8_u16(src.get_unchecked(4 * src_stride..).as_ptr());
        let row5 = xvld1q_u8_u16(src.get_unchecked(5 * src_stride..).as_ptr());
        let row6 = xvld1q_u8_u16(src.get_unchecked(6 * src_stride..).as_ptr());
        let row7 = xvld1q_u8_u16(src.get_unchecked(7 * src_stride..).as_ptr());

        let (v0, v1) = crate::neon::u16_8x8::neon_transpose_u16_4x4_impl::<FLIP>(
            uint16x8x4_t(row0, row1, row2, row3),
            uint16x8x4_t(row4, row5, row6, row7),
        );

        if FLOP {
            xvst1q_u8_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            xvst1q_u8_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            xvst1q_u8_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            xvst1q_u8_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
            xvst1q_u8_u16(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v1.0);
            xvst1q_u8_u16(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v1.1);
            xvst1q_u8_u16(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v1.2);
            xvst1q_u8_u16(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v1.3);
        } else {
            xvst1q_u8_u16(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v0.0);
            xvst1q_u8_u16(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v0.1);
            xvst1q_u8_u16(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v0.2);
            xvst1q_u8_u16(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v0.3);
            xvst1q_u8_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v1.0);
            xvst1q_u8_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v1.1);
            xvst1q_u8_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v1.2);
            xvst1q_u8_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v1.3);
        }
    }
}
