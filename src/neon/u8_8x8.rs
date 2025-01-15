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

use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_transpose_8x8_impl<const FLIP: bool>(
    v0: uint8x8x4_t,
    v1: uint8x8x4_t,
) -> (uint8x8x4_t, uint8x8x4_t) {
    let t0 = vtrn_u8(v0.0, v0.1);
    let t1 = vtrn_u8(v0.2, v0.3);
    let t2 = vtrn_u8(v1.0, v1.1);
    let t3 = vtrn_u8(v1.2, v1.3);

    let t4 = vtrn_u16(vreinterpret_u16_u8(t0.0), vreinterpret_u16_u8(t1.0));
    let t5 = vtrn_u16(vreinterpret_u16_u8(t0.1), vreinterpret_u16_u8(t1.1));
    let t6 = vtrn_u16(vreinterpret_u16_u8(t2.0), vreinterpret_u16_u8(t3.0));
    let t7 = vtrn_u16(vreinterpret_u16_u8(t2.1), vreinterpret_u16_u8(t3.1));

    let t8 = vtrn_u32(vreinterpret_u32_u16(t4.0), vreinterpret_u32_u16(t6.0));
    let t9 = vtrn_u32(vreinterpret_u32_u16(t4.1), vreinterpret_u32_u16(t6.1));
    let t10 = vtrn_u32(vreinterpret_u32_u16(t5.0), vreinterpret_u32_u16(t7.0));
    let t11 = vtrn_u32(vreinterpret_u32_u16(t5.1), vreinterpret_u32_u16(t7.1));

    if FLIP {
        (
            uint8x8x4_t(
                vrev64_u8(vreinterpret_u8_u32(t8.0)),
                vrev64_u8(vreinterpret_u8_u32(t10.0)),
                vrev64_u8(vreinterpret_u8_u32(t9.0)),
                vrev64_u8(vreinterpret_u8_u32(t11.0)),
            ),
            uint8x8x4_t(
                vrev64_u8(vreinterpret_u8_u32(t8.1)),
                vrev64_u8(vreinterpret_u8_u32(t10.1)),
                vrev64_u8(vreinterpret_u8_u32(t9.1)),
                vrev64_u8(vreinterpret_u8_u32(t11.1)),
            ),
        )
    } else {
        (
            uint8x8x4_t(
                vreinterpret_u8_u32(t8.0),
                vreinterpret_u8_u32(t10.0),
                vreinterpret_u8_u32(t9.0),
                vreinterpret_u8_u32(t11.0),
            ),
            uint8x8x4_t(
                vreinterpret_u8_u32(t8.1),
                vreinterpret_u8_u32(t10.1),
                vreinterpret_u8_u32(t9.1),
                vreinterpret_u8_u32(t11.1),
            ),
        )
    }
}

#[inline(always)]
pub(crate) fn neon_transpose_u8_8x8<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1_u8(src.get_unchecked(0..).as_ptr());
        let row1 = vld1_u8(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld1_u8(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld1_u8(src.get_unchecked(3 * src_stride..).as_ptr());
        let row4 = vld1_u8(src.get_unchecked(4 * src_stride..).as_ptr());
        let row5 = vld1_u8(src.get_unchecked(5 * src_stride..).as_ptr());
        let row6 = vld1_u8(src.get_unchecked(6 * src_stride..).as_ptr());
        let row7 = vld1_u8(src.get_unchecked(7 * src_stride..).as_ptr());

        let (v0, v1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0, row1, row2, row3),
            uint8x8x4_t(row4, row5, row6, row7),
        );

        if FLOP {
            vst1_u8(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            vst1_u8(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            vst1_u8(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            vst1_u8(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
            vst1_u8(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v1.0);
            vst1_u8(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v1.1);
            vst1_u8(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v1.2);
            vst1_u8(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v1.3);
        } else {
            vst1_u8(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v0.0);
            vst1_u8(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v0.1);
            vst1_u8(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v0.2);
            vst1_u8(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v0.3);
            vst1_u8(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v1.0);
            vst1_u8(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v1.1);
            vst1_u8(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v1.2);
            vst1_u8(dst.get_unchecked_mut(0..).as_mut_ptr(), v1.3);
        }
    }
}
