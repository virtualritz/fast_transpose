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

use crate::neon::utils::{vrev128_u8, vtrnq_u64_to_u16};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_transpose_16x16_impl<const FLIP: bool>(
    v0: uint8x16x4_t,
    v1: uint8x16x4_t,
    v2: uint8x16x4_t,
    v3: uint8x16x4_t,
) -> (uint8x16x4_t, uint8x16x4_t, uint8x16x4_t, uint8x16x4_t) {
    let w0 = vzipq_u8(v0.0, v0.1);
    let w1 = vzipq_u8(v0.2, v0.3);
    let w2 = vzipq_u8(v1.0, v1.1);
    let w3 = vzipq_u8(v1.2, v1.3);

    let w4 = vzipq_u8(v2.0, v2.1);
    let w5 = vzipq_u8(v2.2, v2.3);
    let w6 = vzipq_u8(v3.0, v3.1);
    let w7 = vzipq_u8(v3.2, v3.3);

    let w8 = vzipq_u16(vreinterpretq_u16_u8(w0.0), vreinterpretq_u16_u8(w1.0));
    let w9 = vzipq_u16(vreinterpretq_u16_u8(w2.0), vreinterpretq_u16_u8(w3.0));
    let w10 = vzipq_u16(vreinterpretq_u16_u8(w4.0), vreinterpretq_u16_u8(w5.0));
    let w11 = vzipq_u16(vreinterpretq_u16_u8(w6.0), vreinterpretq_u16_u8(w7.0));

    let w12 = vzipq_u32(vreinterpretq_u32_u16(w8.0), vreinterpretq_u32_u16(w9.0));
    let w13 = vzipq_u32(vreinterpretq_u32_u16(w10.0), vreinterpretq_u32_u16(w11.0));
    let w14 = vzipq_u32(vreinterpretq_u32_u16(w8.1), vreinterpretq_u32_u16(w9.1));
    let w15 = vzipq_u32(vreinterpretq_u32_u16(w10.1), vreinterpretq_u32_u16(w11.1));

    let d01 = vtrnq_u64_to_u16(w12.0, w13.0);
    let rd0 = vreinterpretq_u8_u16(d01.0);
    let rd1 = vreinterpretq_u8_u16(d01.1);
    let d23 = vtrnq_u64_to_u16(w12.1, w13.1);
    let rd2 = vreinterpretq_u8_u16(d23.0);
    let rd3 = vreinterpretq_u8_u16(d23.1);
    let d45 = vtrnq_u64_to_u16(w14.0, w15.0);
    let rd4 = vreinterpretq_u8_u16(d45.0);
    let rd5 = vreinterpretq_u8_u16(d45.1);
    let d67 = vtrnq_u64_to_u16(w14.1, w15.1);
    let rd6 = vreinterpretq_u8_u16(d67.0);
    let rd7 = vreinterpretq_u8_u16(d67.1);

    // upper half
    let w8 = vzipq_u16(vreinterpretq_u16_u8(w0.1), vreinterpretq_u16_u8(w1.1));
    let w9 = vzipq_u16(vreinterpretq_u16_u8(w2.1), vreinterpretq_u16_u8(w3.1));
    let w10 = vzipq_u16(vreinterpretq_u16_u8(w4.1), vreinterpretq_u16_u8(w5.1));
    let w11 = vzipq_u16(vreinterpretq_u16_u8(w6.1), vreinterpretq_u16_u8(w7.1));

    let w12 = vzipq_u32(vreinterpretq_u32_u16(w8.0), vreinterpretq_u32_u16(w9.0));
    let w13 = vzipq_u32(vreinterpretq_u32_u16(w10.0), vreinterpretq_u32_u16(w11.0));
    let w14 = vzipq_u32(vreinterpretq_u32_u16(w8.1), vreinterpretq_u32_u16(w9.1));
    let w15 = vzipq_u32(vreinterpretq_u32_u16(w10.1), vreinterpretq_u32_u16(w11.1));

    let hd01 = vtrnq_u64_to_u16(w12.0, w13.0);
    let rd8 = vreinterpretq_u8_u16(hd01.0);
    let rd9 = vreinterpretq_u8_u16(hd01.1);
    let hd23 = vtrnq_u64_to_u16(w12.1, w13.1);
    let rd10 = vreinterpretq_u8_u16(hd23.0);
    let rd11 = vreinterpretq_u8_u16(hd23.1);
    let hd45 = vtrnq_u64_to_u16(w14.0, w15.0);
    let rd12 = vreinterpretq_u8_u16(hd45.0);
    let rd13 = vreinterpretq_u8_u16(hd45.1);
    let hd67 = vtrnq_u64_to_u16(w14.1, w15.1);
    let rd14 = vreinterpretq_u8_u16(hd67.0);
    let rd15 = vreinterpretq_u8_u16(hd67.1);

    if FLIP {
        (
            uint8x16x4_t(
                vrev128_u8(rd0),
                vrev128_u8(rd1),
                vrev128_u8(rd2),
                vrev128_u8(rd3),
            ),
            uint8x16x4_t(
                vrev128_u8(rd4),
                vrev128_u8(rd5),
                vrev128_u8(rd6),
                vrev128_u8(rd7),
            ),
            uint8x16x4_t(
                vrev128_u8(rd8),
                vrev128_u8(rd9),
                vrev128_u8(rd10),
                vrev128_u8(rd11),
            ),
            uint8x16x4_t(
                vrev128_u8(rd12),
                vrev128_u8(rd13),
                vrev128_u8(rd14),
                vrev128_u8(rd15),
            ),
        )
    } else {
        (
            uint8x16x4_t(rd0, rd1, rd2, rd3),
            uint8x16x4_t(rd4, rd5, rd6, rd7),
            uint8x16x4_t(rd8, rd9, rd10, rd11),
            uint8x16x4_t(rd12, rd13, rd14, rd15),
        )
    }
}

#[inline(always)]
pub(crate) fn neon_transpose_u8_16x16<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_u8(src.get_unchecked(0..).as_ptr());
        let row1 = vld1q_u8(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld1q_u8(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld1q_u8(src.get_unchecked(3 * src_stride..).as_ptr());
        let row4 = vld1q_u8(src.get_unchecked(4 * src_stride..).as_ptr());
        let row5 = vld1q_u8(src.get_unchecked(5 * src_stride..).as_ptr());
        let row6 = vld1q_u8(src.get_unchecked(6 * src_stride..).as_ptr());
        let row7 = vld1q_u8(src.get_unchecked(7 * src_stride..).as_ptr());
        let row8 = vld1q_u8(src.get_unchecked(8 * src_stride..).as_ptr());
        let row9 = vld1q_u8(src.get_unchecked(9 * src_stride..).as_ptr());
        let row10 = vld1q_u8(src.get_unchecked(10 * src_stride..).as_ptr());
        let row11 = vld1q_u8(src.get_unchecked(11 * src_stride..).as_ptr());
        let row12 = vld1q_u8(src.get_unchecked(12 * src_stride..).as_ptr());
        let row13 = vld1q_u8(src.get_unchecked(13 * src_stride..).as_ptr());
        let row14 = vld1q_u8(src.get_unchecked(14 * src_stride..).as_ptr());
        let row15 = vld1q_u8(src.get_unchecked(15 * src_stride..).as_ptr());

        let set0 = uint8x16x4_t(row0, row1, row2, row3);
        let set1 = uint8x16x4_t(row4, row5, row6, row7);
        let set2 = uint8x16x4_t(row8, row9, row10, row11);
        let set3 = uint8x16x4_t(row12, row13, row14, row15);

        let (v0, v1, v2, v3) = neon_transpose_16x16_impl::<FLIP>(set0, set1, set2, set3);

        if FLOP {
            vst1q_u8(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            vst1q_u8(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            vst1q_u8(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            vst1q_u8(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
            vst1q_u8(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v1.0);
            vst1q_u8(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v1.1);
            vst1q_u8(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v1.2);
            vst1q_u8(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v1.3);
            vst1q_u8(dst.get_unchecked_mut(8 * dst_stride..).as_mut_ptr(), v2.0);
            vst1q_u8(dst.get_unchecked_mut(9 * dst_stride..).as_mut_ptr(), v2.1);
            vst1q_u8(dst.get_unchecked_mut(10 * dst_stride..).as_mut_ptr(), v2.2);
            vst1q_u8(dst.get_unchecked_mut(11 * dst_stride..).as_mut_ptr(), v2.3);
            vst1q_u8(dst.get_unchecked_mut(12 * dst_stride..).as_mut_ptr(), v3.0);
            vst1q_u8(dst.get_unchecked_mut(13 * dst_stride..).as_mut_ptr(), v3.1);
            vst1q_u8(dst.get_unchecked_mut(14 * dst_stride..).as_mut_ptr(), v3.2);
            vst1q_u8(dst.get_unchecked_mut(15 * dst_stride..).as_mut_ptr(), v3.3);
        } else {
            vst1q_u8(dst.get_unchecked_mut(15 * dst_stride..).as_mut_ptr(), v0.0);
            vst1q_u8(dst.get_unchecked_mut(14 * dst_stride..).as_mut_ptr(), v0.1);
            vst1q_u8(dst.get_unchecked_mut(13 * dst_stride..).as_mut_ptr(), v0.2);
            vst1q_u8(dst.get_unchecked_mut(12 * dst_stride..).as_mut_ptr(), v0.3);
            vst1q_u8(dst.get_unchecked_mut(11 * dst_stride..).as_mut_ptr(), v1.0);
            vst1q_u8(dst.get_unchecked_mut(10 * dst_stride..).as_mut_ptr(), v1.1);
            vst1q_u8(dst.get_unchecked_mut(9 * dst_stride..).as_mut_ptr(), v1.2);
            vst1q_u8(dst.get_unchecked_mut(8 * dst_stride..).as_mut_ptr(), v1.3);
            vst1q_u8(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v2.0);
            vst1q_u8(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v2.1);
            vst1q_u8(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v2.2);
            vst1q_u8(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v2.3);
            vst1q_u8(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v3.0);
            vst1q_u8(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v3.1);
            vst1q_u8(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v3.2);
            vst1q_u8(dst.get_unchecked_mut(0..).as_mut_ptr(), v3.3);
        }
    }
}
