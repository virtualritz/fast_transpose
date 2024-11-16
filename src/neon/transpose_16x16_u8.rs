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

pub(crate) fn neon_transpose_16x16<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_u8(&src[0]);
        let row1 = vld1q_u8(&src[src_stride]);
        let row2 = vld1q_u8(&src[2 * src_stride]);
        let row3 = vld1q_u8(&src[3 * src_stride]);
        let row4 = vld1q_u8(&src[4 * src_stride]);
        let row5 = vld1q_u8(&src[5 * src_stride]);
        let row6 = vld1q_u8(&src[6 * src_stride]);
        let row7 = vld1q_u8(&src[7 * src_stride]);
        let row8 = vld1q_u8(&src[8 * src_stride]);
        let row9 = vld1q_u8(&src[9 * src_stride]);
        let row10 = vld1q_u8(&src[10 * src_stride]);
        let row11 = vld1q_u8(&src[11 * src_stride]);
        let row12 = vld1q_u8(&src[12 * src_stride]);
        let row13 = vld1q_u8(&src[13 * src_stride]);
        let row14 = vld1q_u8(&src[14 * src_stride]);
        let row15 = vld1q_u8(&src[15 * src_stride]);

        let set0 = uint8x16x4_t(row0, row1, row2, row3);
        let set1 = uint8x16x4_t(row4, row5, row6, row7);
        let set2 = uint8x16x4_t(row8, row9, row10, row11);
        let set3 = uint8x16x4_t(row12, row13, row14, row15);

        let (v0, v1, v2, v3) = neon_transpose_16x16_impl::<FLIP>(set0, set1, set2, set3);

        if FLOP {
            vst1q_u8(&mut dst[0], v0.0);
            vst1q_u8(&mut dst[dst_stride], v0.1);
            vst1q_u8(&mut dst[2 * dst_stride], v0.2);
            vst1q_u8(&mut dst[3 * dst_stride], v0.3);
            vst1q_u8(&mut dst[4 * dst_stride], v1.0);
            vst1q_u8(&mut dst[5 * dst_stride], v1.1);
            vst1q_u8(&mut dst[6 * dst_stride], v1.2);
            vst1q_u8(&mut dst[7 * dst_stride], v1.3);
            vst1q_u8(&mut dst[8 * dst_stride], v2.0);
            vst1q_u8(&mut dst[9 * dst_stride], v2.1);
            vst1q_u8(&mut dst[10 * dst_stride], v2.2);
            vst1q_u8(&mut dst[11 * dst_stride], v2.3);
            vst1q_u8(&mut dst[12 * dst_stride], v3.0);
            vst1q_u8(&mut dst[13 * dst_stride], v3.1);
            vst1q_u8(&mut dst[14 * dst_stride], v3.2);
            vst1q_u8(&mut dst[15 * dst_stride], v3.3);
        } else {
            vst1q_u8(&mut dst[15 * dst_stride], v0.0);
            vst1q_u8(&mut dst[14 * dst_stride], v0.1);
            vst1q_u8(&mut dst[13 * dst_stride], v0.2);
            vst1q_u8(&mut dst[12 * dst_stride], v0.3);
            vst1q_u8(&mut dst[11 * dst_stride], v1.0);
            vst1q_u8(&mut dst[10 * dst_stride], v1.1);
            vst1q_u8(&mut dst[9 * dst_stride], v1.2);
            vst1q_u8(&mut dst[8 * dst_stride], v1.3);
            vst1q_u8(&mut dst[7 * dst_stride], v2.0);
            vst1q_u8(&mut dst[6 * dst_stride], v2.1);
            vst1q_u8(&mut dst[5 * dst_stride], v2.2);
            vst1q_u8(&mut dst[4 * dst_stride], v2.3);
            vst1q_u8(&mut dst[3 * dst_stride], v3.0);
            vst1q_u8(&mut dst[2 * dst_stride], v3.1);
            vst1q_u8(&mut dst[dst_stride], v3.2);
            vst1q_u8(&mut dst[0], v3.3);
        }
    }
}

pub(crate) fn neon_transpose_16x16_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld3q_u8(&src[0]);
        let row1 = vld3q_u8(&src[src_stride]);
        let row2 = vld3q_u8(&src[2 * src_stride]);
        let row3 = vld3q_u8(&src[3 * src_stride]);
        let row4 = vld3q_u8(&src[4 * src_stride]);
        let row5 = vld3q_u8(&src[5 * src_stride]);
        let row6 = vld3q_u8(&src[6 * src_stride]);
        let row7 = vld3q_u8(&src[7 * src_stride]);
        let row8 = vld3q_u8(&src[8 * src_stride]);
        let row9 = vld3q_u8(&src[9 * src_stride]);
        let row10 = vld3q_u8(&src[10 * src_stride]);
        let row11 = vld3q_u8(&src[11 * src_stride]);
        let row12 = vld3q_u8(&src[12 * src_stride]);
        let row13 = vld3q_u8(&src[13 * src_stride]);
        let row14 = vld3q_u8(&src[14 * src_stride]);
        let row15 = vld3q_u8(&src[15 * src_stride]);

        let rset0 = uint8x16x4_t(row0.0, row1.0, row2.0, row3.0);
        let rset1 = uint8x16x4_t(row4.0, row5.0, row6.0, row7.0);
        let rset2 = uint8x16x4_t(row8.0, row9.0, row10.0, row11.0);
        let rset3 = uint8x16x4_t(row12.0, row13.0, row14.0, row15.0);

        let (r0, r1, r2, r3) = neon_transpose_16x16_impl::<FLIP>(rset0, rset1, rset2, rset3);

        let gset0 = uint8x16x4_t(row0.1, row1.1, row2.1, row3.1);
        let gset1 = uint8x16x4_t(row4.1, row5.1, row6.1, row7.1);
        let gset2 = uint8x16x4_t(row8.1, row9.1, row10.1, row11.1);
        let gset3 = uint8x16x4_t(row12.1, row13.1, row14.1, row15.1);

        let (g0, g1, g2, g3) = neon_transpose_16x16_impl::<FLIP>(gset0, gset1, gset2, gset3);

        let bset0 = uint8x16x4_t(row0.2, row1.2, row2.2, row3.2);
        let bset1 = uint8x16x4_t(row4.2, row5.2, row6.2, row7.2);
        let bset2 = uint8x16x4_t(row8.2, row9.2, row10.2, row11.2);
        let bset3 = uint8x16x4_t(row12.2, row13.2, row14.2, row15.2);

        let (b0, b1, b2, b3) = neon_transpose_16x16_impl::<FLIP>(bset0, bset1, bset2, bset3);

        if FLOP {
            vst3q_u8(&mut dst[0], uint8x16x3_t(r0.0, g0.0, b0.0));
            vst3q_u8(&mut dst[dst_stride], uint8x16x3_t(r0.1, g0.1, b0.1));
            vst3q_u8(&mut dst[2 * dst_stride], uint8x16x3_t(r0.2, g0.2, b0.2));
            vst3q_u8(&mut dst[3 * dst_stride], uint8x16x3_t(r0.3, g0.3, b0.3));
            vst3q_u8(&mut dst[4 * dst_stride], uint8x16x3_t(r1.0, g1.0, b1.0));
            vst3q_u8(&mut dst[5 * dst_stride], uint8x16x3_t(r1.1, g1.1, b1.1));
            vst3q_u8(&mut dst[6 * dst_stride], uint8x16x3_t(r1.2, g1.2, b1.2));
            vst3q_u8(&mut dst[7 * dst_stride], uint8x16x3_t(r1.3, g1.3, b1.3));
            vst3q_u8(&mut dst[8 * dst_stride], uint8x16x3_t(r2.0, g2.0, b2.0));
            vst3q_u8(&mut dst[9 * dst_stride], uint8x16x3_t(r2.1, g2.1, b2.1));
            vst3q_u8(&mut dst[10 * dst_stride], uint8x16x3_t(r2.2, g2.2, b2.2));
            vst3q_u8(&mut dst[11 * dst_stride], uint8x16x3_t(r2.3, g2.3, b2.3));
            vst3q_u8(&mut dst[12 * dst_stride], uint8x16x3_t(r3.0, g3.0, b3.0));
            vst3q_u8(&mut dst[13 * dst_stride], uint8x16x3_t(r3.1, g3.1, b3.1));
            vst3q_u8(&mut dst[14 * dst_stride], uint8x16x3_t(r3.2, g3.2, b3.2));
            vst3q_u8(&mut dst[15 * dst_stride], uint8x16x3_t(r3.3, g3.3, b3.3));
        } else {
            vst3q_u8(&mut dst[15 * dst_stride], uint8x16x3_t(r0.0, g0.0, b0.0));
            vst3q_u8(&mut dst[14 * dst_stride], uint8x16x3_t(r0.1, g0.1, b0.1));
            vst3q_u8(&mut dst[13 * dst_stride], uint8x16x3_t(r0.2, g0.2, b0.2));
            vst3q_u8(&mut dst[12 * dst_stride], uint8x16x3_t(r0.3, g0.3, b0.3));
            vst3q_u8(&mut dst[11 * dst_stride], uint8x16x3_t(r1.0, g1.0, b1.0));
            vst3q_u8(&mut dst[10 * dst_stride], uint8x16x3_t(r1.1, g1.1, b1.1));
            vst3q_u8(&mut dst[9 * dst_stride], uint8x16x3_t(r1.2, g1.2, b1.2));
            vst3q_u8(&mut dst[8 * dst_stride], uint8x16x3_t(r1.3, g1.3, b1.3));
            vst3q_u8(&mut dst[7 * dst_stride], uint8x16x3_t(r2.0, g2.0, b2.0));
            vst3q_u8(&mut dst[6 * dst_stride], uint8x16x3_t(r2.1, g2.1, b2.1));
            vst3q_u8(&mut dst[5 * dst_stride], uint8x16x3_t(r2.2, g2.2, b2.2));
            vst3q_u8(&mut dst[4 * dst_stride], uint8x16x3_t(r2.3, g2.3, b2.3));
            vst3q_u8(&mut dst[3 * dst_stride], uint8x16x3_t(r3.0, g3.0, b3.0));
            vst3q_u8(&mut dst[2 * dst_stride], uint8x16x3_t(r3.1, g3.1, b3.1));
            vst3q_u8(&mut dst[dst_stride], uint8x16x3_t(r3.2, g3.2, b3.2));
            vst3q_u8(&mut dst[0], uint8x16x3_t(r3.3, g3.3, b3.3));
        }
    }
}
pub(crate) fn neon_transpose_16x16_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld2q_u8(&src[0]);
        let row1 = vld2q_u8(&src[src_stride]);
        let row2 = vld2q_u8(&src[2 * src_stride]);
        let row3 = vld2q_u8(&src[3 * src_stride]);
        let row4 = vld2q_u8(&src[4 * src_stride]);
        let row5 = vld2q_u8(&src[5 * src_stride]);
        let row6 = vld2q_u8(&src[6 * src_stride]);
        let row7 = vld2q_u8(&src[7 * src_stride]);
        let row8 = vld2q_u8(&src[8 * src_stride]);
        let row9 = vld2q_u8(&src[9 * src_stride]);
        let row10 = vld2q_u8(&src[10 * src_stride]);
        let row11 = vld2q_u8(&src[11 * src_stride]);
        let row12 = vld2q_u8(&src[12 * src_stride]);
        let row13 = vld2q_u8(&src[13 * src_stride]);
        let row14 = vld2q_u8(&src[14 * src_stride]);
        let row15 = vld2q_u8(&src[15 * src_stride]);

        let rset0 = uint8x16x4_t(row0.0, row1.0, row2.0, row3.0);
        let rset1 = uint8x16x4_t(row4.0, row5.0, row6.0, row7.0);
        let rset2 = uint8x16x4_t(row8.0, row9.0, row10.0, row11.0);
        let rset3 = uint8x16x4_t(row12.0, row13.0, row14.0, row15.0);

        let (r0, r1, r2, r3) = neon_transpose_16x16_impl::<FLIP>(rset0, rset1, rset2, rset3);

        let gset0 = uint8x16x4_t(row0.1, row1.1, row2.1, row3.1);
        let gset1 = uint8x16x4_t(row4.1, row5.1, row6.1, row7.1);
        let gset2 = uint8x16x4_t(row8.1, row9.1, row10.1, row11.1);
        let gset3 = uint8x16x4_t(row12.1, row13.1, row14.1, row15.1);

        let (g0, g1, g2, g3) = neon_transpose_16x16_impl::<FLIP>(gset0, gset1, gset2, gset3);

        if FLOP {
            vst2q_u8(&mut dst[0], uint8x16x2_t(r0.0, g0.0));
            vst2q_u8(&mut dst[dst_stride], uint8x16x2_t(r0.1, g0.1));
            vst2q_u8(&mut dst[2 * dst_stride], uint8x16x2_t(r0.2, g0.2));
            vst2q_u8(&mut dst[3 * dst_stride], uint8x16x2_t(r0.3, g0.3));
            vst2q_u8(&mut dst[4 * dst_stride], uint8x16x2_t(r1.0, g1.0));
            vst2q_u8(&mut dst[5 * dst_stride], uint8x16x2_t(r1.1, g1.1));
            vst2q_u8(&mut dst[6 * dst_stride], uint8x16x2_t(r1.2, g1.2));
            vst2q_u8(&mut dst[7 * dst_stride], uint8x16x2_t(r1.3, g1.3));
            vst2q_u8(&mut dst[8 * dst_stride], uint8x16x2_t(r2.0, g2.0));
            vst2q_u8(&mut dst[9 * dst_stride], uint8x16x2_t(r2.1, g2.1));
            vst2q_u8(&mut dst[10 * dst_stride], uint8x16x2_t(r2.2, g2.2));
            vst2q_u8(&mut dst[11 * dst_stride], uint8x16x2_t(r2.3, g2.3));
            vst2q_u8(&mut dst[12 * dst_stride], uint8x16x2_t(r3.0, g3.0));
            vst2q_u8(&mut dst[13 * dst_stride], uint8x16x2_t(r3.1, g3.1));
            vst2q_u8(&mut dst[14 * dst_stride], uint8x16x2_t(r3.2, g3.2));
            vst2q_u8(&mut dst[15 * dst_stride], uint8x16x2_t(r3.3, g3.3));
        } else {
            vst2q_u8(&mut dst[15 * dst_stride], uint8x16x2_t(r0.0, g0.0));
            vst2q_u8(&mut dst[14 * dst_stride], uint8x16x2_t(r0.1, g0.1));
            vst2q_u8(&mut dst[13 * dst_stride], uint8x16x2_t(r0.2, g0.2));
            vst2q_u8(&mut dst[12 * dst_stride], uint8x16x2_t(r0.3, g0.3));
            vst2q_u8(&mut dst[11 * dst_stride], uint8x16x2_t(r1.0, g1.0));
            vst2q_u8(&mut dst[10 * dst_stride], uint8x16x2_t(r1.1, g1.1));
            vst2q_u8(&mut dst[9 * dst_stride], uint8x16x2_t(r1.2, g1.2));
            vst2q_u8(&mut dst[8 * dst_stride], uint8x16x2_t(r1.3, g1.3));
            vst2q_u8(&mut dst[7 * dst_stride], uint8x16x2_t(r2.0, g2.0));
            vst2q_u8(&mut dst[6 * dst_stride], uint8x16x2_t(r2.1, g2.1));
            vst2q_u8(&mut dst[5 * dst_stride], uint8x16x2_t(r2.2, g2.2));
            vst2q_u8(&mut dst[4 * dst_stride], uint8x16x2_t(r2.3, g2.3));
            vst2q_u8(&mut dst[3 * dst_stride], uint8x16x2_t(r3.0, g3.0));
            vst2q_u8(&mut dst[2 * dst_stride], uint8x16x2_t(r3.1, g3.1));
            vst2q_u8(&mut dst[dst_stride], uint8x16x2_t(r3.2, g3.2));
            vst2q_u8(&mut dst[0], uint8x16x2_t(r3.3, g3.3));
        }
    }
}

pub(crate) fn neon_transpose_16x16_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld4q_u8(&src[0]);
        let row1 = vld4q_u8(&src[src_stride]);
        let row2 = vld4q_u8(&src[2 * src_stride]);
        let row3 = vld4q_u8(&src[3 * src_stride]);
        let row4 = vld4q_u8(&src[4 * src_stride]);
        let row5 = vld4q_u8(&src[5 * src_stride]);
        let row6 = vld4q_u8(&src[6 * src_stride]);
        let row7 = vld4q_u8(&src[7 * src_stride]);
        let row8 = vld4q_u8(&src[8 * src_stride]);
        let row9 = vld4q_u8(&src[9 * src_stride]);
        let row10 = vld4q_u8(&src[10 * src_stride]);
        let row11 = vld4q_u8(&src[11 * src_stride]);
        let row12 = vld4q_u8(&src[12 * src_stride]);
        let row13 = vld4q_u8(&src[13 * src_stride]);
        let row14 = vld4q_u8(&src[14 * src_stride]);
        let row15 = vld4q_u8(&src[15 * src_stride]);

        let rset0 = uint8x16x4_t(row0.0, row1.0, row2.0, row3.0);
        let rset1 = uint8x16x4_t(row4.0, row5.0, row6.0, row7.0);
        let rset2 = uint8x16x4_t(row8.0, row9.0, row10.0, row11.0);
        let rset3 = uint8x16x4_t(row12.0, row13.0, row14.0, row15.0);

        let (r0, r1, r2, r3) = neon_transpose_16x16_impl::<FLIP>(rset0, rset1, rset2, rset3);

        let gset0 = uint8x16x4_t(row0.1, row1.1, row2.1, row3.1);
        let gset1 = uint8x16x4_t(row4.1, row5.1, row6.1, row7.1);
        let gset2 = uint8x16x4_t(row8.1, row9.1, row10.1, row11.1);
        let gset3 = uint8x16x4_t(row12.1, row13.1, row14.1, row15.1);

        let (g0, g1, g2, g3) = neon_transpose_16x16_impl::<FLIP>(gset0, gset1, gset2, gset3);

        let bset0 = uint8x16x4_t(row0.2, row1.2, row2.2, row3.2);
        let bset1 = uint8x16x4_t(row4.2, row5.2, row6.2, row7.2);
        let bset2 = uint8x16x4_t(row8.2, row9.2, row10.2, row11.2);
        let bset3 = uint8x16x4_t(row12.2, row13.2, row14.2, row15.2);

        let (b0, b1, b2, b3) = neon_transpose_16x16_impl::<FLIP>(bset0, bset1, bset2, bset3);

        let aset0 = uint8x16x4_t(row0.3, row1.3, row2.3, row3.3);
        let aset1 = uint8x16x4_t(row4.3, row5.3, row6.3, row7.3);
        let aset2 = uint8x16x4_t(row8.3, row9.3, row10.3, row11.2);
        let aset3 = uint8x16x4_t(row12.3, row13.3, row14.3, row15.3);

        let (a0, a1, a2, a3) = neon_transpose_16x16_impl::<FLIP>(aset0, aset1, aset2, aset3);

        if FLOP {
            vst4q_u8(&mut dst[0], uint8x16x4_t(r0.0, g0.0, b0.0, a0.0));
            vst4q_u8(&mut dst[dst_stride], uint8x16x4_t(r0.1, g0.1, b0.1, a0.1));
            vst4q_u8(
                &mut dst[2 * dst_stride],
                uint8x16x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4q_u8(
                &mut dst[3 * dst_stride],
                uint8x16x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4q_u8(
                &mut dst[4 * dst_stride],
                uint8x16x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4q_u8(
                &mut dst[5 * dst_stride],
                uint8x16x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4q_u8(
                &mut dst[6 * dst_stride],
                uint8x16x4_t(r1.2, g1.2, b1.2, a1.2),
            );
            vst4q_u8(
                &mut dst[7 * dst_stride],
                uint8x16x4_t(r1.3, g1.3, b1.3, a1.3),
            );
            vst4q_u8(
                &mut dst[8 * dst_stride],
                uint8x16x4_t(r2.0, g2.0, b2.0, a2.0),
            );
            vst4q_u8(
                &mut dst[9 * dst_stride],
                uint8x16x4_t(r2.1, g2.1, b2.1, a2.1),
            );
            vst4q_u8(
                &mut dst[10 * dst_stride],
                uint8x16x4_t(r2.2, g2.2, b2.2, a2.2),
            );
            vst4q_u8(
                &mut dst[11 * dst_stride],
                uint8x16x4_t(r2.3, g2.3, b2.3, a2.3),
            );
            vst4q_u8(
                &mut dst[12 * dst_stride],
                uint8x16x4_t(r3.0, g3.0, b3.0, a3.0),
            );
            vst4q_u8(
                &mut dst[13 * dst_stride],
                uint8x16x4_t(r3.1, g3.1, b3.1, a3.1),
            );
            vst4q_u8(
                &mut dst[14 * dst_stride],
                uint8x16x4_t(r3.2, g3.2, b3.2, a3.2),
            );
            vst4q_u8(
                &mut dst[15 * dst_stride],
                uint8x16x4_t(r3.3, g3.3, b3.3, a3.3),
            );
        } else {
            vst4q_u8(
                &mut dst[15 * dst_stride],
                uint8x16x4_t(r0.0, g0.0, b0.0, a0.0),
            );
            vst4q_u8(
                &mut dst[14 * dst_stride],
                uint8x16x4_t(r0.1, g0.1, b0.1, a0.1),
            );
            vst4q_u8(
                &mut dst[13 * dst_stride],
                uint8x16x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4q_u8(
                &mut dst[12 * dst_stride],
                uint8x16x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4q_u8(
                &mut dst[11 * dst_stride],
                uint8x16x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4q_u8(
                &mut dst[10 * dst_stride],
                uint8x16x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4q_u8(
                &mut dst[9 * dst_stride],
                uint8x16x4_t(r1.2, g1.2, b1.2, a1.2),
            );
            vst4q_u8(
                &mut dst[8 * dst_stride],
                uint8x16x4_t(r1.3, g1.3, b1.3, a1.3),
            );
            vst4q_u8(
                &mut dst[7 * dst_stride],
                uint8x16x4_t(r2.0, g2.0, b2.0, a2.0),
            );
            vst4q_u8(
                &mut dst[6 * dst_stride],
                uint8x16x4_t(r2.1, g2.1, b2.1, a2.1),
            );
            vst4q_u8(
                &mut dst[5 * dst_stride],
                uint8x16x4_t(r2.2, g2.2, b2.2, a2.2),
            );
            vst4q_u8(
                &mut dst[4 * dst_stride],
                uint8x16x4_t(r2.3, g2.3, b2.3, a2.3),
            );
            vst4q_u8(
                &mut dst[3 * dst_stride],
                uint8x16x4_t(r3.0, g3.0, b3.0, a3.0),
            );
            vst4q_u8(
                &mut dst[2 * dst_stride],
                uint8x16x4_t(r3.1, g3.1, b3.1, a3.1),
            );
            vst4q_u8(&mut dst[dst_stride], uint8x16x4_t(r3.2, g3.2, b3.2, a3.2));
            vst4q_u8(&mut dst[0], uint8x16x4_t(r3.3, g3.3, b3.3, a3.3));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_transpose_16x16() {
        // Define a 16x16 source matrix
        let mut src: Vec<u8> = vec![0u8; 256];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as u8;
        }

        // Expected output: transpose of the 16x16 matrix
        let mut expected = vec![0u8; 256];
        for (y, chunk) in expected.chunks_exact_mut(16).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 16 + y) as u8;
            }
        }

        // Create the destination matrix
        let mut dst = vec![0u8; 256];

        // Call the function
        neon_transpose_16x16::<true, false>(
            &src, 16, // src_stride
            &mut dst, 16, // dst_stride
        );

        // Compare the result with the expected matrix
        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }
}
