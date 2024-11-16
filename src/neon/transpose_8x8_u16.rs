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
use crate::neon::utils::{vrev128_u16, vtrnq_u64_to_u16};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_transpose_8x8_impl<const FLIP: bool>(
    v0: uint16x8x4_t,
    v1: uint16x8x4_t,
) -> (uint16x8x4_t, uint16x8x4_t) {
    // Swap 16 bit elements. Goes from:
    // a0: 00 01 02 03 04 05 06 07
    // a1: 10 11 12 13 14 15 16 17
    // a2: 20 21 22 23 24 25 26 27
    // a3: 30 31 32 33 34 35 36 37
    // a4: 40 41 42 43 44 45 46 47
    // a5: 50 51 52 53 54 55 56 57
    // a6: 60 61 62 63 64 65 66 67
    // a7: 70 71 72 73 74 75 76 77
    // to:
    // b0.0: 00 10 02 12 04 14 06 16
    // b0.1: 01 11 03 13 05 15 07 17
    // b1.0: 20 30 22 32 24 34 26 36
    // b1.1: 21 31 23 33 25 35 27 37
    // b2.0: 40 50 42 52 44 54 46 56
    // b2.1: 41 51 43 53 45 55 47 57
    // b3.0: 60 70 62 72 64 74 66 76
    // b3.1: 61 71 63 73 65 75 67 77

    let b0 = vtrnq_u16(v0.0, v0.1);
    let b1 = vtrnq_u16(v0.2, v0.3);
    let b2 = vtrnq_u16(v1.0, v1.1);
    let b3 = vtrnq_u16(v1.2, v1.3);

    // Swap 32 bit elements resulting in:
    // c0.0: 00 10 20 30 04 14 24 34
    // c0.1: 02 12 22 32 06 16 26 36
    // c1.0: 01 11 21 31 05 15 25 35
    // c1.1: 03 13 23 33 07 17 27 37
    // c2.0: 40 50 60 70 44 54 64 74
    // c2.1: 42 52 62 72 46 56 66 76
    // c3.0: 41 51 61 71 45 55 65 75
    // c3.1: 43 53 63 73 47 57 67 77

    let c0 = vtrnq_u32(vreinterpretq_u32_u16(b0.0), vreinterpretq_u32_u16(b1.0));
    let c1 = vtrnq_u32(vreinterpretq_u32_u16(b0.1), vreinterpretq_u32_u16(b1.1));
    let c2 = vtrnq_u32(vreinterpretq_u32_u16(b2.0), vreinterpretq_u32_u16(b3.0));
    let c3 = vtrnq_u32(vreinterpretq_u32_u16(b2.1), vreinterpretq_u32_u16(b3.1));

    // Swap 64 bit elements resulting in:
    // d0.0: 00 10 20 30 40 50 60 70
    // d0.1: 04 14 24 34 44 54 64 74
    // d1.0: 01 11 21 31 41 51 61 71
    // d1.1: 05 15 25 35 45 55 65 75
    // d2.0: 02 12 22 32 42 52 62 72
    // d2.1: 06 16 26 36 46 56 66 76
    // d3.0: 03 13 23 33 43 53 63 73
    // d3.1: 07 17 27 37 47 57 67 77

    let d0 = vtrnq_u64_to_u16(c0.0, c2.0);
    let d1 = vtrnq_u64_to_u16(c1.0, c3.0);
    let d2 = vtrnq_u64_to_u16(c0.1, c2.1);
    let d3 = vtrnq_u64_to_u16(c1.1, c3.1);

    if FLIP {
        (
            uint16x8x4_t(
                vrev128_u16(d0.0),
                vrev128_u16(d1.0),
                vrev128_u16(d2.0),
                vrev128_u16(d3.0),
            ),
            uint16x8x4_t(
                vrev128_u16(d0.1),
                vrev128_u16(d1.1),
                vrev128_u16(d2.0),
                vrev128_u16(d3.1),
            ),
        )
    } else {
        (
            uint16x8x4_t(d0.0, d1.0, d2.0, d3.0),
            uint16x8x4_t(d0.1, d1.1, d2.1, d3.1),
        )
    }
}

#[inline]
pub(crate) fn neon_transpose_8x8_u16<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_u16(&src[0]);
        let row1 = vld1q_u16(&src[src_stride]);
        let row2 = vld1q_u16(&src[2 * src_stride]);
        let row3 = vld1q_u16(&src[3 * src_stride]);
        let row4 = vld1q_u16(&src[4 * src_stride]);
        let row5 = vld1q_u16(&src[5 * src_stride]);
        let row6 = vld1q_u16(&src[6 * src_stride]);
        let row7 = vld1q_u16(&src[7 * src_stride]);

        let (v0, v1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0, row1, row2, row3),
            uint16x8x4_t(row4, row5, row6, row7),
        );

        if FLOP {
            vst1q_u16(&mut dst[7 * dst_stride], v0.0);
            vst1q_u16(&mut dst[6 * dst_stride], v0.1);
            vst1q_u16(&mut dst[5 * dst_stride], v0.2);
            vst1q_u16(&mut dst[4 * dst_stride], v0.3);
            vst1q_u16(&mut dst[3 * dst_stride], v1.0);
            vst1q_u16(&mut dst[2 * dst_stride], v1.1);
            vst1q_u16(&mut dst[dst_stride], v1.2);
            vst1q_u16(&mut dst[0], v1.3);
        } else {
            vst1q_u16(&mut dst[0], v0.0);
            vst1q_u16(&mut dst[dst_stride], v0.1);
            vst1q_u16(&mut dst[2 * dst_stride], v0.2);
            vst1q_u16(&mut dst[3 * dst_stride], v0.3);
            vst1q_u16(&mut dst[4 * dst_stride], v1.0);
            vst1q_u16(&mut dst[5 * dst_stride], v1.1);
            vst1q_u16(&mut dst[6 * dst_stride], v1.2);
            vst1q_u16(&mut dst[7 * dst_stride], v1.3);
        }
    }
}

pub(crate) fn neon_transpose_8x8_u16_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld2q_u16(&src[0]);
        let row1 = vld2q_u16(&src[src_stride]);
        let row2 = vld2q_u16(&src[2 * src_stride]);
        let row3 = vld2q_u16(&src[3 * src_stride]);
        let row4 = vld2q_u16(&src[4 * src_stride]);
        let row5 = vld2q_u16(&src[5 * src_stride]);
        let row6 = vld2q_u16(&src[6 * src_stride]);
        let row7 = vld2q_u16(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint16x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint16x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        if FLOP {
            vst2q_u16(&mut dst[0], uint16x8x2_t(r0.0, g0.0));
            vst2q_u16(&mut dst[dst_stride], uint16x8x2_t(r0.1, g0.1));
            vst2q_u16(&mut dst[2 * dst_stride], uint16x8x2_t(r0.2, g0.2));
            vst2q_u16(&mut dst[3 * dst_stride], uint16x8x2_t(r0.3, g0.3));
            vst2q_u16(&mut dst[4 * dst_stride], uint16x8x2_t(r1.0, g1.0));
            vst2q_u16(&mut dst[5 * dst_stride], uint16x8x2_t(r1.1, g1.1));
            vst2q_u16(&mut dst[6 * dst_stride], uint16x8x2_t(r1.2, g1.2));
            vst2q_u16(&mut dst[7 * dst_stride], uint16x8x2_t(r1.3, g1.3));
        } else {
            vst2q_u16(&mut dst[7 * dst_stride], uint16x8x2_t(r0.0, g0.0));
            vst2q_u16(&mut dst[6 * dst_stride], uint16x8x2_t(r0.1, g0.1));
            vst2q_u16(&mut dst[5 * dst_stride], uint16x8x2_t(r0.2, g0.2));
            vst2q_u16(&mut dst[4 * dst_stride], uint16x8x2_t(r0.3, g0.3));
            vst2q_u16(&mut dst[3 * dst_stride], uint16x8x2_t(r1.0, g1.0));
            vst2q_u16(&mut dst[2 * dst_stride], uint16x8x2_t(r1.1, g1.1));
            vst2q_u16(&mut dst[dst_stride], uint16x8x2_t(r1.2, g1.2));
            vst2q_u16(&mut dst[0], uint16x8x2_t(r1.3, g1.3));
        }
    }
}

pub(crate) fn neon_transpose_8x8_u16_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld3q_u16(&src[0]);
        let row1 = vld3q_u16(&src[src_stride]);
        let row2 = vld3q_u16(&src[2 * src_stride]);
        let row3 = vld3q_u16(&src[3 * src_stride]);
        let row4 = vld3q_u16(&src[4 * src_stride]);
        let row5 = vld3q_u16(&src[5 * src_stride]);
        let row6 = vld3q_u16(&src[6 * src_stride]);
        let row7 = vld3q_u16(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint16x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint16x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        let (b0, b1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.2, row1.2, row2.2, row3.2),
            uint16x8x4_t(row4.2, row5.2, row6.2, row7.2),
        );

        if FLOP {
            vst3q_u16(&mut dst[0], uint16x8x3_t(r0.0, g0.0, b0.0));
            vst3q_u16(&mut dst[dst_stride], uint16x8x3_t(r0.1, g0.1, b0.1));
            vst3q_u16(&mut dst[2 * dst_stride], uint16x8x3_t(r0.2, g0.2, b0.2));
            vst3q_u16(&mut dst[3 * dst_stride], uint16x8x3_t(r0.3, g0.3, b0.3));
            vst3q_u16(&mut dst[4 * dst_stride], uint16x8x3_t(r1.0, g1.0, b1.0));
            vst3q_u16(&mut dst[5 * dst_stride], uint16x8x3_t(r1.1, g1.1, b1.1));
            vst3q_u16(&mut dst[6 * dst_stride], uint16x8x3_t(r1.2, g1.2, b1.2));
            vst3q_u16(&mut dst[7 * dst_stride], uint16x8x3_t(r1.3, g1.3, b1.3));
        } else {
            vst3q_u16(&mut dst[7 * dst_stride], uint16x8x3_t(r0.0, g0.0, b0.0));
            vst3q_u16(&mut dst[6 * dst_stride], uint16x8x3_t(r0.1, g0.1, b0.1));
            vst3q_u16(&mut dst[5 * dst_stride], uint16x8x3_t(r0.2, g0.2, b0.2));
            vst3q_u16(&mut dst[4 * dst_stride], uint16x8x3_t(r0.3, g0.3, b0.3));
            vst3q_u16(&mut dst[3 * dst_stride], uint16x8x3_t(r1.0, g1.0, b1.0));
            vst3q_u16(&mut dst[2 * dst_stride], uint16x8x3_t(r1.1, g1.1, b1.1));
            vst3q_u16(&mut dst[dst_stride], uint16x8x3_t(r1.2, g1.2, b1.2));
            vst3q_u16(&mut dst[0], uint16x8x3_t(r1.3, g1.3, b1.3));
        }
    }
}

pub(crate) fn neon_transpose_8x8_u16_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld4q_u16(&src[0]);
        let row1 = vld4q_u16(&src[src_stride]);
        let row2 = vld4q_u16(&src[2 * src_stride]);
        let row3 = vld4q_u16(&src[3 * src_stride]);
        let row4 = vld4q_u16(&src[4 * src_stride]);
        let row5 = vld4q_u16(&src[5 * src_stride]);
        let row6 = vld4q_u16(&src[6 * src_stride]);
        let row7 = vld4q_u16(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint16x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint16x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        let (b0, b1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.2, row1.2, row2.2, row3.2),
            uint16x8x4_t(row4.2, row5.2, row6.2, row7.2),
        );

        let (a0, a1) = neon_transpose_8x8_impl::<FLIP>(
            uint16x8x4_t(row0.3, row1.3, row2.3, row3.3),
            uint16x8x4_t(row4.3, row5.3, row6.3, row7.3),
        );

        if FLOP {
            vst4q_u16(&mut dst[0], uint16x8x4_t(r0.0, g0.0, b0.0, a0.0));
            vst4q_u16(&mut dst[dst_stride], uint16x8x4_t(r0.1, g0.1, b0.1, a0.1));
            vst4q_u16(
                &mut dst[2 * dst_stride],
                uint16x8x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4q_u16(
                &mut dst[3 * dst_stride],
                uint16x8x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4q_u16(
                &mut dst[4 * dst_stride],
                uint16x8x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4q_u16(
                &mut dst[5 * dst_stride],
                uint16x8x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4q_u16(
                &mut dst[6 * dst_stride],
                uint16x8x4_t(r1.2, g1.2, b1.2, a1.2),
            );
            vst4q_u16(
                &mut dst[7 * dst_stride],
                uint16x8x4_t(r1.3, g1.3, b1.3, a1.3),
            );
        } else {
            vst4q_u16(
                &mut dst[7 * dst_stride],
                uint16x8x4_t(r0.0, g0.0, b0.0, a0.0),
            );
            vst4q_u16(
                &mut dst[6 * dst_stride],
                uint16x8x4_t(r0.1, g0.1, b0.1, a0.1),
            );
            vst4q_u16(
                &mut dst[5 * dst_stride],
                uint16x8x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4q_u16(
                &mut dst[4 * dst_stride],
                uint16x8x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4q_u16(
                &mut dst[3 * dst_stride],
                uint16x8x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4q_u16(
                &mut dst[2 * dst_stride],
                uint16x8x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4q_u16(&mut dst[dst_stride], uint16x8x4_t(r1.2, g1.2, b1.2, a1.2));
            vst4q_u16(&mut dst[0], uint16x8x4_t(r1.3, g1.3, b1.3, a1.3));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_transpose_8x8() {
        let mut src: Vec<u16> = vec![0u16; 64];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as u16;
        }

        let mut expected = vec![0u16; 64];
        for (y, chunk) in expected.chunks_exact_mut(8).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 8 + y) as u16;
            }
        }

        let mut dst = vec![0u16; 64];

        neon_transpose_8x8_u16::<false, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }

    #[test]
    fn test_neon_transpose_8x8_flip() {
        let mut src: Vec<u16> = vec![0u16; 64];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as u16;
        }

        // Expected output: transpose of the 8x8 matrix
        let mut expected = vec![0u16; 64];
        for (y, chunk) in expected.chunks_exact_mut(8).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 8 + (7 - y)) as u16;
            }
        }

        let mut dst = vec![0u16; 64];

        // Call the function
        neon_transpose_8x8_u16::<true, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }
}
