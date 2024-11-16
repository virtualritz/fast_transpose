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

#[inline]
pub(crate) fn neon_transpose_8x8<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1_u8(&src[0]);
        let row1 = vld1_u8(&src[src_stride]);
        let row2 = vld1_u8(&src[2 * src_stride]);
        let row3 = vld1_u8(&src[3 * src_stride]);
        let row4 = vld1_u8(&src[4 * src_stride]);
        let row5 = vld1_u8(&src[5 * src_stride]);
        let row6 = vld1_u8(&src[6 * src_stride]);
        let row7 = vld1_u8(&src[7 * src_stride]);

        let (v0, v1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0, row1, row2, row3),
            uint8x8x4_t(row4, row5, row6, row7),
        );

        if FLOP {
            vst1_u8(&mut dst[7 * dst_stride], v0.0);
            vst1_u8(&mut dst[6 * dst_stride], v0.1);
            vst1_u8(&mut dst[5 * dst_stride], v0.2);
            vst1_u8(&mut dst[4 * dst_stride], v0.3);
            vst1_u8(&mut dst[3 * dst_stride], v1.0);
            vst1_u8(&mut dst[2 * dst_stride], v1.1);
            vst1_u8(&mut dst[dst_stride], v1.2);
            vst1_u8(&mut dst[0], v1.3);
        } else {
            vst1_u8(&mut dst[0], v0.0);
            vst1_u8(&mut dst[dst_stride], v0.1);
            vst1_u8(&mut dst[2 * dst_stride], v0.2);
            vst1_u8(&mut dst[3 * dst_stride], v0.3);
            vst1_u8(&mut dst[4 * dst_stride], v1.0);
            vst1_u8(&mut dst[5 * dst_stride], v1.1);
            vst1_u8(&mut dst[6 * dst_stride], v1.2);
            vst1_u8(&mut dst[7 * dst_stride], v1.3);
        }
    }
}

pub(crate) fn neon_transpose_8x8_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld3_u8(&src[0]);
        let row1 = vld3_u8(&src[src_stride]);
        let row2 = vld3_u8(&src[2 * src_stride]);
        let row3 = vld3_u8(&src[3 * src_stride]);
        let row4 = vld3_u8(&src[4 * src_stride]);
        let row5 = vld3_u8(&src[5 * src_stride]);
        let row6 = vld3_u8(&src[6 * src_stride]);
        let row7 = vld3_u8(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint8x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint8x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        let (b0, b1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.2, row1.2, row2.2, row3.2),
            uint8x8x4_t(row4.2, row5.2, row6.2, row7.2),
        );

        if FLOP {
            vst3_u8(&mut dst[0], uint8x8x3_t(r0.0, g0.0, b0.0));
            vst3_u8(&mut dst[dst_stride], uint8x8x3_t(r0.1, g0.1, b0.1));
            vst3_u8(&mut dst[2 * dst_stride], uint8x8x3_t(r0.2, g0.2, b0.2));
            vst3_u8(&mut dst[3 * dst_stride], uint8x8x3_t(r0.3, g0.3, b0.3));
            vst3_u8(&mut dst[4 * dst_stride], uint8x8x3_t(r1.0, g1.0, b1.0));
            vst3_u8(&mut dst[5 * dst_stride], uint8x8x3_t(r1.1, g1.1, b1.1));
            vst3_u8(&mut dst[6 * dst_stride], uint8x8x3_t(r1.2, g1.2, b1.2));
            vst3_u8(&mut dst[7 * dst_stride], uint8x8x3_t(r1.3, g1.3, b1.3));
        } else {
            vst3_u8(&mut dst[7 * dst_stride], uint8x8x3_t(r0.0, g0.0, b0.0));
            vst3_u8(&mut dst[6 * dst_stride], uint8x8x3_t(r0.1, g0.1, b0.1));
            vst3_u8(&mut dst[5 * dst_stride], uint8x8x3_t(r0.2, g0.2, b0.2));
            vst3_u8(&mut dst[4 * dst_stride], uint8x8x3_t(r0.3, g0.3, b0.3));
            vst3_u8(&mut dst[3 * dst_stride], uint8x8x3_t(r1.0, g1.0, b1.0));
            vst3_u8(&mut dst[2 * dst_stride], uint8x8x3_t(r1.1, g1.1, b1.1));
            vst3_u8(&mut dst[dst_stride], uint8x8x3_t(r1.2, g1.2, b1.2));
            vst3_u8(&mut dst[0], uint8x8x3_t(r1.3, g1.3, b1.3));
        }
    }
}

pub(crate) fn neon_transpose_8x8_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld2_u8(&src[0]);
        let row1 = vld2_u8(&src[src_stride]);
        let row2 = vld2_u8(&src[2 * src_stride]);
        let row3 = vld2_u8(&src[3 * src_stride]);
        let row4 = vld2_u8(&src[4 * src_stride]);
        let row5 = vld2_u8(&src[5 * src_stride]);
        let row6 = vld2_u8(&src[6 * src_stride]);
        let row7 = vld2_u8(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint8x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint8x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        if FLOP {
            vst2_u8(&mut dst[0], uint8x8x2_t(r0.0, g0.0));
            vst2_u8(&mut dst[dst_stride], uint8x8x2_t(r0.1, g0.1));
            vst2_u8(&mut dst[2 * dst_stride], uint8x8x2_t(r0.2, g0.2));
            vst2_u8(&mut dst[3 * dst_stride], uint8x8x2_t(r0.3, g0.3));
            vst2_u8(&mut dst[4 * dst_stride], uint8x8x2_t(r1.0, g1.0));
            vst2_u8(&mut dst[5 * dst_stride], uint8x8x2_t(r1.1, g1.1));
            vst2_u8(&mut dst[6 * dst_stride], uint8x8x2_t(r1.2, g1.2));
            vst2_u8(&mut dst[7 * dst_stride], uint8x8x2_t(r1.3, g1.3));
        } else {
            vst2_u8(&mut dst[7 * dst_stride], uint8x8x2_t(r0.0, g0.0));
            vst2_u8(&mut dst[6 * dst_stride], uint8x8x2_t(r0.1, g0.1));
            vst2_u8(&mut dst[5 * dst_stride], uint8x8x2_t(r0.2, g0.2));
            vst2_u8(&mut dst[4 * dst_stride], uint8x8x2_t(r0.3, g0.3));
            vst2_u8(&mut dst[3 * dst_stride], uint8x8x2_t(r1.0, g1.0));
            vst2_u8(&mut dst[2 * dst_stride], uint8x8x2_t(r1.1, g1.1));
            vst2_u8(&mut dst[dst_stride], uint8x8x2_t(r1.2, g1.2));
            vst2_u8(&mut dst[0], uint8x8x2_t(r1.3, g1.3));
        }
    }
}

pub(crate) fn neon_transpose_8x8_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld4_u8(&src[0]);
        let row1 = vld4_u8(&src[src_stride]);
        let row2 = vld4_u8(&src[2 * src_stride]);
        let row3 = vld4_u8(&src[3 * src_stride]);
        let row4 = vld4_u8(&src[4 * src_stride]);
        let row5 = vld4_u8(&src[5 * src_stride]);
        let row6 = vld4_u8(&src[6 * src_stride]);
        let row7 = vld4_u8(&src[7 * src_stride]);

        let (r0, r1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.0, row1.0, row2.0, row3.0),
            uint8x8x4_t(row4.0, row5.0, row6.0, row7.0),
        );

        let (g0, g1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.1, row1.1, row2.1, row3.1),
            uint8x8x4_t(row4.1, row5.1, row6.1, row7.1),
        );

        let (b0, b1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.2, row1.2, row2.2, row3.2),
            uint8x8x4_t(row4.2, row5.2, row6.2, row7.2),
        );

        let (a0, a1) = neon_transpose_8x8_impl::<FLIP>(
            uint8x8x4_t(row0.3, row1.3, row2.3, row3.3),
            uint8x8x4_t(row4.3, row5.3, row6.3, row7.3),
        );

        if FLOP {
            vst4_u8(
                &mut dst[0],
                uint8x8x4_t(r0.0, g0.0, b0.0, a0.0),
            );
            vst4_u8(
                &mut dst[dst_stride],
                uint8x8x4_t(r0.1, g0.1, b0.1, a0.1),
            );
            vst4_u8(
                &mut dst[2 * dst_stride],
                uint8x8x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4_u8(
                &mut dst[3 * dst_stride],
                uint8x8x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4_u8(
                &mut dst[4 * dst_stride],
                uint8x8x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4_u8(
                &mut dst[5 * dst_stride],
                uint8x8x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4_u8(
                &mut dst[6 * dst_stride],
                uint8x8x4_t(r1.2, g1.2, b1.2, a1.2),
            );
            vst4_u8(
                &mut dst[7 * dst_stride],
                uint8x8x4_t(r1.3, g1.3, b1.3, a1.3),
            );
        } else {
            vst4_u8(
                &mut dst[7 * dst_stride],
                uint8x8x4_t(r0.0, g0.0, b0.0, a0.0),
            );
            vst4_u8(
                &mut dst[6 * dst_stride],
                uint8x8x4_t(r0.1, g0.1, b0.1, a0.1),
            );
            vst4_u8(
                &mut dst[5 * dst_stride],
                uint8x8x4_t(r0.2, g0.2, b0.2, a0.2),
            );
            vst4_u8(
                &mut dst[4 * dst_stride],
                uint8x8x4_t(r0.3, g0.3, b0.3, a0.3),
            );
            vst4_u8(
                &mut dst[3 * dst_stride],
                uint8x8x4_t(r1.0, g1.0, b1.0, a1.0),
            );
            vst4_u8(
                &mut dst[2 * dst_stride],
                uint8x8x4_t(r1.1, g1.1, b1.1, a1.1),
            );
            vst4_u8(
                &mut dst[dst_stride],
                uint8x8x4_t(r1.2, g1.2, b1.2, a1.2),
            );
            vst4_u8(
                &mut dst[0],
                uint8x8x4_t(r1.3, g1.3, b1.3, a1.3),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_transpose_8x8() {
        let mut src: Vec<u8> = vec![0u8; 64];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as u8;
        }

        let mut expected = vec![0u8; 64];
        for (y, chunk) in expected.chunks_exact_mut(8).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 8 + y) as u8;
            }
        }

        let mut dst = vec![0u8; 64];

        neon_transpose_8x8::<false, false>(
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
        let mut src: Vec<u8> = vec![0u8; 64];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as u8;
        }

        // Expected output: transpose of the 8x8 matrix
        let mut expected = vec![0u8; 64];
        for (y, chunk) in expected.chunks_exact_mut(8).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 8 + (7 - y)) as u8;
            }
        }

        let mut dst = vec![0u8; 64];

        // Call the function
        neon_transpose_8x8::<true, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }
}
