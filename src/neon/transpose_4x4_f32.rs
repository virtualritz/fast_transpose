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
use crate::neon::utils::{vrev128_u32, vtrnq_s64_to_u32};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_transpose_4x4_impl<const FLIP: bool>(v0: float32x4x4_t) -> float32x4x4_t {
    // Swap 32 bit elements. Goes from:
    // a0: 00 01 02 03
    // a1: 10 11 12 13
    // a2: 20 21 22 23
    // a3: 30 31 32 33
    // to:
    // b0.0: 00 10 02 12
    // b0.1: 01 11 03 13
    // b1.0: 20 30 22 32
    // b1.1: 21 31 23 33

    let b0 = vtrnq_u32(vreinterpretq_u32_f32(v0.0), vreinterpretq_u32_f32(v0.1));
    let b1 = vtrnq_u32(vreinterpretq_u32_f32(v0.2), vreinterpretq_u32_f32(v0.3));

    // Swap 64 bit elements resulting in:
    // c0.0: 00 10 20 30
    // c0.1: 02 12 22 32
    // c1.0: 01 11 21 31
    // c1.1: 03 13 23 33

    let c0 = vtrnq_s64_to_u32(b0.0, b1.0);
    let c1 = vtrnq_s64_to_u32(b0.1, b1.1);

    if FLIP {
        float32x4x4_t(
            vreinterpretq_f32_u32(vrev128_u32(c0.0)),
            vreinterpretq_f32_u32(vrev128_u32(c1.0)),
            vreinterpretq_f32_u32(vrev128_u32(c0.1)),
            vreinterpretq_f32_u32(vrev128_u32(c1.1)),
        )
    } else {
        float32x4x4_t(
            vreinterpretq_f32_u32(c0.0),
            vreinterpretq_f32_u32(c1.0),
            vreinterpretq_f32_u32(c0.1),
            vreinterpretq_f32_u32(c1.1),
        )
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_f32(src.get_unchecked(0..).as_ptr());
        let row1 = vld1q_f32(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld1q_f32(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld1q_f32(src.get_unchecked(3 * src_stride..).as_ptr());

        let v0 = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0, row1, row2, row3));

        if FLOP {
            vst1q_f32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.0);
            vst1q_f32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.1);
            vst1q_f32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.2);
            vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.3);
        } else {
            vst1q_f32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            vst1q_f32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            vst1q_f32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            vst1q_f32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld2q_f32(src.get_unchecked(0..).as_ptr());
        let row1 = vld2q_f32(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld2q_f32(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld2q_f32(src.get_unchecked(3 * src_stride..).as_ptr());

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));

        if FLOP {
            vst2q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.0, g.0),
            );
            vst2q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.1, g.1),
            );
            vst2q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.2, g.2),
            );
            vst2q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x2_t(r.3, g.3),
            );
        } else {
            vst2q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x2_t(r.0, g.0),
            );
            vst2q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.1, g.1),
            );
            vst2q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.2, g.2),
            );
            vst2q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(r.3, g.3),
            );
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld3q_f32(src.get_unchecked(0..).as_ptr());
        let row1 = vld3q_f32(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld3q_f32(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld3q_f32(src.get_unchecked(3 * src_stride..).as_ptr());

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));
        let b = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.2, row1.2, row2.2, row3.2));

        if FLOP {
            vst3q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.0, g.0, b.0),
            );
            vst3q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.1, g.1, b.1),
            );
            vst3q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.2, g.2, b.2),
            );
            vst3q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x3_t(r.3, g.3, b.3),
            );
        } else {
            vst3q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x3_t(r.0, g.0, b.0),
            );
            vst3q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.1, g.1, b.1),
            );
            vst3q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.2, g.2, b.2),
            );
            vst3q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x3_t(r.3, g.3, b.3),
            );
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld4q_f32(src.get_unchecked(0..).as_ptr());
        let row1 = vld4q_f32(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld4q_f32(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld4q_f32(src.get_unchecked(3 * src_stride..).as_ptr());

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));
        let b = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.2, row1.2, row2.2, row3.2));
        let a = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.3, row1.3, row2.3, row3.3));

        if FLOP {
            vst4q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.0, g.0, b.0, a.0),
            );
            vst4q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.1, g.1, b.1, a.1),
            );
            vst4q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.2, g.2, b.2, a.2),
            );
            vst4q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x4_t(r.3, g.3, b.3, a.3),
            );
        } else {
            vst4q_f32(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x4_t(r.0, g.0, b.0, a.0),
            );
            vst4q_f32(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.1, g.1, b.1, a.1),
            );
            vst4q_f32(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.2, g.2, b.2, a.2),
            );
            vst4q_f32(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x4_t(r.3, g.3, b.3, a.3),
            );
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_8x8_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let a0 = vld1q_f32_x2(src.get_unchecked(0..).as_ptr());
        let a1 = vld1q_f32_x2(src.get_unchecked(src_stride..).as_ptr());
        let a2 = vld1q_f32_x2(src.get_unchecked(2 * src_stride..).as_ptr());
        let a3 = vld1q_f32_x2(src.get_unchecked(3 * src_stride..).as_ptr());

        let c0 = vld1q_f32_x2(src.get_unchecked(4 * src_stride..).as_ptr());
        let c1 = vld1q_f32_x2(src.get_unchecked(5 * src_stride..).as_ptr());
        let c2 = vld1q_f32_x2(src.get_unchecked(6 * src_stride..).as_ptr());
        let c3 = vld1q_f32_x2(src.get_unchecked(7 * src_stride..).as_ptr());

        let a = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(a0.0, a1.0, a2.0, a3.0));
        let b = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(a0.1, a1.1, a2.1, a3.1));
        let c = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(c0.0, c1.0, c2.0, c3.0));
        let d = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(c0.1, c1.1, c2.1, c3.1));

        if FLOP {
            vst1q_f32_x2(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.0, c.0),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.1, c.1),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.2, c.2),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.3, c.3),
            );

            vst1q_f32_x2(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.0, d.0),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.1, d.1),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.2, d.2),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x2_t(b.3, d.3),
            );
        } else {
            vst1q_f32_x2(
                dst.get_unchecked_mut(0..).as_mut_ptr(),
                float32x4x2_t(a.0, c.0),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.1, c.1),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.2, c.2),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(a.3, c.3),
            );

            vst1q_f32_x2(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.0, d.0),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.1, d.1),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.2, d.2),
            );
            vst1q_f32_x2(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(),
                float32x4x2_t(b.3, d.3),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_transpose_4x4_f32() {
        let mut src: Vec<f32> = vec![0f32; 16];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as f32;
        }

        // Expected output: transpose of the 16x16 matrix
        let mut expected = vec![0f32; 16];
        for (y, chunk) in expected.chunks_exact_mut(4).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 4 + y) as f32;
            }
        }

        // Create the destination matrix
        let mut dst = vec![0f32; 16];

        // Call the function
        neon_transpose_4x4_f32::<false, false>(
            &src, 4, // src_stride
            &mut dst, 4, // dst_stride
        );

        println!("Expected");
        for lane in expected.chunks_exact(4) {
            println!("{:?}", lane);
        }

        println!("Received");
        for lane in dst.chunks_exact(4) {
            println!("{:?}", lane);
        }

        // Compare the result with the expected matrix
        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }

    #[test]
    fn test_neon_transpose_16x16_f32() {
        // Define a 16x16 source matrix
        let mut src: Vec<f32> = vec![0f32; 64];
        for c in src.iter_mut().enumerate() {
            *c.1 = c.0 as f32;
        }

        // Expected output: transpose of the 16x16 matrix
        let mut expected = vec![0f32; 64];
        for (y, chunk) in expected.chunks_exact_mut(8).enumerate() {
            for (x, dst) in chunk.iter_mut().enumerate() {
                *dst = (x * 8 + y) as f32;
            }
        }

        // Create the destination matrix
        let mut dst = vec![0f32; 64];

        // Call the function
        neon_transpose_8x8_f32::<false, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        println!("Expected");
        for lane in expected.chunks_exact(8) {
            println!("{:?}", lane);
        }

        println!("Received");
        for lane in dst.chunks_exact(8) {
            println!("{:?}", lane);
        }

        // Compare the result with the expected matrix
        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }
}
