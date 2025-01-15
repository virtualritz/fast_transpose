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
use crate::neon::utils::{xvld1q_u8_u32, xvst1q_u8_u32};
use crate::neon::x4_u32::neon_transpose_4x4_impl;
use std::arch::aarch64::uint32x4x4_t;

#[inline]
pub(crate) fn neon_transpose_4x4_u8x4x8<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let q0_1 = xvld1q_u8_u32(src.get_unchecked(0..).as_ptr());
        let q0_2 = xvld1q_u8_u32(src.get_unchecked(src_stride..).as_ptr());
        let q0_3 = xvld1q_u8_u32(src.get_unchecked(2 * src_stride..).as_ptr());
        let q0_4 = xvld1q_u8_u32(src.get_unchecked(3 * src_stride..).as_ptr());

        let q1_1 = xvld1q_u8_u32(src.get_unchecked(16..).as_ptr());
        let q1_2 = xvld1q_u8_u32(src.get_unchecked(16 + src_stride..).as_ptr());
        let q1_3 = xvld1q_u8_u32(src.get_unchecked(16 + 2 * src_stride..).as_ptr());
        let q1_4 = xvld1q_u8_u32(src.get_unchecked(16 + 3 * src_stride..).as_ptr());

        let q2_1 = xvld1q_u8_u32(src.get_unchecked(4 * src_stride..).as_ptr());
        let q2_2 = xvld1q_u8_u32(src.get_unchecked(5 * src_stride..).as_ptr());
        let q2_3 = xvld1q_u8_u32(src.get_unchecked(6 * src_stride..).as_ptr());
        let q2_4 = xvld1q_u8_u32(src.get_unchecked(7 * src_stride..).as_ptr());

        let q3_1 = xvld1q_u8_u32(src.get_unchecked(16 + 4 * src_stride..).as_ptr());
        let q3_2 = xvld1q_u8_u32(src.get_unchecked(16 + 5 * src_stride..).as_ptr());
        let q3_3 = xvld1q_u8_u32(src.get_unchecked(16 + 6 * src_stride..).as_ptr());
        let q3_4 = xvld1q_u8_u32(src.get_unchecked(16 + 7 * src_stride..).as_ptr());

        let mut q0 = neon_transpose_4x4_impl::<FLIP>(uint32x4x4_t(q0_1, q0_2, q0_3, q0_4)); // A
        let mut q1 = neon_transpose_4x4_impl::<FLIP>(uint32x4x4_t(q1_1, q1_2, q1_3, q1_4)); // B
        let mut q2 = neon_transpose_4x4_impl::<FLIP>(uint32x4x4_t(q2_1, q2_2, q2_3, q2_4)); // C
        let mut q3 = neon_transpose_4x4_impl::<FLIP>(uint32x4x4_t(q3_1, q3_2, q3_3, q3_4)); // D

        if FLIP {
            std::mem::swap(&mut q0, &mut q2);
            std::mem::swap(&mut q1, &mut q3);
        }

        // Perform an 8 x 8 matrix transpose by building on top of the existing 4 x 4
        // matrix transpose implementation:
        // [ A B ]^T => [ A^T C^T ]
        // [ C D ]      [ B^T D^T ]

        if FLOP {
            xvst1q_u8_u32(dst.get_unchecked_mut(0..).as_mut_ptr(), q0.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), q0.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), q0.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), q0.3);

            xvst1q_u8_u32(dst.get_unchecked_mut(16..).as_mut_ptr(), q2.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(16 + dst_stride..).as_mut_ptr(), q2.1);
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 2 * dst_stride..).as_mut_ptr(),
                q2.2,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 3 * dst_stride..).as_mut_ptr(),
                q2.3,
            );

            xvst1q_u8_u32(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), q1.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), q1.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), q1.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), q1.3);

            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 4 * dst_stride..).as_mut_ptr(),
                q3.0,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 5 * dst_stride..).as_mut_ptr(),
                q3.1,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 6 * dst_stride..).as_mut_ptr(),
                q3.2,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 7 * dst_stride..).as_mut_ptr(),
                q3.3,
            );
        } else {
            xvst1q_u8_u32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), q1.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), q1.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), q1.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(0..).as_mut_ptr(), q1.3);

            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 3 * dst_stride..).as_mut_ptr(),
                q3.0,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 2 * dst_stride..).as_mut_ptr(),
                q3.1,
            );
            xvst1q_u8_u32(dst.get_unchecked_mut(16 + dst_stride..).as_mut_ptr(), q3.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(16..).as_mut_ptr(), q3.3);

            xvst1q_u8_u32(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), q0.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), q0.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), q0.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), q0.3);

            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 7 * dst_stride..).as_mut_ptr(),
                q2.0,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 6 * dst_stride..).as_mut_ptr(),
                q2.1,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 5 * dst_stride..).as_mut_ptr(),
                q2.2,
            );
            xvst1q_u8_u32(
                dst.get_unchecked_mut(16 + 4 * dst_stride..).as_mut_ptr(),
                q2.3,
            );
        }
    }
}
