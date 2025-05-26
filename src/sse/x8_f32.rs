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
use crate::sse::x4_f32::sse_transpose_4x4_impl_f32;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn sse_transpose_8x8_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let q0_1 = _mm_loadu_ps(src.get_unchecked(0..).as_ptr() as *const _);
        let q0_2 = _mm_loadu_ps(src.get_unchecked(src_stride..).as_ptr() as *const _);
        let q0_3 = _mm_loadu_ps(src.get_unchecked(2 * src_stride..).as_ptr() as *const _);
        let q0_4 = _mm_loadu_ps(src.get_unchecked(3 * src_stride..).as_ptr() as *const _);

        let q1_1 = _mm_loadu_ps(src.get_unchecked(4..).as_ptr() as *const _);
        let q1_2 = _mm_loadu_ps(src.get_unchecked(4 + src_stride..).as_ptr() as *const _);
        let q1_3 = _mm_loadu_ps(src.get_unchecked(4 + 2 * src_stride..).as_ptr() as *const _);
        let q1_4 = _mm_loadu_ps(src.get_unchecked(4 + 3 * src_stride..).as_ptr() as *const _);

        let q2_1 = _mm_loadu_ps(src.get_unchecked(4 * src_stride..).as_ptr() as *const _);
        let q2_2 = _mm_loadu_ps(src.get_unchecked(5 * src_stride..).as_ptr() as *const _);
        let q2_3 = _mm_loadu_ps(src.get_unchecked(6 * src_stride..).as_ptr() as *const _);
        let q2_4 = _mm_loadu_ps(src.get_unchecked(7 * src_stride..).as_ptr() as *const _);

        let q3_1 = _mm_loadu_ps(src.get_unchecked(4 + 4 * src_stride..).as_ptr() as *const _);
        let q3_2 = _mm_loadu_ps(src.get_unchecked(4 + 5 * src_stride..).as_ptr() as *const _);
        let q3_3 = _mm_loadu_ps(src.get_unchecked(4 + 6 * src_stride..).as_ptr() as *const _);
        let q3_4 = _mm_loadu_ps(src.get_unchecked(4 + 7 * src_stride..).as_ptr() as *const _);

        let mut q0 = sse_transpose_4x4_impl_f32::<FLIP>((q0_1, q0_2, q0_3, q0_4)); // A
        let mut q1 = sse_transpose_4x4_impl_f32::<FLIP>((q1_1, q1_2, q1_3, q1_4)); // B
        let mut q2 = sse_transpose_4x4_impl_f32::<FLIP>((q2_1, q2_2, q2_3, q2_4)); // C
        let mut q3 = sse_transpose_4x4_impl_f32::<FLIP>((q3_1, q3_2, q3_3, q3_4)); // D

        if FLIP {
            std::mem::swap(&mut q0, &mut q2);
            std::mem::swap(&mut q1, &mut q3);
        }

        // Perform an 8 x 8 matrix transpose by building on top of the existing 4 x 4
        // matrix transpose implementation:
        // [ A B ]^T => [ A^T C^T ]
        // [ C D ]      [ B^T D^T ]

        if FLOP {
            _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, q0.0);
            _mm_storeu_ps(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                q0.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                q0.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                q0.3,
            );

            _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr() as *mut _, q2.0);
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + dst_stride..).as_mut_ptr() as *mut _,
                q2.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 2 * dst_stride..).as_mut_ptr() as *mut _,
                q2.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 3 * dst_stride..).as_mut_ptr() as *mut _,
                q2.3,
            );

            _mm_storeu_ps(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                q1.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                q1.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                q1.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                q1.3,
            );

            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 4 * dst_stride..).as_mut_ptr() as *mut _,
                q3.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 5 * dst_stride..).as_mut_ptr() as *mut _,
                q3.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 6 * dst_stride..).as_mut_ptr() as *mut _,
                q3.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 7 * dst_stride..).as_mut_ptr() as *mut _,
                q3.3,
            );
        } else {
            _mm_storeu_ps(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                q1.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                q1.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                q1.2,
            );
            _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, q1.3);

            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 3 * dst_stride..).as_mut_ptr() as *mut _,
                q3.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 2 * dst_stride..).as_mut_ptr() as *mut _,
                q3.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + dst_stride..).as_mut_ptr() as *mut _,
                q3.2,
            );
            _mm_storeu_ps(dst.get_unchecked_mut(4..).as_mut_ptr() as *mut _, q3.3);

            _mm_storeu_ps(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                q0.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                q0.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                q0.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                q0.3,
            );

            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 7 * dst_stride..).as_mut_ptr() as *mut _,
                q2.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 6 * dst_stride..).as_mut_ptr() as *mut _,
                q2.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 5 * dst_stride..).as_mut_ptr() as *mut _,
                q2.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(4 + 4 * dst_stride..).as_mut_ptr() as *mut _,
                q2.3,
            );
        }
    }
}
