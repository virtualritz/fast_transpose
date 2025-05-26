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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sse_transpose_4x4_impl_f32<const FLIP: bool>(
    v0: (__m128, __m128, __m128, __m128),
) -> (__m128, __m128, __m128, __m128) {
    // Unpack 32 bit elements. Goes from:
    // in[0]: 00 01 02 03
    // in[1]: 10 11 12 13
    // in[2]: 20 21 22 23
    // in[3]: 30 31 32 33
    // to:
    // a0:    00 10 01 11
    // a1:    20 30 21 31
    // a2:    02 12 03 13
    // a3:    22 32 23 33

    let a0 = _mm_unpacklo_ps(v0.0, v0.1);
    let a1 = _mm_unpacklo_ps(v0.2, v0.3);
    let a2 = _mm_unpackhi_ps(v0.0, v0.1);
    let a3 = _mm_unpackhi_ps(v0.2, v0.3);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    let r0 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(a0), _mm_castps_pd(a1)));
    let r1 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(a0), _mm_castps_pd(a1)));
    let r2 = _mm_castpd_ps(_mm_unpacklo_pd(_mm_castps_pd(a2), _mm_castps_pd(a3)));
    let r3 = _mm_castpd_ps(_mm_unpackhi_pd(_mm_castps_pd(a2), _mm_castps_pd(a3)));

    if FLIP {
        let rsh = _mm_setr_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
        (
            _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r0), rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r1), rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r2), rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(r3), rsh)),
        )
    } else {
        (r0, r1, r2, r3)
    }
}

#[inline]
#[target_feature(enable = "ssse3")]
pub(crate) unsafe fn sse_transpose_4x4_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm_loadu_ps(src.get_unchecked(0..).as_ptr() as *const _);
        let row1 = _mm_loadu_ps(src.get_unchecked(src_stride..).as_ptr() as *const _);
        let row2 = _mm_loadu_ps(src.get_unchecked(2 * src_stride..).as_ptr() as *const _);
        let row3 = _mm_loadu_ps(src.get_unchecked(3 * src_stride..).as_ptr() as *const _);

        let v0 = sse_transpose_4x4_impl_f32::<FLIP>((row0, row1, row2, row3));

        if FLOP {
            _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.0);
            _mm_storeu_ps(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v0.3,
            );
        } else {
            _mm_storeu_ps(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v0.0,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm_storeu_ps(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.3);
        }
    }
}
