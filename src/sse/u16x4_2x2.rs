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

use crate::sse::_mm_shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sse_transpose_u64_2x2_impl<const FLIP: bool>(
    v0: (__m128i, __m128i),
) -> (__m128i, __m128i) {
    let l = _mm_unpacklo_epi64(v0.0, v0.1);
    let h = _mm_unpackhi_epi64(v0.0, v0.1);

    if FLIP {
        const S: i32 = _mm_shuffle(1, 0, 3, 2);
        (_mm_shuffle_epi32::<S>(l), _mm_shuffle_epi32::<S>(h))
    } else {
        (l, h)
    }
}

#[inline]
pub(crate) fn ssse_transpose_u16x4_2x2<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm_loadu_si128(src.get_unchecked(0..).as_ptr() as *const _);
        let row1 = _mm_loadu_si128(src.get_unchecked(src_stride..).as_ptr() as *const _);

        let v0 = sse_transpose_u64_2x2_impl::<FLIP>((row0, row1));

        if FLOP {
            _mm_storeu_si128(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.0);
            _mm_storeu_si128(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
        } else {
            _mm_storeu_si128(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.0,
            );
            _mm_storeu_si128(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.1);
        }
    }
}
