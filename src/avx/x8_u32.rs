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
unsafe fn _mm256_swaphi(v: __m256i) -> __m256i {
    _mm256_permute2x128_si256::<0x01>(v, v)
}

#[inline(always)]
#[allow(clippy::type_complexity)]
unsafe fn avx_transpose_8x8_impl<const FLIP: bool>(
    v0: (__m256i, __m256i, __m256i, __m256i),
    v1: (__m256i, __m256i, __m256i, __m256i),
) -> (
    (__m256i, __m256i, __m256i, __m256i),
    (__m256i, __m256i, __m256i, __m256i),
) {
    // let t0 = _mm256_unpacklo_epi32(_mm256_castsi256_ps(v0.0), _mm256_castsi256_ps(v0.1));
    // let t1 = _mm256_unpackhi_epi32(_mm256_castsi256_ps(v0.0), _mm256_castsi256_ps(v0.1));
    // let t2 = _mm256_unpacklo_epi32(_mm256_castsi256_ps(v0.2), _mm256_castsi256_ps(v0.3));
    // let t3 = _mm256_unpackhi_epi32(_mm256_castsi256_ps(v0.2), _mm256_castsi256_ps(v0.3));
    // let t4 = _mm256_unpacklo_epi32(_mm256_castsi256_ps(v1.0), _mm256_castsi256_ps(v1.1));
    // let t5 = _mm256_unpackhi_epi32(_mm256_castsi256_ps(v1.0), _mm256_castsi256_ps(v1.1));
    // let t6 = _mm256_unpacklo_epi32(_mm256_castsi256_ps(v1.2), _mm256_castsi256_ps(v1.3));
    // let t7 = _mm256_unpackhi_epi32(_mm256_castsi256_ps(v1.2), _mm256_castsi256_ps(v1.3));
    // const FLAG_1: i32 = _mm_shuffle(1, 0, 1, 0);
    // let tt0 = _mm256_shuffle_ps::<FLAG_1>(t0, t2);
    // const FLAG_2: i32 = _mm_shuffle(3, 2, 3, 2);
    // let tt1 = _mm256_shuffle_ps::<FLAG_2>(t0, t2);
    // const FLAG_3: i32 = _mm_shuffle(1, 0, 1, 0);
    // let tt2 = _mm256_shuffle_ps::<FLAG_3>(t1, t3);
    // const FLAG_4: i32 = _mm_shuffle(3, 2, 3, 2);
    // let tt3 = _mm256_shuffle_ps::<FLAG_4>(t1, t3);
    // const FLAG_5: i32 = _mm_shuffle(1, 0, 1, 0);
    // let tt4 = _mm256_shuffle_ps::<FLAG_5>(t4, t6);
    // const FLAG_6: i32 = _mm_shuffle(3, 2, 3, 2);
    // let tt5 = _mm256_shuffle_ps::<FLAG_6>(t4, t6);
    // const FLAG_7: i32 = _mm_shuffle(1, 0, 1, 0);
    // let tt6 = _mm256_shuffle_ps::<FLAG_7>(t5, t7);
    // const FLAG_8: i32 = _mm_shuffle(3, 2, 3, 2);
    // let tt7 = _mm256_shuffle_ps::<FLAG_8>(t5, t7);
    // let r0 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x20>(tt0, tt4));
    // let r1 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x20>(tt1, tt5));
    // let r2 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x20>(tt2, tt6));
    // let r3 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x20>(tt3, tt7));
    // let r4 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x31>(tt0, tt4));
    // let r5 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x31>(tt1, tt5));
    // let r6 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x31>(tt2, tt6));
    // let r7 = _mm256_castps_si256(_mm256_permute2f128_ps::<0x31>(tt3, tt7));

    let t0 = _mm256_unpacklo_epi32(v0.0, v0.1);
    let t1 = _mm256_unpackhi_epi32(v0.0, v0.1);
    let t2 = _mm256_unpacklo_epi32(v0.2, v0.3);
    let t3 = _mm256_unpackhi_epi32(v0.2, v0.3);
    let t4 = _mm256_unpacklo_epi32(v1.0, v1.1);
    let t5 = _mm256_unpackhi_epi32(v1.0, v1.1);
    let t6 = _mm256_unpacklo_epi32(v1.2, v1.3);
    let t7 = _mm256_unpackhi_epi32(v1.2, v1.3);
    let tt0 = _mm256_unpacklo_epi64(t0, t2);
    let tt1 = _mm256_unpackhi_epi64(t0, t2);
    let tt2 = _mm256_unpacklo_epi64(t1, t3);
    let tt3 = _mm256_unpackhi_epi64(t1, t3);
    let tt4 = _mm256_unpacklo_epi64(t4, t6);
    let tt5 = _mm256_unpackhi_epi64(t4, t6);
    let tt6 = _mm256_unpacklo_epi64(t5, t7);
    let tt7 = _mm256_unpackhi_epi64(t5, t7);
    let r0 = _mm256_permute2x128_si256::<0x20>(tt0, tt4);
    let r1 = _mm256_permute2x128_si256::<0x20>(tt1, tt5);
    let r2 = _mm256_permute2x128_si256::<0x20>(tt2, tt6);
    let r3 = _mm256_permute2x128_si256::<0x20>(tt3, tt7);
    let r4 = _mm256_permute2x128_si256::<0x31>(tt0, tt4);
    let r5 = _mm256_permute2x128_si256::<0x31>(tt1, tt5);
    let r6 = _mm256_permute2x128_si256::<0x31>(tt2, tt6);
    let r7 = _mm256_permute2x128_si256::<0x31>(tt3, tt7);

    if FLIP {
        let rsh = _mm256_setr_epi8(
            12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11, 4,
            5, 6, 7, 0, 1, 2, 3,
        );
        (
            (
                _mm256_swaphi(_mm256_shuffle_epi8(r0, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r1, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r2, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r3, rsh)),
            ),
            (
                _mm256_swaphi(_mm256_shuffle_epi8(r4, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r5, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r6, rsh)),
                _mm256_swaphi(_mm256_shuffle_epi8(r7, rsh)),
            ),
        )
    } else {
        ((r0, r1, r2, r3), (r4, r5, r6, r7))
    }
}

#[inline(always)]
pub(crate) fn avx_transpose_8x8_u32<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm256_loadu_si256(src.get_unchecked(0..).as_ptr() as *const _);
        let row1 = _mm256_loadu_si256(src.get_unchecked(src_stride..).as_ptr() as *const _);
        let row2 = _mm256_loadu_si256(src.get_unchecked(2 * src_stride..).as_ptr() as *const _);
        let row3 = _mm256_loadu_si256(src.get_unchecked(3 * src_stride..).as_ptr() as *const _);
        let row4 = _mm256_loadu_si256(src.get_unchecked(4 * src_stride..).as_ptr() as *const _);
        let row5 = _mm256_loadu_si256(src.get_unchecked(5 * src_stride..).as_ptr() as *const _);
        let row6 = _mm256_loadu_si256(src.get_unchecked(6 * src_stride..).as_ptr() as *const _);
        let row7 = _mm256_loadu_si256(src.get_unchecked(7 * src_stride..).as_ptr() as *const _);

        let (v0, v1) =
            avx_transpose_8x8_impl::<FLIP>((row0, row1, row2, row3), (row4, row5, row6, row7));

        if FLOP {
            _mm256_storeu_si256(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.0);
            _mm256_storeu_si256(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v0.3,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                v1.0,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                v1.1,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                v1.2,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                v1.3,
            );
        } else {
            _mm256_storeu_si256(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                v0.0,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                v0.3,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v1.0,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v1.1,
            );
            _mm256_storeu_si256(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v1.2,
            );
            _mm256_storeu_si256(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v1.3);
        }
    }
}
