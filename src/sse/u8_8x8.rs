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
#[allow(clippy::type_complexity)]
unsafe fn sse_transpose_8x8_impl<const FLIP: bool>(
    v0: (__m128i, __m128i, __m128i, __m128i),
    v1: (__m128i, __m128i, __m128i, __m128i),
) -> (
    (__m128i, __m128i, __m128i, __m128i),
    (__m128i, __m128i, __m128i, __m128i),
) {
    // Unpack 8 bit elements. Goes from:
    // in[0]: 00 01 02 03 04 05 06 07
    // in[1]: 10 11 12 13 14 15 16 17
    // in[2]: 20 21 22 23 24 25 26 27
    // in[3]: 30 31 32 33 34 35 36 37
    // in[4]: 40 41 42 43 44 45 46 47
    // in[5]: 50 51 52 53 54 55 56 57
    // in[6]: 60 61 62 63 64 65 66 67
    // in[7]: 70 71 72 73 74 75 76 77
    // to:
    // a0:    00 10 01 11 02 12 03 13  04 14 05 15 06 16 07 17
    // a1:    20 30 21 31 22 32 23 33  24 34 25 35 26 36 27 37
    // a2:    40 50 41 51 42 52 43 53  44 54 45 55 46 56 47 57
    // a3:    60 70 61 71 62 72 63 73  64 74 65 75 66 76 67 77
    let a0 = _mm_unpacklo_epi8(v0.0, v0.1);
    let a1 = _mm_unpacklo_epi8(v0.2, v0.3);
    let a2 = _mm_unpacklo_epi8(v1.0, v1.1);
    let a3 = _mm_unpacklo_epi8(v1.2, v1.3);

    // Unpack 16 bit elements resulting in:
    // b0: 00 10 20 30 01 11 21 31  02 12 22 32 03 13 23 33
    // b1: 40 50 60 70 41 51 61 71  42 52 62 72 43 53 63 73
    // b2: 04 14 24 34 05 15 25 35  06 16 26 36 07 17 27 37
    // b3: 44 54 64 74 45 55 65 75  46 56 66 76 47 57 67 77
    let b0 = _mm_unpacklo_epi16(a0, a1);
    let b1 = _mm_unpackhi_epi16(a0, a1);
    let b2 = _mm_unpacklo_epi16(a2, a3);
    let b3 = _mm_unpackhi_epi16(a2, a3);

    // Unpack 32 bit elements resulting in:
    // c0: 00 10 20 30 40 50 60 70  01 11 21 31 41 51 61 71
    // c1: 02 12 22 32 42 52 62 72  03 13 23 33 43 53 63 73
    // c2: 04 14 24 34 44 54 64 74  05 15 25 35 45 55 65 75
    // c3: 06 16 26 36 46 56 66 76  07 17 27 37 47 57 67 77
    let c0 = _mm_unpacklo_epi32(b0, b2);
    let c1 = _mm_unpackhi_epi32(b0, b2);
    let c2 = _mm_unpacklo_epi32(b1, b3);
    let c3 = _mm_unpackhi_epi32(b1, b3);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30 40 50 60 70
    // out[1]: 01 11 21 31 41 51 61 71
    // out[2]: 02 12 22 32 42 52 62 72
    // out[3]: 03 13 23 33 43 53 63 73
    // out[4]: 04 14 24 34 44 54 64 74
    // out[5]: 05 15 25 35 45 55 65 75
    // out[6]: 06 16 26 36 46 56 66 76
    // out[7]: 07 17 27 37 47 57 67 77
    let r0 = _mm_unpacklo_epi64(c0, c0);
    let r1 = _mm_unpackhi_epi64(c0, c0);
    let r2 = _mm_unpacklo_epi64(c1, c1);
    let r3 = _mm_unpackhi_epi64(c1, c1);
    let r4 = _mm_unpacklo_epi64(c2, c2);
    let r5 = _mm_unpackhi_epi64(c2, c2);
    let r6 = _mm_unpacklo_epi64(c3, c3);
    let r7 = _mm_unpackhi_epi64(c3, c3);

    if FLIP {
        let rsh = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        (
            (
                _mm_shuffle_epi8(r0, rsh),
                _mm_shuffle_epi8(r1, rsh),
                _mm_shuffle_epi8(r2, rsh),
                _mm_shuffle_epi8(r3, rsh),
            ),
            (
                _mm_shuffle_epi8(r4, rsh),
                _mm_shuffle_epi8(r5, rsh),
                _mm_shuffle_epi8(r6, rsh),
                _mm_shuffle_epi8(r7, rsh),
            ),
        )
    } else {
        ((r0, r1, r2, r3), (r4, r5, r6, r7))
    }
}

#[inline(always)]
pub(crate) fn sse_transpose_u8_8x8<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm_loadu_si64(src.get_unchecked(0..).as_ptr());
        let row1 = _mm_loadu_si64(src.get_unchecked(src_stride..).as_ptr());
        let row2 = _mm_loadu_si64(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = _mm_loadu_si64(src.get_unchecked(3 * src_stride..).as_ptr());
        let row4 = _mm_loadu_si64(src.get_unchecked(4 * src_stride..).as_ptr());
        let row5 = _mm_loadu_si64(src.get_unchecked(5 * src_stride..).as_ptr());
        let row6 = _mm_loadu_si64(src.get_unchecked(6 * src_stride..).as_ptr());
        let row7 = _mm_loadu_si64(src.get_unchecked(7 * src_stride..).as_ptr());

        let (v0, v1) =
            sse_transpose_8x8_impl::<FLIP>((row0, row1, row2, row3), (row4, row5, row6, row7));

        if FLOP {
            _mm_storeu_si64(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            _mm_storeu_si64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            _mm_storeu_si64(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            _mm_storeu_si64(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
            _mm_storeu_si64(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v1.0);
            _mm_storeu_si64(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v1.1);
            _mm_storeu_si64(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v1.2);
            _mm_storeu_si64(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v1.3);
        } else {
            _mm_storeu_si64(dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr(), v0.0);
            _mm_storeu_si64(dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr(), v0.1);
            _mm_storeu_si64(dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr(), v0.2);
            _mm_storeu_si64(dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr(), v0.3);
            _mm_storeu_si64(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v1.0);
            _mm_storeu_si64(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v1.1);
            _mm_storeu_si64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v1.2);
            _mm_storeu_si64(dst.get_unchecked_mut(0..).as_mut_ptr(), v1.3);
        }
    }
}
