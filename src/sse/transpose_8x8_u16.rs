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

use crate::sse::utils::{_mm_loadu_si128_u16, _mm_storeu_si128_u16};
use crate::sse::vld::{
    _mm_load_deinterleave_la16, _mm_load_deinterleave_rgb16, _mm_load_deinterleave_rgba16,
    _mm_store_interleave_la16, _mm_store_interleave_rgb16, _mm_store_interleave_rgba16,
};
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
    // Unpack 16 bit elements. Goes from:
    // in[0]: 00 01 02 03  04 05 06 07
    // in[1]: 10 11 12 13  14 15 16 17
    // in[2]: 20 21 22 23  24 25 26 27
    // in[3]: 30 31 32 33  34 35 36 37
    // in[4]: 40 41 42 43  44 45 46 47
    // in[5]: 50 51 52 53  54 55 56 57
    // in[6]: 60 61 62 63  64 65 66 67
    // in[7]: 70 71 72 73  74 75 76 77
    // to:
    // a0:    00 10 01 11  02 12 03 13
    // a1:    20 30 21 31  22 32 23 33
    // a2:    40 50 41 51  42 52 43 53
    // a3:    60 70 61 71  62 72 63 73
    // a4:    04 14 05 15  06 16 07 17
    // a5:    24 34 25 35  26 36 27 37
    // a6:    44 54 45 55  46 56 47 57
    // a7:    64 74 65 75  66 76 67 77
    let a0 = _mm_unpacklo_epi16(v0.0, v0.1);
    let a1 = _mm_unpacklo_epi16(v0.2, v0.3);
    let a2 = _mm_unpacklo_epi16(v1.0, v1.1);
    let a3 = _mm_unpacklo_epi16(v1.2, v1.3);
    let a4 = _mm_unpackhi_epi16(v0.0, v0.1);
    let a5 = _mm_unpackhi_epi16(v0.2, v0.3);
    let a6 = _mm_unpackhi_epi16(v1.0, v1.1);
    let a7 = _mm_unpackhi_epi16(v1.2, v1.3);

    // Unpack 32 bit elements resulting in:
    // b0: 00 10 20 30  01 11 21 31
    // b1: 40 50 60 70  41 51 61 71
    // b2: 04 14 24 34  05 15 25 35
    // b3: 44 54 64 74  45 55 65 75
    // b4: 02 12 22 32  03 13 23 33
    // b5: 42 52 62 72  43 53 63 73
    // b6: 06 16 26 36  07 17 27 37
    // b7: 46 56 66 76  47 57 67 77
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpacklo_epi32(a2, a3);
    let b2 = _mm_unpacklo_epi32(a4, a5);
    let b3 = _mm_unpacklo_epi32(a6, a7);
    let b4 = _mm_unpackhi_epi32(a0, a1);
    let b5 = _mm_unpackhi_epi32(a2, a3);
    let b6 = _mm_unpackhi_epi32(a4, a5);
    let b7 = _mm_unpackhi_epi32(a6, a7);

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30  40 50 60 70
    // out[1]: 01 11 21 31  41 51 61 71
    // out[2]: 02 12 22 32  42 52 62 72
    // out[3]: 03 13 23 33  43 53 63 73
    // out[4]: 04 14 24 34  44 54 64 74
    // out[5]: 05 15 25 35  45 55 65 75
    // out[6]: 06 16 26 36  46 56 66 76
    // out[7]: 07 17 27 37  47 57 67 77
    let r0 = _mm_unpacklo_epi64(b0, b1);
    let r1 = _mm_unpackhi_epi64(b0, b1);
    let r2 = _mm_unpacklo_epi64(b4, b5);
    let r3 = _mm_unpackhi_epi64(b4, b5);
    let r4 = _mm_unpacklo_epi64(b2, b3);
    let r5 = _mm_unpackhi_epi64(b2, b3);
    let r6 = _mm_unpacklo_epi64(b6, b7);
    let r7 = _mm_unpackhi_epi64(b6, b7);

    if FLIP {
        let rsh = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
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

#[inline]
pub(crate) fn sse_transpose_8x8_u16<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe { sse_transpose_8x8_u16_impl_1::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "ssse3")]
unsafe fn sse_transpose_8x8_u16_impl_1<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let row0 = _mm_loadu_si128_u16(src.get_unchecked(0..));
    let row1 = _mm_loadu_si128_u16(src.get_unchecked(src_stride..));
    let row2 = _mm_loadu_si128_u16(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_loadu_si128_u16(src.get_unchecked(3 * src_stride..));
    let row4 = _mm_loadu_si128_u16(src.get_unchecked(4 * src_stride..));
    let row5 = _mm_loadu_si128_u16(src.get_unchecked(5 * src_stride..));
    let row6 = _mm_loadu_si128_u16(src.get_unchecked(6 * src_stride..));
    let row7 = _mm_loadu_si128_u16(src.get_unchecked(7 * src_stride..));

    let (v0, v1) =
        sse_transpose_8x8_impl::<FLIP>((row0, row1, row2, row3), (row4, row5, row6, row7));

    if FLOP {
        _mm_storeu_si128_u16(dst.get_unchecked_mut(7 * dst_stride..), v0.0);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(6 * dst_stride..), v0.1);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(5 * dst_stride..), v0.2);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(4 * dst_stride..), v0.3);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(3 * dst_stride..), v1.0);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(2 * dst_stride..), v1.1);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(dst_stride..), v1.2);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(0..), v1.3);
    } else {
        _mm_storeu_si128_u16(dst.get_unchecked_mut(0..), v0.0);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(dst_stride..), v0.1);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(2 * dst_stride..), v0.2);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(3 * dst_stride..), v0.3);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(4 * dst_stride..), v1.0);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(5 * dst_stride..), v1.1);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(6 * dst_stride..), v1.2);
        _mm_storeu_si128_u16(dst.get_unchecked_mut(7 * dst_stride..), v1.3);
    }
}

#[inline]
pub(crate) fn sse_transpose_8x8_u16_intl3<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe { sse_transpose_8x8_u16_impl_x3::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_8x8_u16_impl_x3<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_rgb16(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_rgb16(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_rgb16(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_rgb16(src.get_unchecked(3 * src_stride..));
    let row4 = _mm_load_deinterleave_rgb16(src.get_unchecked(4 * src_stride..));
    let row5 = _mm_load_deinterleave_rgb16(src.get_unchecked(5 * src_stride..));
    let row6 = _mm_load_deinterleave_rgb16(src.get_unchecked(6 * src_stride..));
    let row7 = _mm_load_deinterleave_rgb16(src.get_unchecked(7 * src_stride..));

    let (r0, r1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.0, row1.0, row2.0, row3.0),
        (row4.0, row5.0, row6.0, row7.0),
    );

    let (g0, g1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.1, row1.1, row2.1, row3.1),
        (row4.1, row5.1, row6.1, row7.1),
    );

    let (b0, b1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.2, row1.2, row2.2, row3.2),
        (row4.2, row5.2, row6.2, row7.2),
    );

    if FLOP {
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(0..), (r0.0, g0.0, b0.0));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(dst_stride..), (r0.1, g0.1, b0.1));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(2 * dst_stride..), (r0.2, g0.2, b0.2));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(3 * dst_stride..), (r0.3, g0.3, b0.3));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(4 * dst_stride..), (r1.0, g1.0, b1.0));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(5 * dst_stride..), (r1.1, g1.1, b1.1));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(6 * dst_stride..), (r1.2, g1.2, b1.2));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(7 * dst_stride..), (r1.3, g1.3, b1.3));
    } else {
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(7 * dst_stride..), (r0.0, g0.0, b0.0));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(6 * dst_stride..), (r0.1, g0.1, b0.1));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(5 * dst_stride..), (r0.2, g0.2, b0.2));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(4 * dst_stride..), (r0.3, g0.3, b0.3));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(3 * dst_stride..), (r1.0, g1.0, b1.0));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(2 * dst_stride..), (r1.1, g1.1, b1.1));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(dst_stride..), (r1.2, g1.2, b1.2));
        _mm_store_interleave_rgb16(dst.get_unchecked_mut(0..), (r1.3, g1.3, b1.3));
    }
}

#[inline]
pub(crate) fn sse_transpose_8x8_u16_intl4<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe { sse_transpose_8x8_impl_intl_4::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_8x8_impl_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_rgba16(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_rgba16(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_rgba16(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_rgba16(src.get_unchecked(3 * src_stride..));
    let row4 = _mm_load_deinterleave_rgba16(src.get_unchecked(4 * src_stride..));
    let row5 = _mm_load_deinterleave_rgba16(src.get_unchecked(5 * src_stride..));
    let row6 = _mm_load_deinterleave_rgba16(src.get_unchecked(6 * src_stride..));
    let row7 = _mm_load_deinterleave_rgba16(src.get_unchecked(7 * src_stride..));

    let (r0, r1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.0, row1.0, row2.0, row3.0),
        (row4.0, row5.0, row6.0, row7.0),
    );

    let (g0, g1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.1, row1.1, row2.1, row3.1),
        (row4.1, row5.1, row6.1, row7.1),
    );

    let (b0, b1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.2, row1.2, row2.2, row3.2),
        (row4.2, row5.2, row6.2, row7.2),
    );

    let (a0, a1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.3, row1.3, row2.3, row3.3),
        (row4.3, row5.3, row6.3, row7.3),
    );

    if FLOP {
        _mm_store_interleave_rgba16(dst.get_unchecked_mut(0..), (r0.0, g0.0, b0.0, a0.0));
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(dst_stride..),
            (r0.1, g0.1, b0.1, a0.1),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(2 * dst_stride..),
            (r0.2, g0.2, b0.2, a0.2),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(3 * dst_stride..),
            (r0.3, g0.3, b0.3, a0.3),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(4 * dst_stride..),
            (r1.0, g1.0, b1.0, a1.0),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(5 * dst_stride..),
            (r1.1, g1.1, b1.1, a1.1),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(6 * dst_stride..),
            (r1.2, g1.2, b1.2, a1.2),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(7 * dst_stride..),
            (r1.3, g1.3, b1.3, a1.3),
        );
    } else {
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(7 * dst_stride..),
            (r0.0, g0.0, b0.0, a0.0),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(6 * dst_stride..),
            (r0.1, g0.1, b0.1, a0.1),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(5 * dst_stride..),
            (r0.2, g0.2, b0.2, a0.2),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(4 * dst_stride..),
            (r0.3, g0.3, b0.3, a0.3),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(3 * dst_stride..),
            (r1.0, g1.0, b1.0, a1.0),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(2 * dst_stride..),
            (r1.1, g1.1, b1.1, a1.1),
        );
        _mm_store_interleave_rgba16(
            dst.get_unchecked_mut(dst_stride..),
            (r1.2, g1.2, b1.2, a1.2),
        );
        _mm_store_interleave_rgba16(dst.get_unchecked_mut(0..), (r1.3, g1.3, b1.3, a1.3));
    }
}

#[inline]
pub(crate) fn sse_transpose_8x8_u16_intl2<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe { sse_transpose_8x8_impl_intl_2::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_8x8_impl_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_la16(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_la16(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_la16(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_la16(src.get_unchecked(3 * src_stride..));
    let row4 = _mm_load_deinterleave_la16(src.get_unchecked(4 * src_stride..));
    let row5 = _mm_load_deinterleave_la16(src.get_unchecked(5 * src_stride..));
    let row6 = _mm_load_deinterleave_la16(src.get_unchecked(6 * src_stride..));
    let row7 = _mm_load_deinterleave_la16(src.get_unchecked(7 * src_stride..));

    let (r0, r1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.0, row1.0, row2.0, row3.0),
        (row4.0, row5.0, row6.0, row7.0),
    );

    let (g0, g1) = sse_transpose_8x8_impl::<FLIP>(
        (row0.1, row1.1, row2.1, row3.1),
        (row4.1, row5.1, row6.1, row7.1),
    );

    if FLOP {
        _mm_store_interleave_la16(dst.get_unchecked_mut(0..), (r0.0, g0.0));
        _mm_store_interleave_la16(dst.get_unchecked_mut(dst_stride..), (r0.1, g0.1));
        _mm_store_interleave_la16(dst.get_unchecked_mut(2 * dst_stride..), (r0.2, g0.2));
        _mm_store_interleave_la16(dst.get_unchecked_mut(3 * dst_stride..), (r0.3, g0.3));
        _mm_store_interleave_la16(dst.get_unchecked_mut(4 * dst_stride..), (r1.0, g1.0));
        _mm_store_interleave_la16(dst.get_unchecked_mut(5 * dst_stride..), (r1.1, g1.1));
        _mm_store_interleave_la16(dst.get_unchecked_mut(6 * dst_stride..), (r1.2, g1.2));
        _mm_store_interleave_la16(dst.get_unchecked_mut(7 * dst_stride..), (r1.3, g1.3));
    } else {
        _mm_store_interleave_la16(dst.get_unchecked_mut(7 * dst_stride..), (r0.0, g0.0));
        _mm_store_interleave_la16(dst.get_unchecked_mut(6 * dst_stride..), (r0.1, g0.1));
        _mm_store_interleave_la16(dst.get_unchecked_mut(5 * dst_stride..), (r0.2, g0.2));
        _mm_store_interleave_la16(dst.get_unchecked_mut(4 * dst_stride..), (r0.3, g0.3));
        _mm_store_interleave_la16(dst.get_unchecked_mut(3 * dst_stride..), (r1.0, g1.0));
        _mm_store_interleave_la16(dst.get_unchecked_mut(2 * dst_stride..), (r1.1, g1.1));
        _mm_store_interleave_la16(dst.get_unchecked_mut(dst_stride..), (r1.2, g1.2));
        _mm_store_interleave_la16(dst.get_unchecked_mut(0..), (r1.3, g1.3));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_transpose_8x8() {
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

        sse_transpose_8x8_u16::<false, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }

    #[test]
    fn test_sse_transpose_8x8_flip() {
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
        sse_transpose_8x8_u16::<true, false>(
            &src, 8, // src_stride
            &mut dst, 8, // dst_stride
        );

        assert_eq!(
            expected, dst,
            "The transposed matrix does not match the expected result"
        );
    }
}
