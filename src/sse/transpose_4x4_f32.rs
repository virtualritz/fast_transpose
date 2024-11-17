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

use crate::sse::vld::{
    _mm_load_deinterleave_la_ps, _mm_load_deinterleave_rgb_ps, _mm_load_deinterleave_rgba_ps,
    _mm_store_interleave_la_ps, _mm_store_interleave_rgb_ps, _mm_store_interleave_rgba_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
unsafe fn sse_transpose_4x4_impl<const FLIP: bool>(
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

    let a0 = _mm_unpacklo_epi32(_mm_castps_si128(v0.0), _mm_castps_si128(v0.1));
    let a1 = _mm_unpacklo_epi32(_mm_castps_si128(v0.2), _mm_castps_si128(v0.3));
    let a2 = _mm_unpackhi_epi32(_mm_castps_si128(v0.0), _mm_castps_si128(v0.1));
    let a3 = _mm_unpackhi_epi32(_mm_castps_si128(v0.2), _mm_castps_si128(v0.3));

    // Unpack 64 bit elements resulting in:
    // out[0]: 00 10 20 30
    // out[1]: 01 11 21 31
    // out[2]: 02 12 22 32
    // out[3]: 03 13 23 33
    let r0 = _mm_unpacklo_epi64(a0, a1);
    let r1 = _mm_unpackhi_epi64(a0, a1);
    let r2 = _mm_unpacklo_epi64(a2, a3);
    let r3 = _mm_unpackhi_epi64(a2, a3);

    if FLIP {
        let rsh = _mm_setr_epi8(12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
        (
            _mm_castsi128_ps(_mm_shuffle_epi8(r0, rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(r1, rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(r2, rsh)),
            _mm_castsi128_ps(_mm_shuffle_epi8(r3, rsh)),
        )
    } else {
        (
            _mm_castsi128_ps(r0),
            _mm_castsi128_ps(r1),
            _mm_castsi128_ps(r2),
            _mm_castsi128_ps(r3),
        )
    }
}

#[inline]
pub(crate) fn sse_transpose_4x4_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe { sse_transpose_4x4_f32_impl_1::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "ssse3")]
unsafe fn sse_transpose_4x4_f32_impl_1<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let row0 = _mm_loadu_ps(src.get_unchecked(0..).as_ptr());
    let row1 = _mm_loadu_ps(src.get_unchecked(src_stride..).as_ptr());
    let row2 = _mm_loadu_ps(src.get_unchecked(2 * src_stride..).as_ptr());
    let row3 = _mm_loadu_ps(src.get_unchecked(3 * src_stride..).as_ptr());

    let v0 = sse_transpose_4x4_impl::<FLIP>((row0, row1, row2, row3));

    if FLOP {
        _mm_storeu_ps(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.0);
        _mm_storeu_ps(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.1);
        _mm_storeu_ps(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.2);
        _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.3);
    } else {
        _mm_storeu_ps(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
        _mm_storeu_ps(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
        _mm_storeu_ps(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
        _mm_storeu_ps(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
    }
}

#[inline]
pub(crate) fn sse_transpose_4x4_f32_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe { sse_transpose_4x4_f32_impl_2::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_4x4_f32_impl_2<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_la_ps(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_la_ps(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_la_ps(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_la_ps(src.get_unchecked(3 * src_stride..));

    let r = sse_transpose_4x4_impl::<FLIP>((row0.0, row1.0, row2.0, row3.0));
    let g = sse_transpose_4x4_impl::<FLIP>((row0.1, row1.1, row2.1, row3.1));

    if FLOP {
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(3 * dst_stride..), (r.0, g.0));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(2 * dst_stride..), (r.1, g.1));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(dst_stride..), (r.2, g.2));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(0..), (r.3, g.3));
    } else {
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(0..), (r.0, g.0));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(dst_stride..), (r.1, g.1));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(2 * dst_stride..), (r.2, g.2));
        _mm_store_interleave_la_ps(dst.get_unchecked_mut(3 * dst_stride..), (r.3, g.3));
    }
}

#[inline]
pub(crate) fn sse_transpose_4x4_f32_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe { sse_transpose_4x4_f32_impl_3::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_4x4_f32_impl_3<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_rgb_ps(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_rgb_ps(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_rgb_ps(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_rgb_ps(src.get_unchecked(3 * src_stride..));

    let r = sse_transpose_4x4_impl::<FLIP>((row0.0, row1.0, row2.0, row3.0));
    let g = sse_transpose_4x4_impl::<FLIP>((row0.1, row1.1, row2.1, row3.1));
    let b = sse_transpose_4x4_impl::<FLIP>((row0.2, row1.2, row2.2, row3.2));

    if FLOP {
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(3 * dst_stride..), (r.0, g.0, b.0));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(2 * dst_stride..), (r.1, g.1, b.1));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(dst_stride..), (r.2, g.2, b.2));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(0..), (r.3, g.3, b.3));
    } else {
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(0..), (r.0, g.0, b.0));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(dst_stride..), (r.1, g.1, b.1));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(2 * dst_stride..), (r.2, g.2, b.2));
        _mm_store_interleave_rgb_ps(dst.get_unchecked_mut(3 * dst_stride..), (r.3, g.3, b.3));
    }
}

#[inline]
pub(crate) fn sse_transpose_4x4_f32_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe { sse_transpose_4x4_f32_impl_4::<FLOP, FLIP>(src, src_stride, dst, dst_stride) }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_transpose_4x4_f32_impl_4<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    let row0 = _mm_load_deinterleave_rgba_ps(src.get_unchecked(0..));
    let row1 = _mm_load_deinterleave_rgba_ps(src.get_unchecked(src_stride..));
    let row2 = _mm_load_deinterleave_rgba_ps(src.get_unchecked(2 * src_stride..));
    let row3 = _mm_load_deinterleave_rgba_ps(src.get_unchecked(3 * src_stride..));

    let r = sse_transpose_4x4_impl::<FLIP>((row0.0, row1.0, row2.0, row3.0));
    let g = sse_transpose_4x4_impl::<FLIP>((row0.1, row1.1, row2.1, row3.1));
    let b = sse_transpose_4x4_impl::<FLIP>((row0.2, row1.2, row2.2, row3.2));
    let a = sse_transpose_4x4_impl::<FLIP>((row0.3, row1.3, row2.3, row3.3));

    if FLOP {
        _mm_store_interleave_rgba_ps(
            dst.get_unchecked_mut(3 * dst_stride..),
            (r.0, g.0, b.0, a.0),
        );
        _mm_store_interleave_rgba_ps(
            dst.get_unchecked_mut(2 * dst_stride..),
            (r.1, g.1, b.1, a.1),
        );
        _mm_store_interleave_rgba_ps(dst.get_unchecked_mut(dst_stride..), (r.2, g.2, b.2, a.2));
        _mm_store_interleave_rgba_ps(dst.get_unchecked_mut(0..), (r.3, g.3, b.3, a.3));
    } else {
        _mm_store_interleave_rgba_ps(dst.get_unchecked_mut(0..), (r.0, g.0, b.0, a.0));
        _mm_store_interleave_rgba_ps(dst.get_unchecked_mut(dst_stride..), (r.1, g.1, b.1, a.1));
        _mm_store_interleave_rgba_ps(
            dst.get_unchecked_mut(2 * dst_stride..),
            (r.2, g.2, b.2, a.2),
        );
        _mm_store_interleave_rgba_ps(
            dst.get_unchecked_mut(3 * dst_stride..),
            (r.3, g.3, b.3, a.3),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sse_transpose_4x4_f32() {
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
        sse_transpose_4x4_f32::<false, false>(
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
}
