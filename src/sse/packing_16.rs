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
pub(crate) unsafe fn _mm_deinterleave_rgba_epi16(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v0 = _mm_unpacklo_epi16(rgba0, rgba2); // a0 a4 b0 b4 ...
    let v1 = _mm_unpackhi_epi16(rgba0, rgba2); // a1 a5 b1 b5 ...
    let v2 = _mm_unpacklo_epi16(rgba1, rgba3); // a2 a6 b2 b6 ...
    let v3 = _mm_unpackhi_epi16(rgba1, rgba3); // a3 a7 b3 b7 ...

    let u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    let u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    let u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    let u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    let a = _mm_unpacklo_epi16(u0, u1);
    let b = _mm_unpackhi_epi16(u0, u1);
    let c = _mm_unpacklo_epi16(u2, u3);
    let d = _mm_unpackhi_epi16(u2, u3);
    (a, b, c, d)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgba_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    let u0 = _mm_unpacklo_epi16(a, c); // a0 c0 a1 c1 ...
    let u1 = _mm_unpackhi_epi16(a, c); // a4 c4 a5 c5 ...
    let u2 = _mm_unpacklo_epi16(b, d); // b0 d0 b1 d1 ...
    let u3 = _mm_unpackhi_epi16(b, d); // b4 d4 b5 d5 ...

    let v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    let v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    let v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    let v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...
    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgb_epi16(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let a0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba0, rgba1), rgba2);
    let b0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba2, rgba0), rgba1);
    let c0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba1, rgba2), rgba0);

    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a0, sh_a);
    let b0 = _mm_shuffle_epi8(b0, sh_b);
    let c0 = _mm_shuffle_epi8(c0, sh_c);
    (a0, b0, c0)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgb_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a, sh_a);
    let b0 = _mm_shuffle_epi8(b, sh_b);
    let c0 = _mm_shuffle_epi8(c, sh_c);

    let v0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(a0, b0), c0);
    let v1 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(c0, a0), b0);
    let v2 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(b0, c0), a0);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_la_epi16(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    let v2 = _mm_unpacklo_epi16(a, b); // a0 a4 b0 b4 a1 a5 b1 b5
    let v3 = _mm_unpackhi_epi16(a, b); // a2 a6 b2 b6 a3 a7 b3 b7
    let v4 = _mm_unpacklo_epi16(v2, v3); // a0 a2 a4 a6 b0 b2 b4 b6
    let v5 = _mm_unpackhi_epi16(v2, v3); // a1 a3 a5 a7 b1 b3 b5 b7

    let av = _mm_unpacklo_epi16(v4, v5); // a0 a1 a2 a3 a4 a5 a6 a7
    let bv = _mm_unpackhi_epi16(v4, v5); // b0 b1 ab b3 b4 b5 b6 b7
    (av, bv)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_la_epi16(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    let v0 = _mm_unpacklo_epi16(a, b);
    let v1 = _mm_unpackhi_epi16(a, b);
    (v0, v1)
}
