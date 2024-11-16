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
const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_la_ps(a: __m128, b: __m128) -> (__m128, __m128) {
    let v0 = _mm_unpacklo_ps(a, b); // a0 b0 a1 b1
    let v1 = _mm_unpackhi_ps(a, b); // a2 b2 a3 b3
    (v0, v1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_la_ps(a: __m128, b: __m128) -> (__m128, __m128) {
    const FLAG_1: i32 = shuffle(2, 0, 2, 0);

    let v0 = _mm_shuffle_ps::<FLAG_1>(a, b); // a0 a1 a2 a3
    const FLAG_2: i32 = shuffle(3, 1, 3, 1);
    let v1 = _mm_shuffle_ps::<FLAG_2>(a, b); // b0 b1 ab b3
    (v0, v1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgb_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
) -> (__m128, __m128, __m128) {
    const FLAG_1: i32 = shuffle(0, 1, 0, 2);
    let at12 = _mm_shuffle_ps::<FLAG_1>(t1, t2);
    const FLAG_2: i32 = shuffle(2, 0, 3, 0);
    let v0 = _mm_shuffle_ps::<FLAG_2>(t0, at12);
    const FLAG_3: i32 = shuffle(0, 0, 0, 1);
    let bt01 = _mm_shuffle_ps::<FLAG_3>(t0, t1);
    const FLAG_4: i32 = shuffle(0, 2, 0, 3);
    let bt12 = _mm_shuffle_ps::<FLAG_4>(t1, t2);
    const FLAG_5: i32 = shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<FLAG_5>(bt01, bt12);

    const FLAG_6: i32 = shuffle(0, 1, 0, 2);
    let ct01 = _mm_shuffle_ps::<FLAG_6>(t0, t1);
    const FLAG_7: i32 = shuffle(3, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<FLAG_7>(ct01, t2);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgb_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
) -> (__m128, __m128, __m128) {
    const FLAG_1: i32 = shuffle(0, 0, 0, 0);
    let u0 = _mm_shuffle_ps::<FLAG_1>(t0, t1);
    const FLAG_2: i32 = shuffle(1, 1, 0, 0);
    let u1 = _mm_shuffle_ps::<FLAG_2>(t2, t0);
    const FLAG_3: i32 = shuffle(2, 0, 2, 0);
    let v0 = _mm_shuffle_ps::<FLAG_3>(u0, u1);
    const FLAG_4: i32 = shuffle(1, 1, 1, 1);
    let u2 = _mm_shuffle_ps::<FLAG_4>(t1, t2);
    const FLAG_5: i32 = shuffle(2, 2, 2, 2);
    let u3 = _mm_shuffle_ps::<FLAG_5>(t0, t1);
    const FLAG_6: i32 = shuffle(2, 0, 2, 0);
    let v1 = _mm_shuffle_ps::<FLAG_6>(u2, u3);
    const FLAG_7: i32 = shuffle(3, 3, 2, 2);
    let u4 = _mm_shuffle_ps::<FLAG_7>(t2, t0);
    const FLAG_8: i32 = shuffle(3, 3, 3, 3);
    let u5 = _mm_shuffle_ps::<FLAG_8>(t1, t2);
    const FLAG_9: i32 = shuffle(2, 0, 2, 0);
    let v2 = _mm_shuffle_ps::<FLAG_9>(u4, u5);
    (v0, v1, v2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_deinterleave_rgba_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
    t3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let t02lo = _mm_unpacklo_ps(t0, t2);
    let t13lo = _mm_unpacklo_ps(t1, t3);
    let t02hi = _mm_unpackhi_ps(t0, t2);
    let t13hi = _mm_unpackhi_ps(t1, t3);
    let v0 = _mm_unpacklo_ps(t02lo, t13lo);
    let v1 = _mm_unpackhi_ps(t02lo, t13lo);
    let v2 = _mm_unpacklo_ps(t02hi, t13hi);
    let v3 = _mm_unpackhi_ps(t02hi, t13hi);
    (v0, v1, v2, v3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_interleave_rgba_ps(
    t0: __m128,
    t1: __m128,
    t2: __m128,
    t3: __m128,
) -> (__m128, __m128, __m128, __m128) {
    let u0 = _mm_unpacklo_ps(t0, t2);
    let u1 = _mm_unpacklo_ps(t1, t3);
    let u2 = _mm_unpackhi_ps(t0, t2);
    let u3 = _mm_unpackhi_ps(t1, t3);
    let v0 = _mm_unpacklo_ps(u0, u1);
    let v2 = _mm_unpacklo_ps(u2, u3);
    let v1 = _mm_unpackhi_ps(u0, u1);
    let v3 = _mm_unpackhi_ps(u2, u3);
    (v0, v1, v2, v3)
}
