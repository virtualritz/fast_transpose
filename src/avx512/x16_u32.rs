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
#[allow(clippy::type_complexity)]
unsafe fn avx512_transpose_16x16_impl<const FLIP: bool>(
    v0: (__m512i, __m512i, __m512i, __m512i),
    v1: (__m512i, __m512i, __m512i, __m512i),
    v2: (__m512i, __m512i, __m512i, __m512i),
    v3: (__m512i, __m512i, __m512i, __m512i),
) -> (
    (__m512i, __m512i, __m512i, __m512i),
    (__m512i, __m512i, __m512i, __m512i),
    (__m512i, __m512i, __m512i, __m512i),
    (__m512i, __m512i, __m512i, __m512i),
) {
    // let r0 = _mm512_castsi512_ps(v0.0);
    // let r1 = _mm512_castsi512_ps(v0.1);
    // let r2 = _mm512_castsi512_ps(v0.2);
    // let r3 = _mm512_castsi512_ps(v0.3);
    // let r4 = _mm512_castsi512_ps(v1.0);
    // let r5 = _mm512_castsi512_ps(v1.1);
    // let r6 = _mm512_castsi512_ps(v1.2);
    // let r7 = _mm512_castsi512_ps(v1.3);
    // let r8 = _mm512_castsi512_ps(v2.0);
    // let r9 = _mm512_castsi512_ps(v2.1);
    // let r10 = _mm512_castsi512_ps(v2.2);
    // let r11 = _mm512_castsi512_ps(v2.3);
    // let r12 = _mm512_castsi512_ps(v3.0);
    // let r13 = _mm512_castsi512_ps(v3.1);
    // let r14 = _mm512_castsi512_ps(v3.2);
    // let r15 = _mm512_castsi512_ps(v3.3);
    //
    // let _tmp0 = _mm512_unpacklo_epi32(r0, r1);
    // let _tmp1 = _mm512_unpackhi_epi32(r0, r1);
    // let _tmp2 = _mm512_unpacklo_epi32(r2, r3);
    // let _tmp3 = _mm512_unpackhi_epi32(r2, r3);
    // let _tmp4 = _mm512_unpacklo_epi32(r4, r5);
    // let _tmp5 = _mm512_unpackhi_epi32(r4, r5);
    // let _tmp6 = _mm512_unpacklo_epi32(r6, r7);
    // let _tmp7 = _mm512_unpackhi_epi32(r6, r7);
    // let _tmp8 = _mm512_unpacklo_epi32(r8, r9);
    // let _tmp9 = _mm512_unpackhi_epi32(r8, r9);
    // let _tmpa = _mm512_unpacklo_epi32(r10, r11);
    // let _tmpb = _mm512_unpackhi_epi32(r10, r11);
    // let _tmpc = _mm512_unpacklo_epi32(r12, r13);
    // let _tmpd = _mm512_unpackhi_epi32(r12, r13);
    // let _tmpe = _mm512_unpacklo_epi32(r14, r15);
    // let _tmpf = _mm512_unpackhi_epi32(r14, r15);
    //
    // const S_1: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpg = _mm512_shuffle_ps::<S_1>(_tmp0, _tmp2);
    // const S_2: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmph = _mm512_shuffle_ps::<S_2>(_tmp0, _tmp2);
    // const S_3: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpi = _mm512_shuffle_ps::<S_3>(_tmp1, _tmp3);
    // const S_4: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpj = _mm512_shuffle_ps::<S_4>(_tmp1, _tmp3);
    // const S_5: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpk = _mm512_shuffle_ps::<S_5>(_tmp4, _tmp6);
    // const S_6: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpl = _mm512_shuffle_ps::<S_6>(_tmp4, _tmp6);
    // const S_7: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpm = _mm512_shuffle_ps::<S_7>(_tmp5, _tmp7);
    // const S_8: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpn = _mm512_shuffle_ps::<S_8>(_tmp5, _tmp7);
    // const S_9: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpo = _mm512_shuffle_ps::<S_9>(_tmp8, _tmpa);
    // const S_10: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpp = _mm512_shuffle_ps::<S_10>(_tmp8, _tmpa);
    // const S_11: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpq = _mm512_shuffle_ps::<S_11>(_tmp9, _tmpb);
    // const S_12: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpr = _mm512_shuffle_ps::<S_12>(_tmp9, _tmpb);
    // const S_13: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmps = _mm512_shuffle_ps::<S_13>(_tmpc, _tmpe);
    // const S_14: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpt = _mm512_shuffle_ps::<S_14>(_tmpc, _tmpe);
    // const S_15: i32 = _mm_shuffle(1, 0, 1, 0);
    // let _tmpu = _mm512_shuffle_ps::<S_15>(_tmpd, _tmpf);
    // const S_16: i32 = _mm_shuffle(3, 2, 3, 2);
    // let _tmpv = _mm512_shuffle_ps::<S_16>(_tmpd, _tmpf);
    //
    // const V_1: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp0 = _mm512_shuffle_f32x4::<V_1>(_tmpg, _tmpk);
    // const V_2: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp1 = _mm512_shuffle_f32x4::<V_2>(_tmpo, _tmps);
    // const V_3: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp2 = _mm512_shuffle_f32x4::<V_3>(_tmph, _tmpl);
    // const V_4: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp3 = _mm512_shuffle_f32x4::<V_4>(_tmpp, _tmpt);
    // const V_5: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp4 = _mm512_shuffle_f32x4::<V_5>(_tmpi, _tmpm);
    // const V_6: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp5 = _mm512_shuffle_f32x4::<V_6>(_tmpq, _tmpu);
    // const V_7: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp6 = _mm512_shuffle_f32x4::<V_7>(_tmpj, _tmpn);
    // const V_8: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _tmp7 = _mm512_shuffle_f32x4::<V_8>(_tmpr, _tmpv);
    // const V_9: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmp8 = _mm512_shuffle_f32x4::<V_9>(_tmpg, _tmpk);
    // const V_10: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmp9 = _mm512_shuffle_f32x4::<V_10>(_tmpo, _tmps);
    // const V_11: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpa = _mm512_shuffle_f32x4::<V_11>(_tmph, _tmpl);
    // const V_12: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpb = _mm512_shuffle_f32x4::<V_12>(_tmpp, _tmpt);
    // const V_13: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpc = _mm512_shuffle_f32x4::<V_13>(_tmpi, _tmpm);
    // const V_14: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpd = _mm512_shuffle_f32x4::<V_14>(_tmpq, _tmpu);
    // const V_15: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpe = _mm512_shuffle_f32x4::<V_15>(_tmpj, _tmpn);
    // const V_16: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _tmpf = _mm512_shuffle_f32x4::<V_16>(_tmpr, _tmpv);
    //
    // const R_1: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r0 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_1>(_tmp0, _tmp1));
    // const R_2: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r1 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_2>(_tmp2, _tmp3));
    // const R_3: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r2 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_3>(_tmp4, _tmp5));
    // const R_4: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r3 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_4>(_tmp6, _tmp7));
    // const R_5: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r4 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_5>(_tmp8, _tmp9));
    // const R_6: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r5 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_6>(_tmpa, _tmpb));
    // const R_7: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r6 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_7>(_tmpc, _tmpd));
    // const R_8: i32 = _mm_shuffle(2, 0, 2, 0);
    // let _r7 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_8>(_tmpe, _tmpf));
    // const R_9: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r8 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_9>(_tmp0, _tmp1));
    // const R_10: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r9 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_10>(_tmp2, _tmp3));
    // const R_11: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r10 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_11>(_tmp4, _tmp5));
    // const R_12: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r11 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_12>(_tmp6, _tmp7));
    // const R_13: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r12 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_13>(_tmp8, _tmp9));
    // const R_14: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r13 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_14>(_tmpa, _tmpb));
    // const R_15: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r14 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_15>(_tmpc, _tmpd));
    // const R_16: i32 = _mm_shuffle(3, 1, 3, 1);
    // let _r15 = _mm512_castps_si512(_mm512_shuffle_f32x4::<R_16>(_tmpe, _tmpf));

    let r0 = v0.0;
    let r1 = v0.1;
    let r2 = v0.2;
    let r3 = v0.3;
    let r4 = v1.0;
    let r5 = v1.1;
    let r6 = v1.2;
    let r7 = v1.3;
    let r8 = v2.0;
    let r9 = v2.1;
    let r10 = v2.2;
    let r11 = v2.3;
    let r12 = v3.0;
    let r13 = v3.1;
    let r14 = v3.2;
    let r15 = v3.3;

    let _tmp0 = _mm512_unpacklo_epi32(r0, r1);
    let _tmp1 = _mm512_unpackhi_epi32(r0, r1);
    let _tmp2 = _mm512_unpacklo_epi32(r2, r3);
    let _tmp3 = _mm512_unpackhi_epi32(r2, r3);
    let _tmp4 = _mm512_unpacklo_epi32(r4, r5);
    let _tmp5 = _mm512_unpackhi_epi32(r4, r5);
    let _tmp6 = _mm512_unpacklo_epi32(r6, r7);
    let _tmp7 = _mm512_unpackhi_epi32(r6, r7);
    let _tmp8 = _mm512_unpacklo_epi32(r8, r9);
    let _tmp9 = _mm512_unpackhi_epi32(r8, r9);
    let _tmpa = _mm512_unpacklo_epi32(r10, r11);
    let _tmpb = _mm512_unpackhi_epi32(r10, r11);
    let _tmpc = _mm512_unpacklo_epi32(r12, r13);
    let _tmpd = _mm512_unpackhi_epi32(r12, r13);
    let _tmpe = _mm512_unpacklo_epi32(r14, r15);
    let _tmpf = _mm512_unpackhi_epi32(r14, r15);

    let _tmpg = _mm512_unpacklo_epi64(_tmp0, _tmp2);
    let _tmph = _mm512_unpackhi_epi64(_tmp0, _tmp2);
    let _tmpi = _mm512_unpacklo_epi64(_tmp1, _tmp3);
    let _tmpj = _mm512_unpackhi_epi64(_tmp1, _tmp3);
    let _tmpk = _mm512_unpacklo_epi64(_tmp4, _tmp6);
    let _tmpl = _mm512_unpackhi_epi64(_tmp4, _tmp6);
    let _tmpm = _mm512_unpacklo_epi64(_tmp5, _tmp7);
    let _tmpn = _mm512_unpackhi_epi64(_tmp5, _tmp7);
    let _tmpo = _mm512_unpacklo_epi64(_tmp8, _tmpa);
    let _tmpp = _mm512_unpackhi_epi64(_tmp8, _tmpa);
    let _tmpq = _mm512_unpacklo_epi64(_tmp9, _tmpb);
    let _tmpr = _mm512_unpackhi_epi64(_tmp9, _tmpb);
    let _tmps = _mm512_unpacklo_epi64(_tmpc, _tmpe);
    let _tmpt = _mm512_unpackhi_epi64(_tmpc, _tmpe);
    let _tmpu = _mm512_unpacklo_epi64(_tmpd, _tmpf);
    let _tmpv = _mm512_unpackhi_epi64(_tmpd, _tmpf);

    const V_1: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp0 = _mm512_shuffle_i32x4::<V_1>(_tmpg, _tmpk);
    const V_2: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp1 = _mm512_shuffle_i32x4::<V_2>(_tmpo, _tmps);
    const V_3: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp2 = _mm512_shuffle_i32x4::<V_3>(_tmph, _tmpl);
    const V_4: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp3 = _mm512_shuffle_i32x4::<V_4>(_tmpp, _tmpt);
    const V_5: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp4 = _mm512_shuffle_i32x4::<V_5>(_tmpi, _tmpm);
    const V_6: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp5 = _mm512_shuffle_i32x4::<V_6>(_tmpq, _tmpu);
    const V_7: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp6 = _mm512_shuffle_i32x4::<V_7>(_tmpj, _tmpn);
    const V_8: i32 = _mm_shuffle(2, 0, 2, 0);
    let _tmp7 = _mm512_shuffle_i32x4::<V_8>(_tmpr, _tmpv);
    const V_9: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmp8 = _mm512_shuffle_i32x4::<V_9>(_tmpg, _tmpk);
    const V_10: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmp9 = _mm512_shuffle_i32x4::<V_10>(_tmpo, _tmps);
    const V_11: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpa = _mm512_shuffle_i32x4::<V_11>(_tmph, _tmpl);
    const V_12: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpb = _mm512_shuffle_i32x4::<V_12>(_tmpp, _tmpt);
    const V_13: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpc = _mm512_shuffle_i32x4::<V_13>(_tmpi, _tmpm);
    const V_14: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpd = _mm512_shuffle_i32x4::<V_14>(_tmpq, _tmpu);
    const V_15: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpe = _mm512_shuffle_i32x4::<V_15>(_tmpj, _tmpn);
    const V_16: i32 = _mm_shuffle(3, 1, 3, 1);
    let _tmpf = _mm512_shuffle_i32x4::<V_16>(_tmpr, _tmpv);

    const R_1: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r0 = _mm512_shuffle_i32x4::<R_1>(_tmp0, _tmp1);
    const R_2: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r1 = _mm512_shuffle_i32x4::<R_2>(_tmp2, _tmp3);
    const R_3: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r2 = _mm512_shuffle_i32x4::<R_3>(_tmp4, _tmp5);
    const R_4: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r3 = _mm512_shuffle_i32x4::<R_4>(_tmp6, _tmp7);
    const R_5: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r4 = _mm512_shuffle_i32x4::<R_5>(_tmp8, _tmp9);
    const R_6: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r5 = _mm512_shuffle_i32x4::<R_6>(_tmpa, _tmpb);
    const R_7: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r6 = _mm512_shuffle_i32x4::<R_7>(_tmpc, _tmpd);
    const R_8: i32 = _mm_shuffle(2, 0, 2, 0);
    let _r7 = _mm512_shuffle_i32x4::<R_8>(_tmpe, _tmpf);
    const R_9: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r8 = _mm512_shuffle_i32x4::<R_9>(_tmp0, _tmp1);
    const R_10: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r9 = _mm512_shuffle_i32x4::<R_10>(_tmp2, _tmp3);
    const R_11: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r10 = _mm512_shuffle_i32x4::<R_11>(_tmp4, _tmp5);
    const R_12: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r11 = _mm512_shuffle_i32x4::<R_12>(_tmp6, _tmp7);
    const R_13: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r12 = _mm512_shuffle_i32x4::<R_13>(_tmp8, _tmp9);
    const R_14: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r13 = _mm512_shuffle_i32x4::<R_14>(_tmpa, _tmpb);
    const R_15: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r14 = _mm512_shuffle_i32x4::<R_15>(_tmpc, _tmpd);
    const R_16: i32 = _mm_shuffle(3, 1, 3, 1);
    let _r15 = _mm512_shuffle_i32x4::<R_16>(_tmpe, _tmpf);

    if FLIP {
        let flipper = _mm512_set_epi8(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
            46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        );
        (
            (
                _mm512_shuffle_epi8(_r0, flipper),
                _mm512_shuffle_epi8(_r1, flipper),
                _mm512_shuffle_epi8(_r2, flipper),
                _mm512_shuffle_epi8(_r3, flipper),
            ),
            (
                _mm512_shuffle_epi8(_r4, flipper),
                _mm512_shuffle_epi8(_r5, flipper),
                _mm512_shuffle_epi8(_r6, flipper),
                _mm512_shuffle_epi8(_r7, flipper),
            ),
            (
                _mm512_shuffle_epi8(_r8, flipper),
                _mm512_shuffle_epi8(_r9, flipper),
                _mm512_shuffle_epi8(_r10, flipper),
                _mm512_shuffle_epi8(_r11, flipper),
            ),
            (
                _mm512_shuffle_epi8(_r12, flipper),
                _mm512_shuffle_epi8(_r13, flipper),
                _mm512_shuffle_epi8(_r14, flipper),
                _mm512_shuffle_epi8(_r15, flipper),
            ),
        )
    } else {
        (
            (_r0, _r1, _r2, _r3),
            (_r4, _r5, _r6, _r7),
            (_r8, _r9, _r10, _r11),
            (_r12, _r13, _r14, _r15),
        )
    }
}

#[inline(always)]
pub(crate) fn avx512_transpose_16x16_u32<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = _mm512_loadu_si512(src.get_unchecked(0..).as_ptr() as *const _);
        let row1 = _mm512_loadu_si512(src.get_unchecked(src_stride..).as_ptr() as *const _);
        let row2 = _mm512_loadu_si512(src.get_unchecked(2 * src_stride..).as_ptr() as *const _);
        let row3 = _mm512_loadu_si512(src.get_unchecked(3 * src_stride..).as_ptr() as *const _);

        let row4 = _mm512_loadu_si512(src.get_unchecked(4 * src_stride..).as_ptr() as *const _);
        let row5 = _mm512_loadu_si512(src.get_unchecked(5 * src_stride..).as_ptr() as *const _);
        let row6 = _mm512_loadu_si512(src.get_unchecked(6 * src_stride..).as_ptr() as *const _);
        let row7 = _mm512_loadu_si512(src.get_unchecked(7 * src_stride..).as_ptr() as *const _);

        let row8 = _mm512_loadu_si512(src.get_unchecked(8 * src_stride..).as_ptr() as *const _);
        let row9 = _mm512_loadu_si512(src.get_unchecked(9 * src_stride..).as_ptr() as *const _);
        let row10 = _mm512_loadu_si512(src.get_unchecked(10 * src_stride..).as_ptr() as *const _);
        let row11 = _mm512_loadu_si512(src.get_unchecked(11 * src_stride..).as_ptr() as *const _);

        let row12 = _mm512_loadu_si512(src.get_unchecked(12 * src_stride..).as_ptr() as *const _);
        let row13 = _mm512_loadu_si512(src.get_unchecked(13 * src_stride..).as_ptr() as *const _);
        let row14 = _mm512_loadu_si512(src.get_unchecked(14 * src_stride..).as_ptr() as *const _);
        let row15 = _mm512_loadu_si512(src.get_unchecked(15 * src_stride..).as_ptr() as *const _);

        let (v0, v1, v2, v3) = avx512_transpose_16x16_impl::<FLIP>(
            (row0, row1, row2, row3),
            (row4, row5, row6, row7),
            (row8, row9, row10, row11),
            (row12, row13, row14, row15),
        );

        if FLOP {
            _mm512_storeu_si512(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v0.0);
            _mm512_storeu_si512(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v0.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                v1.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                v1.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                v1.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                v1.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(8 * dst_stride..).as_mut_ptr() as *mut _,
                v2.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(9 * dst_stride..).as_mut_ptr() as *mut _,
                v2.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(10 * dst_stride..).as_mut_ptr() as *mut _,
                v2.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(11 * dst_stride..).as_mut_ptr() as *mut _,
                v2.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(12 * dst_stride..).as_mut_ptr() as *mut _,
                v3.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(13 * dst_stride..).as_mut_ptr() as *mut _,
                v3.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(14 * dst_stride..).as_mut_ptr() as *mut _,
                v3.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(15 * dst_stride..).as_mut_ptr() as *mut _,
                v3.3,
            );
        } else {
            _mm512_storeu_si512(
                dst.get_unchecked_mut(15 * dst_stride..).as_mut_ptr() as *mut _,
                v0.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(14 * dst_stride..).as_mut_ptr() as *mut _,
                v0.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(13 * dst_stride..).as_mut_ptr() as *mut _,
                v0.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(12 * dst_stride..).as_mut_ptr() as *mut _,
                v0.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(11 * dst_stride..).as_mut_ptr() as *mut _,
                v1.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(10 * dst_stride..).as_mut_ptr() as *mut _,
                v1.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(9 * dst_stride..).as_mut_ptr() as *mut _,
                v1.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(8 * dst_stride..).as_mut_ptr() as *mut _,
                v1.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(7 * dst_stride..).as_mut_ptr() as *mut _,
                v2.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(6 * dst_stride..).as_mut_ptr() as *mut _,
                v2.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(5 * dst_stride..).as_mut_ptr() as *mut _,
                v2.2,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(4 * dst_stride..).as_mut_ptr() as *mut _,
                v2.3,
            );

            _mm512_storeu_si512(
                dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr() as *mut _,
                v3.0,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr() as *mut _,
                v3.1,
            );
            _mm512_storeu_si512(
                dst.get_unchecked_mut(dst_stride..).as_mut_ptr() as *mut _,
                v3.2,
            );
            _mm512_storeu_si512(dst.get_unchecked_mut(0..).as_mut_ptr() as *mut _, v3.3);
        }
    }
}
