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
use crate::neon::utils::{vrev128_u32, vtrnq_s64_to_u32};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn neon_transpose_4x4_impl<const FLIP: bool>(v0: float32x4x4_t) -> float32x4x4_t {
    // Swap 32 bit elements. Goes from:
    // a0: 00 01 02 03
    // a1: 10 11 12 13
    // a2: 20 21 22 23
    // a3: 30 31 32 33
    // to:
    // b0.0: 00 10 02 12
    // b0.1: 01 11 03 13
    // b1.0: 20 30 22 32
    // b1.1: 21 31 23 33

    let b0 = vtrnq_u32(vreinterpretq_u32_f32(v0.0), vreinterpretq_u32_f32(v0.1));
    let b1 = vtrnq_u32(vreinterpretq_u32_f32(v0.2), vreinterpretq_u32_f32(v0.3));

    // Swap 64 bit elements resulting in:
    // c0.0: 00 10 20 30
    // c0.1: 02 12 22 32
    // c1.0: 01 11 21 31
    // c1.1: 03 13 23 33

    let c0 = vtrnq_s64_to_u32(b0.0, b1.0);
    let c1 = vtrnq_s64_to_u32(b0.1, b1.1);
    if FLIP {
        float32x4x4_t(
            vreinterpretq_f32_u32(vrev128_u32(c0.0)),
            vreinterpretq_f32_u32(vrev128_u32(c1.0)),
            vreinterpretq_f32_u32(vrev128_u32(c0.1)),
            vreinterpretq_f32_u32(vrev128_u32(c1.1)),
        )
    } else {
        float32x4x4_t(
            vreinterpretq_f32_u32(c0.0),
            vreinterpretq_f32_u32(c1.0),
            vreinterpretq_f32_u32(c0.1),
            vreinterpretq_f32_u32(c1.1),
        )
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1q_f32(&src[0]);
        let row1 = vld1q_f32(&src[src_stride]);
        let row2 = vld1q_f32(&src[2 * src_stride]);
        let row3 = vld1q_f32(&src[3 * src_stride]);

        let v0 = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0, row1, row2, row3));

        if FLOP {
            vst1q_f32(&mut dst[3 * dst_stride], v0.0);
            vst1q_f32(&mut dst[4 * dst_stride], v0.1);
            vst1q_f32(&mut dst[2 * dst_stride], v0.2);
            vst1q_f32(&mut dst[dst_stride], v0.3);
        } else {
            vst1q_f32(&mut dst[0], v0.0);
            vst1q_f32(&mut dst[dst_stride], v0.1);
            vst1q_f32(&mut dst[2 * dst_stride], v0.2);
            vst1q_f32(&mut dst[3 * dst_stride], v0.3);
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_2<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld2q_f32(&src[0]);
        let row1 = vld2q_f32(&src[src_stride]);
        let row2 = vld2q_f32(&src[2 * src_stride]);
        let row3 = vld2q_f32(&src[3 * src_stride]);

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));

        if FLOP {
            vst2q_f32(&mut dst[3 * dst_stride], float32x4x2_t(r.0, g.0));
            vst2q_f32(&mut dst[4 * dst_stride], float32x4x2_t(r.1, g.1));
            vst2q_f32(&mut dst[2 * dst_stride], float32x4x2_t(r.2, g.2));
            vst2q_f32(&mut dst[dst_stride], float32x4x2_t(r.3, g.3));
        } else {
            vst2q_f32(&mut dst[0], float32x4x2_t(r.0, g.0));
            vst2q_f32(&mut dst[dst_stride], float32x4x2_t(r.1, g.1));
            vst2q_f32(&mut dst[2 * dst_stride], float32x4x2_t(r.2, g.2));
            vst2q_f32(&mut dst[3 * dst_stride], float32x4x2_t(r.3, g.3));
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_3<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld3q_f32(&src[0]);
        let row1 = vld3q_f32(&src[src_stride]);
        let row2 = vld3q_f32(&src[2 * src_stride]);
        let row3 = vld3q_f32(&src[3 * src_stride]);

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));
        let b = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.2, row1.2, row2.2, row3.2));

        if FLOP {
            vst3q_f32(&mut dst[3 * dst_stride], float32x4x3_t(r.0, g.0, b.0));
            vst3q_f32(&mut dst[4 * dst_stride], float32x4x3_t(r.1, g.1, b.1));
            vst3q_f32(&mut dst[2 * dst_stride], float32x4x3_t(r.2, g.2, b.2));
            vst3q_f32(&mut dst[dst_stride], float32x4x3_t(r.3, g.3, b.3));
        } else {
            vst3q_f32(&mut dst[0], float32x4x3_t(r.0, g.0, b.0));
            vst3q_f32(&mut dst[dst_stride], float32x4x3_t(r.1, g.1, b.1));
            vst3q_f32(&mut dst[2 * dst_stride], float32x4x3_t(r.2, g.2, b.2));
            vst3q_f32(&mut dst[3 * dst_stride], float32x4x3_t(r.3, g.3, b.3));
        }
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_f32_intl_4<const FLOP: bool, const FLIP: bool>(
    src: &[f32],
    src_stride: usize,
    dst: &mut [f32],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld4q_f32(&src[0]);
        let row1 = vld4q_f32(&src[src_stride]);
        let row2 = vld4q_f32(&src[2 * src_stride]);
        let row3 = vld4q_f32(&src[3 * src_stride]);

        let r = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.0, row1.0, row2.0, row3.0));
        let g = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.1, row1.1, row2.1, row3.1));
        let b = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.2, row1.2, row2.2, row3.2));
        let a = neon_transpose_4x4_impl::<FLIP>(float32x4x4_t(row0.3, row1.3, row2.3, row3.3));

        if FLOP {
            vst4q_f32(&mut dst[3 * dst_stride], float32x4x4_t(r.0, g.0, b.0, a.0));
            vst4q_f32(&mut dst[4 * dst_stride], float32x4x4_t(r.1, g.1, b.1, a.1));
            vst4q_f32(&mut dst[2 * dst_stride], float32x4x4_t(r.2, g.2, b.2, a.2));
            vst4q_f32(&mut dst[dst_stride], float32x4x4_t(r.3, g.3, b.3, a.3));
        } else {
            vst4q_f32(&mut dst[0], float32x4x4_t(r.0, g.0, b.0, a.0));
            vst4q_f32(&mut dst[dst_stride], float32x4x4_t(r.1, g.1, b.1, a.1));
            vst4q_f32(&mut dst[2 * dst_stride], float32x4x4_t(r.2, g.2, b.2, a.2));
            vst4q_f32(&mut dst[3 * dst_stride], float32x4x4_t(r.3, g.3, b.3, a.3));
        }
    }
}
