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

use crate::neon::utils::{vrev128_u32, vtrnq_s64_to_u32, xvld1q_u8_u32, xvst1q_u8_u32};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_transpose_4x4_impl<const FLIP: bool>(v0: uint32x4x4_t) -> uint32x4x4_t {
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

    let b0 = vtrnq_u32(v0.0, v0.1);
    let b1 = vtrnq_u32(v0.2, v0.3);

    // Swap 64 bit elements resulting in:
    // c0.0: 00 10 20 30
    // c0.1: 02 12 22 32
    // c1.0: 01 11 21 31
    // c1.1: 03 13 23 33

    let c0 = vtrnq_s64_to_u32(b0.0, b1.0);
    let c1 = vtrnq_s64_to_u32(b0.1, b1.1);

    if FLIP {
        uint32x4x4_t(
            vrev128_u32(c0.0),
            vrev128_u32(c1.0),
            vrev128_u32(c0.1),
            vrev128_u32(c1.1),
        )
    } else {
        uint32x4x4_t(c0.0, c1.0, c0.1, c1.1)
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_u8x4<const FLOP: bool, const FLIP: bool>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
) {
    unsafe {
        let row0 = xvld1q_u8_u32(src.get_unchecked(0..).as_ptr());
        let row1 = xvld1q_u8_u32(src.get_unchecked(src_stride..).as_ptr());
        let row2 = xvld1q_u8_u32(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = xvld1q_u8_u32(src.get_unchecked(3 * src_stride..).as_ptr());

        let v0 = neon_transpose_4x4_impl::<FLIP>(uint32x4x4_t(row0, row1, row2, row3));

        if FLOP {
            xvst1q_u8_u32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
        } else {
            xvst1q_u8_u32(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.0);
            xvst1q_u8_u32(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.1);
            xvst1q_u8_u32(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.2);
            xvst1q_u8_u32(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.3);
        }
    }
}
