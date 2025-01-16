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

use crate::neon::utils::{vrev128q_u64, xvld1q_u16_u64, xvst1q_u16_u64};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_transpose_u64_2x2_impl<const FLIP: bool>(
    v0: uint64x2x2_t,
) -> uint64x2x2_t {
    let l = vtrn1q_u64(v0.0, v0.1);
    let h = vtrn2q_u64(v0.0, v0.1);

    if FLIP {
        uint64x2x2_t(vrev128q_u64(l), vrev128q_u64(h))
    } else {
        uint64x2x2_t(l, h)
    }
}

#[inline]
pub(crate) fn neon_transpose_u16x4_2x2<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = xvld1q_u16_u64(src.get_unchecked(0..).as_ptr());
        let row1 = xvld1q_u16_u64(src.get_unchecked(src_stride..).as_ptr());

        let v0 = neon_transpose_u64_2x2_impl::<FLIP>(uint64x2x2_t(row0, row1));

        if FLOP {
            xvst1q_u16_u64(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
        } else {
            xvst1q_u16_u64(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.0);
            xvst1q_u16_u64(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.1);
        }
    }
}
