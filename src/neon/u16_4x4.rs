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

use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn neon_transpose_u16_4x4_impl<const FLIP: bool>(
    v0: uint16x4x4_t,
) -> uint16x4x4_t {
    // Input:
    // 00 01 02 03
    // 10 11 12 13
    // 20 21 22 23
    // 30 31 32 33

    // b:
    // 00 10 02 12
    // 01 11 03 13
    let b = vtrn_u16(v0.0, v0.1);
    // c:
    // 20 30 22 32
    // 21 31 23 33
    let c = vtrn_u16(v0.2, v0.3);
    // d:
    // 00 10 20 30
    // 02 12 22 32
    let d = vtrn_u32(vreinterpret_u32_u16(b.0), vreinterpret_u32_u16(c.0));
    // e:
    // 01 11 21 31
    // 03 13 23 33
    let e = vtrn_u32(vreinterpret_u32_u16(b.1), vreinterpret_u32_u16(c.1));

    // Output:
    // 00 10 20 30
    // 01 11 21 31
    // 02 12 22 32
    // 03 13 23 33

    if FLIP {
        uint16x4x4_t(
            vrev64_u16(vreinterpret_u16_u32(d.0)),
            vrev64_u16(vreinterpret_u16_u32(e.0)),
            vrev64_u16(vreinterpret_u16_u32(d.1)),
            vrev64_u16(vreinterpret_u16_u32(e.1)),
        )
    } else {
        uint16x4x4_t(
            vreinterpret_u16_u32(d.0),
            vreinterpret_u16_u32(e.0),
            vreinterpret_u16_u32(d.1),
            vreinterpret_u16_u32(e.1),
        )
    }
}

#[inline]
pub(crate) fn neon_transpose_4x4_u16<const FLOP: bool, const FLIP: bool>(
    src: &[u16],
    src_stride: usize,
    dst: &mut [u16],
    dst_stride: usize,
) {
    unsafe {
        let row0 = vld1_u16(src.get_unchecked(0..).as_ptr());
        let row1 = vld1_u16(src.get_unchecked(src_stride..).as_ptr());
        let row2 = vld1_u16(src.get_unchecked(2 * src_stride..).as_ptr());
        let row3 = vld1_u16(src.get_unchecked(3 * src_stride..).as_ptr());

        let v0 = neon_transpose_u16_4x4_impl::<FLIP>(uint16x4x4_t(row0, row1, row2, row3));

        if FLOP {
            vst1_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.0);
            vst1_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.1);
            vst1_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.2);
            vst1_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.3);
        } else {
            vst1_u16(dst.get_unchecked_mut(3 * dst_stride..).as_mut_ptr(), v0.0);
            vst1_u16(dst.get_unchecked_mut(2 * dst_stride..).as_mut_ptr(), v0.1);
            vst1_u16(dst.get_unchecked_mut(dst_stride..).as_mut_ptr(), v0.2);
            vst1_u16(dst.get_unchecked_mut(0..).as_mut_ptr(), v0.3);
        }
    }
}
