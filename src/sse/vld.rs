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

use crate::sse::packing::{
    _mm_deinterleave_la, _mm_deinterleave_rgb, _mm_deinterleave_rgba, _mm_interleave_la,
    _mm_interleave_rgb, _mm_interleave_rgba,
};
use crate::sse::packing_16::{
    _mm_deinterleave_la_epi16, _mm_deinterleave_rgb_epi16, _mm_deinterleave_rgba_epi16,
    _mm_interleave_la_epi16, _mm_interleave_rgb_epi16, _mm_interleave_rgba_epi16,
};
use crate::sse::packing_32::{
    _mm_deinterleave_la_ps, _mm_deinterleave_rgb_ps, _mm_deinterleave_rgba_ps,
    _mm_interleave_la_ps, _mm_interleave_rgb_ps, _mm_interleave_rgba_ps,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgb_half(ptr: *const u8) -> (__m128i, __m128i, __m128i) {
    let row0 = _mm_loadu_si128(ptr as *const __m128i);
    let row1 = _mm_loadu_si64(ptr.add(16));
    _mm_deinterleave_rgb(row0, row1, _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgb_half(ptr: *mut u8, v: (__m128i, __m128i, __m128i)) {
    let (v0, v1, _) = _mm_interleave_rgb(v.0, v.1, v.2);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, ptr.add(16), 8);
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgba_half(
    ptr: *const u8,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let row0 = _mm_loadu_si128(ptr as *const __m128i);
    let row1 = _mm_loadu_si128(ptr.add(16) as *const __m128i);
    _mm_deinterleave_rgba(row0, row1, _mm_setzero_si128(), _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgba_half(
    ptr: *mut u8,
    v: (__m128i, __m128i, __m128i, __m128i),
) {
    let (v0, v1, _, _) = _mm_interleave_rgba(v.0, v.1, v.2, v.3);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v1);
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_la_half(ptr: *const u8) -> (__m128i, __m128i) {
    let row0 = _mm_loadu_si128(ptr as *const __m128i);
    _mm_deinterleave_la(row0, _mm_setzero_si128())
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_la_half(ptr: *mut u8, v: (__m128i, __m128i)) {
    let (v0, _) = _mm_interleave_la(v.0, v.1);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_pack_x4(ptr: *mut u8, v: (__m128i, __m128i, __m128i, __m128i)) {
    _mm_storeu_si128(ptr as *mut __m128i, v.0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v.1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, v.2);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, v.3);
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_pack_x3(ptr: *mut u8, v: (__m128i, __m128i, __m128i)) {
    _mm_storeu_si128(ptr as *mut __m128i, v.0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v.1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, v.2);
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_pack_x2(ptr: *mut u8, v: (__m128i, __m128i)) {
    _mm_storeu_si128(ptr as *mut __m128i, v.0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v.1);
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgba16(
    mem: &[u16],
) -> (__m128i, __m128i, __m128i, __m128i) {
    let row0 = _mm_loadu_si128(mem.as_ptr() as *const __m128i);
    let row1 = _mm_loadu_si128(mem.get_unchecked(8..).as_ptr() as *const __m128i);
    let row2 = _mm_loadu_si128(mem.get_unchecked(16..).as_ptr() as *const __m128i);
    let row3 = _mm_loadu_si128(mem.get_unchecked(24..).as_ptr() as *const __m128i);
    _mm_deinterleave_rgba_epi16(row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgba16(
    mem: &mut [u16],
    v: (__m128i, __m128i, __m128i, __m128i),
) {
    let (v0, v1, v2, v3) = _mm_interleave_rgba_epi16(v.0, v.1, v.2, v.3);
    _mm_store_pack_x4(mem.as_ptr() as *mut u8, (v0, v1, v2, v3));
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgb16(mem: &[u16]) -> (__m128i, __m128i, __m128i) {
    let row0 = _mm_loadu_si128(mem.as_ptr() as *const __m128i);
    let row1 = _mm_loadu_si128(mem.get_unchecked(8..).as_ptr() as *const __m128i);
    let row2 = _mm_loadu_si128(mem.get_unchecked(16..).as_ptr() as *const __m128i);
    _mm_deinterleave_rgb_epi16(row0, row1, row2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgb16(mem: &mut [u16], v: (__m128i, __m128i, __m128i)) {
    let (v0, v1, v2) = _mm_interleave_rgb_epi16(v.0, v.1, v.2);
    _mm_store_pack_x3(mem.as_ptr() as *mut u8, (v0, v1, v2));
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_la16(mem: &[u16]) -> (__m128i, __m128i) {
    let row0 = _mm_loadu_si128(mem.as_ptr() as *const __m128i);
    let row1 = _mm_loadu_si128(mem.get_unchecked(8..).as_ptr() as *const __m128i);
    _mm_deinterleave_la_epi16(row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_la16(mem: &mut [u16], v: (__m128i, __m128i)) {
    let (v0, v1) = _mm_interleave_la_epi16(v.0, v.1);
    _mm_store_pack_x2(mem.as_ptr() as *mut u8, (v0, v1));
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_la_ps(mem: &[f32]) -> (__m128, __m128) {
    let row0 = _mm_loadu_ps(mem.as_ptr());
    let row1 = _mm_loadu_ps(mem.get_unchecked(4..).as_ptr());
    _mm_deinterleave_la_ps(row0, row1)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_la_ps(mem: &mut [f32], v: (__m128, __m128)) {
    let (v0, v1) = _mm_interleave_la_ps(v.0, v.1);
    _mm_storeu_ps(mem.as_mut_ptr(), v0);
    _mm_storeu_ps(mem.get_unchecked_mut(4..).as_mut_ptr(), v1);
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgb_ps(mem: &[f32]) -> (__m128, __m128, __m128) {
    let row0 = _mm_loadu_ps(mem.as_ptr());
    let row1 = _mm_loadu_ps(mem.get_unchecked(4..).as_ptr());
    let row2 = _mm_loadu_ps(mem.get_unchecked(8..).as_ptr());
    _mm_deinterleave_rgb_ps(row0, row1, row2)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgb_ps(mem: &mut [f32], v: (__m128, __m128, __m128)) {
    let (v0, v1, v2) = _mm_interleave_rgb_ps(v.0, v.1, v.2);
    _mm_storeu_ps(mem.as_mut_ptr(), v0);
    _mm_storeu_ps(mem.get_unchecked_mut(4..).as_mut_ptr(), v1);
    _mm_storeu_ps(mem.get_unchecked_mut(8..).as_mut_ptr(), v2);
}

#[inline(always)]
pub(crate) unsafe fn _mm_load_deinterleave_rgba_ps(
    mem: &[f32],
) -> (__m128, __m128, __m128, __m128) {
    let row0 = _mm_loadu_ps(mem.as_ptr());
    let row1 = _mm_loadu_ps(mem.get_unchecked(4..).as_ptr());
    let row2 = _mm_loadu_ps(mem.get_unchecked(8..).as_ptr());
    let row3 = _mm_loadu_ps(mem.get_unchecked(12..).as_ptr());
    _mm_deinterleave_rgba_ps(row0, row1, row2, row3)
}

#[inline(always)]
pub(crate) unsafe fn _mm_store_interleave_rgba_ps(
    mem: &mut [f32],
    v: (__m128, __m128, __m128, __m128),
) {
    let (v0, v1, v2, v3) = _mm_interleave_rgba_ps(v.0, v.1, v.2, v.3);
    _mm_storeu_ps(mem.as_mut_ptr(), v0);
    _mm_storeu_ps(mem.get_unchecked_mut(4..).as_mut_ptr(), v1);
    _mm_storeu_ps(mem.get_unchecked_mut(8..).as_mut_ptr(), v2);
    _mm_storeu_ps(mem.get_unchecked_mut(12..).as_mut_ptr(), v3);
}
