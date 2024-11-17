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
#![forbid(unsafe_code)]

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn common_process<
    V: Copy,
    const FLOP: bool,
    const FLIP: bool,
    const PIXEL_STRIDE: usize,
>(
    src_row: &[V],
    row_size: usize,
    target: &mut [V],
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    take_rows: usize,
) {
    let dst_stride = height * PIXEL_STRIDE;
    
    for y in start_y..(start_y + take_rows) {
        let s_start_y = if FLIP { height - 1 - y } else { y };
        let start_row_offset_x = s_start_y * row_size + PIXEL_STRIDE * start_x;
        let end_row = (s_start_y + 1) * row_size;
        let start_row = &src_row[start_row_offset_x..end_row];

        let target = &mut target[y * PIXEL_STRIDE..];
        for (x, src) in start_row.chunks_exact(PIXEL_STRIDE).enumerate() {
            let offset = dst_stride
                * if FLOP {
                    x + start_x
                } else {
                    width - 1 - (x + start_x)
                };
            let dst = &mut target[offset..(offset + PIXEL_STRIDE)];
            if PIXEL_STRIDE == 1 {
                dst[0] = src[0];
            } else if PIXEL_STRIDE == 2 {
                dst[0] = src[0];
                dst[1] = src[1];
            } else if PIXEL_STRIDE == 3 {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
            } else if PIXEL_STRIDE == 4 {
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
            }
        }
    }
}
