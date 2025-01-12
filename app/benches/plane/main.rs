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

use criterion::{criterion_group, criterion_main, Criterion};
use fast_transpose::{
    flip_plane, rotate180_plane, transpose_plane, transpose_plane16, transpose_plane_f32, FlipMode,
    FlopMode,
};
use image::{DynamicImage, ImageReader};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/sonderland.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_luma8();
    let dimensions = img.dimensions();
    let components = 1;

    c.bench_function("FT Rotate 90: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_plane(
                &img,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 90: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_RotatePlane90(
                img.as_ptr(),
                dimensions.0 as i32,
                transposed.as_mut_ptr(),
                dimensions.1 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        });
    });

    c.bench_function("FT Rotate 180: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            rotate180_plane(
                &img,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.0 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 180: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_RotatePlane180(
                img.as_ptr(),
                dimensions.0 as i32,
                transposed.as_mut_ptr(),
                dimensions.0 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        });
    });

    c.bench_function("FT Rotate 270: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_plane(
                &img,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::Flip,
                FlopMode::Flop,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 270: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_RotatePlane270(
                img.as_ptr(),
                dimensions.0 as i32,
                transposed.as_mut_ptr(),
                dimensions.1 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        });
    });

    c.bench_function("FT Mirror: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            flip_plane(
                &img,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.0 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Mirror: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_MirrorPlane(
                img.as_ptr(),
                dimensions.0 as i32,
                transposed.as_mut_ptr(),
                dimensions.0 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        });
    });

    c.bench_function("Transpose: Plane u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose::transpose(
                &img,
                &mut transposed,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
        });
    });

    let dyn_image = DynamicImage::ImageLuma8(img);

    c.bench_function("Image Transpose: Plane u8", |b| {
        b.iter(|| {
            _ = dyn_image.rotate90();
        });
    });

    let img16 = dyn_image.to_luma16();

    c.bench_function("Fast transpose: Plane u16", |b| {
        let mut transposed = vec![0u16; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_plane16(
                &img16,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    c.bench_function("Transpose: Plane u16", |b| {
        let mut transposed = vec![0u16; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose::transpose(
                &img16,
                &mut transposed,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
        });
    });

    let dyn_image16 = DynamicImage::ImageLuma16(img16);

    c.bench_function("Image Transpose: Plane u16", |b| {
        b.iter(|| {
            _ = dyn_image16.rotate90();
        });
    });

    let data = dyn_image16
        .to_luma16()
        .iter()
        .map(|&x| x as f32)
        .collect::<Vec<_>>();

    c.bench_function("Fast transpose: Plane f32", |b| {
        let mut transposed = vec![0.; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_plane_f32(
                &data,
                dimensions.0 as usize,
                &mut transposed,
                dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    c.bench_function("Transpose: Plane f32", |b| {
        let mut transposed = vec![0.; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose::transpose(
                &data,
                &mut transposed,
                dimensions.0 as usize,
                dimensions.1 as usize,
            );
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
