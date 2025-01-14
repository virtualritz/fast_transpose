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
    flip_rgba, rotate180_rgba, transpose_rgba, transpose_rgba16, transpose_rgba8_chunked,
    transpose_rgba_f32, FlipMode, FlopMode,
};
use image::{DynamicImage, ImageReader};
use yuv_sys::{RotationMode_kRotate180, RotationMode_kRotate270, RotationMode_kRotate90};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/bench.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let img = img.to_rgba8();
    let dimensions = img.dimensions();
    let components = 4;

    c.bench_function("FT Rotate 90: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_rgba(
                &img,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.1 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    c.bench_function("FT Rotate 90: Rgba N u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_rgba8_chunked(
                &img,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.1 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::Flip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 90: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                img.as_ptr(),
                dimensions.0 as i32 * 4,
                transposed.as_mut_ptr(),
                dimensions.1 as i32 * 4,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate90,
            );
        });
    });

    c.bench_function("FT Rotate 180: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            rotate180_rgba(
                &img,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.0 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 180: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                img.as_ptr(),
                dimensions.0 as i32 * 4,
                transposed.as_mut_ptr(),
                dimensions.0 as i32 * 4,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate180,
            );
        });
    });

    c.bench_function("FT Rotate 270: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_rgba(
                &img,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.1 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::Flip,
                FlopMode::Flop,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Rotate 270: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                img.as_ptr(),
                dimensions.0 as i32 * 4,
                transposed.as_mut_ptr(),
                dimensions.1 as i32 * 4,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate270,
            );
        });
    });

    c.bench_function("FT Mirror: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            flip_rgba(
                &img,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.0 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
            )
            .unwrap();
        });
    });

    c.bench_function("Libyuv Mirror: Rgba u8", |b| {
        let mut transposed = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBMirror(
                img.as_ptr(),
                dimensions.0 as i32 * 4,
                transposed.as_mut_ptr(),
                dimensions.0 as i32 * 4,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        });
    });

    let dyn_image = DynamicImage::ImageRgba8(img);

    c.bench_function("Image Transpose: Rgba u8", |b| {
        b.iter(|| {
            _ = dyn_image.rotate90();
        });
    });

    let img16 = dyn_image.to_rgba16();

    c.bench_function("Fast transpose: Rgba u16", |b| {
        let mut transposed = vec![0u16; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_rgba16(
                &img16,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.1 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    let dyn_image16 = DynamicImage::ImageRgba16(img16);

    c.bench_function("Image Transpose: Rgb u16", |b| {
        b.iter(|| {
            _ = dyn_image16.rotate90();
        });
    });

    let image_f32 = dyn_image16.to_rgba32f();

    c.bench_function("Fast transpose: Rgba f32", |b| {
        let mut transposed = vec![0.; dimensions.0 as usize * dimensions.1 as usize * components];
        b.iter(|| {
            transpose_rgba_f32(
                &image_f32,
                dimensions.0 as usize * 4,
                &mut transposed,
                dimensions.1 as usize * 4,
                dimensions.0 as usize,
                dimensions.1 as usize,
                FlipMode::NoFlip,
                FlopMode::NoFlop,
            )
            .unwrap();
        });
    });

    let dyn_image_f32 = DynamicImage::ImageRgba32F(image_f32);

    c.bench_function("Image Transpose: Rgba f32", |b| {
        b.iter(|| {
            _ = dyn_image_f32.rotate90();
        });
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
