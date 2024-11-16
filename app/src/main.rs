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
use fast_transpose::{
    transpose_plane_f32, transpose_rgb, transpose_rgb16, transpose_rgb_f32, FlipMode, FlopMode,
};
use image::{ColorType, DynamicImage, GenericImageView, ImageReader};
use std::time::Instant;

fn main() {
    let img = ImageReader::open("assets/sonderland.jpg")
        .unwrap()
        .decode()
        .unwrap();

    let dimensions = img.dimensions();
    let components = if img.color() == ColorType::Rgb8 { 3 } else { 4 };
    let img = img.to_rgb32f();

    let mut transposed = vec![0.; dimensions.0 as usize * dimensions.1 as usize * components];

    let start = Instant::now();

    transpose::transpose(
        &img,
        &mut transposed,
        dimensions.0 as usize * 3,
        dimensions.1 as usize,
    );

    println!("Transpose exec time {:?}", start.elapsed());

    let start = Instant::now();

    transpose_plane_f32(
        &img,
        &mut transposed,
        dimensions.0 as usize * 3,
        dimensions.1 as usize,
        FlipMode::NoFlip,
        FlopMode::NoFlop,
    )
    .unwrap();

    println!("Exec time {:?}", start.elapsed());

    let start = Instant::now();

    let dyn_img = DynamicImage::ImageRgb32F(img);
    _ = dyn_img.rotate90();

    println!("Exec time {:?}", start.elapsed());

    let transposed = transposed
        .iter()
        .map(|&x| (x * 255.) as u8)
        .collect::<Vec<_>>();

    if components == 3 {
        image::save_buffer(
            "transposed.jpg",
            &transposed,
            dimensions.1,
            dimensions.0,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    } else {
        image::save_buffer(
            "transposed.png",
            &transposed,
            dimensions.1,
            dimensions.0,
            if components == 3 {
                image::ExtendedColorType::Rgb8
            } else {
                image::ExtendedColorType::Rgba8
            },
        )
        .unwrap();
    }
}
