# Fast image transpose

Fast and simple image transposing in Rust.

### Adding to project

```bash
cargo add fast_transpose
```

### Transpose RGB image

```rust
transpose_rgb(
    &img,
    &mut transposed,
    dimensions.0 as usize,
    dimensions.1 as usize,
    FlipMode::NoFlip,
    FlopMode::NoFlop,
)
.unwrap();
```

### Benchmarks

Tests are performed on the image 5000x4000

Transpose plane

```bash
cargo bench --bench plane --manifest-path ./app/Cargo.toml
```

|                           | Time(NEON) | Time(x86) | Time(Scalar) |
|---------------------------|:----------:|:---------:|:------------:|
| fast_transpose(plane u8)  |   1.95ms   |     -     |   11.31ms    |
| transpose(plane u8)       |   4.40ms   |     -     |    4.40ms    |
| image(plane u8)           |  16.84ms   |     -     |   16.84ms    |
| fast_transpose(plane u16) |   2.64ms   |     -     |   10.25ms    |
| transpose(plane u16)      |   5.14ms   |     -     |    5.34ms    |
| image(plane u16)          |  17.45ms   |     -     |   17.46ms    |
| fast_transpose(plane f32) |   9.05ms   |     -     |   18.08ms    |
| transpose(plane f32)      |   7.71ms   |     -     |    8.17ms    |
| image(plane f32)          |     -      |     -     |      -       |

Transpose RGB image

```bash
cargo bench --bench rgb --manifest-path ./app/Cargo.toml
```

|                           | Time(ARM) | Time(x86) | Time(Scalar) |
|---------------------------|:---------:|:---------:|:------------:|
| fast_transpose(plane u8)  |  10.55ms  |     -     |   20.05ms    |
| image(plane u8)           |  79.61ms  |     -     |   79.61ms    |
| fast_transpose(plane u16) |  18.52ms  |     -     |   21.68ms    |
| image(plane u16)          |  82.29ms  |     -     |   82.29ms    |
| fast_transpose(plane f32) |  18.06ms  |     -     |   29.27ms    |
| image(plane f32)          |  72.51ms  |     -     |   72.51ms    |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
