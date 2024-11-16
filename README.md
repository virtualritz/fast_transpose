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

|                           | Time(ARM) | Time(x86) | 
|---------------------------|:---------:|:---------:| 
| fast_transpose(plane u8)  |  2.01ms   |     -     | 
| transpose(plane u8)       |  4.40ms   |     -     | 
| image(plane u8)           |  16.84ms  |     -     | 
| fast_transpose(plane u16) |  2.77ms   |     -     | 
| transpose(plane u16)      |  5.34ms   |     -     | 
| image(plane u16)          |  17.46ms  |     -     | 
| fast_transpose(plane f32) |  18.92ms  |     -     | 
| transpose(plane f32)      |  7.71ms   |     -     | 
| image(plane f32)          |     -     |     -     | 

Transpose RGB image

```bash
cargo bench --bench rgb --manifest-path ./app/Cargo.toml
```

|                           | Time(ARM) | Time(x86) | 
|---------------------------|:---------:|:---------:| 
| fast_transpose(plane u8)  |  10.75ms  |     -     | 
| image(plane u8)           |  79.61ms  |     -     | 
| fast_transpose(plane u16) |  18.52ms  |     -     | 
| image(plane u16)          |  82.29ms  |     -     | 
| fast_transpose(plane f32) |  18.06ms  |     -     | 
| image(plane f32)          |  72.51ms  |     -     |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
