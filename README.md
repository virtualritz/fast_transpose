# Fast image transpose

Fast and simple image rotating in Rust with flipping and flopping in-place.

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

### Features

Turning off `unsafe` feature will activate `forbid unsafe` mode.

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
