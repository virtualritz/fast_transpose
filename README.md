# Fast Image Transpose

High-performance image transposition library with SIMD acceleration for multiple CPU architectures.
Supports flipping, flopping, and rotation operations on images with various pixel formats.

## Features

- **Fast transposition**: Optimized algorithms for 90°, 180°, and 270° rotations
- **Multiple data types**: Support for 8-bit, 16-bit, and 32-bit float pixels
- **Arbitrary channels**: Works with grayscale, RGB, RGBA, and custom channel counts
- **SIMD optimizations**: Architecture-specific implementations for x86 (SSE/AVX) and ARM (NEON)
- **In-place operations**: Memory-efficient transformations where possible
- **Safe mode**: Optional pure-Rust implementation without unsafe code

## Installation

```bash
cargo add fast_transpose
```

## Usage

### Transpose RGB Image

```rust
use fast_transpose::{transpose_rgb, FlipMode, FlopMode};

let input = vec![0u8; width * height * 3];
let mut output = vec![0u8; height * width * 3];

// Transpose (90° clockwise rotation) without additional flipping
transpose_rgb(
    &input,
    &mut output,
    width,
    height,
    FlipMode::NoFlip,
    FlopMode::NoFlop,
)
.unwrap();
```

### Rotation Modes

- **90° clockwise**: `transpose` with `FlipMode::NoFlip`, `FlopMode::NoFlop`
- **90° counter-clockwise (270° clockwise)**: `transpose` with `FlipMode::Flip`, `FlopMode::Flop`
- **180°**: Use `rotate180_*` functions or `flip` + `flop`
- **Horizontal mirror**: Use `flip_*` functions
- **Vertical mirror**: Use `flop_*` functions

### Arbitrary Channel Support

For images with non-standard channel counts (e.g., 5-channel scientific imagery):

```rust
use fast_transpose::{transpose_arbitrary_grouped, FlipMode, FlopMode};

// Transpose a 5-channel image
let input = vec![0u8; width * height * 5];
let mut output = vec![0u8; height * width * 5];

transpose_arbitrary_grouped::<u8, 5>(
    &input,
    width * 5,  // input stride
    &mut output,
    height * 5, // output stride
    width,
    height,
    FlipMode::NoFlip,
    FlopMode::NoFlop,
).unwrap();
```

## Cargo Features

- `unsafe` (default): Enables SIMD optimizations. Disabling activates `forbid(unsafe_code)`
- `sse` (default): SSE optimizations for x86
- `avx` (default): AVX optimizations for x86_64
- `neon` (default): NEON optimizations for ARM
- `nightly_avx512`: AVX-512 support (requires nightly Rust)

### Building without unsafe code

```bash
cargo build --no-default-features
```

## Performance

This library prioritizes performance through:
- SIMD vectorization for all common operations
- Cache-friendly memory access patterns
- Specialized implementations for each data type and channel count
- Zero-cost abstractions

Benchmarks are available in the `app/benches` directory.

## License

This project is licensed under either of:

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE-APACHE](LICENSE-APACHE.md))

at your option.
