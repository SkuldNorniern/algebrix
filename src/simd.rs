//! SIMD optimizations for math operations
//!
//! This module provides SIMD-accelerated implementations that are
//! conditionally compiled based on platform and feature flags.
//!
//! ## Features
//!
//! - `simd`: Enable SIMD optimizations (auto-detects platform)
//! - `simd-x86`: Force x86_64 SIMD (SSE/AVX) - automatically enabled on x86_64 with `simd`
//! - `simd-arm`: Force ARM SIMD (NEON) - automatically enabled on aarch64 with `simd`
//!
//! ## Cross-Platform Support
//!
//! - **x86_64**: Uses SSE2 intrinsics (SSE2 is guaranteed on x86_64, no runtime detection)
//! - **aarch64**: Uses NEON intrinsics (NEON is effectively always available on aarch64)
//! - **Other platforms**: Automatically falls back to scalar implementations
//!
//! All SIMD functions automatically fall back to scalar code if:
//! - The platform doesn't support SIMD
//! - The `simd` feature is not enabled

pub mod scalar;

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
pub mod x86_64_impl;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
pub mod aarch64_impl;

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
pub use x86_64_impl::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
pub use aarch64_impl::*;

#[cfg(not(any(
    all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
    all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
)))]
pub use scalar::*;
