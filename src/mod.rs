//! Custom math library for the game engine
//!
//! This module provides high-performance custom implementations of vector, matrix, and quaternion types.
//! The library is fully optimized with SIMD support, FMA utilization, and fast variants for game code.
//! It has replaced glam as the primary math library throughout the codebase.
//!
//! ## SIMD Support
//!
//! SIMD optimizations are available via feature flags:
//! - `simd`: Enable SIMD optimizations (auto-detects platform)
//! - `simd-x86`: Force x86_64 SIMD (SSE/AVX)
//! - `simd-arm`: Force ARM SIMD (NEON)
//!
//! SIMD implementations automatically fall back to scalar code on unsupported platforms.

pub mod simd;

pub mod dmat4;
pub mod dquat;
pub mod dvec3;
pub mod mat3;
pub mod mat4;
pub mod quat;
pub mod utils;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub use dmat4::DMat4;
pub use dquat::DQuat;
pub use dvec3::DVec3;
pub use mat3::Mat3;
pub use mat4::Mat4;
pub use quat::Quat;
pub use utils::*;
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;
