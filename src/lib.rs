//! Math for game engines: vectors, matrices, quaternions, geometry.
//!
//! SIMD is used when the `simd` feature is on and the target is x86_64 or aarch64;
//! otherwise scalar. Column vectors; multiply by matrices on the left. See README for feature flags.
//!
//! # Modules
//!
//! Vectors: `vec2`, `vec3`, `vec4`, `ivec2`–`ivec4`, `uvec2`–`uvec4`, `bvec`, `dvec3`.
//! Matrices: `mat2`, `mat3`, `mat4`, `dmat4`. Quaternions: `quat`, `dquat`.
//! Transforms: `affine2`, `affine3`. Geometry: `geometry` (plane, ray, AABB, frustum).
//! Angles: `angle` (Rad, Deg, EulerRot). Utilities: `utils`. Optional interop: `compat`.
//!
//! # Examples
//!
//! Vectors and normalization:
//!
//! ```rust
//! use algebrix::{Vec3, Rad, Deg};
//!
//! let v = Vec3::new(3.0, 4.0, 0.0);
//! assert!((v.length() - 5.0).abs() < 1e-5);
//! let u = v.normalize();
//! let half_turn = Rad::from(Deg::new(90.0));
//! ```
//!
//! Transform and quaternion:
//!
//! ```rust
//! use algebrix::{Mat4, Quat, Vec3};
//!
//! let t = Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0));
//! let p = t.transform_point3(Vec3::ZERO);
//! assert_eq!(p, Vec3::new(1.0, 0.0, 0.0));
//!
//! let q = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2);
//! let x_rotated = q * Vec3::X;
//! assert!((x_rotated - Vec3::Y).length() < 1e-5);
//! ```

pub mod bvec;
pub mod ivec2;
pub mod ivec3;
pub mod ivec4;
pub mod swizzles;
pub mod uvec2;
pub mod uvec3;
pub mod uvec4;
pub mod vec2;
pub mod vec3;
pub mod vec4;

pub mod dmat4;
pub mod mat2;
pub mod mat3;
pub mod mat4;

pub mod dquat;
pub mod quat;

pub mod dvec3;

pub mod affine2;
pub mod affine3;

pub mod geometry;

pub mod angle;

pub mod utils;

pub mod compat;

pub use bvec::{BVec2, BVec3, BVec4};
pub use ivec2::IVec2;
pub use ivec3::IVec3;
pub use ivec4::IVec4;
pub use swizzles::{Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};
pub use uvec2::UVec2;
pub use uvec3::UVec3;
pub use uvec4::UVec4;
pub use vec2::Vec2;
pub use vec3::Vec3;
pub use vec4::Vec4;

pub use dmat4::DMat4;
pub use mat2::Mat2;
pub use mat3::Mat3;
pub use mat4::Mat4;

pub use dquat::DQuat;
pub use quat::Quat;

pub use dvec3::DVec3;

pub use affine2::Affine2;
pub use affine3::Affine3;

pub use geometry::{Aabb2, Aabb3, Frustum, Plane, Ray};

pub use angle::{Deg, EulerRot, Rad};

pub use utils::*;
