# Algebrix

Math for game engines: vectors, matrices, quaternions, and geometry. Uses SIMD on x86_64 and aarch64 when the `simd` feature is on; otherwise scalar. Originally split out from a larger codebase and kept as a separate crate.

## What's in it

- **Vectors**: `Vec2`, `Vec3`, `Vec4` (float); `IVec2/3/4`, `UVec2/3/4`, `BVec2/3/4`; `DVec3` for double. Swizzles supported. Float vectors also have `normalize_fast()` and `length_fast()` (rsqrt-based) when you care more about speed than precision.
- **Matrices**: `Mat2`, `Mat3`, `Mat4`, and `DMat4` for double 4x4.
- **Quaternions**: `Quat` and `DQuat` for rotations.
- **Transforms**: `Affine2` and `Affine3` for combined rotation, scale, and translation.
- **Geometry**: `Plane`, `Ray`, `Aabb2`, `Aabb3`, `Frustum` for culling and raycasting.
- **Angles**: `Rad`, `Deg`, and `EulerRot` for type-safe radians/degrees.

## Usage

```rust
use algebrix::{Vec3, Mat4, Quat, Rad, Deg};

let v = Vec3::new(1.0, 2.0, 3.0);
let normalized = v.normalize();
let angle = Rad::from(Deg::new(90.0));
```

## Feature flags

- `std` – standard library (default).
- `simd` – SIMD on; backend chosen from target (default).
- `simd-x86` – force x86_64 (SSE/AVX).
- `simd-arm` – force ARM (NEON).
- `scalar-math` – no SIMD.
- `glam-compatibility` – `From`/`Into` between algebrix and glam for `Vec2`/`Vec3`/`Vec4`, `Mat2`/`Mat3`/`Mat4`, `Quat`.
- `nalgebra-compatibility` – `From`/`Into` between algebrix and nalgebra for `Vector2`/`3`/`4`, `Matrix2`/`3`/`4`, `Quaternion`.

Default features: `["std", "simd"]`. Use `default-features = false` and opt in to what you need if you want a smaller dependency set or a specific SIMD backend.
