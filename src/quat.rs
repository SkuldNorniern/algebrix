//! Unit quaternion for 3D rotation. Layout (x, y, z, w); multiply quats for combined rotation.
//!
//! Use [`from_axis_angle`](Quat::from_axis_angle) for axis+angle, [`from_mat3`](Quat::from_mat3) from a matrix,
//! [`slerp`](Quat::slerp) for interpolation. Rotate a vector with `quat * vec` or [`mul_vec3`](Quat::mul_vec3).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Quat, Vec3};
//!
//! let axis = Vec3::Z;
//! let q = Quat::from_axis_angle(axis, std::f32::consts::FRAC_PI_2);
//! let x = Vec3::X;
//! let y = q * x;
//! assert!((y - Vec3::Y).length() < 1e-5);
//!
//! let a = Quat::IDENTITY;
//! let b = Quat::from_axis_angle(Vec3::Y, 0.5);
//! let mid = a.slerp(b, 0.5);
//! assert!(mid.w > 0.9);
//! ```

use crate::vec4::XYZW;
use crate::{Vec3, Vec4, utils};

/// Stored as a [`Vec4`], so with the SIMD feature it lives in a SIMD register.
/// `x`/`y`/`z`/`w` are accessible through `Deref`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Quat(pub(crate) Vec4);

impl std::ops::Deref for Quat {
    type Target = XYZW;
    #[inline(always)]
    fn deref(&self) -> &XYZW {
        // Quat is repr(transparent) over Vec4; XYZW matches its layout.
        unsafe { &*(self as *const Self as *const XYZW) }
    }
}

impl std::ops::DerefMut for Quat {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZW {
        unsafe { &mut *(self as *mut Self as *mut XYZW) }
    }
}

impl std::fmt::Debug for Quat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Quat")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .field("w", &self.w)
            .finish()
    }
}

impl PartialEq for Quat {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Quat {
    pub const IDENTITY: Quat = Quat::new(0.0, 0.0, 0.0, 1.0);

    #[inline]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }

    #[inline]
    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self(Vec4::new(x, y, z, w))
    }

    /// Rotation around `axis` (will be normalized) by `angle` radians. Right-hand rule.
    ///
    /// # Example
    ///
    /// ```rust
    /// use algebrix::{Quat, Vec3};
    /// let q = Quat::from_axis_angle(Vec3::Z, std::f32::consts::FRAC_PI_2);
    /// let v = q * Vec3::X;
    /// assert!((v - Vec3::Y).length() < 1e-5);
    /// ```
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let half_angle = angle * 0.5;
        let s = half_angle.sin();
        let c = half_angle.cos();
        let normalized_axis = axis.normalize();
        Self::new(normalized_axis.x * s, normalized_axis.y * s, normalized_axis.z * s, c)
    }

    /// Rotation around +X by `angle` radians.
    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(s, 0.0, 0.0, c)
    }

    /// Rotation around +Y by `angle` radians.
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(0.0, s, 0.0, c)
    }

    /// Rotation around +Z by `angle` radians.
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        Self::new(0.0, 0.0, s, c)
    }

    /// Rotation from a rotation vector: direction is the axis, length is the angle in radians.
    /// Zero vector gives identity.
    #[inline]
    pub fn from_scaled_axis(v: Vec3) -> Self {
        let angle = v.length();
        if angle < 1e-8 {
            // first order, keeps tiny rotations instead of snapping to identity
            return Self::new(v.x * 0.5, v.y * 0.5, v.z * 0.5, 1.0).normalize();
        }
        let (s, c) = (angle * 0.5).sin_cos();
        let k = s / angle;
        Self::new(v.x * k, v.y * k, v.z * k, c)
    }

    /// Rotation vector of this rotation, angle in `[0, pi]`. Inverse of [`from_scaled_axis`](Quat::from_scaled_axis).
    #[inline]
    pub fn to_scaled_axis(self) -> Vec3 {
        // q and -q are the same rotation, take the short way
        let q = if self.w < 0.0 { -self.normalize() } else { self.normalize() };
        let xyz = Vec3::new(q.x, q.y, q.z);
        let sin_half = xyz.length();
        if sin_half < 1e-8 {
            return xyz * 2.0;
        }
        let angle = 2.0 * sin_half.atan2(q.w);
        xyz * (angle / sin_half)
    }

    /// Angle in radians of the rotation that takes `self` to `other`, in `[0, pi]`.
    #[inline]
    pub fn angle_between(self, other: Self) -> f32 {
        // atan2 of the relative rotation, acos(dot) loses precision near zero
        let d = self.conjugate() * other;
        let sin_half = Vec3::new(d.x, d.y, d.z).length();
        2.0 * sin_half.atan2(d.w.abs())
    }

    /// True when every component differs by at most `epsilon`. `q` and `-q` count as different.
    #[inline]
    pub fn abs_diff_eq(self, other: Self, epsilon: f32) -> bool {
        self.0.abs_diff_eq(other.0, epsilon)
    }

    #[inline]
    pub fn is_finite(self) -> bool {
        self.0.is_finite()
    }

    #[inline]
    pub fn is_nan(self) -> bool {
        self.0.is_nan()
    }

    /// Quaternion from a 3x3 rotation matrix. Use when you have a Mat3 and need a Quat.
    #[inline]
    pub fn from_mat3(mat: &crate::Mat3) -> Self {
        let trace = mat.trace();
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            let inv_s = s.recip();
            Self::new((mat.y_axis.z - mat.z_axis.y) * inv_s, (mat.z_axis.x - mat.x_axis.z) * inv_s, (mat.x_axis.y - mat.y_axis.x) * inv_s, s * 0.25)
        } else if mat.x_axis.x > mat.y_axis.y && mat.x_axis.x > mat.z_axis.z {
            let s = (1.0 + mat.x_axis.x - mat.y_axis.y - mat.z_axis.z).sqrt() * 2.0;
            let inv_s = s.recip();
            Self::new(s * 0.25, (mat.x_axis.y + mat.y_axis.x) * inv_s, (mat.z_axis.x + mat.x_axis.z) * inv_s, (mat.y_axis.z - mat.z_axis.y) * inv_s)
        } else if mat.y_axis.y > mat.z_axis.z {
            let s = (1.0 + mat.y_axis.y - mat.x_axis.x - mat.z_axis.z).sqrt() * 2.0;
            let inv_s = s.recip();
            Self::new((mat.x_axis.y + mat.y_axis.x) * inv_s, s * 0.25, (mat.y_axis.z + mat.z_axis.y) * inv_s, (mat.z_axis.x - mat.x_axis.z) * inv_s)
        } else {
            let s = (1.0 + mat.z_axis.z - mat.x_axis.x - mat.y_axis.y).sqrt() * 2.0;
            let inv_s = s.recip();
            Self::new((mat.z_axis.x + mat.x_axis.z) * inv_s, (mat.y_axis.z + mat.z_axis.y) * inv_s, s * 0.25, (mat.x_axis.y - mat.y_axis.x) * inv_s)
        }
    }

    pub fn from_rotation_arc(from: Vec3, to: Vec3) -> Self {
        let from_norm = from.normalize();
        let to_norm = to.normalize();
        let dot = from_norm.dot(to_norm);

        if dot >= 1.0 {
            return Self::IDENTITY;
        }

        if dot <= -1.0 {
            let axis = if from_norm.x.abs() < 0.9 {
                from_norm.cross(Vec3::X).normalize()
            } else {
                from_norm.cross(Vec3::Y).normalize()
            };
            return Self::from_axis_angle(axis, utils::PI);
        }

        let axis = from_norm.cross(to_norm).normalize();
        let angle = dot.acos();
        Self::from_axis_angle(axis, angle)
    }

    pub fn from_euler_xyz(x: f32, y: f32, z: f32) -> Self {
        let half_x = x * 0.5;
        let half_y = y * 0.5;
        let half_z = z * 0.5;

        let sx = half_x.sin();
        let cx = half_x.cos();
        let sy = half_y.sin();
        let cy = half_y.cos();
        let sz = half_z.sin();
        let cz = half_z.cos();

        Self::new(sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz)
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.0.dot(self.0)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            Self(self.0 * len_sq.sqrt().recip())
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self::new(-self.x, -self.y, -self.z, self.w)
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len_sq = len_sq.recip();
            Self::new(-self.x * inv_len_sq, -self.y * inv_len_sq, -self.z * inv_len_sq, self.w * inv_len_sq)
        } else {
            Self::IDENTITY
        }
    }

    /// Spherical linear interpolation. Uses nlerp when the quaternions are very
    /// close (avoids acos/sin) and full slerp otherwise.
    #[inline]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let dot = self.dot(other);
        let abs_dot = dot.abs();

        if abs_dot >= 1.0 {
            return self;
        }

        const DOT_THRESHOLD: f32 = 0.9995;
        if abs_dot > DOT_THRESHOLD {
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            let result = Self(self.0 + (other.0 * sign - self.0) * t);
            return result.normalize();
        }

        let theta = utils::acos_approx(abs_dot);
        let sin_theta = theta.sin();
        let inv_sin_theta = sin_theta.recip();
        let t_inv = 1.0 - t;
        let scale0 = (t_inv * theta).sin() * inv_sin_theta;
        let scale1 = (t * theta).sin() * inv_sin_theta;

        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        // Exact slerp of unit quaternions is unit length; no re-normalize needed.
        Self(self.0 * scale0 + other.0 * (scale1 * sign))
    }

    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        // v' = v + w*t + q_xyz x t, with t = 2 * (q_xyz x v).
        // Expressed through Vec3 ops so the SIMD cross/add/mul paths are used.
        let q_xyz = Vec3::new(self.x, self.y, self.z);
        let t = q_xyz.cross(other) * 2.0;
        other + t * self.w + q_xyz.cross(t)
    }

    /// Rotate a vector by this quaternion (alias for mul_vec3)
    #[inline]
    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        self.mul_vec3(v)
    }

    /// Extract axis and angle from this quaternion
    /// Returns (axis, angle) where axis is a unit vector and angle is in radians
    #[inline]
    pub fn to_axis_angle(self) -> (Vec3, f32) {
        let normalized = self.normalize();
        let angle = 2.0 * normalized.w.acos();
        let sin_half_angle = (1.0 - normalized.w * normalized.w).sqrt();

        if sin_half_angle < 0.0001 {
            (Vec3::X, angle)
        } else {
            let axis = Vec3::new(
                normalized.x / sin_half_angle,
                normalized.y / sin_half_angle,
                normalized.z / sin_half_angle,
            );
            (axis, angle)
        }
    }

    /// Create from Euler angles with specified rotation order
    #[inline]
    pub fn from_euler(order: crate::EulerRot, x: f32, y: f32, z: f32) -> Self {
        match order {
            crate::EulerRot::XYZ => Self::from_euler_xyz(x, y, z),
            crate::EulerRot::XZY => {
                let qx = Self::from_axis_angle(crate::Vec3::X, x);
                let qz = Self::from_axis_angle(crate::Vec3::Z, z);
                let qy = Self::from_axis_angle(crate::Vec3::Y, y);
                qx * qz * qy
            }
            crate::EulerRot::YXZ => {
                let qy = Self::from_axis_angle(crate::Vec3::Y, y);
                let qx = Self::from_axis_angle(crate::Vec3::X, x);
                let qz = Self::from_axis_angle(crate::Vec3::Z, z);
                qy * qx * qz
            }
            crate::EulerRot::YZX => {
                let qy = Self::from_axis_angle(crate::Vec3::Y, y);
                let qz = Self::from_axis_angle(crate::Vec3::Z, z);
                let qx = Self::from_axis_angle(crate::Vec3::X, x);
                qy * qz * qx
            }
            crate::EulerRot::ZXY => {
                let qz = Self::from_axis_angle(crate::Vec3::Z, z);
                let qx = Self::from_axis_angle(crate::Vec3::X, x);
                let qy = Self::from_axis_angle(crate::Vec3::Y, y);
                qz * qx * qy
            }
            crate::EulerRot::ZYX => {
                let qz = Self::from_axis_angle(crate::Vec3::Z, z);
                let qy = Self::from_axis_angle(crate::Vec3::Y, y);
                let qx = Self::from_axis_angle(crate::Vec3::X, x);
                qz * qy * qx
            }
        }
    }

    /// Extract Euler angles (XYZ order) from this quaternion
    /// Returns (pitch, yaw, roll) in radians
    #[inline]
    pub fn to_euler_xyz(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.x + self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.y - self.z * self.x);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Dot product of two quaternions
    #[inline(always)]
    pub fn dot(self, other: Self) -> f32 {
        self.0.dot(other.0)
    }

    /// Normalized linear interpolation (faster than slerp for close quaternions)
    #[inline]
    pub fn nlerp(self, other: Self, t: f32) -> Self {
        let dot = self.dot(other);
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        Self(self.0 + (other.0 * sign - self.0) * t).normalize()
    }

    /// Create from array [x, y, z, w]
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// Convert to array [x, y, z, w]
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Check if the quaternion is approximately normalized
    #[inline(always)]
    pub fn is_normalized(self) -> bool {
        (self.length_squared() - 1.0).abs() < 0.0001
    }

    /// Extract Euler angles with specified rotation order
    /// Returns (x, y, z) angles in radians
    #[inline]
    pub fn to_euler(self, order: crate::EulerRot) -> (f32, f32, f32) {
        match order {
            crate::EulerRot::XYZ => self.to_euler_xyz(),
            crate::EulerRot::XZY => {
                let (x, z, y) = self.to_euler_xzy();
                (x, y, z)
            }
            crate::EulerRot::YXZ => {
                let (y, x, z) = self.to_euler_yxz();
                (x, y, z)
            }
            crate::EulerRot::YZX => {
                let (y, z, x) = self.to_euler_yzx();
                (x, y, z)
            }
            crate::EulerRot::ZXY => {
                let (z, x, y) = self.to_euler_zxy();
                (x, y, z)
            }
            crate::EulerRot::ZYX => {
                let (z, y, x) = self.to_euler_zyx();
                (x, y, z)
            }
        }
    }

    /// Extract Euler angles in XZY order
    #[inline]
    fn to_euler_xzy(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.x - self.y * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.z * self.z);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.z - self.x * self.y);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.y - self.z * self.x);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Extract Euler angles in YXZ order
    #[inline]
    fn to_euler_yxz(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.y - self.z * self.x);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let yaw = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.x - self.y * self.z);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.z - self.x * self.y);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let roll = siny_cosp.atan2(cosy_cosp);

        (yaw, pitch, roll)
    }

    /// Extract Euler angles in YZX order
    #[inline]
    fn to_euler_yzx(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.y + self.x * self.z);
        let cosr_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let yaw = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.z - self.x * self.y);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.x - self.y * self.z);
        let cosy_cosp = 1.0 - 2.0 * (self.x * self.x + self.z * self.z);
        let roll = siny_cosp.atan2(cosy_cosp);

        (yaw, pitch, roll)
    }

    /// Extract Euler angles in ZXY order
    #[inline]
    fn to_euler_zxy(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.z + self.x * self.y);
        let cosr_cosp = 1.0 - 2.0 * (self.x * self.x + self.z * self.z);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.x - self.y * self.z);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.y - self.z * self.x);
        let cosy_cosp = 1.0 - 2.0 * (self.y * self.y + self.x * self.x);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }

    /// Extract Euler angles in ZYX order
    #[inline]
    fn to_euler_zyx(self) -> (f32, f32, f32) {
        let sinr_cosp = 2.0 * (self.w * self.z - self.x * self.y);
        let cosr_cosp = 1.0 - 2.0 * (self.y * self.y + self.z * self.z);
        let roll = sinr_cosp.atan2(cosr_cosp);

        let sinp = 2.0 * (self.w * self.y + self.x * self.z);
        let pitch = if sinp.abs() >= 1.0 {
            std::f32::consts::FRAC_PI_2.copysign(sinp)
        } else {
            sinp.asin()
        };

        let siny_cosp = 2.0 * (self.w * self.x - self.y * self.z);
        let cosy_cosp = 1.0 - 2.0 * (self.x * self.x + self.y * self.y);
        let yaw = siny_cosp.atan2(cosy_cosp);

        (roll, pitch, yaw)
    }
}

impl std::ops::Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            // a * b = a.w*b + a.x*[bw,-bz,by,-bx] + a.y*[bz,bw,-bx,-by] + a.z*[-by,bx,bw,-bz]
            use std::arch::x86_64::*;
            unsafe {
                let a = (self.0).0;
                let b = (other.0).0;
                let b_wzyx = _mm_shuffle_ps(b, b, 0b00_01_10_11);
                let b_zwxy = _mm_shuffle_ps(b, b, 0b01_00_11_10);
                let b_yxwz = _mm_shuffle_ps(b, b, 0b10_11_00_01);
                let t0 = _mm_mul_ps(_mm_shuffle_ps(a, a, 0b11_11_11_11), b);
                let t1 = _mm_mul_ps(_mm_mul_ps(_mm_shuffle_ps(a, a, 0b00_00_00_00), b_wzyx), _mm_setr_ps(1.0, -1.0, 1.0, -1.0));
                let t2 = _mm_mul_ps(_mm_mul_ps(_mm_shuffle_ps(a, a, 0b01_01_01_01), b_zwxy), _mm_setr_ps(1.0, 1.0, -1.0, -1.0));
                let t3 = _mm_mul_ps(_mm_mul_ps(_mm_shuffle_ps(a, a, 0b10_10_10_10), b_yxwz), _mm_setr_ps(-1.0, 1.0, 1.0, -1.0));
                Self(Vec4(_mm_add_ps(_mm_add_ps(t0, t1), _mm_add_ps(t2, t3))))
            }
        }
        #[cfg(not(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86"))))]
        {
            self.mul_scalar(other)
        }
    }
}

impl Quat {
    /// Hamilton product written out per lane. Reference for the SIMD path.
    #[inline]
    #[cfg_attr(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86"), not(test)), allow(dead_code))]
    fn mul_scalar(self, other: Self) -> Self {
        Self::new(self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y, self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z, self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x, self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z)
    }
}

impl std::ops::Mul<Vec3> for Quat {
    type Output = Vec3;
    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        self.mul_vec3(other)
    }
}

impl std::ops::Mul<f32> for Quat {
    type Output = Quat;
    #[inline]
    fn mul(self, scalar: f32) -> Quat {
        Quat(self.0 * scalar)
    }
}

impl std::ops::Add for Quat {
    type Output = Quat;
    #[inline]
    fn add(self, other: Quat) -> Quat {
        Quat(self.0 + other.0)
    }
}

impl std::ops::Sub for Quat {
    type Output = Quat;
    #[inline]
    fn sub(self, other: Quat) -> Quat {
        Quat(self.0 - other.0)
    }
}

impl std::ops::Neg for Quat {
    type Output = Quat;
    #[inline]
    fn neg(self) -> Quat {
        Quat(-self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quat_identity() {
        let q = Quat::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(q * v, v);
    }

    #[test]
    fn test_quat_from_axis_angle() {
        let q = Quat::from_axis_angle(Vec3::X, 0.0);
        assert!((q.w - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_quat_normalize() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let normalized = q.normalize();
        assert!((normalized.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_quat_conjugate() {
        let q = Quat::new(1.0, 2.0, 3.0, 4.0);
        let conj = q.conjugate();
        assert_eq!(conj.x, -q.x);
        assert_eq!(conj.y, -q.y);
        assert_eq!(conj.z, -q.z);
        assert_eq!(conj.w, q.w);
    }

    #[test]
    fn test_quat_mul() {
        let q1 = Quat::IDENTITY;
        let q2 = Quat::IDENTITY;
        assert_eq!(q1 * q2, Quat::IDENTITY);
    }

    fn quats() -> Vec<Quat> {
        // fixed pseudo random set, no rand dependency
        let mut seed = 0x2545_f491_u32;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        (0..256).map(|_| Quat::new(next(), next(), next(), next())).collect()
    }

    #[test]
    fn test_quat_mul_matches_scalar() {
        let qs = quats();
        for pair in qs.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            assert!((a * b).abs_diff_eq(a.mul_scalar(b), 1e-5), "{a} * {b}");
        }
    }

    #[test]
    fn test_quat_mul_composes_rotations() {
        let q = Quat::from_rotation_z(0.3) * Quat::from_rotation_x(1.1);
        let v = Vec3::new(0.2, -0.7, 1.5);
        let expected = Quat::from_rotation_z(0.3) * (Quat::from_rotation_x(1.1) * v);
        assert!((q * v).abs_diff_eq(expected, 1e-5));
    }

    #[test]
    fn test_quat_from_rotation_axes() {
        let angle = 0.7;
        assert!(Quat::from_rotation_x(angle).abs_diff_eq(Quat::from_axis_angle(Vec3::X, angle), 1e-6));
        assert!(Quat::from_rotation_y(angle).abs_diff_eq(Quat::from_axis_angle(Vec3::Y, angle), 1e-6));
        assert!(Quat::from_rotation_z(angle).abs_diff_eq(Quat::from_axis_angle(Vec3::Z, angle), 1e-6));
    }

    #[test]
    fn test_quat_scaled_axis_round_trip() {
        for q in quats() {
            let q = q.normalize();
            let back = Quat::from_scaled_axis(q.to_scaled_axis());
            assert!(back.angle_between(q) < 1e-3, "{q} -> {back}");
            assert!(q.to_scaled_axis().length() <= std::f32::consts::PI + 1e-5);
        }
        let v = Vec3::new(0.0, 0.0, 1e-9);
        assert!(Quat::from_scaled_axis(v).to_scaled_axis().abs_diff_eq(v, 1e-12));
        assert_eq!(Quat::from_scaled_axis(Vec3::ZERO), Quat::IDENTITY);
    }

    #[test]
    fn test_quat_angle_between() {
        let a = Quat::from_rotation_y(0.2);
        let b = Quat::from_rotation_y(0.9);
        assert!((a.angle_between(b) - 0.7).abs() < 1e-4);
        assert!(a.angle_between(-a) < 1e-3);
    }

    #[test]
    fn test_quat_finite() {
        assert!(Quat::IDENTITY.is_finite());
        assert!(Quat::new(f32::NAN, 0.0, 0.0, 1.0).is_nan());
        assert!(!Quat::new(f32::INFINITY, 0.0, 0.0, 1.0).is_finite());
    }

    #[test]
    fn test_quat_slerp() {
        let q1 = Quat::IDENTITY;
        let q2 = Quat::from_axis_angle(Vec3::X, 1.0);
        let result = q1.slerp(q2, 0.5);
        assert!((result.length() - 1.0).abs() < 0.0001);
    }
}

impl Default for Quat {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl std::fmt::Display for Quat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl From<[f32; 4]> for Quat {
    #[inline]
    fn from(a: [f32; 4]) -> Self {
        Self::from_xyzw(a[0], a[1], a[2], a[3])
    }
}

impl From<Quat> for [f32; 4] {
    #[inline]
    fn from(q: Quat) -> Self {
        [q.x, q.y, q.z, q.w]
    }
}
