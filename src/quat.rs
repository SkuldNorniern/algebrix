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

use crate::{Vec3, utils};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quat {
    pub const IDENTITY: Quat = Quat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub const fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
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
        Self {
            x: normalized_axis.x * s,
            y: normalized_axis.y * s,
            z: normalized_axis.z * s,
            w: c,
        }
    }

    /// Quaternion from a 3x3 rotation matrix. Use when you have a Mat3 and need a Quat.
    #[inline]
    pub fn from_mat3(mat: &crate::Mat3) -> Self {
        let trace = mat.trace();
        if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            let inv_s = s.recip();
            Self {
                x: (mat.y_axis.z - mat.z_axis.y) * inv_s,
                y: (mat.z_axis.x - mat.x_axis.z) * inv_s,
                z: (mat.x_axis.y - mat.y_axis.x) * inv_s,
                w: s * 0.25,
            }
        } else if mat.x_axis.x > mat.y_axis.y && mat.x_axis.x > mat.z_axis.z {
            let s = (1.0 + mat.x_axis.x - mat.y_axis.y - mat.z_axis.z).sqrt() * 2.0;
            let inv_s = s.recip();
            Self {
                x: s * 0.25,
                y: (mat.x_axis.y + mat.y_axis.x) * inv_s,
                z: (mat.z_axis.x + mat.x_axis.z) * inv_s,
                w: (mat.y_axis.z - mat.z_axis.y) * inv_s,
            }
        } else if mat.y_axis.y > mat.z_axis.z {
            let s = (1.0 + mat.y_axis.y - mat.x_axis.x - mat.z_axis.z).sqrt() * 2.0;
            let inv_s = s.recip();
            Self {
                x: (mat.x_axis.y + mat.y_axis.x) * inv_s,
                y: s * 0.25,
                z: (mat.y_axis.z + mat.z_axis.y) * inv_s,
                w: (mat.z_axis.x - mat.x_axis.z) * inv_s,
            }
        } else {
            let s = (1.0 + mat.z_axis.z - mat.x_axis.x - mat.y_axis.y).sqrt() * 2.0;
            let inv_s = s.recip();
            Self {
                x: (mat.z_axis.x + mat.x_axis.z) * inv_s,
                y: (mat.y_axis.z + mat.z_axis.y) * inv_s,
                z: s * 0.25,
                w: (mat.x_axis.y - mat.y_axis.x) * inv_s,
            }
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

        Self {
            x: sx * cy * cz - cx * sy * sz,
            y: cx * sy * cz + sx * cy * sz,
            z: cx * cy * sz - sx * sy * cz,
            w: cx * cy * cz + sx * sy * sz,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                w: self.w * inv_len,
            }
        } else {
            Self::IDENTITY
        }
    }

    #[inline]
    pub fn conjugate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        }
    }

    #[inline]
    pub fn inverse(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len_sq = len_sq.recip();
            Self {
                x: -self.x * inv_len_sq,
                y: -self.y * inv_len_sq,
                z: -self.z * inv_len_sq,
                w: self.w * inv_len_sq,
            }
        } else {
            Self::IDENTITY
        }
    }

    /// Spherical linear interpolation. Uses nlerp when the quaternions are very
    /// close (avoids acos/sin) and full slerp otherwise.
    #[inline]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let abs_dot = dot.abs();

        if abs_dot >= 1.0 {
            return self;
        }

        const DOT_THRESHOLD: f32 = 0.9995;
        if abs_dot > DOT_THRESHOLD {
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            let result = Self {
                x: self.x + (other.x * sign - self.x) * t,
                y: self.y + (other.y * sign - self.y) * t,
                z: self.z + (other.z * sign - self.z) * t,
                w: self.w + (other.w * sign - self.w) * t,
            };
            return result.normalize();
        }

        let theta = abs_dot.acos();
        let sin_theta = theta.sin();
        let inv_sin_theta = sin_theta.recip();
        let t_inv = 1.0 - t;
        let scale0 = (t_inv * theta).sin() * inv_sin_theta;
        let scale1 = (t * theta).sin() * inv_sin_theta;

        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let result = Self {
            x: scale0 * self.x + scale1 * other.x * sign,
            y: scale0 * self.y + scale1 * other.y * sign,
            z: scale0 * self.z + scale1 * other.z * sign,
            w: scale0 * self.w + scale1 * other.w * sign,
        };

        result.normalize()
    }

    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        let qx = self.x;
        let qy = self.y;
        let qz = self.z;
        let qw = self.w;
        let vx = other.x;
        let vy = other.y;
        let vz = other.z;

        let tx = 2.0 * (qy * vz - qz * vy);
        let ty = 2.0 * (qz * vx - qx * vz);
        let tz = 2.0 * (qx * vy - qy * vx);

        Vec3::new(
            vx + qw.mul_add(tx, qy.mul_add(tz, -qz * ty)),
            vy + qw.mul_add(ty, qz.mul_add(tx, -qx * tz)),
            vz + qw.mul_add(tz, qx.mul_add(ty, -qy * tx)),
        )
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
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    /// Normalized linear interpolation (faster than slerp for close quaternions)
    #[inline]
    pub fn nlerp(self, other: Self, t: f32) -> Self {
        let dot = self.dot(other);
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        Self {
            x: self.x + (other.x * sign - self.x) * t,
            y: self.y + (other.y * sign - self.y) * t,
            z: self.z + (other.z * sign - self.z) * t,
            w: self.w + (other.w * sign - self.w) * t,
        }.normalize()
    }

    /// Create from array [x, y, z, w]
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
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
        Self {
            x: self.w.mul_add(
                other.x,
                self.x
                    .mul_add(other.w, self.y.mul_add(other.z, -self.z * other.y)),
            ),
            y: self.w.mul_add(
                other.y,
                self.y
                    .mul_add(other.w, self.z.mul_add(other.x, -self.x * other.z)),
            ),
            z: self.w.mul_add(
                other.z,
                self.z
                    .mul_add(other.w, self.x.mul_add(other.y, -self.y * other.x)),
            ),
            w: self.w.mul_add(
                other.w,
                -(self
                    .x
                    .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))),
            ),
        }
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
        Quat {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl std::ops::Add for Quat {
    type Output = Quat;
    #[inline]
    fn add(self, other: Quat) -> Quat {
        Quat {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl std::ops::Sub for Quat {
    type Output = Quat;
    #[inline]
    fn sub(self, other: Quat) -> Quat {
        Quat {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl std::ops::Neg for Quat {
    type Output = Quat;
    #[inline]
    fn neg(self) -> Quat {
        Quat {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
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

    #[test]
    fn test_quat_slerp() {
        let q1 = Quat::IDENTITY;
        let q2 = Quat::from_axis_angle(Vec3::X, 1.0);
        let result = q1.slerp(q2, 0.5);
        assert!((result.length() - 1.0).abs() < 0.0001);
    }
}
