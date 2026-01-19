//! Quaternion implementation

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

    #[inline]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let dot = self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w;
        let abs_dot = dot.abs();

        if abs_dot >= 1.0 {
            return self;
        }

        // Fast path: when quaternions are very close, use normalized lerp instead of expensive trig
        // This avoids acos/sin calls in animation-heavy code paths
        const DOT_THRESHOLD: f32 = 0.9995;
        if abs_dot > DOT_THRESHOLD {
            // Normalized lerp (nlerp) - much faster than slerp for close quaternions
            let sign = if dot < 0.0 { -1.0 } else { 1.0 };
            let result = Self {
                x: self.x + (other.x * sign - self.x) * t,
                y: self.y + (other.y * sign - self.y) * t,
                z: self.z + (other.z * sign - self.z) * t,
                w: self.w + (other.w * sign - self.w) * t,
            };
            return result.normalize();
        }

        // Full slerp for quaternions that are further apart
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

        // Use mul_add for the final computation (FMA optimization)
        Vec3::new(
            vx + qw.mul_add(tx, qy.mul_add(tz, -qz * ty)),
            vy + qw.mul_add(ty, qz.mul_add(tx, -qx * tz)),
            vz + qw.mul_add(tz, qx.mul_add(ty, -qy * tx)),
        )
    }
}

impl std::ops::Mul for Quat {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        // Use mul_add where possible for FMA optimization
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
