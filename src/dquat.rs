//! Quaternion implementation (f64)

use crate::{DVec3, Quat};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DQuat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

impl DQuat {
    pub const IDENTITY: DQuat = DQuat {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };

    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn from_axis_angle(axis: DVec3, angle: f64) -> Self {
        let (s, c) = (angle * 0.5).sin_cos();
        let axis = axis.normalize();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: c,
        }
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w;
        if len_sq > 0.0 {
            let inv_len = 1.0 / len_sq.sqrt();
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
        self.normalize().conjugate()
    }

    #[inline]
    pub fn as_quat(self) -> Quat {
        Quat::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }

    #[inline]
    pub fn from_quat(q: Quat) -> Self {
        Self::new(q.x as f64, q.y as f64, q.z as f64, q.w as f64)
    }

    #[inline]
    pub fn mul_vec3(self, v: DVec3) -> DVec3 {
        let q = DVec3::new(self.x, self.y, self.z);
        let t = q.cross(v) * 2.0;
        v + q.cross(t) + t * self.w
    }
}

impl std::ops::Mul for DQuat {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

impl std::ops::Mul<DVec3> for DQuat {
    type Output = DVec3;
    #[inline]
    fn mul(self, v: DVec3) -> DVec3 {
        self.mul_vec3(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;

    #[test]
    fn test_dquat_conversion() {
        let q32 = Quat::from_axis_angle(Vec3::Y, 1.0);
        let q64 = DQuat::from_quat(q32);

        let back = q64.as_quat();
        assert!((back.x - q32.x).abs() < 1e-6);
        assert!((back.y - q32.y).abs() < 1e-6);
        assert!((back.z - q32.z).abs() < 1e-6);
        assert!((back.w - q32.w).abs() < 1e-6);
    }
}
