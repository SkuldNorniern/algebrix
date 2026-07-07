//! 3D float vector in double precision (f64). Use when you need more precision than Vec3; convert
//! to/from Vec3 with [`from_vec3`](DVec3::from_vec3) / [`as_vec3`](DVec3::as_vec3).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{DVec3, Vec3};
//! let a = DVec3::new(1.0, 2.0, 3.0);
//! let b = DVec3::from_vec3(Vec3::new(1.0, 0.0, 0.0));
//! let c = a + b;
//! assert!((c.x - 2.0).abs() < 1e-10);
//! ```
//!

/// 3D vector in double precision (f64).
///
/// Use when you need more precision than [`Vec3`](crate::Vec3); convert with [`from_vec3`](DVec3::from_vec3) and [`as_vec3`](DVec3::as_vec3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl DVec3 {
    /// Zero vector.
    pub const ZERO: DVec3 = DVec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    /// All components 1.
    pub const ONE: DVec3 = DVec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
    };
    /// Unit vector along +X.
    pub const X: DVec3 = DVec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    };
    /// Unit vector along +Y.
    pub const Y: DVec3 = DVec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
    };
    /// Unit vector along +Z.
    pub const Z: DVec3 = DVec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Build from x, y, z.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// All components set to `value`.
    #[inline]
    pub fn splat(value: f64) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Squared length (avoids sqrt).
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Distance between two points.
    #[inline]
    pub fn distance(self, other: Self) -> f64 {
        (self - other).length()
    }

    /// Squared distance (avoids sqrt).
    #[inline]
    pub fn distance_squared(self, other: Self) -> f64 {
        (self - other).length_squared()
    }

    /// Unit vector in the same direction. Returns zero if length is zero.
    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = 1.0 / len_sq.sqrt();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
            }
        } else {
            Self::ZERO
        }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product (right-handed).
    #[inline]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    /// Linear interpolation: `self * (1 - t) + other * t`.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        let t_inv = 1.0 - t;
        Self {
            x: self.x * t_inv + other.x * t,
            y: self.y * t_inv + other.y * t,
            z: self.z * t_inv + other.z * t,
        }
    }

    /// Convert to single-precision [`Vec3`](crate::Vec3).
    #[inline]
    pub fn as_vec3(self) -> crate::Vec3 {
        crate::Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Build from a single-precision [`Vec3`](crate::Vec3).
    #[inline]
    pub fn from_vec3(v: crate::Vec3) -> Self {
        Self::new(v.x as f64, v.y as f64, v.z as f64)
    }
}

impl std::ops::Add for DVec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

impl std::ops::Sub for DVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl std::ops::Mul<f64> for DVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f64) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

impl std::ops::Div<f64> for DVec3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f64) -> Self {
        let inv = 1.0 / scalar;
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }
}

impl std::ops::Neg for DVec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl std::ops::AddAssign for DVec3 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl std::ops::SubAssign for DVec3 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

impl std::ops::MulAssign<f64> for DVec3 {
    #[inline]
    fn mul_assign(&mut self, scalar: f64) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl std::ops::DivAssign<f64> for DVec3 {
    #[inline]
    fn div_assign(&mut self, scalar: f64) {
        let inv = 1.0 / scalar;
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

impl std::ops::Mul<DVec3> for f64 {
    type Output = DVec3;
    #[inline]
    fn mul(self, v: DVec3) -> DVec3 {
        DVec3 {
            x: self * v.x,
            y: self * v.y,
            z: self * v.z,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;

    #[test]
    fn test_dvec3_conversion() {
        let v32 = Vec3::new(1.0, 2.0, 3.0);
        let v64 = DVec3::from_vec3(v32);
        assert_eq!(v64.x, 1.0);
        assert_eq!(v64.y, 2.0);
        assert_eq!(v64.z, 3.0);

        let back = v64.as_vec3();
        assert_eq!(back, v32);
    }

    #[test]
    fn test_dvec3_ops() {
        let a = DVec3::new(1.0, 2.0, 3.0);
        let b = DVec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, DVec3::new(5.0, 7.0, 9.0));
        assert_eq!(a.dot(b), 32.0);
    }
}
