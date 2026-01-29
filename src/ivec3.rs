//! 3D signed integer vector (IVec3). Same layout as Vec3 but i32; use [`as_vec3`](IVec3::as_vec3) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::IVec3;
//! let a = IVec3::new(1, 0, 0);
//! let b = IVec3::new(0, 1, 0);
//! let c = a.cross(b);
//! assert_eq!(c, IVec3::new(0, 0, 1));
//! ```
//!

use crate::Vec3;

/// 3D signed integer vector (i32). Same layout as [`Vec3`]; use [`as_vec3`](IVec3::as_vec3) to convert to float.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IVec3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl IVec3 {
    /// Zero vector (0, 0, 0).
    pub const ZERO: IVec3 = IVec3 { x: 0, y: 0, z: 0 };
    /// Vector (1, 1, 1).
    pub const ONE: IVec3 = IVec3 { x: 1, y: 1, z: 1 };
    /// Unit vector (1, 0, 0).
    pub const X: IVec3 = IVec3 { x: 1, y: 0, z: 0 };
    /// Unit vector (0, 1, 0).
    pub const Y: IVec3 = IVec3 { x: 0, y: 1, z: 0 };
    /// Unit vector (0, 0, 1).
    pub const Z: IVec3 = IVec3 { x: 0, y: 0, z: 1 };
    /// Unit vector (-1, 0, 0).
    pub const NEG_X: IVec3 = IVec3 { x: -1, y: 0, z: 0 };
    /// Unit vector (0, -1, 0).
    pub const NEG_Y: IVec3 = IVec3 { x: 0, y: -1, z: 0 };
    /// Unit vector (0, 0, -1).
    pub const NEG_Z: IVec3 = IVec3 { x: 0, y: 0, z: -1 };

    /// Build from x, y, z.
    #[inline(always)]
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub const fn splat(v: i32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs(), z: self.z.abs() }
    }

    /// Component-wise minimum.
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    /// Component-wise maximum.
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    /// Smallest component.
    #[inline(always)]
    pub fn min_element(self) -> i32 {
        self.x.min(self.y).min(self.z)
    }

    /// Largest component.
    #[inline(always)]
    pub fn max_element(self) -> i32 {
        self.x.max(self.y).max(self.z)
    }

    /// Clamp each component to the range [min, max] per axis.
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
        }
    }

    /// Convert to float [`Vec3`] (component cast).
    #[inline(always)]
    pub fn as_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    /// Build from a 3-element array.
    #[inline(always)]
    pub fn from_array(a: [i32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    /// Copy into a 3-element array [x, y, z].
    #[inline(always)]
    pub fn to_array(self) -> [i32; 3] {
        [self.x, self.y, self.z]
    }

    /// Dot product (x*x + y*y + z*z).
    #[inline(always)]
    pub fn dot(self, other: Self) -> i32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Cross product (right-handed): self x other.
    #[inline(always)]
    pub fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl std::ops::Add for IVec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl std::ops::Sub for IVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl std::ops::Mul<i32> for IVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: i32) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }
}

impl std::ops::Mul for IVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z }
    }
}

impl std::ops::Div<i32> for IVec3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: i32) -> Self {
        Self { x: self.x / scalar, y: self.y / scalar, z: self.z / scalar }
    }
}

impl std::ops::Neg for IVec3 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl From<Vec3> for IVec3 {
    fn from(v: Vec3) -> Self {
        Self { x: v.x as i32, y: v.y as i32, z: v.z as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivec3_new() {
        let v = IVec3::new(1, 2, 3);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
        assert_eq!(v.z, 3);
    }

    #[test]
    fn test_ivec3_cross() {
        let v1 = IVec3::X;
        let v2 = IVec3::Y;
        assert_eq!(v1.cross(v2), IVec3::Z);
    }
}
