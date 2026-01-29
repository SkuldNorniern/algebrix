//! 3D unsigned integer vector (UVec3). Same layout as Vec3 but u32; use [`as_vec3`](UVec3::as_vec3) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::UVec3;
//! let a = UVec3::new(1, 2, 3);
//! let b = UVec3::splat(1);
//! let c = a + b;
//! assert_eq!(c.z, 4);
//! ```
//!

use crate::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UVec3 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl UVec3 {
    pub const ZERO: UVec3 = UVec3 { x: 0, y: 0, z: 0 };
    pub const ONE: UVec3 = UVec3 { x: 1, y: 1, z: 1 };
    pub const X: UVec3 = UVec3 { x: 1, y: 0, z: 0 };
    pub const Y: UVec3 = UVec3 { x: 0, y: 1, z: 0 };
    pub const Z: UVec3 = UVec3 { x: 0, y: 0, z: 1 };

    #[inline(always)]
    pub const fn new(x: u32, y: u32, z: u32) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub const fn splat(v: u32) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
        }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
        }
    }

    #[inline(always)]
    pub fn min_element(self) -> u32 {
        self.x.min(self.y).min(self.z)
    }

    #[inline(always)]
    pub fn max_element(self) -> u32 {
        self.x.max(self.y).max(self.z)
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
        }
    }

    #[inline(always)]
    pub fn as_vec3(self) -> Vec3 {
        Vec3::new(self.x as f32, self.y as f32, self.z as f32)
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 3]) -> Self {
        Self { x: a[0], y: a[1], z: a[2] }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> u32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_add(other.x),
            y: self.y.saturating_add(other.y),
            z: self.z.saturating_add(other.z),
        }
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_sub(other.x),
            y: self.y.saturating_sub(other.y),
            z: self.z.saturating_sub(other.z),
        }
    }
}

impl std::ops::Add for UVec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }
}

impl std::ops::Sub for UVec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }
}

impl std::ops::Mul<u32> for UVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: u32) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar, z: self.z * scalar }
    }
}

impl std::ops::Mul for UVec3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y, z: self.z * other.z }
    }
}

impl std::ops::Div<u32> for UVec3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: u32) -> Self {
        Self { x: self.x / scalar, y: self.y / scalar, z: self.z / scalar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uvec3_new() {
        let v = UVec3::new(1, 2, 3);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
        assert_eq!(v.z, 3);
    }
}
