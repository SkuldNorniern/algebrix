//! 4D unsigned integer vector (UVec4). Same layout as Vec4 but u32; use [`as_vec4`](UVec4::as_vec4) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::UVec4;
//! let a = UVec4::new(1, 2, 3, 4);
//! let b = UVec4::splat(2);
//! let c = a * b;
//! assert_eq!(c.w, 8);
//! ```
//!

use crate::Vec4;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UVec4 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
    pub w: u32,
}

impl UVec4 {
    pub const ZERO: UVec4 = UVec4 { x: 0, y: 0, z: 0, w: 0 };
    pub const ONE: UVec4 = UVec4 { x: 1, y: 1, z: 1, w: 1 };

    #[inline(always)]
    pub const fn new(x: u32, y: u32, z: u32, w: u32) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub const fn splat(v: u32) -> Self {
        Self { x: v, y: v, z: v, w: v }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
            z: self.z.min(other.z),
            w: self.w.min(other.w),
        }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
            z: self.z.max(other.z),
            w: self.w.max(other.w),
        }
    }

    #[inline(always)]
    pub fn min_element(self) -> u32 {
        self.x.min(self.y).min(self.z).min(self.w)
    }

    #[inline(always)]
    pub fn max_element(self) -> u32 {
        self.x.max(self.y).max(self.z).max(self.w)
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
            z: self.z.clamp(min.z, max.z),
            w: self.w.clamp(min.w, max.w),
        }
    }

    #[inline(always)]
    pub fn as_vec4(self) -> Vec4 {
        Vec4::new(self.x as f32, self.y as f32, self.z as f32, self.w as f32)
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 4]) -> Self {
        Self { x: a[0], y: a[1], z: a[2], w: a[3] }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> u32 {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_add(other.x),
            y: self.y.saturating_add(other.y),
            z: self.z.saturating_add(other.z),
            w: self.w.saturating_add(other.w),
        }
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_sub(other.x),
            y: self.y.saturating_sub(other.y),
            z: self.z.saturating_sub(other.z),
            w: self.w.saturating_sub(other.w),
        }
    }
}

impl std::ops::Add for UVec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

impl std::ops::Sub for UVec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

impl std::ops::Mul<u32> for UVec4 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: u32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl std::ops::Mul for UVec4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z,
            w: self.w * other.w,
        }
    }
}

impl std::ops::Div<u32> for UVec4 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: u32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            w: self.w / scalar,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uvec4_new() {
        let v = UVec4::new(1, 2, 3, 4);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
        assert_eq!(v.z, 3);
        assert_eq!(v.w, 4);
    }

    #[test]
    fn test_uvec4_dot() {
        let v1 = UVec4::new(1, 2, 3, 4);
        let v2 = UVec4::new(5, 6, 7, 8);
        assert_eq!(v1.dot(v2), 70);
    }
}
