//! 2D unsigned integer vector (UVec2). Same layout as Vec2 but u32; use [`as_vec2`](UVec2::as_vec2) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::UVec2;
//! let a = UVec2::new(1, 2);
//! let b = UVec2::splat(2);
//! let c = a.saturating_add(b);
//! assert_eq!(c.x, 3);
//! ```
//!

use crate::Vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UVec2 {
    pub x: u32,
    pub y: u32,
}

impl UVec2 {
    pub const ZERO: UVec2 = UVec2 { x: 0, y: 0 };
    pub const ONE: UVec2 = UVec2 { x: 1, y: 1 };
    pub const X: UVec2 = UVec2 { x: 1, y: 0 };
    pub const Y: UVec2 = UVec2 { x: 0, y: 1 };

    #[inline(always)]
    pub const fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub const fn splat(v: u32) -> Self {
        Self { x: v, y: v }
    }

    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self { x: self.x.min(other.x), y: self.y.min(other.y) }
    }

    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self { x: self.x.max(other.x), y: self.y.max(other.y) }
    }

    #[inline(always)]
    pub fn min_element(self) -> u32 {
        self.x.min(self.y)
    }

    #[inline(always)]
    pub fn max_element(self) -> u32 {
        self.x.max(self.y)
    }

    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
        }
    }

    #[inline(always)]
    pub fn as_vec2(self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }

    #[inline(always)]
    pub fn from_array(a: [u32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    #[inline(always)]
    pub fn to_array(self) -> [u32; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> u32 {
        self.x * other.x + self.y * other.y
    }

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_add(other.x),
            y: self.y.saturating_add(other.y),
        }
    }

    #[inline(always)]
    pub fn saturating_sub(self, other: Self) -> Self {
        Self {
            x: self.x.saturating_sub(other.x),
            y: self.y.saturating_sub(other.y),
        }
    }
}

impl std::ops::Add for UVec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl std::ops::Sub for UVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

impl std::ops::Mul<u32> for UVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: u32) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

impl std::ops::Mul for UVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y }
    }
}

impl std::ops::Div<u32> for UVec2 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: u32) -> Self {
        Self { x: self.x / scalar, y: self.y / scalar }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uvec2_new() {
        let v = UVec2::new(1, 2);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
    }

    #[test]
    fn test_uvec2_saturating_sub() {
        let v1 = UVec2::new(5, 3);
        let v2 = UVec2::new(3, 10);
        let result = v1.saturating_sub(v2);
        assert_eq!(result, UVec2::new(2, 0));
    }
}
