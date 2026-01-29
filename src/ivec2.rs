//! 2D signed integer vector (IVec2). Same layout as Vec2 but i32; use [`as_vec2`](IVec2::as_vec2) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::IVec2;
//! let a = IVec2::new(1, 2);
//! let b = IVec2::splat(3);
//! let c = a + b;
//! assert_eq!(c.x, 4);
//! assert_eq!(c.y, 5);
//! let min = a.min(b);
//! assert_eq!(min.x, 1);
//! ```
//!

use crate::Vec2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IVec2 {
    pub x: i32,
    pub y: i32,
}

impl IVec2 {
    pub const ZERO: IVec2 = IVec2 { x: 0, y: 0 };
    pub const ONE: IVec2 = IVec2 { x: 1, y: 1 };
    pub const X: IVec2 = IVec2 { x: 1, y: 0 };
    pub const Y: IVec2 = IVec2 { x: 0, y: 1 };
    pub const NEG_X: IVec2 = IVec2 { x: -1, y: 0 };
    pub const NEG_Y: IVec2 = IVec2 { x: 0, y: -1 };

    #[inline(always)]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub const fn splat(v: i32) -> Self {
        Self { x: v, y: v }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        Self { x: self.x.abs(), y: self.y.abs() }
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
    pub fn min_element(self) -> i32 {
        self.x.min(self.y)
    }

    #[inline(always)]
    pub fn max_element(self) -> i32 {
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
    pub fn from_array(a: [i32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 2] {
        [self.x, self.y]
    }

    #[inline(always)]
    pub fn dot(self, other: Self) -> i32 {
        self.x * other.x + self.y * other.y
    }

    #[inline(always)]
    pub fn perp_dot(self, other: Self) -> i32 {
        self.x * other.y - self.y * other.x
    }
}

impl std::ops::Add for IVec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self { x: self.x + other.x, y: self.y + other.y }
    }
}

impl std::ops::Sub for IVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self { x: self.x - other.x, y: self.y - other.y }
    }
}

impl std::ops::Mul<i32> for IVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: i32) -> Self {
        Self { x: self.x * scalar, y: self.y * scalar }
    }
}

impl std::ops::Mul for IVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self { x: self.x * other.x, y: self.y * other.y }
    }
}

impl std::ops::Div<i32> for IVec2 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: i32) -> Self {
        Self { x: self.x / scalar, y: self.y / scalar }
    }
}

impl std::ops::Neg for IVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { x: -self.x, y: -self.y }
    }
}

impl From<Vec2> for IVec2 {
    fn from(v: Vec2) -> Self {
        Self { x: v.x as i32, y: v.y as i32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivec2_new() {
        let v = IVec2::new(1, 2);
        assert_eq!(v.x, 1);
        assert_eq!(v.y, 2);
    }

    #[test]
    fn test_ivec2_add() {
        let v1 = IVec2::new(1, 2);
        let v2 = IVec2::new(3, 4);
        assert_eq!(v1 + v2, IVec2::new(4, 6));
    }

    #[test]
    fn test_ivec2_as_vec2() {
        let v = IVec2::new(3, 4);
        let fv = v.as_vec2();
        assert_eq!(fv.x, 3.0);
        assert_eq!(fv.y, 4.0);
    }
}
