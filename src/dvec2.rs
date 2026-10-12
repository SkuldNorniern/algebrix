//! 2D float vector in double precision (f64). For document geometry where f32 runs out: large
//! canvases at deep zoom. Convert with [`From<Vec2>`] and [`as_vec2`](DVec2::as_vec2).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{DVec2, Vec2};
//! let a = DVec2::new(30000.0, 0.25);
//! let b = DVec2::from(Vec2::new(1.0, 0.0));
//! assert_eq!(a + b, DVec2::new(30001.0, 0.25));
//! ```

use crate::Vec2;

/// 2D vector in double precision (f64).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DVec2 {
    pub x: f64,
    pub y: f64,
}

impl DVec2 {
    /// Zero vector.
    pub const ZERO: DVec2 = DVec2 { x: 0.0, y: 0.0 };
    /// All components 1.
    pub const ONE: DVec2 = DVec2 { x: 1.0, y: 1.0 };
    /// Unit vector along +X.
    pub const X: DVec2 = DVec2 { x: 1.0, y: 0.0 };
    /// Unit vector along +Y.
    pub const Y: DVec2 = DVec2 { x: 0.0, y: 1.0 };

    /// Build from x, y.
    #[inline]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Both components set to `value`.
    #[inline]
    pub const fn splat(value: f64) -> Self {
        Self { x: value, y: value }
    }

    /// Dot product.
    #[inline]
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// 2D cross product (z of the 3D cross): positive when `other` is counter-clockwise.
    #[inline]
    pub fn perp_dot(self, other: Self) -> f64 {
        self.x * other.y - self.y * other.x
    }

    /// Euclidean length.
    #[inline]
    pub fn length(self) -> f64 {
        self.length_squared().sqrt()
    }

    /// Squared length (avoids sqrt).
    #[inline]
    pub fn length_squared(self) -> f64 {
        self.dot(self)
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
        self.try_normalize().unwrap_or(Self::ZERO)
    }

    /// Unit vector in the same direction, or `None` for a zero or non-finite length.
    #[inline]
    pub fn try_normalize(self) -> Option<Self> {
        let recip = 1.0 / self.length();
        (recip.is_finite() && recip > 0.0).then(|| self * recip)
    }

    /// Linear interpolation: `self * (1 - t) + other * t`.
    #[inline]
    pub fn lerp(self, other: Self, t: f64) -> Self {
        self + (other - self) * t
    }

    /// Componentwise minimum.
    #[inline]
    pub fn min(self, other: Self) -> Self {
        Self::new(self.x.min(other.x), self.y.min(other.y))
    }

    /// Componentwise maximum.
    #[inline]
    pub fn max(self, other: Self) -> Self {
        Self::new(self.x.max(other.x), self.y.max(other.y))
    }

    /// Componentwise absolute value.
    #[inline]
    pub fn abs(self) -> Self {
        Self::new(self.x.abs(), self.y.abs())
    }

    /// Componentwise floor.
    #[inline]
    pub fn floor(self) -> Self {
        Self::new(self.x.floor(), self.y.floor())
    }

    /// Componentwise ceil.
    #[inline]
    pub fn ceil(self) -> Self {
        Self::new(self.x.ceil(), self.y.ceil())
    }

    /// Componentwise round, half away from zero.
    #[inline]
    pub fn round(self) -> Self {
        Self::new(self.x.round(), self.y.round())
    }

    /// To single precision. Lossy: f32 keeps about 7 significant digits.
    #[inline]
    pub fn as_vec2(self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

impl From<Vec2> for DVec2 {
    #[inline]
    fn from(v: Vec2) -> Self {
        Self::new(f64::from(v.x), f64::from(v.y))
    }
}

impl std::ops::Add for DVec2 {
    type Output = Self;
    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y)
    }
}

impl std::ops::Add<f64> for DVec2 {
    type Output = Self;
    #[inline]
    fn add(self, s: f64) -> Self {
        Self::new(self.x + s, self.y + s)
    }
}

impl std::ops::AddAssign for DVec2 {
    #[inline]
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl std::ops::AddAssign<f64> for DVec2 {
    #[inline]
    fn add_assign(&mut self, s: f64) {
        *self = *self + s;
    }
}

impl std::ops::Sub for DVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y)
    }
}

impl std::ops::Sub<f64> for DVec2 {
    type Output = Self;
    #[inline]
    fn sub(self, s: f64) -> Self {
        Self::new(self.x - s, self.y - s)
    }
}

impl std::ops::SubAssign for DVec2 {
    #[inline]
    fn sub_assign(&mut self, o: Self) {
        *self = *self - o;
    }
}

impl std::ops::SubAssign<f64> for DVec2 {
    #[inline]
    fn sub_assign(&mut self, s: f64) {
        *self = *self - s;
    }
}

impl std::ops::Mul for DVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(self.x * o.x, self.y * o.y)
    }
}

impl std::ops::Mul<f64> for DVec2 {
    type Output = Self;
    #[inline]
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s)
    }
}

impl std::ops::MulAssign for DVec2 {
    #[inline]
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl std::ops::MulAssign<f64> for DVec2 {
    #[inline]
    fn mul_assign(&mut self, s: f64) {
        *self = *self * s;
    }
}

impl std::ops::Div for DVec2 {
    type Output = Self;
    #[inline]
    fn div(self, o: Self) -> Self {
        Self::new(self.x / o.x, self.y / o.y)
    }
}

impl std::ops::Div<f64> for DVec2 {
    type Output = Self;
    #[inline]
    fn div(self, s: f64) -> Self {
        Self::new(self.x / s, self.y / s)
    }
}

impl std::ops::DivAssign for DVec2 {
    #[inline]
    fn div_assign(&mut self, o: Self) {
        *self = *self / o;
    }
}

impl std::ops::DivAssign<f64> for DVec2 {
    #[inline]
    fn div_assign(&mut self, s: f64) {
        *self = *self / s;
    }
}

impl std::ops::Mul<DVec2> for f64 {
    type Output = DVec2;
    #[inline]
    fn mul(self, v: DVec2) -> DVec2 {
        v * self
    }
}

impl std::ops::Neg for DVec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_are_componentwise() {
        let a = DVec2::new(1.0, 2.0);
        let b = DVec2::new(4.0, 8.0);
        assert_eq!(a + b, DVec2::new(5.0, 10.0));
        assert_eq!(b - a, DVec2::new(3.0, 6.0));
        assert_eq!(a * b, DVec2::new(4.0, 16.0));
        assert_eq!(b / a, DVec2::new(4.0, 4.0));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(-a, DVec2::new(-1.0, -2.0));
        assert_eq!(a.dot(b), 20.0);
        assert_eq!(DVec2::X.perp_dot(DVec2::Y), 1.0);
    }

    #[test]
    fn normalize_handles_zero() {
        assert_eq!(DVec2::ZERO.try_normalize(), None);
        assert_eq!(DVec2::ZERO.normalize(), DVec2::ZERO);
        assert!((DVec2::new(3.0, 4.0).normalize() - DVec2::new(0.6, 0.8)).length() < 1e-15);
    }

    #[test]
    fn keeps_sub_pixel_detail_far_out() {
        // A 30000 px document at 3200% puts pixels near 1e6; f32 steps there are 0.0625.
        let p = DVec2::new(960_000.0, 960_000.0) + DVec2::splat(0.001);
        assert_eq!(p.x - 960_000.0, 0.001_000_000_047_497_451_4);
        assert_ne!(p.as_vec2().x - 960_000.0, 0.001);
    }

    #[test]
    fn rounding_and_conversion() {
        let v = DVec2::new(-1.5, 2.5);
        assert_eq!(v.floor(), DVec2::new(-2.0, 2.0));
        assert_eq!(v.ceil(), DVec2::new(-1.0, 3.0));
        assert_eq!(v.round(), DVec2::new(-2.0, 3.0));
        assert_eq!(v.abs(), DVec2::new(1.5, 2.5));
        assert_eq!(
            DVec2::from(Vec2::new(1.0, 2.0)).as_vec2(),
            Vec2::new(1.0, 2.0)
        );
    }
}
