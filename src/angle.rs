//! Type-safe angles: [`Rad`] and [`Deg`], plus [`EulerRot`] for Euler order.
//!
//! Use `Rad::from(Deg::new(90.0))` to convert; [`Rad::normalize`](Rad::normalize) / [`Deg::normalize`](Deg::normalize)
//! to wrap into [0, 2π) or [0, 360).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Rad, Deg};
//!
//! let half_turn = Deg::new(180.0);
//! let rad = Rad::from(half_turn);
//! assert!((rad.as_rad() - std::f32::consts::PI).abs() < 1e-5);
//!
//! let a = Rad::new(3.0 * std::f32::consts::TAU);
//! let b = a.normalize();
//! assert!(b.as_rad() >= 0.0 && b.as_rad() < std::f32::consts::TAU);
//! ```

use std::ops::{Add, Sub, Mul, Div, Neg};

/// Angle in radians. Use [`new`](Rad::new) or `Rad::from(deg)` from degrees.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Rad(pub f32);

impl Rad {
    /// Create from radians
    #[inline(always)]
    pub const fn new(radians: f32) -> Self {
        Self(radians)
    }

    /// Convert to degrees
    #[inline(always)]
    pub fn to_deg(self) -> Deg {
        Deg(self.0 * 180.0 / std::f32::consts::PI)
    }

    /// Get the value in radians
    #[inline(always)]
    pub fn as_rad(self) -> f32 {
        self.0
    }

    /// Normalize to [0, 2π)
    #[inline]
    pub fn normalize(self) -> Self {
        let two_pi = std::f32::consts::TAU;
        let mut angle = self.0 % two_pi;
        if angle < 0.0 {
            angle += two_pi;
        }
        Self(angle)
    }
}

impl From<Deg> for Rad {
    #[inline(always)]
    fn from(deg: Deg) -> Self {
        deg.to_rad()
    }
}

impl From<f32> for Rad {
    #[inline(always)]
    fn from(rad: f32) -> Self {
        Self(rad)
    }
}

impl Add for Rad {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Rad {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<f32> for Rad {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self(self.0 * scalar)
    }
}

impl Div<f32> for Rad {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self(self.0 / scalar)
    }
}

impl Neg for Rad {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Angle in degrees. Use [`new`](Deg::new); convert to radians with [`to_rad`](Deg::to_rad) or `Rad::from(deg)`.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Deg(pub f32);

impl Deg {
    /// Create from degrees
    #[inline(always)]
    pub const fn new(degrees: f32) -> Self {
        Self(degrees)
    }

    /// Convert to radians
    #[inline(always)]
    pub fn to_rad(self) -> Rad {
        Rad(self.0 * std::f32::consts::PI / 180.0)
    }

    /// Get the value in degrees
    #[inline(always)]
    pub fn as_deg(self) -> f32 {
        self.0
    }

    /// Normalize to [0, 360)
    #[inline]
    pub fn normalize(self) -> Self {
        let mut angle = self.0 % 360.0;
        if angle < 0.0 {
            angle += 360.0;
        }
        Self(angle)
    }
}

impl From<Rad> for Deg {
    #[inline(always)]
    fn from(rad: Rad) -> Self {
        rad.to_deg()
    }
}

impl From<f32> for Deg {
    #[inline(always)]
    fn from(deg: f32) -> Self {
        Self(deg)
    }
}

impl Add for Deg {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }
}

impl Sub for Deg {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0)
    }
}

impl Mul<f32> for Deg {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self(self.0 * scalar)
    }
}

impl Div<f32> for Deg {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        Self(self.0 / scalar)
    }
}

impl Neg for Deg {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

/// Euler rotation order for type-safe Euler angle handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum EulerRot {
    XYZ,
    XZY,
    YXZ,
    YZX,
    ZXY,
    ZYX,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rad_deg_conversion() {
        let rad = Rad::new(std::f32::consts::PI);
        let deg = rad.to_deg();
        assert!((deg.as_deg() - 180.0).abs() < 0.0001);
    }

    #[test]
    fn test_deg_rad_conversion() {
        let deg = Deg::new(90.0);
        let rad = deg.to_rad();
        assert!((rad.as_rad() - std::f32::consts::FRAC_PI_2).abs() < 0.0001);
    }
}
