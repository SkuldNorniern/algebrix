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

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Angle in radians. Use [`new`](Rad::new) or `Rad::from(deg)` from degrees.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Rad(pub f32);

impl Rad {
    /// Angle from radians. For degrees use `Rad::from(Deg::new(90.0))`.
    #[inline(always)]
    pub const fn new(radians: f32) -> Self {
        Self(radians)
    }

    /// Same angle in degrees.
    #[inline(always)]
    pub fn to_deg(self) -> Deg {
        Deg(self.0 * 180.0 / std::f32::consts::PI)
    }

    /// Value in radians (f32).
    #[inline(always)]
    pub fn as_rad(self) -> f32 {
        self.0
    }

    /// Wrap into [0, 2π); useful for canonical angle comparison.
    #[inline]
    pub fn normalize(self) -> Self {
        let two_pi = std::f32::consts::TAU;
        let mut angle = self.0 % two_pi;
        if angle < 0.0 {
            angle += two_pi;
        }
        Self(angle)
    }

    /// Wrap into (-π, π]; the shortest signed turn, e.g. for angle
    /// differences.
    #[inline]
    pub fn normalize_signed(self) -> Self {
        use std::f32::consts::{PI, TAU};
        let a = self.0;
        // in range already: untouched, so repeated wrapping loses nothing
        if a > -PI && a <= PI {
            return self;
        }
        let w = a - TAU * (a / TAU).round();
        Self(if w <= -PI {
            w + TAU
        } else if w > PI {
            w - TAU
        } else {
            w
        })
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
    /// Angle from degrees.
    #[inline(always)]
    pub const fn new(degrees: f32) -> Self {
        Self(degrees)
    }

    /// Same angle in radians.
    #[inline(always)]
    pub fn to_rad(self) -> Rad {
        Rad(self.0 * std::f32::consts::PI / 180.0)
    }

    /// Value in degrees (f32).
    #[inline(always)]
    pub fn as_deg(self) -> f32 {
        self.0
    }

    /// Wrap into [0, 360); useful for canonical angle comparison.
    #[inline]
    pub fn normalize(self) -> Self {
        let mut angle = self.0 % 360.0;
        if angle < 0.0 {
            angle += 360.0;
        }
        Self(angle)
    }

    /// Wrap into (-180, 180]; the shortest signed turn.
    #[inline]
    pub fn normalize_signed(self) -> Self {
        let a = self.0;
        if a > -180.0 && a <= 180.0 {
            return self;
        }
        let w = a - 360.0 * (a / 360.0).round();
        Self(if w <= -180.0 {
            w + 360.0
        } else if w > 180.0 {
            w - 360.0
        } else {
            w
        })
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

/// Order of axis rotations for Euler angles (e.g. XYZ = rotate X, then Y, then Z). Used with [`Quat::from_euler`](crate::Quat::from_euler) and [`Quat::to_euler`](crate::Quat::to_euler).
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

    #[test]
    fn normalize_signed_takes_the_short_way() {
        use std::f32::consts::{PI, TAU};
        assert!((Rad::new(TAU - 0.1).normalize_signed().0 + 0.1).abs() < 1e-5);
        assert!((Rad::new(-TAU + 0.1).normalize_signed().0 - 0.1).abs() < 1e-5);
        assert!((Rad::new(PI).normalize_signed().0 - PI).abs() < 1e-5);
        assert!((Rad::new(3.0 * PI + 0.2).normalize_signed().0 - (-PI + 0.2)).abs() < 1e-4);
        assert!((Deg::new(350.0).normalize_signed().0 + 10.0).abs() < 1e-4);
        assert!((Deg::new(-190.0).normalize_signed().0 - 170.0).abs() < 1e-4);
        // in range comes back bit for bit, so repeated wrapping does not creep
        assert_eq!(
            Rad::new(-0.3).normalize_signed().0.to_bits(),
            (-0.3_f32).to_bits()
        );
        assert!((Rad::new(-PI).normalize_signed().0 - PI).abs() < 1e-5);
    }
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
