//! 2x2 column-major matrix. Used for 2D rotation, scale, or combined linear transform.
//!
//! Multiply a Vec2 on the right: `matrix * point`. Use [`from_angle`](Mat2::from_angle) for
//! rotation (radians), [`from_scale`](Mat2::from_scale) for scale, [`from_diagonal`](Mat2::from_diagonal) for non-uniform scale.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Mat2, Vec2};
//!
//! let rot = Mat2::from_angle(std::f32::consts::FRAC_PI_2);
//! let x = Vec2::X;
//! let y = rot * x;
//! assert!((y.x.abs() < 1e-5) && (y.y - 1.0).abs() < 1e-5);
//!
//! let scale = Mat2::from_scale(Vec2::new(2.0, 3.0));
//! let p = Vec2::new(1.0, 1.0);
//! assert_eq!(scale * p, Vec2::new(2.0, 3.0));
//! ```

use crate::Vec2;

/// 2x2 column-major matrix for 2D rotation, scale, or combined linear transform.
///
/// Multiply a Vec2 on the right: `matrix * point`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat2 {
    pub x_axis: Vec2,
    pub y_axis: Vec2,
}

impl Mat2 {
    /// Identity matrix (no rotation, no scale).
    pub const IDENTITY: Mat2 = Mat2 {
        x_axis: Vec2::X,
        y_axis: Vec2::Y,
    };

    /// Zero matrix (all elements 0).
    pub const ZERO: Mat2 = Mat2 {
        x_axis: Vec2::ZERO,
        y_axis: Vec2::ZERO,
    };

    /// Build from two column vectors.
    #[inline(always)]
    pub const fn new(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self { x_axis, y_axis }
    }

    /// Same as [`new`](Mat2::new); build from columns.
    #[inline(always)]
    pub const fn from_cols(x_axis: Vec2, y_axis: Vec2) -> Self {
        Self { x_axis, y_axis }
    }

    /// Diagonal matrix with (diagonal.x, diagonal.y) on the diagonal.
    #[inline(always)]
    pub const fn from_diagonal(diagonal: Vec2) -> Self {
        Self {
            x_axis: Vec2::new(diagonal.x, 0.0),
            y_axis: Vec2::new(0.0, diagonal.y),
        }
    }

    /// Rotation matrix: angle in radians, counter-clockwise. Column 0 is (cos, sin), column 1 is (-sin, cos).
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            x_axis: Vec2::new(cos, sin),
            y_axis: Vec2::new(-sin, cos),
        }
    }

    /// Scale matrix: diagonal (scale.x, scale.y). Use `matrix * point` to scale a point.
    #[inline(always)]
    pub fn from_scale(scale: Vec2) -> Self {
        Self::from_diagonal(scale)
    }

    /// Transpose: swap rows and columns.
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            x_axis: Vec2::new(self.x_axis.x, self.y_axis.x),
            y_axis: Vec2::new(self.x_axis.y, self.y_axis.y),
        }
    }

    /// Determinant (x_axis.x * y_axis.y - x_axis.y * y_axis.x).
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x
    }

    /// Inverse matrix. Returns `None` if determinant is zero (singular).
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = det.recip();
        Some(Self {
            x_axis: Vec2::new(self.y_axis.y * inv_det, -self.x_axis.y * inv_det),
            y_axis: Vec2::new(-self.y_axis.x * inv_det, self.x_axis.x * inv_det),
        })
    }

    /// Multiply matrix by a 2D vector.
    #[inline]
    pub fn mul_vec2(self, other: Vec2) -> Vec2 {
        Vec2::new(
            self.x_axis.x * other.x + self.y_axis.x * other.y,
            self.x_axis.y * other.x + self.y_axis.y * other.y,
        )
    }
}

impl std::ops::Mul for Mat2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x_axis: self.mul_vec2(other.x_axis),
            y_axis: self.mul_vec2(other.y_axis),
        }
    }
}

impl std::ops::Mul<Vec2> for Mat2 {
    type Output = Vec2;
    #[inline]
    fn mul(self, other: Vec2) -> Vec2 {
        self.mul_vec2(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat2_identity() {
        let m = Mat2::IDENTITY;
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(m * v, v);
    }

    #[test]
    fn test_mat2_determinant() {
        assert_eq!(Mat2::IDENTITY.determinant(), 1.0);
    }

    #[test]
    fn test_mat2_inverse() {
        let m = Mat2::IDENTITY;
        let inv = m.inverse().expect("IDENTITY is invertible");
        assert_eq!(inv, Mat2::IDENTITY);
    }
}
