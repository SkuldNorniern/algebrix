//! 2D affine transform: scale/rotation plus translation. No perspective; cheaper than a full 3x3 for 2D.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Affine2, Vec2};
//!
//! let t = Affine2::from_translation(Vec2::new(10.0, 0.0));
//! let p = Vec2::ZERO;
//! assert_eq!(t.transform_point2(p), Vec2::new(10.0, 0.0));
//!
//! let r = Affine2::from_angle(std::f32::consts::FRAC_PI_2);
//! let v = Vec2::X;
//! let w = r.transform_vector2(v);
//! assert!((w.y - 1.0).abs() < 1e-5);
//! ```

use crate::{Mat2, Vec2};

/// 2D affine transform: a 2x2 linear part (rotation/scale) plus a translation.
///
/// Use [`transform_point2`](Affine2::transform_point2) for points and
/// [`transform_vector2`](Affine2::transform_vector2) for directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2 {
    pub matrix: Mat2,
    pub translation: Vec2,
}

impl Affine2 {
    /// Identity (no rotation, no scale, no translation).
    pub const IDENTITY: Affine2 = Affine2 {
        matrix: Mat2::IDENTITY,
        translation: Vec2::ZERO,
    };

    /// Build from a 2x2 matrix and translation.
    #[inline(always)]
    pub const fn new(matrix: Mat2, translation: Vec2) -> Self {
        Self { matrix, translation }
    }

    /// Translation only (identity rotation/scale).
    #[inline]
    pub fn from_translation(translation: Vec2) -> Self {
        Self {
            matrix: Mat2::IDENTITY,
            translation,
        }
    }

    /// Rotation by `angle` radians (counter-clockwise); no translation.
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        Self {
            matrix: Mat2::from_angle(angle),
            translation: Vec2::ZERO,
        }
    }

    /// Non-uniform scale only; no rotation or translation.
    #[inline]
    pub fn from_scale(scale: Vec2) -> Self {
        Self {
            matrix: Mat2::from_scale(scale),
            translation: Vec2::ZERO,
        }
    }

    /// Rotation by `angle` and translation.
    #[inline]
    pub fn from_angle_translation(angle: f32, translation: Vec2) -> Self {
        Self {
            matrix: Mat2::from_angle(angle),
            translation,
        }
    }

    /// Full 2D transform: scale, then rotation by `angle`, then translation.
    #[inline]
    pub fn from_scale_angle_translation(scale: Vec2, angle: f32, translation: Vec2) -> Self {
        Self {
            matrix: Mat2::from_angle(angle) * Mat2::from_scale(scale),
            translation,
        }
    }

    /// Transform a 2D point: apply rotation/scale then add translation.
    #[inline]
    pub fn transform_point2(self, point: Vec2) -> Vec2 {
        self.matrix * point + self.translation
    }

    /// Transform a 2D direction: rotation/scale only (translation ignored).
    #[inline]
    pub fn transform_vector2(self, vector: Vec2) -> Vec2 {
        self.matrix * vector
    }

    /// Inverse affine transform. Returns `None` if the linear part is singular.
    #[inline]
    pub fn inverse(self) -> Option<Self> {
        let mat_inv = self.matrix.inverse()?;
        Some(Self {
            matrix: mat_inv,
            translation: mat_inv * (-self.translation),
        })
    }
}

impl std::ops::Mul for Affine2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            matrix: self.matrix * other.matrix,
            translation: self.matrix * other.translation + self.translation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine2_identity() {
        let a = Affine2::IDENTITY;
        let p = Vec2::new(1.0, 2.0);
        assert_eq!(a.transform_point2(p), p);
    }

    #[test]
    fn test_affine2_translation() {
        let a = Affine2::from_translation(Vec2::new(1.0, 2.0));
        let p = Vec2::ZERO;
        assert_eq!(a.transform_point2(p), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_affine2_inverse() {
        let a = Affine2::from_translation(Vec2::new(1.0, 2.0));
        let inv = a.inverse().expect("translation is invertible");
        let result = a * inv;
        assert!((result.translation.x - 0.0).abs() < 0.0001);
        assert!((result.translation.y - 0.0).abs() < 0.0001);
    }
}
