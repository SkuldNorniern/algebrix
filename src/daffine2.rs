//! 2D affine transform in double precision (f64): [`Affine2`](crate::Affine2) for document
//! geometry. Columns are `x_axis`, `y_axis` and `translation`.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{DAffine2, DVec2};
//!
//! let t = DAffine2::from_translation(DVec2::new(1e6, 0.0)) * DAffine2::from_scale(DVec2::splat(2.0));
//! assert_eq!(t.transform_point2(DVec2::new(0.25, 1.0)), DVec2::new(1_000_000.5, 2.0));
//! let back = t.inverse().unwrap().transform_point2(DVec2::new(1_000_000.5, 2.0));
//! assert_eq!(back, DVec2::new(0.25, 1.0));
//! ```

use crate::{Affine2, DVec2, Mat2, Vec2};

/// 2D affine transform in f64: a 2x2 linear part (as two columns) plus a translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DAffine2 {
    pub x_axis: DVec2,
    pub y_axis: DVec2,
    pub translation: DVec2,
}

/// The parts [`DAffine2::to_parts`] splits a transform into: it equals
/// `translate * rotate(angle) * skew_x(skew) * scale(scale)`.
///
/// A flip goes into `scale.y`: `scale.x` is never negative, so a mirrored
/// transform has a negative `scale.y` and keeps the angle of its x axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine2Parts {
    pub translation: DVec2,
    /// Radians, counter-clockwise, in `(-pi, pi]`.
    pub angle: f64,
    pub scale: DVec2,
    /// Horizontal shear factor: `x' = x + skew * y`, before rotation.
    pub skew: f64,
}

impl DAffine2 {
    /// Identity (no rotation, no scale, no translation).
    pub const IDENTITY: DAffine2 = DAffine2 {
        x_axis: DVec2::X,
        y_axis: DVec2::Y,
        translation: DVec2::ZERO,
    };

    /// Build from the linear part's columns and a translation.
    #[inline]
    pub const fn new(x_axis: DVec2, y_axis: DVec2, translation: DVec2) -> Self {
        Self {
            x_axis,
            y_axis,
            translation,
        }
    }

    /// Same as [`new`](Self::new): columns in order.
    #[inline]
    pub const fn from_cols(x_axis: DVec2, y_axis: DVec2, translation: DVec2) -> Self {
        Self::new(x_axis, y_axis, translation)
    }

    /// Translation only.
    #[inline]
    pub const fn from_translation(translation: DVec2) -> Self {
        Self::new(DVec2::X, DVec2::Y, translation)
    }

    /// Rotation by `angle` radians, counter-clockwise.
    #[inline]
    pub fn from_angle(angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self::new(DVec2::new(cos, sin), DVec2::new(-sin, cos), DVec2::ZERO)
    }

    /// Non-uniform scale only.
    #[inline]
    pub const fn from_scale(scale: DVec2) -> Self {
        Self::new(
            DVec2::new(scale.x, 0.0),
            DVec2::new(0.0, scale.y),
            DVec2::ZERO,
        )
    }

    /// Scale, then rotation by `angle`, then translation.
    #[inline]
    pub fn from_scale_angle_translation(scale: DVec2, angle: f64, translation: DVec2) -> Self {
        let r = Self::from_angle(angle);
        Self::new(r.x_axis * scale.x, r.y_axis * scale.y, translation)
    }

    /// Transform a point: linear part, then translation.
    #[inline]
    pub fn transform_point2(&self, p: DVec2) -> DVec2 {
        self.x_axis * p.x + self.y_axis * p.y + self.translation
    }

    /// Transform a direction: linear part only.
    #[inline]
    pub fn transform_vector2(&self, v: DVec2) -> DVec2 {
        self.x_axis * v.x + self.y_axis * v.y
    }

    /// Determinant of the linear part. Negative when the transform mirrors.
    #[inline]
    pub fn determinant(&self) -> f64 {
        self.x_axis.perp_dot(self.y_axis)
    }

    /// Inverse transform, or `None` when the linear part is singular or not finite.
    pub fn inverse(&self) -> Option<Self> {
        let inv = 1.0 / self.determinant();
        if !inv.is_finite() {
            return None;
        }
        let x_axis = DVec2::new(self.y_axis.y, -self.x_axis.y) * inv;
        let y_axis = DVec2::new(-self.y_axis.x, self.x_axis.x) * inv;
        let translation = -(x_axis * self.translation.x + y_axis * self.translation.y);
        Some(Self::new(x_axis, y_axis, translation))
    }

    /// Split into translation, rotation, skew and scale. See [`Affine2Parts`]
    /// for the order and the rule for flips. `None` when the x axis is zero.
    pub fn to_parts(&self) -> Option<Affine2Parts> {
        let sx = self.x_axis.length();
        if sx == 0.0 || !sx.is_finite() {
            return None;
        }
        let angle = self.x_axis.y.atan2(self.x_axis.x);
        let (sin, cos) = angle.sin_cos();
        // The y axis with the rotation taken back off: (skew * sy, sy).
        let local = DVec2::new(
            cos * self.y_axis.x + sin * self.y_axis.y,
            -sin * self.y_axis.x + cos * self.y_axis.y,
        );
        let sy = local.y;
        let skew = if sy == 0.0 { 0.0 } else { local.x / sy };
        Some(Affine2Parts {
            translation: self.translation,
            angle,
            scale: DVec2::new(sx, sy),
            skew,
        })
    }

    /// Rebuild from [`to_parts`](Self::to_parts)' output.
    pub fn from_parts(parts: Affine2Parts) -> Self {
        let r = Self::from_angle(parts.angle);
        let Affine2Parts { scale, skew, .. } = parts;
        Self::new(
            r.x_axis * scale.x,
            r.x_axis * (skew * scale.y) + r.y_axis * scale.y,
            parts.translation,
        )
    }

    /// To single precision. Lossy.
    #[inline]
    pub fn as_affine2(&self) -> Affine2 {
        Affine2::new(
            Mat2::from_cols(self.x_axis.as_vec2(), self.y_axis.as_vec2()),
            self.translation.as_vec2(),
        )
    }
}

impl Default for DAffine2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<Affine2> for DAffine2 {
    fn from(a: Affine2) -> Self {
        let x: Vec2 = a.matrix * Vec2::X;
        let y: Vec2 = a.matrix * Vec2::Y;
        Self::new(DVec2::from(x), DVec2::from(y), DVec2::from(a.translation))
    }
}

impl std::ops::Mul for DAffine2 {
    type Output = Self;
    /// `self * other` applies `other` first.
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self::new(
            self.transform_vector2(other.x_axis),
            self.transform_vector2(other.y_axis),
            self.transform_point2(other.translation),
        )
    }
}

impl std::ops::MulAssign for DAffine2 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_6};

    fn close(a: DVec2, b: DVec2, tol: f64) -> bool {
        (a - b).abs().x <= tol && (a - b).abs().y <= tol
    }

    #[test]
    fn rotation_and_inverse() {
        let r = DAffine2::from_angle(FRAC_PI_2);
        assert!(close(r.transform_vector2(DVec2::X), DVec2::Y, 1e-15));
        let t = DAffine2::from_scale_angle_translation(
            DVec2::new(2.0, 3.0),
            0.3,
            DVec2::new(5.0, -7.0),
        );
        let id = t * t.inverse().unwrap();
        assert!(close(id.x_axis, DVec2::X, 1e-15));
        assert!(close(id.y_axis, DVec2::Y, 1e-15));
        assert!(close(id.translation, DVec2::ZERO, 1e-14));
        assert_eq!(DAffine2::from_scale(DVec2::new(1.0, 0.0)).inverse(), None);
    }

    /// A pan/zoom/rotate chain applied thousands of times out at 1e6 px stays
    /// within 1e-6 px of where it started once its inverse is applied.
    #[test]
    fn long_chains_stay_exact_far_out() {
        let step = DAffine2::from_translation(DVec2::new(1e6, -1e6))
            * DAffine2::from_angle(0.001)
            * DAffine2::from_scale(DVec2::splat(1.0001))
            * DAffine2::from_translation(DVec2::new(-1e6, 1e6));
        let mut chain = DAffine2::IDENTITY;
        for _ in 0..2000 {
            chain *= step;
        }
        let back = chain.inverse().unwrap();
        let p = DVec2::new(1_000_123.456, -999_876.543);
        let round_trip = back.transform_point2(chain.transform_point2(p));
        assert!(close(round_trip, p, 1e-6), "{round_trip:?}");
    }

    #[test]
    fn parts_round_trip() {
        let parts = Affine2Parts {
            translation: DVec2::new(10.0, 20.0),
            angle: FRAC_PI_6,
            scale: DVec2::new(2.0, 0.5),
            skew: 0.25,
        };
        let t = DAffine2::from_parts(parts);
        let got = t.to_parts().unwrap();
        assert!((got.angle - parts.angle).abs() < 1e-12);
        assert!(close(got.scale, parts.scale, 1e-12));
        assert!((got.skew - parts.skew).abs() < 1e-12);
        assert!(close(DAffine2::from_parts(got).y_axis, t.y_axis, 1e-12));
    }

    #[test]
    fn a_flip_lands_in_scale_y() {
        let mirror = DAffine2::from_scale(DVec2::new(-1.0, 1.0));
        let parts = mirror.to_parts().unwrap();
        assert_eq!(parts.scale.x, 1.0);
        assert!(parts.scale.y < 0.0);
        assert!(close(
            DAffine2::from_parts(parts).x_axis,
            mirror.x_axis,
            1e-15
        ));
        assert!(close(
            DAffine2::from_parts(parts).y_axis,
            mirror.y_axis,
            1e-15
        ));
    }

    #[test]
    fn converts_with_affine2() {
        let a =
            Affine2::from_scale_angle_translation(Vec2::new(2.0, 3.0), 0.5, Vec2::new(1.0, 2.0));
        let d = DAffine2::from(a);
        let p = Vec2::new(0.5, -0.25);
        assert!((d.as_affine2().transform_point2(p) - a.transform_point2(p)).length() < 1e-6);
    }
}
