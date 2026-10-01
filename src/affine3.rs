//! 3D affine transform: scale/rotation plus translation. No perspective; cheaper than a full 4x4 for 3D.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Affine3, Quat, Vec3};
//!
//! let t = Affine3::from_translation(Vec3::new(1.0, 2.0, 3.0));
//! let p = Vec3::ZERO;
//! assert_eq!(t.transform_point3(p), Vec3::new(1.0, 2.0, 3.0));
//!
//! let r = Affine3::from_quat(Quat::IDENTITY);
//! let v = Vec3::X;
//! assert!((r.transform_vector3(v) - v).length() < 1e-5);
//! ```

use crate::{Mat3, Quat, Vec3};

/// 3D affine transform: a 3x3 linear part (rotation/scale) plus a translation.
///
/// No perspective; use [`transform_point3`](Affine3::transform_point3) for points and
/// [`transform_vector3`](Affine3::transform_vector3) for directions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine3 {
    pub matrix: Mat3,
    pub translation: Vec3,
}

impl Affine3 {
    /// Identity (no rotation, no scale, no translation).
    pub const IDENTITY: Affine3 = Affine3 {
        matrix: Mat3::IDENTITY,
        translation: Vec3::ZERO,
    };

    /// Build from a 3x3 matrix and translation.
    #[inline(always)]
    pub const fn new(matrix: Mat3, translation: Vec3) -> Self {
        Self {
            matrix,
            translation,
        }
    }

    /// Translation only (identity rotation/scale).
    #[inline]
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            matrix: Mat3::IDENTITY,
            translation,
        }
    }

    /// Rotation from a unit quaternion; no translation.
    #[inline]
    pub fn from_quat(quat: Quat) -> Self {
        Self {
            matrix: Mat3::from_quat(quat),
            translation: Vec3::ZERO,
        }
    }

    /// Non-uniform scale only; no rotation or translation.
    #[inline]
    pub fn from_scale(scale: Vec3) -> Self {
        Self {
            matrix: Mat3::from_nonuniform_scale(scale),
            translation: Vec3::ZERO,
        }
    }

    /// Rotation from quat and translation.
    #[inline]
    pub fn from_rotation_translation(quat: Quat, translation: Vec3) -> Self {
        Self {
            matrix: Mat3::from_quat(quat),
            translation,
        }
    }

    /// Full SRT: scale, then rotation from quat, then translation.
    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, quat: Quat, translation: Vec3) -> Self {
        let mut mat = Mat3::from_quat(quat);
        mat.x_axis *= scale.x;
        mat.y_axis *= scale.y;
        mat.z_axis *= scale.z;
        Self {
            matrix: mat,
            translation,
        }
    }

    /// Transform a 3D point: apply rotation/scale then add translation.
    #[inline]
    pub fn transform_point3(self, point: Vec3) -> Vec3 {
        self.matrix * point + self.translation
    }

    /// Transform a 3D direction: rotation/scale only (translation ignored).
    #[inline]
    pub fn transform_vector3(self, vector: Vec3) -> Vec3 {
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

    /// Right-handed look-at: from `eye` toward `center`, with `up` as the up vector. Useful for camera view.
    #[inline]
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            matrix: Mat3::from_cols(
                Vec3::new(s.x, u.x, -f.x),
                Vec3::new(s.y, u.y, -f.y),
                Vec3::new(s.z, u.z, -f.z),
            ),
            translation: Vec3::new(-s.dot(eye), -u.dot(eye), f.dot(eye)),
        }
    }
}

impl std::ops::Mul for Affine3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            matrix: self.matrix * other.matrix,
            translation: self.matrix * other.translation + self.translation,
        }
    }
}

impl std::ops::Mul<Vec3> for Affine3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, point: Vec3) -> Vec3 {
        self.transform_point3(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine3_identity() {
        let a = Affine3::IDENTITY;
        let p = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(a.transform_point3(p), p);
    }

    #[test]
    fn test_affine3_translation() {
        let a = Affine3::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::ZERO;
        assert_eq!(a.transform_point3(p), Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_affine3_mul() {
        let a = Affine3::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let b = Affine3::from_translation(Vec3::new(0.0, 1.0, 0.0));
        let c = a * b;
        let p = Vec3::ZERO;
        assert_eq!(c.transform_point3(p), Vec3::new(1.0, 1.0, 0.0));
    }
}
