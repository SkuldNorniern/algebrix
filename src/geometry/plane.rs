//! Infinite plane: normal and distance from origin.
//!
//! Use [`from_normal_point`](Plane::from_normal_point) to build from a point and normal;
//! [`distance_to_point`](Plane::distance_to_point) for signed distance, [`project_point`](Plane::project_point) to snap a point onto the plane.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Plane, Vec3};
//! let plane = Plane::from_normal_point(Vec3::Z, Vec3::new(0.0, 0.0, 5.0));
//! assert!((plane.distance_to_point(Vec3::ZERO) + 5.0).abs() < 1e-5);
//! let on_plane = plane.project_point(Vec3::new(1.0, 2.0, 10.0));
//! assert!((plane.distance_to_point(on_plane)).abs() < 1e-5);
//! ```

use crate::Vec3;

/// Plane equation: `normal · point + d = 0`. Normal should be unit length for correct distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    pub normal: Vec3,
    pub d: f32,
}

impl Plane {
    /// Plane through `point` with the given `normal` (will be normalized). Positive half-space is where normal points.
    ///
    /// # Example
    ///
    /// ```rust
    /// use algebrix::{Plane, Vec3};
    /// let p = Plane::from_normal_point(Vec3::Y, Vec3::ZERO);
    /// assert!(p.distance_to_point(Vec3::new(0.0, 1.0, 0.0)) > 0.0);
    /// ```
    #[inline]
    pub fn from_normal_point(normal: Vec3, point: Vec3) -> Self {
        let normal = normal.normalize();
        Self {
            normal,
            d: -normal.dot(point),
        }
    }

    /// Plane from precomputed normal and distance (equation: `normal · p + d = 0`).
    #[inline]
    pub const fn from_normal_d(normal: Vec3, d: f32) -> Self {
        Self { normal, d }
    }

    /// Signed distance from a point to the plane. Positive means on the side the normal points to.
    #[inline]
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }

    /// Orthogonal projection of a point onto the plane.
    #[inline]
    pub fn project_point(&self, point: Vec3) -> Vec3 {
        let dist = self.distance_to_point(point);
        point - self.normal * dist
    }

    /// True if the point is on the side the normal points to (positive half-space).
    #[inline]
    pub fn is_point_in_front(&self, point: Vec3) -> bool {
        self.distance_to_point(point) > 0.0
    }

    /// Normalize the plane so the normal has length 1; `d` is scaled accordingly.
    #[inline]
    pub fn normalize(self) -> Self {
        let len = self.normal.length();
        if len > 0.0 {
            Self {
                normal: self.normal / len,
                d: self.d / len,
            }
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_from_normal_point() {
        let normal = Vec3::Z;
        let point = Vec3::new(0.0, 0.0, 5.0);
        let plane = Plane::from_normal_point(normal, point);
        assert!((plane.distance_to_point(point)).abs() < 0.0001);
    }

    #[test]
    fn test_plane_project_point() {
        let plane = Plane::from_normal_point(Vec3::Z, Vec3::ZERO);
        let point = Vec3::new(1.0, 2.0, 5.0);
        let projected = plane.project_point(point);
        assert!((projected.z).abs() < 0.0001);
    }
}
