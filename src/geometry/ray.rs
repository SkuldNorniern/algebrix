//! Ray: origin and direction (direction is stored normalized).
//!
//! Use [`at`](Ray::at) to get a point along the ray; [`intersect_plane`](Ray::intersect_plane) for plane hit;
//! [`closest_point_to`](Ray::closest_point_to) and [`distance_to_point`](Ray::distance_to_point) for point-to-ray distance.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Plane, Ray, Vec3};
//! let ray = Ray::new(Vec3::ZERO, Vec3::Z);
//! let plane = Plane::from_normal_point(Vec3::Z, Vec3::new(0.0, 0.0, 5.0));
//! let t = ray.intersect_plane(&plane);
//! assert!(t.is_some());
//! assert!((t.unwrap() - 5.0).abs() < 1e-5);
//! ```

use super::Plane;
use crate::Vec3;

/// Ray from `origin` along `direction`. Direction is normalized on creation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    /// Ray starting at `origin` and going along `direction` (normalized on creation).
    #[inline]
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    /// Point at `t`: `origin + direction * t`. Use `t >= 0` for points on the ray.
    #[inline]
    pub fn at(&self, t: f32) -> Vec3 {
        self.origin + self.direction * t
    }

    /// Closest point on the ray to `point`. If the perpendicular from `point` hits behind the origin, returns the origin.
    #[inline]
    pub fn closest_point_to(&self, point: Vec3) -> Vec3 {
        let to_point = point - self.origin;
        let t = self.direction.dot(to_point);
        self.at(t.max(0.0))
    }

    /// Distance from `point` to the ray (length of the perpendicular, or to the origin if the foot is behind).
    #[inline]
    pub fn distance_to_point(&self, point: Vec3) -> f32 {
        let closest = self.closest_point_to(point);
        (point - closest).length()
    }

    /// Intersection with a plane. Returns `Some(t)` so that [`at`](Self::at)(t) is on the plane, or `None` if parallel or behind.
    #[inline]
    pub fn intersect_plane(&self, plane: &Plane) -> Option<f32> {
        let denom = self.direction.dot(plane.normal);
        if denom.abs() < f32::EPSILON {
            return None;
        }
        let t = -(plane.normal.dot(self.origin) + plane.d) / denom;
        if t >= 0.0 { Some(t) } else { None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_at() {
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        assert_eq!(ray.at(5.0), Vec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn test_ray_closest_point() {
        let ray = Ray::new(Vec3::ZERO, Vec3::X);
        let point = Vec3::new(5.0, 3.0, 0.0);
        let closest = ray.closest_point_to(point);
        assert_eq!(closest, Vec3::new(5.0, 0.0, 0.0));
    }
}
