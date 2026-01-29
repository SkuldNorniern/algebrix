//! View frustum: 6 planes from a projection matrix. Use [`from_projection`](Frustum::from_projection);
//! then [`contains_point`](Frustum::contains_point), [`contains_sphere`](Frustum::contains_sphere), [`contains_aabb`](Frustum::contains_aabb) for culling.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Frustum, Mat4, Vec3};
//!
//! let proj = Mat4::perspective_lh(
//!     std::f32::consts::FRAC_PI_4,
//!     16.0 / 9.0,
//!     0.1,
//!     100.0,
//! );
//! let frustum = Frustum::from_projection(proj);
//! let origin = Vec3::ZERO;
//! let inside = frustum.contains_point(origin);
//! let sphere_inside = frustum.contains_sphere(Vec3::new(0.0, 0.0, -1.0), 0.5);
//! ```

use crate::{Mat4, Vec3};
use super::Plane;

/// View frustum: six planes (left, right, bottom, top, near, far) from a projection matrix.
///
/// Use [`from_projection`](Frustum::from_projection) with a view-projection matrix; then
/// [`contains_point`](Frustum::contains_point), [`contains_sphere`](Frustum::contains_sphere), or [`contains_aabb`](Frustum::contains_aabb) for culling.
#[derive(Debug, Clone, Copy)]
pub struct Frustum {
    pub planes: [Plane; 6],
}

impl Frustum {
    /// Build the six frustum planes from a projection matrix (perspective or orthographic). Plane order: left, right, bottom, top, near, far.
    #[inline]
    pub fn from_projection(projection: Mat4) -> Self {
        let m = projection.as_ref();
        
        let left = Plane::from_normal_d(
            Vec3::new(m[3] + m[0], m[7] + m[4], m[11] + m[8]).normalize(),
            m[15] + m[12],
        ).normalize();

        let right = Plane::from_normal_d(
            Vec3::new(m[3] - m[0], m[7] - m[4], m[11] - m[8]).normalize(),
            m[15] - m[12],
        ).normalize();

        let bottom = Plane::from_normal_d(
            Vec3::new(m[3] + m[1], m[7] + m[5], m[11] + m[9]).normalize(),
            m[15] + m[13],
        ).normalize();

        let top = Plane::from_normal_d(
            Vec3::new(m[3] - m[1], m[7] - m[5], m[11] - m[9]).normalize(),
            m[15] - m[13],
        ).normalize();

        let near = Plane::from_normal_d(
            Vec3::new(m[3] + m[2], m[7] + m[6], m[11] + m[10]).normalize(),
            m[15] + m[14],
        ).normalize();

        let far = Plane::from_normal_d(
            Vec3::new(m[3] - m[2], m[7] - m[6], m[11] - m[10]).normalize(),
            m[15] - m[14],
        ).normalize();

        Self {
            planes: [left, right, bottom, top, near, far],
        }
    }

    /// True if the point is inside all six frustum planes.
    #[inline]
    pub fn contains_point(&self, point: Vec3) -> bool {
        self.planes.iter().all(|plane| plane.distance_to_point(point) >= 0.0)
    }

    /// True if the sphere (center, radius) is inside or intersects all six planes.
    #[inline]
    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes.iter().all(|plane| plane.distance_to_point(center) >= -radius)
    }

    /// True if the AABB intersects the frustum (at least one corner in front of each plane).
    #[inline]
    pub fn contains_aabb(&self, aabb: &super::Aabb3) -> bool {
        for plane in &self.planes {
            let corners = aabb.corners();
            let mut all_behind = true;
            for corner in &corners {
                if plane.distance_to_point(*corner) >= 0.0 {
                    all_behind = false;
                    break;
                }
            }
            if all_behind {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::aabb::Aabb3;

    #[test]
    fn test_frustum_creation() {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 1.0, 0.1, 100.0);
        let frustum = Frustum::from_projection(proj);
        assert_eq!(frustum.planes.len(), 6);
    }

    #[test]
    fn test_frustum_contains_aabb() {
        let proj = Mat4::orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.1, 100.0);
        let frustum = Frustum::from_projection(proj);
        let aabb = Aabb3::new(Vec3::new(-0.5, -0.5, -1.0), Vec3::new(0.5, 0.5, -0.5));
        let result = frustum.contains_aabb(&aabb);
        assert!(result);
    }
}
