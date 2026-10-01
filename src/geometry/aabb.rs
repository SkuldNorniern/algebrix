//! Axis-aligned bounding boxes: [`Aabb2`] and [`Aabb3`]. Min/max corners; [`contains`](Aabb2::contains),
//! [`intersects`](Aabb2::intersects), [`union`](Aabb2::union), [`expand`](Aabb2::expand).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Aabb2, Vec2};
//!
//! let box1 = Aabb2::new(Vec2::ZERO, Vec2::new(2.0, 2.0));
//! assert!(box1.contains(Vec2::new(1.0, 1.0)));
//! assert!(!box1.contains(Vec2::new(3.0, 0.0)));
//!
//! let box2 = Aabb2::new(Vec2::new(1.0, 1.0), Vec2::new(3.0, 3.0));
//! assert!(box1.intersects(&box2));
//! let u = box1.union(&box2);
//! assert!(u.contains(Vec2::new(2.0, 2.0)));
//! ```

use crate::{Vec2, Vec3};

/// 2D axis-aligned bounding box (min and max corners).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb2 {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb2 {
    /// Box with the given min and max corners.
    #[inline]
    pub const fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// Box centered at `center` with the given `size` (half-extents applied on each axis).
    #[inline]
    pub fn from_center_size(center: Vec2, size: Vec2) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    /// Center of the box (midpoint of min and max).
    #[inline]
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Size per axis (max - min).
    #[inline]
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// True if the point is inside or on the boundary of the box.
    #[inline]
    pub fn contains(&self, point: Vec2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }

    /// True if this box and `other` overlap (intersect).
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Smallest box that contains both this box and `other`.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vec2::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
            ),
            max: Vec2::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
            ),
        }
    }

    /// Box expanded by `margin` on each side (min -= margin, max += margin).
    #[inline]
    pub fn expand(&self, margin: Vec2) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }
}

/// 3D axis-aligned bounding box (min and max corners).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb3 {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb3 {
    /// Box with the given min and max corners.
    #[inline]
    pub const fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Box centered at `center` with the given `size` (half-extents applied on each axis).
    #[inline]
    pub fn from_center_size(center: Vec3, size: Vec3) -> Self {
        let half = size * 0.5;
        Self {
            min: center - half,
            max: center + half,
        }
    }

    /// Center of the box (midpoint of min and max).
    #[inline]
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Size per axis (max - min).
    #[inline]
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// True if the point is inside or on the boundary of the box.
    #[inline]
    pub fn contains(&self, point: Vec3) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
            && point.z >= self.min.z
            && point.z <= self.max.z
    }

    /// True if this box and `other` overlap (intersect).
    #[inline]
    pub fn intersects(&self, other: &Self) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
            && self.min.z <= other.max.z
            && self.max.z >= other.min.z
    }

    /// Smallest box that contains both this box and `other`.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: Vec3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            max: Vec3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        }
    }

    /// Box expanded by `margin` on each side (min -= margin, max += margin).
    #[inline]
    pub fn expand(&self, margin: Vec3) -> Self {
        Self {
            min: self.min - margin,
            max: self.max + margin,
        }
    }

    /// The eight corners of the box (min, then variations of max components).
    #[inline]
    pub fn corners(&self) -> [Vec3; 8] {
        [
            Vec3::new(self.min.x, self.min.y, self.min.z),
            Vec3::new(self.max.x, self.min.y, self.min.z),
            Vec3::new(self.min.x, self.max.y, self.min.z),
            Vec3::new(self.max.x, self.max.y, self.min.z),
            Vec3::new(self.min.x, self.min.y, self.max.z),
            Vec3::new(self.max.x, self.min.y, self.max.z),
            Vec3::new(self.min.x, self.max.y, self.max.z),
            Vec3::new(self.max.x, self.max.y, self.max.z),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb3_contains() {
        let aabb = Aabb3::new(Vec3::ZERO, Vec3::ONE);
        assert!(aabb.contains(Vec3::new(0.5, 0.5, 0.5)));
        assert!(!aabb.contains(Vec3::new(2.0, 2.0, 2.0)));
    }

    #[test]
    fn test_aabb3_intersects() {
        let aabb1 = Aabb3::new(Vec3::ZERO, Vec3::ONE);
        let aabb2 = Aabb3::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.5, 1.5, 1.5));
        assert!(aabb1.intersects(&aabb2));
    }

    #[test]
    fn test_aabb3_union() {
        let aabb1 = Aabb3::new(Vec3::ZERO, Vec3::ONE);
        let aabb2 = Aabb3::new(Vec3::ONE, Vec3::new(2.0, 2.0, 2.0));
        let union = aabb1.union(&aabb2);
        assert_eq!(union.min, Vec3::ZERO);
        assert_eq!(union.max, Vec3::new(2.0, 2.0, 2.0));
    }
}
