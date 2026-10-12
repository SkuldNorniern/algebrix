//! 2D axis-aligned box in double precision: layer bounds and dirty regions in document space.
//! A box with `min > max` on either axis is empty; [`DAabb2::EMPTY`] is the start for unions.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{DAabb2, DAffine2, DVec2};
//!
//! let mut dirty = DAabb2::EMPTY;
//! dirty = dirty.union(&DAabb2::from_min_max(DVec2::ZERO, DVec2::splat(10.0)));
//! let moved = dirty.transform(&DAffine2::from_translation(DVec2::new(5.0, 0.0)));
//! assert_eq!(moved.min, DVec2::new(5.0, 0.0));
//! assert!(dirty.intersection(&moved).contains_point(DVec2::new(7.0, 7.0)));
//! ```

use crate::{DAffine2, DVec2};

/// 2D axis-aligned box in f64 (min and max corners, inclusive).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DAabb2 {
    pub min: DVec2,
    pub max: DVec2,
}

impl DAabb2 {
    /// Contains nothing; the identity for [`union`](Self::union).
    pub const EMPTY: DAabb2 = DAabb2 {
        min: DVec2::splat(f64::INFINITY),
        max: DVec2::splat(f64::NEG_INFINITY),
    };

    /// Box with the given corners.
    #[inline]
    pub const fn from_min_max(min: DVec2, max: DVec2) -> Self {
        Self { min, max }
    }

    /// Smallest box holding every point, or [`EMPTY`](Self::EMPTY) for none.
    pub fn from_points(points: impl IntoIterator<Item = DVec2>) -> Self {
        points.into_iter().fold(Self::EMPTY, |b, p| Self {
            min: b.min.min(p),
            max: b.max.max(p),
        })
    }

    /// True when the box holds no point.
    #[inline]
    pub fn is_empty(&self) -> bool {
        !(self.min.x <= self.max.x && self.min.y <= self.max.y)
    }

    /// Size per axis, zero when empty.
    #[inline]
    pub fn size(&self) -> DVec2 {
        if self.is_empty() {
            DVec2::ZERO
        } else {
            self.max - self.min
        }
    }

    /// Midpoint of the corners.
    #[inline]
    pub fn center(&self) -> DVec2 {
        (self.min + self.max) * 0.5
    }

    /// True if `p` is inside or on the edge.
    #[inline]
    pub fn contains_point(&self, p: DVec2) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Smallest box holding both.
    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// The overlap of both; [`is_empty`](Self::is_empty) when they don't meet.
    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    /// Bounds of the box's four corners after `t`. Empty stays empty.
    pub fn transform(&self, t: &DAffine2) -> Self {
        if self.is_empty() {
            return Self::EMPTY;
        }
        Self::from_points([
            t.transform_point2(self.min),
            t.transform_point2(DVec2::new(self.max.x, self.min.y)),
            t.transform_point2(DVec2::new(self.min.x, self.max.y)),
            t.transform_point2(self.max),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_is_the_union_identity() {
        let b = DAabb2::from_min_max(DVec2::new(1.0, 2.0), DVec2::new(3.0, 4.0));
        assert!(DAabb2::EMPTY.is_empty());
        assert_eq!(DAabb2::EMPTY.union(&b), b);
        assert_eq!(DAabb2::EMPTY.size(), DVec2::ZERO);
        assert!(DAabb2::EMPTY.transform(&DAffine2::IDENTITY).is_empty());
    }

    #[test]
    fn apart_boxes_do_not_intersect() {
        let a = DAabb2::from_min_max(DVec2::ZERO, DVec2::ONE);
        let b = DAabb2::from_min_max(DVec2::splat(2.0), DVec2::splat(3.0));
        assert!(a.intersection(&b).is_empty());
        let edge = DAabb2::from_min_max(DVec2::ONE, DVec2::splat(2.0));
        assert!(!a.intersection(&edge).is_empty());
    }

    #[test]
    fn rotated_box_bounds_its_corners() {
        let b = DAabb2::from_min_max(DVec2::ZERO, DVec2::new(2.0, 1.0));
        let t = DAffine2::from_angle(std::f64::consts::FRAC_PI_2);
        let r = b.transform(&t);
        assert!((r.min - DVec2::new(-1.0, 0.0)).length() < 1e-12);
        assert!((r.max - DVec2::new(0.0, 2.0)).length() < 1e-12);
    }
}
