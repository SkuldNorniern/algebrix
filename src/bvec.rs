//! Boolean vectors (BVec2/3/4): component-wise masks for select, all, any.
//!
//! Use [`select`](BVec2::select) to pick components from two vectors by mask (where true take first,
//! where false take second). [`all`](BVec2::all) / [`any`](BVec2::any) / [`none`](BVec2::none) for reduction.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{BVec2, Vec2};
//!
//! let mask = BVec2::new(true, false);
//! let a = Vec2::new(1.0, 2.0);
//! let b = Vec2::new(10.0, 20.0);
//! let c = mask.select(a, b);
//! assert_eq!(c.x, 1.0);
//! assert_eq!(c.y, 20.0);
//!
//! assert!(BVec2::TRUE.all());
//! assert!(BVec2::new(true, false).any());
//! ```

use crate::{Vec2, Vec3, Vec4};

/// 2D boolean vector (component-wise mask).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BVec2 {
    pub x: bool,
    pub y: bool,
}

impl BVec2 {
    pub const FALSE: BVec2 = BVec2 { x: false, y: false };
    pub const TRUE: BVec2 = BVec2 { x: true, y: true };

    #[inline(always)]
    pub const fn new(x: bool, y: bool) -> Self {
        Self { x, y }
    }

    #[inline(always)]
    pub const fn splat(v: bool) -> Self {
        Self { x: v, y: v }
    }

    /// Returns true if all components are true
    #[inline(always)]
    pub fn all(self) -> bool {
        self.x && self.y
    }

    /// Returns true if any component is true
    #[inline(always)]
    pub fn any(self) -> bool {
        self.x || self.y
    }

    /// Returns true if no components are true
    #[inline(always)]
    pub fn none(self) -> bool {
        !self.x && !self.y
    }

    /// Component-wise AND
    #[inline(always)]
    pub fn and(self, other: Self) -> Self {
        Self {
            x: self.x && other.x,
            y: self.y && other.y,
        }
    }

    /// Component-wise OR
    #[inline(always)]
    pub fn or(self, other: Self) -> Self {
        Self {
            x: self.x || other.x,
            y: self.y || other.y,
        }
    }

    /// Component-wise XOR
    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        Self {
            x: self.x ^ other.x,
            y: self.y ^ other.y,
        }
    }

    /// Select components: if mask is true, use a, else use b
    #[inline(always)]
    pub fn select(self, a: Vec2, b: Vec2) -> Vec2 {
        Vec2::new(
            if self.x { a.x } else { b.x },
            if self.y { a.y } else { b.y },
        )
    }
}

impl std::ops::Not for BVec2 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self {
            x: !self.x,
            y: !self.y,
        }
    }
}

impl std::ops::BitAnd for BVec2 {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.and(other)
    }
}

impl std::ops::BitOr for BVec2 {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.or(other)
    }
}

impl std::ops::BitXor for BVec2 {
    type Output = Self;
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        self.xor(other)
    }
}

/// 3D boolean vector (component-wise mask).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BVec3 {
    pub x: bool,
    pub y: bool,
    pub z: bool,
}

impl BVec3 {
    pub const FALSE: BVec3 = BVec3 {
        x: false,
        y: false,
        z: false,
    };
    pub const TRUE: BVec3 = BVec3 {
        x: true,
        y: true,
        z: true,
    };

    #[inline(always)]
    pub const fn new(x: bool, y: bool, z: bool) -> Self {
        Self { x, y, z }
    }

    #[inline(always)]
    pub const fn splat(v: bool) -> Self {
        Self { x: v, y: v, z: v }
    }

    #[inline(always)]
    pub fn all(self) -> bool {
        self.x && self.y && self.z
    }

    #[inline(always)]
    pub fn any(self) -> bool {
        self.x || self.y || self.z
    }

    #[inline(always)]
    pub fn none(self) -> bool {
        !self.x && !self.y && !self.z
    }

    #[inline(always)]
    pub fn and(self, other: Self) -> Self {
        Self {
            x: self.x && other.x,
            y: self.y && other.y,
            z: self.z && other.z,
        }
    }

    #[inline(always)]
    pub fn or(self, other: Self) -> Self {
        Self {
            x: self.x || other.x,
            y: self.y || other.y,
            z: self.z || other.z,
        }
    }

    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        Self {
            x: self.x ^ other.x,
            y: self.y ^ other.y,
            z: self.z ^ other.z,
        }
    }

    #[inline(always)]
    pub fn select(self, a: Vec3, b: Vec3) -> Vec3 {
        Vec3::new(
            if self.x { a.x } else { b.x },
            if self.y { a.y } else { b.y },
            if self.z { a.z } else { b.z },
        )
    }
}

impl std::ops::Not for BVec3 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self {
            x: !self.x,
            y: !self.y,
            z: !self.z,
        }
    }
}

impl std::ops::BitAnd for BVec3 {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.and(other)
    }
}

impl std::ops::BitOr for BVec3 {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.or(other)
    }
}

impl std::ops::BitXor for BVec3 {
    type Output = Self;
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        self.xor(other)
    }
}

/// 4D boolean vector (component-wise mask).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BVec4 {
    pub x: bool,
    pub y: bool,
    pub z: bool,
    pub w: bool,
}

impl BVec4 {
    pub const FALSE: BVec4 = BVec4 {
        x: false,
        y: false,
        z: false,
        w: false,
    };
    pub const TRUE: BVec4 = BVec4 {
        x: true,
        y: true,
        z: true,
        w: true,
    };

    #[inline(always)]
    pub const fn new(x: bool, y: bool, z: bool, w: bool) -> Self {
        Self { x, y, z, w }
    }

    #[inline(always)]
    pub const fn splat(v: bool) -> Self {
        Self {
            x: v,
            y: v,
            z: v,
            w: v,
        }
    }

    #[inline(always)]
    pub fn all(self) -> bool {
        self.x && self.y && self.z && self.w
    }

    #[inline(always)]
    pub fn any(self) -> bool {
        self.x || self.y || self.z || self.w
    }

    #[inline(always)]
    pub fn none(self) -> bool {
        !self.x && !self.y && !self.z && !self.w
    }

    #[inline(always)]
    pub fn and(self, other: Self) -> Self {
        Self {
            x: self.x && other.x,
            y: self.y && other.y,
            z: self.z && other.z,
            w: self.w && other.w,
        }
    }

    #[inline(always)]
    pub fn or(self, other: Self) -> Self {
        Self {
            x: self.x || other.x,
            y: self.y || other.y,
            z: self.z || other.z,
            w: self.w || other.w,
        }
    }

    #[inline(always)]
    pub fn xor(self, other: Self) -> Self {
        Self {
            x: self.x ^ other.x,
            y: self.y ^ other.y,
            z: self.z ^ other.z,
            w: self.w ^ other.w,
        }
    }

    #[inline(always)]
    pub fn select(self, a: Vec4, b: Vec4) -> Vec4 {
        Vec4::new(
            if self.x { a.x } else { b.x },
            if self.y { a.y } else { b.y },
            if self.z { a.z } else { b.z },
            if self.w { a.w } else { b.w },
        )
    }
}

impl std::ops::Not for BVec4 {
    type Output = Self;
    #[inline]
    fn not(self) -> Self {
        Self {
            x: !self.x,
            y: !self.y,
            z: !self.z,
            w: !self.w,
        }
    }
}

impl std::ops::BitAnd for BVec4 {
    type Output = Self;
    #[inline]
    fn bitand(self, other: Self) -> Self {
        self.and(other)
    }
}

impl std::ops::BitOr for BVec4 {
    type Output = Self;
    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.or(other)
    }
}

impl std::ops::BitXor for BVec4 {
    type Output = Self;
    #[inline]
    fn bitxor(self, other: Self) -> Self {
        self.xor(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bvec2_all_any() {
        assert!(BVec2::TRUE.all());
        assert!(!BVec2::FALSE.all());
        assert!(BVec2::new(true, false).any());
        assert!(!BVec2::FALSE.any());
    }

    #[test]
    fn test_bvec3_select() {
        let mask = BVec3::new(true, false, true);
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        let result = mask.select(a, b);
        assert_eq!(result.x, 1.0);
        assert_eq!(result.y, 5.0);
        assert_eq!(result.z, 3.0);
    }

    #[test]
    fn test_bvec4_bitops() {
        let a = BVec4::new(true, false, true, false);
        let b = BVec4::new(true, true, false, false);
        assert_eq!(a & b, BVec4::new(true, false, false, false));
        assert_eq!(a | b, BVec4::new(true, true, true, false));
        assert_eq!(a ^ b, BVec4::new(false, true, true, false));
    }
}
