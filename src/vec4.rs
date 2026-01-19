//! 4D vector implementation

use crate::{simd, utils};

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    pub const ZERO: Vec4 = Vec4 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 0.0,
    };
    pub const ONE: Vec4 = Vec4 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        w: 1.0,
    };

    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        simd::vec4_dot(self, self)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                w: self.w * inv_len,
            }
        } else {
            Self::ZERO
        }
    }

    /// Fast normalize using reciprocal square root approximation (SIMD-accelerated)
    /// Less accurate than `normalize()` but faster. Vec4 benefits from SIMD here.
    #[inline]
    pub fn normalize_fast(self) -> Self {
        let len_sq = self.length_squared();
        simd::vec4_normalize_fast(self, len_sq)
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        simd::vec4_dot(self, other)
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        // Vec4 is 4-wide, perfect for SIMD logic (implemented in centralized lib if needed)
        // For now, let's keep the scalar lerp for simplicity or move it to SIMD if t is broadcastable
        Self {
            x: utils::lerp(self.x, other.x, t),
            y: utils::lerp(self.y, other.y, t),
            z: utils::lerp(self.z, other.z, t),
            w: utils::lerp(self.w, other.w, t),
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        simd::vec4_min(self, other)
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        simd::vec4_max(self, other)
    }
}

impl std::ops::Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        simd::vec4_add(self, other)
    }
}

impl std::ops::Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        simd::vec4_sub(self, other)
    }
}

impl std::ops::Mul<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
            w: self.w * scalar,
        }
    }
}

impl std::ops::Mul<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        simd::vec4_mul(self, other)
    }
}

impl std::ops::Mul<Vec4> for f32 {
    type Output = Vec4;
    #[inline]
    fn mul(self, vec: Vec4) -> Vec4 {
        Vec4 {
            x: self * vec.x,
            y: self * vec.y,
            z: self * vec.z,
            w: self * vec.w,
        }
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv = scalar.recip();
        Self {
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
            w: self.w * inv,
        }
    }
}

impl std::ops::Div<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        simd::vec4_div(self, other)
    }
}

impl std::ops::Neg for Vec4 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec4_new() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_length() {
        let v = Vec4::new(2.0, 0.0, 0.0, 0.0);
        assert!((v.length() - 2.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec4_normalize() {
        let v = Vec4::new(2.0, 0.0, 0.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec4_dot() {
        let v1 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let v2 = Vec4::new(5.0, 6.0, 7.0, 8.0);
        assert_eq!(v1.dot(v2), 70.0);
    }

    #[test]
    fn test_vec4_normalize_fast() {
        let v = Vec4::new(2.0, 0.0, 0.0, 0.0);
        let normalized = v.normalize_fast();
        let len = normalized.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Fast normalize length should be close to 1.0, got {}",
            len
        );
    }
}
