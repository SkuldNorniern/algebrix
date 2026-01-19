//! 3D vector implementation

use crate::simd;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    _padding: f32,
}

impl Vec3 {
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        _padding: 0.0,
    };
    pub const ONE: Vec3 = Vec3 {
        x: 1.0,
        y: 1.0,
        z: 1.0,
        _padding: 0.0,
    };
    pub const X: Vec3 = Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
        _padding: 0.0,
    };
    pub const Y: Vec3 = Vec3 {
        x: 0.0,
        y: 1.0,
        z: 0.0,
        _padding: 0.0,
    };
    pub const Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 1.0,
        _padding: 0.0,
    };
    pub const NEG_X: Vec3 = Vec3 {
        x: -1.0,
        y: 0.0,
        z: 0.0,
        _padding: 0.0,
    };
    pub const NEG_Y: Vec3 = Vec3 {
        x: 0.0,
        y: -1.0,
        z: 0.0,
        _padding: 0.0,
    };
    pub const NEG_Z: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: -1.0,
        _padding: 0.0,
    };

    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z, _padding: 0.0 }
    }
    
    #[inline(always)]
    pub fn as_ptr(&self) -> *const f32 {
        &self.x as *const f32
    }
    
    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        &mut self.x as *mut f32
    }

    /// Create a vector with all components set to the same value
    /// Equivalent to glam's `splat` method
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
            _padding: 0.0,
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Fast length using reciprocal square root approximation
    /// Less accurate than `length()` but faster. Returns approximate length.
    #[inline]
    pub fn length_fast(self) -> f32 {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            len_sq * simd::f32_rsqrt(len_sq)
        } else {
            0.0
        }
    }

    // Actually, I'll just keep the logic clean and call specialized SIMD where provided.

    /// Distance between two points
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Fast distance using reciprocal square root approximation
    #[inline]
    pub fn distance_fast(self, other: Self) -> f32 {
        (self - other).length_fast()
    }

    /// Squared distance between two points (faster than distance, avoids sqrt)
    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        (self - other).length_squared()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        simd::vec3_length_squared(self)
    }

    #[inline(always)]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
                z: self.z * inv_len,
                _padding: 0.0,
            }
        } else {
            Self::ZERO
        }
    }

    /// Fast normalize using reciprocal square root approximation
    /// Less accurate than `normalize()` but faster. Suitable for game code where
    /// slight precision loss is acceptable (e.g., normalizing direction vectors).
    #[inline]
    pub fn normalize_fast(self) -> Self {
        let len_sq = self.length_squared();
        simd::vec3_normalize_fast(self, len_sq)
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        simd::vec3_dot(self, other)
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        simd::vec3_cross(self, other)
    }

    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        simd::vec3_lerp(self, other, t)
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        simd::vec3_abs(self)
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        simd::vec3_min(self, other)
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        simd::vec3_max(self, other)
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        simd::vec3_add(self, other)
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        simd::vec3_sub(self, other)
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        simd::vec3_mul(self, other)
    }
}

impl std::ops::Div for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        simd::vec3_div(self, other)
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, scalar: f32) -> Self {
        simd::vec3_mul_scalar(self, scalar)
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, vec: Vec3) -> Vec3 {
        simd::vec3_mul_scalar(vec, self)
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, scalar: f32) -> Self {
        simd::vec3_mul_scalar(self, scalar.recip())
    }
}

impl std::iter::Sum for Vec3 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |acc, v| acc + v)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        simd::vec3_neg(self)
    }
}

impl std::ops::AddAssign for Vec3 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for Vec3 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign<f32> for Vec3 {
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        self.x *= scalar;
        self.y *= scalar;
        self.z *= scalar;
    }
}

impl std::ops::DivAssign<f32> for Vec3 {
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        let inv = scalar.recip();
        self.x *= inv;
        self.y *= inv;
        self.z *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_new() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
    }

    #[test]
    fn test_vec3_constants() {
        assert_eq!(Vec3::ZERO, Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(Vec3::ONE, Vec3::new(1.0, 1.0, 1.0));
        assert_eq!(Vec3::X, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(Vec3::Y, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(Vec3::Z, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec3_add() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        let result = v1 + v2;
        assert_eq!(result, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3_sub() {
        let v1 = Vec3::new(5.0, 7.0, 9.0);
        let v2 = Vec3::new(1.0, 2.0, 3.0);
        let result = v1 - v2;
        assert_eq!(result, Vec3::new(4.0, 5.0, 6.0));
    }

    #[test]
    fn test_vec3_mul_scalar() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let result = v * 2.0;
        assert_eq!(result, Vec3::new(2.0, 4.0, 6.0));
    }

    #[test]
    fn test_vec3_length() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert!((v.length() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_length_squared() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v.length_squared(), 25.0);
    }

    #[test]
    fn test_vec3_normalize() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_normalize_zero() {
        let v = Vec3::ZERO;
        let normalized = v.normalize();
        assert_eq!(normalized, Vec3::ZERO);
    }

    #[test]
    fn test_vec3_dot() {
        let v1 = Vec3::new(1.0, 2.0, 3.0);
        let v2 = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(v1.dot(v2), 32.0);
    }

    #[test]
    fn test_vec3_cross() {
        let v1 = Vec3::X;
        let v2 = Vec3::Y;
        let result = v1.cross(v2);
        assert_eq!(result, Vec3::Z);
    }

    #[test]
    fn test_vec3_lerp() {
        let v1 = Vec3::ZERO;
        let v2 = Vec3::ONE;
        let result = v1.lerp(v2, 0.5);
        assert_eq!(result, Vec3::new(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_vec3_neg() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(-v, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn test_vec3_add_assign() {
        let mut v = Vec3::new(1.0, 2.0, 3.0);
        v += Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(v, Vec3::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vec3_normalize_fast() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let normalized = v.normalize_fast();
        // Fast normalize should be close to unit length (may have slight precision differences)
        let len = normalized.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Fast normalize length should be close to 1.0, got {}",
            len
        );
    }

    #[test]
    fn test_vec3_normalize_fast_zero() {
        let v = Vec3::ZERO;
        let normalized = v.normalize_fast();
        assert_eq!(normalized, Vec3::ZERO);
    }

    #[test]
    fn test_vec3_length_fast() {
        let v = Vec3::new(3.0, 4.0, 0.0);
        let len = v.length();
        // length_fast is implemented via normalize_fast which might be slightly off
        let len_fast = v.length_fast();
        assert!(
            (len_fast - len).abs() < 0.1,
            "Fast length should be close to regular length"
        );
    }

    #[test]
    fn test_vec3_distance() {
        let v1 = Vec3::new(0.0, 0.0, 0.0);
        let v2 = Vec3::new(3.0, 4.0, 0.0);
        assert!((v1.distance(v2) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec3_distance_squared() {
        let v1 = Vec3::new(0.0, 0.0, 0.0);
        let v2 = Vec3::new(3.0, 4.0, 0.0);
        assert_eq!(v1.distance_squared(v2), 25.0);
    }

    #[test]
    fn test_vec3_distance_fast() {
        let v1 = Vec3::new(0.0, 0.0, 0.0);
        let v2 = Vec3::new(3.0, 4.0, 0.0);
        let dist = v1.distance(v2);
        let dist_fast = v1.distance_fast(v2);
        assert!(
            (dist_fast - dist).abs() < 0.1,
            "Fast distance should be close to regular distance"
        );
    }
}
