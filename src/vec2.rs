//! 2D vector implementation

use crate::utils;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const ZERO: Vec2 = Vec2 { x: 0.0, y: 0.0 };
    pub const ONE: Vec2 = Vec2 { x: 1.0, y: 1.0 };
    pub const X: Vec2 = Vec2 { x: 1.0, y: 0.0 };
    pub const Y: Vec2 = Vec2 { x: 0.0, y: 1.0 };

    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Self {
                x: self.x * inv_len,
                y: self.y * inv_len,
            }
        } else {
            Self::ZERO
        }
    }

    /// Fast normalize using reciprocal square root approximation
    /// Less accurate than `normalize()` but faster. Suitable for game code.
    #[inline]
    pub fn normalize_fast(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            #[cfg(target_arch = "x86_64")]
            {
                #[cfg(any(feature = "simd", feature = "simd-x86"))]
                unsafe {
                    use std::arch::x86_64::*;
                    let len_sq_simd = _mm_set_ss(len_sq);
                    let rsqrt_approx = _mm_rsqrt_ss(len_sq_simd);
                    let half = _mm_set_ss(0.5);
                    let three = _mm_set_ss(3.0);
                    let refined = _mm_mul_ss(
                        rsqrt_approx,
                        _mm_sub_ss(
                            three,
                            _mm_mul_ss(
                                _mm_mul_ss(half, len_sq_simd),
                                _mm_mul_ss(rsqrt_approx, rsqrt_approx),
                            ),
                        ),
                    );
                    let inv_len = _mm_cvtss_f32(refined);
                    Self {
                        x: self.x * inv_len,
                        y: self.y * inv_len,
                    }
                }
                #[cfg(not(any(feature = "simd", feature = "simd-x86")))]
                {
                    let inv_len = len_sq.sqrt().recip();
                    Self {
                        x: self.x * inv_len,
                        y: self.y * inv_len,
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                #[cfg(any(feature = "simd", feature = "simd-arm"))]
                unsafe {
                    use std::arch::aarch64::*;
                    let len_sq_simd = vdupq_n_f32(len_sq);
                    let rsqrt_approx = vrsqrteq_f32(len_sq_simd);
                    let muls = vmulq_f32(rsqrt_approx, rsqrt_approx);
                    let rsqrt = vmulq_f32(rsqrt_approx, vrsqrtsq_f32(len_sq_simd, muls));
                    let muls2 = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(len_sq_simd, muls2));
                    let inv_len = vgetq_lane_f32(rsqrt, 0);
                    Self {
                        x: self.x * inv_len,
                        y: self.y * inv_len,
                    }
                }
                #[cfg(not(any(feature = "simd", feature = "simd-arm")))]
                {
                    let inv_len = len_sq.sqrt().recip();
                    Self {
                        x: self.x * inv_len,
                        y: self.y * inv_len,
                    }
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let inv_len = len_sq.sqrt().recip();
                Self {
                    x: self.x * inv_len,
                    y: self.y * inv_len,
                }
            }
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: utils::lerp(self.x, other.x, t),
            y: utils::lerp(self.y, other.y, t),
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl std::ops::Mul<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

impl std::ops::Mul<Vec2> for f32 {
    type Output = Vec2;
    #[inline]
    fn mul(self, vec: Vec2) -> Vec2 {
        Vec2 {
            x: self * vec.x,
            y: self * vec.y,
        }
    }
}

impl std::ops::Div<f32> for Vec2 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv = scalar.recip();
        Self {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

impl std::ops::Neg for Vec2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_new() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
    }

    #[test]
    fn test_vec2_constants() {
        assert_eq!(Vec2::ZERO, Vec2::new(0.0, 0.0));
        assert_eq!(Vec2::ONE, Vec2::new(1.0, 1.0));
        assert_eq!(Vec2::X, Vec2::new(1.0, 0.0));
        assert_eq!(Vec2::Y, Vec2::new(0.0, 1.0));
    }

    #[test]
    fn test_vec2_add() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1 + v2, Vec2::new(4.0, 6.0));
    }

    #[test]
    fn test_vec2_length() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec2_normalize() {
        let v = Vec2::new(3.0, 4.0);
        let normalized = v.normalize();
        assert!((normalized.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec2_dot() {
        let v1 = Vec2::new(1.0, 2.0);
        let v2 = Vec2::new(3.0, 4.0);
        assert_eq!(v1.dot(v2), 11.0);
    }

    #[test]
    fn test_vec2_normalize_fast() {
        let v = Vec2::new(3.0, 4.0);
        let normalized = v.normalize_fast();
        let len = normalized.length();
        assert!(
            (len - 1.0).abs() < 0.01,
            "Fast normalize length should be close to 1.0, got {}",
            len
        );
    }
}
