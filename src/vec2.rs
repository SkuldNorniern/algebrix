//! 2D float vector: positions, directions, UVs.
//!
//! Supports the usual ops (add, sub, mul by scalar, dot, length, normalize).
//! Use [`normalize_fast`](Vec2::normalize_fast) when you care more about speed than precision.
//!
//! # Example
//!
//! ```rust
//! use algebrix::Vec2;
//!
//! let a = Vec2::new(3.0, 4.0);
//! assert!((a.length() - 5.0).abs() < 1e-5);
//! let u = a.normalize();
//! assert!((u.dot(u) - 1.0).abs() < 1e-5);
//!
//! let right = Vec2::from_angle(0.0);
//! let up = Vec2::from_angle(std::f32::consts::FRAC_PI_2);
//! assert!(right.dot(up).abs() < 1e-5);
//! ```

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

    /// Normalize using rsqrt. Slightly less accurate than [`normalize`](Self::normalize) but faster; use for directions where exact length 1 is not required.
    ///
    /// # Example
    ///
    /// ```rust
    /// use algebrix::Vec2;
    /// let v = Vec2::new(3.0, 4.0);
    /// let u = v.normalize_fast();
    /// assert!(u.length() > 0.99 && u.length() < 1.01);
    /// ```
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

    /// All components set to the same value. Same as `Vec2::new(v, v)`.
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self { x: value, y: value }
    }

    /// Perpendicular vector: rotate 90 degrees counter-clockwise in the XY plane. For (x, y) returns (-y, x).
    #[inline(always)]
    pub fn perp(self) -> Self {
        Self { x: -self.y, y: self.x }
    }

    /// Perpendicular dot product (2D cross product)
    #[inline(always)]
    pub fn perp_dot(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    /// Angle between two vectors in radians
    #[inline]
    pub fn angle_between(self, other: Self) -> f32 {
        let dot = self.dot(other);
        let det = self.perp_dot(other);
        det.atan2(dot)
    }

    /// Unit vector at the given angle (radians). X = cos(angle), Y = sin(angle); angle 0 is +X.
    ///
    /// # Example
    ///
    /// ```rust
    /// use algebrix::Vec2;
    /// let right = Vec2::from_angle(0.0);
    /// assert!((right.x - 1.0).abs() < 1e-5 && right.y.abs() < 1e-5);
    /// ```
    #[inline]
    pub fn from_angle(angle: f32) -> Self {
        Self {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    /// Convert to angle in radians
    #[inline]
    pub fn to_angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Component-wise reciprocal
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self {
            x: self.x.recip(),
            y: self.y.recip(),
        }
    }

    /// Component-wise absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        Self {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }

    /// Component-wise signum
    #[inline(always)]
    pub fn signum(self) -> Self {
        Self {
            x: self.x.signum(),
            y: self.y.signum(),
        }
    }

    /// Minimum element
    #[inline(always)]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y)
    }

    /// Maximum element
    #[inline(always)]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y)
    }

    /// Component-wise clamp
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self {
            x: self.x.clamp(min.x, max.x),
            y: self.y.clamp(min.y, max.y),
        }
    }

    /// Component-wise minimum
    #[inline(always)]
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Component-wise maximum
    #[inline(always)]
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Extend to Vec3
    #[inline(always)]
    pub fn extend(self, z: f32) -> crate::Vec3 {
        crate::Vec3::new(self.x, self.y, z)
    }

    /// Project onto another vector
    #[inline]
    pub fn project(self, onto: Self) -> Self {
        let dot = self.dot(onto);
        let len_sq = onto.length_squared();
        if len_sq > 0.0 {
            onto * (dot / len_sq)
        } else {
            Self::ZERO
        }
    }

    /// Reject from another vector (component perpendicular to onto)
    #[inline]
    pub fn reject(self, from: Self) -> Self {
        self - self.project(from)
    }

    /// Reflect across a normal
    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Check if all components are finite
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Check if any component is NaN
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan()
    }

    /// Approximate equality with epsilon
    #[inline]
    pub fn abs_diff_eq(self, other: Self, epsilon: f32) -> bool {
        (self.x - other.x).abs() <= epsilon && (self.y - other.y).abs() <= epsilon
    }

    /// Create from array
    #[inline(always)]
    pub fn from_array(a: [f32; 2]) -> Self {
        Self { x: a[0], y: a[1] }
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 2] {
        [self.x, self.y]
    }

    /// Distance between two points
    #[inline]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Squared distance between two points
    #[inline]
    pub fn distance_squared(self, other: Self) -> f32 {
        (self - other).length_squared()
    }

    /// Create from slice, returns None if slice is too short
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() >= 2 {
            Some(Self::new(slice[0], slice[1]))
        } else {
            None
        }
    }

    /// Write to slice, panics if slice is too short
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        assert!(slice.len() >= 2, "slice must have at least 2 elements");
        slice[0] = self.x;
        slice[1] = self.y;
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 2] {
        self.as_ref()
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 2] {
        self.as_mut()
    }
}

impl std::convert::AsRef<[f32; 2]> for Vec2 {
    #[inline(always)]
    fn as_ref(&self) -> &[f32; 2] {
        unsafe { &*(self as *const Self as *const [f32; 2]) }
    }
}

impl std::convert::AsMut<[f32; 2]> for Vec2 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f32; 2] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 2]) }
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
