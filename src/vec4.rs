//! 4D float vector: homogenous coords, quaternion-like (x,y,z,w), colors (RGBA).
//!
//! Column vectors; multiply by Mat4 on the left. Use `.truncate()` to drop w and get Vec3.
//! [`normalize_fast`](Vec4::normalize_fast) uses rsqrt when the SIMD feature is on.
//!
//! # Example
//!
//! ```rust
//! use algebrix::Vec4;
//!
//! let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
//! assert!((v.length() - 1.0).abs() < 1e-5);
//! let p = Vec4::new(2.0, 2.0, 2.0, 1.0);
//! let p3 = p.truncate();
//! assert_eq!(p3.x, 2.0);
//!
//! let a = Vec4::new(1.0, 0.0, 0.0, 0.5);
//! let b = Vec4::new(0.0, 1.0, 0.0, 0.5);
//! let mid = a.lerp(b, 0.5);
//! assert!(mid.x > 0.0 && mid.y > 0.0);
//! ```

use crate::utils;

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
use std::arch::aarch64::*;

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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ref().as_ptr());
                let mul = _mm_mul_ps(v_v, v_v);
                let shuf = _mm_movehdup_ps(mul);
                let sums = _mm_add_ps(mul, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let result = _mm_add_ss(sums, shuf2);
                _mm_cvtss_f32(result)
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ref().as_ptr());
                let mul = vmulq_f32(v_v, v_v);
                vaddvq_f32(mul)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
        }
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

    /// Normalize using rsqrt approximation. Less accurate than `normalize()` but faster; uses SIMD when the feature is on.
    #[inline]
    pub fn normalize_fast(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
            {
                unsafe {
                    let v_v = _mm_load_ps(self.as_ref().as_ptr());
                    let v_len_sq = _mm_set1_ps(len_sq);
                    let rsqrt = _mm_rsqrt_ps(v_len_sq);
                    let half = _mm_set1_ps(0.5);
                    let three = _mm_set1_ps(3.0);
                    let muls = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
                    let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls));
                    let muls2 = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
                    let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls2));
                    let res = _mm_mul_ps(v_v, rsqrt);
                    let mut out = Self::ZERO;
                    _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                    out
                }
            }
            #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
            {
                unsafe {
                    let v_v = vld1q_f32(self.as_ref().as_ptr());
                    let v_len_sq = vdupq_n_f32(len_sq);
                    let rsqrt = vrsqrteq_f32(v_len_sq);
                    let muls = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls));
                    let muls2 = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls2));
                    let res = vmulq_f32(v_v, rsqrt);
                    let mut out = Self::ZERO;
                    vst1q_f32(out.as_mut().as_mut_ptr(), res);
                    out
                }
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
                all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
            )))]
            {
                let inv_len = len_sq.sqrt().recip();
                Self::new(
                    self.x * inv_len,
                    self.y * inv_len,
                    self.z * inv_len,
                    self.w * inv_len,
                )
            }
        } else {
            Self::ZERO
        }
    }

    #[inline]
    pub fn dot(self, other: Self) -> f32 {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let mul = _mm_mul_ps(v_a, v_b);
                let shuf = _mm_movehdup_ps(mul);
                let sums = _mm_add_ps(mul, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let result = _mm_add_ss(sums, shuf2);
                _mm_cvtss_f32(result)
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let mul = vmulq_f32(v_a, v_b);
                vaddvq_f32(mul)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
        }
    }

    #[inline]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        Self {
            x: utils::lerp(self.x, other.x, t),
            y: utils::lerp(self.y, other.y, t),
            z: utils::lerp(self.z, other.z, t),
            w: utils::lerp(self.w, other.w, t),
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_min_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vminq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x.min(other.x),
                self.y.min(other.y),
                self.z.min(other.z),
                self.w.min(other.w),
            )
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_max_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vmaxq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x.max(other.x),
                self.y.max(other.y),
                self.z.max(other.z),
                self.w.max(other.w),
            )
        }
    }

    /// All components set to the same value.
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        Self { x: value, y: value, z: value, w: value }
    }

    /// Component-wise absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ref().as_ptr());
                let mask = _mm_set1_ps(-0.0);
                let res = _mm_andnot_ps(mask, v_v);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ref().as_ptr());
                let res = vabsq_f32(v_v);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x.abs(), self.y.abs(), self.z.abs(), self.w.abs())
        }
    }

    /// Component-wise reciprocal
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip(), self.w.recip())
    }

    /// Component-wise signum
    #[inline(always)]
    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum(), self.z.signum(), self.w.signum())
    }

    /// Minimum element
    #[inline(always)]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z).min(self.w)
    }

    /// Maximum element
    #[inline(always)]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z).max(self.w)
    }

    /// Component-wise clamp
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.clamp(min.x, max.x),
            self.y.clamp(min.y, max.y),
            self.z.clamp(min.z, max.z),
            self.w.clamp(min.w, max.w),
        )
    }

    /// Truncate to Vec3 (drop w component)
    #[inline(always)]
    pub fn truncate(self) -> crate::Vec3 {
        crate::Vec3::new(self.x, self.y, self.z)
    }

    /// Check if all components are finite
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite() && self.w.is_finite()
    }

    /// Check if any component is NaN
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan() || self.w.is_nan()
    }

    /// Approximate equality with epsilon
    #[inline]
    pub fn abs_diff_eq(self, other: Self, epsilon: f32) -> bool {
        (self.x - other.x).abs() <= epsilon
            && (self.y - other.y).abs() <= epsilon
            && (self.z - other.z).abs() <= epsilon
            && (self.w - other.w).abs() <= epsilon
    }

    /// Create from array
    #[inline(always)]
    pub fn from_array(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 4] {
        [self.x, self.y, self.z, self.w]
    }

    /// Create from slice, returns None if slice is too short
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() >= 4 {
            Some(Self::new(slice[0], slice[1], slice[2], slice[3]))
        } else {
            None
        }
    }

    /// Write to slice, panics if slice is too short
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        assert!(slice.len() >= 4, "slice must have at least 4 elements");
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
        slice[3] = self.w;
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 4] {
        self.as_ref()
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 4] {
        self.as_mut()
    }
}

impl std::convert::AsRef<[f32; 4]> for Vec4 {
    #[inline(always)]
    fn as_ref(&self) -> &[f32; 4] {
        unsafe { &*(self as *const Self as *const [f32; 4]) }
    }
}

impl std::convert::AsMut<[f32; 4]> for Vec4 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f32; 4] {
        unsafe { &mut *(self as *mut Self as *mut [f32; 4]) }
    }
}

impl std::ops::Add for Vec4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_add_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vaddq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x + other.x,
                self.y + other.y,
                self.z + other.z,
                self.w + other.w,
            )
        }
    }
}

impl std::ops::Sub for Vec4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_sub_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vsubq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x - other.x,
                self.y - other.y,
                self.z - other.z,
                self.w - other.w,
            )
        }
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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_mul_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vmulq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x * other.x,
                self.y * other.y,
                self.z * other.z,
                self.w * other.w,
            )
        }
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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ref().as_ptr());
                let v_b = _mm_load_ps(other.as_ref().as_ptr());
                let res = _mm_div_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ref().as_ptr());
                let v_b = vld1q_f32(other.as_ref().as_ptr());
                let res = vdivq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut().as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x / other.x,
                self.y / other.y,
                self.z / other.z,
                self.w / other.w,
            )
        }
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
