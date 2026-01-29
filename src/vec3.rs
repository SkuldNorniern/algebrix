//! 3D float vector: positions, directions, normals.
//!
//! Column vectors; multiply by Mat3/Mat4 on the left. Has [`normalize_fast`](Vec3::normalize_fast)
//! and [`length_fast`](Vec3::length_fast) (rsqrt) when you prefer speed over precision.
//!
//! # Example
//!
//! ```rust
//! use algebrix::Vec3;
//!
//! let a = Vec3::new(1.0, 0.0, 0.0);
//! let b = Vec3::new(0.0, 1.0, 0.0);
//! assert!(a.dot(b).abs() < 1e-5);
//! let c = a.cross(b);
//! assert!((c - Vec3::Z).length() < 1e-5);
//!
//! let p = Vec3::new(3.0, 4.0, 0.0);
//! assert!((p.distance(Vec3::ZERO) - 5.0).abs() < 1e-5);
//! ```

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
use std::arch::aarch64::*;

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

    /// All components set to the same value.
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
            #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
            {
                unsafe {
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
                    len_sq * _mm_cvtss_f32(refined)
                }
            }
            #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
            {
                unsafe {
                    let len_sq_simd = vdupq_n_f32(len_sq);
                    let rsqrt_approx = vrsqrteq_f32(len_sq_simd);
                    let muls = vmulq_f32(rsqrt_approx, rsqrt_approx);
                    let rsqrt = vmulq_f32(rsqrt_approx, vrsqrtsq_f32(len_sq_simd, muls));
                    let muls2 = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(len_sq_simd, muls2));
                    len_sq * vgetq_lane_f32(rsqrt, 0)
                }
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
                all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
            )))]
            {
                len_sq * len_sq.sqrt().recip()
            }
        } else {
            0.0
        }
    }

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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ptr());
                let mul = _mm_mul_ps(v_v, v_v);
                let masked = _mm_blend_ps(mul, _mm_setzero_ps(), 0b1000);
                let shuf = _mm_movehdup_ps(masked);
                let sums = _mm_add_ps(masked, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let result = _mm_add_ss(sums, shuf2);
                _mm_cvtss_f32(result)
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ptr());
                let mul = vmulq_f32(v_v, v_v);
                let zero_w = vsetq_lane_f32(0.0, mul, 3);
                vaddvq_f32(zero_w)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            self.x.mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
        }
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
        if len_sq > 0.0 {
            #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
            {
                unsafe {
                    let v_v = _mm_load_ps(self.as_ptr());
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
                    _mm_store_ps(out.as_mut_ptr(), res);
                    out
                }
            }
            #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
            {
                unsafe {
                    let v_v = vld1q_f32(self.as_ptr());
                    let v_len_sq = vdupq_n_f32(len_sq);
                    let rsqrt = vrsqrteq_f32(v_len_sq);
                    let muls = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls));
                    let muls2 = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls2));
                    let res = vmulq_f32(v_v, rsqrt);
                    let mut out = Self::ZERO;
                    vst1q_f32(out.as_mut_ptr(), res);
                    out
                }
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
                all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
            )))]
            {
                let inv_len = len_sq.sqrt().recip();
                Self::new(self.x * inv_len, self.y * inv_len, self.z * inv_len)
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
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let mul = _mm_mul_ps(v_a, v_b);
                let masked = _mm_blend_ps(mul, _mm_setzero_ps(), 0b1000);
                let shuf = _mm_movehdup_ps(masked);
                let sums = _mm_add_ps(masked, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                let result = _mm_add_ss(sums, shuf2);
                _mm_cvtss_f32(result)
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let mul = vmulq_f32(v_a, v_b);
                let zero_w = vsetq_lane_f32(0.0, mul, 3);
                vaddvq_f32(zero_w)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            self.x.mul_add(other.x, self.y.mul_add(other.y, self.z * other.z))
        }
    }

    #[inline]
    pub fn cross(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let a_yzx = _mm_shuffle_ps(v_a, v_a, _MM_SHUFFLE(3, 0, 2, 1));
                let b_yzx = _mm_shuffle_ps(v_b, v_b, _MM_SHUFFLE(3, 0, 2, 1));
                let a_zxy = _mm_shuffle_ps(v_a, v_a, _MM_SHUFFLE(3, 1, 0, 2));
                let b_zxy = _mm_shuffle_ps(v_b, v_b, _MM_SHUFFLE(3, 1, 0, 2));
                let mul1 = _mm_mul_ps(a_yzx, b_zxy);
                let mul2 = _mm_mul_ps(a_zxy, b_yzx);
                let res = _mm_sub_ps(mul1, mul2);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            Self::new(
                self.y.mul_add(other.z, -(self.z * other.y)),
                self.z.mul_add(other.x, -(self.x * other.z)),
                self.x.mul_add(other.y, -(self.y * other.x)),
            )
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.y.mul_add(other.z, -(self.z * other.y)),
                self.z.mul_add(other.x, -(self.x * other.z)),
                self.x.mul_add(other.y, -(self.y * other.x)),
            )
        }
    }

    #[inline(always)]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        let t_inv = 1.0 - t;
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let v_t = _mm_set1_ps(t);
                let v_t_inv = _mm_set1_ps(t_inv);
                let res = _mm_add_ps(_mm_mul_ps(v_a, v_t_inv), _mm_mul_ps(v_b, v_t));
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let v_t = vdupq_n_f32(t);
                let v_t_inv = vdupq_n_f32(t_inv);
                let res = vmlaq_f32(vmulq_f32(v_a, v_t_inv), v_b, v_t);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x.mul_add(t_inv, other.x * t),
                self.y.mul_add(t_inv, other.y * t),
                self.z.mul_add(t_inv, other.z * t),
            )
        }
    }

    #[inline(always)]
    pub fn abs(self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ptr());
                let mask = _mm_set1_ps(-0.0);
                let res = _mm_andnot_ps(mask, v_v);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ptr());
                let res = vabsq_f32(v_v);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x.abs(), self.y.abs(), self.z.abs())
        }
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_min_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vminq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x.min(other.x), self.y.min(other.y), self.z.min(other.z))
        }
    }

    #[inline]
    pub fn max(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_max_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vmaxq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x.max(other.x), self.y.max(other.y), self.z.max(other.z))
        }
    }

    /// Component-wise reciprocal
    #[inline(always)]
    pub fn recip(self) -> Self {
        Self::new(self.x.recip(), self.y.recip(), self.z.recip())
    }

    /// Component-wise signum
    #[inline(always)]
    pub fn signum(self) -> Self {
        Self::new(self.x.signum(), self.y.signum(), self.z.signum())
    }

    /// Minimum element
    #[inline(always)]
    pub fn min_element(self) -> f32 {
        self.x.min(self.y).min(self.z)
    }

    /// Maximum element
    #[inline(always)]
    pub fn max_element(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Component-wise clamp
    #[inline(always)]
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self::new(
            self.x.clamp(min.x, max.x),
            self.y.clamp(min.y, max.y),
            self.z.clamp(min.z, max.z),
        )
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

    /// Reject from another vector (component perpendicular to from)
    #[inline]
    pub fn reject(self, from: Self) -> Self {
        self - self.project(from)
    }

    /// Reflect across a normal
    #[inline]
    pub fn reflect(self, normal: Self) -> Self {
        self - normal * (2.0 * self.dot(normal))
    }

    /// Refract through a surface with given normal and ratio of indices of refraction
    #[inline]
    pub fn refract(self, normal: Self, eta: f32) -> Self {
        let dot_ni = self.dot(normal);
        let k = 1.0 - eta * eta * (1.0 - dot_ni * dot_ni);
        if k < 0.0 {
            Self::ZERO
        } else {
            self * eta - normal * (eta * dot_ni + k.sqrt())
        }
    }

    /// Truncate to Vec2 (drop z component)
    #[inline(always)]
    pub fn truncate(self) -> crate::Vec2 {
        crate::Vec2::new(self.x, self.y)
    }

    /// Extend to Vec4
    #[inline(always)]
    pub fn extend(self, w: f32) -> crate::Vec4 {
        crate::Vec4::new(self.x, self.y, self.z, w)
    }

    /// Check if all components are finite
    #[inline(always)]
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Check if any component is NaN
    #[inline(always)]
    pub fn is_nan(self) -> bool {
        self.x.is_nan() || self.y.is_nan() || self.z.is_nan()
    }

    /// Approximate equality with epsilon
    #[inline]
    pub fn abs_diff_eq(self, other: Self, epsilon: f32) -> bool {
        (self.x - other.x).abs() <= epsilon
            && (self.y - other.y).abs() <= epsilon
            && (self.z - other.z).abs() <= epsilon
    }

    /// Angle between two vectors in radians
    #[inline]
    pub fn angle_between(self, other: Self) -> f32 {
        let dot = self.dot(other);
        let len_product = (self.length_squared() * other.length_squared()).sqrt();
        if len_product > 0.0 {
            (dot / len_product).clamp(-1.0, 1.0).acos()
        } else {
            0.0
        }
    }

    /// Create from array
    #[inline(always)]
    pub fn from_array(a: [f32; 3]) -> Self {
        Self::new(a[0], a[1], a[2])
    }

    /// Convert to array
    #[inline(always)]
    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    /// Create from slice, returns None if slice is too short
    #[inline]
    pub fn from_slice(slice: &[f32]) -> Option<Self> {
        if slice.len() >= 3 {
            Some(Self::new(slice[0], slice[1], slice[2]))
        } else {
            None
        }
    }

    /// Write to slice, panics if slice is too short
    #[inline]
    pub fn write_to_slice(self, slice: &mut [f32]) {
        assert!(slice.len() >= 3, "slice must have at least 3 elements");
        slice[0] = self.x;
        slice[1] = self.y;
        slice[2] = self.z;
    }

    /// Get reference to underlying array
    #[inline(always)]
    pub fn as_array(&self) -> &[f32; 3] {
        self.as_ref()
    }

    /// Get mutable reference to underlying array
    #[inline(always)]
    pub fn as_array_mut(&mut self) -> &mut [f32; 3] {
        self.as_mut()
    }
}

impl std::convert::AsRef<[f32; 3]> for Vec3 {
    #[inline(always)]
    fn as_ref(&self) -> &[f32; 3] {
        unsafe { &*(&self.x as *const f32 as *const [f32; 3]) }
    }
}

impl std::convert::AsMut<[f32; 3]> for Vec3 {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [f32; 3] {
        unsafe { &mut *(&mut self.x as *mut f32 as *mut [f32; 3]) }
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_add_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vaddq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
        }
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_sub_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vsubq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
        }
    }
}

impl std::ops::Mul for Vec3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_mul_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vmulq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x * other.x, self.y * other.y, self.z * other.z)
        }
    }
}

impl std::ops::Div for Vec3 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_a = _mm_load_ps(self.as_ptr());
                let v_b = _mm_load_ps(other.as_ptr());
                let res = _mm_div_ps(v_a, v_b);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_a = vld1q_f32(self.as_ptr());
                let v_b = vld1q_f32(other.as_ptr());
                let res = vdivq_f32(v_a, v_b);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x / other.x, self.y / other.y, self.z / other.z)
        }
    }
}

impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, scalar: f32) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ptr());
                let v_s = _mm_set1_ps(scalar);
                let res = _mm_mul_ps(v_v, v_s);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ptr());
                let res = vmulq_n_f32(v_v, scalar);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
        }
    }
}

impl std::ops::Mul<Vec3> for f32 {
    type Output = Vec3;
    #[inline(always)]
    fn mul(self, vec: Vec3) -> Vec3 {
        vec * self
    }
}

impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    #[inline(always)]
    fn div(self, scalar: f32) -> Self {
        Self {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
            _padding: 0.0,
        }
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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let v_v = _mm_load_ps(self.as_ptr());
                let zero = _mm_setzero_ps();
                let res = _mm_sub_ps(zero, v_v);
                let mut out = Self::ZERO;
                _mm_store_ps(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let v_v = vld1q_f32(self.as_ptr());
                let res = vnegq_f32(v_v);
                let mut out = Self::ZERO;
                vst1q_f32(out.as_mut_ptr(), res);
                out
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(-self.x, -self.y, -self.z)
        }
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
