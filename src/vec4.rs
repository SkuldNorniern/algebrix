//! 4D float vector: homogenous coords, quaternion-like (x,y,z,w), colors (RGBA).
//!
//! Column vectors; multiply by Mat4 on the left. Use `.truncate()` to drop w and get Vec3.
//! [`normalize_fast`](Vec4::normalize_fast) uses rsqrt when the SIMD feature is on.
//!
//! With the SIMD feature the vector is stored directly in a SIMD register type
//! (`__m128` / `float32x4_t`); `x`/`y`/`z`/`w` remain accessible through `Deref`.
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

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) __m128);

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Vec4(pub(crate) float32x4_t);

#[cfg(not(any(
    all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
    all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
)))]
#[derive(Clone, Copy)]
#[repr(C, align(16))]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

/// Field-access view of a [`Vec4`]. With SIMD storage `Vec4` derefs to this,
/// so `v.x` etc. keep working.
#[repr(C, align(16))]
pub struct XYZW {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

#[cfg(any(
    all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
    all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
))]
impl std::ops::Deref for Vec4 {
    type Target = XYZW;
    #[inline(always)]
    fn deref(&self) -> &XYZW {
        // Same size and alignment; XYZW is repr(C, align(16)).
        unsafe { &*(self as *const Self as *const XYZW) }
    }
}

#[cfg(any(
    all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
    all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
))]
impl std::ops::DerefMut for Vec4 {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut XYZW {
        unsafe { &mut *(self as *mut Self as *mut XYZW) }
    }
}

/// Used for const construction of the SIMD-backed Vec4.
#[cfg(any(
    all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
    all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
))]
#[repr(C)]
union UnionCast {
    a: [f32; 4],
    v: Vec4,
}

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
impl Vec4 {
    /// The underlying SIMD register value.
    #[inline(always)]
    pub(crate) fn to_simd(self) -> __m128 {
        self.0
    }

    /// Wrap a SIMD register value.
    #[inline(always)]
    pub(crate) fn from_simd(v: __m128) -> Self {
        Self(v)
    }
}

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
impl Vec4 {
    /// The underlying SIMD register value.
    #[inline(always)]
    pub(crate) fn to_simd(self) -> float32x4_t {
        self.0
    }

    /// Wrap a SIMD register value.
    #[inline(always)]
    pub(crate) fn from_simd(v: float32x4_t) -> Self {
        Self(v)
    }
}

impl Vec4 {
    pub const ZERO: Vec4 = Vec4::new(0.0, 0.0, 0.0, 0.0);
    pub const ONE: Vec4 = Vec4::new(1.0, 1.0, 1.0, 1.0);

    #[cfg(any(
        all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
        all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
    ))]
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        unsafe { UnionCast { a: [x, y, z, w] }.v }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
        all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
    )))]
    #[inline(always)]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// All components set to the same value.
    #[inline(always)]
    pub fn splat(value: f32) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_set1_ps(value)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vdupq_n_f32(value)) }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(value, value, value, value)
        }
    }

    #[inline]
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    #[inline]
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    pub fn normalize(self) -> Self {
        let len_sq = self.length_squared();
        if len_sq > 0.0 {
            self * len_sq.sqrt().recip()
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
                    let v_len_sq = _mm_set1_ps(len_sq);
                    let rsqrt = _mm_rsqrt_ps(v_len_sq);
                    let half = _mm_set1_ps(0.5);
                    let three = _mm_set1_ps(3.0);
                    // Two Newton-Raphson steps: y' = 0.5 * y * (3 - x * y * y)
                    let muls = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
                    let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls));
                    let muls2 = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
                    let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls2));
                    Self(_mm_mul_ps(self.0, rsqrt))
                }
            }
            #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
            {
                unsafe {
                    let v_len_sq = vdupq_n_f32(len_sq);
                    let rsqrt = vrsqrteq_f32(v_len_sq);
                    let muls = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls));
                    let muls2 = vmulq_f32(rsqrt, rsqrt);
                    let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls2));
                    Self(vmulq_f32(self.0, rsqrt))
                }
            }
            #[cfg(not(any(
                all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
                all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
            )))]
            {
                self * len_sq.sqrt().recip()
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
                let mul = _mm_mul_ps(self.0, other.0);
                // SSE2-safe stand-in for _mm_movehdup_ps (duplicate odd lanes).
                let shuf = _mm_shuffle_ps(mul, mul, 0b11_11_01_01);
                let sums = _mm_add_ps(mul, shuf);
                let shuf2 = _mm_movehl_ps(sums, sums);
                _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { vaddvq_f32(vmulq_f32(self.0, other.0)) }
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
        self + (other - self) * t
    }

    #[inline]
    pub fn min(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_min_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vminq_f32(self.0, other.0)) }
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
            unsafe { Self(_mm_max_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vmaxq_f32(self.0, other.0)) }
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

    /// Component-wise absolute value
    #[inline(always)]
    pub fn abs(self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_andnot_ps(_mm_set1_ps(-0.0), self.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vabsq_f32(self.0)) }
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
        self.max(min).min(max)
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
        (self - other).abs().max_element() <= epsilon
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

impl std::fmt::Debug for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vec4")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .field("w", &self.w)
            .finish()
    }
}

impl PartialEq for Vec4 {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y && self.z == other.z && self.w == other.w
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
            unsafe { Self(_mm_add_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vaddq_f32(self.0, other.0)) }
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
            unsafe { Self(_mm_sub_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vsubq_f32(self.0, other.0)) }
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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_mul_ps(self.0, _mm_set1_ps(scalar))) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vmulq_n_f32(self.0, scalar)) }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(
                self.x * scalar,
                self.y * scalar,
                self.z * scalar,
                self.w * scalar,
            )
        }
    }
}

impl std::ops::Mul<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_mul_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vmulq_f32(self.0, other.0)) }
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
        vec * self
    }
}

impl std::ops::Div<f32> for Vec4 {
    type Output = Self;
    #[inline]
    // Multiplying by the reciprocal is intentional: one divide + four mults
    // beats four divides, and matches the previous behavior.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, scalar: f32) -> Self {
        self * scalar.recip()
    }
}

impl std::ops::Div<Vec4> for Vec4 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_div_ps(self.0, other.0)) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vdivq_f32(self.0, other.0)) }
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
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe { Self(_mm_xor_ps(self.0, _mm_set1_ps(-0.0))) }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe { Self(vnegq_f32(self.0)) }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self::new(-self.x, -self.y, -self.z, -self.w)
        }
    }
}

impl Default for Vec4 {
    #[inline]
    fn default() -> Self {
        Self::ZERO
    }
}

impl std::fmt::Display for Vec4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}, {}, {}]", self.x, self.y, self.z, self.w)
    }
}

impl std::ops::Index<usize> for Vec4 {
    type Output = f32;
    #[inline]
    fn index(&self, index: usize) -> &f32 {
        &self.as_ref()[index]
    }
}

impl std::ops::IndexMut<usize> for Vec4 {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut f32 {
        &mut self.as_mut()[index]
    }
}

impl From<[f32; 4]> for Vec4 {
    #[inline]
    fn from(a: [f32; 4]) -> Self {
        Self::new(a[0], a[1], a[2], a[3])
    }
}

impl From<Vec4> for [f32; 4] {
    #[inline]
    fn from(v: Vec4) -> Self {
        v.to_array()
    }
}

impl From<(f32, f32, f32, f32)> for Vec4 {
    #[inline]
    fn from(t: (f32, f32, f32, f32)) -> Self {
        Self::new(t.0, t.1, t.2, t.3)
    }
}

impl std::ops::AddAssign for Vec4 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl std::ops::SubAssign for Vec4 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl std::ops::MulAssign<f32> for Vec4 {
    #[inline]
    fn mul_assign(&mut self, scalar: f32) {
        *self = *self * scalar;
    }
}

impl std::ops::DivAssign<f32> for Vec4 {
    #[inline]
    fn div_assign(&mut self, scalar: f32) {
        *self = *self / scalar;
    }
}

impl std::iter::Sum for Vec4 {
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |a, b| a + b)
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

    #[test]
    fn test_vec4_const_new() {
        const V: Vec4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(V.to_array(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(Vec4::ZERO.to_array(), [0.0; 4]);
        assert_eq!(Vec4::ONE.to_array(), [1.0; 4]);
    }

    #[test]
    fn test_vec4_field_write() {
        let mut v = Vec4::ZERO;
        v.x = 1.0;
        v.w = 4.0;
        assert_eq!(v, Vec4::new(1.0, 0.0, 0.0, 4.0));
    }
}
