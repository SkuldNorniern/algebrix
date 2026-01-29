//! Scalar math: constants (PI, TAU, etc.), [`lerp`], [`clamp`], [`clamp01`], [`smoothstep`],
//! angle conversion, [`min`]/[`max`], and [`f32_rsqrt`] (approximate 1/sqrt(x), SIMD when feature is on).
//!
//! # Example
//!
//! ```rust
//! use algebrix::utils;
//!
//! let t = 0.5;
//! let a = 0.0_f32;
//! let b = 10.0_f32;
//! assert!((utils::lerp(a, b, t) - 5.0).abs() < 1e-5);
//!
//! let x = 15.0;
//! assert_eq!(utils::clamp(x, 0.0, 10.0), 10.0);
//! assert_eq!(utils::clamp01(-0.5), 0.0);
//!
//! let r = utils::f32_rsqrt(4.0);
//! assert!(r > 0.49 && r < 0.51);
//! ```

use std::f32::consts;

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
use std::arch::aarch64::*;

/// Pi (3.14159...).
pub const PI: f32 = consts::PI;
/// 2 * Pi (full circle in radians).
pub const TAU: f32 = consts::TAU;
/// Pi / 2.
pub const FRAC_PI_2: f32 = consts::FRAC_PI_2;
/// Pi / 3.
pub const FRAC_PI_3: f32 = consts::FRAC_PI_3;
/// Pi / 4.
pub const FRAC_PI_4: f32 = consts::FRAC_PI_4;
/// Pi / 6.
pub const FRAC_PI_6: f32 = consts::FRAC_PI_6;
/// 2 / Pi.
pub const FRAC_2_PI: f32 = consts::FRAC_2_PI;
/// Sqrt(2).
pub const SQRT_2: f32 = consts::SQRT_2;
/// Euler's number e.
pub const E: f32 = consts::E;

/// Linear interpolation: `a + (b - a) * t`. At t=0 returns a, at t=1 returns b.
#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Clamp `value` to the range [min, max].
#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Clamp `value` to [0, 1].
#[inline]
pub fn clamp01(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

/// Hermite interpolation between 0 and 1. Returns 0 for x <= edge0, 1 for x >= edge1, smooth in between.
#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp01((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

/// Convert degrees to radians.
pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * PI / 180.0
}

/// Convert radians to degrees.
pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * 180.0 / PI
}

/// Minimum of two floats.
#[inline]
pub fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

/// Maximum of two floats.
#[inline]
pub fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Approximate reciprocal square root. Uses SIMD when the feature is on.
#[inline]
pub fn f32_rsqrt(x: f32) -> f32 {
    #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
    {
        unsafe {
            let x_simd = _mm_set_ss(x);
            let rsqrt_approx = _mm_rsqrt_ss(x_simd);
            let half = _mm_set_ss(0.5);
            let three = _mm_set_ss(3.0);
            let refined = _mm_mul_ss(
                rsqrt_approx,
                _mm_sub_ss(
                    three,
                    _mm_mul_ps(
                        _mm_mul_ss(half, x_simd),
                        _mm_mul_ss(rsqrt_approx, rsqrt_approx),
                    ),
                ),
            );
            _mm_cvtss_f32(refined)
        }
    }
    #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
    {
        unsafe {
            let x_simd = vdupq_n_f32(x);
            let rsqrt_approx = vrsqrteq_f32(x_simd);
            let muls = vmulq_f32(rsqrt_approx, rsqrt_approx);
            let rsqrt = vmulq_f32(rsqrt_approx, vrsqrtsq_f32(x_simd, muls));
            let muls2 = vmulq_f32(rsqrt, rsqrt);
            let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(x_simd, muls2));
            vgetq_lane_f32(rsqrt, 0)
        }
    }
    #[cfg(not(any(
        all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
        all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
    )))]
    {
        x.sqrt().recip()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
    }

    #[test]
    fn test_min_max() {
        assert_eq!(min(5.0, 10.0), 5.0);
        assert_eq!(max(5.0, 10.0), 10.0);
    }
}
