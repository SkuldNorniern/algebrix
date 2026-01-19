//! x86_64 SIMD optimizations using SSE/AVX
//!
//! This module provides SIMD-accelerated implementations for x86_64 platforms.
//! Vec3 is now 16-byte aligned with padding, enabling direct aligned loads.

use crate::{Mat4, Vec3, Vec4};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// --- Vec3 Operations (16-byte aligned, direct loads) ---

#[inline(always)]
pub fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_add_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
    }
}

#[inline(always)]
pub fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_sub_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
    }
}

#[inline(always)]
pub fn vec3_mul(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_mul_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
    }
}

#[inline(always)]
pub fn vec3_div(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_div_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x / b.x, a.y / b.y, a.z / b.z)
    }
}

#[inline(always)]
pub fn vec3_mul_scalar(v: Vec3, s: f32) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(v.as_ptr());
        let v_s = _mm_set1_ps(s);
        let res = _mm_mul_ps(v_v, v_s);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(v.x * s, v.y * s, v.z * s)
    }
}

#[inline(always)]
pub fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let mul = _mm_mul_ps(v_a, v_b);
        let masked = _mm_blend_ps(mul, _mm_setzero_ps(), 0b1000);
        let shuf = _mm_movehdup_ps(masked);
        let sums = _mm_add_ps(masked, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

#[inline(always)]
pub fn vec3_length_squared(v: Vec3) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(v.as_ptr());
        let mul = _mm_mul_ps(v_v, v_v);
        let masked = _mm_blend_ps(mul, _mm_setzero_ps(), 0b1000);
        let shuf = _mm_movehdup_ps(masked);
        let sums = _mm_add_ps(masked, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v.x * v.x + v.y * v.y + v.z * v.z
    }
}

#[inline(always)]
pub fn vec3_normalize_fast(v: Vec3, len_sq: f32) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if len_sq > 0.0 {
            let v_v = _mm_load_ps(v.as_ptr());
            let v_len_sq = _mm_set1_ps(len_sq);
            let rsqrt = _mm_rsqrt_ps(v_len_sq);
            let half = _mm_set1_ps(0.5);
            let three = _mm_set1_ps(3.0);
            let muls = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
            let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls));
            let muls2 = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
            let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls2));
            let res = _mm_mul_ps(v_v, rsqrt);
            let mut out = Vec3::ZERO;
            _mm_store_ps(out.as_mut_ptr(), res);
            out
        } else {
            Vec3::ZERO
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Vec3::new(v.x * inv_len, v.y * inv_len, v.z * inv_len)
        } else {
            Vec3::ZERO
        }
    }
}

#[inline(always)]
pub fn vec3_min(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_min_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }
}

#[inline(always)]
pub fn vec3_max(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let res = _mm_max_ps(v_a, v_b);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }
}

#[inline(always)]
pub fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let a_yzx = _mm_shuffle_ps(v_a, v_a, 0b11_00_10_01);
        let b_yzx = _mm_shuffle_ps(v_b, v_b, 0b11_00_10_01);
        let a_zxy = _mm_shuffle_ps(v_a, v_a, 0b11_01_00_10);
        let b_zxy = _mm_shuffle_ps(v_b, v_b, 0b11_01_00_10);
        let res = _mm_sub_ps(_mm_mul_ps(a_yzx, b_zxy), _mm_mul_ps(a_zxy, b_yzx));
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x,
        )
    }
}

#[inline(always)]
pub fn vec3_lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let v_t = _mm_set1_ps(t);
        let diff = _mm_sub_ps(v_b, v_a);
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(diff, v_t, v_a)
        } else {
            _mm_add_ps(v_a, _mm_mul_ps(diff, v_t))
        };
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let t_inv = 1.0 - t;
        Vec3::new(
            a.x * t_inv + b.x * t,
            a.y * t_inv + b.y * t,
            a.z * t_inv + b.z * t,
        )
    }
}

#[inline(always)]
pub fn vec3_abs(v: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(v.as_ptr());
        let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF_u32 as i32));
        let res = _mm_and_ps(v_v, mask);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(v.x.abs(), v.y.abs(), v.z.abs())
    }
}

#[inline(always)]
pub fn vec3_neg(v: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(v.as_ptr());
        let res = _mm_sub_ps(_mm_setzero_ps(), v_v);
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(-v.x, -v.y, -v.z)
    }
}

#[inline(always)]
pub fn vec3_fma(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_load_ps(b.as_ptr());
        let v_c = _mm_load_ps(c.as_ptr());
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(v_a, v_b, v_c)
        } else {
            _mm_add_ps(_mm_mul_ps(v_a, v_b), v_c)
        };
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(
            a.x.mul_add(b.x, c.x),
            a.y.mul_add(b.y, c.y),
            a.z.mul_add(b.z, c.z),
        )
    }
}

#[inline(always)]
pub fn vec3_fma_scalar(a: Vec3, b: f32, c: Vec3) -> Vec3 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(a.as_ptr());
        let v_b = _mm_set1_ps(b);
        let v_c = _mm_load_ps(c.as_ptr());
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(v_a, v_b, v_c)
        } else {
            _mm_add_ps(_mm_mul_ps(v_a, v_b), v_c)
        };
        let mut out = Vec3::ZERO;
        _mm_store_ps(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec3::new(
            a.x.mul_add(b, c.x),
            a.y.mul_add(b, c.y),
            a.z.mul_add(b, c.z),
        )
    }
}

// --- Vec4 Operations ---

#[inline(always)]
pub fn vec4_add(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_add_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
    }
}

#[inline(always)]
pub fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_sub_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
    }
}

#[inline(always)]
pub fn vec4_mul(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_mul_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w)
    }
}

#[inline(always)]
pub fn vec4_div(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_div_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w)
    }
}

#[inline(always)]
pub fn vec4_mul_scalar(v: Vec4, s: f32) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(&v.x);
        let v_s = _mm_set1_ps(s);
        let res = _mm_mul_ps(v_v, v_s);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(v.x * s, v.y * s, v.z * s, v.w * s)
    }
}

#[inline(always)]
pub fn vec4_dot(a: Vec4, b: Vec4) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let mul = _mm_mul_ps(v_a, v_b);
        let shuf = _mm_movehdup_ps(mul);
        let sums = _mm_add_ps(mul, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }
}

#[inline(always)]
pub fn vec4_length_squared(v: Vec4) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(&v.x);
        let mul = _mm_mul_ps(v_v, v_v);
        let shuf = _mm_movehdup_ps(mul);
        let sums = _mm_add_ps(mul, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        _mm_cvtss_f32(result)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w
    }
}

#[inline(always)]
pub fn vec4_normalize_fast(v: Vec4, len_sq: f32) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if len_sq > 0.0 {
            let v_v = _mm_load_ps(&v.x);
            let v_len_sq = _mm_set1_ps(len_sq);
            let rsqrt = _mm_rsqrt_ps(v_len_sq);
            let half = _mm_set1_ps(0.5);
            let three = _mm_set1_ps(3.0);
            let muls = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
            let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls));
            let muls2 = _mm_mul_ps(_mm_mul_ps(v_len_sq, rsqrt), rsqrt);
            let rsqrt = _mm_mul_ps(_mm_mul_ps(half, rsqrt), _mm_sub_ps(three, muls2));
            let res = _mm_mul_ps(v_v, rsqrt);
            let mut out = Vec4::ZERO;
            _mm_store_ps(&mut out.x, res);
            out
        } else {
            Vec4::ZERO
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if len_sq > 0.0 {
            let inv_len = len_sq.sqrt().recip();
            Vec4::new(v.x * inv_len, v.y * inv_len, v.z * inv_len, v.w * inv_len)
        } else {
            Vec4::ZERO
        }
    }
}

#[inline(always)]
pub fn vec4_min(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_min_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z), a.w.min(b.w))
    }
}

#[inline(always)]
pub fn vec4_max(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let res = _mm_max_ps(v_a, v_b);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z), a.w.max(b.w))
    }
}

#[inline(always)]
pub fn vec4_lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let v_t = _mm_set1_ps(t);
        let diff = _mm_sub_ps(v_b, v_a);
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(diff, v_t, v_a)
        } else {
            _mm_add_ps(v_a, _mm_mul_ps(diff, v_t))
        };
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let t_inv = 1.0 - t;
        Vec4::new(
            a.x * t_inv + b.x * t,
            a.y * t_inv + b.y * t,
            a.z * t_inv + b.z * t,
            a.w * t_inv + b.w * t,
        )
    }
}

#[inline(always)]
pub fn vec4_abs(v: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(&v.x);
        let mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFF_FFFF_u32 as i32));
        let res = _mm_and_ps(v_v, mask);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(v.x.abs(), v.y.abs(), v.z.abs(), v.w.abs())
    }
}

#[inline(always)]
pub fn vec4_neg(v: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_v = _mm_load_ps(&v.x);
        let res = _mm_sub_ps(_mm_setzero_ps(), v_v);
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(-v.x, -v.y, -v.z, -v.w)
    }
}

#[inline(always)]
pub fn vec4_fma(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_load_ps(&b.x);
        let v_c = _mm_load_ps(&c.x);
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(v_a, v_b, v_c)
        } else {
            _mm_add_ps(_mm_mul_ps(v_a, v_b), v_c)
        };
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(
            a.x.mul_add(b.x, c.x),
            a.y.mul_add(b.y, c.y),
            a.z.mul_add(b.z, c.z),
            a.w.mul_add(b.w, c.w),
        )
    }
}

#[inline(always)]
pub fn vec4_fma_scalar(a: Vec4, b: f32, c: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let v_a = _mm_load_ps(&a.x);
        let v_b = _mm_set1_ps(b);
        let v_c = _mm_load_ps(&c.x);
        let res = if is_x86_feature_detected!("fma") {
            _mm_fmadd_ps(v_a, v_b, v_c)
        } else {
            _mm_add_ps(_mm_mul_ps(v_a, v_b), v_c)
        };
        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(
            a.x.mul_add(b, c.x),
            a.y.mul_add(b, c.y),
            a.z.mul_add(b, c.z),
            a.w.mul_add(b, c.w),
        )
    }
}

// --- Mat4 Operations ---

#[inline(always)]
pub fn mat4_mul_vec4(m: Mat4, v: Vec4) -> Vec4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let col0 = _mm_load_ps(&m.x_axis.x);
        let col1 = _mm_load_ps(&m.y_axis.x);
        let col2 = _mm_load_ps(&m.z_axis.x);
        let col3 = _mm_load_ps(&m.w_axis.x);

        let v_x = _mm_set1_ps(v.x);
        let v_y = _mm_set1_ps(v.y);
        let v_z = _mm_set1_ps(v.z);
        let v_w = _mm_set1_ps(v.w);

        let mut res = _mm_mul_ps(col0, v_x);
        if is_x86_feature_detected!("fma") {
            res = _mm_fmadd_ps(col1, v_y, res);
            res = _mm_fmadd_ps(col2, v_z, res);
            res = _mm_fmadd_ps(col3, v_w, res);
        } else {
            res = _mm_add_ps(res, _mm_mul_ps(col1, v_y));
            res = _mm_add_ps(res, _mm_mul_ps(col2, v_z));
            res = _mm_add_ps(res, _mm_mul_ps(col3, v_w));
        }

        let mut out = Vec4::ZERO;
        _mm_store_ps(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Vec4::new(
            m.x_axis.x * v.x + m.y_axis.x * v.y + m.z_axis.x * v.z + m.w_axis.x * v.w,
            m.x_axis.y * v.x + m.y_axis.y * v.y + m.z_axis.y * v.z + m.w_axis.y * v.w,
            m.x_axis.z * v.x + m.y_axis.z * v.y + m.z_axis.z * v.z + m.w_axis.z * v.w,
            m.x_axis.w * v.x + m.y_axis.w * v.y + m.z_axis.w * v.z + m.w_axis.w * v.w,
        )
    }
}

#[inline(always)]
pub fn mat4_mul_mat4(a: Mat4, b: Mat4) -> Mat4 {
    Mat4::new(
        mat4_mul_vec4(a, b.x_axis),
        mat4_mul_vec4(a, b.y_axis),
        mat4_mul_vec4(a, b.z_axis),
        mat4_mul_vec4(a, b.w_axis),
    )
}

#[inline(always)]
pub fn mat4_transpose(m: Mat4) -> Mat4 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        let col0 = _mm_load_ps(&m.x_axis.x);
        let col1 = _mm_load_ps(&m.y_axis.x);
        let col2 = _mm_load_ps(&m.z_axis.x);
        let col3 = _mm_load_ps(&m.w_axis.x);

        let mut row0 = col0;
        let mut row1 = col1;
        let mut row2 = col2;
        let mut row3 = col3;

        _MM_TRANSPOSE4_PS(&mut row0, &mut row1, &mut row2, &mut row3);

        let mut res0 = Vec4::ZERO;
        let mut res1 = Vec4::ZERO;
        let mut res2 = Vec4::ZERO;
        let mut res3 = Vec4::ZERO;

        _mm_store_ps(&mut res0.x, row0);
        _mm_store_ps(&mut res1.x, row1);
        _mm_store_ps(&mut res2.x, row2);
        _mm_store_ps(&mut res3.x, row3);

        Mat4::new(res0, res1, res2, res3)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        Mat4::new(
            Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x),
            Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y),
            Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z),
            Vec4::new(m.x_axis.w, m.y_axis.w, m.z_axis.w, m.w_axis.w),
        )
    }
}

// --- Scalar Utilities ---

#[inline(always)]
pub fn f32_rsqrt(x: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        if x <= 0.0 {
            return 0.0;
        }
        let v = _mm_set_ss(x);
        let rsqrt = _mm_rsqrt_ss(v);
        let half = _mm_set_ss(0.5);
        let three = _mm_set_ss(3.0);
        let muls = _mm_mul_ss(_mm_mul_ss(v, rsqrt), rsqrt);
        let rsqrt = _mm_mul_ss(_mm_mul_ss(half, rsqrt), _mm_sub_ss(three, muls));
        let muls2 = _mm_mul_ss(_mm_mul_ss(v, rsqrt), rsqrt);
        let rsqrt = _mm_mul_ss(_mm_mul_ss(half, rsqrt), _mm_sub_ss(three, muls2));
        _mm_cvtss_f32(rsqrt)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        if x <= 0.0 {
            return 0.0;
        }
        x.sqrt().recip()
    }
}

// --- Batch Operations ---

#[inline]
pub fn vec3_batch_dot(a: &[Vec3], b: &[Vec3], out: &mut [f32]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = vec3_dot(a[i], b[i]);
    }
}

#[inline]
pub fn vec3_batch_add(a: &[Vec3], b: &[Vec3], out: &mut [Vec3]) {
    let len = a.len().min(b.len()).min(out.len());
    for i in 0..len {
        out[i] = vec3_add(a[i], b[i]);
    }
}

#[inline]
pub fn vec3_batch_fma_scalar(a: &[Vec3], b: f32, c: &[Vec3], out: &mut [Vec3]) {
    let len = a.len().min(c.len()).min(out.len());
    for i in 0..len {
        out[i] = vec3_fma_scalar(a[i], b, c[i]);
    }
}
