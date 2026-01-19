//! ARM64 (aarch64) SIMD optimizations using NEON
//!
//! This module provides SIMD-accelerated implementations for ARM64 platforms.
//! NEON is always available on aarch64, so we use it directly.
//!
//! Vec3 is now 16-byte aligned with padding, enabling direct aligned loads.

use crate::{Mat4, Vec3, Vec4};
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// --- Vec3 Operations (16-byte aligned, direct loads) ---

#[inline(always)]
pub fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vaddq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
    }
}

#[inline(always)]
pub fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vsubq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
    }
}

#[inline(always)]
pub fn vec3_mul(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vmulq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
    }
}

#[inline(always)]
pub fn vec3_div(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vdivq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x / b.x, a.y / b.y, a.z / b.z)
    }
}

#[inline(always)]
pub fn vec3_mul_scalar(v: Vec3, s: f32) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(v.as_ptr());
        let res = vmulq_n_f32(v_v, s);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(v.x * s, v.y * s, v.z * s)
    }
}

#[inline(always)]
pub fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let mul = vmulq_f32(v_a, v_b);
        let zero_w = vsetq_lane_f32(0.0, mul, 3);
        vaddvq_f32(zero_w)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}

#[inline(always)]
pub fn vec3_length_squared(v: Vec3) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(v.as_ptr());
        let mul = vmulq_f32(v_v, v_v);
        let zero_w = vsetq_lane_f32(0.0, mul, 3);
        vaddvq_f32(zero_w)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        v.x * v.x + v.y * v.y + v.z * v.z
    }
}

#[inline(always)]
pub fn vec3_normalize_fast(v: Vec3, len_sq: f32) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if len_sq > 0.0 {
            let v_v = vld1q_f32(v.as_ptr());
            let v_len_sq = vdupq_n_f32(len_sq);
            let rsqrt = vrsqrteq_f32(v_len_sq);
            let muls = vmulq_f32(rsqrt, rsqrt);
            let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls));
            let muls2 = vmulq_f32(rsqrt, rsqrt);
            let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls2));
            let res = vmulq_f32(v_v, rsqrt);
            let mut out = Vec3::ZERO;
            vst1q_f32(out.as_mut_ptr(), res);
            out
        } else {
            Vec3::ZERO
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vminq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
    }
}

#[inline(always)]
pub fn vec3_max(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let res = vmaxq_f32(v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
    }
}

#[inline(always)]
pub fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        
        let a_yzxw: [u8; 16] = [4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 12, 13, 14, 15];
        let b_zxyw: [u8; 16] = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15];
        
        let a_yzx = vqtbl1q_u8(vreinterpretq_u8_f32(v_a), vld1q_u8(a_yzxw.as_ptr()));
        let a_zxy = vqtbl1q_u8(vreinterpretq_u8_f32(v_a), vld1q_u8(b_zxyw.as_ptr()));
        let b_zxy = vqtbl1q_u8(vreinterpretq_u8_f32(v_b), vld1q_u8(b_zxyw.as_ptr()));
        let b_yzx = vqtbl1q_u8(vreinterpretq_u8_f32(v_b), vld1q_u8(a_yzxw.as_ptr()));
        
        let mul1 = vmulq_f32(vreinterpretq_f32_u8(a_yzx), vreinterpretq_f32_u8(b_zxy));
        let res = vfmsq_f32(mul1, vreinterpretq_f32_u8(a_zxy), vreinterpretq_f32_u8(b_yzx));
        
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let diff = vsubq_f32(v_b, v_a);
        let res = vfmaq_n_f32(v_a, diff, t);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(v.as_ptr());
        let res = vabsq_f32(v_v);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(v.x.abs(), v.y.abs(), v.z.abs())
    }
}

#[inline(always)]
pub fn vec3_neg(v: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(v.as_ptr());
        let res = vnegq_f32(v_v);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec3::new(-v.x, -v.y, -v.z)
    }
}

#[inline(always)]
pub fn vec3_fma(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_b = vld1q_f32(b.as_ptr());
        let v_c = vld1q_f32(c.as_ptr());
        let res = vfmaq_f32(v_c, v_a, v_b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(a.as_ptr());
        let v_c = vld1q_f32(c.as_ptr());
        let res = vfmaq_n_f32(v_c, v_a, b);
        let mut out = Vec3::ZERO;
        vst1q_f32(out.as_mut_ptr(), res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vaddq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
    }
}

#[inline(always)]
pub fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vsubq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
    }
}

#[inline(always)]
pub fn vec4_mul(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vmulq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w)
    }
}

#[inline(always)]
pub fn vec4_div(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vdivq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w)
    }
}

#[inline(always)]
pub fn vec4_mul_scalar(v: Vec4, s: f32) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(&v.x);
        let res = vmulq_n_f32(v_v, s);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(v.x * s, v.y * s, v.z * s, v.w * s)
    }
}

#[inline(always)]
pub fn vec4_dot(a: Vec4, b: Vec4) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let mul = vmulq_f32(v_a, v_b);
        vaddvq_f32(mul)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    }
}

#[inline(always)]
pub fn vec4_length_squared(v: Vec4) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(&v.x);
        let mul = vmulq_f32(v_v, v_v);
        vaddvq_f32(mul)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w
    }
}

#[inline(always)]
pub fn vec4_normalize_fast(v: Vec4, len_sq: f32) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if len_sq > 0.0 {
            let v_v = vld1q_f32(&v.x);
            let v_len_sq = vdupq_n_f32(len_sq);
            let rsqrt = vrsqrteq_f32(v_len_sq);
            let muls = vmulq_f32(rsqrt, rsqrt);
            let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls));
            let muls2 = vmulq_f32(rsqrt, rsqrt);
            let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v_len_sq, muls2));
            let res = vmulq_f32(v_v, rsqrt);
            let mut out = Vec4::ZERO;
            vst1q_f32(&mut out.x, res);
            out
        } else {
            Vec4::ZERO
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vminq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z), a.w.min(b.w))
    }
}

#[inline(always)]
pub fn vec4_max(a: Vec4, b: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let res = vmaxq_f32(v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z), a.w.max(b.w))
    }
}

#[inline(always)]
pub fn vec4_lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let diff = vsubq_f32(v_b, v_a);
        let res = vfmaq_n_f32(v_a, diff, t);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(&v.x);
        let res = vabsq_f32(v_v);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(v.x.abs(), v.y.abs(), v.z.abs(), v.w.abs())
    }
}

#[inline(always)]
pub fn vec4_neg(v: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_v = vld1q_f32(&v.x);
        let res = vnegq_f32(v_v);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        Vec4::new(-v.x, -v.y, -v.z, -v.w)
    }
}

#[inline(always)]
pub fn vec4_fma(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_b = vld1q_f32(&b.x);
        let v_c = vld1q_f32(&c.x);
        let res = vfmaq_f32(v_c, v_a, v_b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let v_a = vld1q_f32(&a.x);
        let v_c = vld1q_f32(&c.x);
        let res = vfmaq_n_f32(v_c, v_a, b);
        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let col0 = vld1q_f32(&m.x_axis.x);
        let col1 = vld1q_f32(&m.y_axis.x);
        let col2 = vld1q_f32(&m.z_axis.x);
        let col3 = vld1q_f32(&m.w_axis.x);

        let mut res = vmulq_n_f32(col0, v.x);
        res = vfmaq_n_f32(res, col1, v.y);
        res = vfmaq_n_f32(res, col2, v.z);
        res = vfmaq_n_f32(res, col3, v.w);

        let mut out = Vec4::ZERO;
        vst1q_f32(&mut out.x, res);
        out
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        let col0 = vld1q_f32(&m.x_axis.x);
        let col1 = vld1q_f32(&m.y_axis.x);
        let col2 = vld1q_f32(&m.z_axis.x);
        let col3 = vld1q_f32(&m.w_axis.x);

        let trans01 = vtrnq_f32(col0, col1);
        let trans23 = vtrnq_f32(col2, col3);

        let row0 = vcombine_f32(vget_low_f32(trans01.0), vget_low_f32(trans23.0));
        let row1 = vcombine_f32(vget_low_f32(trans01.1), vget_low_f32(trans23.1));
        let row2 = vcombine_f32(vget_high_f32(trans01.0), vget_high_f32(trans23.0));
        let row3 = vcombine_f32(vget_high_f32(trans01.1), vget_high_f32(trans23.1));

        let mut res_col0 = Vec4::ZERO;
        let mut res_col1 = Vec4::ZERO;
        let mut res_col2 = Vec4::ZERO;
        let mut res_col3 = Vec4::ZERO;

        vst1q_f32(&mut res_col0.x, row0);
        vst1q_f32(&mut res_col1.x, row1);
        vst1q_f32(&mut res_col2.x, row2);
        vst1q_f32(&mut res_col3.x, row3);

        Mat4::new(res_col0, res_col1, res_col2, res_col3)
    }
    #[cfg(not(target_arch = "aarch64"))]
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
    #[cfg(target_arch = "aarch64")]
    unsafe {
        if x <= 0.0 {
            return 0.0;
        }
        let v = vdupq_n_f32(x);
        let rsqrt = vrsqrteq_f32(v);
        let muls = vmulq_f32(rsqrt, rsqrt);
        let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v, muls));
        let muls2 = vmulq_f32(rsqrt, rsqrt);
        let rsqrt = vmulq_f32(rsqrt, vrsqrtsq_f32(v, muls2));
        vgetq_lane_f32(rsqrt, 0)
    }
    #[cfg(not(target_arch = "aarch64"))]
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
