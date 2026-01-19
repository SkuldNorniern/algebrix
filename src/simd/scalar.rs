//! Scalar fallback implementations (no SIMD)

use crate::{Mat4, Vec3, Vec4};

// --- Vec3 Operations ---

#[inline(always)]
pub fn vec3_add(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x + b.x, a.y + b.y, a.z + b.z)
}

#[inline(always)]
pub fn vec3_sub(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z)
}

#[inline(always)]
pub fn vec3_mul(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x * b.x, a.y * b.y, a.z * b.z)
}

#[inline(always)]
pub fn vec3_div(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x / b.x, a.y / b.y, a.z / b.z)
}

#[inline(always)]
pub fn vec3_mul_scalar(v: Vec3, s: f32) -> Vec3 {
    Vec3::new(v.x * s, v.y * s, v.z * s)
}

#[inline(always)]
pub fn vec3_dot(a: Vec3, b: Vec3) -> f32 {
    a.x.mul_add(b.x, a.y.mul_add(b.y, a.z * b.z))
}

#[inline(always)]
pub fn vec3_length_squared(v: Vec3) -> f32 {
    v.x.mul_add(v.x, v.y.mul_add(v.y, v.z * v.z))
}

#[inline(always)]
pub fn vec3_normalize_fast(v: Vec3, len_sq: f32) -> Vec3 {
    if len_sq > 0.0 {
        let inv_len = f32_rsqrt(len_sq);
        Vec3::new(v.x * inv_len, v.y * inv_len, v.z * inv_len)
    } else {
        Vec3::ZERO
    }
}

#[inline(always)]
pub fn vec3_min(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z))
}

#[inline(always)]
pub fn vec3_max(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z))
}

#[inline(always)]
pub fn vec3_cross(a: Vec3, b: Vec3) -> Vec3 {
    Vec3::new(
        a.y.mul_add(b.z, -(a.z * b.y)),
        a.z.mul_add(b.x, -(a.x * b.z)),
        a.x.mul_add(b.y, -(a.y * b.x)),
    )
}

#[inline(always)]
pub fn vec3_lerp(a: Vec3, b: Vec3, t: f32) -> Vec3 {
    let t_inv = 1.0 - t;
    Vec3::new(
        a.x.mul_add(t_inv, b.x * t),
        a.y.mul_add(t_inv, b.y * t),
        a.z.mul_add(t_inv, b.z * t),
    )
}

#[inline(always)]
pub fn vec3_abs(v: Vec3) -> Vec3 {
    Vec3::new(v.x.abs(), v.y.abs(), v.z.abs())
}

#[inline(always)]
pub fn vec3_neg(v: Vec3) -> Vec3 {
    Vec3::new(-v.x, -v.y, -v.z)
}

#[inline(always)]
pub fn vec3_fma(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    Vec3::new(
        a.x.mul_add(b.x, c.x),
        a.y.mul_add(b.y, c.y),
        a.z.mul_add(b.z, c.z),
    )
}

#[inline(always)]
pub fn vec3_fma_scalar(a: Vec3, b: f32, c: Vec3) -> Vec3 {
    Vec3::new(
        a.x.mul_add(b, c.x),
        a.y.mul_add(b, c.y),
        a.z.mul_add(b, c.z),
    )
}

// --- Vec4 Operations ---

#[inline(always)]
pub fn vec4_add(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w)
}

#[inline(always)]
pub fn vec4_sub(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w)
}

#[inline(always)]
pub fn vec4_mul(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w)
}

#[inline(always)]
pub fn vec4_div(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w)
}

#[inline(always)]
pub fn vec4_mul_scalar(v: Vec4, s: f32) -> Vec4 {
    Vec4::new(v.x * s, v.y * s, v.z * s, v.w * s)
}

#[inline(always)]
pub fn vec4_dot(a: Vec4, b: Vec4) -> f32 {
    a.x.mul_add(b.x, a.y.mul_add(b.y, a.z.mul_add(b.z, a.w * b.w)))
}

#[inline(always)]
pub fn vec4_length_squared(v: Vec4) -> f32 {
    v.x.mul_add(v.x, v.y.mul_add(v.y, v.z.mul_add(v.z, v.w * v.w)))
}

#[inline(always)]
pub fn vec4_normalize_fast(v: Vec4, len_sq: f32) -> Vec4 {
    if len_sq > 0.0 {
        let inv_len = f32_rsqrt(len_sq);
        Vec4::new(v.x * inv_len, v.y * inv_len, v.z * inv_len, v.w * inv_len)
    } else {
        Vec4::ZERO
    }
}

#[inline(always)]
pub fn vec4_min(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z), a.w.min(b.w))
}

#[inline(always)]
pub fn vec4_max(a: Vec4, b: Vec4) -> Vec4 {
    Vec4::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z), a.w.max(b.w))
}

#[inline(always)]
pub fn vec4_lerp(a: Vec4, b: Vec4, t: f32) -> Vec4 {
    let t_inv = 1.0 - t;
    Vec4::new(
        a.x.mul_add(t_inv, b.x * t),
        a.y.mul_add(t_inv, b.y * t),
        a.z.mul_add(t_inv, b.z * t),
        a.w.mul_add(t_inv, b.w * t),
    )
}

#[inline(always)]
pub fn vec4_abs(v: Vec4) -> Vec4 {
    Vec4::new(v.x.abs(), v.y.abs(), v.z.abs(), v.w.abs())
}

#[inline(always)]
pub fn vec4_neg(v: Vec4) -> Vec4 {
    Vec4::new(-v.x, -v.y, -v.z, -v.w)
}

#[inline(always)]
pub fn vec4_fma(a: Vec4, b: Vec4, c: Vec4) -> Vec4 {
    Vec4::new(
        a.x.mul_add(b.x, c.x),
        a.y.mul_add(b.y, c.y),
        a.z.mul_add(b.z, c.z),
        a.w.mul_add(b.w, c.w),
    )
}

#[inline(always)]
pub fn vec4_fma_scalar(a: Vec4, b: f32, c: Vec4) -> Vec4 {
    Vec4::new(
        a.x.mul_add(b, c.x),
        a.y.mul_add(b, c.y),
        a.z.mul_add(b, c.z),
        a.w.mul_add(b, c.w),
    )
}

// --- Mat4 Operations ---

#[inline(always)]
pub fn mat4_mul_vec4(m: Mat4, v: Vec4) -> Vec4 {
    Vec4::new(
        m.x_axis.x.mul_add(v.x, m.y_axis.x.mul_add(v.y, m.z_axis.x.mul_add(v.z, m.w_axis.x * v.w))),
        m.x_axis.y.mul_add(v.x, m.y_axis.y.mul_add(v.y, m.z_axis.y.mul_add(v.z, m.w_axis.y * v.w))),
        m.x_axis.z.mul_add(v.x, m.y_axis.z.mul_add(v.y, m.z_axis.z.mul_add(v.z, m.w_axis.z * v.w))),
        m.x_axis.w.mul_add(v.x, m.y_axis.w.mul_add(v.y, m.z_axis.w.mul_add(v.z, m.w_axis.w * v.w))),
    )
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
    Mat4::new(
        Vec4::new(m.x_axis.x, m.y_axis.x, m.z_axis.x, m.w_axis.x),
        Vec4::new(m.x_axis.y, m.y_axis.y, m.z_axis.y, m.w_axis.y),
        Vec4::new(m.x_axis.z, m.y_axis.z, m.z_axis.z, m.w_axis.z),
        Vec4::new(m.x_axis.w, m.y_axis.w, m.z_axis.w, m.w_axis.w),
    )
}

// --- Scalar Utilities ---

#[inline(always)]
pub fn f32_rsqrt(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    let y = f32::from_bits(0x5f37_5a86 - (x.to_bits() >> 1));
    let y = y * (1.5 - 0.5 * x * y * y);
    y * (1.5 - 0.5 * x * y * y)
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
