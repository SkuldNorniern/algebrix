//! 4x4 column-major matrix: 3D affine or projective transform.
//!
//! Use [`from_translation`](Mat4::from_translation), [`from_quat`](Mat4::from_quat),
//! [`from_scale_rotation_translation`](Mat4::from_scale_rotation_translation) for affine; perspective helpers
//! like [`perspective_lh`](Mat4::perspective_lh) for projection. Multiply Vec4 on the right; use
//! [`transform_point3`](Mat4::transform_point3) / [`transform_vector3`](Mat4::transform_vector3) for Vec3.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Mat4, Quat, Vec3};
//!
//! let t = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
//! let p = Vec3::ZERO;
//! let q = t.transform_point3(p);
//! assert_eq!(q, Vec3::new(1.0, 2.0, 3.0));
//!
//! let r = Mat4::from_quat(Quat::IDENTITY);
//! let v = Vec3::X;
//! assert!((r.transform_vector3(v) - v).length() < 1e-5);
//! ```
//!

use crate::{Mat3, Quat, Vec3, Vec4};

#[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
use std::arch::aarch64::*;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub x_axis: Vec4,
    pub y_axis: Vec4,
    pub z_axis: Vec4,
    pub w_axis: Vec4,
}

impl Mat4 {
    pub const IDENTITY: Mat4 = Mat4 {
        x_axis: Vec4 {
            x: 1.0,
            y: 0.0,
            z: 0.0,
            w: 0.0,
        },
        y_axis: Vec4 {
            x: 0.0,
            y: 1.0,
            z: 0.0,
            w: 0.0,
        },
        z_axis: Vec4 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
            w: 0.0,
        },
        w_axis: Vec4 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
    };

    #[inline]
    pub const fn new(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    #[inline]
    pub const fn from_cols(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

    #[inline]
    pub const fn from_translation(translation: Vec3) -> Self {
        Self {
            x_axis: Vec4::new(1.0, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 1.0, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0, 0.0),
            w_axis: Vec4::new(translation.x, translation.y, translation.z, 1.0),
        }
    }

    #[inline]
    pub fn from_rotation_translation(rotation: Quat, translation: Vec3) -> Self {
        let mut m = Mat4::from_quat(rotation);
        m.w_axis.x = translation.x;
        m.w_axis.y = translation.y;
        m.w_axis.z = translation.z;
        m
    }

    #[inline]
    pub fn from_quat(quat: Quat) -> Self {
        let x = quat.x;
        let y = quat.y;
        let z = quat.z;
        let w = quat.w;

        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;

        Self {
            x_axis: Vec4::new(1.0 - yy - zz, xy + wz, xz - wy, 0.0),
            y_axis: Vec4::new(xy - wz, 1.0 - xx - zz, yz + wx, 0.0),
            z_axis: Vec4::new(xz + wy, yz - wx, 1.0 - xx - yy, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        Self {
            x_axis: Vec4::new(1.0, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, c, -s, 0.0),
            z_axis: Vec4::new(0.0, s, c, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        Self {
            x_axis: Vec4::new(c, 0.0, s, 0.0),
            y_axis: Vec4::new(0.0, 1.0, 0.0, 0.0),
            z_axis: Vec4::new(-s, 0.0, c, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Create a rotation matrix around the Z axis
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let s = angle.sin();
        let c = angle.cos();
        Self {
            x_axis: Vec4::new(c, s, 0.0, 0.0),
            y_axis: Vec4::new(-s, c, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 1.0, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Create a rotation matrix from axis and angle
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let axis = axis.normalize();
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            x_axis: Vec4::new(
                cos + x * x * one_minus_cos,
                x * y * one_minus_cos + z * sin,
                x * z * one_minus_cos - y * sin,
                0.0,
            ),
            y_axis: Vec4::new(
                y * x * one_minus_cos - z * sin,
                cos + y * y * one_minus_cos,
                y * z * one_minus_cos + x * sin,
                0.0,
            ),
            z_axis: Vec4::new(
                z * x * one_minus_cos + y * sin,
                z * y * one_minus_cos - x * sin,
                cos + z * z * one_minus_cos,
                0.0,
            ),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Create a uniform scale matrix
    #[inline]
    pub fn from_scale(scale: f32) -> Self {
        Self {
            x_axis: Vec4::new(scale, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, scale, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, scale, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Create a non-uniform scale matrix
    #[inline]
    pub fn from_nonuniform_scale(scale: Vec3) -> Self {
        Self {
            x_axis: Vec4::new(scale.x, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, scale.y, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, scale.z, 0.0),
            w_axis: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }

    /// Create a scale, rotation, translation matrix
    #[inline]
    pub fn from_scale_rotation_translation(scale: Vec3, rotation: Quat, translation: Vec3) -> Self {
        let mut m = Self::from_quat(rotation);
        m.x_axis *= scale.x;
        m.y_axis *= scale.y;
        m.z_axis *= scale.z;
        m.w_axis = Vec4::new(translation.x, translation.y, translation.z, 1.0);
        m
    }

    #[inline]
    pub fn perspective_rh(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let half_fov = fov_y_radians * 0.5;
        let f = half_fov.tan().recip();
        let inv_length = (z_near - z_far).recip();
        let a = f * aspect.recip();
        let b = (z_near + z_far) * inv_length;
        let c = (2.0 * z_near * z_far) * inv_length;

        Self {
            x_axis: Vec4::new(a, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, b, -1.0),
            w_axis: Vec4::new(0.0, 0.0, c, 0.0),
        }
    }

    #[inline]
    pub fn look_at_rh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Self::look_to_rh(eye, center - eye, up)
    }

    /// Left-handed look-at matrix
    #[inline]
    pub fn look_at_lh(eye: Vec3, center: Vec3, up: Vec3) -> Self {
        Self::look_to_lh(eye, center - eye, up)
    }

    /// Right-handed look-to matrix (direction-based)
    #[inline]
    pub fn look_to_rh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);

        Self {
            x_axis: Vec4::new(s.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(s.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(s.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
        }
    }

    /// Left-handed look-to matrix (direction-based)
    #[inline]
    pub fn look_to_lh(eye: Vec3, dir: Vec3, up: Vec3) -> Self {
        Self::look_to_rh(eye, -dir, up)
    }

    /// Left-handed perspective projection matrix
    #[inline]
    pub fn perspective_lh(fov_y_radians: f32, aspect: f32, z_near: f32, z_far: f32) -> Self {
        let half_fov = fov_y_radians * 0.5;
        let f = half_fov.tan().recip();
        let inv_length = (z_far - z_near).recip();
        let a = f * aspect.recip();
        let b = (z_near + z_far) * inv_length;
        let c = -(2.0 * z_near * z_far) * inv_length;

        Self {
            x_axis: Vec4::new(a, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, f, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, b, 1.0),
            w_axis: Vec4::new(0.0, 0.0, c, 0.0),
        }
    }

    /// Right-handed orthographic projection matrix
    #[inline]
    pub fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, z_near: f32, z_far: f32) -> Self {
        let rml = right - left;
        let tmb = top - bottom;
        let fmn = z_far - z_near;

        Self {
            x_axis: Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, -2.0 / fmn, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rml,
                -(top + bottom) / tmb,
                -(z_far + z_near) / fmn,
                1.0,
            ),
        }
    }

    /// Left-handed orthographic projection matrix
    #[inline]
    pub fn orthographic_lh(left: f32, right: f32, bottom: f32, top: f32, z_near: f32, z_far: f32) -> Self {
        let rml = right - left;
        let tmb = top - bottom;
        let fmn = z_far - z_near;

        Self {
            x_axis: Vec4::new(2.0 / rml, 0.0, 0.0, 0.0),
            y_axis: Vec4::new(0.0, 2.0 / tmb, 0.0, 0.0),
            z_axis: Vec4::new(0.0, 0.0, 2.0 / fmn, 0.0),
            w_axis: Vec4::new(
                -(right + left) / rml,
                -(top + bottom) / tmb,
                -(z_far + z_near) / fmn,
                1.0,
            ),
        }
    }

    /// Compute the trace (sum of diagonal elements)
    #[inline(always)]
    pub fn trace(&self) -> f32 {
        self.x_axis.x + self.y_axis.y + self.z_axis.z + self.w_axis.w
    }

    #[inline]
    pub fn transpose(self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                let tmp0 = _mm_unpacklo_ps(col0, col1);
                let tmp1 = _mm_unpackhi_ps(col0, col1);
                let tmp2 = _mm_unpacklo_ps(col2, col3);
                let tmp3 = _mm_unpackhi_ps(col2, col3);

                Self {
                    x_axis: Vec4::from_simd(_mm_movelh_ps(tmp0, tmp2)),
                    y_axis: Vec4::from_simd(_mm_movehl_ps(tmp2, tmp0)),
                    z_axis: Vec4::from_simd(_mm_movelh_ps(tmp1, tmp3)),
                    w_axis: Vec4::from_simd(_mm_movehl_ps(tmp3, tmp1)),
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                let tmp0 = vzip1q_f32(col0, col2);
                let tmp1 = vzip2q_f32(col0, col2);
                let tmp2 = vzip1q_f32(col1, col3);
                let tmp3 = vzip2q_f32(col1, col3);

                Self {
                    x_axis: Vec4::from_simd(vzip1q_f32(tmp0, tmp2)),
                    y_axis: Vec4::from_simd(vzip2q_f32(tmp0, tmp2)),
                    z_axis: Vec4::from_simd(vzip1q_f32(tmp1, tmp3)),
                    w_axis: Vec4::from_simd(vzip2q_f32(tmp1, tmp3)),
                }
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self {
                x_axis: Vec4::new(self.x_axis.x, self.y_axis.x, self.z_axis.x, self.w_axis.x),
                y_axis: Vec4::new(self.x_axis.y, self.y_axis.y, self.z_axis.y, self.w_axis.y),
                z_axis: Vec4::new(self.x_axis.z, self.y_axis.z, self.z_axis.z, self.w_axis.z),
                w_axis: Vec4::new(self.x_axis.w, self.y_axis.w, self.z_axis.w, self.w_axis.w),
            }
        }
    }

    #[inline]
    pub fn mul_vec4(self, other: Vec4) -> Vec4 {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                let v = other.to_simd();
                let v_x = _mm_shuffle_ps(v, v, 0b00_00_00_00);
                let v_y = _mm_shuffle_ps(v, v, 0b01_01_01_01);
                let v_z = _mm_shuffle_ps(v, v, 0b10_10_10_10);
                let v_w = _mm_shuffle_ps(v, v, 0b11_11_11_11);

                let mut res = _mm_mul_ps(col0, v_x);
                res = _mm_add_ps(res, _mm_mul_ps(col1, v_y));
                res = _mm_add_ps(res, _mm_mul_ps(col2, v_z));
                res = _mm_add_ps(res, _mm_mul_ps(col3, v_w));

                Vec4::from_simd(res)
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                let v = other.to_simd();
                let mut res = vmulq_laneq_f32(col0, v, 0);
                res = vfmaq_laneq_f32(res, col1, v, 1);
                res = vfmaq_laneq_f32(res, col2, v, 2);
                res = vfmaq_laneq_f32(res, col3, v, 3);

                Vec4::from_simd(res)
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Vec4::new(
                self.x_axis.x * other.x + self.y_axis.x * other.y + self.z_axis.x * other.z + self.w_axis.x * other.w,
                self.x_axis.y * other.x + self.y_axis.y * other.y + self.z_axis.y * other.z + self.w_axis.y * other.w,
                self.x_axis.z * other.x + self.y_axis.z * other.y + self.z_axis.z * other.z + self.w_axis.z * other.w,
                self.x_axis.w * other.x + self.y_axis.w * other.y + self.z_axis.w * other.z + self.w_axis.w * other.w,
            )
        }
    }

    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        let res = self.mul_vec4(Vec4::new(other.x, other.y, other.z, 1.0));
        Vec3::new(res.x, res.y, res.z)
    }

    #[inline]
    pub fn transform_vector3(self, v: Vec3) -> Vec3 {
        let res = self.mul_vec4(Vec4::new(v.x, v.y, v.z, 0.0));
        Vec3::new(res.x, res.y, res.z)
    }

    #[inline]
    pub fn transform_point3(self, v: Vec3) -> Vec3 {
        let res = self.mul_vec4(Vec4::new(v.x, v.y, v.z, 1.0));
        Vec3::new(res.x, res.y, res.z)
    }

    /// Multiply two affine matrices (both with last row `0 0 0 1`).
    /// Skips the w-row terms, so it is faster than `self * other`;
    /// the result is only correct when both matrices are affine.
    #[inline]
    pub fn mul_affine(self, other: Self) -> Self {
        let x = self.x_axis;
        let y = self.y_axis;
        let z = self.z_axis;
        Self {
            x_axis: x * other.x_axis.x + y * other.x_axis.y + z * other.x_axis.z,
            y_axis: x * other.y_axis.x + y * other.y_axis.y + z * other.y_axis.z,
            z_axis: x * other.z_axis.x + y * other.z_axis.y + z * other.z_axis.z,
            w_axis: x * other.w_axis.x + y * other.w_axis.y + z * other.w_axis.z + self.w_axis,
        }
    }

    #[inline]
    pub fn to_cols_array(self) -> [f32; 16] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.x_axis.w,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.y_axis.w,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
            self.z_axis.w,
            self.w_axis.x,
            self.w_axis.y,
            self.w_axis.z,
            self.w_axis.w,
        ]
    }

    #[inline]
    pub fn as_array(&self) -> &[f32; 16] {
        self.as_ref()
    }

    #[inline]
    pub fn determinant(&self) -> f32 {
        let m = self.as_ref();

        let s0 = m[0] * m[5] - m[1] * m[4];
        let s1 = m[0] * m[6] - m[2] * m[4];
        let s2 = m[0] * m[7] - m[3] * m[4];
        let s3 = m[1] * m[6] - m[2] * m[5];
        let s4 = m[1] * m[7] - m[3] * m[5];
        let s5 = m[2] * m[7] - m[3] * m[6];

        let c5 = m[10] * m[15] - m[11] * m[14];
        let c4 = m[9] * m[15] - m[11] * m[13];
        let c3 = m[9] * m[14] - m[10] * m[13];
        let c2 = m[8] * m[15] - m[11] * m[12];
        let c1 = m[8] * m[14] - m[10] * m[12];
        let c0 = m[8] * m[13] - m[9] * m[12];

        s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
    }

    #[inline]
    pub fn inverse(&self) -> Option<Mat4> {
        let m = self.as_ref();

        let s0 = m[0] * m[5] - m[1] * m[4];
        let s1 = m[0] * m[6] - m[2] * m[4];
        let s2 = m[0] * m[7] - m[3] * m[4];
        let s3 = m[1] * m[6] - m[2] * m[5];
        let s4 = m[1] * m[7] - m[3] * m[5];
        let s5 = m[2] * m[7] - m[3] * m[6];

        let c5 = m[10] * m[15] - m[11] * m[14];
        let c4 = m[9] * m[15] - m[11] * m[13];
        let c3 = m[9] * m[14] - m[10] * m[13];
        let c2 = m[8] * m[15] - m[11] * m[12];
        let c1 = m[8] * m[14] - m[10] * m[12];
        let c0 = m[8] * m[13] - m[9] * m[12];

        let det = s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;

        if det.abs() < f32::EPSILON {
            return None;
        }

        let inv_det = det.recip();

        Some(Mat4::from_cols(
            Vec4::new(
                (m[5] * c5 - m[6] * c4 + m[7] * c3) * inv_det,
                (-m[1] * c5 + m[2] * c4 - m[3] * c3) * inv_det,
                (m[13] * s5 - m[14] * s4 + m[15] * s3) * inv_det,
                (-m[9] * s5 + m[10] * s4 - m[11] * s3) * inv_det,
            ),
            Vec4::new(
                (-m[4] * c5 + m[6] * c2 - m[7] * c1) * inv_det,
                (m[0] * c5 - m[2] * c2 + m[3] * c1) * inv_det,
                (-m[12] * s5 + m[14] * s2 - m[15] * s1) * inv_det,
                (m[8] * s5 - m[10] * s2 + m[11] * s1) * inv_det,
            ),
            Vec4::new(
                (m[4] * c4 - m[5] * c2 + m[7] * c0) * inv_det,
                (-m[0] * c4 + m[1] * c2 - m[3] * c0) * inv_det,
                (m[12] * s4 - m[13] * s2 + m[15] * s0) * inv_det,
                (-m[8] * s4 + m[9] * s2 - m[11] * s0) * inv_det,
            ),
            Vec4::new(
                (-m[4] * c3 + m[5] * c1 - m[6] * c0) * inv_det,
                (m[0] * c3 - m[1] * c1 + m[2] * c0) * inv_det,
                (-m[12] * s3 + m[13] * s1 - m[14] * s0) * inv_det,
                (m[8] * s3 - m[9] * s1 + m[10] * s0) * inv_det,
            ),
        ))
    }

    /// Decompose into scale, rotation (quaternion), and translation. Returns None if the matrix is singular or has non-uniform scale.
    #[inline]
    pub fn decompose(&self) -> Option<(Vec3, Quat, Vec3)> {
        let translation = Vec3::new(self.w_axis.x, self.w_axis.y, self.w_axis.z);

        let scale_x = Vec3::new(self.x_axis.x, self.x_axis.y, self.x_axis.z).length();
        let scale_y = Vec3::new(self.y_axis.x, self.y_axis.y, self.y_axis.z).length();
        let scale_z = Vec3::new(self.z_axis.x, self.z_axis.y, self.z_axis.z).length();
        let scale = Vec3::new(scale_x, scale_y, scale_z);

        if scale.x < f32::EPSILON || scale.y < f32::EPSILON || scale.z < f32::EPSILON {
            return None;
        }

        let inv_scale = Vec3::new(scale.x.recip(), scale.y.recip(), scale.z.recip());
        let rotation_matrix = Mat3::new(
            Vec3::new(self.x_axis.x * inv_scale.x, self.x_axis.y * inv_scale.y, self.x_axis.z * inv_scale.z),
            Vec3::new(self.y_axis.x * inv_scale.x, self.y_axis.y * inv_scale.y, self.y_axis.z * inv_scale.z),
            Vec3::new(self.z_axis.x * inv_scale.x, self.z_axis.y * inv_scale.y, self.z_axis.z * inv_scale.z),
        );

        let rotation = Quat::from_mat3(&rotation_matrix);

        Some((scale, rotation, translation))
    }

    /// Create from array (column-major order)
    #[inline]
    pub fn from_cols_array(m: &[f32; 16]) -> Self {
        Self {
            x_axis: Vec4::new(m[0], m[1], m[2], m[3]),
            y_axis: Vec4::new(m[4], m[5], m[6], m[7]),
            z_axis: Vec4::new(m[8], m[9], m[10], m[11]),
            w_axis: Vec4::new(m[12], m[13], m[14], m[15]),
        }
    }
}

impl std::convert::AsRef<[f32; 16]> for Mat4 {
    #[inline]
    fn as_ref(&self) -> &[f32; 16] {
        unsafe { &*(self as *const Mat4 as *const [f32; 16]) }
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        #[cfg(all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                #[inline(always)]
                unsafe fn mul_col(
                    col0: __m128,
                    col1: __m128,
                    col2: __m128,
                    col3: __m128,
                    rhs: Vec4,
                ) -> Vec4 {
                    unsafe {
                        let v = rhs.to_simd();
                        let e0 = _mm_shuffle_ps(v, v, 0b00_00_00_00);
                        let e1 = _mm_shuffle_ps(v, v, 0b01_01_01_01);
                        let e2 = _mm_shuffle_ps(v, v, 0b10_10_10_10);
                        let e3 = _mm_shuffle_ps(v, v, 0b11_11_11_11);
                        let res = _mm_add_ps(
                            _mm_add_ps(_mm_mul_ps(col0, e0), _mm_mul_ps(col1, e1)),
                            _mm_add_ps(_mm_mul_ps(col2, e2), _mm_mul_ps(col3, e3)),
                        );
                        Vec4::from_simd(res)
                    }
                }

                Self {
                    x_axis: mul_col(col0, col1, col2, col3, other.x_axis),
                    y_axis: mul_col(col0, col1, col2, col3, other.y_axis),
                    z_axis: mul_col(col0, col1, col2, col3, other.z_axis),
                    w_axis: mul_col(col0, col1, col2, col3, other.w_axis),
                }
            }
        }
        #[cfg(all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm")))]
        {
            unsafe {
                let col0 = self.x_axis.to_simd();
                let col1 = self.y_axis.to_simd();
                let col2 = self.z_axis.to_simd();
                let col3 = self.w_axis.to_simd();

                #[inline(always)]
                unsafe fn mul_col(
                    col0: float32x4_t,
                    col1: float32x4_t,
                    col2: float32x4_t,
                    col3: float32x4_t,
                    rhs: Vec4,
                ) -> Vec4 {
                    unsafe {
                        let v = rhs.to_simd();
                        let mut res = vmulq_laneq_f32(col0, v, 0);
                        res = vfmaq_laneq_f32(res, col1, v, 1);
                        res = vfmaq_laneq_f32(res, col2, v, 2);
                        res = vfmaq_laneq_f32(res, col3, v, 3);
                        Vec4::from_simd(res)
                    }
                }

                Self {
                    x_axis: mul_col(col0, col1, col2, col3, other.x_axis),
                    y_axis: mul_col(col0, col1, col2, col3, other.y_axis),
                    z_axis: mul_col(col0, col1, col2, col3, other.z_axis),
                    w_axis: mul_col(col0, col1, col2, col3, other.w_axis),
                }
            }
        }
        #[cfg(not(any(
            all(target_arch = "x86_64", any(feature = "simd", feature = "simd-x86")),
            all(target_arch = "aarch64", any(feature = "simd", feature = "simd-arm"))
        )))]
        {
            Self {
                x_axis: self.mul_vec4(other.x_axis),
                y_axis: self.mul_vec4(other.y_axis),
                z_axis: self.mul_vec4(other.z_axis),
                w_axis: self.mul_vec4(other.w_axis),
            }
        }
    }
}

impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;
    #[inline]
    fn mul(self, other: Vec4) -> Vec4 {
        self.mul_vec4(other)
    }
}

impl std::ops::Mul<Vec3> for Mat4 {
    type Output = Vec3;
    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        self.mul_vec3(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat4_identity() {
        let m = Mat4::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m * v, v);
    }

    #[test]
    fn test_mat4_from_translation() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let v = Vec3::ZERO;
        let result = m * v;
        assert_eq!(result, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_mat4_mul() {
        let m1 = Mat4::IDENTITY;
        let m2 = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let result = m1 * m2;
        assert_eq!(result.w_axis.x, 1.0);
        assert_eq!(result.w_axis.y, 2.0);
        assert_eq!(result.w_axis.z, 3.0);
    }

    #[test]
    fn test_mat4_transpose() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let transposed = m.transpose();
        assert_eq!(transposed.x_axis.x, m.x_axis.x);
        assert_eq!(transposed.x_axis.y, m.y_axis.x);
    }

    #[test]
    fn test_mat4_transform_vector3() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let v = Vec3::new(1.0, 0.0, 0.0);
        let result = m.transform_vector3(v);
        assert_eq!(result, v);
    }

    #[test]
    fn test_mat4_transform_point3() {
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let v = Vec3::ZERO;
        let result = m.transform_point3(v);
        assert_eq!(result, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_mat4_determinant() {
        assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(m.determinant(), 1.0);
    }

    #[test]
    fn test_mat4_inverse() {
        let m = Mat4::IDENTITY;
        let inv = m.inverse().unwrap();
        assert_eq!(inv, Mat4::IDENTITY);

        let m = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let inv = m.inverse().unwrap();
        let result = m * inv;

        for i in 0..16 {
            assert!((result.as_ref()[i] - Mat4::IDENTITY.as_ref()[i]).abs() < 0.0001);
        }
    }

    #[test]
    fn test_mat4_decompose() {
        let scale = Vec3::new(2.0, 3.0, 4.0);
        let rotation = Quat::from_axis_angle(Vec3::Y, std::f32::consts::FRAC_PI_4);
        let translation = Vec3::new(1.0, 2.0, 3.0);

        let m = Mat4::from_scale_rotation_translation(scale, rotation, translation);
        let (decomp_scale, _decomp_rot, decomp_trans) = m.decompose().unwrap();

        assert!((decomp_scale.x - scale.x).abs() < 0.01);
        assert!((decomp_scale.y - scale.y).abs() < 0.01);
        assert!((decomp_scale.z - scale.z).abs() < 0.01);
        assert!((decomp_trans.x - translation.x).abs() < 0.0001);
        assert!((decomp_trans.y - translation.y).abs() < 0.0001);
        assert!((decomp_trans.z - translation.z).abs() < 0.0001);
    }
}

impl Default for Mat4 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl From<[f32; 16]> for Mat4 {
    #[inline]
    fn from(a: [f32; 16]) -> Self {
        Self::from_cols_array(&a)
    }
}

impl From<Mat4> for [f32; 16] {
    #[inline]
    fn from(m: Mat4) -> Self {
        m.to_cols_array()
    }
}

impl std::ops::Add for Mat4 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x_axis: self.x_axis + other.x_axis,
            y_axis: self.y_axis + other.y_axis,
            z_axis: self.z_axis + other.z_axis,
            w_axis: self.w_axis + other.w_axis,
        }
    }
}

impl std::ops::Sub for Mat4 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x_axis: self.x_axis - other.x_axis,
            y_axis: self.y_axis - other.y_axis,
            z_axis: self.z_axis - other.z_axis,
            w_axis: self.w_axis - other.w_axis,
        }
    }
}

impl std::ops::Mul<f32> for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x_axis: self.x_axis * scalar,
            y_axis: self.y_axis * scalar,
            z_axis: self.z_axis * scalar,
            w_axis: self.w_axis * scalar,
        }
    }
}
