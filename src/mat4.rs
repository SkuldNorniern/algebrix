//! 4x4 matrix implementation

use crate::{Quat, Vec3, Vec4, simd};

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

    pub const fn new(x_axis: Vec4, y_axis: Vec4, z_axis: Vec4, w_axis: Vec4) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
            w_axis,
        }
    }

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
        let f = (center - eye).normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        let s_dot_eye = s.dot(eye);
        let u_dot_eye = u.dot(eye);
        let f_dot_eye = f.dot(eye);

        Self {
            x_axis: Vec4::new(s.x, u.x, -f.x, 0.0),
            y_axis: Vec4::new(s.y, u.y, -f.y, 0.0),
            z_axis: Vec4::new(s.z, u.z, -f.z, 0.0),
            w_axis: Vec4::new(-s_dot_eye, -u_dot_eye, f_dot_eye, 1.0),
        }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        simd::mat4_transpose(self)
    }

    #[inline]
    pub fn mul_vec4(self, other: Vec4) -> Vec4 {
        simd::mat4_mul_vec4(self, other)
    }

    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        // Scalar is fine for single vec3, but we can use mul_vec4 for consistency
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

    #[inline]
    pub fn mul_affine(self, other: Self) -> Self {
        simd::mat4_mul_mat4(self, other)
    }

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
    pub fn as_ref(&self) -> &[f32; 16] {
        unsafe { &*(self as *const Mat4 as *const [f32; 16]) }
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
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        simd::mat4_mul_mat4(self, other)
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
}
