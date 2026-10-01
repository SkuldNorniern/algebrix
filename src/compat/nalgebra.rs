//! `From`/`Into` conversions between algebrix and nalgebra. Enable with the `nalgebra-compatibility` feature.

use crate::{Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};
use nalgebra::{Matrix2, Matrix3, Matrix4, Quaternion, Vector2, Vector3, Vector4};

impl From<Vector2<f32>> for Vec2 {
    #[inline]
    fn from(v: Vector2<f32>) -> Self {
        Self::new(v.x, v.y)
    }
}

impl From<Vec2> for Vector2<f32> {
    #[inline]
    fn from(v: Vec2) -> Self {
        Self::new(v.x, v.y)
    }
}

impl From<Vector3<f32>> for Vec3 {
    #[inline]
    fn from(v: Vector3<f32>) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl From<Vec3> for Vector3<f32> {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl From<Vector4<f32>> for Vec4 {
    #[inline]
    fn from(v: Vector4<f32>) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl From<Vec4> for Vector4<f32> {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl From<Matrix2<f32>> for Mat2 {
    #[inline]
    fn from(m: Matrix2<f32>) -> Self {
        Self::from_cols(
            Vec2::new(m[(0, 0)], m[(1, 0)]),
            Vec2::new(m[(0, 1)], m[(1, 1)]),
        )
    }
}

impl From<Mat2> for Matrix2<f32> {
    #[inline]
    fn from(m: Mat2) -> Self {
        Self::from_column_slice(&[m.x_axis.x, m.x_axis.y, m.y_axis.x, m.y_axis.y])
    }
}

impl From<Matrix3<f32>> for Mat3 {
    #[inline]
    fn from(m: Matrix3<f32>) -> Self {
        Self::from_cols(
            Vec3::new(m[(0, 0)], m[(1, 0)], m[(2, 0)]),
            Vec3::new(m[(0, 1)], m[(1, 1)], m[(2, 1)]),
            Vec3::new(m[(0, 2)], m[(1, 2)], m[(2, 2)]),
        )
    }
}

impl From<Mat3> for Matrix3<f32> {
    #[inline]
    fn from(m: Mat3) -> Self {
        Self::from_column_slice(&[
            m.x_axis.x, m.x_axis.y, m.x_axis.z, m.y_axis.x, m.y_axis.y, m.y_axis.z, m.z_axis.x,
            m.z_axis.y, m.z_axis.z,
        ])
    }
}

impl From<Matrix4<f32>> for Mat4 {
    #[inline]
    fn from(m: Matrix4<f32>) -> Self {
        Self::from_cols(
            Vec4::new(m[(0, 0)], m[(1, 0)], m[(2, 0)], m[(3, 0)]),
            Vec4::new(m[(0, 1)], m[(1, 1)], m[(2, 1)], m[(3, 1)]),
            Vec4::new(m[(0, 2)], m[(1, 2)], m[(2, 2)], m[(3, 2)]),
            Vec4::new(m[(0, 3)], m[(1, 3)], m[(2, 3)], m[(3, 3)]),
        )
    }
}

impl From<Mat4> for Matrix4<f32> {
    #[inline]
    fn from(m: Mat4) -> Self {
        Self::from_column_slice(&[
            m.x_axis.x, m.x_axis.y, m.x_axis.z, m.x_axis.w, m.y_axis.x, m.y_axis.y, m.y_axis.z,
            m.y_axis.w, m.z_axis.x, m.z_axis.y, m.z_axis.z, m.z_axis.w, m.w_axis.x, m.w_axis.y,
            m.w_axis.z, m.w_axis.w,
        ])
    }
}

impl From<Quaternion<f32>> for Quat {
    #[inline]
    fn from(q: Quaternion<f32>) -> Self {
        Self::new(q.i, q.j, q.k, q.w)
    }
}

impl From<Quat> for Quaternion<f32> {
    #[inline]
    fn from(q: Quat) -> Self {
        Self::new(q.w, q.x, q.y, q.z)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Mat4, Quat, Vec2, Vec3, Vec4};
    use nalgebra::{Matrix4, Quaternion, Vector2, Vector3, Vector4};

    #[test]
    fn roundtrip_vec2() {
        let a = Vec2::new(1.0, 2.0);
        let n: Vector2<f32> = a.into();
        let back: Vec2 = n.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_vec3() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let n: Vector3<f32> = a.into();
        let back: Vec3 = n.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_vec4() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let n: Vector4<f32> = a.into();
        let back: Vec4 = n.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_quat() {
        let a = Quat::IDENTITY;
        let n: Quaternion<f32> = a.into();
        let back: Quat = n.into();
        assert_eq!(a.x, back.x);
        assert_eq!(a.y, back.y);
        assert_eq!(a.z, back.z);
        assert_eq!(a.w, back.w);
    }

    #[test]
    fn roundtrip_mat4() {
        let a = Mat4::IDENTITY;
        let n: Matrix4<f32> = a.into();
        let back: Mat4 = n.into();
        assert_eq!(a.x_axis, back.x_axis);
        assert_eq!(a.y_axis, back.y_axis);
        assert_eq!(a.z_axis, back.z_axis);
        assert_eq!(a.w_axis, back.w_axis);
    }
}
