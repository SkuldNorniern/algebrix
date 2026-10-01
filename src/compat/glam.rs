//! `From`/`Into` conversions between algebrix and glam. Enable with the `glam-compatibility` feature.

use crate::{Mat2, Mat3, Mat4, Quat, Vec2, Vec3, Vec4};

impl From<glam::Vec2> for Vec2 {
    #[inline]
    fn from(g: glam::Vec2) -> Self {
        Self::new(g.x, g.y)
    }
}

impl From<Vec2> for glam::Vec2 {
    #[inline]
    fn from(v: Vec2) -> Self {
        Self::new(v.x, v.y)
    }
}

impl From<glam::Vec3> for Vec3 {
    #[inline]
    fn from(g: glam::Vec3) -> Self {
        Self::new(g.x, g.y, g.z)
    }
}

impl From<Vec3> for glam::Vec3 {
    #[inline]
    fn from(v: Vec3) -> Self {
        Self::new(v.x, v.y, v.z)
    }
}

impl From<glam::Vec4> for Vec4 {
    #[inline]
    fn from(g: glam::Vec4) -> Self {
        Self::new(g.x, g.y, g.z, g.w)
    }
}

impl From<Vec4> for glam::Vec4 {
    #[inline]
    fn from(v: Vec4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

impl From<glam::Mat2> for Mat2 {
    #[inline]
    fn from(g: glam::Mat2) -> Self {
        Self::from_cols(g.x_axis.into(), g.y_axis.into())
    }
}

impl From<Mat2> for glam::Mat2 {
    #[inline]
    fn from(m: Mat2) -> Self {
        Self::from_cols(m.x_axis.into(), m.y_axis.into())
    }
}

impl From<glam::Mat3> for Mat3 {
    #[inline]
    fn from(g: glam::Mat3) -> Self {
        Self::from_cols(g.x_axis.into(), g.y_axis.into(), g.z_axis.into())
    }
}

impl From<Mat3> for glam::Mat3 {
    #[inline]
    fn from(m: Mat3) -> Self {
        Self::from_cols(m.x_axis.into(), m.y_axis.into(), m.z_axis.into())
    }
}

impl From<glam::Mat4> for Mat4 {
    #[inline]
    fn from(g: glam::Mat4) -> Self {
        Self::from_cols(
            g.x_axis.into(),
            g.y_axis.into(),
            g.z_axis.into(),
            g.w_axis.into(),
        )
    }
}

impl From<Mat4> for glam::Mat4 {
    #[inline]
    fn from(m: Mat4) -> Self {
        Self::from_cols(
            m.x_axis.into(),
            m.y_axis.into(),
            m.z_axis.into(),
            m.w_axis.into(),
        )
    }
}

impl From<glam::Quat> for Quat {
    #[inline]
    fn from(g: glam::Quat) -> Self {
        Self::new(g.x, g.y, g.z, g.w)
    }
}

impl From<Quat> for glam::Quat {
    #[inline]
    fn from(q: Quat) -> Self {
        Self::from_xyzw(q.x, q.y, q.z, q.w)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Mat4, Quat, Vec2, Vec3, Vec4};

    #[test]
    fn roundtrip_vec2() {
        let a = Vec2::new(1.0, 2.0);
        let g: glam::Vec2 = a.into();
        let back: Vec2 = g.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_vec3() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let g: glam::Vec3 = a.into();
        let back: Vec3 = g.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_vec4() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let g: glam::Vec4 = a.into();
        let back: Vec4 = g.into();
        assert_eq!(a, back);
    }

    #[test]
    fn roundtrip_quat() {
        let a = Quat::new(0.0, 0.0, 0.0, 1.0);
        let g: glam::Quat = a.into();
        let back: Quat = g.into();
        assert_eq!(a.x, back.x);
        assert_eq!(a.y, back.y);
        assert_eq!(a.z, back.z);
        assert_eq!(a.w, back.w);
    }

    #[test]
    fn roundtrip_mat4() {
        let a = Mat4::IDENTITY;
        let g: glam::Mat4 = a.into();
        let back: Mat4 = g.into();
        assert_eq!(a.x_axis, back.x_axis);
        assert_eq!(a.y_axis, back.y_axis);
        assert_eq!(a.z_axis, back.z_axis);
        assert_eq!(a.w_axis, back.w_axis);
    }
}
