//! Component reordering (swizzling): `.xy()`, `.xyz()`, `.xx()`, etc. on Vec2/3/4.
//!
//! Implemented via [`Vec2Swizzles`], [`Vec3Swizzles`], [`Vec4Swizzles`]. Use to build smaller vectors
//! or repeat components (e.g. `vec.xx()` for Vec2 with both components x).
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Vec3, Vec4, Vec3Swizzles, Vec4Swizzles};
//!
//! let v3 = Vec3::new(1.0, 2.0, 3.0);
//! let v2 = v3.xy();
//! assert_eq!(v2.x, 1.0);
//! assert_eq!(v2.y, 2.0);
//!
//! let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
//! let v3_from_4 = v4.xyz();
//! assert_eq!(v3_from_4.z, 3.0);
//! let repeated = v3.xxx();
//! assert_eq!(repeated.x, 1.0);
//! assert_eq!(repeated.y, 1.0);
//! assert_eq!(repeated.z, 1.0);
//! ```

use crate::{Vec2, Vec3, Vec4};

/// Vec2 component reordering.
pub trait Vec2Swizzles: Sized + Copy {
    fn x(self) -> f32;
    fn y(self) -> f32;

    #[inline(always)]
    fn xx(self) -> Vec2 {
        Vec2::new(self.x(), self.x())
    }

    #[inline(always)]
    fn xy(self) -> Vec2 {
        Vec2::new(self.x(), self.y())
    }

    #[inline(always)]
    fn yx(self) -> Vec2 {
        Vec2::new(self.y(), self.x())
    }

    #[inline(always)]
    fn yy(self) -> Vec2 {
        Vec2::new(self.y(), self.y())
    }

    #[inline(always)]
    fn xxx(self) -> Vec3 {
        Vec3::new(self.x(), self.x(), self.x())
    }

    #[inline(always)]
    fn xxy(self) -> Vec3 {
        Vec3::new(self.x(), self.x(), self.y())
    }

    #[inline(always)]
    fn xyx(self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.x())
    }

    #[inline(always)]
    fn xyy(self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.y())
    }

    #[inline(always)]
    fn yxx(self) -> Vec3 {
        Vec3::new(self.y(), self.x(), self.x())
    }

    #[inline(always)]
    fn yxy(self) -> Vec3 {
        Vec3::new(self.y(), self.x(), self.y())
    }

    #[inline(always)]
    fn yyx(self) -> Vec3 {
        Vec3::new(self.y(), self.y(), self.x())
    }

    #[inline(always)]
    fn yyy(self) -> Vec3 {
        Vec3::new(self.y(), self.y(), self.y())
    }
}

impl Vec2Swizzles for Vec2 {
    #[inline(always)]
    fn x(self) -> f32 {
        self.x
    }
    #[inline(always)]
    fn y(self) -> f32 {
        self.y
    }
}

/// Vec3 component reordering.
pub trait Vec3Swizzles: Sized + Copy {
    fn x(self) -> f32;
    fn y(self) -> f32;
    fn z(self) -> f32;

    #[inline(always)]
    fn xx(self) -> Vec2 {
        Vec2::new(self.x(), self.x())
    }
    #[inline(always)]
    fn xy(self) -> Vec2 {
        Vec2::new(self.x(), self.y())
    }
    #[inline(always)]
    fn xz(self) -> Vec2 {
        Vec2::new(self.x(), self.z())
    }
    #[inline(always)]
    fn yx(self) -> Vec2 {
        Vec2::new(self.y(), self.x())
    }
    #[inline(always)]
    fn yy(self) -> Vec2 {
        Vec2::new(self.y(), self.y())
    }
    #[inline(always)]
    fn yz(self) -> Vec2 {
        Vec2::new(self.y(), self.z())
    }
    #[inline(always)]
    fn zx(self) -> Vec2 {
        Vec2::new(self.z(), self.x())
    }
    #[inline(always)]
    fn zy(self) -> Vec2 {
        Vec2::new(self.z(), self.y())
    }
    #[inline(always)]
    fn zz(self) -> Vec2 {
        Vec2::new(self.z(), self.z())
    }

    #[inline(always)]
    fn xyz(self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.z())
    }
    #[inline(always)]
    fn xzy(self) -> Vec3 {
        Vec3::new(self.x(), self.z(), self.y())
    }
    #[inline(always)]
    fn yxz(self) -> Vec3 {
        Vec3::new(self.y(), self.x(), self.z())
    }
    #[inline(always)]
    fn yzx(self) -> Vec3 {
        Vec3::new(self.y(), self.z(), self.x())
    }
    #[inline(always)]
    fn zxy(self) -> Vec3 {
        Vec3::new(self.z(), self.x(), self.y())
    }
    #[inline(always)]
    fn zyx(self) -> Vec3 {
        Vec3::new(self.z(), self.y(), self.x())
    }
    #[inline(always)]
    fn xxx(self) -> Vec3 {
        Vec3::new(self.x(), self.x(), self.x())
    }
    #[inline(always)]
    fn yyy(self) -> Vec3 {
        Vec3::new(self.y(), self.y(), self.y())
    }
    #[inline(always)]
    fn zzz(self) -> Vec3 {
        Vec3::new(self.z(), self.z(), self.z())
    }
}

impl Vec3Swizzles for Vec3 {
    #[inline(always)]
    fn x(self) -> f32 {
        self.x
    }
    #[inline(always)]
    fn y(self) -> f32 {
        self.y
    }
    #[inline(always)]
    fn z(self) -> f32 {
        self.z
    }
}

/// Vec4 component reordering.
pub trait Vec4Swizzles: Sized + Copy {
    fn x(self) -> f32;
    fn y(self) -> f32;
    fn z(self) -> f32;
    fn w(self) -> f32;

    #[inline(always)]
    fn xx(self) -> Vec2 {
        Vec2::new(self.x(), self.x())
    }
    #[inline(always)]
    fn xy(self) -> Vec2 {
        Vec2::new(self.x(), self.y())
    }
    #[inline(always)]
    fn xz(self) -> Vec2 {
        Vec2::new(self.x(), self.z())
    }
    #[inline(always)]
    fn xw(self) -> Vec2 {
        Vec2::new(self.x(), self.w())
    }
    #[inline(always)]
    fn yx(self) -> Vec2 {
        Vec2::new(self.y(), self.x())
    }
    #[inline(always)]
    fn yy(self) -> Vec2 {
        Vec2::new(self.y(), self.y())
    }
    #[inline(always)]
    fn yz(self) -> Vec2 {
        Vec2::new(self.y(), self.z())
    }
    #[inline(always)]
    fn yw(self) -> Vec2 {
        Vec2::new(self.y(), self.w())
    }
    #[inline(always)]
    fn zx(self) -> Vec2 {
        Vec2::new(self.z(), self.x())
    }
    #[inline(always)]
    fn zy(self) -> Vec2 {
        Vec2::new(self.z(), self.y())
    }
    #[inline(always)]
    fn zz(self) -> Vec2 {
        Vec2::new(self.z(), self.z())
    }
    #[inline(always)]
    fn zw(self) -> Vec2 {
        Vec2::new(self.z(), self.w())
    }
    #[inline(always)]
    fn wx(self) -> Vec2 {
        Vec2::new(self.w(), self.x())
    }
    #[inline(always)]
    fn wy(self) -> Vec2 {
        Vec2::new(self.w(), self.y())
    }
    #[inline(always)]
    fn wz(self) -> Vec2 {
        Vec2::new(self.w(), self.z())
    }
    #[inline(always)]
    fn ww(self) -> Vec2 {
        Vec2::new(self.w(), self.w())
    }

    #[inline(always)]
    fn xyz(self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.z())
    }
    #[inline(always)]
    fn xyw(self) -> Vec3 {
        Vec3::new(self.x(), self.y(), self.w())
    }
    #[inline(always)]
    fn xzy(self) -> Vec3 {
        Vec3::new(self.x(), self.z(), self.y())
    }
    #[inline(always)]
    fn xzw(self) -> Vec3 {
        Vec3::new(self.x(), self.z(), self.w())
    }
    #[inline(always)]
    fn xwy(self) -> Vec3 {
        Vec3::new(self.x(), self.w(), self.y())
    }
    #[inline(always)]
    fn xwz(self) -> Vec3 {
        Vec3::new(self.x(), self.w(), self.z())
    }
    #[inline(always)]
    fn yxz(self) -> Vec3 {
        Vec3::new(self.y(), self.x(), self.z())
    }
    #[inline(always)]
    fn yxw(self) -> Vec3 {
        Vec3::new(self.y(), self.x(), self.w())
    }
    #[inline(always)]
    fn yzx(self) -> Vec3 {
        Vec3::new(self.y(), self.z(), self.x())
    }
    #[inline(always)]
    fn yzw(self) -> Vec3 {
        Vec3::new(self.y(), self.z(), self.w())
    }
    #[inline(always)]
    fn ywx(self) -> Vec3 {
        Vec3::new(self.y(), self.w(), self.x())
    }
    #[inline(always)]
    fn ywz(self) -> Vec3 {
        Vec3::new(self.y(), self.w(), self.z())
    }
    #[inline(always)]
    fn zxy(self) -> Vec3 {
        Vec3::new(self.z(), self.x(), self.y())
    }
    #[inline(always)]
    fn zxw(self) -> Vec3 {
        Vec3::new(self.z(), self.x(), self.w())
    }
    #[inline(always)]
    fn zyx(self) -> Vec3 {
        Vec3::new(self.z(), self.y(), self.x())
    }
    #[inline(always)]
    fn zyw(self) -> Vec3 {
        Vec3::new(self.z(), self.y(), self.w())
    }
    #[inline(always)]
    fn zwx(self) -> Vec3 {
        Vec3::new(self.z(), self.w(), self.x())
    }
    #[inline(always)]
    fn zwy(self) -> Vec3 {
        Vec3::new(self.z(), self.w(), self.y())
    }
    #[inline(always)]
    fn wxy(self) -> Vec3 {
        Vec3::new(self.w(), self.x(), self.y())
    }
    #[inline(always)]
    fn wxz(self) -> Vec3 {
        Vec3::new(self.w(), self.x(), self.z())
    }
    #[inline(always)]
    fn wyx(self) -> Vec3 {
        Vec3::new(self.w(), self.y(), self.x())
    }
    #[inline(always)]
    fn wyz(self) -> Vec3 {
        Vec3::new(self.w(), self.y(), self.z())
    }
    #[inline(always)]
    fn wzx(self) -> Vec3 {
        Vec3::new(self.w(), self.z(), self.x())
    }
    #[inline(always)]
    fn wzy(self) -> Vec3 {
        Vec3::new(self.w(), self.z(), self.y())
    }

    #[inline(always)]
    fn xyzw(self) -> Vec4 {
        Vec4::new(self.x(), self.y(), self.z(), self.w())
    }
    #[inline(always)]
    fn xywz(self) -> Vec4 {
        Vec4::new(self.x(), self.y(), self.w(), self.z())
    }
    #[inline(always)]
    fn xzyw(self) -> Vec4 {
        Vec4::new(self.x(), self.z(), self.y(), self.w())
    }
    #[inline(always)]
    fn xzwy(self) -> Vec4 {
        Vec4::new(self.x(), self.z(), self.w(), self.y())
    }
    #[inline(always)]
    fn xwyz(self) -> Vec4 {
        Vec4::new(self.x(), self.w(), self.y(), self.z())
    }
    #[inline(always)]
    fn xwzy(self) -> Vec4 {
        Vec4::new(self.x(), self.w(), self.z(), self.y())
    }
    #[inline(always)]
    fn yxzw(self) -> Vec4 {
        Vec4::new(self.y(), self.x(), self.z(), self.w())
    }
    #[inline(always)]
    fn yxwz(self) -> Vec4 {
        Vec4::new(self.y(), self.x(), self.w(), self.z())
    }
    #[inline(always)]
    fn yzxw(self) -> Vec4 {
        Vec4::new(self.y(), self.z(), self.x(), self.w())
    }
    #[inline(always)]
    fn yzwx(self) -> Vec4 {
        Vec4::new(self.y(), self.z(), self.w(), self.x())
    }
    #[inline(always)]
    fn ywxz(self) -> Vec4 {
        Vec4::new(self.y(), self.w(), self.x(), self.z())
    }
    #[inline(always)]
    fn ywzx(self) -> Vec4 {
        Vec4::new(self.y(), self.w(), self.z(), self.x())
    }
    #[inline(always)]
    fn zxyw(self) -> Vec4 {
        Vec4::new(self.z(), self.x(), self.y(), self.w())
    }
    #[inline(always)]
    fn zxwy(self) -> Vec4 {
        Vec4::new(self.z(), self.x(), self.w(), self.y())
    }
    #[inline(always)]
    fn zyxw(self) -> Vec4 {
        Vec4::new(self.z(), self.y(), self.x(), self.w())
    }
    #[inline(always)]
    fn zywx(self) -> Vec4 {
        Vec4::new(self.z(), self.y(), self.w(), self.x())
    }
    #[inline(always)]
    fn zwxy(self) -> Vec4 {
        Vec4::new(self.z(), self.w(), self.x(), self.y())
    }
    #[inline(always)]
    fn zwyx(self) -> Vec4 {
        Vec4::new(self.z(), self.w(), self.y(), self.x())
    }
    #[inline(always)]
    fn wxyz(self) -> Vec4 {
        Vec4::new(self.w(), self.x(), self.y(), self.z())
    }
    #[inline(always)]
    fn wxzy(self) -> Vec4 {
        Vec4::new(self.w(), self.x(), self.z(), self.y())
    }
    #[inline(always)]
    fn wyxz(self) -> Vec4 {
        Vec4::new(self.w(), self.y(), self.x(), self.z())
    }
    #[inline(always)]
    fn wyzx(self) -> Vec4 {
        Vec4::new(self.w(), self.y(), self.z(), self.x())
    }
    #[inline(always)]
    fn wzxy(self) -> Vec4 {
        Vec4::new(self.w(), self.z(), self.x(), self.y())
    }
    #[inline(always)]
    fn wzyx(self) -> Vec4 {
        Vec4::new(self.w(), self.z(), self.y(), self.x())
    }
}

impl Vec4Swizzles for Vec4 {
    #[inline(always)]
    fn x(self) -> f32 {
        self.x
    }
    #[inline(always)]
    fn y(self) -> f32 {
        self.y
    }
    #[inline(always)]
    fn z(self) -> f32 {
        self.z
    }
    #[inline(always)]
    fn w(self) -> f32 {
        self.w
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec2_swizzles() {
        let v = Vec2::new(1.0, 2.0);
        assert_eq!(v.xy(), Vec2::new(1.0, 2.0));
        assert_eq!(v.yx(), Vec2::new(2.0, 1.0));
        assert_eq!(v.xx(), Vec2::new(1.0, 1.0));
    }

    #[test]
    fn test_vec3_swizzles() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.xyz(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.zyx(), Vec3::new(3.0, 2.0, 1.0));
        assert_eq!(v.xy(), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn test_vec4_swizzles() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.xyz(), Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(v.wzyx(), Vec4::new(4.0, 3.0, 2.0, 1.0));
        assert_eq!(v.zw(), Vec2::new(3.0, 4.0));
    }
}
