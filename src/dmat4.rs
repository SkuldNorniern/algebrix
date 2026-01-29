//! 4x4 column-major matrix in double precision (f64). Same API as Mat4; use [`as_mat4`](DMat4::as_mat4) to convert to float.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{DMat4, DVec3};
//! let t = DMat4::from_translation(DVec3::new(1.0, 2.0, 3.0));
//! let p = DVec3::ZERO;
//! let q = t.transform_point3(p);
//! assert!((q.x - 1.0).abs() < 1e-10);
//! assert!((q.y - 2.0).abs() < 1e-10);
//! ```
//!

use crate::Vec4;
use crate::{DQuat, DVec3, Mat4};

/// 4x4 column-major matrix in double precision (f64).
///
/// Same layout and operations as [`Mat4`]; use [`as_mat4`](DMat4::as_mat4) to convert to single precision
/// or [`from_mat4`](DMat4::from_mat4) to convert from it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DMat4 {
    pub x_axis: [f64; 4],
    pub y_axis: [f64; 4],
    pub z_axis: [f64; 4],
    pub w_axis: [f64; 4],
}

impl DMat4 {
    /// Identity matrix (no translation, no rotation).
    pub const IDENTITY: DMat4 = DMat4 {
        x_axis: [1.0, 0.0, 0.0, 0.0],
        y_axis: [0.0, 1.0, 0.0, 0.0],
        z_axis: [0.0, 0.0, 1.0, 0.0],
        w_axis: [0.0, 0.0, 0.0, 1.0],
    };

    /// Build from four column vectors (column-major order).
    pub const fn from_cols(x: [f64; 4], y: [f64; 4], z: [f64; 4], w: [f64; 4]) -> Self {
        Self {
            x_axis: x,
            y_axis: y,
            z_axis: z,
            w_axis: w,
        }
    }

    /// Translation matrix: identity with `translation` in the fourth column.
    #[inline]
    pub fn from_translation(translation: DVec3) -> Self {
        Self {
            x_axis: [1.0, 0.0, 0.0, 0.0],
            y_axis: [0.0, 1.0, 0.0, 0.0],
            z_axis: [0.0, 0.0, 1.0, 0.0],
            w_axis: [translation.x, translation.y, translation.z, 1.0],
        }
    }

    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(quat: DQuat) -> Self {
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
            x_axis: [1.0 - yy - zz, xy + wz, xz - wy, 0.0],
            y_axis: [xy - wz, 1.0 - xx - zz, yz + wx, 0.0],
            z_axis: [xz + wy, yz - wx, 1.0 - xx - yy, 0.0],
            w_axis: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Convert to single-precision [`Mat4`] (f32). Useful when passing to GPU or float APIs.
    #[inline]
    pub fn as_mat4(self) -> Mat4 {
        Mat4::from_cols(
            Vec4::new(
                self.x_axis[0] as f32,
                self.x_axis[1] as f32,
                self.x_axis[2] as f32,
                self.x_axis[3] as f32,
            ),
            Vec4::new(
                self.y_axis[0] as f32,
                self.y_axis[1] as f32,
                self.y_axis[2] as f32,
                self.y_axis[3] as f32,
            ),
            Vec4::new(
                self.z_axis[0] as f32,
                self.z_axis[1] as f32,
                self.z_axis[2] as f32,
                self.z_axis[3] as f32,
            ),
            Vec4::new(
                self.w_axis[0] as f32,
                self.w_axis[1] as f32,
                self.w_axis[2] as f32,
                self.w_axis[3] as f32,
            ),
        )
    }

    /// Build from a single-precision [`Mat4`]. Components are cast to f64.
    #[inline]
    pub fn from_mat4(m: Mat4) -> Self {
        let cols = m.to_cols_array();
        Self {
            x_axis: [
                cols[0] as f64,
                cols[1] as f64,
                cols[2] as f64,
                cols[3] as f64,
            ],
            y_axis: [
                cols[4] as f64,
                cols[5] as f64,
                cols[6] as f64,
                cols[7] as f64,
            ],
            z_axis: [
                cols[8] as f64,
                cols[9] as f64,
                cols[10] as f64,
                cols[11] as f64,
            ],
            w_axis: [
                cols[12] as f64,
                cols[13] as f64,
                cols[14] as f64,
                cols[15] as f64,
            ],
        }
    }

    /// Transform a 3D point (rotation/scale then translation). Same as `Mat4::transform_point3` but in f64.
    #[inline]
    pub fn transform_point3(self, v: DVec3) -> DVec3 {
        DVec3::new(
            v.x * self.x_axis[0] + v.y * self.y_axis[0] + v.z * self.z_axis[0] + self.w_axis[0],
            v.x * self.x_axis[1] + v.y * self.y_axis[1] + v.z * self.z_axis[1] + self.w_axis[1],
            v.x * self.x_axis[2] + v.y * self.y_axis[2] + v.z * self.z_axis[2] + self.w_axis[2],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vec3;

    #[test]
    fn test_dmat4_conversion() {
        let m32 = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let m64 = DMat4::from_mat4(m32);

        let back = m64.as_mat4();
        for i in 0..16 {
            assert!((back.as_ref()[i] - m32.as_ref()[i]).abs() < 1e-6);
        }
    }
}
