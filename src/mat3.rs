//! 3x3 column-major matrix: 3D rotation, scale, or combined linear transform.
//!
//! Multiply a Vec3 on the right: `matrix * point`. Use [`from_quat`](Mat3::from_quat),
//! [`from_axis_angle`](Mat3::from_axis_angle), or [`from_rotation_x/y/z`](Mat3::from_rotation_z) for rotation.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Mat3, Vec3};
//!
//! let rot = Mat3::from_rotation_z(std::f32::consts::FRAC_PI_2);
//! let x = Vec3::X;
//! let y = rot * x;
//! assert!((y - Vec3::Y).length() < 1e-5);
//!
//! let scale = Mat3::from_nonuniform_scale(Vec3::new(2.0, 3.0, 1.0));
//! let p = Vec3::new(1.0, 1.0, 0.0);
//! assert_eq!(scale * p, Vec3::new(2.0, 3.0, 0.0));
//! ```

use crate::Vec3;

/// 3x3 column-major matrix for 3D rotation, scale, or combined linear transform.
///
/// Multiply a Vec3 on the right: `matrix * point`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub x_axis: Vec3,
    pub y_axis: Vec3,
    pub z_axis: Vec3,
}

impl Mat3 {
    /// Zero matrix (all elements 0).
    pub const ZERO: Mat3 = Mat3 {
        x_axis: Vec3::ZERO,
        y_axis: Vec3::ZERO,
        z_axis: Vec3::ZERO,
    };

    /// Identity matrix (no rotation, no scale).
    pub const IDENTITY: Mat3 = Mat3 {
        x_axis: Vec3::X,
        y_axis: Vec3::Y,
        z_axis: Vec3::Z,
    };

    /// Build from three column vectors.
    #[inline]
    pub const fn new(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Same as [`new`](Mat3::new); build from columns.
    #[inline]
    pub const fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    /// Diagonal matrix with (diagonal.x, diagonal.y, diagonal.z) on the diagonal.
    #[inline]
    pub fn from_diagonal(diagonal: Vec3) -> Self {
        Self {
            x_axis: Vec3::new(diagonal.x, 0.0, 0.0),
            y_axis: Vec3::new(0.0, diagonal.y, 0.0),
            z_axis: Vec3::new(0.0, 0.0, diagonal.z),
        }
    }

    /// Diagonal matrix with `value` on the diagonal (uniform scale).
    pub const fn from_diagonal_value(value: f32) -> Self {
        Self {
            x_axis: Vec3::new(value, 0.0, 0.0),
            y_axis: Vec3::new(0.0, value, 0.0),
            z_axis: Vec3::new(0.0, 0.0, value),
        }
    }

    /// Column at `index` (0, 1, or 2). Panics if `index > 2`.
    pub fn col(&self, index: usize) -> Vec3 {
        match index {
            0 => self.x_axis,
            1 => self.y_axis,
            2 => self.z_axis,
            _ => panic!("Column index out of bounds: {}", index),
        }
    }

    /// Transpose: swap rows and columns.
    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            x_axis: Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            y_axis: Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            z_axis: Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        }
    }

    /// Multiply matrix by a 3D vector.
    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        Vec3::new(
            self.x_axis.x * other.x + self.y_axis.x * other.y + self.z_axis.x * other.z,
            self.x_axis.y * other.x + self.y_axis.y * other.y + self.z_axis.y * other.z,
            self.x_axis.z * other.x + self.y_axis.z * other.y + self.z_axis.z * other.z,
        )
    }

    /// Determinant (signed volume scale). Zero if the matrix is singular.
    #[inline]
    pub fn determinant(&self) -> f32 {
        self.x_axis.x * (self.y_axis.y * self.z_axis.z - self.y_axis.z * self.z_axis.y)
            - self.y_axis.x * (self.x_axis.y * self.z_axis.z - self.x_axis.z * self.z_axis.y)
            + self.z_axis.x * (self.x_axis.y * self.y_axis.z - self.x_axis.z * self.y_axis.y)
    }

    /// Trace: sum of diagonal elements (x_axis.x + y_axis.y + z_axis.z).
    #[inline(always)]
    pub fn trace(&self) -> f32 {
        self.x_axis.x + self.y_axis.y + self.z_axis.z
    }

    /// Inverse matrix. Returns `None` if determinant is zero (singular).
    #[inline]
    pub fn inverse(&self) -> Option<Self> {
        let det = self.determinant();
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv_det = det.recip();

        let c00 = self.y_axis.y * self.z_axis.z - self.y_axis.z * self.z_axis.y;
        let c01 = self.x_axis.z * self.z_axis.y - self.x_axis.y * self.z_axis.z;
        let c02 = self.x_axis.y * self.y_axis.z - self.x_axis.z * self.y_axis.y;

        let c10 = self.y_axis.z * self.z_axis.x - self.y_axis.x * self.z_axis.z;
        let c11 = self.x_axis.x * self.z_axis.z - self.x_axis.z * self.z_axis.x;
        let c12 = self.x_axis.z * self.y_axis.x - self.x_axis.x * self.y_axis.z;

        let c20 = self.y_axis.x * self.z_axis.y - self.y_axis.y * self.z_axis.x;
        let c21 = self.x_axis.y * self.z_axis.x - self.x_axis.x * self.z_axis.y;
        let c22 = self.x_axis.x * self.y_axis.y - self.x_axis.y * self.y_axis.x;

        Some(Self {
            x_axis: Vec3::new(c00 * inv_det, c01 * inv_det, c02 * inv_det),
            y_axis: Vec3::new(c10 * inv_det, c11 * inv_det, c12 * inv_det),
            z_axis: Vec3::new(c20 * inv_det, c21 * inv_det, c22 * inv_det),
        })
    }

    /// Uniform scale matrix (scale along all axes).
    #[inline]
    pub fn from_scale(scale: f32) -> Self {
        Self::from_diagonal(Vec3::splat(scale))
    }

    /// Non-uniform scale matrix (scale.x, scale.y, scale.z per axis).
    #[inline]
    pub fn from_nonuniform_scale(scale: Vec3) -> Self {
        Self::from_diagonal(scale)
    }

    /// Rotation around the X axis by `angle` radians (right-handed).
    #[inline]
    pub fn from_rotation_x(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            x_axis: Vec3::X,
            y_axis: Vec3::new(0.0, cos, sin),
            z_axis: Vec3::new(0.0, -sin, cos),
        }
    }

    /// Rotation around the Y axis by `angle` radians (right-handed).
    #[inline]
    pub fn from_rotation_y(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            x_axis: Vec3::new(cos, 0.0, -sin),
            y_axis: Vec3::Y,
            z_axis: Vec3::new(sin, 0.0, cos),
        }
    }

    /// Rotation around the Z axis by `angle` radians (right-handed).
    #[inline]
    pub fn from_rotation_z(angle: f32) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            x_axis: Vec3::new(cos, sin, 0.0),
            y_axis: Vec3::new(-sin, cos, 0.0),
            z_axis: Vec3::Z,
        }
    }

    /// Rotation around `axis` (normalized) by `angle` radians (right-handed).
    #[inline]
    pub fn from_axis_angle(axis: Vec3, angle: f32) -> Self {
        let axis = axis.normalize();
        let (sin, cos) = angle.sin_cos();
        let one_minus_cos = 1.0 - cos;

        let x = axis.x;
        let y = axis.y;
        let z = axis.z;

        Self {
            x_axis: Vec3::new(
                cos + x * x * one_minus_cos,
                x * y * one_minus_cos + z * sin,
                x * z * one_minus_cos - y * sin,
            ),
            y_axis: Vec3::new(
                y * x * one_minus_cos - z * sin,
                cos + y * y * one_minus_cos,
                y * z * one_minus_cos + x * sin,
            ),
            z_axis: Vec3::new(
                z * x * one_minus_cos + y * sin,
                z * y * one_minus_cos - x * sin,
                cos + z * z * one_minus_cos,
            ),
        }
    }

    /// Rotation matrix from a unit quaternion.
    #[inline]
    pub fn from_quat(quat: crate::Quat) -> Self {
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
            x_axis: Vec3::new(1.0 - yy - zz, xy + wz, xz - wy),
            y_axis: Vec3::new(xy - wz, 1.0 - xx - zz, yz + wx),
            z_axis: Vec3::new(xz + wy, yz - wx, 1.0 - xx - yy),
        }
    }

    /// Build from a 9-element array in column-major order.
    #[inline]
    pub fn from_cols_array(m: &[f32; 9]) -> Self {
        Self {
            x_axis: Vec3::new(m[0], m[1], m[2]),
            y_axis: Vec3::new(m[3], m[4], m[5]),
            z_axis: Vec3::new(m[6], m[7], m[8]),
        }
    }

    /// Copy into a 9-element array [col0, col1, col2] in column-major order.
    #[inline]
    pub fn to_cols_array(self) -> [f32; 9] {
        [
            self.x_axis.x,
            self.x_axis.y,
            self.x_axis.z,
            self.y_axis.x,
            self.y_axis.y,
            self.y_axis.z,
            self.z_axis.x,
            self.z_axis.y,
            self.z_axis.z,
        ]
    }
}

impl std::ops::Mul for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x_axis: self.mul_vec3(other.x_axis),
            y_axis: self.mul_vec3(other.y_axis),
            z_axis: self.mul_vec3(other.z_axis),
        }
    }
}

impl std::ops::Mul<Vec3> for Mat3 {
    type Output = Vec3;
    #[inline]
    fn mul(self, other: Vec3) -> Vec3 {
        self.mul_vec3(other)
    }
}

impl std::ops::Mul<f32> for Mat3 {
    type Output = Self;
    #[inline]
    fn mul(self, scalar: f32) -> Self {
        Self {
            x_axis: self.x_axis * scalar,
            y_axis: self.y_axis * scalar,
            z_axis: self.z_axis * scalar,
        }
    }
}

impl std::ops::Mul<Mat3> for f32 {
    type Output = Mat3;
    #[inline]
    fn mul(self, matrix: Mat3) -> Mat3 {
        Mat3 {
            x_axis: matrix.x_axis * self,
            y_axis: matrix.y_axis * self,
            z_axis: matrix.z_axis * self,
        }
    }
}

impl std::ops::Div<f32> for Mat3 {
    type Output = Self;
    #[inline]
    fn div(self, scalar: f32) -> Self {
        let inv = 1.0 / scalar;
        Self {
            x_axis: self.x_axis * inv,
            y_axis: self.y_axis * inv,
            z_axis: self.z_axis * inv,
        }
    }
}

impl std::ops::Add for Mat3 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x_axis: self.x_axis + other.x_axis,
            y_axis: self.y_axis + other.y_axis,
            z_axis: self.z_axis + other.z_axis,
        }
    }
}

impl std::ops::Sub for Mat3 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x_axis: self.x_axis - other.x_axis,
            y_axis: self.y_axis - other.y_axis,
            z_axis: self.z_axis - other.z_axis,
        }
    }
}

impl std::ops::AddAssign for Mat3 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.x_axis += other.x_axis;
        self.y_axis += other.y_axis;
        self.z_axis += other.z_axis;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mat3_identity() {
        let m = Mat3::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m * v, v);
    }

    #[test]
    fn test_mat3_mul_vec3() {
        let m = Mat3::IDENTITY;
        let v = Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(m.mul_vec3(v), v);
    }

    #[test]
    fn test_mat3_transpose() {
        let m = Mat3::new(
            Vec3::new(1.0, 2.0, 3.0),
            Vec3::new(4.0, 5.0, 6.0),
            Vec3::new(7.0, 8.0, 9.0),
        );
        let transposed = m.transpose();
        assert_eq!(transposed.x_axis.x, m.x_axis.x);
        assert_eq!(transposed.x_axis.y, m.y_axis.x);
        assert_eq!(transposed.y_axis.x, m.x_axis.y);
    }
}

impl Default for Mat3 {
    #[inline]
    fn default() -> Self {
        Self::IDENTITY
    }
}
