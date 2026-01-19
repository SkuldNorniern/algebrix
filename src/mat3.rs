//! 3x3 matrix implementation

use crate::Vec3;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3 {
    pub x_axis: Vec3,
    pub y_axis: Vec3,
    pub z_axis: Vec3,
}

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3 {
        x_axis: Vec3::X,
        y_axis: Vec3::Y,
        z_axis: Vec3::Z,
    };

    pub const fn new(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    pub const fn from_cols(x_axis: Vec3, y_axis: Vec3, z_axis: Vec3) -> Self {
        Self {
            x_axis,
            y_axis,
            z_axis,
        }
    }

    pub const fn from_diagonal(diagonal: Vec3) -> Self {
        Self {
            x_axis: Vec3::new(diagonal.x, 0.0, 0.0),
            y_axis: Vec3::new(0.0, diagonal.y, 0.0),
            z_axis: Vec3::new(0.0, 0.0, diagonal.z),
        }
    }

    #[inline]
    pub fn transpose(self) -> Self {
        Self {
            x_axis: Vec3::new(self.x_axis.x, self.y_axis.x, self.z_axis.x),
            y_axis: Vec3::new(self.x_axis.y, self.y_axis.y, self.z_axis.y),
            z_axis: Vec3::new(self.x_axis.z, self.y_axis.z, self.z_axis.z),
        }
    }

    #[inline]
    pub fn mul_vec3(self, other: Vec3) -> Vec3 {
        // Use mul_add for FMA optimization and better numerical stability
        Vec3::new(
            self.x_axis.x.mul_add(
                other.x,
                self.y_axis.x.mul_add(other.y, self.z_axis.x * other.z),
            ),
            self.x_axis.y.mul_add(
                other.x,
                self.y_axis.y.mul_add(other.y, self.z_axis.y * other.z),
            ),
            self.x_axis.z.mul_add(
                other.x,
                self.y_axis.z.mul_add(other.y, self.z_axis.z * other.z),
            ),
        )
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
