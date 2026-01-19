//! Math utility functions and constants

use std::f32::consts;

pub const PI: f32 = consts::PI;
pub const TAU: f32 = consts::TAU;
pub const FRAC_PI_2: f32 = consts::FRAC_PI_2;
pub const FRAC_PI_3: f32 = consts::FRAC_PI_3;
pub const FRAC_PI_4: f32 = consts::FRAC_PI_4;
pub const FRAC_PI_6: f32 = consts::FRAC_PI_6;
pub const FRAC_2_PI: f32 = consts::FRAC_2_PI;
pub const SQRT_2: f32 = consts::SQRT_2;
pub const E: f32 = consts::E;

#[inline]
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
pub fn clamp(value: f32, min: f32, max: f32) -> f32 {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

#[inline]
pub fn clamp01(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

#[inline]
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp01((x - edge0) / (edge1 - edge0));
    t * t * (3.0 - 2.0 * t)
}

pub fn degrees_to_radians(degrees: f32) -> f32 {
    degrees * PI / 180.0
}

pub fn radians_to_degrees(radians: f32) -> f32 {
    radians * 180.0 / PI
}

#[inline]
pub fn min(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}

#[inline]
pub fn max(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(-1.0), 0.0);
        assert_eq!(clamp01(2.0), 1.0);
    }

    #[test]
    fn test_min_max() {
        assert_eq!(min(5.0, 10.0), 5.0);
        assert_eq!(max(5.0, 10.0), 10.0);
    }
}
