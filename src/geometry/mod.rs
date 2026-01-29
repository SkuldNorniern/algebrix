//! Planes, rays, AABBs, and frustums for culling and spatial queries.
//!
//! # Example
//!
//! ```rust
//! use algebrix::{Plane, Ray, Vec3};
//!
//! let plane = Plane::from_normal_point(Vec3::NEG_Z, Vec3::new(0.0, 0.0, 5.0));
//! let ray = Ray::new(Vec3::ZERO, Vec3::Z);
//! let t = ray.intersect_plane(&plane);
//! assert!(t.is_some());
//! assert!((t.unwrap() - 5.0).abs() < 1e-5);
//!
//! let aabb = algebrix::Aabb3::new(Vec3::ZERO, Vec3::ONE);
//! assert!(aabb.contains(Vec3::new(0.5, 0.5, 0.5)));
//! ```

pub mod plane;
pub mod ray;
pub mod aabb;
pub mod frustum;

pub use plane::Plane;
pub use ray::Ray;
pub use aabb::{Aabb2, Aabb3};
pub use frustum::Frustum;
