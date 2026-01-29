//! Optional interop with glam and nalgebra.
//!
//! Enable with the `glam-compatibility` or `nalgebra-compatibility` feature in Cargo.toml.
//! Then use `From`/`Into` to convert between algebrix types and the other crate's types.
//!
//! # Example (with glam-compatibility)
//!
//! ```rust,ignore
//! use algebrix::Vec3;
//! use glam::Vec3 as GlamVec3;
//! let v = Vec3::new(1.0, 2.0, 3.0);
//! let g: GlamVec3 = v.into();
//! let back: Vec3 = g.into();
//! ```

#[cfg(feature = "glam-compatibility")]
pub mod glam;

#[cfg(feature = "nalgebra-compatibility")]
pub mod nalgebra;
