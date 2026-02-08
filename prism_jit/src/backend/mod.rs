//! Code generation backends for JIT compilation.
//!
//! This module provides architecture-specific code generation:
//! - `x64`: Intel/AMD 64-bit (default, fully implemented)
//! - `arm64`: ARM AArch64 (complete infrastructure)

pub mod x64;

#[cfg(any(target_arch = "aarch64", feature = "arm64"))]
pub mod arm64;

// Re-export x64 as the default on x86_64
#[cfg(target_arch = "x86_64")]
pub use x64::*;
