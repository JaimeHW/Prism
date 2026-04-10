//! Prism AOT build planning.
//!
//! This crate owns the whole-program planning stage that sits between
//! Python/module discovery and future native object generation.

mod error;
mod imports;
mod manifest;
mod planner;

pub use error::AotError;
pub use manifest::{BuildManifest, EntryManifest, InvocationManifest, ModuleManifest};
pub use planner::{
    BuildEntry, BuildOptions, BuildPlan, BuildPlanner, ModuleKind, PlannedEntry, PlannedModule,
};
