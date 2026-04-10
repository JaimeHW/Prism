//! Prism AOT build planning.
//!
//! This crate owns the whole-program planning stage that sits between
//! Python/module discovery and future native object generation.

mod bundle;
mod error;
mod imports;
mod link;
mod manifest;
mod planner;

pub use bundle::{
    CodeImage, ConstantImage, ExceptionTableImage, FrozenEntryImage, FrozenModuleBundle,
    FrozenModuleImage, LineTableImage,
};
pub use error::AotError;
pub use link::{
    FROZEN_BUNDLE_END_SYMBOL, FROZEN_BUNDLE_START_SYMBOL, LinkArtifactFormat,
    LinkableBundleArtifact,
};
pub use manifest::{BuildManifest, EntryManifest, InvocationManifest, ModuleManifest};
pub use planner::{
    BuildEntry, BuildOptions, BuildPlan, BuildPlanner, ModuleKind, PlannedEntry, PlannedModule,
};
