//! JIT runtime integration.
//!
//! This module exposes the executable code cache, entry/exit stubs, profiling
//! data, and speculation feedback used by the Prism JIT at execution time.
//! Background compilation is orchestrated by the VM's production compilation
//! queue so the runtime surface here only exports components that are actually
//! wired into execution.

pub mod code_cache;
pub mod entry_stub;
pub mod profile_collector;
pub mod profile_data;
pub mod type_feedback;

#[cfg(test)]
mod profile_data_tests;

pub use code_cache::{CodeCache, CompiledEntry, DeoptSite, ReturnAbi};
pub use entry_stub::{EntryStub, ExitReason};
pub use profile_data::{
    AtomicBranchCounter, BranchProfile, CallProfile, CallTarget, ProfileData, ProfileError,
    TypeProfile, TypeProfileEntry,
};
pub use type_feedback::{
    CallSpeculation, GenericReason, ObservedType, OracleConfig, SpeculationDecision,
    TypeFeedbackOracle, TypeStability,
};

// =============================================================================
// Runtime Configuration
// =============================================================================

/// Configuration for the JIT runtime.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum size of code cache in bytes.
    pub max_code_size: usize,
    /// Number of background compiler threads requested by the embedding VM.
    pub compiler_threads: usize,
    /// Compilation tier-up threshold (execution count).
    pub tier_up_threshold: u32,
    /// Enable OSR for hot loops.
    pub enable_osr: bool,
    /// Enable speculative optimizations.
    pub enable_speculation: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_code_size: 64 * 1024 * 1024, // 64MB
            compiler_threads: 1,
            tier_up_threshold: 1000,
            enable_osr: true,
            enable_speculation: true,
        }
    }
}

impl RuntimeConfig {
    /// Create a config for testing (smaller limits, synchronous compilation).
    pub fn for_testing() -> Self {
        Self {
            max_code_size: 1024 * 1024, // 1MB
            compiler_threads: 0,
            tier_up_threshold: 10,
            enable_osr: false,
            enable_speculation: false,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
