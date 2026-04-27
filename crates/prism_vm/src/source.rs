//! Source compilation settings used by VM-owned imports and dynamic execution.

/// Optimization level for source compiled by the VM at runtime.
///
/// The VM keeps this as its own public configuration type so callers do not
/// need to depend on the compiler crate just to configure imported modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceOptimization {
    /// Preserve source structure and skip compiler optimizations.
    #[default]
    None,
    /// Enable inexpensive optimizations.
    Basic,
    /// Enable the full source optimization pipeline.
    Full,
}

impl SourceOptimization {
    #[inline]
    pub(crate) fn to_compiler(self) -> prism_compiler::OptimizationLevel {
        match self {
            Self::None => prism_compiler::OptimizationLevel::None,
            Self::Basic => prism_compiler::OptimizationLevel::Basic,
            Self::Full => prism_compiler::OptimizationLevel::Full,
        }
    }
}
