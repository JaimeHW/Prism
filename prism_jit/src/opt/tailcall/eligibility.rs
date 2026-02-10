//! Tail Call Eligibility Analysis
//!
//! Determines whether a tail call can actually be optimized based on:
//! - Frame size compatibility
//! - ABI compatibility  
//! - No escaping local references

use super::detection::TailCallInfo;

// =============================================================================
// Calling Convention
// =============================================================================

/// Calling convention for compatibility checking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallingConvention {
    /// Standard Python calling convention.
    Python,
    /// System V AMD64 ABI.
    SysV64,
    /// Microsoft x64 ABI.
    Win64,
    /// Unknown or custom.
    Unknown,
}

impl Default for CallingConvention {
    fn default() -> Self {
        CallingConvention::Python
    }
}

// =============================================================================
// Frame Info
// =============================================================================

/// Information about a function's stack frame.
#[derive(Debug, Clone, Default)]
pub struct FrameInfo {
    /// Stack frame size in bytes.
    pub size: usize,
    /// Number of locals.
    pub local_count: usize,
    /// Calling convention.
    pub convention: CallingConvention,
    /// Whether frame has escaping pointers.
    pub has_escaping_refs: bool,
}

impl FrameInfo {
    /// Create a new frame info.
    pub fn new(size: usize, locals: usize) -> Self {
        Self {
            size,
            local_count: locals,
            convention: CallingConvention::Python,
            has_escaping_refs: false,
        }
    }
}

// =============================================================================
// Eligibility Result
// =============================================================================

/// Result of eligibility analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eligibility {
    /// Eligible for optimization.
    Eligible,
    /// Not in tail position.
    NotTailPosition,
    /// Frame too large for reuse.
    FrameTooLarge,
    /// ABI mismatch.
    AbiMismatch,
    /// Has escaping local references.
    EscapingLocals,
    /// Requires cleanup code.
    RequiresCleanup,
    /// Unknown callee.
    UnknownCallee,
}

impl Eligibility {
    /// Check if eligible for optimization.
    pub fn is_eligible(self) -> bool {
        self == Eligibility::Eligible
    }

    /// Get a description of this status.
    pub fn description(self) -> &'static str {
        match self {
            Eligibility::Eligible => "eligible for optimization",
            Eligibility::NotTailPosition => "not in tail position",
            Eligibility::FrameTooLarge => "callee frame too large",
            Eligibility::AbiMismatch => "ABI mismatch",
            Eligibility::EscapingLocals => "escaping local references",
            Eligibility::RequiresCleanup => "requires cleanup",
            Eligibility::UnknownCallee => "unknown callee",
        }
    }
}

// =============================================================================
// Eligibility Config
// =============================================================================

/// Configuration for eligibility analysis.
#[derive(Debug, Clone)]
pub struct EligibilityConfig {
    /// Maximum frame size growth allowed.
    pub max_frame_growth: usize,
    /// Allow cross-ABI optimization.
    pub allow_cross_abi: bool,
}

impl Default for EligibilityConfig {
    fn default() -> Self {
        Self {
            max_frame_growth: 256,
            allow_cross_abi: false,
        }
    }
}

// =============================================================================
// Eligibility Analyzer
// =============================================================================

/// Analyzes tail call eligibility.
#[derive(Debug)]
pub struct EligibilityAnalyzer {
    /// Caller frame info.
    caller_frame: FrameInfo,
    /// Configuration.
    config: EligibilityConfig,
}

impl EligibilityAnalyzer {
    /// Create a new analyzer.
    pub fn new(caller_frame: FrameInfo) -> Self {
        Self {
            caller_frame,
            config: EligibilityConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(caller_frame: FrameInfo, config: EligibilityConfig) -> Self {
        Self {
            caller_frame,
            config,
        }
    }

    /// Analyze a tail call's eligibility.
    pub fn analyze(&self, info: &TailCallInfo, callee_frame: Option<&FrameInfo>) -> Eligibility {
        // Must be in tail position
        if !info.status.is_optimizable() {
            return Eligibility::NotTailPosition;
        }

        // Check caller frame for escaping refs
        if self.caller_frame.has_escaping_refs {
            return Eligibility::EscapingLocals;
        }

        // If we have callee info, check compatibility
        if let Some(callee) = callee_frame {
            // Check ABI compatibility
            if !self.config.allow_cross_abi && self.caller_frame.convention != callee.convention {
                return Eligibility::AbiMismatch;
            }

            // Check frame size
            let growth = callee.size.saturating_sub(self.caller_frame.size);
            if growth > self.config.max_frame_growth {
                return Eligibility::FrameTooLarge;
            }
        }

        Eligibility::Eligible
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Analyze eligibility of tail calls.
pub fn analyze_batch(calls: &[TailCallInfo], caller_frame: &FrameInfo) -> Vec<Eligibility> {
    let analyzer = EligibilityAnalyzer::new(caller_frame.clone());
    calls
        .iter()
        .map(|info| analyzer.analyze(info, None))
        .collect()
}

/// Filter to only eligible tail calls.
pub fn filter_eligible(calls: &[TailCallInfo], caller_frame: &FrameInfo) -> Vec<TailCallInfo> {
    let analyzer = EligibilityAnalyzer::new(caller_frame.clone());
    calls
        .iter()
        .filter(|info| analyzer.analyze(info, None).is_eligible())
        .cloned()
        .collect()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::node::NodeId;
    use crate::opt::tailcall::detection::TailCallStatus;

    fn make_tail_info() -> TailCallInfo {
        TailCallInfo {
            call_node: NodeId::new(1),
            is_self_call: false,
            status: TailCallStatus::TailPosition,
            return_node: None,
            arg_count: 0,
        }
    }

    // =========================================================================
    // FrameInfo Tests
    // =========================================================================

    #[test]
    fn test_frame_info_new() {
        let frame = FrameInfo::new(64, 4);
        assert_eq!(frame.size, 64);
        assert_eq!(frame.local_count, 4);
    }

    // =========================================================================
    // Eligibility Tests
    // =========================================================================

    #[test]
    fn test_eligibility_is_eligible() {
        assert!(Eligibility::Eligible.is_eligible());
        assert!(!Eligibility::NotTailPosition.is_eligible());
    }

    #[test]
    fn test_eligibility_description() {
        assert!(!Eligibility::AbiMismatch.description().is_empty());
    }

    // =========================================================================
    // Analyzer Tests
    // =========================================================================

    #[test]
    fn test_analyzer_eligible() {
        let caller = FrameInfo::new(64, 2);
        let analyzer = EligibilityAnalyzer::new(caller);
        let info = make_tail_info();

        let result = analyzer.analyze(&info, None);
        assert_eq!(result, Eligibility::Eligible);
    }

    #[test]
    fn test_analyzer_not_tail_position() {
        let caller = FrameInfo::new(64, 2);
        let analyzer = EligibilityAnalyzer::new(caller);

        let mut info = make_tail_info();
        info.status = TailCallStatus::ResultUsed;

        let result = analyzer.analyze(&info, None);
        assert_eq!(result, Eligibility::NotTailPosition);
    }

    #[test]
    fn test_analyzer_escaping_locals() {
        let mut caller = FrameInfo::new(64, 2);
        caller.has_escaping_refs = true;
        let analyzer = EligibilityAnalyzer::new(caller);
        let info = make_tail_info();

        let result = analyzer.analyze(&info, None);
        assert_eq!(result, Eligibility::EscapingLocals);
    }

    #[test]
    fn test_analyzer_abi_mismatch() {
        let mut caller = FrameInfo::new(64, 2);
        caller.convention = CallingConvention::SysV64;

        let mut callee = FrameInfo::new(64, 2);
        callee.convention = CallingConvention::Win64;

        let analyzer = EligibilityAnalyzer::new(caller);
        let info = make_tail_info();

        let result = analyzer.analyze(&info, Some(&callee));
        assert_eq!(result, Eligibility::AbiMismatch);
    }

    #[test]
    fn test_analyzer_frame_too_large() {
        let caller = FrameInfo::new(64, 2);
        let callee = FrameInfo::new(1024, 32);

        let config = EligibilityConfig {
            max_frame_growth: 128,
            allow_cross_abi: true,
        };

        let analyzer = EligibilityAnalyzer::with_config(caller, config);
        let info = make_tail_info();

        let result = analyzer.analyze(&info, Some(&callee));
        assert_eq!(result, Eligibility::FrameTooLarge);
    }

    // =========================================================================
    // Convenience Function Tests
    // =========================================================================

    #[test]
    fn test_filter_eligible() {
        let caller = FrameInfo::new(64, 2);

        let mut info1 = make_tail_info();
        info1.status = TailCallStatus::TailPosition;

        let mut info2 = make_tail_info();
        info2.status = TailCallStatus::ResultUsed;

        let calls = vec![info1, info2];
        let eligible = filter_eligible(&calls, &caller);

        assert_eq!(eligible.len(), 1);
    }
}
