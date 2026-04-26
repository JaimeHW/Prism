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
