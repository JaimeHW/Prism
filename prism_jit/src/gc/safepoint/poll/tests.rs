use super::*;

#[test]
fn test_poll_size() {
    assert_eq!(SAFEPOINT_POLL_SIZE, 3);
}

#[test]
fn test_elision_leaf_no_loops() {
    let traits = FunctionTraits {
        instruction_count: 10,
        has_loops: false,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 30,
    };
    assert!(should_elide_safepoints(&traits));
}

#[test]
fn test_no_elision_has_loops() {
    let traits = FunctionTraits {
        instruction_count: 10,
        has_loops: true,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 30,
    };
    assert!(!should_elide_safepoints(&traits));
}

#[test]
fn test_no_elision_has_calls() {
    let traits = FunctionTraits {
        instruction_count: 10,
        has_loops: false,
        has_calls: true,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 30,
    };
    assert!(!should_elide_safepoints(&traits));
}

#[test]
fn test_no_elision_has_allocations() {
    let traits = FunctionTraits {
        instruction_count: 10,
        has_loops: false,
        has_calls: false,
        has_allocations: true,
        can_throw: false,
        bytecode_size: 30,
    };
    assert!(!should_elide_safepoints(&traits));
}

#[test]
fn test_no_elision_too_large() {
    let traits = FunctionTraits {
        instruction_count: 100,
        has_loops: false,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 200,
    };
    assert!(!should_elide_safepoints(&traits));
}

#[test]
fn test_placement_none() {
    let traits = FunctionTraits {
        instruction_count: 10,
        has_loops: false,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 30,
    };
    assert_eq!(
        analyze_safepoint_placement(&traits),
        SafepointPlacement::None
    );
}

#[test]
fn test_placement_return_only() {
    let traits = FunctionTraits {
        instruction_count: 100, // Too large for elision
        has_loops: false,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 200,
    };
    assert_eq!(
        analyze_safepoint_placement(&traits),
        SafepointPlacement::ReturnOnly
    );
}

#[test]
fn test_placement_back_edges() {
    let traits = FunctionTraits {
        instruction_count: 50,
        has_loops: true,
        has_calls: false,
        has_allocations: false,
        can_throw: false,
        bytecode_size: 100,
    };
    assert_eq!(
        analyze_safepoint_placement(&traits),
        SafepointPlacement::BackEdgesAndReturn
    );
}

#[test]
fn test_placement_full() {
    let traits = FunctionTraits {
        instruction_count: 50,
        has_loops: true,
        has_calls: true,
        has_allocations: true,
        can_throw: false,
        bytecode_size: 100,
    };
    assert_eq!(
        analyze_safepoint_placement(&traits),
        SafepointPlacement::Full
    );
}
