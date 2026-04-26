use super::*;

// =========================================================================
// InductionInit Tests
// =========================================================================

#[test]
fn test_init_constant() {
    let init = InductionInit::Constant(0);
    assert!(init.is_constant());
    assert_eq!(init.as_constant(), Some(0));
    assert_eq!(init.as_node(), None);
}

#[test]
fn test_init_node() {
    let init = InductionInit::Node(NodeId::new(5));
    assert!(!init.is_constant());
    assert_eq!(init.as_constant(), None);
    assert_eq!(init.as_node(), Some(NodeId::new(5)));
}

// =========================================================================
// InductionStep Tests
// =========================================================================

#[test]
fn test_step_constant() {
    let step = InductionStep::Constant(1);
    assert!(step.is_constant());
    assert_eq!(step.as_constant(), Some(1));
    assert_eq!(step.as_node(), None);
}

#[test]
fn test_step_node() {
    let step = InductionStep::Node(NodeId::new(10));
    assert!(!step.is_constant());
    assert_eq!(step.as_constant(), None);
    assert_eq!(step.as_node(), Some(NodeId::new(10)));
}

#[test]
fn test_step_direction() {
    assert_eq!(
        InductionStep::Constant(1).direction(),
        InductionDirection::Increasing
    );
    assert_eq!(
        InductionStep::Constant(5).direction(),
        InductionDirection::Increasing
    );
    assert_eq!(
        InductionStep::Constant(-1).direction(),
        InductionDirection::Decreasing
    );
    assert_eq!(
        InductionStep::Constant(-10).direction(),
        InductionDirection::Decreasing
    );
    assert_eq!(
        InductionStep::Constant(0).direction(),
        InductionDirection::Unknown
    );
    assert_eq!(
        InductionStep::Node(NodeId::new(0)).direction(),
        InductionDirection::Unknown
    );
}

// =========================================================================
// InductionVariable Tests
// =========================================================================

fn make_canonical_iv() -> InductionVariable {
    InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        Some(NodeId::new(1)),
    )
}

fn make_simple_iv(init: i64, step: i64) -> InductionVariable {
    let direction = if step > 0 {
        InductionDirection::Increasing
    } else if step < 0 {
        InductionDirection::Decreasing
    } else {
        InductionDirection::Unknown
    };
    InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(init),
        InductionStep::Constant(step),
        direction,
        None,
    )
}

#[test]
fn test_iv_is_simple() {
    assert!(make_canonical_iv().is_simple());
    assert!(make_simple_iv(5, 2).is_simple());

    let non_simple = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Node(NodeId::new(5)),
        InductionDirection::Unknown,
        None,
    );
    assert!(!non_simple.is_simple());
}

#[test]
fn test_iv_has_constant_init() {
    assert!(make_canonical_iv().has_constant_init());

    let node_init = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Node(NodeId::new(10)),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );
    assert!(!node_init.has_constant_init());
}

#[test]
fn test_iv_is_canonical() {
    assert!(make_canonical_iv().is_canonical());
    assert!(!make_simple_iv(0, 2).is_canonical());
    assert!(!make_simple_iv(1, 1).is_canonical());
    assert!(!make_simple_iv(0, -1).is_canonical());
}

#[test]
fn test_iv_constant_step() {
    assert_eq!(make_canonical_iv().constant_step(), Some(1));
    assert_eq!(make_simple_iv(0, 5).constant_step(), Some(5));
    assert_eq!(make_simple_iv(0, -3).constant_step(), Some(-3));

    let node_step = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Node(NodeId::new(5)),
        InductionDirection::Unknown,
        None,
    );
    assert_eq!(node_step.constant_step(), None);
}

#[test]
fn test_iv_constant_init() {
    assert_eq!(make_canonical_iv().constant_init(), Some(0));
    assert_eq!(make_simple_iv(10, 1).constant_init(), Some(10));

    let node_init = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Node(NodeId::new(10)),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    );
    assert_eq!(node_init.constant_init(), None);
}

#[test]
fn test_iv_step_magnitude() {
    assert_eq!(make_simple_iv(0, 1).step_magnitude(), Some(1));
    assert_eq!(make_simple_iv(0, 5).step_magnitude(), Some(5));
    assert_eq!(make_simple_iv(0, -3).step_magnitude(), Some(3));
    assert_eq!(make_simple_iv(0, -100).step_magnitude(), Some(100));
}

#[test]
fn test_iv_direction_checks() {
    let inc = make_simple_iv(0, 1);
    assert!(inc.is_increasing());
    assert!(!inc.is_decreasing());
    assert!(inc.is_monotonic());

    let dec = make_simple_iv(10, -1);
    assert!(!dec.is_increasing());
    assert!(dec.is_decreasing());
    assert!(dec.is_monotonic());

    let unknown = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Node(NodeId::new(5)),
        InductionDirection::Unknown,
        None,
    );
    assert!(!unknown.is_increasing());
    assert!(!unknown.is_decreasing());
    assert!(!unknown.is_monotonic());
}

// =========================================================================
// InductionAnalysis Tests
// =========================================================================

#[test]
fn test_analysis_empty() {
    let analysis = InductionAnalysis::empty();
    assert_eq!(analysis.total(), 0);
    assert_eq!(analysis.num_loops(), 0);
    assert!(analysis.get(0).is_none());
}

#[test]
fn test_analysis_with_capacity() {
    let analysis = InductionAnalysis::with_capacity(5);
    assert_eq!(analysis.total(), 0);
    assert_eq!(analysis.num_loops(), 0);
}

#[test]
fn test_analysis_add_loop() {
    let mut analysis = InductionAnalysis::empty();

    let mut ivs = HashMap::new();
    ivs.insert(NodeId::new(0), make_canonical_iv());
    ivs.insert(NodeId::new(1), make_simple_iv(5, 2));

    analysis.add_loop(ivs);

    assert_eq!(analysis.total(), 2);
    assert_eq!(analysis.num_loops(), 1);
    assert!(analysis.get(0).is_some());
}

#[test]
fn test_analysis_is_induction_variable() {
    let mut analysis = InductionAnalysis::empty();

    let mut ivs = HashMap::new();
    ivs.insert(NodeId::new(5), make_canonical_iv());
    analysis.add_loop(ivs);

    assert!(analysis.is_induction_variable(0, NodeId::new(5)));
    assert!(!analysis.is_induction_variable(0, NodeId::new(6)));
    assert!(!analysis.is_induction_variable(1, NodeId::new(5)));
}

#[test]
fn test_analysis_get_iv() {
    let mut analysis = InductionAnalysis::empty();

    let iv = make_canonical_iv();
    let mut ivs = HashMap::new();
    ivs.insert(NodeId::new(5), iv.clone());
    analysis.add_loop(ivs);

    let retrieved = analysis.get_iv(0, NodeId::new(5));
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().phi, iv.phi);

    assert!(analysis.get_iv(0, NodeId::new(6)).is_none());
    assert!(analysis.get_iv(1, NodeId::new(5)).is_none());
}

#[test]
fn test_analysis_iter_all() {
    let mut analysis = InductionAnalysis::empty();

    let mut ivs1 = HashMap::new();
    ivs1.insert(NodeId::new(0), make_canonical_iv());
    analysis.add_loop(ivs1);

    let mut ivs2 = HashMap::new();
    ivs2.insert(NodeId::new(1), make_simple_iv(0, 2));
    ivs2.insert(NodeId::new(2), make_simple_iv(10, -1));
    analysis.add_loop(ivs2);

    let all: Vec<_> = analysis.iter_all().collect();
    assert_eq!(all.len(), 3);
}

#[test]
fn test_analysis_count_canonical() {
    let mut analysis = InductionAnalysis::empty();

    let mut ivs = HashMap::new();
    ivs.insert(NodeId::new(0), make_canonical_iv());
    ivs.insert(NodeId::new(1), make_simple_iv(0, 2)); // not canonical
    ivs.insert(NodeId::new(2), make_simple_iv(1, 1)); // not canonical
    analysis.add_loop(ivs);

    assert_eq!(analysis.count_canonical(), 1);
}

#[test]
fn test_analysis_count_simple() {
    let mut analysis = InductionAnalysis::empty();

    let node_step = InductionVariable::new(
        NodeId::new(3),
        InductionInit::Constant(0),
        InductionStep::Node(NodeId::new(10)),
        InductionDirection::Unknown,
        None,
    );

    let mut ivs = HashMap::new();
    ivs.insert(NodeId::new(0), make_canonical_iv());
    ivs.insert(NodeId::new(1), make_simple_iv(0, 2));
    ivs.insert(NodeId::new(2), node_step);
    analysis.add_loop(ivs);

    assert_eq!(analysis.count_simple(), 2);
}
