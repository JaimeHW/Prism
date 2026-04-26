use super::*;

// =========================================================================
// BoundValue Tests
// =========================================================================

#[test]
fn test_bound_constant() {
    let bound = BoundValue::Constant(100);
    assert!(bound.is_constant());
    assert_eq!(bound.as_constant(), Some(100));
    assert_eq!(bound.as_node(), None);
}

#[test]
fn test_bound_node() {
    let bound = BoundValue::Node(NodeId::new(5));
    assert!(!bound.is_constant());
    assert_eq!(bound.as_constant(), None);
    assert_eq!(bound.as_node(), Some(NodeId::new(5)));
}

// =========================================================================
// RangeCheck Tests
// =========================================================================

fn make_lower_check() -> RangeCheck {
    RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    )
}

fn make_upper_check() -> RangeCheck {
    RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    )
}

#[test]
fn test_check_is_lower_bound() {
    assert!(make_lower_check().is_lower_bound());
    assert!(!make_upper_check().is_lower_bound());
}

#[test]
fn test_check_is_upper_bound() {
    assert!(!make_lower_check().is_upper_bound());
    assert!(make_upper_check().is_upper_bound());

    let inclusive = RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Constant(99),
        RangeCheckKind::UpperBoundInclusive,
        0,
    );
    assert!(inclusive.is_upper_bound());
}

#[test]
fn test_check_has_constant_bound() {
    assert!(make_lower_check().has_constant_bound());
    assert!(make_upper_check().has_constant_bound());

    let node_bound = RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Node(NodeId::new(10)),
        RangeCheckKind::UpperBound,
        0,
    );
    assert!(!node_bound.has_constant_bound());
}

#[test]
fn test_check_constant_bound() {
    assert_eq!(make_lower_check().constant_bound(), Some(0));
    assert_eq!(make_upper_check().constant_bound(), Some(100));

    let node_bound = RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Node(NodeId::new(10)),
        RangeCheckKind::UpperBound,
        0,
    );
    assert_eq!(node_bound.constant_bound(), None);
}

// =========================================================================
// RangeCheckCollection Tests
// =========================================================================

#[test]
fn test_collection_new() {
    let coll = RangeCheckCollection::new();
    assert!(coll.is_empty());
    assert_eq!(coll.len(), 0);
}

#[test]
fn test_collection_add() {
    let mut coll = RangeCheckCollection::new();
    coll.add(make_lower_check());
    coll.add(make_upper_check());

    assert_eq!(coll.len(), 2);
    assert!(!coll.is_empty());
}

#[test]
fn test_collection_add_all() {
    let mut coll = RangeCheckCollection::new();
    coll.add_all(vec![make_lower_check(), make_upper_check()]);

    assert_eq!(coll.len(), 2);
}

#[test]
fn test_collection_get() {
    let mut coll = RangeCheckCollection::new();
    coll.add(make_lower_check());

    assert!(coll.get(0).is_some());
    assert!(coll.get(1).is_none());
}

#[test]
fn test_collection_for_loop() {
    let mut coll = RangeCheckCollection::new();

    let check1 = RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0, // loop 0
    );
    let check2 = RangeCheck::new(
        NodeId::new(3),
        NodeId::new(4),
        NodeId::new(5),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        1, // loop 1
    );

    coll.add(check1);
    coll.add(check2);

    let loop0_checks: Vec<_> = coll.for_loop(0).collect();
    assert_eq!(loop0_checks.len(), 1);

    let loop1_checks: Vec<_> = coll.for_loop(1).collect();
    assert_eq!(loop1_checks.len(), 1);

    let loop2_checks: Vec<_> = coll.for_loop(2).collect();
    assert_eq!(loop2_checks.len(), 0);
}

#[test]
fn test_collection_for_iv() {
    let mut coll = RangeCheckCollection::new();

    let iv = NodeId::new(1);
    let check1 = RangeCheck::new(
        NodeId::new(0),
        iv,
        NodeId::new(2),
        BoundValue::Constant(0),
        RangeCheckKind::LowerBound,
        0,
    );
    let check2 = RangeCheck::new(
        NodeId::new(3),
        iv,
        NodeId::new(4),
        BoundValue::Constant(100),
        RangeCheckKind::UpperBound,
        0,
    );

    coll.add(check1);
    coll.add(check2);

    let iv_checks: Vec<_> = coll.for_iv(iv).collect();
    assert_eq!(iv_checks.len(), 2);

    let other_checks: Vec<_> = coll.for_iv(NodeId::new(999)).collect();
    assert_eq!(other_checks.len(), 0);
}

#[test]
fn test_collection_count_lower_bounds() {
    let mut coll = RangeCheckCollection::new();
    coll.add(make_lower_check());
    coll.add(make_upper_check());
    coll.add(make_lower_check());

    assert_eq!(coll.count_lower_bounds(), 2);
}

#[test]
fn test_collection_count_upper_bounds() {
    let mut coll = RangeCheckCollection::new();
    coll.add(make_lower_check());
    coll.add(make_upper_check());
    coll.add(make_upper_check());

    assert_eq!(coll.count_upper_bounds(), 2);
}

#[test]
fn test_collection_count_constant_bounds() {
    let mut coll = RangeCheckCollection::new();
    coll.add(make_lower_check());
    coll.add(make_upper_check());

    let node_bound = RangeCheck::new(
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        BoundValue::Node(NodeId::new(10)),
        RangeCheckKind::UpperBound,
        0,
    );
    coll.add(node_bound);

    assert_eq!(coll.count_constant_bounds(), 2);
}
