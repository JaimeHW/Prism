use super::*;

// =========================================================================
// TripCount Tests
// =========================================================================

#[test]
fn test_trip_count_constant() {
    let tc = TripCount::constant(100);
    assert!(tc.is_exact());
    assert!(!tc.is_unknown());
    assert!(!tc.is_symbolic());
    assert_eq!(tc.as_constant(), Some(100));
}

#[test]
fn test_trip_count_unknown() {
    let tc = TripCount::unknown();
    assert!(!tc.is_exact());
    assert!(tc.is_unknown());
    assert!(!tc.is_symbolic());
    assert_eq!(tc.as_constant(), None);
}

#[test]
fn test_trip_count_at_most() {
    let tc = TripCount::AtMost(50);
    assert!(!tc.is_exact());
    assert!(!tc.is_unknown());
    assert_eq!(tc.as_constant(), Some(50));
}

#[test]
fn test_trip_count_symbolic() {
    let sym = SymbolicTripCount::new(NodeId::new(5), 0, TripCountValue::Constant(0), 1);
    let tc = TripCount::Symbolic(sym);
    assert!(!tc.is_exact());
    assert!(tc.is_symbolic());
    assert!(tc.as_symbolic().is_some());
}

#[test]
fn test_trip_count_executes_at_least_once() {
    assert!(TripCount::Constant(1).executes_at_least_once());
    assert!(TripCount::Constant(100).executes_at_least_once());
    assert!(!TripCount::Constant(0).executes_at_least_once());
    assert!(!TripCount::AtMost(100).executes_at_least_once());
    assert!(!TripCount::Unknown.executes_at_least_once());
}

#[test]
fn test_trip_count_maybe_zero() {
    let inner = TripCount::Constant(10);
    let tc = TripCount::MaybeZero(Box::new(inner));
    assert!(!tc.executes_at_least_once());
}

// =========================================================================
// MaxIVValue Tests
// =========================================================================

#[test]
fn test_max_iv_constant() {
    let iv = make_canonical_iv();
    let tc = TripCount::Constant(100);
    let max = tc.max_iv_value(&iv);
    assert_eq!(max, Some(MaxIVValue::Constant(99)));
}

#[test]
fn test_max_iv_with_step() {
    let iv = InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Constant(2),
        InductionDirection::Increasing,
        None,
    );
    let tc = TripCount::Constant(50);
    let max = tc.max_iv_value(&iv);
    // max = 0 + 2 * (50 - 1) = 98
    assert_eq!(max, Some(MaxIVValue::Constant(98)));
}

#[test]
fn test_max_iv_zero_trip() {
    let iv = make_canonical_iv();
    let tc = TripCount::Constant(0);
    let max = tc.max_iv_value(&iv);
    assert_eq!(max, None);
}

#[test]
fn test_max_iv_symbolic() {
    let iv = make_canonical_iv();
    let sym = SymbolicTripCount::new(NodeId::new(5), 0, TripCountValue::Constant(0), 1);
    let tc = TripCount::Symbolic(sym);
    let max = tc.max_iv_value(&iv);
    // For i < n with step 1: max = n - 1 (offset = 0 - 1 = -1)
    assert_eq!(
        max,
        Some(MaxIVValue::Symbolic {
            bound: NodeId::new(5),
            offset: -1
        })
    );
}

#[test]
fn test_max_iv_definitely_less_than() {
    let max = MaxIVValue::Constant(99);
    assert!(max.definitely_less_than(100));
    assert!(!max.definitely_less_than(99));
    assert!(!max.definitely_less_than(50));
}

#[test]
fn test_max_iv_definitely_at_most() {
    let max = MaxIVValue::Constant(99);
    assert!(max.definitely_at_most(100));
    assert!(max.definitely_at_most(99));
    assert!(!max.definitely_at_most(50));
}

#[test]
fn test_max_iv_symbolic_cannot_prove() {
    let max = MaxIVValue::Symbolic {
        bound: NodeId::new(5),
        offset: -1,
    };
    assert!(!max.definitely_less_than(100));
    assert!(!max.definitely_at_most(100));
}

// =========================================================================
// SymbolicTripCount Tests
// =========================================================================

#[test]
fn test_symbolic_trip_count_new() {
    let sym = SymbolicTripCount::new(NodeId::new(10), 0, TripCountValue::Constant(0), 1);
    assert_eq!(sym.bound_node, NodeId::new(10));
    assert_eq!(sym.offset, 0);
    assert!(sym.exact);
}

#[test]
fn test_symbolic_trip_count_upper_bound() {
    let sym = SymbolicTripCount::upper_bound(NodeId::new(10), 0, TripCountValue::Constant(0), 1);
    assert!(!sym.exact);
}

// =========================================================================
// TripCountValue Tests
// =========================================================================

#[test]
fn test_trip_count_value_constant() {
    let v = TripCountValue::Constant(42);
    assert!(v.is_constant());
    assert_eq!(v.as_constant(), Some(42));
}

#[test]
fn test_trip_count_value_node() {
    let v = TripCountValue::Node(NodeId::new(5));
    assert!(!v.is_constant());
    assert_eq!(v.as_constant(), None);
}

// =========================================================================
// TripCountCache Tests
// =========================================================================

#[test]
fn test_cache_new() {
    let cache = TripCountCache::new();
    assert_eq!(cache.get(0), None);
    assert_eq!(cache.count_known(), 0);
}

#[test]
fn test_cache_with_capacity() {
    let cache = TripCountCache::with_capacity(5);
    assert_eq!(cache.get(0), None);
    for i in 0..5 {
        assert_eq!(cache.get(i), None);
    }
}

#[test]
fn test_cache_set_get() {
    let mut cache = TripCountCache::new();
    cache.set(0, TripCount::Constant(100));
    cache.set(2, TripCount::Unknown);

    assert_eq!(cache.get(0), Some(&TripCount::Constant(100)));
    assert_eq!(cache.get(1), None);
    assert_eq!(cache.get(2), Some(&TripCount::Unknown));
}

#[test]
fn test_cache_has_constant() {
    let mut cache = TripCountCache::new();
    cache.set(0, TripCount::Constant(100));
    cache.set(1, TripCount::Unknown);

    assert!(cache.has_constant(0));
    assert!(!cache.has_constant(1));
    assert!(!cache.has_constant(2));
}

#[test]
fn test_cache_constant() {
    let mut cache = TripCountCache::new();
    cache.set(0, TripCount::Constant(100));
    cache.set(1, TripCount::AtMost(50));

    assert_eq!(cache.constant(0), Some(100));
    assert_eq!(cache.constant(1), Some(50));
    assert_eq!(cache.constant(2), None);
}

#[test]
fn test_cache_count_known() {
    let mut cache = TripCountCache::new();
    cache.set(0, TripCount::Constant(100));
    cache.set(1, TripCount::Unknown);
    cache.set(2, TripCount::AtMost(50));

    assert_eq!(cache.count_known(), 2);
}

#[test]
fn test_cache_count_constant() {
    let mut cache = TripCountCache::new();
    cache.set(0, TripCount::Constant(100));
    cache.set(1, TripCount::Unknown);
    cache.set(2, TripCount::AtMost(50));

    assert_eq!(cache.count_constant(), 1);
}

// =========================================================================
// Helper Functions
// =========================================================================

fn make_canonical_iv() -> InductionVariable {
    InductionVariable::new(
        NodeId::new(0),
        InductionInit::Constant(0),
        InductionStep::Constant(1),
        InductionDirection::Increasing,
        None,
    )
}

use super::super::induction::InductionVariable;
