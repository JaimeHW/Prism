use super::super::RegClass;
use super::super::interval::{LiveRange, ProgPoint};
use super::*;

fn make_interval(vreg: u32, start: u32, end: u32) -> LiveInterval {
    let mut interval = LiveInterval::new(VReg::new(vreg), RegClass::Int);
    interval.add_range(LiveRange::new(
        ProgPoint::before(start),
        ProgPoint::before(end),
    ));
    interval
}

#[test]
fn test_no_interference() {
    // [0, 10) and [20, 30) don't overlap
    let intervals = vec![make_interval(0, 0, 10), make_interval(1, 20, 30)];

    let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

    assert!(!graph.interferes(VReg::new(0), VReg::new(1)));
}

#[test]
fn test_interference() {
    // [0, 20) and [10, 30) overlap
    let intervals = vec![make_interval(0, 0, 20), make_interval(1, 10, 30)];

    let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

    assert!(graph.interferes(VReg::new(0), VReg::new(1)));
    assert!(graph.interferes(VReg::new(1), VReg::new(0)));
}

#[test]
fn test_degree() {
    // v0: [0, 30), v1: [10, 20), v2: [25, 35)
    // v0 interferes with v1 and v2
    // v1 interferes with v0 only
    // v2 interferes with v0 only
    let intervals = vec![
        make_interval(0, 0, 30),
        make_interval(1, 10, 20),
        make_interval(2, 25, 35),
    ];

    let graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

    assert_eq!(graph.degree(VReg::new(0)), 2);
    assert_eq!(graph.degree(VReg::new(1)), 1);
    assert_eq!(graph.degree(VReg::new(2)), 1);
}

#[test]
fn test_coalesce() {
    let intervals = vec![
        make_interval(0, 0, 30),
        make_interval(1, 10, 20),
        make_interval(2, 25, 35),
    ];

    let mut graph = InterferenceGraph::build_legacy(&intervals, 14, 15);

    // Coalesce v1 into v0
    graph.coalesce(VReg::new(0), VReg::new(1));

    // v1 should be gone
    assert!(!graph.adjacency.contains_key(&VReg::new(1)));

    // v0 should still interfere with v2
    assert!(graph.interferes(VReg::new(0), VReg::new(2)));
}

// =========================================================================
// Vector Register Class Tests
// =========================================================================

#[test]
fn test_vector_register_counts_full_api() {
    // Test with explicit YMM and ZMM counts
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    assert_eq!(graph.k_gpr(), 14);
    assert_eq!(graph.k_xmm(), 16);
    assert_eq!(graph.k_ymm(), 15);
    assert_eq!(graph.k_zmm(), 31);
}

#[test]
fn test_vector_register_counts_legacy_api() {
    // Legacy API should provide default YMM=16, ZMM=32
    let graph = InterferenceGraph::new_legacy(14, 16);

    assert_eq!(graph.k_gpr(), 14);
    assert_eq!(graph.k_xmm(), 16);
    assert_eq!(graph.k_ymm(), 16); // Default
    assert_eq!(graph.k_zmm(), 32); // Default
}

#[test]
fn test_k_for_class_int() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    assert_eq!(graph.k_for_class(RegClass::Int), 14);
    assert_eq!(graph.k_for_class(RegClass::Any), 14);
}

#[test]
fn test_k_for_class_float() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    assert_eq!(graph.k_for_class(RegClass::Float), 16);
}

#[test]
fn test_k_for_class_vec256() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    assert_eq!(graph.k_for_class(RegClass::Vec256), 15);
}

#[test]
fn test_k_for_class_vec512() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    assert_eq!(graph.k_for_class(RegClass::Vec512), 31);
}

#[test]
fn test_legacy_k_method_matches_k_for_class() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    // Legacy k() method should match GPR/XMM
    assert_eq!(graph.k(false), graph.k_for_class(RegClass::Int));
    assert_eq!(graph.k(true), graph.k_for_class(RegClass::Float));
}

#[test]
fn test_build_with_vector_counts() {
    let intervals = vec![make_interval(0, 0, 10)];

    let graph = InterferenceGraph::build(&intervals, 13, 15, 14, 30);

    assert_eq!(graph.k_gpr(), 13);
    assert_eq!(graph.k_xmm(), 15);
    assert_eq!(graph.k_ymm(), 14);
    assert_eq!(graph.k_zmm(), 30);
}

#[test]
fn test_build_legacy_with_default_vector_counts() {
    let intervals = vec![make_interval(0, 0, 10)];

    let graph = InterferenceGraph::build_legacy(&intervals, 13, 15);

    assert_eq!(graph.k_gpr(), 13);
    assert_eq!(graph.k_xmm(), 15);
    assert_eq!(graph.k_ymm(), 16); // Default
    assert_eq!(graph.k_zmm(), 32); // Default
}

#[test]
fn test_all_register_classes_accessible() {
    let graph = InterferenceGraph::new(14, 16, 15, 31);

    // Verify all register classes return correct values
    let test_cases = [
        (RegClass::Int, 14),
        (RegClass::Any, 14),
        (RegClass::Float, 16),
        (RegClass::Vec256, 15),
        (RegClass::Vec512, 31),
    ];

    for (class, expected) in test_cases {
        assert_eq!(
            graph.k_for_class(class),
            expected,
            "k_for_class({:?}) should return {}",
            class,
            expected
        );
    }
}
