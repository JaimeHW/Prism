use super::*;

#[test]
fn test_vreg_encoding() {
    assert_eq!(Vreg::V0.encoding(), 0);
    assert_eq!(Vreg::V31.encoding(), 31);
}

#[test]
fn test_vreg_set() {
    let set = VregSet::singleton(Vreg::V5).insert(Vreg::V10);
    assert!(set.contains(Vreg::V5));
    assert!(set.contains(Vreg::V10));
    assert!(!set.contains(Vreg::V0));
    assert_eq!(set.count(), 2);
}

#[test]
fn test_arrangement() {
    assert_eq!(ArrangementSpec::S4.q_bit(), 1);
    assert_eq!(ArrangementSpec::S2.q_bit(), 0);
    assert_eq!(ArrangementSpec::S4.count(), 4);
    assert_eq!(ArrangementSpec::D2.element_bits(), 64);
}

#[test]
fn test_callee_saved() {
    assert!(Vreg::V8.is_callee_saved());
    assert!(Vreg::V15.is_callee_saved());
    assert!(!Vreg::V0.is_callee_saved());
    assert!(!Vreg::V16.is_callee_saved());
}
