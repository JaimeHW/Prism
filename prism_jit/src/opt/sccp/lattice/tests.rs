use super::*;

// =========================================================================
// Constant Tests
// =========================================================================

#[test]
fn test_constant_int() {
    let c = Constant::int(42);
    assert!(c.is_int());
    assert!(!c.is_float());
    assert_eq!(c.as_int(), Some(42));
}

#[test]
fn test_constant_float() {
    let c = Constant::float(3.125);
    assert!(c.is_float());
    assert!(!c.is_int());
    assert_eq!(c.as_float(), Some(3.125));
}

#[test]
fn test_constant_bool() {
    let t = Constant::bool(true);
    let f = Constant::bool(false);
    assert!(t.is_bool());
    assert_eq!(t.as_bool(), Some(true));
    assert_eq!(f.as_bool(), Some(false));
}

#[test]
fn test_constant_string() {
    let c = Constant::string("hello");
    assert!(matches!(c, Constant::String(_)));
    if let Constant::String(s) = c {
        assert_eq!(&*s, "hello");
    }
}

#[test]
fn test_constant_truthiness() {
    assert!(Constant::int(1).truthiness());
    assert!(!Constant::int(0).truthiness());
    assert!(Constant::float(0.1).truthiness());
    assert!(!Constant::float(0.0).truthiness());
    assert!(Constant::bool(true).truthiness());
    assert!(!Constant::bool(false).truthiness());
    assert!(!Constant::None.truthiness());
    assert!(Constant::string("x").truthiness());
    assert!(!Constant::string("").truthiness());
    assert!(!Constant::EmptyTuple.truthiness());
    assert!(!Constant::EmptyList.truthiness());
    assert!(!Constant::EmptyDict.truthiness());
}

#[test]
fn test_constant_negate() {
    assert_eq!(Constant::int(5).negate(), Some(Constant::int(-5)));
    assert_eq!(Constant::int(-3).negate(), Some(Constant::int(3)));
    assert_eq!(Constant::float(2.5).negate(), Some(Constant::float(-2.5)));
    assert_eq!(Constant::bool(true).negate(), None);
}

#[test]
fn test_constant_logical_not() {
    assert_eq!(Constant::bool(true).logical_not(), Constant::bool(false));
    assert_eq!(Constant::bool(false).logical_not(), Constant::bool(true));
    assert_eq!(Constant::int(0).logical_not(), Constant::bool(true));
    assert_eq!(Constant::int(1).logical_not(), Constant::bool(false));
}

#[test]
fn test_constant_bitwise_not() {
    assert_eq!(Constant::int(0).bitwise_not(), Some(Constant::int(-1)));
    assert_eq!(Constant::int(-1).bitwise_not(), Some(Constant::int(0)));
    assert_eq!(Constant::float(1.0).bitwise_not(), None);
}

#[test]
fn test_constant_int_to_float_coercion() {
    let c = Constant::int(42);
    assert_eq!(c.as_float(), Some(42.0));
}

// =========================================================================
// LatticeValue Construction Tests
// =========================================================================

#[test]
fn test_lattice_undef() {
    let v = LatticeValue::undef();
    assert!(v.is_undef());
    assert!(!v.is_constant());
    assert!(!v.is_overdefined());
}

#[test]
fn test_lattice_overdefined() {
    let v = LatticeValue::overdefined();
    assert!(v.is_overdefined());
    assert!(!v.is_constant());
    assert!(!v.is_undef());
}

#[test]
fn test_lattice_constant() {
    let v = LatticeValue::int(42);
    assert!(v.is_constant());
    assert!(!v.is_undef());
    assert!(!v.is_overdefined());
    assert_eq!(v.as_constant(), Some(&Constant::Int(42)));
}

#[test]
fn test_lattice_into_constant() {
    let v = LatticeValue::int(100);
    assert_eq!(v.into_constant(), Some(Constant::Int(100)));

    let v = LatticeValue::overdefined();
    assert_eq!(v.into_constant(), None);
}

#[test]
fn test_lattice_default() {
    let v = LatticeValue::default();
    assert!(v.is_undef());
}

// =========================================================================
// Meet Operation Tests
// =========================================================================

#[test]
fn test_meet_undef_identity() {
    let undef = LatticeValue::undef();
    let const_42 = LatticeValue::int(42);
    let overdefined = LatticeValue::overdefined();

    // meet(Undef, x) = x
    assert_eq!(undef.meet(&const_42), const_42);
    assert_eq!(undef.meet(&overdefined), overdefined);
    assert_eq!(undef.meet(&undef), undef);

    // meet(x, Undef) = x
    assert_eq!(const_42.meet(&undef), const_42);
    assert_eq!(overdefined.meet(&undef), overdefined);
}

#[test]
fn test_meet_overdefined_absorbs() {
    let overdefined = LatticeValue::overdefined();
    let const_42 = LatticeValue::int(42);
    let undef = LatticeValue::undef();

    // meet(Overdefined, x) = Overdefined
    assert_eq!(overdefined.meet(&const_42), overdefined);
    assert_eq!(overdefined.meet(&undef), overdefined);
    assert_eq!(overdefined.meet(&overdefined), overdefined);

    // meet(x, Overdefined) = Overdefined
    assert_eq!(const_42.meet(&overdefined), overdefined);
}

#[test]
fn test_meet_same_constants() {
    let a = LatticeValue::int(42);
    let b = LatticeValue::int(42);
    assert_eq!(a.meet(&b), LatticeValue::int(42));

    let x = LatticeValue::float(3.125);
    let y = LatticeValue::float(3.125);
    assert_eq!(x.meet(&y), LatticeValue::float(3.125));
}

#[test]
fn test_meet_different_constants() {
    let a = LatticeValue::int(1);
    let b = LatticeValue::int(2);
    assert_eq!(a.meet(&b), LatticeValue::overdefined());

    let x = LatticeValue::bool(true);
    let y = LatticeValue::bool(false);
    assert_eq!(x.meet(&y), LatticeValue::overdefined());
}

// =========================================================================
// Higher Than Tests
// =========================================================================

#[test]
fn test_higher_than() {
    let undef = LatticeValue::undef();
    let constant = LatticeValue::int(0);
    let overdefined = LatticeValue::overdefined();

    // Overdefined is higher than everything except itself
    assert!(overdefined.higher_than(&constant));
    assert!(overdefined.higher_than(&undef));
    assert!(!overdefined.higher_than(&overdefined));

    // Constant is higher than Undef only
    assert!(constant.higher_than(&undef));
    assert!(!constant.higher_than(&constant));
    assert!(!constant.higher_than(&overdefined));

    // Undef is lower than everything
    assert!(!undef.higher_than(&undef));
    assert!(!undef.higher_than(&constant));
    assert!(!undef.higher_than(&overdefined));
}

// =========================================================================
// Merge Tests
// =========================================================================

#[test]
fn test_merge_undef_to_constant() {
    let mut v = LatticeValue::undef();
    let changed = v.merge(&LatticeValue::int(42));
    assert!(changed);
    assert_eq!(v, LatticeValue::int(42));
}

#[test]
fn test_merge_constant_to_overdefined() {
    let mut v = LatticeValue::int(1);
    let changed = v.merge(&LatticeValue::int(2));
    assert!(changed);
    assert_eq!(v, LatticeValue::overdefined());
}

#[test]
fn test_merge_same_constant_no_change() {
    let mut v = LatticeValue::int(42);
    let changed = v.merge(&LatticeValue::int(42));
    assert!(!changed);
    assert_eq!(v, LatticeValue::int(42));
}

#[test]
fn test_merge_overdefined_no_change() {
    let mut v = LatticeValue::overdefined();
    let changed = v.merge(&LatticeValue::int(42));
    assert!(!changed);
    assert_eq!(v, LatticeValue::overdefined());
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_float_nan_handling() {
    // NaN is not equal to itself, but our constants use Rust's PartialEq
    let nan1 = Constant::float(f64::NAN);
    let nan2 = Constant::float(f64::NAN);
    // Due to NaN != NaN, these will not be equal
    assert_ne!(nan1, nan2);
}

#[test]
fn test_constant_negate_overflow() {
    // i64::MIN cannot be negated without overflow
    let min = Constant::int(i64::MIN);
    assert_eq!(min.negate(), None);
}
