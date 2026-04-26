use super::*;

#[test]
fn test_mono_ic_hit() {
    let mut ic = MonoIC::empty();
    ic.update(42, 10);

    assert_eq!(ic.check(42), Some(10));
    assert_eq!(ic.check(43), None);
}

#[test]
fn test_mono_ic_counters() {
    let mut ic = MonoIC::empty();
    ic.update(1, 0);

    for _ in 0..100 {
        ic.record_hit();
    }
    for _ in 0..10 {
        ic.record_miss();
    }

    assert_eq!(ic.hits, 100);
    assert_eq!(ic.misses, 10);
    assert!(ic.hit_rate() > 90.0);
}

#[test]
fn test_poly_ic() {
    let mut ic = PolyIC::empty();

    assert!(ic.add(1, 10));
    assert!(ic.add(2, 20));
    assert!(ic.add(3, 30));
    assert!(ic.add(4, 40));
    assert!(!ic.add(5, 50)); // Full

    assert_eq!(ic.lookup(1), Some(10));
    assert_eq!(ic.lookup(3), Some(30));
    assert_eq!(ic.lookup(5), None);
    assert!(ic.is_full());
}

#[test]
fn test_call_ic() {
    let mut ic = CallIC::empty();
    let func: fn() = || {};
    let ptr = func as *const ();

    ic.update(ptr, 2);
    assert!(ic.check(ptr, 2));
    assert!(!ic.check(ptr, 3)); // Wrong argc
}
