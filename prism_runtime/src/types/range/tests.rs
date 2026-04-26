use super::*;

fn collect_i64(range: &RangeObject) -> Vec<i64> {
    range
        .iter()
        .map(|value| value_to_i64(value).expect("range item should fit in i64"))
        .collect()
}

#[test]
fn test_range_from_stop() {
    let r = RangeObject::from_stop(5);
    assert_eq!(r.start_i64(), Some(0));
    assert_eq!(r.stop_i64(), Some(5));
    assert_eq!(r.step_i64(), Some(1));
    assert_eq!(r.len(), 5);
}

#[test]
fn test_range_from_start_stop() {
    let r = RangeObject::from_start_stop(2, 8);
    assert_eq!(r.start_i64(), Some(2));
    assert_eq!(r.stop_i64(), Some(8));
    assert_eq!(r.len(), 6);
}

#[test]
fn test_range_with_step() {
    let r = RangeObject::new(0, 10, 2);
    assert_eq!(r.len(), 5);
    assert_eq!(collect_i64(&r), vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_range_negative_step() {
    let r = RangeObject::new(10, 0, -1);
    assert_eq!(r.len(), 10);
    assert_eq!(collect_i64(&r), vec![10, 9, 8, 7, 6, 5, 4, 3, 2, 1]);
}

#[test]
fn test_empty_range() {
    let r = RangeObject::from_start_stop(5, 5);
    assert!(r.is_empty());
    assert_eq!(r.len(), 0);
    assert!(r.iter().next().is_none());
}

#[test]
fn test_range_get_and_contains() {
    let r = RangeObject::new(10, 20, 2);
    assert_eq!(r.get(0), Some(10));
    assert_eq!(r.get(4), Some(18));
    assert_eq!(r.get(-1), Some(18));
    assert_eq!(r.get(5), None);
    assert!(r.contains(12));
    assert!(!r.contains(13));
}

#[test]
fn test_range_reverse() {
    let r = RangeObject::new(1, 6, 1);
    let reversed = r.reverse();
    assert_eq!(collect_i64(&reversed), vec![5, 4, 3, 2, 1]);
}

#[test]
fn test_big_range_iterates_lazily() {
    let stop = BigInt::from(1_u8) << 1000_u32;
    let range = RangeObject::from_bigints(BigInt::zero(), stop, BigInt::one());

    let mut iter = range.iter();
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(0));
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(1));
    assert_eq!(value_to_i64(iter.next().unwrap()), Some(2));
}

#[test]
fn test_big_range_try_len_reports_overflow() {
    let stop = BigInt::from(1_u8) << 1000_u32;
    let range = RangeObject::from_bigints(BigInt::zero(), stop, BigInt::one());
    assert_eq!(range.try_len(), None);
    assert_eq!(range.len(), usize::MAX);
}

#[test]
fn test_big_range_display() {
    let stop = BigInt::from(1_u8) << 80_u32;
    let range = RangeObject::from_bigints(BigInt::zero(), stop.clone(), BigInt::one());
    assert_eq!(range.to_string(), format!("range(0, {stop})"));
}
