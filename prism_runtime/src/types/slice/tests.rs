use super::*;

// =========================================================================
// SliceValue Tests
// =========================================================================

#[test]
fn test_slice_value_none() {
    let v = SliceValue::none();
    assert!(v.is_none());
    assert!(!v.is_some());
    assert_eq!(v.get(), None);
    assert_eq!(v.unwrap_or(42), 42);
}

#[test]
fn test_slice_value_some() {
    let v = SliceValue::some(10);
    assert!(v.is_some());
    assert!(!v.is_none());
    assert_eq!(v.get(), Some(10));
    assert_eq!(v.unwrap_or(42), 10);
}

#[test]
fn test_slice_value_negative() {
    let v = SliceValue::some(-5);
    assert!(v.is_some());
    assert_eq!(v.get(), Some(-5));
}

#[test]
fn test_slice_value_zero() {
    let v = SliceValue::some(0);
    assert!(v.is_some());
    assert_eq!(v.get(), Some(0));
}

#[test]
fn test_slice_value_from_option() {
    let v1: SliceValue = Some(10).into();
    let v2: SliceValue = None.into();
    assert_eq!(v1.get(), Some(10));
    assert_eq!(v2.get(), None);
}

#[test]
fn test_slice_value_map() {
    let v = SliceValue::some(5);
    let mapped = v.map(|x| x * 2);
    assert_eq!(mapped.get(), Some(10));

    let none = SliceValue::none();
    let mapped_none = none.map(|x| x * 2);
    assert!(mapped_none.is_none());
}

// =========================================================================
// SliceObject Construction Tests
// =========================================================================

#[test]
fn test_slice_new_all_values() {
    let s = SliceObject::new(Some(1), Some(10), Some(2));
    assert_eq!(s.start(), Some(1));
    assert_eq!(s.stop(), Some(10));
    assert_eq!(s.step(), Some(2));
}

#[test]
fn test_slice_new_all_none() {
    let s = SliceObject::new(None, None, None);
    assert_eq!(s.start(), None);
    assert_eq!(s.stop(), None);
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_stop_only() {
    let s = SliceObject::stop_only(5);
    assert_eq!(s.start(), None);
    assert_eq!(s.stop(), Some(5));
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_start_stop() {
    let s = SliceObject::start_stop(2, 8);
    assert_eq!(s.start(), Some(2));
    assert_eq!(s.stop(), Some(8));
    assert_eq!(s.step(), None);
}

#[test]
fn test_slice_full() {
    let s = SliceObject::full(0, 10, 2);
    assert_eq!(s.start(), Some(0));
    assert_eq!(s.stop(), Some(10));
    assert_eq!(s.step(), Some(2));
}

#[test]
#[should_panic(expected = "slice step cannot be zero")]
fn test_slice_zero_step_panics() {
    SliceObject::new(Some(0), Some(10), Some(0));
}

#[test]
#[should_panic(expected = "slice step cannot be zero")]
fn test_slice_full_zero_step_panics() {
    SliceObject::full(0, 10, 0);
}

// =========================================================================
// SliceIndices Resolution Tests
// =========================================================================

#[test]
fn test_indices_simple_forward() {
    // [1:5] on a length-10 sequence
    let s = SliceObject::start_stop(1, 5);
    let idx = s.indices(10);
    assert_eq!(idx.start, 1);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 4);
}

#[test]
fn test_indices_full_slice() {
    // [:] on a length-5 sequence
    let s = SliceObject::new(None, None, None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_negative_start() {
    // [-3:] on a length-5 sequence = [2:]
    let s = SliceObject::new(Some(-3), None, None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 2);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_negative_stop() {
    // [:-2] on a length-5 sequence = [:3]
    let s = SliceObject::new(None, Some(-2), None);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 3);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_negative_both() {
    // [-4:-1] on a length-5 sequence = [1:4]
    let s = SliceObject::start_stop(-4, -1);
    let idx = s.indices(5);
    assert_eq!(idx.start, 1);
    assert_eq!(idx.stop, 4);
    assert_eq!(idx.step, 1);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_with_step() {
    // [0:10:2] on a length-10 sequence
    let s = SliceObject::full(0, 10, 2);
    let idx = s.indices(10);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 10);
    assert_eq!(idx.step, 2);
    assert_eq!(idx.length, 5); // 0, 2, 4, 6, 8
}

#[test]
fn test_indices_reverse() {
    // [::-1] on a length-5 sequence
    let s = SliceObject::new(None, None, Some(-1));
    let idx = s.indices(5);
    assert_eq!(idx.start, 4);
    assert_eq!(idx.step, -1);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_reverse_with_bounds() {
    // [4:1:-1] on a length-5 sequence
    let s = SliceObject::full(4, 1, -1);
    let idx = s.indices(5);
    assert_eq!(idx.start, 4);
    assert_eq!(idx.step, -1);
    assert_eq!(idx.length, 3); // 4, 3, 2
}

#[test]
fn test_indices_empty_forward() {
    // [5:3] on any sequence = empty (start > stop with positive step)
    let s = SliceObject::start_stop(5, 3);
    let idx = s.indices(10);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_empty_reverse() {
    // [3:5:-1] = empty (start < stop with negative step)
    let s = SliceObject::full(3, 5, -1);
    let idx = s.indices(10);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_out_of_bounds_clamped() {
    // [0:100] on a length-5 sequence = [0:5]
    let s = SliceObject::start_stop(0, 100);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 5);
    assert_eq!(idx.length, 5);
}

#[test]
fn test_indices_negative_out_of_bounds() {
    // [-100:3] on a length-5 sequence = [0:3]
    let s = SliceObject::start_stop(-100, 3);
    let idx = s.indices(5);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.stop, 3);
    assert_eq!(idx.length, 3);
}

#[test]
fn test_indices_empty_sequence() {
    let s = SliceObject::new(None, None, None);
    let idx = s.indices(0);
    assert_eq!(idx.length, 0);
}

#[test]
fn test_indices_step_2() {
    // [::2] on a length-7 sequence
    let s = SliceObject::new(None, None, Some(2));
    let idx = s.indices(7);
    assert_eq!(idx.start, 0);
    assert_eq!(idx.step, 2);
    assert_eq!(idx.length, 4); // 0, 2, 4, 6
}

#[test]
fn test_indices_step_3() {
    // [1:10:3] on a length-10 sequence
    let s = SliceObject::full(1, 10, 3);
    let idx = s.indices(10);
    assert_eq!(idx.start, 1);
    assert_eq!(idx.step, 3);
    assert_eq!(idx.length, 3); // 1, 4, 7
}

// =========================================================================
// SliceIndexIter Tests
// =========================================================================

#[test]
fn test_slice_iter_forward() {
    let s = SliceObject::start_stop(1, 5);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![1, 2, 3, 4]);
}

#[test]
fn test_slice_iter_step() {
    let s = SliceObject::full(0, 10, 2);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_slice_iter_reverse() {
    let s = SliceObject::new(None, None, Some(-1));
    let indices: Vec<usize> = s.indices(5).iter().collect();
    assert_eq!(indices, vec![4, 3, 2, 1, 0]);
}

#[test]
fn test_slice_iter_reverse_step() {
    let s = SliceObject::full(8, 0, -2);
    let indices: Vec<usize> = s.indices(10).iter().collect();
    assert_eq!(indices, vec![8, 6, 4, 2]);
}

#[test]
fn test_slice_iter_size_hint() {
    let s = SliceObject::start_stop(0, 5);
    let iter = s.indices(10).iter();
    assert_eq!(iter.size_hint(), (5, Some(5)));
    assert_eq!(iter.len(), 5);
}

// =========================================================================
// Display Tests
// =========================================================================

#[test]
fn test_slice_display() {
    let s = SliceObject::new(Some(1), Some(10), Some(2));
    assert_eq!(format!("{}", s), "slice(1, 10, 2)");

    let s2 = SliceObject::new(None, Some(5), None);
    assert_eq!(format!("{}", s2), "slice(None, 5, None)");
}

// =========================================================================
// Memory Layout Verification
// =========================================================================

#[test]
fn test_slice_value_size() {
    // SliceValue should be exactly 8 bytes
    assert_eq!(std::mem::size_of::<SliceValue>(), 8);
}

#[test]
fn test_slice_object_size() {
    // SliceObject should be compact: header (16) + 3 * SliceValue (24) = 40 bytes
    assert_eq!(std::mem::size_of::<SliceObject>(), 40);
}

#[test]
fn test_slice_alignment() {
    // SliceObject should be 8-byte aligned for optimal cache line usage
    assert!(std::mem::align_of::<SliceObject>() >= 8);
}
