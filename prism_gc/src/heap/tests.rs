use super::*;

#[test]
fn test_align_up() {
    assert_eq!(align_up(0, 8), 0);
    assert_eq!(align_up(1, 8), 8);
    assert_eq!(align_up(7, 8), 8);
    assert_eq!(align_up(8, 8), 8);
    assert_eq!(align_up(9, 8), 16);
}

#[test]
fn test_heap_creation() {
    let heap = GcHeap::with_defaults();
    assert!(!heap.should_minor_collect());
}
