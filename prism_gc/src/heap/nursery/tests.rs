use super::*;

#[test]
fn test_nursery_creation() {
    let nursery = Nursery::new(1024 * 1024); // 1MB
    assert_eq!(nursery.size(), 1024 * 1024);
    assert_eq!(nursery.allocated(), 0);
}

#[test]
fn test_nursery_allocation() {
    let nursery = Nursery::new(1024);

    let ptr1 = nursery.alloc(64).expect("Alloc 1 failed");
    assert_eq!(nursery.allocated(), 64);

    let ptr2 = nursery.alloc(64).expect("Alloc 2 failed");
    assert_eq!(nursery.allocated(), 128);

    // Pointers should be consecutive
    assert_eq!(ptr2.as_ptr() as usize - ptr1.as_ptr() as usize, 64);
}

#[test]
fn test_nursery_exhaustion() {
    let nursery = Nursery::new(128);

    // Fill the nursery
    let _ = nursery.alloc(64);
    let _ = nursery.alloc(64);

    // Should fail - nursery is full
    assert!(nursery.alloc(1).is_none());
    assert!(nursery.is_full());
}

#[test]
fn test_nursery_contains() {
    let nursery = Nursery::new(1024);
    let ptr = nursery.alloc(64).expect("Alloc failed");

    assert!(nursery.contains(ptr.as_ptr() as *const ()));
    assert!(nursery.in_from_space(ptr.as_ptr() as *const ()));
    assert!(!nursery.in_to_space(ptr.as_ptr() as *const ()));
}

#[test]
fn test_nursery_swap() {
    let mut nursery = Nursery::new(1024);

    // Allocate in from-space
    let ptr1 = nursery.alloc(64).expect("Alloc failed");
    assert!(nursery.in_from_space(ptr1.as_ptr() as *const ()));

    // Allocate in to-space (simulating copy)
    let ptr2 = nursery.alloc_to_space(64).expect("ToSpace alloc failed");
    assert!(nursery.in_to_space(ptr2.as_ptr() as *const ()));

    // Swap spaces
    nursery.swap_spaces();

    // ptr2 should now be in from-space
    assert!(nursery.in_from_space(ptr2.as_ptr() as *const ()));
}
