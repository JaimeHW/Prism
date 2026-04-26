use super::*;

#[test]
fn test_old_space_creation() {
    let old_space = OldSpace::new(64 * 1024, 16 * 1024);
    assert!(old_space.capacity() > 0);
    assert_eq!(old_space.usage(), 0);
}

#[test]
fn test_old_space_allocation() {
    let old_space = OldSpace::new(64 * 1024, 16 * 1024);

    let ptr = old_space.alloc(256).expect("Alloc failed");
    assert_eq!(old_space.usage(), 256);
    assert!(old_space.contains(ptr.as_ptr() as *const ()));
}

#[test]
fn test_old_space_multiple_blocks() {
    let old_space = OldSpace::new(1024, 512);

    // Fill first block
    for _ in 0..4 {
        old_space.alloc(128).expect("Alloc failed");
    }

    // This should allocate a new block
    let ptr = old_space.alloc(128).expect("Alloc in new block failed");
    assert!(old_space.contains(ptr.as_ptr() as *const ()));
}

#[test]
fn test_mark_block_live_preserves_allocated_block() {
    let mut old_space = OldSpace::new(1024, 512);
    let ptr = old_space.alloc(128).expect("Alloc failed");

    old_space.clear_live_counts();
    assert!(old_space.mark_block_live(ptr.as_ptr() as *const ()));
    let (bytes_freed, _) = old_space.sweep();

    assert_eq!(bytes_freed, 0);
    assert_eq!(old_space.usage(), 128);
}

#[test]
fn test_sweep_reclaims_unmarked_block() {
    let mut old_space = OldSpace::new(1024, 512);
    old_space.alloc(128).expect("Alloc failed");

    old_space.clear_live_counts();
    let (bytes_freed, _) = old_space.sweep();

    assert_eq!(bytes_freed, 128);
    assert_eq!(old_space.usage(), 0);
}
