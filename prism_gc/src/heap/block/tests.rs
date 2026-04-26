use super::*;

#[test]
fn test_block_creation() {
    let block = Block::new(1024).expect("Failed to create block");
    assert_eq!(block.size(), 1024);
    assert_eq!(block.header().state(), BlockState::Empty);
    assert_eq!(block.header().allocated(), 0);
}

#[test]
fn test_block_allocation() {
    let block = Block::new(1024).expect("Failed to create block");

    let ptr1 = block.alloc(64).expect("First alloc failed");
    assert_eq!(block.header().allocated(), 64);
    assert_eq!(block.header().state(), BlockState::Partial);

    let ptr2 = block.alloc(64).expect("Second alloc failed");
    assert_eq!(block.header().allocated(), 128);
    assert_ne!(ptr1, ptr2);
}

#[test]
fn test_block_full() {
    let block = Block::new(128).expect("Failed to create block");

    // Fill the block
    let _ = block.alloc(64);
    let _ = block.alloc(64);

    // Should fail - block is full
    assert!(block.alloc(1).is_none());
    assert_eq!(block.header().state(), BlockState::Full);
}

#[test]
fn test_block_contains() {
    let block = Block::new(1024).expect("Failed to create block");
    let ptr = block.alloc(64).expect("Alloc failed");

    assert!(block.contains(ptr.as_ptr() as *const ()));
    assert!(!block.contains(std::ptr::null()));
}

#[test]
fn test_block_reset() {
    let mut block = Block::new(1024).expect("Failed to create block");

    let _ = block.alloc(64);
    assert_eq!(block.header().allocated(), 64);

    block.reset();
    assert_eq!(block.header().allocated(), 0);
    assert_eq!(block.header().state(), BlockState::Empty);
}
