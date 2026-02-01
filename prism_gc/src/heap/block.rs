//! Memory block management for the old generation.
//!
//! Blocks are the unit of allocation and sweeping in the old space.
//! Each block contains a header followed by object data.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Default block size (16KB).
pub const DEFAULT_BLOCK_SIZE: usize = 16 * 1024;

/// Minimum object size (8 bytes for alignment).
pub const MIN_OBJECT_SIZE: usize = 8;

/// Block state for sweep logic.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockState {
    /// Block is empty and available for allocation.
    Empty = 0,
    /// Block is partially filled.
    Partial = 1,
    /// Block is full.
    Full = 2,
    /// Block is being swept.
    Sweeping = 3,
}

/// Header at the beginning of each memory block.
#[repr(C)]
pub struct BlockHeader {
    /// Next block in the linked list.
    pub next: AtomicUsize,
    /// State of this block.
    pub state: AtomicU32,
    /// Bytes allocated in this block.
    pub allocated: AtomicUsize,
    /// Number of live objects after last GC.
    pub live_count: AtomicU32,
    /// Block size (excluding header).
    pub size: usize,
}

impl BlockHeader {
    /// Size of the block header.
    pub const SIZE: usize = std::mem::size_of::<BlockHeader>();

    /// Create a new block header.
    pub fn new(size: usize) -> Self {
        Self {
            next: AtomicUsize::new(0),
            state: AtomicU32::new(BlockState::Empty as u32),
            allocated: AtomicUsize::new(0),
            live_count: AtomicU32::new(0),
            size,
        }
    }

    /// Get the block state.
    #[inline]
    pub fn state(&self) -> BlockState {
        match self.state.load(Ordering::Relaxed) {
            0 => BlockState::Empty,
            1 => BlockState::Partial,
            2 => BlockState::Full,
            _ => BlockState::Sweeping,
        }
    }

    /// Set the block state.
    #[inline]
    pub fn set_state(&self, state: BlockState) {
        self.state.store(state as u32, Ordering::Relaxed);
    }

    /// Get bytes allocated.
    #[inline]
    pub fn allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Get remaining free space.
    #[inline]
    pub fn free_space(&self) -> usize {
        self.size.saturating_sub(self.allocated())
    }

    /// Check if block has room for an allocation.
    #[inline]
    pub fn can_alloc(&self, size: usize) -> bool {
        self.free_space() >= size
    }
}

/// A memory block in the old generation.
///
/// Blocks have a fixed size and contain a header followed by
/// object data. Objects are allocated linearly within a block
/// until it's full.
pub struct Block {
    /// Pointer to the start of the block (header).
    ptr: NonNull<u8>,
    /// Total size including header.
    total_size: usize,
}

impl Block {
    /// Allocate a new block with the given size.
    pub fn new(size: usize) -> Option<Self> {
        let total_size = BlockHeader::SIZE + size;

        // Allocate aligned memory
        let layout = std::alloc::Layout::from_size_align(total_size, 8).ok()?;
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };

        if ptr.is_null() {
            return None;
        }

        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        // Initialize header
        let header = unsafe { &mut *(ptr.as_ptr() as *mut BlockHeader) };
        *header = BlockHeader::new(size);

        Some(Self { ptr, total_size })
    }

    /// Get the block header.
    #[inline]
    pub fn header(&self) -> &BlockHeader {
        unsafe { &*(self.ptr.as_ptr() as *const BlockHeader) }
    }

    /// Get mutable header access.
    #[inline]
    pub fn header_mut(&mut self) -> &mut BlockHeader {
        unsafe { &mut *(self.ptr.as_ptr() as *mut BlockHeader) }
    }

    /// Get the data region start.
    #[inline]
    pub fn data_start(&self) -> *mut u8 {
        unsafe { self.ptr.as_ptr().add(BlockHeader::SIZE) }
    }

    /// Get the data region end.
    #[inline]
    pub fn data_end(&self) -> *mut u8 {
        unsafe { self.data_start().add(self.header().size) }
    }

    /// Check if a pointer is within this block's data region.
    #[inline]
    pub fn contains(&self, ptr: *const ()) -> bool {
        let addr = ptr as usize;
        let start = self.data_start() as usize;
        let end = self.data_end() as usize;
        addr >= start && addr < end
    }

    /// Allocate memory within this block.
    ///
    /// Returns None if there's not enough space.
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        let header = self.header();

        // Try to bump-allocate within block
        loop {
            let current = header.allocated.load(Ordering::Relaxed);
            let new_offset = current + size;

            if new_offset > header.size {
                // Update state to full
                header.set_state(BlockState::Full);
                return None;
            }

            // CAS to claim space
            if header
                .allocated
                .compare_exchange_weak(current, new_offset, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
            {
                let ptr = unsafe { self.data_start().add(current) };

                // Update state to partial if it was empty
                if current == 0 {
                    header.set_state(BlockState::Partial);
                }

                return NonNull::new(ptr);
            }
        }
    }

    /// Reset the block for reuse.
    pub fn reset(&mut self) {
        let header = self.header_mut();
        header.allocated.store(0, Ordering::Relaxed);
        header.live_count.store(0, Ordering::Relaxed);
        header.set_state(BlockState::Empty);

        // Zero the data region for safety
        let data_size = header.size;
        unsafe {
            std::ptr::write_bytes(self.data_start(), 0, data_size);
        }
    }

    /// Get the usable size of this block.
    #[inline]
    pub fn size(&self) -> usize {
        self.header().size
    }

    /// Get total size including header.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if let Ok(layout) = std::alloc::Layout::from_size_align(self.total_size, 8) {
            unsafe {
                std::alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

// Safety: Block can be sent between threads.
unsafe impl Send for Block {}

#[cfg(test)]
mod tests {
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
}
