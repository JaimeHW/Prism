//! Old space (tenured generation) with block-based allocation.
//!
//! The old space contains long-lived objects that have survived
//! multiple minor GC cycles. It uses a block-based allocator
//! with free-list management.

use super::block::{Block, BlockState};

use parking_lot::Mutex;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Old space (tenured generation) with block-based mark-sweep.
pub struct OldSpace {
    /// Blocks in the old space.
    blocks: Mutex<Vec<Block>>,
    /// Current allocation block index.
    current_block: AtomicUsize,
    /// Total capacity in bytes.
    capacity: AtomicUsize,
    /// Total bytes allocated.
    allocated: AtomicUsize,
    /// Block size for new blocks.
    block_size: usize,
}

impl OldSpace {
    /// Create a new old space with initial capacity.
    pub fn new(initial_capacity: usize, block_size: usize) -> Self {
        let mut blocks = Vec::new();
        let mut total_capacity = 0;

        // Pre-allocate initial blocks
        let num_blocks = initial_capacity / block_size;
        for _ in 0..num_blocks {
            if let Some(block) = Block::new(block_size) {
                total_capacity += block.size();
                blocks.push(block);
            }
        }

        Self {
            blocks: Mutex::new(blocks),
            current_block: AtomicUsize::new(0),
            capacity: AtomicUsize::new(total_capacity),
            allocated: AtomicUsize::new(0),
            block_size,
        }
    }

    /// Allocate memory in the old space.
    pub fn alloc(&self, size: usize) -> Option<NonNull<u8>> {
        // Try current block first
        let current = self.current_block.load(Ordering::Relaxed);

        {
            let blocks = self.blocks.lock();
            if current < blocks.len() {
                if let Some(ptr) = blocks[current].alloc(size) {
                    self.allocated.fetch_add(size, Ordering::Relaxed);
                    return Some(ptr);
                }
            }
        }

        // Current block is full, try other partial blocks
        self.alloc_slow(size)
    }

    /// Slow path: search for a partial block or allocate new one.
    fn alloc_slow(&self, size: usize) -> Option<NonNull<u8>> {
        let mut blocks = self.blocks.lock();

        // Try to find a partial block with space
        for (i, block) in blocks.iter().enumerate() {
            if block.header().state() != BlockState::Full {
                if let Some(ptr) = block.alloc(size) {
                    self.current_block.store(i, Ordering::Relaxed);
                    self.allocated.fetch_add(size, Ordering::Relaxed);
                    return Some(ptr);
                }
            }
        }

        // No space, allocate a new block
        let new_block = Block::new(self.block_size.max(size))?;
        let ptr = new_block.alloc(size)?;

        let new_index = blocks.len();
        self.capacity.fetch_add(new_block.size(), Ordering::Relaxed);
        blocks.push(new_block);

        self.current_block.store(new_index, Ordering::Relaxed);
        self.allocated.fetch_add(size, Ordering::Relaxed);

        Some(ptr)
    }

    /// Check if a pointer is in the old space.
    pub fn contains(&self, ptr: *const ()) -> bool {
        let blocks = self.blocks.lock();
        blocks.iter().any(|block| block.contains(ptr))
    }

    /// Get total capacity in bytes.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// Get total bytes allocated.
    #[inline]
    pub fn usage(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    /// Get number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.lock().len()
    }

    /// Iterate over all blocks for sweeping.
    pub fn for_each_block<F>(&self, mut f: F)
    where
        F: FnMut(&Block),
    {
        let blocks = self.blocks.lock();
        for block in blocks.iter() {
            f(block);
        }
    }

    /// Iterate over all blocks mutably.
    pub fn for_each_block_mut<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Block),
    {
        let blocks = self.blocks.get_mut();
        for block in blocks.iter_mut() {
            f(block);
        }
    }

    /// Reset allocation tracking after GC.
    pub fn reset_allocated(&self, live_bytes: usize) {
        self.allocated.store(live_bytes, Ordering::Relaxed);
    }

    /// Sweep dead objects and reclaim space.
    ///
    /// Returns (bytes_freed, objects_freed).
    pub fn sweep(&mut self) -> (usize, usize) {
        let blocks = self.blocks.get_mut();
        let mut bytes_freed = 0;
        let objects_freed = 0;

        for block in blocks.iter_mut() {
            // Mark blocks with no live objects as empty
            if block.header().live_count.load(Ordering::Relaxed) == 0 {
                bytes_freed += block.header().allocated();
                block.reset();
            }
        }

        // Update allocated count
        let old_allocated = self.allocated.load(Ordering::Relaxed);
        self.allocated
            .store(old_allocated.saturating_sub(bytes_freed), Ordering::Relaxed);

        (bytes_freed, objects_freed)
    }

    /// Compact the old space by removing empty blocks.
    pub fn compact_blocks(&mut self) {
        let blocks = self.blocks.get_mut();

        // Count how many blocks we have before filtering
        let initial_len = blocks.len();

        // Remove completely empty blocks (keep at least one)
        if initial_len > 1 {
            let mut freed_capacity = 0;
            let mut kept = 0;

            blocks.retain(|block| {
                // Keep if: not empty, or we need to keep at least one block
                if block.header().state() != BlockState::Empty || kept == 0 {
                    kept += 1;
                    true
                } else {
                    freed_capacity += block.size();
                    false
                }
            });

            self.capacity.fetch_sub(freed_capacity, Ordering::Relaxed);
        }

        // Reset current block index
        self.current_block.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
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
}
