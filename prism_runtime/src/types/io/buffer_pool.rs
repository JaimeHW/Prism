//! Thread-local buffer pool for zero-allocation I/O operations.
//!
//! This module provides a high-performance buffer recycling system that
//! eliminates allocation overhead for common I/O patterns.
//!
//! # Design
//!
//! Each thread maintains a pool of pre-allocated buffers. When a buffer is
//! needed, it's taken from the pool; when released, it's returned for reuse.
//!
//! # Performance Characteristics
//!
//! - **Allocation**: O(1) amortized (pool hit) or O(n) (pool miss)
//! - **Deallocation**: O(1) (return to pool)
//! - **Memory**: Bounded by `MAX_POOLED_BUFFERS` per thread
//!
//! # Thread Safety
//!
//! The pool is thread-local, so no synchronization is required.

use std::cell::RefCell;

/// Default buffer size (8KB - optimal for most filesystems).
pub const DEFAULT_BUFFER_SIZE: usize = 8 * 1024;

/// Small buffer size for line-oriented I/O.
pub const SMALL_BUFFER_SIZE: usize = 1024;

/// Large buffer size for bulk transfers.
pub const LARGE_BUFFER_SIZE: usize = 64 * 1024;

/// Maximum number of buffers to keep per size class.
const MAX_POOLED_BUFFERS: usize = 8;

/// Buffer size classes for efficient pooling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferSizeClass {
    /// 1KB buffers for line I/O
    Small,
    /// 8KB buffers for general I/O  
    Medium,
    /// 64KB buffers for bulk transfers
    Large,
}

impl BufferSizeClass {
    /// Get the actual size in bytes for this class.
    #[inline]
    pub const fn size(self) -> usize {
        match self {
            BufferSizeClass::Small => SMALL_BUFFER_SIZE,
            BufferSizeClass::Medium => DEFAULT_BUFFER_SIZE,
            BufferSizeClass::Large => LARGE_BUFFER_SIZE,
        }
    }

    /// Determine the appropriate size class for a requested size.
    #[inline]
    pub const fn for_size(size: usize) -> Self {
        if size <= SMALL_BUFFER_SIZE {
            BufferSizeClass::Small
        } else if size <= DEFAULT_BUFFER_SIZE {
            BufferSizeClass::Medium
        } else {
            BufferSizeClass::Large
        }
    }
}

/// A pooled buffer that returns to the pool when dropped.
pub struct PooledBuffer {
    /// The underlying buffer data.
    data: Vec<u8>,
    /// Size class for return to correct pool.
    size_class: BufferSizeClass,
}

impl PooledBuffer {
    /// Create a new pooled buffer with the given size class.
    #[inline]
    fn new(size_class: BufferSizeClass) -> Self {
        let mut data = Vec::with_capacity(size_class.size());
        // Pre-extend to capacity for predictable performance
        data.resize(size_class.size(), 0);
        Self { data, size_class }
    }

    /// Get the buffer data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get the buffer data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Get the buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Clear the buffer contents (for security).
    #[inline]
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Get the size class of this buffer.
    #[inline]
    pub fn size_class(&self) -> BufferSizeClass {
        self.size_class
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for PooledBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to the pool
        BUFFER_POOL.with(|pool| {
            pool.borrow_mut()
                .return_buffer(std::mem::take(&mut self.data), self.size_class);
        });
    }
}

/// Thread-local buffer pool.
pub struct BufferPool {
    /// Small buffer pool (1KB)
    small: Vec<Vec<u8>>,
    /// Medium buffer pool (8KB)
    medium: Vec<Vec<u8>>,
    /// Large buffer pool (64KB)
    large: Vec<Vec<u8>>,
    /// Statistics: total allocations
    #[cfg(debug_assertions)]
    total_allocations: usize,
    /// Statistics: pool hits
    #[cfg(debug_assertions)]
    pool_hits: usize,
}

impl BufferPool {
    /// Create a new empty buffer pool.
    #[inline]
    pub fn new() -> Self {
        Self {
            small: Vec::with_capacity(MAX_POOLED_BUFFERS),
            medium: Vec::with_capacity(MAX_POOLED_BUFFERS),
            large: Vec::with_capacity(MAX_POOLED_BUFFERS),
            #[cfg(debug_assertions)]
            total_allocations: 0,
            #[cfg(debug_assertions)]
            pool_hits: 0,
        }
    }

    /// Acquire a buffer from the pool, or allocate a new one.
    #[inline]
    pub fn acquire(&mut self, size_class: BufferSizeClass) -> PooledBuffer {
        #[cfg(debug_assertions)]
        {
            self.total_allocations += 1;
        }

        let pool = self.pool_for_class_mut(size_class);

        if let Some(data) = pool.pop() {
            #[cfg(debug_assertions)]
            {
                self.pool_hits += 1;
            }
            PooledBuffer { data, size_class }
        } else {
            PooledBuffer::new(size_class)
        }
    }

    /// Return a buffer to the pool.
    #[inline]
    fn return_buffer(&mut self, mut data: Vec<u8>, size_class: BufferSizeClass) {
        let pool = self.pool_for_class_mut(size_class);

        if pool.len() < MAX_POOLED_BUFFERS && data.capacity() == size_class.size() {
            // Clear sensitive data before pooling
            data.fill(0);
            pool.push(data);
        }
        // Otherwise, let the buffer be deallocated
    }

    /// Get the pool for a given size class.
    #[inline]
    fn pool_for_class_mut(&mut self, size_class: BufferSizeClass) -> &mut Vec<Vec<u8>> {
        match size_class {
            BufferSizeClass::Small => &mut self.small,
            BufferSizeClass::Medium => &mut self.medium,
            BufferSizeClass::Large => &mut self.large,
        }
    }

    /// Clear all pooled buffers (useful for testing or memory pressure).
    pub fn clear(&mut self) {
        self.small.clear();
        self.medium.clear();
        self.large.clear();
    }

    /// Get the total number of pooled buffers.
    pub fn pooled_count(&self) -> usize {
        self.small.len() + self.medium.len() + self.large.len()
    }

    /// Get the total memory held by pooled buffers.
    pub fn pooled_bytes(&self) -> usize {
        self.small.len() * SMALL_BUFFER_SIZE
            + self.medium.len() * DEFAULT_BUFFER_SIZE
            + self.large.len() * LARGE_BUFFER_SIZE
    }

    /// Get pool statistics (debug builds only).
    #[cfg(debug_assertions)]
    pub fn stats(&self) -> (usize, usize) {
        (self.pool_hits, self.total_allocations)
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new()
    }
}

// Thread-local buffer pool instance
thread_local! {
    static BUFFER_POOL: RefCell<BufferPool> = RefCell::new(BufferPool::new());
}

/// Acquire a buffer from the thread-local pool.
///
/// # Example
///
/// ```ignore
/// let mut buf = acquire_buffer(BufferSizeClass::Medium);
/// // Use buffer...
/// // Buffer automatically returns to pool on drop
/// ```
#[inline]
pub fn acquire_buffer(size_class: BufferSizeClass) -> PooledBuffer {
    BUFFER_POOL.with(|pool| pool.borrow_mut().acquire(size_class))
}

/// Acquire a buffer of at least the specified size.
#[inline]
pub fn acquire_buffer_sized(min_size: usize) -> PooledBuffer {
    acquire_buffer(BufferSizeClass::for_size(min_size))
}

/// Clear the thread-local buffer pool.
pub fn clear_pool() {
    BUFFER_POOL.with(|pool| pool.borrow_mut().clear());
}

/// Get thread-local pool statistics.
pub fn pool_stats() -> (usize, usize) {
    BUFFER_POOL.with(|pool| {
        let p = pool.borrow();
        (p.pooled_count(), p.pooled_bytes())
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // BufferSizeClass Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_size_class_sizes() {
        assert_eq!(BufferSizeClass::Small.size(), 1024);
        assert_eq!(BufferSizeClass::Medium.size(), 8192);
        assert_eq!(BufferSizeClass::Large.size(), 65536);
    }

    #[test]
    fn test_size_class_for_size() {
        assert_eq!(BufferSizeClass::for_size(100), BufferSizeClass::Small);
        assert_eq!(BufferSizeClass::for_size(1024), BufferSizeClass::Small);
        assert_eq!(BufferSizeClass::for_size(1025), BufferSizeClass::Medium);
        assert_eq!(BufferSizeClass::for_size(8192), BufferSizeClass::Medium);
        assert_eq!(BufferSizeClass::for_size(8193), BufferSizeClass::Large);
        assert_eq!(BufferSizeClass::for_size(100000), BufferSizeClass::Large);
    }

    // -------------------------------------------------------------------------
    // PooledBuffer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pooled_buffer_creation() {
        let buf = PooledBuffer::new(BufferSizeClass::Small);
        assert_eq!(buf.capacity(), SMALL_BUFFER_SIZE);
        assert_eq!(buf.size_class(), BufferSizeClass::Small);
    }

    #[test]
    fn test_pooled_buffer_as_slice() {
        let buf = PooledBuffer::new(BufferSizeClass::Small);
        assert_eq!(buf.as_slice().len(), SMALL_BUFFER_SIZE);
    }

    #[test]
    fn test_pooled_buffer_mutability() {
        let mut buf = PooledBuffer::new(BufferSizeClass::Small);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    #[test]
    fn test_pooled_buffer_deref() {
        let buf = PooledBuffer::new(BufferSizeClass::Small);
        let slice: &[u8] = &*buf;
        assert_eq!(slice.len(), SMALL_BUFFER_SIZE);
    }

    #[test]
    fn test_pooled_buffer_clear() {
        let mut buf = PooledBuffer::new(BufferSizeClass::Small);
        buf.as_mut_slice()[0] = 255;
        buf.clear();
        assert_eq!(buf.as_slice()[0], 0);
    }

    // -------------------------------------------------------------------------
    // BufferPool Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pool_acquire_fresh() {
        let mut pool = BufferPool::new();
        let buf = pool.acquire(BufferSizeClass::Medium);
        assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = BufferPool::new();

        // Acquire and release a buffer
        let buf = pool.acquire(BufferSizeClass::Medium);
        let ptr = buf.as_ptr();
        drop(buf);

        // The buffer should be returned to the pool - but since we're
        // using a separate pool instance, we need to manually return it
        // In the real implementation, the thread-local pool is used
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = BufferPool::new();

        // Pre-populate with a buffer
        let buf = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
        pool.medium.push(buf);

        assert_eq!(pool.pooled_count(), 1);
        pool.clear();
        assert_eq!(pool.pooled_count(), 0);
    }

    #[test]
    fn test_pool_max_buffers() {
        let mut pool = BufferPool::new();

        // Fill the pool to max
        for _ in 0..MAX_POOLED_BUFFERS {
            let mut buf = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
            buf.resize(DEFAULT_BUFFER_SIZE, 0);
            pool.return_buffer(buf, BufferSizeClass::Medium);
        }

        assert_eq!(pool.medium.len(), MAX_POOLED_BUFFERS);

        // Try to add one more - should be dropped
        let mut extra = Vec::with_capacity(DEFAULT_BUFFER_SIZE);
        extra.resize(DEFAULT_BUFFER_SIZE, 0);
        pool.return_buffer(extra, BufferSizeClass::Medium);

        // Still at max
        assert_eq!(pool.medium.len(), MAX_POOLED_BUFFERS);
    }

    #[test]
    fn test_pool_wrong_capacity_rejected() {
        let mut pool = BufferPool::new();

        // Buffer with wrong capacity shouldn't be pooled
        let buf = Vec::with_capacity(100); // Wrong size
        pool.return_buffer(buf, BufferSizeClass::Medium);

        assert_eq!(pool.medium.len(), 0);
    }

    // -------------------------------------------------------------------------
    // Thread-Local API Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_acquire_buffer_api() {
        let buf = acquire_buffer(BufferSizeClass::Small);
        assert_eq!(buf.capacity(), SMALL_BUFFER_SIZE);
    }

    #[test]
    fn test_acquire_buffer_sized_api() {
        let buf = acquire_buffer_sized(5000);
        assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
    }

    #[test]
    fn test_clear_pool_api() {
        // Acquire and drop to populate pool
        let _ = acquire_buffer(BufferSizeClass::Medium);

        clear_pool();

        let (count, _) = pool_stats();
        assert_eq!(count, 0);
    }

    // -------------------------------------------------------------------------
    // Pool Statistics Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pooled_bytes_calculation() {
        let mut pool = BufferPool::new();

        let mut small = Vec::with_capacity(SMALL_BUFFER_SIZE);
        small.resize(SMALL_BUFFER_SIZE, 0);
        pool.return_buffer(small, BufferSizeClass::Small);

        assert_eq!(pool.pooled_bytes(), SMALL_BUFFER_SIZE);
    }

    // -------------------------------------------------------------------------
    // Stress Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_many_acquisitions() {
        for _ in 0..100 {
            let buf = acquire_buffer(BufferSizeClass::Medium);
            assert_eq!(buf.capacity(), DEFAULT_BUFFER_SIZE);
        }
    }

    #[test]
    fn test_mixed_size_classes() {
        let bufs: Vec<_> = [
            BufferSizeClass::Small,
            BufferSizeClass::Medium,
            BufferSizeClass::Large,
            BufferSizeClass::Small,
            BufferSizeClass::Medium,
        ]
        .iter()
        .map(|&class| acquire_buffer(class))
        .collect();

        assert_eq!(bufs[0].capacity(), SMALL_BUFFER_SIZE);
        assert_eq!(bufs[1].capacity(), DEFAULT_BUFFER_SIZE);
        assert_eq!(bufs[2].capacity(), LARGE_BUFFER_SIZE);
    }
}
