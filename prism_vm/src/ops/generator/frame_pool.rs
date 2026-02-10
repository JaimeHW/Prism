//! Generator frame pool for high-performance suspension/resumption.
//!
//! This module provides arena-based frame storage allocation for generators,
//! eliminating heap allocations during yield/resume cycles.
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        GeneratorFramePool                                │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Small Frames (≤8 registers)          Large Frames (>8 registers)       │
//! │  ┌────────────────────────┐           ┌────────────────────────┐        │
//! │  │ InlineFrameStorage[16] │           │ HeapFrameStorage[8]    │        │
//! │  │ - 64 bytes inline      │           │ - Vec<Value> backed    │        │
//! │  │ - O(1) alloc/free      │           │ - Variable size        │        │
//! │  └────────────────────────┘           └────────────────────────┘        │
//! │                                                                          │
//! │  Free Lists:                                                            │
//! │  small_frames ──▶ [Box<Inline>, Box<Inline>, ...]                       │
//! │  large_frames ──▶ [Box<Heap>, Box<Heap>, ...]                           │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | Pool Hit | Pool Miss |
//! |-----------|----------|-----------|
//! | Allocate | O(1) pop | O(1) Box::new |
//! | Release | O(1) push | N/A |
//! | Memory | 0 bytes | 64-512 bytes |
//!
//! # Thread Safety
//!
//! The pool is NOT thread-safe. Each VM instance owns its own pool.
//! For multi-threaded scenarios, use thread-local pools.

use prism_core::Value;
use smallvec::SmallVec;
use std::mem::MaybeUninit;

// =============================================================================
// Constants
// =============================================================================

/// Maximum registers stored inline (fits in cache line).
const INLINE_REGISTER_CAPACITY: usize = 8;

/// Initial small frame pool capacity.
const INITIAL_SMALL_POOL_SIZE: usize = 16;

/// Initial large frame pool capacity.
const INITIAL_LARGE_POOL_SIZE: usize = 8;

/// Maximum frames to retain in pool (prevent unbounded growth).
const MAX_SMALL_POOL_SIZE: usize = 64;
const MAX_LARGE_POOL_SIZE: usize = 32;

// =============================================================================
// Frame Storage Types
// =============================================================================

/// Inline frame storage for small generators (≤8 registers).
///
/// Uses a fixed-size array to avoid heap allocation.
/// Fits in a cache line for optimal access patterns.
#[repr(C, align(64))] // Cache-line aligned
pub struct InlineFrameStorage {
    /// Register values (8 x 8 bytes = 64 bytes).
    registers: [MaybeUninit<Value>; INLINE_REGISTER_CAPACITY],
    /// Number of registers actually stored.
    count: u8,
    /// Liveness bitmap for stored registers.
    liveness: u64,
    /// Resume point index.
    resume_index: u32,
    /// Padding to fill cache line.
    _pad: [u8; 3],
}

/// Heap frame storage for large generators (>8 registers).
///
/// Uses a Vec for variable-size storage.
pub struct HeapFrameStorage {
    /// Register values.
    registers: Vec<Value>,
    /// Liveness bitmap.
    liveness: u64,
    /// Resume point index.
    resume_index: u32,
}

/// Unified frame handle returned by the pool.
///
/// Transparently handles both inline and heap storage.
pub enum PooledFrame {
    /// Inline storage (small generators).
    Inline(Box<InlineFrameStorage>),
    /// Heap storage (large generators).
    Heap(Box<HeapFrameStorage>),
}

// =============================================================================
// Pool Statistics
// =============================================================================

/// Statistics for monitoring pool behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Total frames allocated.
    pub allocations: u64,
    /// Total frames released back to pool.
    pub releases: u64,
    /// Allocations served from pool (cache hit).
    pub pool_hits: u64,
    /// Allocations requiring new Box (cache miss).
    pub pool_misses: u64,
    /// Heap allocations (Box::new calls).
    pub heap_allocations: u64,
    /// Frames evicted due to pool size limit.
    pub evictions: u64,
    /// Current small pool size.
    pub small_pool_size: usize,
    /// Current large pool size.
    pub large_pool_size: usize,
}

// =============================================================================
// Generator Frame Pool
// =============================================================================

/// High-performance frame pool for generator suspension.
///
/// Maintains separate free lists for small and large frames,
/// with inline storage optimization for common case.
pub struct GeneratorFramePool {
    /// Free list of small frame storage.
    small_frames: SmallVec<[Box<InlineFrameStorage>; INITIAL_SMALL_POOL_SIZE]>,
    /// Free list of large frame storage.
    large_frames: SmallVec<[Box<HeapFrameStorage>; INITIAL_LARGE_POOL_SIZE]>,
    /// Pool statistics.
    stats: PoolStats,
}

// =============================================================================
// InlineFrameStorage Implementation
// =============================================================================

impl InlineFrameStorage {
    /// Create new zeroed inline storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            registers: [const { MaybeUninit::uninit() }; INLINE_REGISTER_CAPACITY],
            count: 0,
            liveness: 0,
            resume_index: 0,
            _pad: [0; 3],
        }
    }

    /// Get the capacity.
    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        INLINE_REGISTER_CAPACITY
    }

    /// Get the number of stored registers.
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.count as usize
    }

    /// Get the liveness bitmap.
    #[inline(always)]
    pub fn liveness(&self) -> u64 {
        self.liveness
    }

    /// Get the resume index.
    #[inline(always)]
    pub fn resume_index(&self) -> u32 {
        self.resume_index
    }

    /// Set metadata.
    #[inline]
    pub fn set_metadata(&mut self, liveness: u64, resume_index: u32) {
        self.liveness = liveness;
        self.resume_index = resume_index;
    }

    /// Store a register value.
    ///
    /// # Safety
    /// Caller must ensure index < INLINE_REGISTER_CAPACITY.
    #[inline]
    pub fn store(&mut self, index: usize, value: Value) {
        debug_assert!(index < INLINE_REGISTER_CAPACITY);
        self.registers[index].write(value);
        if index >= self.count as usize {
            self.count = (index + 1) as u8;
        }
    }

    /// Load a register value.
    ///
    /// # Safety
    /// Caller must ensure index < count and the slot was previously stored.
    #[inline]
    pub unsafe fn load(&self, index: usize) -> Value {
        debug_assert!(index < self.count as usize);
        // SAFETY: Caller guarantees index < count and the slot was initialized via store()
        unsafe { self.registers[index].assume_init_read() }
    }

    /// Clear storage for reuse.
    #[inline]
    pub fn clear(&mut self) {
        // Drop any stored values
        for i in 0..self.count as usize {
            unsafe {
                std::ptr::drop_in_place(self.registers[i].as_mut_ptr());
            }
        }
        self.count = 0;
        self.liveness = 0;
        self.resume_index = 0;
    }
}

impl Default for InlineFrameStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for InlineFrameStorage {
    fn drop(&mut self) {
        // Ensure all stored values are dropped
        for i in 0..self.count as usize {
            unsafe {
                std::ptr::drop_in_place(self.registers[i].as_mut_ptr());
            }
        }
    }
}

// =============================================================================
// HeapFrameStorage Implementation
// =============================================================================

impl HeapFrameStorage {
    /// Create new heap storage with given capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            registers: Vec::with_capacity(capacity),
            liveness: 0,
            resume_index: 0,
        }
    }

    /// Get the capacity.
    #[inline(always)]
    pub fn capacity(&self) -> usize {
        self.registers.capacity()
    }

    /// Get the number of stored registers.
    #[inline(always)]
    pub fn count(&self) -> usize {
        self.registers.len()
    }

    /// Get the liveness bitmap.
    #[inline(always)]
    pub fn liveness(&self) -> u64 {
        self.liveness
    }

    /// Get the resume index.
    #[inline(always)]
    pub fn resume_index(&self) -> u32 {
        self.resume_index
    }

    /// Set metadata.
    #[inline]
    pub fn set_metadata(&mut self, liveness: u64, resume_index: u32) {
        self.liveness = liveness;
        self.resume_index = resume_index;
    }

    /// Store a register value.
    #[inline]
    pub fn store(&mut self, value: Value) {
        self.registers.push(value);
    }

    /// Load a register value.
    #[inline]
    pub fn load(&self, index: usize) -> Option<&Value> {
        self.registers.get(index)
    }

    /// Get registers slice.
    #[inline(always)]
    pub fn registers(&self) -> &[Value] {
        &self.registers
    }

    /// Clear storage for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.registers.clear();
        self.liveness = 0;
        self.resume_index = 0;
    }

    /// Ensure capacity for a given size.
    #[inline]
    pub fn ensure_capacity(&mut self, capacity: usize) {
        if self.registers.capacity() < capacity {
            // Reserve additional capacity to reach the target
            let additional = capacity - self.registers.len();
            self.registers.reserve(additional);
        }
    }
}

// =============================================================================
// PooledFrame Implementation
// =============================================================================

impl PooledFrame {
    /// Get the capacity of this frame.
    #[inline]
    pub fn capacity(&self) -> usize {
        match self {
            PooledFrame::Inline(frame) => frame.capacity(),
            PooledFrame::Heap(frame) => frame.capacity(),
        }
    }

    /// Get the number of stored registers.
    #[inline]
    pub fn count(&self) -> usize {
        match self {
            PooledFrame::Inline(frame) => frame.count(),
            PooledFrame::Heap(frame) => frame.count(),
        }
    }

    /// Get the liveness bitmap.
    #[inline]
    pub fn liveness(&self) -> u64 {
        match self {
            PooledFrame::Inline(frame) => frame.liveness(),
            PooledFrame::Heap(frame) => frame.liveness(),
        }
    }

    /// Get the resume index.
    #[inline]
    pub fn resume_index(&self) -> u32 {
        match self {
            PooledFrame::Inline(frame) => frame.resume_index(),
            PooledFrame::Heap(frame) => frame.resume_index(),
        }
    }

    /// Check if this is inline storage.
    #[inline(always)]
    pub fn is_inline(&self) -> bool {
        matches!(self, PooledFrame::Inline(_))
    }

    /// Get raw pointer for identity comparison.
    #[inline]
    pub fn as_ptr(&self) -> *const () {
        match self {
            PooledFrame::Inline(frame) => frame.as_ref() as *const _ as *const (),
            PooledFrame::Heap(frame) => frame.as_ref() as *const _ as *const (),
        }
    }

    /// Set metadata (liveness and resume index).
    #[inline]
    pub fn set_metadata(&mut self, liveness: u64, resume_index: u32) {
        match self {
            PooledFrame::Inline(frame) => frame.set_metadata(liveness, resume_index),
            PooledFrame::Heap(frame) => frame.set_metadata(liveness, resume_index),
        }
    }

    /// Store registers from a slice.
    ///
    /// Only stores registers marked as live in the liveness bitmap.
    pub fn store_registers(&mut self, registers: &[Value], liveness: u64) {
        match self {
            PooledFrame::Inline(frame) => {
                let mut stored = 0;
                for (i, reg) in registers.iter().enumerate() {
                    if i >= INLINE_REGISTER_CAPACITY {
                        break;
                    }
                    if (liveness >> i) & 1 == 1 {
                        frame.store(stored, reg.clone());
                        stored += 1;
                    }
                }
            }
            PooledFrame::Heap(frame) => {
                for (i, reg) in registers.iter().enumerate() {
                    if (liveness >> i) & 1 == 1 {
                        frame.store(reg.clone());
                    }
                }
            }
        }
    }

    /// Restore registers to a slice.
    ///
    /// Restores only registers that were marked as live.
    pub fn restore_registers(&self, registers: &mut [Value], liveness: u64) {
        match self {
            PooledFrame::Inline(frame) => {
                let mut loaded = 0;
                for i in 0..registers.len() {
                    if (liveness >> i) & 1 == 1 {
                        if loaded < frame.count() {
                            // SAFETY: We only load up to count, and store guarantees initialization
                            registers[i] = unsafe { frame.load(loaded) };
                            loaded += 1;
                        }
                    }
                }
            }
            PooledFrame::Heap(frame) => {
                let mut loaded = 0;
                for i in 0..registers.len() {
                    if (liveness >> i) & 1 == 1 {
                        if let Some(val) = frame.load(loaded) {
                            registers[i] = val.clone();
                            loaded += 1;
                        }
                    }
                }
            }
        }
    }

    /// Clear the frame for reuse.
    pub fn clear(&mut self) {
        match self {
            PooledFrame::Inline(frame) => frame.clear(),
            PooledFrame::Heap(frame) => frame.clear(),
        }
    }
}

// =============================================================================
// GeneratorFramePool Implementation
// =============================================================================

impl GeneratorFramePool {
    /// Create a new empty frame pool.
    #[inline]
    pub fn new() -> Self {
        Self {
            small_frames: SmallVec::new(),
            large_frames: SmallVec::new(),
            stats: PoolStats::default(),
        }
    }

    /// Create a pool with pre-allocated capacity.
    pub fn with_capacity(small: usize, large: usize) -> Self {
        let mut pool = Self::new();

        // Pre-allocate small frames
        for _ in 0..small.min(MAX_SMALL_POOL_SIZE) {
            pool.small_frames.push(Box::new(InlineFrameStorage::new()));
            pool.stats.heap_allocations += 1;
        }
        pool.stats.small_pool_size = pool.small_frames.len();

        // Pre-allocate large frames
        for _ in 0..large.min(MAX_LARGE_POOL_SIZE) {
            pool.large_frames
                .push(Box::new(HeapFrameStorage::with_capacity(32)));
            pool.stats.heap_allocations += 1;
        }
        pool.stats.large_pool_size = pool.large_frames.len();

        pool
    }

    /// Allocate a frame with at least the given register capacity.
    #[inline]
    pub fn allocate(&mut self, register_count: usize) -> PooledFrame {
        self.stats.allocations += 1;

        if register_count <= INLINE_REGISTER_CAPACITY {
            self.allocate_small()
        } else {
            self.allocate_large(register_count)
        }
    }

    /// Allocate a small (inline) frame.
    #[inline]
    fn allocate_small(&mut self) -> PooledFrame {
        if let Some(mut frame) = self.small_frames.pop() {
            self.stats.pool_hits += 1;
            self.stats.small_pool_size = self.small_frames.len();
            frame.clear();
            PooledFrame::Inline(frame)
        } else {
            self.stats.pool_misses += 1;
            self.stats.heap_allocations += 1;
            PooledFrame::Inline(Box::new(InlineFrameStorage::new()))
        }
    }

    /// Allocate a large (heap) frame.
    #[inline]
    fn allocate_large(&mut self, capacity: usize) -> PooledFrame {
        if let Some(mut frame) = self.large_frames.pop() {
            self.stats.pool_hits += 1;
            self.stats.large_pool_size = self.large_frames.len();
            frame.clear();
            frame.ensure_capacity(capacity);
            PooledFrame::Heap(frame)
        } else {
            self.stats.pool_misses += 1;
            self.stats.heap_allocations += 1;
            PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(capacity)))
        }
    }

    /// Release a frame back to the pool.
    #[inline]
    pub fn release(&mut self, mut frame: PooledFrame) {
        self.stats.releases += 1;

        match frame {
            PooledFrame::Inline(inline) => {
                if self.small_frames.len() < MAX_SMALL_POOL_SIZE {
                    self.small_frames.push(inline);
                    self.stats.small_pool_size = self.small_frames.len();
                } else {
                    self.stats.evictions += 1;
                    // Frame is dropped
                }
            }
            PooledFrame::Heap(heap) => {
                if self.large_frames.len() < MAX_LARGE_POOL_SIZE {
                    self.large_frames.push(heap);
                    self.stats.large_pool_size = self.large_frames.len();
                } else {
                    self.stats.evictions += 1;
                    // Frame is dropped
                }
            }
        }
    }

    /// Get pool statistics.
    #[inline]
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            small_pool_size: self.small_frames.len(),
            large_pool_size: self.large_frames.len(),
            ..self.stats
        }
    }

    /// Clear the pool, dropping all cached frames.
    pub fn clear(&mut self) {
        self.small_frames.clear();
        self.large_frames.clear();
        self.stats.small_pool_size = 0;
        self.stats.large_pool_size = 0;
    }

    /// Get current pool size (small + large).
    #[inline]
    pub fn len(&self) -> usize {
        self.small_frames.len() + self.large_frames.len()
    }

    /// Check if pool is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.small_frames.is_empty() && self.large_frames.is_empty()
    }
}

impl Default for GeneratorFramePool {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // InlineFrameStorage Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_inline_storage_new() {
        let storage = InlineFrameStorage::new();
        assert_eq!(storage.count(), 0);
        assert_eq!(storage.liveness(), 0);
        assert_eq!(storage.resume_index(), 0);
        assert_eq!(storage.capacity(), INLINE_REGISTER_CAPACITY);
    }

    #[test]
    fn test_inline_storage_store_load() {
        let mut storage = InlineFrameStorage::new();

        storage.store(0, Value::int(42).unwrap());
        storage.store(1, Value::bool(true));
        storage.store(2, Value::none());

        assert_eq!(storage.count(), 3);

        unsafe {
            assert_eq!(storage.load(0).as_int(), Some(42));
            assert_eq!(storage.load(1).as_bool(), Some(true));
            assert!(storage.load(2).is_none());
        }
    }

    #[test]
    fn test_inline_storage_metadata() {
        let mut storage = InlineFrameStorage::new();

        storage.set_metadata(0b1010_1010, 123);

        assert_eq!(storage.liveness(), 0b1010_1010);
        assert_eq!(storage.resume_index(), 123);
    }

    #[test]
    fn test_inline_storage_clear() {
        let mut storage = InlineFrameStorage::new();

        storage.store(0, Value::int(1).unwrap());
        storage.store(1, Value::int(2).unwrap());
        storage.set_metadata(0xFF, 99);

        storage.clear();

        assert_eq!(storage.count(), 0);
        assert_eq!(storage.liveness(), 0);
        assert_eq!(storage.resume_index(), 0);
    }

    #[test]
    fn test_inline_storage_cache_line_aligned() {
        assert_eq!(
            std::mem::align_of::<InlineFrameStorage>(),
            64,
            "InlineFrameStorage should be cache-line aligned"
        );
    }

    // ════════════════════════════════════════════════════════════════════════
    // HeapFrameStorage Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_heap_storage_new() {
        let storage = HeapFrameStorage::with_capacity(16);
        assert_eq!(storage.count(), 0);
        assert!(storage.capacity() >= 16);
        assert_eq!(storage.liveness(), 0);
        assert_eq!(storage.resume_index(), 0);
    }

    #[test]
    fn test_heap_storage_store_load() {
        let mut storage = HeapFrameStorage::with_capacity(4);

        storage.store(Value::int(10).unwrap());
        storage.store(Value::int(20).unwrap());
        storage.store(Value::int(30).unwrap());

        assert_eq!(storage.count(), 3);
        assert_eq!(storage.load(0).unwrap().as_int(), Some(10));
        assert_eq!(storage.load(1).unwrap().as_int(), Some(20));
        assert_eq!(storage.load(2).unwrap().as_int(), Some(30));
        assert!(storage.load(3).is_none());
    }

    #[test]
    fn test_heap_storage_metadata() {
        let mut storage = HeapFrameStorage::with_capacity(4);

        storage.set_metadata(0xDEADBEEF, 456);

        assert_eq!(storage.liveness(), 0xDEADBEEF);
        assert_eq!(storage.resume_index(), 456);
    }

    #[test]
    fn test_heap_storage_clear() {
        let mut storage = HeapFrameStorage::with_capacity(4);

        storage.store(Value::int(1).unwrap());
        storage.store(Value::int(2).unwrap());
        storage.set_metadata(0xFF, 99);

        storage.clear();

        assert_eq!(storage.count(), 0);
        assert_eq!(storage.liveness(), 0);
        assert_eq!(storage.resume_index(), 0);
    }

    #[test]
    fn test_heap_storage_ensure_capacity() {
        let mut storage = HeapFrameStorage::with_capacity(4);
        let initial_cap = storage.capacity();

        storage.ensure_capacity(32);

        assert!(storage.capacity() >= 32);
        assert!(storage.capacity() >= initial_cap);
    }

    // ════════════════════════════════════════════════════════════════════════
    // PooledFrame Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_pooled_frame_inline() {
        let frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

        assert!(frame.is_inline());
        assert_eq!(frame.capacity(), INLINE_REGISTER_CAPACITY);
        assert_eq!(frame.count(), 0);
    }

    #[test]
    fn test_pooled_frame_heap() {
        let frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(32)));

        assert!(!frame.is_inline());
        assert!(frame.capacity() >= 32);
        assert_eq!(frame.count(), 0);
    }

    #[test]
    fn test_pooled_frame_store_restore_inline() {
        let mut frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

        let registers = vec![
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ];
        let liveness = 0b1011u64; // Registers 0, 1, 3 are live

        frame.set_metadata(liveness, 42);
        frame.store_registers(&registers, liveness);

        let mut restored = vec![Value::none(); 4];
        frame.restore_registers(&mut restored, liveness);

        assert_eq!(restored[0].as_int(), Some(1));
        assert_eq!(restored[1].as_int(), Some(2));
        assert!(restored[2].is_none()); // Was not live
        assert_eq!(restored[3].as_int(), Some(4));
    }

    #[test]
    fn test_pooled_frame_store_restore_heap() {
        let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(16)));

        let registers: Vec<Value> = (0..12).map(|i| Value::int(i).unwrap()).collect();
        let liveness = 0b1010_1010_1010u64;

        frame.set_metadata(liveness, 99);
        frame.store_registers(&registers, liveness);

        let mut restored = vec![Value::none(); 12];
        frame.restore_registers(&mut restored, liveness);

        // Check only even-indexed (live) registers
        for i in (1..12).step_by(2) {
            assert_eq!(restored[i].as_int(), Some(i as i64));
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // GeneratorFramePool Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_pool_new() {
        let pool = GeneratorFramePool::new();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
    }

    #[test]
    fn test_pool_with_capacity() {
        let pool = GeneratorFramePool::with_capacity(8, 4);
        let stats = pool.stats();

        assert_eq!(stats.small_pool_size, 8);
        assert_eq!(stats.large_pool_size, 4);
        assert_eq!(stats.heap_allocations, 12);
    }

    #[test]
    fn test_pool_allocate_small() {
        let mut pool = GeneratorFramePool::new();

        let frame = pool.allocate(4);
        assert!(frame.is_inline());
        assert!(frame.capacity() >= 4);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.pool_misses, 1);
        assert_eq!(stats.heap_allocations, 1);
    }

    #[test]
    fn test_pool_allocate_large() {
        let mut pool = GeneratorFramePool::new();

        let frame = pool.allocate(16);
        assert!(!frame.is_inline());
        assert!(frame.capacity() >= 16);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_pool_release_reuse() {
        let mut pool = GeneratorFramePool::new();

        let frame1 = pool.allocate(4);
        let ptr1 = frame1.as_ptr();

        pool.release(frame1);

        let frame2 = pool.allocate(4);
        let ptr2 = frame2.as_ptr();

        assert_eq!(ptr1, ptr2, "Released frame should be reused");

        let stats = pool.stats();
        assert_eq!(stats.allocations, 2);
        assert_eq!(stats.releases, 1);
        assert_eq!(stats.pool_hits, 1);
        assert_eq!(stats.pool_misses, 1);
    }

    #[test]
    fn test_pool_separate_lists() {
        let mut pool = GeneratorFramePool::new();

        // Allocate and release small frame
        let small = pool.allocate(4);
        pool.release(small);

        // Allocate large - should not reuse small
        let large = pool.allocate(16);
        assert!(!large.is_inline());

        let stats = pool.stats();
        assert_eq!(stats.pool_hits, 0); // Large allocation was a miss
        assert_eq!(stats.pool_misses, 2);
    }

    #[test]
    fn test_pool_stats_tracking() {
        let mut pool = GeneratorFramePool::new();

        // Various operations
        let f1 = pool.allocate(4);
        let f2 = pool.allocate(4);
        let f3 = pool.allocate(16);

        pool.release(f1);
        pool.release(f2);

        let _f4 = pool.allocate(4);
        let _f5 = pool.allocate(4);

        let stats = pool.stats();
        assert_eq!(stats.allocations, 5);
        assert_eq!(stats.releases, 2);
        assert_eq!(stats.pool_hits, 2);
    }

    #[test]
    fn test_pool_max_size_limit() {
        let mut pool = GeneratorFramePool::new();

        // Fill pool beyond limit
        let mut frames = Vec::new();
        for _ in 0..(MAX_SMALL_POOL_SIZE + 10) {
            frames.push(pool.allocate(4));
        }

        // Release all
        for frame in frames {
            pool.release(frame);
        }

        // Pool should be capped at max size
        let stats = pool.stats();
        assert!(stats.small_pool_size <= MAX_SMALL_POOL_SIZE);
        assert!(stats.evictions > 0);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = GeneratorFramePool::with_capacity(8, 4);
        assert!(!pool.is_empty());

        pool.clear();

        assert!(pool.is_empty());
        let stats = pool.stats();
        assert_eq!(stats.small_pool_size, 0);
        assert_eq!(stats.large_pool_size, 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Edge Case Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_allocate_boundary_size() {
        let mut pool = GeneratorFramePool::new();

        // Exactly at inline limit
        let frame = pool.allocate(INLINE_REGISTER_CAPACITY);
        assert!(frame.is_inline());

        // One over limit
        let frame = pool.allocate(INLINE_REGISTER_CAPACITY + 1);
        assert!(!frame.is_inline());
    }

    #[test]
    fn test_allocate_zero_registers() {
        let mut pool = GeneratorFramePool::new();

        let frame = pool.allocate(0);
        assert!(frame.is_inline()); // Should still use inline
    }

    #[test]
    fn test_store_restore_empty() {
        let mut frame = PooledFrame::Inline(Box::new(InlineFrameStorage::new()));

        let registers: Vec<Value> = vec![];
        let liveness = 0u64;

        frame.store_registers(&registers, liveness);

        let mut restored: Vec<Value> = vec![];
        frame.restore_registers(&mut restored, liveness);

        assert_eq!(frame.count(), 0);
    }

    #[test]
    fn test_store_restore_all_live() {
        let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(8)));

        let registers: Vec<Value> = (0..8).map(|i| Value::int(i).unwrap()).collect();
        let liveness = 0xFFu64; // All 8 live

        frame.store_registers(&registers, liveness);

        let mut restored = vec![Value::none(); 8];
        frame.restore_registers(&mut restored, liveness);

        for i in 0..8 {
            assert_eq!(restored[i].as_int(), Some(i as i64));
        }
    }

    #[test]
    fn test_pooled_frame_clear() {
        let mut frame = PooledFrame::Heap(Box::new(HeapFrameStorage::with_capacity(8)));

        frame.store_registers(&[Value::int(1).unwrap(), Value::int(2).unwrap()], 0b11);
        frame.set_metadata(0xFF, 42);

        frame.clear();

        assert_eq!(frame.count(), 0);
        assert_eq!(frame.liveness(), 0);
        assert_eq!(frame.resume_index(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Performance Characteristics Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_inline_storage_size() {
        // Ensure inline storage fits in reasonable size
        let size = std::mem::size_of::<InlineFrameStorage>();
        assert!(
            size <= 128,
            "InlineFrameStorage should be compact: {} bytes",
            size
        );
    }

    #[test]
    fn test_heap_storage_overhead() {
        let size = std::mem::size_of::<HeapFrameStorage>();
        // Vec (24) + u64 (8) + u32 (4) + padding
        assert!(
            size <= 48,
            "HeapFrameStorage overhead should be minimal: {} bytes",
            size
        );
    }

    #[test]
    fn test_pooled_frame_size() {
        let size = std::mem::size_of::<PooledFrame>();
        // Should be 8 bytes (pointer) + 8 bytes (tag + padding) at most
        assert!(
            size <= 16,
            "PooledFrame enum should be compact: {} bytes",
            size
        );
    }
}
