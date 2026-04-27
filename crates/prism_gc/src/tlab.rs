//! Thread-Local Allocation Buffers (TLABs)
//!
//! TLABs provide zero-atomic allocation on the fast path by giving each thread
//! its own bump-pointer region carved from the nursery.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        Nursery                               │
//! ├─────────────────┬─────────────────┬─────────────────────────┤
//! │   Thread 1 TLAB │   Thread 2 TLAB │     (unallocated)       │
//! │  [ptr...end]    │  [ptr...end]    │                         │
//! └─────────────────┴─────────────────┴─────────────────────────┘
//! ```
//!
//! # Performance
//!
//! - Fast path: Simple pointer bump, NO atomics
//! - Slow path: Refill from nursery (requires atomic CAS)
//!
//! # Usage
//!
//! ```ignore
//! let mut tlab = Tlab::new(config);
//! tlab.refill(&nursery)?;  // Get region from nursery
//!
//! // Fast-path allocation (no atomics!)
//! if let Some(ptr) = tlab.alloc_fast(size) {
//!     // Success - object allocated inline
//! } else {
//!     // TLAB exhausted - refill and retry
//!     tlab.refill(&nursery)?;
//!     let ptr = tlab.alloc_fast(size).expect("TLAB too small");
//! }
//! ```

use std::ptr::NonNull;

// =============================================================================
// Configuration
// =============================================================================

/// Default TLAB size: 64KB
/// - Large enough to amortize refill cost
/// - Small enough to not waste memory when threads are idle
pub const DEFAULT_TLAB_SIZE: usize = 64 * 1024;

/// Minimum TLAB size: 4KB (one page)
pub const MIN_TLAB_SIZE: usize = 4 * 1024;

/// Maximum TLAB size: 1MB
pub const MAX_TLAB_SIZE: usize = 1024 * 1024;

/// TLAB configuration.
#[derive(Debug, Clone, Copy)]
pub struct TlabConfig {
    /// Initial TLAB size in bytes.
    pub initial_size: usize,
    /// Maximum TLAB size (for adaptive sizing).
    pub max_size: usize,
    /// Minimum object size for TLAB allocation (smaller go to nursery).
    pub min_object_size: usize,
    /// Maximum object size for TLAB allocation (larger go to LOH).
    pub max_object_size: usize,
}

impl Default for TlabConfig {
    fn default() -> Self {
        Self {
            initial_size: DEFAULT_TLAB_SIZE,
            max_size: MAX_TLAB_SIZE,
            min_object_size: 8,
            max_object_size: DEFAULT_TLAB_SIZE / 2, // Half of TLAB
        }
    }
}

// =============================================================================
// Thread-Local Allocation Buffer
// =============================================================================

/// Thread-Local Allocation Buffer.
///
/// Provides zero-atomic bump-pointer allocation for a single thread.
/// When exhausted, must be refilled from the shared nursery.
#[repr(C)]
pub struct Tlab {
    /// Current allocation pointer (grows upward).
    ptr: *mut u8,
    /// End of the TLAB region.
    end: *mut u8,
    /// Start of the TLAB region (for statistics).
    start: *mut u8,
    /// Configured TLAB size.
    size: usize,
    /// Statistics: total bytes allocated through this TLAB.
    bytes_allocated: usize,
    /// Statistics: number of refills.
    refill_count: usize,
}

// Safety: Tlab is only accessed by its owning thread.
// The pointers point to nursery memory which is thread-safe.
unsafe impl Send for Tlab {}

impl Tlab {
    /// Create an empty TLAB that needs to be refilled.
    pub fn new(size: usize) -> Self {
        let size = size.clamp(MIN_TLAB_SIZE, MAX_TLAB_SIZE);
        Self {
            ptr: std::ptr::null_mut(),
            end: std::ptr::null_mut(),
            start: std::ptr::null_mut(),
            size,
            bytes_allocated: 0,
            refill_count: 0,
        }
    }

    /// Create a TLAB with default size.
    pub fn with_default_size() -> Self {
        Self::new(DEFAULT_TLAB_SIZE)
    }

    /// Attempt to allocate `size` bytes from this TLAB.
    ///
    /// This is the fast path: NO atomics, just pointer bump.
    ///
    /// # Safety
    ///
    /// Size must be properly aligned (typically 8-byte aligned).
    #[inline(always)]
    pub fn alloc_fast(&mut self, size: usize) -> Option<NonNull<u8>> {
        // Align size to 8 bytes
        let aligned_size = (size + 7) & !7;

        let new_ptr = unsafe { self.ptr.add(aligned_size) };

        if new_ptr <= self.end {
            let result = self.ptr;
            self.ptr = new_ptr;
            self.bytes_allocated += aligned_size;
            NonNull::new(result)
        } else {
            None // TLAB exhausted
        }
    }

    /// Check if the TLAB is empty (needs refill).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ptr.is_null() || self.ptr >= self.end
    }

    /// Get remaining free bytes in this TLAB.
    #[inline]
    pub fn free(&self) -> usize {
        if self.ptr.is_null() || self.end.is_null() {
            return 0;
        }
        (self.end as usize).saturating_sub(self.ptr as usize)
    }

    /// Get bytes allocated in this TLAB since last refill.
    #[inline]
    pub fn used(&self) -> usize {
        if self.ptr.is_null() || self.start.is_null() {
            return 0;
        }
        (self.ptr as usize).saturating_sub(self.start as usize)
    }

    /// Get configured TLAB size.
    #[inline]
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get total bytes allocated through this TLAB (lifetime).
    #[inline]
    pub fn total_bytes_allocated(&self) -> usize {
        self.bytes_allocated
    }

    /// Get number of refills (lifetime).
    #[inline]
    pub fn refill_count(&self) -> usize {
        self.refill_count
    }

    /// Check if a pointer is within this TLAB's region.
    #[inline]
    pub fn contains(&self, ptr: *const ()) -> bool {
        let addr = ptr as usize;
        let start = self.start as usize;
        let end = self.end as usize;
        addr >= start && addr < end
    }

    /// Refill the TLAB with a new region.
    ///
    /// Called when the TLAB is exhausted. This obtains a new region
    /// from the nursery using atomic operations (slow path).
    ///
    /// Returns the amount of wasted space from the old TLAB.
    pub fn refill_from_region(&mut self, start: *mut u8, end: *mut u8) -> usize {
        let waste = self.free();

        self.start = start;
        self.end = end;
        self.ptr = start;
        self.refill_count += 1;

        waste
    }

    /// Retire this TLAB (return unused space to nursery).
    ///
    /// Called before GC to ensure no unreturned TLAB space.
    /// Returns (start, used_end) for the nursery to know what's actually used.
    pub fn retire(&mut self) -> Option<(*mut u8, *mut u8)> {
        if self.start.is_null() {
            return None;
        }

        let result = Some((self.start, self.ptr));

        self.start = std::ptr::null_mut();
        self.end = std::ptr::null_mut();
        self.ptr = std::ptr::null_mut();

        result
    }

    /// Get TLAB statistics.
    pub fn stats(&self) -> TlabStats {
        TlabStats {
            current_used: self.used(),
            current_free: self.free(),
            total_allocated: self.bytes_allocated,
            refill_count: self.refill_count,
        }
    }
}

impl Default for Tlab {
    fn default() -> Self {
        Self::with_default_size()
    }
}

// =============================================================================
// Statistics
// =============================================================================

/// TLAB statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct TlabStats {
    /// Bytes used in current TLAB region.
    pub current_used: usize,
    /// Bytes free in current TLAB region.
    pub current_free: usize,
    /// Total bytes allocated through this TLAB (lifetime).
    pub total_allocated: usize,
    /// Number of refills (lifetime).
    pub refill_count: usize,
}

// =============================================================================
// TLAB Provider
// =============================================================================

/// Provider trait for obtaining TLAB regions.
///
/// Implemented by the nursery to provide TLAB regions to threads.
pub trait TlabProvider {
    /// Attempt to obtain a TLAB region of the given size.
    ///
    /// Returns (start, end) if successful, None if nursery is full.
    fn obtain_tlab_region(&self, size: usize) -> Option<(*mut u8, *mut u8)>;
}
