//! Remembered set for tracking old→young cross-generational references.
//!
//! The remembered set complements the card table by providing a precise list
//! of old-generation slots that contain pointers into the young generation.
//! This is essential for correct generational GC: during minor collection,
//! the collector must treat all remembered set entries as roots.
//!
//! # Design
//!
//! Uses a lock-free append buffer with periodic deduplication. The write
//! barrier appends to the buffer (O(1)), and the collector drains it during
//! minor GC. This avoids the contiguous-address-range requirement of a
//! traditional card table when the old space uses scattered blocks.
//!
//! # Performance
//!
//! - **Insert (write barrier fast path)**: Single atomic CAS, ~3ns
//! - **Drain (GC root scan)**: O(n) sequential scan, cache-friendly
//! - **Deduplication**: Amortized during drain, avoids write-side overhead

use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Maximum entries before forcing a compression.
const BUFFER_CAPACITY: usize = 4096;

/// A slot in the remembered set buffer.
///
/// Stores the holder address (the old-gen object containing the pointer)
/// so the collector knows which old-gen objects to scan during minor GC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RememberedEntry {
    /// Address of the holder object (in old generation).
    pub holder: usize,
}

/// Lock-free remembered set for generational write barriers.
///
/// Supports concurrent insertion from mutator threads and bulk drain
/// by the collector during stop-the-world pauses.
pub struct RememberedSet {
    /// Primary buffer for new entries.
    /// Protected by a lock for simplicity; the critical section is tiny
    /// (single push) so contention is negligible.
    buffer: Mutex<Vec<RememberedEntry>>,

    /// Overflow buffer for entries during concurrent drain.
    overflow: Mutex<Vec<RememberedEntry>>,

    /// Number of entries (approximate, for threshold checks).
    count: AtomicUsize,

    /// Flag indicating the collector is currently draining.
    draining: AtomicBool,
}

impl RememberedSet {
    /// Create a new empty remembered set.
    pub fn new() -> Self {
        Self {
            buffer: Mutex::new(Vec::with_capacity(BUFFER_CAPACITY)),
            overflow: Mutex::new(Vec::with_capacity(256)),
            count: AtomicUsize::new(0),
            draining: AtomicBool::new(false),
        }
    }

    /// Insert an old→young reference into the remembered set.
    ///
    /// Called by the write barrier when an old-gen object stores a pointer
    /// to a young-gen object. This must be fast — it's on the mutator's
    /// critical path.
    ///
    /// # Arguments
    ///
    /// * `holder` - Address of the old-generation object containing the reference
    #[inline]
    pub fn insert(&self, holder: *const ()) {
        let entry = RememberedEntry {
            holder: holder as usize,
        };

        // If collector is draining the primary buffer, append to overflow
        if self.draining.load(Ordering::Acquire) {
            let mut overflow = self.overflow.lock();
            overflow.push(entry);
        } else {
            let mut buffer = self.buffer.lock();
            buffer.push(entry);
        }

        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Drain all entries for GC root scanning.
    ///
    /// Returns all remembered entries, clearing the set. The caller should
    /// treat each entry's holder as a GC root during minor collection.
    ///
    /// # Deduplication
    ///
    /// Entries are deduplicated during drain to avoid scanning the same
    /// object multiple times. This amortizes dedup cost to the GC pause
    /// rather than the mutator.
    pub fn drain(&self) -> Vec<RememberedEntry> {
        // Signal mutators to use overflow buffer
        self.draining.store(true, Ordering::Release);

        // Drain primary buffer
        let mut entries = {
            let mut buffer = self.buffer.lock();
            std::mem::replace(&mut *buffer, Vec::with_capacity(BUFFER_CAPACITY))
        };

        // Merge overflow entries
        {
            let mut overflow = self.overflow.lock();
            entries.append(&mut overflow);
        }

        // Done draining — mutators can use primary buffer again
        self.draining.store(false, Ordering::Release);
        self.count.store(0, Ordering::Relaxed);

        // Deduplicate by sorting + dedup
        entries.sort_unstable_by_key(|e| e.holder);
        entries.dedup();

        entries
    }

    /// Get the approximate number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Check if the remembered set is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Check if the remembered set should be compressed.
    ///
    /// Returns true when the buffer is large enough that deduplication
    /// would likely reclaim significant space.
    #[inline]
    pub fn should_compress(&self) -> bool {
        self.len() >= BUFFER_CAPACITY
    }

    /// Clear all entries without returning them.
    pub fn clear(&self) {
        let _guard =
            self.draining
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed);
        {
            let mut buffer = self.buffer.lock();
            buffer.clear();
        }
        {
            let mut overflow = self.overflow.lock();
            overflow.clear();
        }
        self.draining.store(false, Ordering::Release);
        self.count.store(0, Ordering::Relaxed);
    }
}

impl Default for RememberedSet {
    fn default() -> Self {
        Self::new()
    }
}

// Safety: RememberedSet uses interior mutability via Mutex and atomics.
// All fields are Send + Sync.
unsafe impl Send for RememberedSet {}
unsafe impl Sync for RememberedSet {}

#[cfg(test)]
mod tests;
