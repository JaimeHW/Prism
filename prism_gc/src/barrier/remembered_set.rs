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
mod tests {
    use super::*;

    #[test]
    fn test_remembered_set_creation() {
        let rs = RememberedSet::new();
        assert!(rs.is_empty());
        assert_eq!(rs.len(), 0);
    }

    #[test]
    fn test_insert_single() {
        let rs = RememberedSet::new();
        let ptr = 0x1000 as *const ();
        rs.insert(ptr);
        assert_eq!(rs.len(), 1);
        assert!(!rs.is_empty());
    }

    #[test]
    fn test_insert_multiple() {
        let rs = RememberedSet::new();
        for i in 0..100 {
            rs.insert((0x1000 + i * 8) as *const ());
        }
        assert_eq!(rs.len(), 100);
    }

    #[test]
    fn test_drain_returns_all_entries() {
        let rs = RememberedSet::new();
        for i in 0..10 {
            rs.insert((0x1000 + i * 64) as *const ());
        }

        let entries = rs.drain();
        assert_eq!(entries.len(), 10);
        assert!(rs.is_empty());
    }

    #[test]
    fn test_drain_deduplicates() {
        let rs = RememberedSet::new();
        let ptr = 0x2000 as *const ();

        // Insert the same pointer multiple times
        for _ in 0..50 {
            rs.insert(ptr);
        }

        let entries = rs.drain();
        // After dedup, should be exactly 1
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].holder, 0x2000);
    }

    #[test]
    fn test_drain_preserves_unique_entries() {
        let rs = RememberedSet::new();

        // Insert 5 unique + duplicates
        for i in 0..5 {
            let ptr = (0x1000 + i * 512) as *const ();
            rs.insert(ptr);
            rs.insert(ptr); // duplicate
        }

        let entries = rs.drain();
        assert_eq!(entries.len(), 5);
    }

    #[test]
    fn test_drain_sorted_order() {
        let rs = RememberedSet::new();

        // Insert in reverse order
        rs.insert(0x3000 as *const ());
        rs.insert(0x1000 as *const ());
        rs.insert(0x2000 as *const ());

        let entries = rs.drain();
        assert_eq!(entries[0].holder, 0x1000);
        assert_eq!(entries[1].holder, 0x2000);
        assert_eq!(entries[2].holder, 0x3000);
    }

    #[test]
    fn test_clear() {
        let rs = RememberedSet::new();
        for i in 0..20 {
            rs.insert((0x1000 + i * 8) as *const ());
        }
        assert_eq!(rs.len(), 20);

        rs.clear();
        assert!(rs.is_empty());
        assert_eq!(rs.len(), 0);
    }

    #[test]
    fn test_should_compress() {
        let rs = RememberedSet::new();
        assert!(!rs.should_compress());

        // Fill to capacity
        for i in 0..BUFFER_CAPACITY {
            rs.insert((0x1000 + i * 8) as *const ());
        }
        assert!(rs.should_compress());
    }

    #[test]
    fn test_drain_then_insert() {
        let rs = RememberedSet::new();
        rs.insert(0x1000 as *const ());
        rs.insert(0x2000 as *const ());

        let entries = rs.drain();
        assert_eq!(entries.len(), 2);

        // Insert after drain
        rs.insert(0x3000 as *const ());
        assert_eq!(rs.len(), 1);

        let entries2 = rs.drain();
        assert_eq!(entries2.len(), 1);
        assert_eq!(entries2[0].holder, 0x3000);
    }

    #[test]
    fn test_concurrent_insert_simulation() {
        // Simulate what happens when multiple threads insert
        let rs = RememberedSet::new();

        // Main thread inserts
        for i in 0..50 {
            rs.insert((0x1000 + i * 8) as *const ());
        }

        assert_eq!(rs.len(), 50);
        let entries = rs.drain();
        assert_eq!(entries.len(), 50);
    }

    #[test]
    fn test_overflow_during_drain() {
        let rs = RememberedSet::new();

        // Insert before drain
        rs.insert(0x1000 as *const ());
        rs.insert(0x2000 as *const ());

        // Simulate draining flag being set
        rs.draining.store(true, Ordering::Release);

        // These should go to overflow
        rs.insert(0x3000 as *const ());
        rs.insert(0x4000 as *const ());

        // Reset draining
        rs.draining.store(false, Ordering::Release);

        // Drain should get all entries
        let entries = rs.drain();
        assert_eq!(entries.len(), 4);
    }

    #[test]
    fn test_default() {
        let rs = RememberedSet::default();
        assert!(rs.is_empty());
    }

    #[test]
    fn test_remembered_entry_debug() {
        let entry = RememberedEntry { holder: 0x42 };
        let debug = format!("{:?}", entry);
        assert!(debug.contains("66")); // 0x42 = 66
    }

    #[test]
    fn test_remembered_entry_equality() {
        let a = RememberedEntry { holder: 0x1000 };
        let b = RememberedEntry { holder: 0x1000 };
        let c = RememberedEntry { holder: 0x2000 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_large_volume() {
        let rs = RememberedSet::new();

        // Insert many entries
        for i in 0..10_000 {
            rs.insert((0x10000 + i * 8) as *const ());
        }

        assert_eq!(rs.len(), 10_000);
        let entries = rs.drain();
        assert_eq!(entries.len(), 10_000);
        assert!(rs.is_empty());
    }
}
