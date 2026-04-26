//! Resume table caching for computed-goto-style generator dispatch.
//!
//! This module provides O(1) lookup of resume points within generators,
//! using a multi-level cache strategy for optimal performance.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                         ResumeTableCache                                 │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  Level 1: InlineResumeCache (4 entries)                                 │
//! │  ┌────────────────────────────────────────────────────────────────┐     │
//! │  │ [code_ptr₀, table₀] [code_ptr₁, table₁] [code_ptr₂, table₂] ... │    │
//! │  └────────────────────────────────────────────────────────────────┘     │
//! │       │                                                                  │
//! │       │ Cache Miss                                                       │
//! │       ▼                                                                  │
//! │  Level 2: HashMap<usize, Arc<ResumeTable>>                              │
//! │  ┌────────────────────────────────────────────────────────────────┐     │
//! │  │ { 0xDEAD... → table, 0xBEEF... → table, ... }                  │     │
//! │  └────────────────────────────────────────────────────────────────┘     │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance
//!
//! | Operation | L1 Hit | L2 Hit | Miss |
//! |-----------|--------|--------|------|
//! | Lookup | ~3 cycles | ~30 cycles | O(1) amortized |
//! | Insert | O(1) | O(1) | O(1) amortized |
//!
//! # ResumeTable Structure
//!
//! Each ResumeTable maps resume indices to PC offsets:
//!
//! ```text
//! ResumeTable {
//!     entries: [
//!         YieldPointEntry { resume_idx: 0, pc: 100 },
//!         YieldPointEntry { resume_idx: 1, pc: 200 },
//!         YieldPointEntry { resume_idx: 2, pc: 350 },
//!     ]
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// Constants
// =============================================================================

/// Number of entries in the inline cache.
/// Power of 2 for efficient modulo via bitwise AND.
const INLINE_CACHE_ENTRIES: usize = 4;

/// Maximum entries per resume table before switching to sparse storage.
const MAX_DENSE_ENTRIES: usize = 256;

/// Initial capacity for the overflow HashMap.
const INITIAL_OVERFLOW_CAPACITY: usize = 16;

// =============================================================================
// Yield Point Entry
// =============================================================================

/// A single yield point entry mapping resume index to PC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YieldPointEntry {
    /// The resume index (0-based).
    pub resume_idx: u32,
    /// The PC offset to jump to.
    pub pc: u32,
}

impl YieldPointEntry {
    /// Create a new yield point entry.
    #[inline]
    pub const fn new(resume_idx: u32, pc: u32) -> Self {
        Self { resume_idx, pc }
    }
}

// =============================================================================
// Resume Table
// =============================================================================

/// Table of yield points for a single code object.
///
/// Uses either dense or sparse storage based on the number of entries.
#[derive(Debug, Clone)]
pub struct ResumeTable {
    /// Storage strategy.
    storage: ResumeTableStorage,
}

/// Storage strategy for resume tables.
#[derive(Debug, Clone)]
enum ResumeTableStorage {
    /// Dense array for small tables (most common).
    /// Index directly by resume_idx.
    Dense(Vec<u32>), // PC values indexed by resume_idx

    /// Sparse HashMap for large tables (rare).
    Sparse(HashMap<u32, u32>),
}

impl ResumeTable {
    /// Create a new empty resume table.
    #[inline]
    pub fn new() -> Self {
        Self {
            storage: ResumeTableStorage::Dense(Vec::new()),
        }
    }

    /// Create a resume table with the given capacity hint.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity <= MAX_DENSE_ENTRIES {
            Self {
                storage: ResumeTableStorage::Dense(Vec::with_capacity(capacity)),
            }
        } else {
            Self {
                storage: ResumeTableStorage::Sparse(HashMap::with_capacity(capacity)),
            }
        }
    }

    /// Get the number of entries.
    #[inline]
    pub fn len(&self) -> usize {
        match &self.storage {
            ResumeTableStorage::Dense(vec) => vec.len(),
            ResumeTableStorage::Sparse(map) => map.len(),
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Insert a yield point.
    #[inline]
    pub fn insert(&mut self, resume_idx: u32, pc: u32) {
        match &mut self.storage {
            ResumeTableStorage::Dense(vec) => {
                let idx = resume_idx as usize;

                // Check if we need to convert to sparse
                if idx >= MAX_DENSE_ENTRIES {
                    self.convert_to_sparse();
                    if let ResumeTableStorage::Sparse(map) = &mut self.storage {
                        map.insert(resume_idx, pc);
                    }
                    return;
                }

                // Extend if needed
                if idx >= vec.len() {
                    vec.resize(idx + 1, u32::MAX); // u32::MAX = invalid
                }
                vec[idx] = pc;
            }
            ResumeTableStorage::Sparse(map) => {
                map.insert(resume_idx, pc);
            }
        }
    }

    /// Get the PC for a resume index.
    #[inline]
    pub fn get_pc(&self, resume_idx: u32) -> Option<u32> {
        match &self.storage {
            ResumeTableStorage::Dense(vec) => {
                let idx = resume_idx as usize;
                vec.get(idx).copied().filter(|&pc| pc != u32::MAX)
            }
            ResumeTableStorage::Sparse(map) => map.get(&resume_idx).copied(),
        }
    }

    /// Check if a resume index exists.
    #[inline]
    pub fn contains(&self, resume_idx: u32) -> bool {
        self.get_pc(resume_idx).is_some()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = YieldPointEntry> + '_ {
        match &self.storage {
            ResumeTableStorage::Dense(vec) => {
                let dense_iter = vec.iter().enumerate().filter_map(|(idx, &pc)| {
                    if pc != u32::MAX {
                        Some(YieldPointEntry::new(idx as u32, pc))
                    } else {
                        None
                    }
                });
                IteratorWrapper::Dense(dense_iter)
            }
            ResumeTableStorage::Sparse(map) => {
                let sparse_iter = map.iter().map(|(&idx, &pc)| YieldPointEntry::new(idx, pc));
                IteratorWrapper::Sparse(sparse_iter)
            }
        }
    }

    /// Convert from dense to sparse storage.
    fn convert_to_sparse(&mut self) {
        if let ResumeTableStorage::Dense(vec) = &self.storage {
            let mut map = HashMap::with_capacity(vec.len());
            for (idx, &pc) in vec.iter().enumerate() {
                if pc != u32::MAX {
                    map.insert(idx as u32, pc);
                }
            }
            self.storage = ResumeTableStorage::Sparse(map);
        }
    }

    /// Check if using dense storage.
    #[inline]
    pub fn is_dense(&self) -> bool {
        matches!(self.storage, ResumeTableStorage::Dense(_))
    }
}

impl Default for ResumeTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator wrapper to unify dense and sparse iteration.
enum IteratorWrapper<D, S> {
    Dense(D),
    Sparse(S),
}

impl<D, S> Iterator for IteratorWrapper<D, S>
where
    D: Iterator<Item = YieldPointEntry>,
    S: Iterator<Item = YieldPointEntry>,
{
    type Item = YieldPointEntry;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IteratorWrapper::Dense(iter) => iter.next(),
            IteratorWrapper::Sparse(iter) => iter.next(),
        }
    }
}

// =============================================================================
// Inline Resume Cache
// =============================================================================

/// Fast inline cache for the most recently used resume tables.
///
/// Uses a simple direct-mapped cache with code object pointer as key.
#[derive(Debug)]
pub struct InlineResumeCache {
    /// Cache entries: (code_ptr, resume_table).
    /// Zero code_ptr indicates empty slot.
    entries: [(usize, Option<Arc<ResumeTable>>); INLINE_CACHE_ENTRIES],
}

impl InlineResumeCache {
    /// Create a new empty inline cache.
    #[inline]
    pub fn new() -> Self {
        Self {
            entries: [(0, None), (0, None), (0, None), (0, None)],
        }
    }

    /// Lookup a resume table by code object pointer.
    #[inline]
    pub fn lookup(&self, code_ptr: usize) -> Option<&Arc<ResumeTable>> {
        let idx = Self::hash(code_ptr);
        let (key, value) = &self.entries[idx];
        if *key == code_ptr {
            value.as_ref()
        } else {
            None
        }
    }

    /// Insert a resume table into the cache.
    #[inline]
    pub fn insert(&mut self, code_ptr: usize, table: Arc<ResumeTable>) {
        let idx = Self::hash(code_ptr);
        self.entries[idx] = (code_ptr, Some(table));
    }

    /// Invalidate an entry.
    #[inline]
    pub fn invalidate(&mut self, code_ptr: usize) {
        let idx = Self::hash(code_ptr);
        if self.entries[idx].0 == code_ptr {
            self.entries[idx] = (0, None);
        }
    }

    /// Clear all entries.
    #[inline]
    pub fn clear(&mut self) {
        for entry in &mut self.entries {
            *entry = (0, None);
        }
    }

    /// Hash function for cache indexing.
    #[inline(always)]
    const fn hash(code_ptr: usize) -> usize {
        // Use golden ratio bits for better distribution
        let hash = code_ptr.wrapping_mul(0x9E3779B97F4A7C15);
        (hash >> 60) & (INLINE_CACHE_ENTRIES - 1)
    }
}

impl Default for InlineResumeCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Cache Statistics
// =============================================================================

/// Statistics for monitoring cache behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResumeCacheStats {
    /// Inline cache hits.
    pub inline_hits: u64,
    /// Inline cache misses.
    pub inline_misses: u64,
    /// Overflow map hits.
    pub overflow_hits: u64,
    /// Total lookups.
    pub lookups: u64,
    /// Total inserts.
    pub inserts: u64,
    /// Tables in overflow map.
    pub overflow_size: usize,
}

// =============================================================================
// Resume Table Cache
// =============================================================================

/// Multi-level cache for resume tables.
///
/// Provides O(1) access to yield point PCs for generator resumption.
pub struct ResumeTableCache {
    /// L1: Inline cache for hot code objects.
    inline: InlineResumeCache,
    /// L2: Overflow map for cold code objects.
    overflow: HashMap<usize, Arc<ResumeTable>>,
    /// Statistics.
    stats: ResumeCacheStats,
}

impl ResumeTableCache {
    /// Create a new empty cache.
    #[inline]
    pub fn new() -> Self {
        Self {
            inline: InlineResumeCache::new(),
            overflow: HashMap::with_capacity(INITIAL_OVERFLOW_CAPACITY),
            stats: ResumeCacheStats::default(),
        }
    }

    /// Lookup a resume table by code object pointer.
    #[inline]
    pub fn lookup(&mut self, code_ptr: usize) -> Option<&Arc<ResumeTable>> {
        self.stats.lookups += 1;

        // Try inline cache first
        if self.inline.lookup(code_ptr).is_some() {
            self.stats.inline_hits += 1;
            return self.inline.lookup(code_ptr);
        }

        self.stats.inline_misses += 1;

        // Check if entry exists in overflow map
        if self.overflow.contains_key(&code_ptr) {
            self.stats.overflow_hits += 1;
            // Clone the Arc to promote to inline cache
            let table = self.overflow.get(&code_ptr).unwrap().clone();
            self.inline.insert(code_ptr, table);
            return self.inline.lookup(code_ptr);
        }

        None
    }

    /// Get or create a resume table for a code object.
    pub fn get_or_create(&mut self, code_ptr: usize) -> &mut Arc<ResumeTable> {
        // Check if exists
        if !self.overflow.contains_key(&code_ptr) {
            // Create new table
            let table = Arc::new(ResumeTable::new());
            self.overflow.insert(code_ptr, table);
            self.stats.inserts += 1;
            self.stats.overflow_size = self.overflow.len();
        }

        self.overflow.get_mut(&code_ptr).unwrap()
    }

    /// Insert a yield point for a code object.
    pub fn insert_yield_point(&mut self, code_ptr: usize, resume_idx: u32, pc: u32) {
        // Get or create table
        let table = self.overflow.entry(code_ptr).or_insert_with(|| {
            self.stats.inserts += 1;
            Arc::new(ResumeTable::new())
        });

        // We need to mutate the table, so make it unique
        Arc::make_mut(table).insert(resume_idx, pc);

        // Update inline cache
        self.inline.insert(code_ptr, Arc::clone(table));
        self.stats.overflow_size = self.overflow.len();
    }

    /// Remove a resume table.
    pub fn remove(&mut self, code_ptr: usize) {
        self.inline.invalidate(code_ptr);
        self.overflow.remove(&code_ptr);
        self.stats.overflow_size = self.overflow.len();
    }

    /// Clear all cached resume tables.
    pub fn clear(&mut self) {
        self.inline.clear();
        self.overflow.clear();
        self.stats.overflow_size = 0;
    }

    /// Get statistics.
    #[inline]
    pub fn stats(&self) -> ResumeCacheStats {
        ResumeCacheStats {
            overflow_size: self.overflow.len(),
            ..self.stats
        }
    }

    /// Get the number of cached tables.
    #[inline]
    pub fn len(&self) -> usize {
        self.overflow.len()
    }

    /// Check if cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.overflow.is_empty()
    }
}

impl Default for ResumeTableCache {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
