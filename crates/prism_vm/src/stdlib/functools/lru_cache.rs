//! High-performance LRU (Least Recently Used) cache implementation.
//!
//! Provides an O(1) amortized cache with configurable maximum size and
//! statistics tracking. Uses a doubly-linked list threaded through a HashMap
//! for O(1) access, insertion, and eviction.
//!
//! # Performance Characteristics
//!
//! | Operation | Bounded Mode | Unbounded Mode |
//! |-----------|-------------|----------------|
//! | Get (hit) | O(1) | O(1) |
//! | Get (miss)| O(1) amortized | O(1) amortized |
//! | Eviction | O(1) | N/A |
//! | Clear | O(n) | O(n) |
//!
//! # Architecture
//!
//! - **Bounded mode** (`maxsize > 0`): HashMap + intrusive doubly-linked list.
//!   Entries are indexed by key hash; on access, the entry is moved to the
//!   head of the LRU list. When capacity is exceeded, the tail is evicted.
//!
//! - **Unbounded mode** (`maxsize = None`): Plain HashMap with no LRU tracking.
//!   No eviction overhead, pure hashtable performance.

use crate::stdlib::collections::counter::HashableValue;
use prism_core::Value;
use std::collections::HashMap;

// =============================================================================
// Cache Statistics
// =============================================================================

/// Cache access statistics, matching Python's `CacheInfo` namedtuple.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheInfo {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Maximum cache size (None for unbounded).
    pub maxsize: Option<usize>,
    /// Current number of cached entries.
    pub currsize: usize,
}

impl CacheInfo {
    /// Hit rate as a percentage (0.0 to 100.0).
    #[inline]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

impl std::fmt::Display for CacheInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CacheInfo(hits={}, misses={}, maxsize={}, currsize={})",
            self.hits,
            self.misses,
            match self.maxsize {
                Some(n) => n.to_string(),
                None => "None".to_string(),
            },
            self.currsize
        )
    }
}

// =============================================================================
// LRU Node (Intrusive Linked List)
// =============================================================================

/// Index into the node arena.
type NodeIndex = usize;

/// Sentinel value for null links.
const NULL_INDEX: NodeIndex = usize::MAX;

/// A node in the intrusive doubly-linked LRU list.
///
/// Each node stores a cached key-value pair and links to its neighbors
/// in the recency order. The most recently used node is at the head;
/// the least recently used is at the tail.
#[derive(Debug, Clone)]
struct LruNode {
    /// The cache key.
    key: HashableValue,
    /// The cached value.
    value: Value,
    /// Index of the previous (more recently used) node.
    prev: NodeIndex,
    /// Index of the next (less recently used) node.
    next: NodeIndex,
}

// =============================================================================
// Bounded LRU Cache
// =============================================================================

/// A bounded LRU cache with O(1) operations using an arena-backed linked list.
///
/// The arena eliminates per-node allocations. Evicted nodes are recycled
/// through a free list, avoiding allocation churn.
#[derive(Debug, Clone)]
struct BoundedLruCache {
    /// Node arena — contiguous storage for all nodes.
    arena: Vec<LruNode>,
    /// Free list — indices of recycled nodes.
    free_list: Vec<NodeIndex>,
    /// Key → node index mapping.
    map: HashMap<HashableValue, NodeIndex>,
    /// Head of the LRU list (most recently used).
    head: NodeIndex,
    /// Tail of the LRU list (least recently used).
    tail: NodeIndex,
    /// Maximum number of entries.
    maxsize: usize,
    /// Cache hit count.
    hits: u64,
    /// Cache miss count.
    misses: u64,
}

impl BoundedLruCache {
    /// Create a new bounded cache with the given maximum size.
    fn new(maxsize: usize) -> Self {
        debug_assert!(maxsize > 0, "Use UnboundedCache for maxsize=0");
        Self {
            arena: Vec::with_capacity(maxsize),
            free_list: Vec::new(),
            map: HashMap::with_capacity(maxsize),
            head: NULL_INDEX,
            tail: NULL_INDEX,
            maxsize,
            hits: 0,
            misses: 0,
        }
    }

    /// Get a cached value, moving it to the head (most recent) position.
    fn get(&mut self, key: &HashableValue) -> Option<&Value> {
        if let Some(&idx) = self.map.get(key) {
            self.hits += 1;
            self.move_to_head(idx);
            Some(&self.arena[idx].value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a key-value pair, evicting the LRU entry if at capacity.
    fn put(&mut self, key: HashableValue, value: Value) {
        if let Some(&idx) = self.map.get(&key) {
            // Update existing entry
            self.arena[idx].value = value;
            self.move_to_head(idx);
            return;
        }

        // Evict LRU if at capacity
        if self.map.len() >= self.maxsize {
            self.evict_tail();
        }

        // Allocate or recycle a node
        let idx = self.alloc_node(LruNode {
            key: key.clone(),
            value,
            prev: NULL_INDEX,
            next: NULL_INDEX,
        });

        self.map.insert(key, idx);
        self.push_head(idx);
    }

    /// Check if a key exists in the cache (does not affect LRU ordering).
    #[inline]
    fn contains(&self, key: &HashableValue) -> bool {
        self.map.contains_key(key)
    }

    /// Get the number of cached entries.
    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }

    /// Clear all entries and reset statistics.
    fn clear(&mut self) {
        self.map.clear();
        self.arena.clear();
        self.free_list.clear();
        self.head = NULL_INDEX;
        self.tail = NULL_INDEX;
        self.hits = 0;
        self.misses = 0;
    }

    // =========================================================================
    // Internal: Node Management
    // =========================================================================

    /// Allocate a node from the free list or by appending to the arena.
    #[inline]
    fn alloc_node(&mut self, node: LruNode) -> NodeIndex {
        if let Some(idx) = self.free_list.pop() {
            self.arena[idx] = node;
            idx
        } else {
            let idx = self.arena.len();
            self.arena.push(node);
            idx
        }
    }

    /// Free a node, adding it to the free list for recycling.
    #[inline]
    fn free_node(&mut self, idx: NodeIndex) {
        self.free_list.push(idx);
    }

    // =========================================================================
    // Internal: Linked List Operations
    // =========================================================================

    /// Detach a node from the linked list without freeing it.
    fn detach(&mut self, idx: NodeIndex) {
        let prev = self.arena[idx].prev;
        let next = self.arena[idx].next;

        if prev != NULL_INDEX {
            self.arena[prev].next = next;
        } else {
            self.head = next;
        }

        if next != NULL_INDEX {
            self.arena[next].prev = prev;
        } else {
            self.tail = prev;
        }

        self.arena[idx].prev = NULL_INDEX;
        self.arena[idx].next = NULL_INDEX;
    }

    /// Push a node to the head (most recently used) position.
    fn push_head(&mut self, idx: NodeIndex) {
        self.arena[idx].prev = NULL_INDEX;
        self.arena[idx].next = self.head;

        if self.head != NULL_INDEX {
            self.arena[self.head].prev = idx;
        }
        self.head = idx;

        if self.tail == NULL_INDEX {
            self.tail = idx;
        }
    }

    /// Move an existing node to the head position.
    fn move_to_head(&mut self, idx: NodeIndex) {
        if self.head == idx {
            return; // Already at head
        }
        self.detach(idx);
        self.push_head(idx);
    }

    /// Evict the tail (least recently used) node.
    fn evict_tail(&mut self) {
        if self.tail == NULL_INDEX {
            return;
        }
        let tail_idx = self.tail;
        let key = self.arena[tail_idx].key.clone();
        self.detach(tail_idx);
        self.map.remove(&key);
        self.free_node(tail_idx);
    }

    /// Get cache statistics.
    fn cache_info(&self) -> CacheInfo {
        CacheInfo {
            hits: self.hits,
            misses: self.misses,
            maxsize: Some(self.maxsize),
            currsize: self.map.len(),
        }
    }
}

// =============================================================================
// Unbounded Cache
// =============================================================================

/// An unbounded cache — pure HashMap with no eviction.
///
/// More efficient than bounded mode since no linked list tracking is needed.
#[derive(Debug, Clone)]
struct UnboundedCache {
    map: HashMap<HashableValue, Value>,
    hits: u64,
    misses: u64,
}

impl UnboundedCache {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    fn with_capacity(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &HashableValue) -> Option<&Value> {
        if self.map.contains_key(key) {
            self.hits += 1;
            self.map.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    fn put(&mut self, key: HashableValue, value: Value) {
        self.map.insert(key, value);
    }

    #[inline]
    fn contains(&self, key: &HashableValue) -> bool {
        self.map.contains_key(key)
    }

    #[inline]
    fn len(&self) -> usize {
        self.map.len()
    }

    fn clear(&mut self) {
        self.map.clear();
        self.hits = 0;
        self.misses = 0;
    }

    fn cache_info(&self) -> CacheInfo {
        CacheInfo {
            hits: self.hits,
            misses: self.misses,
            maxsize: None,
            currsize: self.map.len(),
        }
    }
}

// =============================================================================
// Public LRU Cache (Unified Interface)
// =============================================================================

/// High-performance LRU cache with configurable bounds.
///
/// Automatically selects between bounded (linked-list) and unbounded
/// (plain HashMap) implementations based on `maxsize`.
#[derive(Debug, Clone)]
pub enum LruCache {
    /// Bounded cache with maxsize > 0.
    Bounded(BoundedLruCache),
    /// Unbounded cache (maxsize = None).
    Unbounded(UnboundedCache),
}

impl LruCache {
    /// Create a new bounded LRU cache.
    ///
    /// # Panics
    ///
    /// Panics if `maxsize` is 0. Use `LruCache::unbounded()` instead.
    pub fn new(maxsize: usize) -> Self {
        assert!(maxsize > 0, "Use LruCache::unbounded() for unbounded cache");
        LruCache::Bounded(BoundedLruCache::new(maxsize))
    }

    /// Create a new unbounded cache.
    pub fn unbounded() -> Self {
        LruCache::Unbounded(UnboundedCache::new())
    }

    /// Create an unbounded cache with initial capacity hint.
    pub fn unbounded_with_capacity(capacity: usize) -> Self {
        LruCache::Unbounded(UnboundedCache::with_capacity(capacity))
    }

    /// Get a cached value.
    ///
    /// For bounded caches, this moves the entry to the head (most recently used).
    /// Returns `None` on cache miss.
    pub fn get(&mut self, key: &HashableValue) -> Option<&Value> {
        match self {
            LruCache::Bounded(c) => c.get(key),
            LruCache::Unbounded(c) => c.get(key),
        }
    }

    /// Insert a key-value pair into the cache.
    ///
    /// For bounded caches, evicts the LRU entry if at capacity.
    pub fn put(&mut self, key: HashableValue, value: Value) {
        match self {
            LruCache::Bounded(c) => c.put(key, value),
            LruCache::Unbounded(c) => c.put(key, value),
        }
    }

    /// Check if a key exists without affecting LRU ordering.
    #[inline]
    pub fn contains(&self, key: &HashableValue) -> bool {
        match self {
            LruCache::Bounded(c) => c.contains(key),
            LruCache::Unbounded(c) => c.contains(key),
        }
    }

    /// Get cache statistics.
    pub fn cache_info(&self) -> CacheInfo {
        match self {
            LruCache::Bounded(c) => c.cache_info(),
            LruCache::Unbounded(c) => c.cache_info(),
        }
    }

    /// Clear all entries and reset statistics.
    pub fn cache_clear(&mut self) {
        match self {
            LruCache::Bounded(c) => c.clear(),
            LruCache::Unbounded(c) => c.clear(),
        }
    }

    /// Get the current number of cached entries.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            LruCache::Bounded(c) => c.len(),
            LruCache::Unbounded(c) => c.len(),
        }
    }

    /// Check if the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the maximum size (None for unbounded).
    pub fn maxsize(&self) -> Option<usize> {
        match self {
            LruCache::Bounded(c) => Some(c.maxsize),
            LruCache::Unbounded(_) => None,
        }
    }
}
