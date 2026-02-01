//! Arena allocator for IR nodes.
//!
//! The arena provides:
//! - **O(1) allocation**: Bump pointer with no deallocation overhead
//! - **Cache-friendly**: Nodes are contiguous in memory
//! - **Fast iteration**: Linear traversal of all nodes
//! - **Zero-cost IDs**: NodeId is just an index into the arena
//!
//! This is critical for JIT performance as node manipulation is the
//! dominant cost in optimization passes.

use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

// =============================================================================
// Arena Configuration
// =============================================================================

/// Default initial capacity (nodes).
const DEFAULT_INITIAL_CAPACITY: usize = 256;

/// Growth factor when reallocating.
const GROWTH_FACTOR: usize = 2;

// =============================================================================
// Typed ID
// =============================================================================

/// A type-safe identifier for arena-allocated items.
///
/// The generic parameter `T` ensures you can't mix up IDs from different arenas.
/// We implement traits manually to ensure Id<T> is always Copy/Clone/Hash/Eq
/// regardless of whether T implements those traits.
pub struct Id<T> {
    index: u32,
    _marker: PhantomData<fn() -> T>,
}

// Manual implementations to ensure Copy/Clone work regardless of T
impl<T> Copy for Id<T> {}

impl<T> Clone for Id<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for Id<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for Id<T> {}

impl<T> PartialOrd for Id<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T> Ord for Id<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.index.cmp(&other.index)
    }
}

impl<T> std::hash::Hash for Id<T> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> Id<T> {
    /// Create a new ID from a raw index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Id {
            index,
            _marker: PhantomData,
        }
    }

    /// Get the raw index.
    #[inline]
    pub const fn index(self) -> u32 {
        self.index
    }

    /// Get the index as usize.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.index as usize
    }

    /// Invalid/null ID.
    pub const INVALID: Self = Id {
        index: u32::MAX,
        _marker: PhantomData,
    };

    /// Check if this ID is valid.
    #[inline]
    pub const fn is_valid(self) -> bool {
        self.index != u32::MAX
    }
}

impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_valid() {
            write!(f, "#{}", self.index)
        } else {
            write!(f, "#INVALID")
        }
    }
}

impl<T> std::fmt::Display for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.index)
    }
}

impl<T> Default for Id<T> {
    fn default() -> Self {
        Self::INVALID
    }
}

// =============================================================================
// Arena
// =============================================================================

/// A simple arena allocator for homogeneous items.
///
/// Items are stored contiguously in memory and can be accessed by ID.
/// The arena never deallocates individual items - the entire arena is freed at once.
#[derive(Debug, Clone)]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Arena<T> {
    /// Create a new empty arena.
    #[inline]
    pub fn new() -> Self {
        Arena { items: Vec::new() }
    }

    /// Create a new arena with the given initial capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Arena {
            items: Vec::with_capacity(capacity),
        }
    }

    /// Allocate a new item and return its ID.
    #[inline]
    pub fn alloc(&mut self, item: T) -> Id<T> {
        let index = self.items.len() as u32;
        self.items.push(item);
        Id::new(index)
    }

    /// Get a reference to an item by ID.
    #[inline]
    pub fn get(&self, id: Id<T>) -> Option<&T> {
        self.items.get(id.as_usize())
    }

    /// Get a mutable reference to an item by ID.
    #[inline]
    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        self.items.get_mut(id.as_usize())
    }

    /// Get the number of items in the arena.
    #[inline]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Check if the arena is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Iterate over all items with their IDs.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(i, item)| (Id::new(i as u32), item))
    }

    /// Iterate over all items mutably with their IDs.
    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Id<T>, &mut T)> {
        self.items
            .iter_mut()
            .enumerate()
            .map(|(i, item)| (Id::new(i as u32), item))
    }

    /// Iterate over all IDs.
    #[inline]
    pub fn ids(&self) -> impl Iterator<Item = Id<T>> {
        (0..self.items.len() as u32).map(Id::new)
    }

    /// Reserve capacity for at least `additional` more items.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.items.reserve(additional);
    }

    /// Clear the arena, removing all items.
    #[inline]
    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Get the next ID that will be allocated.
    #[inline]
    pub fn next_id(&self) -> Id<T> {
        Id::new(self.items.len() as u32)
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Index<Id<T>> for Arena<T> {
    type Output = T;

    #[inline]
    fn index(&self, id: Id<T>) -> &Self::Output {
        &self.items[id.as_usize()]
    }
}

impl<T> IndexMut<Id<T>> for Arena<T> {
    #[inline]
    fn index_mut(&mut self, id: Id<T>) -> &mut Self::Output {
        &mut self.items[id.as_usize()]
    }
}

// =============================================================================
// Secondary Map
// =============================================================================

/// A secondary map that associates additional data with arena items.
///
/// This is useful for storing computed properties (e.g., liveness, dominator info)
/// without modifying the node structure.
#[derive(Debug, Clone)]
pub struct SecondaryMap<K, V> {
    values: Vec<V>,
    _marker: PhantomData<K>,
}

impl<K, V: Default + Clone> SecondaryMap<K, V> {
    /// Create a new empty secondary map.
    pub fn new() -> Self {
        SecondaryMap {
            values: Vec::new(),
            _marker: PhantomData,
        }
    }

    /// Create a new secondary map with capacity for the given arena size.
    pub fn with_capacity(capacity: usize) -> Self {
        SecondaryMap {
            values: vec![V::default(); capacity],
            _marker: PhantomData,
        }
    }

    /// Ensure the map can hold up to the given index.
    pub fn resize(&mut self, len: usize) {
        if len > self.values.len() {
            self.values.resize(len, V::default());
        }
    }

    /// Get a value by ID.
    pub fn get(&self, id: Id<K>) -> Option<&V> {
        self.values.get(id.as_usize())
    }

    /// Get a mutable value by ID.
    pub fn get_mut(&mut self, id: Id<K>) -> Option<&mut V> {
        self.values.get_mut(id.as_usize())
    }

    /// Set a value by ID.
    pub fn set(&mut self, id: Id<K>, value: V) {
        let idx = id.as_usize();
        if idx >= self.values.len() {
            self.values.resize(idx + 1, V::default());
        }
        self.values[idx] = value;
    }

    /// Clear all values.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Iterate over all values.
    pub fn iter(&self) -> impl Iterator<Item = (Id<K>, &V)> {
        self.values
            .iter()
            .enumerate()
            .map(|(i, v)| (Id::new(i as u32), v))
    }
}

impl<K, V: Default + Clone> Default for SecondaryMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V: Default + Clone> Index<Id<K>> for SecondaryMap<K, V> {
    type Output = V;

    fn index(&self, id: Id<K>) -> &Self::Output {
        &self.values[id.as_usize()]
    }
}

impl<K, V: Default + Clone> IndexMut<Id<K>> for SecondaryMap<K, V> {
    fn index_mut(&mut self, id: Id<K>) -> &mut Self::Output {
        &mut self.values[id.as_usize()]
    }
}

// =============================================================================
// Bit Set
// =============================================================================

/// A compact bit set for tracking node properties.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BitSet {
    bits: Vec<u64>,
}

impl BitSet {
    /// Create a new empty bit set.
    pub fn new() -> Self {
        BitSet { bits: Vec::new() }
    }

    /// Create a new bit set with capacity for `n` bits.
    pub fn with_capacity(n: usize) -> Self {
        let words = (n + 63) / 64;
        BitSet {
            bits: vec![0; words],
        }
    }

    /// Ensure the bit set can hold at least `n` bits.
    pub fn ensure_capacity(&mut self, n: usize) {
        let words = (n + 63) / 64;
        if words > self.bits.len() {
            self.bits.resize(words, 0);
        }
    }

    /// Set a bit.
    #[inline]
    pub fn insert(&mut self, index: usize) {
        self.ensure_capacity(index + 1);
        let word = index / 64;
        let bit = index % 64;
        self.bits[word] |= 1 << bit;
    }

    /// Clear a bit.
    #[inline]
    pub fn remove(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        if word < self.bits.len() {
            self.bits[word] &= !(1 << bit);
        }
    }

    /// Check if a bit is set.
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        if word < self.bits.len() {
            (self.bits[word] & (1 << bit)) != 0
        } else {
            false
        }
    }

    /// Clear all bits.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Union with another bit set (self |= other).
    pub fn union_with(&mut self, other: &BitSet) {
        if other.bits.len() > self.bits.len() {
            self.bits.resize(other.bits.len(), 0);
        }
        for (i, &word) in other.bits.iter().enumerate() {
            self.bits[i] |= word;
        }
    }

    /// Intersect with another bit set (self &= other).
    pub fn intersect_with(&mut self, other: &BitSet) {
        for (i, word) in self.bits.iter_mut().enumerate() {
            if i < other.bits.len() {
                *word &= other.bits[i];
            } else {
                *word = 0;
            }
        }
    }

    /// Check if the bit set is empty.
    pub fn is_empty(&self) -> bool {
        self.bits.iter().all(|&w| w == 0)
    }

    /// Count the number of set bits.
    pub fn count(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate over set bit indices.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.bits.iter().enumerate().flat_map(|(word_idx, &word)| {
            (0..64).filter_map(move |bit| {
                if (word & (1 << bit)) != 0 {
                    Some(word_idx * 64 + bit)
                } else {
                    None
                }
            })
        })
    }
}

impl Default for BitSet {
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

    struct TestNode {
        value: i32,
    }

    #[test]
    fn test_arena_alloc() {
        let mut arena: Arena<TestNode> = Arena::new();

        let id1 = arena.alloc(TestNode { value: 10 });
        let id2 = arena.alloc(TestNode { value: 20 });
        let id3 = arena.alloc(TestNode { value: 30 });

        assert_eq!(id1.index(), 0);
        assert_eq!(id2.index(), 1);
        assert_eq!(id3.index(), 2);

        assert_eq!(arena[id1].value, 10);
        assert_eq!(arena[id2].value, 20);
        assert_eq!(arena[id3].value, 30);

        arena[id2].value = 200;
        assert_eq!(arena[id2].value, 200);
    }

    #[test]
    fn test_arena_iter() {
        let mut arena: Arena<TestNode> = Arena::new();

        arena.alloc(TestNode { value: 1 });
        arena.alloc(TestNode { value: 2 });
        arena.alloc(TestNode { value: 3 });

        let values: Vec<_> = arena.iter().map(|(_, n)| n.value).collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_secondary_map() {
        let mut arena: Arena<TestNode> = Arena::new();
        let id1 = arena.alloc(TestNode { value: 10 });
        let id2 = arena.alloc(TestNode { value: 20 });

        let mut map: SecondaryMap<TestNode, String> = SecondaryMap::new();
        map.set(id1, "first".to_string());
        map.set(id2, "second".to_string());

        assert_eq!(map[id1], "first");
        assert_eq!(map[id2], "second");
    }

    #[test]
    fn test_bit_set() {
        let mut set = BitSet::new();

        set.insert(0);
        set.insert(5);
        set.insert(63);
        set.insert(64);
        set.insert(100);

        assert!(set.contains(0));
        assert!(set.contains(5));
        assert!(set.contains(63));
        assert!(set.contains(64));
        assert!(set.contains(100));
        assert!(!set.contains(1));
        assert!(!set.contains(65));

        assert_eq!(set.count(), 5);

        let indices: Vec<_> = set.iter().collect();
        assert_eq!(indices, vec![0, 5, 63, 64, 100]);
    }

    #[test]
    fn test_bit_set_union_intersect() {
        let mut set1 = BitSet::new();
        set1.insert(0);
        set1.insert(2);
        set1.insert(4);

        let mut set2 = BitSet::new();
        set2.insert(1);
        set2.insert(2);
        set2.insert(3);

        let mut union = set1.clone();
        union.union_with(&set2);
        assert_eq!(union.count(), 5); // 0,1,2,3,4

        let mut intersect = set1.clone();
        intersect.intersect_with(&set2);
        assert_eq!(intersect.count(), 1); // just 2
        assert!(intersect.contains(2));
    }

    #[test]
    fn test_id_invalid() {
        let id: Id<TestNode> = Id::INVALID;
        assert!(!id.is_valid());

        let valid_id: Id<TestNode> = Id::new(0);
        assert!(valid_id.is_valid());
    }
}
