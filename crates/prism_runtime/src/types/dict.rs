//! Dictionary object implementation.
//!
//! High-performance hash map for Python's dict type.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::hashable::HashableValue;
use crate::types::int::value_to_bigint;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use rustc_hash::FxHashMap;

// =============================================================================
// Dictionary Object
// =============================================================================

/// Python dict object.
///
/// Uses one hash index plus compact ordered slots.
/// Insertion order is preserved to match Python 3.7+ semantics.
#[repr(C)]
#[derive(Debug)]
pub struct DictObject {
    /// Object header.
    pub header: ObjectHeader,
    entries: DictEntries,
    version: u64,
}

#[derive(Debug, Clone, Default)]
struct DictEntries {
    index: FxHashMap<HashableValue, usize>,
    slots: Vec<DictSlot>,
    protocol_key_count: usize,
    tombstones: usize,
}

#[derive(Debug, Clone, Copy)]
struct DictSlot {
    key: HashableValue,
    value: Value,
    hash: i64,
    flags: u8,
}

const SLOT_LIVE: u8 = 0b0001;
const SLOT_PROTOCOL_LOOKUP: u8 = 0b0010;
const SLOT_HAS_HASH: u8 = 0b0100;

impl DictSlot {
    #[inline]
    fn new(
        key: HashableValue,
        value: Value,
        hash: Option<i64>,
        requires_protocol_lookup: bool,
    ) -> Self {
        let mut flags = SLOT_LIVE;
        if requires_protocol_lookup {
            flags |= SLOT_PROTOCOL_LOOKUP;
        }
        if hash.is_some() {
            flags |= SLOT_HAS_HASH;
        }

        Self {
            key,
            value,
            hash: hash.unwrap_or(0),
            flags,
        }
    }

    #[inline]
    fn is_live(self) -> bool {
        self.flags & SLOT_LIVE != 0
    }

    #[inline]
    fn set_live(&mut self, live: bool) {
        if live {
            self.flags |= SLOT_LIVE;
        } else {
            self.flags &= !SLOT_LIVE;
        }
    }

    #[inline]
    fn requires_protocol_lookup(self) -> bool {
        self.flags & SLOT_PROTOCOL_LOOKUP != 0
    }

    #[inline]
    fn mark_protocol_lookup(&mut self) {
        self.flags |= SLOT_PROTOCOL_LOOKUP;
    }

    #[inline]
    fn cached_hash(self) -> Option<i64> {
        (self.flags & SLOT_HAS_HASH != 0).then_some(self.hash)
    }

    #[inline]
    fn set_hash(&mut self, hash: i64) {
        self.hash = hash;
        self.flags |= SLOT_HAS_HASH;
    }
}

impl DictObject {
    /// Create a new empty dict.
    #[inline]
    pub fn new() -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            entries: DictEntries::default(),
            version: 0,
        }
    }

    /// Create a dict with pre-allocated capacity.
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::DICT),
            entries: DictEntries {
                index: FxHashMap::with_capacity_and_hasher(capacity, Default::default()),
                slots: Vec::with_capacity(capacity),
                protocol_key_count: 0,
                tombstones: 0,
            },
            version: 0,
        }
    }

    /// Get the number of items.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.index.len()
    }

    /// Monotonic structural mutation counter for iterator invalidation.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Check if the dict is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.index.is_empty()
    }

    /// Return true when at least one key can require Python-level equality for
    /// collision resolution.
    #[inline]
    pub fn has_protocol_keys(&self) -> bool {
        self.entries.protocol_key_count != 0
    }

    /// Get a value by key.
    #[inline]
    pub fn get(&self, key: Value) -> Option<Value> {
        self.slot_for_key(HashableValue(key)).map(|slot| slot.value)
    }

    /// Set a key-value pair.
    #[inline]
    pub fn set(&mut self, key: Value, value: Value) -> Option<Value> {
        self.insert(key, value, None, requires_protocol_lookup(key))
    }

    /// Set a key-value pair after the VM has computed the Python hash.
    #[inline]
    pub fn set_with_hash(&mut self, key: Value, value: Value, hash: i64) -> Option<Value> {
        self.insert(key, value, Some(hash), requires_protocol_lookup(key))
    }

    /// Set a key-value pair when the caller already classified whether lookup
    /// may need Python-level equality.
    #[inline]
    pub fn set_with_hash_and_protocol_lookup(
        &mut self,
        key: Value,
        value: Value,
        hash: i64,
        requires_protocol_lookup: bool,
    ) -> Option<Value> {
        self.insert(key, value, Some(hash), requires_protocol_lookup)
    }

    #[inline]
    fn insert(
        &mut self,
        key: Value,
        value: Value,
        hash: Option<i64>,
        requires_protocol_lookup: bool,
    ) -> Option<Value> {
        let key = HashableValue(key);
        if let Some(index) = self.entries.index.get(&key).copied() {
            let slot = &mut self.entries.slots[index];
            let old_value = slot.value;
            slot.value = value;
            if let Some(hash) = hash {
                slot.set_hash(hash);
            }
            if requires_protocol_lookup && !slot.requires_protocol_lookup() {
                slot.mark_protocol_lookup();
                self.entries.protocol_key_count += 1;
            }
            return Some(old_value);
        }

        self.bump_version();
        let index = self.entries.slots.len();
        self.entries.index.insert(key, index);
        self.entries
            .slots
            .push(DictSlot::new(key, value, hash, requires_protocol_lookup));
        if requires_protocol_lookup {
            self.entries.protocol_key_count += 1;
        }
        None
    }

    /// Remove a key and return its value.
    #[inline]
    pub fn remove(&mut self, key: Value) -> Option<Value> {
        let key = HashableValue(key);
        let Some((_stored_key, index)) = self.entries.index.remove_entry(&key) else {
            return None;
        };

        self.bump_version();
        let value = {
            let slot = &mut self.entries.slots[index];
            if slot.requires_protocol_lookup() {
                self.entries.protocol_key_count =
                    self.entries.protocol_key_count.saturating_sub(1);
            }
            slot.set_live(false);
            slot.value
        };
        self.entries.tombstones += 1;
        self.maybe_compact_order();
        Some(value)
    }

    /// Check if the dict contains a key.
    #[inline]
    pub fn contains_key(&self, key: Value) -> bool {
        self.entries.index.contains_key(&HashableValue(key))
    }

    /// Clear all items.
    #[inline]
    pub fn clear(&mut self) {
        if !self.entries.index.is_empty() {
            self.bump_version();
        }
        self.entries.index.clear();
        self.entries.slots.clear();
        self.entries.protocol_key_count = 0;
        self.entries.tombstones = 0;
    }

    /// Get an iterator over keys.
    pub fn keys(&self) -> impl Iterator<Item = Value> + '_ {
        self.entries
            .slots
            .iter()
            .filter_map(|slot| slot.is_live().then_some(slot.key.0))
    }

    /// Get an iterator over values.
    pub fn values(&self) -> impl Iterator<Item = Value> + '_ {
        self.entries
            .slots
            .iter()
            .filter_map(|slot| slot.is_live().then_some(slot.value))
    }

    /// Get an iterator over key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Value, Value)> + '_ {
        self.entries
            .slots
            .iter()
            .filter_map(|slot| slot.is_live().then_some((slot.key.0, slot.value)))
    }

    /// Return the next live key-value pair at or after an insertion-order cursor.
    ///
    /// The cursor is advanced past skipped tombstones and the returned entry.
    /// This lets dict-view iterators preserve CPython's live-view semantics
    /// without materializing the values they have not yielded yet.
    pub fn next_entry_from(&self, cursor: &mut usize) -> Option<(Value, Value)> {
        while *cursor < self.entries.slots.len() {
            let index = *cursor;
            *cursor += 1;

            let slot = self.entries.slots[index];
            if slot.is_live() {
                return Some((slot.key.0, slot.value));
            }
        }
        None
    }

    /// Return the cached Python hash for a stored key, when available.
    #[inline]
    pub fn stored_hash(&self, key: Value) -> Option<i64> {
        self.slot_for_key(HashableValue(key))
            .and_then(|slot| slot.cached_hash())
    }

    /// Update this dict with items from another.
    pub fn update(&mut self, other: &DictObject) {
        for (key, value) in other.iter() {
            if let Some(hash) = other.stored_hash(key) {
                self.set_with_hash(key, value, hash);
            } else {
                self.set(key, value);
            }
        }
    }

    /// Get value or insert default.
    pub fn get_or_insert(&mut self, key: Value, default: Value) -> Value {
        if let Some(existing) = self.get(key) {
            return existing;
        }

        self.set(key, default);
        default
    }

    /// Return the current value for a key, inserting a default when absent.
    #[inline]
    pub fn setdefault(&mut self, key: Value, default: Value) -> Value {
        self.get_or_insert(key, default)
    }

    /// Move an existing key to either end of the insertion order.
    pub fn move_to_end(&mut self, key: Value, last: bool) -> bool {
        let key = HashableValue(key);
        let Some(index) = self.entries.index.get(&key).copied() else {
            return false;
        };

        if last {
            if self.last_live_slot_index() == Some(index) {
                return true;
            }

            self.bump_version();
            let mut moved = self.entries.slots[index];
            self.entries.slots[index].set_live(false);
            self.entries.tombstones += 1;
            moved.set_live(true);
            let new_index = self.entries.slots.len();
            self.entries.index.insert(moved.key, new_index);
            self.entries.slots.push(moved);
            self.maybe_compact_order();
            return true;
        }

        if self.first_live_slot_index() != Some(index) {
            self.bump_version();
        }
        self.compact_order_with_prefix(index);
        true
    }

    /// Pop a key and return (key, value) or None.
    pub fn popitem(&mut self) -> Option<(Value, Value)> {
        loop {
            let slot = self.entries.slots.pop()?;
            if !slot.is_live() {
                self.entries.tombstones = self.entries.tombstones.saturating_sub(1);
                continue;
            }

            self.bump_version();
            self.entries.index.remove(&slot.key);
            if slot.requires_protocol_lookup() {
                self.entries.protocol_key_count =
                    self.entries.protocol_key_count.saturating_sub(1);
            }
            return Some((slot.key.0, slot.value));
        }
    }

    #[inline]
    fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    #[inline]
    fn maybe_compact_order(&mut self) {
        if self.entries.tombstones > 16 && self.entries.tombstones > self.entries.index.len() {
            self.compact_order();
        }
    }

    fn compact_order(&mut self) {
        if self.entries.tombstones == 0 {
            return;
        }

        let live_len = self.entries.index.len();
        let old_slots = std::mem::take(&mut self.entries.slots);
        self.entries.index.clear();
        self.entries.protocol_key_count = 0;
        self.entries.slots = Vec::with_capacity(live_len);

        for slot in old_slots.into_iter().filter(|slot| slot.is_live()) {
            self.push_compacted_slot(slot);
        }
        self.entries.tombstones = 0;
    }

    fn compact_order_with_prefix(&mut self, prefix_index: usize) {
        let Some(prefix) = self.entries.slots.get(prefix_index).copied() else {
            return;
        };
        if !prefix.is_live() {
            return;
        }

        let live_len = self.entries.index.len();
        let old_slots = std::mem::take(&mut self.entries.slots);
        self.entries.index.clear();
        self.entries.protocol_key_count = 0;
        self.entries.slots = Vec::with_capacity(live_len);

        self.push_compacted_slot(prefix);
        for slot in old_slots.into_iter().filter(|slot| slot.is_live()) {
            if slot.key == prefix.key {
                continue;
            }
            self.push_compacted_slot(slot);
        }

        self.entries.tombstones = 0;
    }

    #[inline]
    fn slot_for_key(&self, key: HashableValue) -> Option<&DictSlot> {
        let index = self.entries.index.get(&key).copied()?;
        let slot = self.entries.slots.get(index)?;
        debug_assert!(slot.is_live());
        slot.is_live().then_some(slot)
    }

    #[inline]
    fn first_live_slot_index(&self) -> Option<usize> {
        self.entries.slots.iter().position(|slot| slot.is_live())
    }

    #[inline]
    fn last_live_slot_index(&self) -> Option<usize> {
        self.entries.slots.iter().rposition(|slot| slot.is_live())
    }

    #[inline]
    fn push_compacted_slot(&mut self, mut slot: DictSlot) {
        slot.set_live(true);
        let index = self.entries.slots.len();
        self.entries.index.insert(slot.key, index);
        if slot.requires_protocol_lookup() {
            self.entries.protocol_key_count += 1;
        }
        self.entries.slots.push(slot);
    }
}

#[inline]
fn requires_protocol_lookup(value: Value) -> bool {
    !is_fast_lookup_key(value)
}

fn is_fast_lookup_key(value: Value) -> bool {
    if value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_to_bigint(value).is_some()
        || value.is_none()
        || value.is_string()
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    match unsafe { (*(ptr as *const ObjectHeader)).type_id } {
        TypeId::STR => true,
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple.iter().copied().all(is_fast_lookup_key)
        }
        _ => false,
    }
}

impl Default for DictObject {
    fn default() -> Self {
        Self::new()
    }
}

impl PyObject for DictObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn int(value: i64) -> Value {
        Value::int_unchecked(value)
    }

    fn ints(values: impl Iterator<Item = Value>) -> Vec<i64> {
        values
            .map(|value| value.as_int().expect("test value should be an int"))
            .collect()
    }

    #[test]
    fn overwrite_preserves_order_and_structural_version() {
        let mut dict = DictObject::new();

        assert_eq!(dict.set(int(1), int(10)), None);
        assert_eq!(dict.set(int(2), int(20)), None);
        let version_after_inserts = dict.version();
        assert_eq!(dict.set(int(1), int(99)), Some(int(10)));

        assert_eq!(dict.len(), 2);
        assert_eq!(dict.version(), version_after_inserts);
        assert_eq!(ints(dict.keys()), vec![1, 2]);
        assert_eq!(ints(dict.values()), vec![99, 20]);
    }

    #[test]
    fn remove_and_compaction_preserve_live_order() {
        let mut dict = DictObject::with_capacity(48);
        for key in 0..40 {
            dict.set(int(key), int(key * 10));
        }

        for key in 0..25 {
            assert_eq!(dict.remove(int(key)), Some(int(key * 10)));
        }

        assert_eq!(dict.len(), 15);
        assert_eq!(dict.get(int(4)), None);
        assert_eq!(dict.get(int(39)), Some(int(390)));
        assert_eq!(ints(dict.keys()), (25..40).collect::<Vec<_>>());
    }

    #[test]
    fn cached_hash_moves_with_ordered_slot() {
        let mut dict = DictObject::new();
        dict.set_with_hash(int(1), int(10), 111);
        dict.set_with_hash(int(2), int(20), 222);

        assert!(dict.move_to_end(int(1), true));

        assert_eq!(dict.stored_hash(int(1)), Some(111));
        assert_eq!(dict.stored_hash(int(2)), Some(222));
        assert_eq!(ints(dict.keys()), vec![2, 1]);
    }

    #[test]
    fn move_to_front_and_popitem_follow_python_order() {
        let mut dict = DictObject::new();
        for key in 1..=4 {
            dict.set(int(key), int(key * 10));
        }

        assert!(dict.move_to_end(int(3), false));
        assert_eq!(ints(dict.keys()), vec![3, 1, 2, 4]);
        assert_eq!(dict.popitem(), Some((int(4), int(40))));
        assert_eq!(dict.popitem(), Some((int(2), int(20))));
        assert_eq!(ints(dict.keys()), vec![3, 1]);
    }

    #[test]
    fn live_cursor_skips_tombstones_without_materializing() {
        let mut dict = DictObject::new();
        for key in 0..6 {
            dict.set(int(key), int(key + 100));
        }
        dict.remove(int(1));
        dict.remove(int(3));

        let mut cursor = 0;
        let mut seen = Vec::new();
        while let Some((key, value)) = dict.next_entry_from(&mut cursor) {
            seen.push((key.as_int().unwrap(), value.as_int().unwrap()));
        }

        assert_eq!(seen, vec![(0, 100), (2, 102), (4, 104), (5, 105)]);
    }
}
