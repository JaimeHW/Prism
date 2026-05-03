//! Dictionary object implementation.
//!
//! High-performance hash map for Python's dict type.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::hashable::HashableValue;
use crate::types::int::value_to_bigint;
use crate::types::tuple::TupleObject;
use prism_core::Value;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};

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
    index: FxHashMap<DictIndexKey, usize>,
    slots: Vec<DictSlot>,
    live_count: usize,
    hashed_key_count: usize,
    protocol_key_count: usize,
    tombstones: usize,
}

#[derive(Debug, Clone, Copy)]
struct DictIndexKey {
    key: HashableValue,
    hash: u64,
}

impl DictIndexKey {
    #[inline]
    fn new(key: HashableValue, hash: u64) -> Self {
        Self { key, hash }
    }
}

impl PartialEq for DictIndexKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for DictIndexKey {}

impl Hash for DictIndexKey {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
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
    fn has_cached_hash(self) -> bool {
        self.flags & SLOT_HAS_HASH != 0
    }

    #[inline]
    fn set_hash(&mut self, hash: i64) {
        self.hash = hash;
        self.flags |= SLOT_HAS_HASH;
    }
}

#[derive(Debug, Clone, Copy)]
enum DictLookupMode {
    Raw,
    PythonHash,
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
                live_count: 0,
                hashed_key_count: 0,
                protocol_key_count: 0,
                tombstones: 0,
            },
            version: 0,
        }
    }

    /// Get the number of items.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.live_count
    }

    /// Monotonic structural mutation counter for iterator invalidation.
    #[inline]
    pub fn version(&self) -> u64 {
        self.version
    }

    /// Check if the dict is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.live_count == 0
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
        self.slot_for_key(
            HashableValue(key),
            index_hash_for_value(key),
            DictLookupMode::Raw,
        )
        .map(|slot| slot.value)
    }

    /// Get a value by key after the caller has already computed Python hash.
    #[inline]
    pub fn get_with_hash(&self, key: Value, hash: i64) -> Option<Value> {
        self.slot_for_key(
            HashableValue(key),
            index_hash_for_python_hash(hash),
            DictLookupMode::PythonHash,
        )
        .map(|slot| slot.value)
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
        let lookup_mode = if hash.is_some() {
            DictLookupMode::PythonHash
        } else {
            DictLookupMode::Raw
        };
        let index_key = DictIndexKey::new(
            key,
            hash.map_or_else(|| index_hash_for_value(key.0), index_hash_for_python_hash),
        );
        if let Some(index) = self.slot_index_for_key(key, index_key.hash, lookup_mode) {
            let replacement_index_key = hash
                .map(|hash| DictIndexKey::new(key, index_hash_for_python_hash(hash)))
                .unwrap_or_else(|| self.index_key_for_slot(self.entries.slots[index]));
            if self.entries.index.get(&replacement_index_key).copied() != Some(index) {
                self.remove_slot_index_from_index(index);
                self.entries.index.insert(replacement_index_key, index);
            }
            let slot = &mut self.entries.slots[index];
            let old_value = slot.value;
            let was_hashless = !slot.has_cached_hash();
            slot.value = value;
            if let Some(hash) = hash {
                slot.set_hash(hash);
                if was_hashless {
                    self.entries.hashed_key_count += 1;
                }
            }
            if requires_protocol_lookup && !slot.requires_protocol_lookup() {
                slot.mark_protocol_lookup();
                self.entries.protocol_key_count += 1;
            }
            return Some(old_value);
        }

        self.bump_version();
        let index = self.entries.slots.len();
        self.entries.index.insert(index_key, index);
        self.entries
            .slots
            .push(DictSlot::new(key, value, hash, requires_protocol_lookup));
        self.entries.live_count += 1;
        if hash.is_some() {
            self.entries.hashed_key_count += 1;
        }
        if requires_protocol_lookup {
            self.entries.protocol_key_count += 1;
        }
        None
    }

    /// Remove a key and return its value.
    #[inline]
    pub fn remove(&mut self, key: Value) -> Option<Value> {
        let key = HashableValue(key);
        self.remove_indexed(key, index_hash_for_value(key.0), DictLookupMode::Raw)
    }

    /// Remove a key after the caller has already computed Python hash.
    #[inline]
    pub fn remove_with_hash(&mut self, key: Value, hash: i64) -> Option<Value> {
        self.remove_indexed(
            HashableValue(key),
            index_hash_for_python_hash(hash),
            DictLookupMode::PythonHash,
        )
    }

    #[inline]
    fn remove_indexed(
        &mut self,
        key: HashableValue,
        index_hash: u64,
        mode: DictLookupMode,
    ) -> Option<Value> {
        let index_key = DictIndexKey::new(key, index_hash);
        let index = match self.entries.index.remove_entry(&index_key) {
            Some((_stored_key, index)) => index,
            None => {
                if !self.may_need_cross_hash_fallback(mode) {
                    return None;
                }
                let index = self.linear_slot_index_for_key(key)?;
                self.remove_slot_index_from_index(index);
                index
            }
        };

        self.bump_version();
        let value = {
            let slot = &mut self.entries.slots[index];
            if slot.requires_protocol_lookup() {
                self.entries.protocol_key_count = self.entries.protocol_key_count.saturating_sub(1);
            }
            if slot.has_cached_hash() {
                self.entries.hashed_key_count = self.entries.hashed_key_count.saturating_sub(1);
            }
            slot.set_live(false);
            slot.value
        };
        self.entries.live_count = self.entries.live_count.saturating_sub(1);
        self.entries.tombstones += 1;
        self.maybe_compact_order();
        Some(value)
    }

    /// Check if the dict contains a key.
    #[inline]
    pub fn contains_key(&self, key: Value) -> bool {
        let key = HashableValue(key);
        self.slot_index_for_key(key, index_hash_for_value(key.0), DictLookupMode::Raw)
            .is_some()
    }

    /// Check if the dict contains a key after the caller computed Python hash.
    #[inline]
    pub fn contains_key_with_hash(&self, key: Value, hash: i64) -> bool {
        self.slot_index_for_key(
            HashableValue(key),
            index_hash_for_python_hash(hash),
            DictLookupMode::PythonHash,
        )
        .is_some()
    }

    /// Clear all items.
    #[inline]
    pub fn clear(&mut self) {
        if self.entries.live_count != 0 {
            self.bump_version();
        }
        self.entries.index.clear();
        self.entries.slots.clear();
        self.entries.live_count = 0;
        self.entries.hashed_key_count = 0;
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

    /// Get an iterator over key-value-hash triples without re-indexing keys.
    pub fn iter_with_hashes(&self) -> impl Iterator<Item = (Value, Value, Option<i64>)> + '_ {
        self.entries.slots.iter().filter_map(|slot| {
            slot.is_live()
                .then_some((slot.key.0, slot.value, slot.cached_hash()))
        })
    }

    /// Get an iterator over keys and cached hashes without re-indexing keys.
    pub fn keys_with_hashes(&self) -> impl Iterator<Item = (Value, Option<i64>)> + '_ {
        self.entries
            .slots
            .iter()
            .filter_map(|slot| slot.is_live().then_some((slot.key.0, slot.cached_hash())))
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
        self.slot_for_key(
            HashableValue(key),
            index_hash_for_value(key),
            DictLookupMode::Raw,
        )
        .and_then(|slot| slot.cached_hash())
    }

    /// Update this dict with items from another.
    pub fn update(&mut self, other: &DictObject) {
        for (key, value, hash) in other.iter_with_hashes() {
            if let Some(hash) = hash {
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
        let Some(index) =
            self.slot_index_for_key(key, index_hash_for_value(key.0), DictLookupMode::Raw)
        else {
            return false;
        };

        if last {
            if self.last_live_slot_index() == Some(index) {
                return true;
            }

            self.bump_version();
            let mut moved = self.entries.slots[index];
            self.remove_slot_index_from_index(index);
            self.entries.slots[index].set_live(false);
            self.entries.tombstones += 1;
            moved.set_live(true);
            let new_index = self.entries.slots.len();
            self.entries
                .index
                .insert(self.index_key_for_slot(moved), new_index);
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
            self.remove_slot_index_from_index(self.entries.slots.len());
            if slot.requires_protocol_lookup() {
                self.entries.protocol_key_count = self.entries.protocol_key_count.saturating_sub(1);
            }
            if slot.has_cached_hash() {
                self.entries.hashed_key_count = self.entries.hashed_key_count.saturating_sub(1);
            }
            self.entries.live_count = self.entries.live_count.saturating_sub(1);
            return Some((slot.key.0, slot.value));
        }
    }

    #[inline]
    fn bump_version(&mut self) {
        self.version = self.version.wrapping_add(1);
    }

    #[inline]
    fn maybe_compact_order(&mut self) {
        if self.entries.tombstones > 16 && self.entries.tombstones > self.entries.live_count {
            self.compact_order();
        }
    }

    fn compact_order(&mut self) {
        if self.entries.tombstones == 0 {
            return;
        }

        let live_len = self.entries.live_count;
        let old_slots = std::mem::take(&mut self.entries.slots);
        self.entries.index.clear();
        self.entries.live_count = 0;
        self.entries.hashed_key_count = 0;
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

        let live_len = self.entries.live_count;
        let old_slots = std::mem::take(&mut self.entries.slots);
        self.entries.index.clear();
        self.entries.live_count = 0;
        self.entries.hashed_key_count = 0;
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
    fn slot_for_key(
        &self,
        key: HashableValue,
        index_hash: u64,
        mode: DictLookupMode,
    ) -> Option<&DictSlot> {
        let index = self.slot_index_for_key(key, index_hash, mode)?;
        let slot = self.entries.slots.get(index)?;
        debug_assert!(slot.is_live());
        slot.is_live().then_some(slot)
    }

    #[inline]
    fn slot_index_for_key(
        &self,
        key: HashableValue,
        index_hash: u64,
        mode: DictLookupMode,
    ) -> Option<usize> {
        self.entries
            .index
            .get(&DictIndexKey::new(key, index_hash))
            .copied()
            .filter(|index| {
                self.entries
                    .slots
                    .get(*index)
                    .is_some_and(|slot| slot.is_live())
            })
            .or_else(|| {
                self.may_need_cross_hash_fallback(mode)
                    .then(|| self.linear_slot_index_for_key(key))
                    .flatten()
            })
    }

    #[inline]
    fn may_need_cross_hash_fallback(&self, mode: DictLookupMode) -> bool {
        match mode {
            DictLookupMode::Raw => self.entries.hashed_key_count != 0,
            DictLookupMode::PythonHash => self.entries.hashed_key_count != self.entries.live_count,
        }
    }

    #[inline]
    fn linear_slot_index_for_key(&self, key: HashableValue) -> Option<usize> {
        self.entries
            .slots
            .iter()
            .position(|slot| slot.is_live() && slot.key == key)
    }

    #[inline]
    fn remove_slot_index_from_index(&mut self, slot_index: usize) {
        self.entries.index.retain(|_, index| *index != slot_index);
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
        self.entries
            .index
            .insert(self.index_key_for_slot(slot), index);
        if slot.requires_protocol_lookup() {
            self.entries.protocol_key_count += 1;
        }
        if slot.has_cached_hash() {
            self.entries.hashed_key_count += 1;
        }
        self.entries.live_count += 1;
        self.entries.slots.push(slot);
    }

    #[inline]
    fn index_key_for_slot(&self, slot: DictSlot) -> DictIndexKey {
        let hash = slot.cached_hash().map_or_else(
            || index_hash_for_value(slot.key.0),
            index_hash_for_python_hash,
        );
        DictIndexKey::new(slot.key, hash)
    }
}

#[inline]
fn index_hash_for_python_hash(hash: i64) -> u64 {
    hash as u64
}

#[inline]
fn index_hash_for_value(value: Value) -> u64 {
    let mut hasher = FxHasher::default();
    HashableValue(value).hash(&mut hasher);
    hasher.finish()
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
    fn hashed_and_raw_paths_share_one_entry() {
        let mut dict = DictObject::new();

        dict.set(int(1), int(10));
        assert_eq!(dict.set_with_hash(int(1), int(11), 1), Some(int(10)));
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.get_with_hash(int(1), 1), Some(int(11)));
        assert_eq!(dict.stored_hash(int(1)), Some(1));

        assert_eq!(dict.set(int(1), int(12)), Some(int(11)));
        assert_eq!(dict.len(), 1);
        assert_eq!(dict.get(int(1)), Some(int(12)));
        assert_eq!(dict.remove_with_hash(int(1), 1), Some(int(12)));
        assert!(dict.is_empty());
    }

    #[test]
    fn hash_iterators_read_slot_hashes_directly() {
        let mut dict = DictObject::new();
        dict.set_with_hash(int(1), int(10), 101);
        dict.set(int(2), int(20));

        let entries = dict.iter_with_hashes().collect::<Vec<_>>();
        assert_eq!(entries[0], (int(1), int(10), Some(101)));
        assert_eq!(entries[1], (int(2), int(20), None));

        let keys = dict.keys_with_hashes().collect::<Vec<_>>();
        assert_eq!(keys, vec![(int(1), Some(101)), (int(2), None)]);
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

    #[test]
    fn slot_layout_stays_cache_compact() {
        assert!(std::mem::size_of::<DictSlot>() <= 32);
    }
}
