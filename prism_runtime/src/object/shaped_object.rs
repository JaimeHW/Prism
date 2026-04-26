//! ShapedObject - Python object with hidden class optimization.
//!
//! This module provides the core object type that uses Shape-based hidden classes
//! for O(1) property access via inline property storage.
//!
//! # Architecture
//!
//! ShapedObject combines:
//! - A shape pointer describing property layout
//! - Fixed inline slots for fast property access
//! - Optional overflow dictionary for properties beyond inline capacity
//! - Standard ObjectHeader for GC and type information
//!
//! # Performance
//!
//! Property access is O(1) when shapes match:
//! 1. Check shape ID matches cached shape
//! 2. Read/write directly from inline slot at known offset
//!
//! On shape miss, fallback to shape lookup (still fast for small objects).

use super::shape::{MAX_INLINE_SLOTS, PropertyFlags, Shape, ShapeId};
use super::{ObjectHeader, PyObject};
use crate::object::type_obj::TypeId;
use crate::types::bytes::BytesObject;
use crate::types::dict::DictObject;
use crate::types::list::ListObject;
use crate::types::string::StringObject;
use crate::types::tuple::TupleObject;
use num_bigint::BigInt;
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock};

/// Marker payload used to represent a deleted inline attribute slot.
///
/// Deleted attributes must be distinguishable from attributes explicitly set
/// to `None`, so we store a private marker object in inline slots.
#[repr(C)]
struct DeletedPropertyMarker {
    header: ObjectHeader,
}

static DELETED_PROPERTY_MARKER_PTR: LazyLock<usize> = LazyLock::new(|| {
    let marker = DeletedPropertyMarker {
        header: ObjectHeader::new(TypeId::OBJECT),
    };
    Box::into_raw(Box::new(marker)) as usize
});

#[inline(always)]
fn deleted_property_value() -> Value {
    Value::object_ptr(*DELETED_PROPERTY_MARKER_PTR as *const ())
}

#[inline(always)]
fn is_deleted_property(value: Value) -> bool {
    value
        .as_object_ptr()
        .is_some_and(|ptr| ptr as usize == *DELETED_PROPERTY_MARKER_PTR)
}

// =============================================================================
// Inline Slots
// =============================================================================

/// Fixed-size inline storage for object properties.
///
/// Stores up to MAX_INLINE_SLOTS Values directly in the object,
/// avoiding dictionary overhead for common cases.
#[derive(Debug, Clone)]
pub struct InlineSlots {
    /// Fixed array of slots.
    slots: [Value; MAX_INLINE_SLOTS],
    /// Number of slots currently in use.
    used: u16,
}

impl Default for InlineSlots {
    fn default() -> Self {
        Self::new()
    }
}

impl InlineSlots {
    /// Create new empty inline slots.
    #[inline]
    pub fn new() -> Self {
        Self {
            // Initialize all slots to None
            slots: std::array::from_fn(|_| Value::none()),
            used: 0,
        }
    }

    /// Get a slot value by index.
    ///
    /// # Panics
    /// Panics if index >= MAX_INLINE_SLOTS.
    #[inline]
    pub fn get(&self, index: u16) -> Value {
        self.slots[index as usize]
    }

    /// Set a slot value by index.
    ///
    /// # Panics
    /// Panics if index >= MAX_INLINE_SLOTS.
    #[inline]
    pub fn set(&mut self, index: u16, value: Value) {
        let idx = index as usize;
        if idx >= self.used as usize {
            self.used = index + 1;
        }
        self.slots[idx] = value;
    }

    /// Get number of slots in use.
    #[inline]
    pub fn used(&self) -> u16 {
        self.used
    }

    /// Iterate over used slots.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &Value)> {
        self.slots[..self.used as usize]
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u16, v))
    }
}

// =============================================================================
// Overflow Storage
// =============================================================================

/// Storage for properties that exceed inline capacity.
///
/// Used when an object has more than MAX_INLINE_SLOTS properties.
/// Falls back to dictionary-based storage with O(1) amortized access.
#[derive(Debug, Clone, Default)]
pub struct OverflowStorage {
    /// Map from property name to value.
    properties: FxHashMap<InternedString, Value>,
}

impl OverflowStorage {
    /// Create new empty overflow storage.
    #[inline]
    pub fn new() -> Self {
        Self {
            properties: FxHashMap::default(),
        }
    }

    /// Get a property by name.
    #[inline]
    pub fn get(&self, name: &InternedString) -> Option<&Value> {
        self.properties.get(name)
    }

    /// Set a property.
    #[inline]
    pub fn set(&mut self, name: InternedString, value: Value) {
        self.properties.insert(name, value);
    }

    /// Check if a property exists.
    #[inline]
    pub fn contains(&self, name: &InternedString) -> bool {
        self.properties.contains_key(name)
    }

    /// Remove a property.
    #[inline]
    pub fn remove(&mut self, name: &InternedString) -> Option<Value> {
        self.properties.remove(name)
    }

    /// Get number of overflow properties.
    #[inline]
    pub fn len(&self) -> usize {
        self.properties.len()
    }

    /// Check if overflow is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.properties.is_empty()
    }

    /// Iterate over overflow properties.
    pub fn iter(&self) -> impl Iterator<Item = (&InternedString, &Value)> {
        self.properties.iter()
    }
}

// =============================================================================
// ShapedObject
// =============================================================================

/// A Python object with hidden class optimization.
///
/// Uses Shape-based property layout for O(1) access to inline properties.
/// Falls back to overflow dictionary for large objects.
#[repr(C)]
#[derive(Debug)]
pub struct ShapedObject {
    /// Standard object header for GC and type info.
    header: ObjectHeader,

    /// Current shape describing property layout.
    shape: Arc<Shape>,

    /// Inline property storage (fast path).
    inline_slots: InlineSlots,

    /// Overflow storage for properties beyond inline capacity.
    /// Lazily allocated only when needed.
    overflow: Option<Box<OverflowStorage>>,

    /// Python-visible instance dictionary for heap instances.
    ///
    /// The shaped slots above remain the cold-start fast path for normal
    /// attribute access. Once `obj.__dict__` is materialized or explicitly
    /// assigned, this mapping becomes the authoritative Python attribute
    /// dictionary and the VM mirrors normal writes into it.
    instance_dict: Option<InstanceDictStorage>,

    /// Optional native dict storage for heap subclasses of `dict`.
    ///
    /// This keeps Python-visible mapping state separate from instance
    /// attributes while allowing builtin `dict` methods and opcodes to
    /// operate on user-defined subclasses with the same semantics as the
    /// built-in type.
    dict_backing: Option<Box<DictObject>>,

    /// Optional native list storage for heap subclasses of `list`.
    ///
    /// This preserves list protocol semantics for heap subclasses while
    /// keeping user-defined instance attributes in the shaped-object storage.
    list_backing: Option<Box<ListObject>>,

    /// Optional native tuple storage for tuple-like heap objects.
    ///
    /// This supports CPython struct-sequence style objects that expose named
    /// fields while still participating in the immutable tuple protocol.
    tuple_backing: Option<Box<TupleObject>>,

    /// Optional native string storage for heap subclasses of `str`.
    ///
    /// This preserves native string semantics while keeping heap instance
    /// attributes in the shaped-object storage.
    string_backing: Option<Box<StringObject>>,

    /// Optional native bytes storage for heap subclasses of `bytes` and
    /// `bytearray`.
    ///
    /// This preserves byte-sequence semantics while keeping heap instance
    /// attributes in the shaped-object storage.
    bytes_backing: Option<Box<BytesObject>>,

    /// Optional native integer storage for heap subclasses of `int`.
    ///
    /// The payload is arbitrary precision, matching Python's visible `int`
    /// semantics while preserving user-defined instance attributes in the
    /// shaped-object storage.
    int_backing: Option<Box<BigInt>>,
}

#[derive(Debug)]
enum InstanceDictStorage {
    Owned(Box<DictObject>),
    External(Value),
}

impl ShapedObject {
    /// Create a new empty ShapedObject with the given type.
    #[inline]
    pub fn new(type_id: TypeId, empty_shape: Arc<Shape>) -> Self {
        Self {
            header: ObjectHeader::new(type_id),
            shape: empty_shape,
            inline_slots: InlineSlots::new(),
            overflow: None,
            instance_dict: None,
            dict_backing: None,
            list_backing: None,
            tuple_backing: None,
            string_backing: None,
            bytes_backing: None,
            int_backing: None,
        }
    }

    /// Create a new ShapedObject with native dict storage.
    #[inline]
    pub fn new_dict_backed(type_id: TypeId, empty_shape: Arc<Shape>) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.dict_backing = Some(Box::new(DictObject::new()));
        object
    }

    /// Create a new ShapedObject with native list storage.
    #[inline]
    pub fn new_list_backed(type_id: TypeId, empty_shape: Arc<Shape>) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.list_backing = Some(Box::new(ListObject::new()));
        object
    }

    /// Create a new ShapedObject with native tuple storage.
    #[inline]
    pub fn new_tuple_backed(type_id: TypeId, empty_shape: Arc<Shape>, tuple: TupleObject) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.tuple_backing = Some(Box::new(tuple));
        object
    }

    /// Create a new ShapedObject with native string storage.
    #[inline]
    pub fn new_string_backed(
        type_id: TypeId,
        empty_shape: Arc<Shape>,
        string: StringObject,
    ) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.string_backing = Some(Box::new(string));
        object
    }

    /// Create a new ShapedObject with native bytes storage.
    #[inline]
    pub fn new_bytes_backed(type_id: TypeId, empty_shape: Arc<Shape>, bytes: BytesObject) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.bytes_backing = Some(Box::new(bytes));
        object
    }

    /// Create a new ShapedObject with native integer storage.
    #[inline]
    pub fn new_int_backed(type_id: TypeId, empty_shape: Arc<Shape>, integer: BigInt) -> Self {
        let mut object = Self::new(type_id, empty_shape);
        object.int_backing = Some(Box::new(integer));
        object
    }

    /// Create a new ShapedObject with default OBJECT type.
    #[inline]
    pub fn with_empty_shape(empty_shape: Arc<Shape>) -> Self {
        Self::new(TypeId::OBJECT, empty_shape)
    }

    /// Check whether this heap instance carries native dict storage.
    #[inline]
    pub fn has_dict_backing(&self) -> bool {
        self.dict_backing.is_some()
    }

    /// Borrow the native dict storage for heap subclasses of `dict`.
    #[inline]
    pub fn dict_backing(&self) -> Option<&DictObject> {
        self.dict_backing.as_deref()
    }

    /// Mutably borrow the native dict storage for heap subclasses of `dict`.
    #[inline]
    pub fn dict_backing_mut(&mut self) -> Option<&mut DictObject> {
        self.dict_backing.as_deref_mut()
    }

    /// Check whether a Python-visible instance dictionary has been materialized.
    #[inline]
    pub fn has_instance_dict(&self) -> bool {
        self.instance_dict.is_some()
    }

    /// Return the current Python-visible instance dictionary value.
    #[inline]
    pub fn instance_dict_value(&self) -> Option<Value> {
        self.instance_dict.as_ref().map(InstanceDictStorage::value)
    }

    /// Materialize and return the Python-visible instance dictionary.
    ///
    /// Existing shaped properties are copied into the dictionary exactly once.
    /// After materialization, the VM treats the dictionary as authoritative and
    /// mirrors normal attribute writes into both representations.
    pub fn ensure_instance_dict_value(&mut self) -> Value {
        if self.instance_dict.is_none() {
            let mut dict = DictObject::with_capacity(self.property_count());
            for (name, value) in self.iter_properties() {
                dict.set(Value::string(name), value);
            }
            self.instance_dict = Some(InstanceDictStorage::Owned(Box::new(dict)));
        }

        self.instance_dict_value()
            .expect("instance dict should be initialized")
    }

    /// Replace the Python-visible instance dictionary with an externally owned mapping.
    #[inline]
    pub fn set_instance_dict_value(&mut self, value: Value) {
        if self.instance_dict_value() == Some(value) {
            return;
        }
        self.instance_dict = Some(InstanceDictStorage::External(value));
    }

    /// Reset the Python-visible instance dictionary to a fresh empty dict.
    #[inline]
    pub fn reset_instance_dict(&mut self) -> Value {
        self.instance_dict = Some(InstanceDictStorage::Owned(Box::new(DictObject::new())));
        self.instance_dict_value()
            .expect("instance dict should be initialized")
    }

    /// Check whether this heap instance carries native list storage.
    #[inline]
    pub fn has_list_backing(&self) -> bool {
        self.list_backing.is_some()
    }

    /// Borrow the native list storage for heap subclasses of `list`.
    #[inline]
    pub fn list_backing(&self) -> Option<&ListObject> {
        self.list_backing.as_deref()
    }

    /// Mutably borrow the native list storage for heap subclasses of `list`.
    #[inline]
    pub fn list_backing_mut(&mut self) -> Option<&mut ListObject> {
        self.list_backing.as_deref_mut()
    }

    /// Check whether this heap instance carries native tuple storage.
    #[inline]
    pub fn has_tuple_backing(&self) -> bool {
        self.tuple_backing.is_some()
    }

    /// Borrow the native tuple storage for tuple-like heap objects.
    #[inline]
    pub fn tuple_backing(&self) -> Option<&TupleObject> {
        self.tuple_backing.as_deref()
    }

    /// Replace the native tuple storage for tuple-like heap objects.
    #[inline]
    pub fn set_tuple_backing(&mut self, tuple: TupleObject) {
        self.tuple_backing = Some(Box::new(tuple));
    }

    /// Check whether this heap instance carries native string storage.
    #[inline]
    pub fn has_string_backing(&self) -> bool {
        self.string_backing.is_some()
    }

    /// Borrow the native string storage for heap subclasses of `str`.
    #[inline]
    pub fn string_backing(&self) -> Option<&StringObject> {
        self.string_backing.as_deref()
    }

    /// Mutably borrow the native string storage for heap subclasses of `str`.
    #[inline]
    pub fn string_backing_mut(&mut self) -> Option<&mut StringObject> {
        self.string_backing.as_deref_mut()
    }

    /// Check whether this heap instance carries native bytes storage.
    #[inline]
    pub fn has_bytes_backing(&self) -> bool {
        self.bytes_backing.is_some()
    }

    /// Borrow the native bytes storage for heap subclasses of `bytes` and
    /// `bytearray`.
    #[inline]
    pub fn bytes_backing(&self) -> Option<&BytesObject> {
        self.bytes_backing.as_deref()
    }

    /// Mutably borrow the native bytes storage for heap subclasses of `bytes`
    /// and `bytearray`.
    #[inline]
    pub fn bytes_backing_mut(&mut self) -> Option<&mut BytesObject> {
        self.bytes_backing.as_deref_mut()
    }

    /// Check whether this heap instance carries native integer storage.
    #[inline]
    pub fn has_int_backing(&self) -> bool {
        self.int_backing.is_some()
    }

    /// Borrow the native integer storage for heap subclasses of `int`.
    #[inline]
    pub fn int_backing(&self) -> Option<&BigInt> {
        self.int_backing.as_deref()
    }

    /// Get the current shape.
    #[inline]
    pub fn shape(&self) -> &Arc<Shape> {
        &self.shape
    }

    /// Get the shape ID for fast comparison.
    #[inline]
    pub fn shape_id(&self) -> ShapeId {
        self.shape.id()
    }

    /// Get a property by name.
    ///
    /// Returns None if the property doesn't exist.
    pub fn get_property(&self, name: &str) -> Option<Value> {
        let interned = prism_core::intern::intern(name);
        self.get_property_interned(&interned)
    }

    /// Get a property using an interned name (faster).
    #[inline]
    pub fn get_property_interned(&self, name: &InternedString) -> Option<Value> {
        // Fast path: shape lookup with interned name
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                let value = self.inline_slots.get(slot_index);
                return if is_deleted_property(value) {
                    None
                } else {
                    Some(value)
                };
            }
            return self
                .overflow
                .as_ref()
                .and_then(|overflow| overflow.get(name).copied());
        }

        // Check overflow storage
        if let Some(overflow) = &self.overflow {
            return overflow.get(name).copied();
        }

        None
    }

    /// Get a property by cached slot index (fastest).
    ///
    /// This is the IC fast path - caller must verify shape_id matches.
    #[inline]
    pub fn get_property_cached(&self, slot_index: u16) -> Value {
        debug_assert!(
            (slot_index as usize) < MAX_INLINE_SLOTS,
            "Cached access only for inline slots"
        );
        self.inline_slots.get(slot_index)
    }

    /// Set a property by name.
    ///
    /// This may transition to a new shape if the property is new.
    /// Returns the new shape if a transition occurred.
    pub fn set_property(
        &mut self,
        name: InternedString,
        value: Value,
        registry: &super::shape::ShapeRegistry,
    ) -> Option<Arc<Shape>> {
        // Check if property already exists in current shape
        if let Some(slot_index) = self.shape.lookup_interned(&name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                // Property exists in inline storage - just update
                self.inline_slots.set(slot_index, value);
                return None;
            }
            // Property exists in overflow storage for current shape.
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
            return None;
        }

        // Property is new - create transition
        let new_shape = registry.transition_default(&self.shape, name.clone());
        let slot_index = new_shape
            .property()
            .expect("New shape must have property")
            .slot_index;

        if (slot_index as usize) < MAX_INLINE_SLOTS {
            // Store in inline slot
            self.inline_slots.set(slot_index, value);
        } else {
            // Spill to overflow
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
        }

        self.shape = new_shape.clone();
        Some(new_shape)
    }

    /// Set a property with custom flags.
    pub fn set_property_with_flags(
        &mut self,
        name: InternedString,
        value: Value,
        flags: PropertyFlags,
        registry: &super::shape::ShapeRegistry,
    ) -> Option<Arc<Shape>> {
        // Check if property already exists
        if let Some(slot_index) = self.shape.lookup_interned(&name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                self.inline_slots.set(slot_index, value);
                return None;
            }
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
            return None;
        }

        // Create transition with custom flags
        let new_shape = registry.transition(&self.shape, name.clone(), flags);
        let slot_index = new_shape.property().unwrap().slot_index;

        if (slot_index as usize) < MAX_INLINE_SLOTS {
            self.inline_slots.set(slot_index, value);
        } else {
            let overflow = self
                .overflow
                .get_or_insert_with(|| Box::new(OverflowStorage::new()));
            overflow.set(name, value);
        }

        self.shape = new_shape.clone();
        Some(new_shape)
    }

    /// Set a property by cached slot index (fastest).
    ///
    /// Caller must verify shape_id matches and property is writable.
    #[inline]
    pub fn set_property_cached(&mut self, slot_index: u16, value: Value) {
        debug_assert!(
            (slot_index as usize) < MAX_INLINE_SLOTS,
            "Cached access only for inline slots"
        );
        self.inline_slots.set(slot_index, value);
    }

    /// Check if a property exists.
    pub fn has_property(&self, name: &str) -> bool {
        let interned = prism_core::intern::intern(name);
        self.has_property_interned(&interned)
    }

    /// Check if a property exists using an interned name.
    pub fn has_property_interned(&self, name: &InternedString) -> bool {
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                return !is_deleted_property(self.inline_slots.get(slot_index));
            }
            return self
                .overflow
                .as_ref()
                .is_some_and(|overflow| overflow.contains(name));
        }
        self.overflow
            .as_ref()
            .is_some_and(|overflow| overflow.contains(name))
    }

    /// Delete a property.
    ///
    /// Note: This doesn't change the shape. Inline slots use a private tombstone
    /// marker so deletion is distinct from assigning `None`.
    /// A more sophisticated implementation could use "delete shapes" like V8.
    pub fn delete_property(&mut self, name: &str) -> bool {
        let interned = prism_core::intern::intern(name);
        self.delete_property_interned(&interned)
    }

    /// Delete a property by interned name.
    ///
    /// For inline slots we install a tombstone marker so deleted attributes are
    /// not confused with attributes explicitly set to `None`.
    pub fn delete_property_interned(&mut self, name: &InternedString) -> bool {
        if let Some(slot_index) = self.shape.lookup_interned(name) {
            if (slot_index as usize) < MAX_INLINE_SLOTS {
                let current = self.inline_slots.get(slot_index);
                if is_deleted_property(current) {
                    return false;
                }
                self.inline_slots.set(slot_index, deleted_property_value());
                return true;
            }
            return self
                .overflow
                .as_mut()
                .is_some_and(|overflow| overflow.remove(name).is_some());
        }

        self.overflow
            .as_mut()
            .is_some_and(|overflow| overflow.remove(name).is_some())
    }

    /// Get all property names in definition order.
    pub fn property_names(&self) -> Vec<InternedString> {
        let mut names = Vec::new();
        for name in self.shape.property_names() {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS {
                    if !is_deleted_property(self.inline_slots.get(slot_index)) {
                        names.push(name);
                    }
                    continue;
                }
            }
            if self
                .overflow
                .as_ref()
                .is_some_and(|overflow| overflow.contains(&name))
            {
                names.push(name);
            }
        }

        if let Some(overflow) = &self.overflow {
            for (name, _) in overflow.iter() {
                // Only add if not already in shape (shouldn't happen, but defensive)
                if !names.iter().any(|n| n == name) {
                    names.push(name.clone());
                }
            }
        }

        names
    }

    /// Get total property count.
    pub fn property_count(&self) -> usize {
        let mut inline_count = 0usize;
        for name in self.shape.property_names() {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS
                    && !is_deleted_property(self.inline_slots.get(slot_index))
                {
                    inline_count += 1;
                }
            }
        }
        let overflow_count = self.overflow.as_ref().map_or(0, |o| o.len());
        inline_count + overflow_count
    }

    /// Check if object uses only inline storage.
    #[inline]
    pub fn is_inline_only(&self) -> bool {
        self.overflow.is_none() || self.overflow.as_ref().unwrap().is_empty()
    }

    /// Iterate over all properties.
    pub fn iter_properties(&self) -> impl Iterator<Item = (InternedString, Value)> + '_ {
        let shape_props = self.shape.property_names();
        let inline_iter = shape_props.into_iter().filter_map(|name| {
            if let Some(slot_index) = self.shape.lookup_interned(&name) {
                if (slot_index as usize) < MAX_INLINE_SLOTS {
                    let value = self.inline_slots.get(slot_index);
                    if !is_deleted_property(value) {
                        return Some((name, value));
                    }
                }
            }
            None
        });

        let overflow_iter = self
            .overflow
            .iter()
            .flat_map(|o| o.iter().map(|(k, v)| (k.clone(), *v)));

        inline_iter.chain(overflow_iter)
    }
}

impl PyObject for ShapedObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

impl InstanceDictStorage {
    #[inline]
    fn value(&self) -> Value {
        match self {
            Self::Owned(dict) => Value::object_ptr(&**dict as *const DictObject as *const ()),
            Self::External(value) => *value,
        }
    }
}
