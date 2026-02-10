//! Python instance object implementation.
//!
//! A `PyInstanceObject` represents an instance of a user-defined class.
//! It uses a hot/cold memory layout to optimize for cache performance:
//! - Hot data (frequently accessed) fits in one cache line (64 bytes)
//! - Cold data (rarely accessed) is allocated separately
//!
//! # Memory Layout
//!
//! ```text
//! PyInstanceObject (64 bytes, cache-line aligned)
//! ├── header: ObjectHeader (16 bytes) - GC metadata + type ID
//! ├── class_id: ClassId (4 bytes) - Reference to the class
//! ├── shape_id: u32 (4 bytes) - Shape ID for inline cache validation
//! ├── inline_slots: [Value; 4] (32 bytes) - First 4 instance attributes
//! └── overflow: *mut InstanceOverflow (8 bytes) - Pointer to overflow storage
//! ```
//!
//! # Performance
//!
//! - First 4 attributes accessed via inline slots (O(1), cache-friendly)
//! - Shape ID enables O(1) inline cache validation
//! - Class ID lookup is O(1)
//! - Overflow storage only allocated when needed
//!
//! # __slots__ Support
//!
//! Classes with `__slots__` use fixed-offset storage without overflow.
//! This provides O(1) access to all attributes with minimal memory.

use crate::object::mro::ClassId;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::sync::atomic::{AtomicU32, Ordering};

// =============================================================================
// Shape ID Management
// =============================================================================

/// Global counter for shape IDs.
static NEXT_SHAPE_ID: AtomicU32 = AtomicU32::new(1);

/// Allocate a new unique shape ID.
#[inline]
pub fn allocate_shape_id() -> u32 {
    NEXT_SHAPE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Shape ID for empty instances (no attributes).
pub const EMPTY_SHAPE_ID: u32 = 0;

// =============================================================================
// Instance Configuration
// =============================================================================

/// Number of inline slots for instance attributes.
/// Chosen to fit the hot data in a cache line (64 bytes).
///
/// Layout: header(16) + class_id(4) + shape_id(4) + slots(4*8=32) + overflow(8) = 64
pub const INLINE_SLOT_COUNT: usize = 4;

/// Maximum number of overflow slots before switching to hashmap.
/// After this, attributes are stored in a dynamic dictionary.
pub const MAX_OVERFLOW_SLOTS: usize = 12;

// =============================================================================
// Overflow Storage
// =============================================================================

/// Overflow storage for instances with more than 4 attributes.
///
/// Uses a two-tier approach:
/// 1. For 5-16 attributes: contiguous array (cache-friendly)
/// 2. For 17+ attributes: hashmap (flexible)
#[derive(Debug)]
pub enum OverflowStorage {
    /// Contiguous array storage for moderate number of attributes.
    /// Stores (offset -> Value) pairs.
    Slots(Box<[Value; MAX_OVERFLOW_SLOTS]>),

    /// Dynamic dictionary for many attributes.
    Dict(Box<FxHashMap<InternedString, Value>>),
}

impl OverflowStorage {
    /// Create new slots-based overflow storage.
    #[inline]
    pub fn new_slots() -> Self {
        Self::Slots(Box::new([Value::none(); MAX_OVERFLOW_SLOTS]))
    }

    /// Create new dictionary-based overflow storage.
    #[inline]
    pub fn new_dict() -> Self {
        Self::Dict(Box::new(FxHashMap::default()))
    }

    /// Get a value from slot-based storage.
    #[inline]
    pub fn get_slot(&self, offset: usize) -> Option<Value> {
        match self {
            Self::Slots(slots) => {
                if offset < MAX_OVERFLOW_SLOTS {
                    Some(slots[offset])
                } else {
                    None
                }
            }
            Self::Dict(_) => None,
        }
    }

    /// Set a value in slot-based storage.
    #[inline]
    pub fn set_slot(&mut self, offset: usize, value: Value) -> bool {
        match self {
            Self::Slots(slots) => {
                if offset < MAX_OVERFLOW_SLOTS {
                    slots[offset] = value;
                    true
                } else {
                    false
                }
            }
            Self::Dict(_) => false,
        }
    }

    /// Get a value from dictionary-based storage.
    #[inline]
    pub fn get_dict(&self, name: &InternedString) -> Option<Value> {
        match self {
            Self::Dict(dict) => dict.get(name).copied(),
            Self::Slots(_) => None,
        }
    }

    /// Set a value in dictionary-based storage.
    #[inline]
    pub fn set_dict(&mut self, name: InternedString, value: Value) {
        if let Self::Dict(dict) = self {
            dict.insert(name, value);
        }
    }

    /// Convert slots to dictionary (for instances that outgrow slots).
    pub fn promote_to_dict(&mut self, attr_names: &[InternedString]) {
        if let Self::Slots(slots) = self {
            let mut dict = FxHashMap::default();
            for (i, name) in attr_names.iter().enumerate() {
                if i < MAX_OVERFLOW_SLOTS && !slots[i].is_none() {
                    dict.insert(name.clone(), slots[i]);
                }
            }
            *self = Self::Dict(Box::new(dict));
        }
    }
}

impl Clone for OverflowStorage {
    fn clone(&self) -> Self {
        match self {
            Self::Slots(slots) => Self::Slots(slots.clone()),
            Self::Dict(dict) => Self::Dict(Box::new((**dict).clone())),
        }
    }
}

// =============================================================================
// Instance Overflow
// =============================================================================

/// Cold data for instances (rarely accessed).
#[derive(Debug, Clone)]
pub struct InstanceOverflow {
    /// Overflow attribute storage.
    pub storage: OverflowStorage,

    /// Custom `__dict__` if class allows it.
    pub custom_dict: Option<Box<FxHashMap<InternedString, Value>>>,

    /// Weakref list (for instances that support weakrefs).
    pub weakrefs: Option<Vec<Value>>,
}

impl InstanceOverflow {
    /// Create new empty overflow.
    #[inline]
    pub fn new() -> Self {
        Self {
            storage: OverflowStorage::new_slots(),
            custom_dict: None,
            weakrefs: None,
        }
    }

    /// Create new dictionary-based overflow.
    #[inline]
    pub fn new_dict() -> Self {
        Self {
            storage: OverflowStorage::new_dict(),
            custom_dict: None,
            weakrefs: None,
        }
    }
}

impl Default for InstanceOverflow {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Python Instance Object (Hot Data)
// =============================================================================

/// Python instance object - represents an instance of a user-defined class.
///
/// This struct is carefully sized to fit in a single cache line (64 bytes)
/// with proper alignment for optimal performance.
///
/// # Thread Safety
///
/// Instances are not thread-safe by default. Use external synchronization
/// when sharing instances across threads.
#[repr(C, align(64))]
#[derive(Debug)]
pub struct PyInstanceObject {
    /// Object header for GC and type identification.
    header: ObjectHeader,

    /// ClassId of the class this is an instance of.
    class_id: ClassId,

    /// Shape ID for inline cache validation.
    /// Changes when instance's shape changes (attribute add/delete).
    shape_id: u32,

    /// Inline storage for first 4 instance attributes.
    /// Accessed via shape-determined offsets.
    inline_slots: [Value; INLINE_SLOT_COUNT],

    /// Pointer to overflow storage (cold data).
    /// Null if no overflow attributes exist.
    overflow: Option<Box<InstanceOverflow>>,
}

impl PyInstanceObject {
    /// Create a new instance of the given class.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The class this is an instance of
    /// * `type_id` - The TypeId for this instance's class
    ///
    /// # Returns
    ///
    /// A new instance with empty attribute storage.
    #[inline]
    pub fn new(class_id: ClassId, type_id: TypeId) -> Self {
        Self {
            header: ObjectHeader::new(type_id),
            class_id,
            shape_id: EMPTY_SHAPE_ID,
            inline_slots: [Value::none(); INLINE_SLOT_COUNT],
            overflow: None,
        }
    }

    /// Create a new instance with pre-allocated overflow.
    ///
    /// Use this when you know the instance will have more than 4 attributes.
    pub fn new_with_overflow(class_id: ClassId, type_id: TypeId) -> Self {
        Self {
            header: ObjectHeader::new(type_id),
            class_id,
            shape_id: EMPTY_SHAPE_ID,
            inline_slots: [Value::none(); INLINE_SLOT_COUNT],
            overflow: Some(Box::new(InstanceOverflow::new())),
        }
    }

    /// Create a new instance for a `__slots__` class.
    ///
    /// # Arguments
    ///
    /// * `class_id` - The class this is an instance of
    /// * `type_id` - The TypeId for this instance's class
    /// * `slot_count` - Number of `__slots__` defined
    pub fn new_slotted(class_id: ClassId, type_id: TypeId, slot_count: usize) -> Self {
        if slot_count <= INLINE_SLOT_COUNT {
            Self::new(class_id, type_id)
        } else {
            Self::new_with_overflow(class_id, type_id)
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the class ID of this instance.
    #[inline]
    pub fn class_id(&self) -> ClassId {
        self.class_id
    }

    /// Get the current shape ID.
    #[inline]
    pub fn shape_id(&self) -> u32 {
        self.shape_id
    }

    /// Set the shape ID (called when shape changes).
    #[inline]
    pub fn set_shape_id(&mut self, shape_id: u32) {
        self.shape_id = shape_id;
    }

    /// Check if this instance has overflow storage.
    #[inline]
    pub fn has_overflow(&self) -> bool {
        self.overflow.is_some()
    }

    // =========================================================================
    // Inline Slot Access
    // =========================================================================

    /// Get value from inline slot.
    ///
    /// # Safety
    ///
    /// Caller must ensure offset < INLINE_SLOT_COUNT.
    #[inline]
    pub fn get_inline_slot(&self, offset: usize) -> Value {
        debug_assert!(offset < INLINE_SLOT_COUNT);
        self.inline_slots[offset]
    }

    /// Set value in inline slot.
    ///
    /// # Safety
    ///
    /// Caller must ensure offset < INLINE_SLOT_COUNT.
    #[inline]
    pub fn set_inline_slot(&mut self, offset: usize, value: Value) {
        debug_assert!(offset < INLINE_SLOT_COUNT);
        self.inline_slots[offset] = value;
    }

    /// Get value from inline slot with bounds check.
    #[inline]
    pub fn get_inline_slot_checked(&self, offset: usize) -> Option<Value> {
        if offset < INLINE_SLOT_COUNT {
            Some(self.inline_slots[offset])
        } else {
            None
        }
    }

    /// Set value in inline slot with bounds check.
    #[inline]
    pub fn set_inline_slot_checked(&mut self, offset: usize, value: Value) -> bool {
        if offset < INLINE_SLOT_COUNT {
            self.inline_slots[offset] = value;
            true
        } else {
            false
        }
    }

    // =========================================================================
    // Overflow Access
    // =========================================================================

    /// Get reference to overflow storage.
    #[inline]
    pub fn overflow(&self) -> Option<&InstanceOverflow> {
        self.overflow.as_ref().map(|b| &**b)
    }

    /// Get mutable reference to overflow storage.
    #[inline]
    pub fn overflow_mut(&mut self) -> Option<&mut InstanceOverflow> {
        self.overflow.as_mut().map(|b| &mut **b)
    }

    /// Ensure overflow storage exists.
    pub fn ensure_overflow(&mut self) {
        if self.overflow.is_none() {
            self.overflow = Some(Box::new(InstanceOverflow::new()));
        }
    }

    /// Get value from overflow slot.
    #[inline]
    pub fn get_overflow_slot(&self, offset: usize) -> Option<Value> {
        self.overflow()?.storage.get_slot(offset)
    }

    /// Set value in overflow slot.
    #[inline]
    pub fn set_overflow_slot(&mut self, offset: usize, value: Value) -> bool {
        self.ensure_overflow();
        if let Some(overflow) = self.overflow_mut() {
            overflow.storage.set_slot(offset, value)
        } else {
            false
        }
    }

    // =========================================================================
    // Unified Attribute Access
    // =========================================================================

    /// Get attribute by offset (used with shape-based IC).
    ///
    /// This is the hot path for attribute access. The offset is determined
    /// by the shape and cached in inline caches.
    #[inline]
    pub fn get_attr_by_offset(&self, offset: usize) -> Option<Value> {
        if offset < INLINE_SLOT_COUNT {
            Some(self.inline_slots[offset])
        } else {
            self.get_overflow_slot(offset - INLINE_SLOT_COUNT)
        }
    }

    /// Set attribute by offset (used with shape-based IC).
    ///
    /// This is the hot path for attribute setting.
    #[inline]
    pub fn set_attr_by_offset(&mut self, offset: usize, value: Value) -> bool {
        if offset < INLINE_SLOT_COUNT {
            self.inline_slots[offset] = value;
            true
        } else {
            self.set_overflow_slot(offset - INLINE_SLOT_COUNT, value)
        }
    }

    /// Get attribute by name (slow path, used when cache misses).
    pub fn get_attr_by_name(&self, name: &InternedString) -> Option<Value> {
        if let Some(overflow) = self.overflow() {
            overflow.storage.get_dict(name)
        } else {
            None
        }
    }

    /// Set attribute by name (slow path, transitions shape).
    pub fn set_attr_by_name(&mut self, name: InternedString, value: Value) {
        self.ensure_overflow();
        if let Some(overflow) = self.overflow_mut() {
            overflow.storage.set_dict(name, value);
        }
    }

    // =========================================================================
    // __dict__ Access
    // =========================================================================

    /// Check if instance has a custom `__dict__`.
    #[inline]
    pub fn has_dict(&self) -> bool {
        self.overflow().map_or(false, |o| o.custom_dict.is_some())
    }

    /// Get custom `__dict__` if present.
    pub fn get_dict(&self) -> Option<&FxHashMap<InternedString, Value>> {
        self.overflow()?.custom_dict.as_ref().map(|b| &**b)
    }

    /// Create or get mutable `__dict__`.
    pub fn get_or_create_dict(&mut self) -> &mut FxHashMap<InternedString, Value> {
        self.ensure_overflow();
        let overflow = self.overflow_mut().unwrap();
        if overflow.custom_dict.is_none() {
            overflow.custom_dict = Some(Box::new(FxHashMap::default()));
        }
        overflow.custom_dict.as_mut().unwrap()
    }
}

impl PyObject for PyInstanceObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

impl Drop for PyInstanceObject {
    fn drop(&mut self) {
        // Box handles cleanup automatically, no manual drop needed
    }
}

// =============================================================================
// Specialization Hints for JIT
// =============================================================================

/// JIT compilation hints for instance operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstanceSpecHint {
    /// Instance uses only inline slots (fast path).
    InlineOnly,

    /// Instance uses fixed slots from `__slots__`.
    FixedSlots,

    /// Instance has dynamic attributes (general path).
    Dynamic,
}

impl InstanceSpecHint {
    /// Determine hint from attribute count.
    #[inline]
    pub fn from_attr_count(count: usize, has_slots: bool) -> Self {
        if has_slots {
            Self::FixedSlots
        } else if count <= INLINE_SLOT_COUNT {
            Self::InlineOnly
        } else {
            Self::Dynamic
        }
    }
}


// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::intern::intern;

    // =========================================================================
    // Construction Tests
    // =========================================================================

    #[test]
    fn test_instance_new() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new(class_id, type_id);

        assert_eq!(instance.class_id(), class_id);
        assert_eq!(instance.shape_id(), EMPTY_SHAPE_ID);
        assert!(!instance.has_overflow());
    }

    #[test]
    fn test_instance_new_with_overflow() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new_with_overflow(class_id, type_id);

        assert_eq!(instance.class_id(), class_id);
        assert!(instance.has_overflow());
    }

    #[test]
    fn test_instance_new_slotted_small() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new_slotted(class_id, type_id, 3);

        assert!(!instance.has_overflow());
    }

    #[test]
    fn test_instance_new_slotted_large() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new_slotted(class_id, type_id, 8);

        assert!(instance.has_overflow());
    }

    // =========================================================================
    // Inline Slot Tests
    // =========================================================================

    #[test]
    fn test_inline_slot_access() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        // Set and get inline slots
        instance.set_inline_slot(0, Value::int_unchecked(42));
        instance.set_inline_slot(1, Value::int_unchecked(100));

        assert_eq!(instance.get_inline_slot(0).as_int().unwrap(), 42);
        assert_eq!(instance.get_inline_slot(1).as_int().unwrap(), 100);
    }

    #[test]
    fn test_inline_slot_checked() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        assert!(instance.set_inline_slot_checked(0, Value::int_unchecked(1)));
        assert!(instance.set_inline_slot_checked(3, Value::int_unchecked(4)));
        assert!(!instance.set_inline_slot_checked(4, Value::int_unchecked(5)));

        assert!(instance.get_inline_slot_checked(0).is_some());
        assert!(instance.get_inline_slot_checked(4).is_none());
    }

    #[test]
    fn test_all_inline_slots() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        for i in 0..INLINE_SLOT_COUNT {
            instance.set_inline_slot(i, Value::int_unchecked(i as i64));
        }

        for i in 0..INLINE_SLOT_COUNT {
            assert_eq!(instance.get_inline_slot(i).as_int().unwrap(), i as i64);
        }
    }

    // =========================================================================
    // Overflow Storage Tests
    // =========================================================================

    #[test]
    fn test_ensure_overflow() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        assert!(!instance.has_overflow());
        instance.ensure_overflow();
        assert!(instance.has_overflow());
    }

    #[test]
    fn test_overflow_slot_access() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

        assert!(instance.set_overflow_slot(0, Value::int_unchecked(99)));
        assert_eq!(instance.get_overflow_slot(0).unwrap().as_int().unwrap(), 99);
    }

    #[test]
    fn test_unified_attr_access() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

        // Inline slots (offset 0-3)
        for i in 0..INLINE_SLOT_COUNT {
            assert!(instance.set_attr_by_offset(i, Value::int_unchecked(i as i64)));
        }

        // Overflow slots (offset 4+)
        for i in 0..4 {
            let offset = INLINE_SLOT_COUNT + i;
            assert!(instance.set_attr_by_offset(offset, Value::int_unchecked(100 + i as i64)));
        }

        // Verify inline
        for i in 0..INLINE_SLOT_COUNT {
            assert_eq!(
                instance.get_attr_by_offset(i).unwrap().as_int().unwrap(),
                i as i64
            );
        }

        // Verify overflow
        for i in 0..4 {
            let offset = INLINE_SLOT_COUNT + i;
            assert_eq!(
                instance
                    .get_attr_by_offset(offset)
                    .unwrap()
                    .as_int()
                    .unwrap(),
                100 + i as i64
            );
        }
    }

    // =========================================================================
    // Shape ID Tests
    // =========================================================================

    #[test]
    fn test_shape_id() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        assert_eq!(instance.shape_id(), EMPTY_SHAPE_ID);

        let new_shape = allocate_shape_id();
        instance.set_shape_id(new_shape);
        assert_eq!(instance.shape_id(), new_shape);
    }

    #[test]
    fn test_shape_id_allocation() {
        let id1 = allocate_shape_id();
        let id2 = allocate_shape_id();
        let id3 = allocate_shape_id();

        assert!(id1 < id2);
        assert!(id2 < id3);
    }

    // =========================================================================
    // __dict__ Tests
    // =========================================================================

    #[test]
    fn test_dict_creation() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        assert!(!instance.has_dict());

        let dict = instance.get_or_create_dict();
        dict.insert(intern("x"), Value::int_unchecked(1));

        assert!(instance.has_dict());
    }

    #[test]
    fn test_dict_access() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new(class_id, type_id);

        let dict = instance.get_or_create_dict();
        dict.insert(intern("foo"), Value::int_unchecked(42));
        dict.insert(intern("bar"), Value::int_unchecked(99));

        let read_dict = instance.get_dict().unwrap();
        assert_eq!(read_dict.get(&intern("foo")).unwrap().as_int().unwrap(), 42);
        assert_eq!(read_dict.get(&intern("bar")).unwrap().as_int().unwrap(), 99);
    }

    // =========================================================================
    // OverflowStorage Tests
    // =========================================================================

    #[test]
    fn test_overflow_storage_slots() {
        let mut storage = OverflowStorage::new_slots();

        assert!(storage.set_slot(0, Value::int_unchecked(10)));
        assert!(storage.set_slot(5, Value::int_unchecked(50)));

        assert_eq!(storage.get_slot(0).unwrap().as_int().unwrap(), 10);
        assert_eq!(storage.get_slot(5).unwrap().as_int().unwrap(), 50);
    }

    #[test]
    fn test_overflow_storage_dict() {
        let mut storage = OverflowStorage::new_dict();
        let name = intern("attr");

        storage.set_dict(name.clone(), Value::int_unchecked(123));
        assert_eq!(storage.get_dict(&name).unwrap().as_int().unwrap(), 123);
    }

    #[test]
    fn test_overflow_storage_clone() {
        let mut storage = OverflowStorage::new_slots();
        storage.set_slot(0, Value::int_unchecked(42));

        let cloned = storage.clone();
        assert_eq!(cloned.get_slot(0).unwrap().as_int().unwrap(), 42);
    }

    // =========================================================================
    // InstanceOverflow Tests
    // =========================================================================

    #[test]
    fn test_instance_overflow_default() {
        let overflow = InstanceOverflow::default();
        assert!(overflow.custom_dict.is_none());
        assert!(overflow.weakrefs.is_none());
    }

    #[test]
    fn test_instance_overflow_new_dict() {
        let overflow = InstanceOverflow::new_dict();
        // Should have dict-based storage
        assert!(matches!(overflow.storage, OverflowStorage::Dict(_)));
    }

    // =========================================================================
    // PyObject Trait Tests
    // =========================================================================

    #[test]
    fn test_pyobject_trait() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new(class_id, type_id);

        assert_eq!(instance.header().type_id, type_id);
    }

    // =========================================================================
    // Specialization Hint Tests
    // =========================================================================

    #[test]
    fn test_spec_hint_inline_only() {
        let hint = InstanceSpecHint::from_attr_count(3, false);
        assert_eq!(hint, InstanceSpecHint::InlineOnly);
    }

    #[test]
    fn test_spec_hint_fixed_slots() {
        let hint = InstanceSpecHint::from_attr_count(10, true);
        assert_eq!(hint, InstanceSpecHint::FixedSlots);
    }

    #[test]
    fn test_spec_hint_dynamic() {
        let hint = InstanceSpecHint::from_attr_count(5, false);
        assert_eq!(hint, InstanceSpecHint::Dynamic);
    }

    // =========================================================================
    // Memory Layout Tests
    // =========================================================================

    #[test]
    fn test_instance_size() {
        let size = std::mem::size_of::<PyInstanceObject>();
        assert_eq!(size, 64, "PyInstanceObject should be 64 bytes (cache line)");
    }

    #[test]
    fn test_instance_alignment() {
        let align = std::mem::align_of::<PyInstanceObject>();
        assert_eq!(align, 64, "PyInstanceObject should be 64-byte aligned");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_inline_slot_default_none() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new(class_id, type_id);

        for i in 0..INLINE_SLOT_COUNT {
            assert!(instance.get_inline_slot(i).is_none());
        }
    }

    #[test]
    fn test_overflow_boundary() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let mut instance = PyInstanceObject::new_with_overflow(class_id, type_id);

        // Set value at inline boundary
        assert!(instance.set_attr_by_offset(INLINE_SLOT_COUNT - 1, Value::int_unchecked(1)));
        // Set value at overflow boundary
        assert!(instance.set_attr_by_offset(INLINE_SLOT_COUNT, Value::int_unchecked(2)));

        assert_eq!(
            instance
                .get_attr_by_offset(INLINE_SLOT_COUNT - 1)
                .unwrap()
                .as_int()
                .unwrap(),
            1
        );
        assert_eq!(
            instance
                .get_attr_by_offset(INLINE_SLOT_COUNT)
                .unwrap()
                .as_int()
                .unwrap(),
            2
        );
    }

    #[test]
    fn test_drop_frees_overflow() {
        let class_id = ClassId(100);
        let type_id = TypeId::from_raw(100);
        let instance = PyInstanceObject::new_with_overflow(class_id, type_id);

        // Just verify it doesn't leak - Drop will be called
        drop(instance);
    }
}
