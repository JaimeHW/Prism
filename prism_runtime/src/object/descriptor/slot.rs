//! Slot descriptor implementation.
//!
//! Slot descriptors provide optimized fixed-offset attribute access for classes
//! that define `__slots__`. This avoids the overhead of dictionary lookup and
//! reduces memory usage per instance.
//!
//! # Performance Characteristics
//!
//! - **O(1) access**: Fixed offset means direct pointer arithmetic
//! - **Cache-friendly**: Slots are contiguous in memory
//! - **Reduced memory**: No per-instance `__dict__` allocation
//! - **IC-friendly**: Offset can be cached in inline caches
//!
//! # Example (Python)
//!
//! ```python
//! class Point:
//!     __slots__ = ('x', 'y')
//!
//!     def __init__(self, x, y):
//!         self.x = x  # Uses SlotDescriptor, not dict
//!         self.y = y
//! ```
//!
//! # Memory Layout
//!
//! With slots, instance layout is:
//! ```text
//! Instance
//! ├── ObjectHeader (16 bytes)
//! ├── slot[0]: Value (8 bytes)  # e.g., 'x'
//! ├── slot[1]: Value (8 bytes)  # e.g., 'y'
//! └── ... more slots
//! ```
//!
//! Without slots (using __dict__):
//! ```text
//! Instance
//! ├── ObjectHeader (16 bytes)
//! └── __dict__: *Dict (~56 bytes + per-entry overhead)
//! ```

use super::{Descriptor, DescriptorFlags, DescriptorKind};
use crate::object::shape::shape_registry;
use crate::object::shaped_object::ShapedObject;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::intern::InternedString;
use prism_core::{PrismError, PrismResult, Value};

// =============================================================================
// Slot Access Mode
// =============================================================================

/// Access mode for a slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlotAccess {
    /// Slot is readable and writable.
    ReadWrite,
    /// Slot is read-only (set only once, typically in __init__).
    ReadOnly,
    /// Slot is write-only (rare, but supported).
    WriteOnly,
}

impl Default for SlotAccess {
    fn default() -> Self {
        Self::ReadWrite
    }
}

// =============================================================================
// Slot Descriptor
// =============================================================================

/// Slot descriptor for fixed-offset attribute storage.
///
/// # Performance
///
/// Slot access is the fastest possible attribute access pattern:
/// 1. Check instance type matches expected class
/// 2. Load/store at fixed offset from instance pointer
///
/// This is the same pattern used by compiled languages like C++ for members.
///
/// # Memory
///
/// SlotDescriptor itself is small (24 bytes) and stored in the class dict.
/// The actual slot values are stored inline in each instance.
#[derive(Debug)]
#[repr(C)]
pub struct SlotDescriptor {
    /// Object header for heap storage and fast type dispatch.
    pub header: ObjectHeader,
    /// Name of the slot (for error messages and debugging).
    name: InternedString,
    /// Offset in bytes from instance start to this slot.
    offset: u16,
    /// Slot index (0-based position in __slots__).
    index: u16,
    /// Access mode.
    access: SlotAccess,
}

impl SlotDescriptor {
    /// Create a new slot descriptor.
    ///
    /// # Arguments
    ///
    /// * `name` - The attribute name for this slot
    /// * `index` - Zero-based index in the __slots__ tuple
    /// * `offset` - Byte offset from instance start
    /// * `access` - Read/write access mode
    pub fn new(name: InternedString, index: u16, offset: u16, access: SlotAccess) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MEMBER_DESCRIPTOR),
            name,
            offset,
            index,
            access,
        }
    }

    /// Create a read-write slot (most common case).
    #[inline]
    pub fn read_write(name: InternedString, index: u16, offset: u16) -> Self {
        Self::new(name, index, offset, SlotAccess::ReadWrite)
    }

    /// Create a read-only slot.
    #[inline]
    pub fn read_only(name: InternedString, index: u16, offset: u16) -> Self {
        Self::new(name, index, offset, SlotAccess::ReadOnly)
    }

    /// Get the slot name.
    #[inline]
    pub fn name(&self) -> &InternedString {
        &self.name
    }

    /// Get the byte offset.
    #[inline]
    pub fn offset(&self) -> u16 {
        self.offset
    }

    /// Get the slot index.
    #[inline]
    pub fn index(&self) -> u16 {
        self.index
    }

    /// Get the access mode.
    #[inline]
    pub fn access(&self) -> SlotAccess {
        self.access
    }

    /// Check if slot is readable.
    #[inline]
    pub fn is_readable(&self) -> bool {
        !matches!(self.access, SlotAccess::WriteOnly)
    }

    /// Check if slot is writable.
    #[inline]
    pub fn is_writable(&self) -> bool {
        !matches!(self.access, SlotAccess::ReadOnly)
    }

    /// Calculate the slot offset for a given class.
    ///
    /// This computes the offset assuming:
    /// - ObjectHeader is at offset 0 (16 bytes)
    /// - Slots start at offset 16
    /// - Each slot is 8 bytes (size of Value)
    #[inline]
    pub const fn compute_offset(slot_index: u16) -> u16 {
        const HEADER_SIZE: u16 = 16; // ObjectHeader size
        const SLOT_SIZE: u16 = 8; // Value size
        HEADER_SIZE + (slot_index * SLOT_SIZE)
    }

    #[inline]
    fn self_value(&self) -> Value {
        Value::object_ptr(self as *const Self as *const ())
    }

    #[inline]
    fn shaped_object(&self, obj: Value) -> PrismResult<&'static ShapedObject> {
        let ptr = obj
            .as_object_ptr()
            .ok_or_else(|| PrismError::type_error("slot descriptor requires an object"))?;
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id != TypeId::OBJECT && type_id.raw() < TypeId::FIRST_USER_TYPE {
            return Err(PrismError::type_error(format!(
                "slot '{}' cannot be read from '{}'",
                self.name.as_str(),
                type_id.name()
            )));
        }

        Ok(unsafe { &*(ptr as *const ShapedObject) })
    }

    #[inline]
    fn shaped_object_mut(&self, obj: Value) -> PrismResult<&'static mut ShapedObject> {
        let ptr = obj
            .as_object_ptr()
            .ok_or_else(|| PrismError::type_error("slot descriptor requires an object"))?;
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        if type_id != TypeId::OBJECT && type_id.raw() < TypeId::FIRST_USER_TYPE {
            return Err(PrismError::type_error(format!(
                "slot '{}' cannot be written on '{}'",
                self.name.as_str(),
                type_id.name()
            )));
        }

        Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
    }

    /// Read the slot value from an instance.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `instance_ptr` points to a valid instance of the correct type
    /// - The instance has this slot allocated
    #[inline]
    pub unsafe fn read_unchecked(&self, instance_ptr: *const u8) -> Value {
        unsafe {
            let slot_ptr = instance_ptr.add(self.offset as usize) as *const Value;
            std::ptr::read(slot_ptr)
        }
    }

    /// Write the slot value to an instance.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `instance_ptr` points to a valid instance of the correct type
    /// - The instance has this slot allocated
    /// - The slot is writable
    #[inline]
    pub unsafe fn write_unchecked(&self, instance_ptr: *mut u8, value: Value) {
        unsafe {
            let slot_ptr = instance_ptr.add(self.offset as usize) as *mut Value;
            std::ptr::write(slot_ptr, value);
        }
    }
}

impl Clone for SlotDescriptor {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::MEMBER_DESCRIPTOR),
            name: self.name.clone(),
            offset: self.offset,
            index: self.index,
            access: self.access,
        }
    }
}

impl Descriptor for SlotDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::Slot
    }

    fn flags(&self) -> DescriptorFlags {
        let mut flags = DescriptorFlags::DATA_DESCRIPTOR | DescriptorFlags::SLOT;

        if self.is_readable() {
            flags |= DescriptorFlags::HAS_GET;
        }
        if self.is_writable() {
            flags |= DescriptorFlags::HAS_SET;
        }
        // Note: slots typically don't support deletion
        // (deleting sets to an uninitialized sentinel)

        flags
    }

    fn get(&self, obj: Option<Value>, _objtype: Value) -> PrismResult<Value> {
        // If accessed through class (obj is None), return the descriptor itself
        if obj.is_none() {
            return Ok(self.self_value());
        }

        if !self.is_readable() {
            return Err(PrismError::attribute(format!(
                "slot '{}' is not readable",
                self.name.as_str()
            )));
        }

        let shaped = self.shaped_object(obj.expect("checked above"))?;
        shaped.get_property_interned(&self.name).ok_or_else(|| {
            PrismError::attribute(format!(
                "slot '{}' has not been assigned",
                self.name.as_str()
            ))
        })
    }

    fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        if !self.is_writable() {
            return Err(PrismError::attribute(format!(
                "slot '{}' is read-only",
                self.name.as_str()
            )));
        }

        let shaped = self.shaped_object_mut(obj)?;
        shaped.set_property(self.name.clone(), value, shape_registry());
        Ok(())
    }

    fn delete(&self, obj: Value) -> PrismResult<()> {
        let shaped = self.shaped_object_mut(obj)?;
        if shaped.delete_property_interned(&self.name) {
            Ok(())
        } else {
            Err(PrismError::attribute(format!(
                "slot '{}' has not been assigned",
                self.name.as_str()
            )))
        }
    }
}

impl PyObject for SlotDescriptor {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Slot Member Collection
// =============================================================================

/// Collection of slots for a class.
///
/// This is used during class creation to gather all slots and compute offsets.
#[derive(Debug, Clone, Default)]
pub struct SlotCollection {
    /// The slot descriptors in order.
    slots: Vec<SlotDescriptor>,
    /// Total size in bytes for all slots.
    total_size: u16,
}

impl SlotCollection {
    /// Create a new empty slot collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a slot collection from slot names.
    pub fn from_names(names: &[InternedString]) -> Self {
        let mut collection = Self::new();
        for name in names {
            collection.add_slot(name.clone(), SlotAccess::ReadWrite);
        }
        collection
    }

    /// Add a slot to the collection.
    pub fn add_slot(&mut self, name: InternedString, access: SlotAccess) -> u16 {
        let index = self.slots.len() as u16;
        let offset = SlotDescriptor::compute_offset(index);

        self.slots
            .push(SlotDescriptor::new(name, index, offset, access));
        self.total_size = offset + 8; // 8 bytes per Value

        index
    }

    /// Get a slot by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&SlotDescriptor> {
        self.slots.get(index)
    }

    /// Get a slot by name.
    pub fn get_by_name(&self, name: &InternedString) -> Option<&SlotDescriptor> {
        self.slots.iter().find(|s| &s.name == name)
    }

    /// Get the number of slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Get the total size in bytes for all slots.
    #[inline]
    pub fn total_size(&self) -> u16 {
        self.total_size
    }

    /// Iterate over slots.
    pub fn iter(&self) -> impl Iterator<Item = &SlotDescriptor> {
        self.slots.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::shape::shape_registry;
    use prism_core::intern::intern;

    #[test]
    fn slot_descriptor_reads_writes_and_deletes_storage() {
        let name = intern("x");
        let descriptor = SlotDescriptor::read_write(name.clone(), 0, 0);
        let mut instance = ShapedObject::new(TypeId::OBJECT, shape_registry().empty_shape());
        let instance_value = Value::object_ptr(&mut instance as *mut ShapedObject as *const ());
        let assigned = Value::int_unchecked(42);

        assert!(descriptor.get(Some(instance_value), Value::none()).is_err());
        descriptor.set(instance_value, assigned).unwrap();
        assert_eq!(
            descriptor.get(Some(instance_value), Value::none()).unwrap(),
            assigned
        );
        descriptor.delete(instance_value).unwrap();
        assert!(descriptor.get(Some(instance_value), Value::none()).is_err());
    }

    #[test]
    fn slot_descriptor_preserves_none_as_assigned_value() {
        let name = intern("maybe");
        let descriptor = SlotDescriptor::read_write(name, 0, 0);
        let mut instance = ShapedObject::new(TypeId::OBJECT, shape_registry().empty_shape());
        let instance_value = Value::object_ptr(&mut instance as *mut ShapedObject as *const ());

        descriptor.set(instance_value, Value::none()).unwrap();
        assert_eq!(
            descriptor.get(Some(instance_value), Value::none()).unwrap(),
            Value::none()
        );
        descriptor.delete(instance_value).unwrap();
        assert!(descriptor.get(Some(instance_value), Value::none()).is_err());
    }
}
