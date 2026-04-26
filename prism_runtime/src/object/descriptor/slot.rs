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
#[derive(Debug, Clone)]
pub struct SlotDescriptor {
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
            // In a full implementation, we'd return self as a Value
            return Ok(Value::none());
        }

        if !self.is_readable() {
            return Err(PrismError::attribute(format!(
                "slot '{}' is not readable",
                self.name.as_str()
            )));
        }

        let _obj = obj.unwrap();

        // TODO: Actually read from the instance slot
        // In real implementation:
        // let ptr = obj.as_object_ptr().unwrap() as *const u8;
        // unsafe { Ok(self.read_unchecked(ptr)) }

        // Placeholder - return None to indicate uninitialized
        Ok(Value::none())
    }

    fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        if !self.is_writable() {
            return Err(PrismError::attribute(format!(
                "slot '{}' is read-only",
                self.name.as_str()
            )));
        }

        let _ = (obj, value);

        // TODO: Actually write to the instance slot
        // In real implementation:
        // let ptr = obj.as_object_ptr().unwrap() as *mut u8;
        // unsafe { self.write_unchecked(ptr, value) };

        Ok(())
    }

    fn delete(&self, obj: Value) -> PrismResult<()> {
        // Deleting a slot sets it to an uninitialized sentinel value
        // This allows AttributeError on subsequent reads
        let _ = obj;

        // TODO: Set to uninitialized sentinel
        // In real implementation:
        // let ptr = obj.as_object_ptr().unwrap() as *mut u8;
        // unsafe { self.write_unchecked(ptr, Value::UNINITIALIZED) };

        Ok(())
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
