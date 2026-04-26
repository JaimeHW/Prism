//! ClassMethod descriptor implementation.
//!
//! A classmethod receives the class as its first argument instead of an instance.
//! This is useful for alternative constructors and class-level operations.
//!
//! # Example (Python)
//!
//! ```python
//! class Date:
//!     def __init__(self, year, month, day):
//!         self.year = year
//!         self.month = month
//!         self.day = day
//!
//!     @classmethod
//!     def from_string(cls, date_string):
//!         year, month, day = map(int, date_string.split('-'))
//!         return cls(year, month, day)  # cls is the class, not an instance
//!
//!     @classmethod
//!     def today(cls):
//!         # Get current date and return new instance
//!         return cls(2024, 1, 1)
//!
//! d = Date.from_string("2024-06-15")  # Uses classmethod
//! ```
//!
//! # Binding Behavior
//!
//! - Accessed through instance: binds to instance's class
//! - Accessed through class: binds to that class
//! - Works correctly with inheritance
//!
//! # Performance
//!
//! ClassMethodDescriptor creates a bound method that receives the class.
//! This is slightly more expensive than regular methods since we need
//! to determine the objtype.

use super::method::BoundMethod;
use super::{Descriptor, DescriptorFlags, DescriptorKind};
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::{PrismResult, Value};

// =============================================================================
// ClassMethod Descriptor
// =============================================================================

/// Descriptor for @classmethod decorated functions.
///
/// When accessed, binds the function to the class (or instance's class)
/// rather than the instance itself.
#[derive(Debug)]
#[repr(C)]
pub struct ClassMethodDescriptor {
    /// Object header for heap storage and fast type dispatch.
    pub header: ObjectHeader,
    /// The underlying function.
    function: Value,
}

impl ClassMethodDescriptor {
    /// Create a new classmethod descriptor.
    pub fn new(function: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::CLASSMETHOD),
            function,
        }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }

    /// Create a bound method bound to the given class.
    #[inline]
    pub fn bind(&self, class: Value) -> BoundMethod {
        BoundMethod::new(self.function, class)
    }

    /// Create a heap-allocated bound-method value bound to `class`.
    #[inline]
    pub fn bind_value(&self, class: Value) -> Value {
        crate::allocation_context::alloc_value_in_current_heap_or_box(self.bind(class))
    }
}

impl Descriptor for ClassMethodDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::ClassMethod
    }

    fn flags(&self) -> DescriptorFlags {
        DescriptorFlags::HAS_GET | DescriptorFlags::CLASSMETHOD
    }

    fn get(&self, _obj: Option<Value>, objtype: Value) -> PrismResult<Value> {
        // For classmethod, we always bind to objtype (the class)
        // regardless of whether accessed through instance or class
        Ok(self.bind_value(objtype))
    }
}

impl PyObject for ClassMethodDescriptor {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
