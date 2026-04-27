//! StaticMethod descriptor implementation.
//!
//! A staticmethod is a regular function that happens to live in a class namespace.
//! It doesn't receive any implicit first argument (no self, no cls).
//!
//! # Example (Python)
//!
//! ```python
//! class Math:
//!     @staticmethod
//!     def add(x, y):
//!         return x + y
//!
//!     @staticmethod
//!     def is_even(n):
//!         return n % 2 == 0
//!
//! # Both work the same:
//! Math.add(1, 2)      # 3
//! Math().add(1, 2)    # 3
//! ```
//!
//! # Use Cases
//!
//! - Utility functions that logically belong to a class
//! - Functions that don't need instance or class state
//! - Organizing code into namespaces
//!
//! # Performance
//!
//! StaticMethod is the fastest descriptor type:
//! - No binding needed (returns function directly)
//! - No allocation on access
//! - Identical to calling a regular function

use super::{Descriptor, DescriptorFlags, DescriptorKind};
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::{PrismResult, Value};

// =============================================================================
// StaticMethod Descriptor
// =============================================================================

/// Descriptor for @staticmethod decorated functions.
///
/// StaticMethods return the underlying function unchanged, regardless
/// of whether accessed through a class or instance.
#[derive(Debug)]
#[repr(C)]
pub struct StaticMethodDescriptor {
    /// Object header for heap storage and fast type dispatch.
    pub header: ObjectHeader,
    /// The underlying function.
    function: Value,
}

impl StaticMethodDescriptor {
    /// Create a new staticmethod descriptor.
    pub fn new(function: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::STATICMETHOD),
            function,
        }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }
}

impl Descriptor for StaticMethodDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::StaticMethod
    }

    fn flags(&self) -> DescriptorFlags {
        DescriptorFlags::HAS_GET | DescriptorFlags::STATICMETHOD
    }

    fn get(&self, _obj: Option<Value>, _objtype: Value) -> PrismResult<Value> {
        // StaticMethod always returns the function unchanged
        // No binding, no wrapping - just the raw function
        Ok(self.function)
    }
}

impl PyObject for StaticMethodDescriptor {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
