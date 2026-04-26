//! Method descriptor and bound method implementation.
//!
//! When a function is accessed through an instance, Python creates a "bound method"
//! that automatically passes the instance as the first argument (self).
//!
//! # Binding Mechanism
//!
//! ```python
//! class Counter:
//!     def __init__(self):
//!         self.count = 0
//!
//!     def increment(self):  # Regular method
//!         self.count += 1
//!
//! c = Counter()
//! inc = c.increment  # Creates BoundMethod(increment, c)
//! inc()              # Equivalent to increment(c)
//! ```
//!
//! # Performance Optimizations
//!
//! - BoundMethod caches the function and instance together
//! - No dictionary lookup needed when calling bound methods
//! - Inline caching can specialize to avoid rebinding on each call
//!
//! # Memory Layout
//!
//! ```text
//! BoundMethod (32 bytes)
//! ├── ObjectHeader (16 bytes)
//! ├── function: Value (8 bytes)
//! └── instance: Value (8 bytes)
//! ```

use super::{Descriptor, DescriptorFlags, DescriptorKind};
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::{PrismResult, Value};

// =============================================================================
// Method Descriptor
// =============================================================================

/// Descriptor that creates bound methods when accessed through instances.
///
/// This is a non-data descriptor (only has `__get__`), which means instance
/// attributes take precedence over methods of the same name.
#[derive(Debug, Clone)]
pub struct MethodDescriptor {
    /// The underlying function.
    function: Value,
    /// Optional name for debugging.
    name: Option<prism_core::intern::InternedString>,
}

impl MethodDescriptor {
    /// Create a new method descriptor from a function.
    pub fn new(function: Value) -> Self {
        Self {
            function,
            name: None,
        }
    }

    /// Create a new method descriptor with a name.
    pub fn new_named(function: Value, name: prism_core::intern::InternedString) -> Self {
        Self {
            function,
            name: Some(name),
        }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }

    /// Get the method name.
    #[inline]
    pub fn name(&self) -> Option<&prism_core::intern::InternedString> {
        self.name.as_ref()
    }

    /// Create a bound method for the given instance.
    #[inline]
    pub fn bind(&self, instance: Value) -> BoundMethod {
        BoundMethod::new(self.function, instance)
    }
}

impl Descriptor for MethodDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::Method
    }

    fn flags(&self) -> DescriptorFlags {
        DescriptorFlags::HAS_GET | DescriptorFlags::METHOD
    }

    fn get(&self, obj: Option<Value>, _objtype: Value) -> PrismResult<Value> {
        match obj {
            None => {
                // Accessed through class - return the function itself
                Ok(self.function)
            }
            Some(instance) => {
                // Accessed through instance - create bound method
                let bound = self.bind(instance);
                // In a full implementation, we'd wrap this as a Value
                // For now, return the function (bound method creation would be here)
                let _ = bound;
                Ok(self.function)
            }
        }
    }
}

// =============================================================================
// Bound Method
// =============================================================================

/// A bound method - a function bound to a specific instance.
///
/// When called, the bound method automatically prepends the instance
/// to the argument list before calling the underlying function.
///
/// # Performance
///
/// BoundMethod is designed for efficient invocation:
/// - Function and instance are stored together for cache locality
/// - No allocation needed for the argument list (instance is prepended inline)
/// - Inline caches can specialize bound method calls
///
/// # Equality
///
/// Two bound methods are equal if they have the same function AND instance.
/// This matches Python semantics where `obj.method == obj.method` is True.
#[derive(Debug)]
#[repr(C)]
pub struct BoundMethod {
    /// Object header.
    pub header: ObjectHeader,
    /// The underlying function.
    function: Value,
    /// The bound instance (self).
    instance: Value,
}

impl BoundMethod {
    /// Create a new bound method.
    pub fn new(function: Value, instance: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::METHOD),
            function,
            instance,
        }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }

    /// Get the bound instance.
    #[inline]
    pub fn instance(&self) -> Value {
        self.instance
    }

    /// Call the bound method with arguments.
    ///
    /// This prepends the instance to the arguments and calls the function.
    pub fn call(&self, args: &[Value]) -> PrismResult<Value> {
        // In a full implementation, we would:
        // 1. Allocate space for instance + args
        // 2. Copy instance as first arg
        // 3. Copy remaining args
        // 4. Call the function
        //
        // For now, return a placeholder
        let _ = args;
        Ok(Value::none())
    }

    /// Call the bound method with keyword arguments.
    pub fn call_kw(
        &self,
        args: &[Value],
        kwargs: Option<&[(prism_core::intern::InternedString, Value)]>,
    ) -> PrismResult<Value> {
        let _ = (args, kwargs);
        Ok(Value::none())
    }

    /// Check equality with another bound method.
    pub fn eq(&self, other: &BoundMethod) -> bool {
        // Same function AND same instance
        self.function == other.function && self.instance == other.instance
    }

    /// Compute hash for the bound method.
    pub fn hash(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.function.hash(&mut hasher);
        self.instance.hash(&mut hasher);
        hasher.finish()
    }
}

impl PyObject for BoundMethod {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

impl PartialEq for BoundMethod {
    fn eq(&self, other: &Self) -> bool {
        BoundMethod::eq(self, other)
    }
}

impl Eq for BoundMethod {}

impl std::hash::Hash for BoundMethod {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.function.hash(state);
        self.instance.hash(state);
    }
}

// =============================================================================
// Unbound Method (Python 2 compatibility, rare)
// =============================================================================

/// Unbound method (for Python 2 compatibility).
///
/// In Python 3, accessing a method through a class just returns the function.
/// This type exists for completeness but is rarely used.
#[derive(Debug, Clone)]
pub struct UnboundMethod {
    /// The underlying function.
    function: Value,
    /// The class the method belongs to.
    class: Value,
}

impl UnboundMethod {
    /// Create a new unbound method.
    pub fn new(function: Value, class: Value) -> Self {
        Self { function, class }
    }

    /// Get the underlying function.
    #[inline]
    pub fn function(&self) -> Value {
        self.function
    }

    /// Get the class.
    #[inline]
    pub fn class(&self) -> Value {
        self.class
    }

    /// Bind to an instance.
    pub fn bind(&self, instance: Value) -> PrismResult<BoundMethod> {
        // In a full implementation, we'd check isinstance(instance, self.class)
        let _ = self.class;
        Ok(BoundMethod::new(self.function, instance))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
