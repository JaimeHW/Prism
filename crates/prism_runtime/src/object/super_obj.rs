//! Python `super()` object implementation.
//!
//! The `super` type provides a proxy object for delegating method calls to
//! parent or sibling classes in the MRO (Method Resolution Order).
//!
//! # Semantics
//!
//! `super()` searches the MRO starting *after* the specified class, not from
//! the beginning. This enables cooperative multiple inheritance.
//!
//! # Example (Python)
//!
//! ```python
//! class A:
//!     def method(self):
//!         return "A"
//!
//! class B(A):
//!     def method(self):
//!         return "B+" + super().method()  # Calls A.method
//!
//! class C(A):
//!     def method(self):
//!         return "C+" + super().method()  # Calls A.method
//!
//! class D(B, C):
//!     def method(self):
//!         return "D+" + super().method()  # Calls B.method
//!
//! # D().method() returns "D+B+C+A" due to MRO: [D, B, C, A, object]
//! ```
//!
//! # Usage Patterns
//!
//! 1. `super()` - No-arg form (Python 3), infers type and instance
//! 2. `super(type, obj)` - Explicit type and instance
//! 3. `super(type, type2)` - Type-only binding for classmethods
//!
//! # Performance
//!
//! - MRO index is cached after first lookup for repeated calls
//! - Inline caching in JIT can specialize super() dispatch
//! - Attribute access through super is O(n) in MRO length worst case

use crate::object::class::PyClassObject;
use crate::object::mro::ClassId;
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::Value;
use prism_core::intern::InternedString;
use std::sync::Arc;

// =============================================================================
// Super Binding Kind
// =============================================================================

/// The kind of object-to-type binding for super().
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuperBinding {
    /// Bound to an instance: `super(type, obj)` where `isinstance(obj, type)`
    Instance,
    /// Bound to a type: `super(type, type2)` where `issubclass(type2, type)`
    Type,
    /// Unbound: `super(type)` - rarely used in practice
    Unbound,
}

// =============================================================================
// Super Object
// =============================================================================

/// Python super() object - proxy for parent class method dispatch.
///
/// # Memory Layout
///
/// ```text
/// SuperObject (40 bytes on 64-bit)
/// ├── ObjectHeader (16 bytes)
/// ├── this_type: ClassId (4 bytes) - start of MRO search
/// ├── start_index: u16 (2 bytes) - cached MRO index + 1
/// ├── binding: SuperBinding (1 byte)
/// ├── _pad: u8 (1 byte)
/// ├── obj: Value (8 bytes) - bound instance or type
/// └── obj_type: ClassId (4 bytes) - type of bound object
/// ```
///
/// # Thread Safety
///
/// SuperObject is immutable after creation and safe to share across threads.
#[derive(Debug)]
#[repr(C)]
pub struct SuperObject {
    /// Object header (for GC).
    header: ObjectHeader,

    /// The type that super() was called with.
    /// MRO search starts *after* this type.
    this_type: ClassId,

    /// Cached index in MRO (start_index = mro_index + 1).
    /// 0 means not yet computed.
    start_index: u16,

    /// Kind of binding.
    binding: SuperBinding,

    /// The bound instance or type.
    /// - For instance binding: the instance object
    /// - For type binding: the type object
    /// - For unbound: Value::none()
    obj: Value,

    /// ClassId of the object's type (for instance) or the type itself.
    obj_type: ClassId,
}

impl SuperObject {
    /// Create a new super object bound to an instance.
    ///
    /// # Arguments
    ///
    /// * `this_type` - The type to start searching after in the MRO
    /// * `obj` - The bound instance
    /// * `obj_type` - The type of the bound instance
    ///
    /// # Example (Python)
    ///
    /// ```python
    /// class Child(Parent):
    ///     def method(self):
    ///         super().method()  # super(Child, self).method()
    /// ```
    #[inline]
    pub fn new_instance(this_type: ClassId, obj: Value, obj_type: ClassId) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SUPER),
            this_type,
            start_index: 0, // Will be computed on first lookup
            binding: SuperBinding::Instance,
            obj,
            obj_type,
        }
    }

    /// Create a new super object bound to a type (for classmethods).
    ///
    /// # Arguments
    ///
    /// * `this_type` - The type to start searching after in the MRO
    /// * `type_value` - The bound type as a Value
    /// * `bound_type` - ClassId of the bound type
    ///
    /// # Example (Python)
    ///
    /// ```python
    /// class Child(Parent):
    ///     @classmethod
    ///     def method(cls):
    ///         super().method()  # super(Child, cls).method()
    /// ```
    #[inline]
    pub fn new_type(this_type: ClassId, type_value: Value, bound_type: ClassId) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SUPER),
            this_type,
            start_index: 0,
            binding: SuperBinding::Type,
            obj: type_value,
            obj_type: bound_type,
        }
    }

    /// Create an unbound super object.
    ///
    /// Unbound super is rarely used but supported for completeness.
    ///
    /// # Example (Python)
    ///
    /// ```python
    /// s = super(Child)  # Unbound
    /// s.__get__(obj, type(obj)).method()  # Then bind it
    /// ```
    #[inline]
    pub fn new_unbound(this_type: ClassId) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::SUPER),
            this_type,
            start_index: 0,
            binding: SuperBinding::Unbound,
            obj: Value::none(),
            obj_type: ClassId::NONE,
        }
    }

    // =========================================================================
    // Accessors
    // =========================================================================

    /// Get the type that super() was called with.
    #[inline]
    pub fn this_type(&self) -> ClassId {
        self.this_type
    }

    /// Get the binding kind.
    #[inline]
    pub fn binding(&self) -> SuperBinding {
        self.binding
    }

    /// Get the bound object (instance or type).
    #[inline]
    pub fn obj(&self) -> Value {
        self.obj
    }

    /// Get the bound object's type.
    #[inline]
    pub fn obj_type(&self) -> ClassId {
        self.obj_type
    }

    /// Check if this super object is bound.
    #[inline]
    pub fn is_bound(&self) -> bool {
        self.binding != SuperBinding::Unbound
    }

    // =========================================================================
    // Method Lookup
    // =========================================================================

    /// Look up an attribute through super().
    ///
    /// This walks the MRO of obj_type starting *after* this_type.
    ///
    /// # Arguments
    ///
    /// * `name` - The attribute name to look up
    /// * `class_registry` - Function to get class objects by ClassId
    ///
    /// # Returns
    ///
    /// The attribute value and defining class, or None if not found.
    ///
    /// # Algorithm
    ///
    /// 1. Get MRO of `obj_type`
    /// 2. Find position of `this_type` in MRO
    /// 3. Search from position+1 to end of MRO
    /// 4. Return first match
    pub fn lookup_attr<F>(
        &self,
        name: &InternedString,
        class_registry: F,
    ) -> Option<SuperLookupResult>
    where
        F: Fn(ClassId) -> Option<Arc<PyClassObject>>,
    {
        // Get the MRO from the object's type
        let obj_class = class_registry(self.obj_type)?;
        let mro = obj_class.mro();

        // Find starting position (after this_type)
        let start = self.find_start_index(mro);

        // Walk MRO from start position
        for (offset, &class_id) in mro[start..].iter().enumerate() {
            if let Some(klass) = class_registry(class_id) {
                if let Some(value) = klass.get_attr(name) {
                    return Some(SuperLookupResult {
                        value,
                        defining_class: class_id,
                        mro_index: (start + offset) as u16,
                    });
                }
            }
        }

        None
    }

    /// Find the starting index in the MRO (after this_type).
    ///
    /// Uses cached value if available, otherwise computes and caches.
    fn find_start_index(&self, mro: &[ClassId]) -> usize {
        // If already computed, use cached value
        if self.start_index > 0 {
            return self.start_index as usize;
        }

        // Find this_type in MRO
        for (i, &class_id) in mro.iter().enumerate() {
            if class_id == self.this_type {
                // Return index after this_type
                return i + 1;
            }
        }

        // this_type not in MRO - start from beginning (shouldn't happen)
        0
    }

    /// Compute and cache the start index for future lookups.
    ///
    /// This is an optimization - call this after construction if you'll
    /// do multiple lookups through this super object.
    pub fn cache_start_index<F>(&mut self, class_registry: F)
    where
        F: Fn(ClassId) -> Option<Arc<PyClassObject>>,
    {
        if self.start_index > 0 {
            return; // Already cached
        }

        if let Some(obj_class) = class_registry(self.obj_type) {
            let mro = obj_class.mro();
            for (i, &class_id) in mro.iter().enumerate() {
                if class_id == self.this_type {
                    self.start_index = (i + 1) as u16;
                    return;
                }
            }
        }
    }

    // =========================================================================
    // Descriptor Protocol (super is itself a descriptor)
    // =========================================================================

    /// Implement descriptor protocol for super.
    ///
    /// When an unbound super is retrieved via `__get__`, it becomes bound.
    ///
    /// # Arguments
    ///
    /// * `obj` - The instance (or None for class access)
    /// * `obj_type` - The type
    ///
    /// # Returns
    ///
    /// For unbound super, returns a new bound super.
    /// For already-bound super, returns self (reconstructed).
    pub fn __get__(&self, obj: Option<Value>, obj_type: ClassId) -> SuperObject {
        if self.is_bound() || obj.is_none() {
            // Already bound or class access - reconstruct self
            SuperObject {
                header: ObjectHeader::new(TypeId::SUPER),
                this_type: self.this_type,
                start_index: self.start_index,
                binding: self.binding,
                obj: self.obj,
                obj_type: self.obj_type,
            }
        } else {
            // Bind to the instance
            SuperObject::new_instance(self.this_type, obj.unwrap(), obj_type)
        }
    }
}

impl PyObject for SuperObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Super Lookup Result
// =============================================================================

/// Result of a super lookup operation.
#[derive(Debug, Clone)]
pub struct SuperLookupResult {
    /// The attribute value found.
    pub value: Value,
    /// The class where the attribute was found.
    pub defining_class: ClassId,
    /// Index in MRO where found.
    pub mro_index: u16,
}
