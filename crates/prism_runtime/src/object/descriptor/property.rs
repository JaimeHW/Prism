//! Property descriptor implementation.
//!
//! Properties provide computed attribute access via getter, setter, and deleter
//! functions. This is the mechanism behind Python's `@property` decorator.
//!
//! # Example (Python)
//!
//! ```python
//! class Circle:
//!     def __init__(self, radius):
//!         self._radius = radius
//!
//!     @property
//!     def radius(self):
//!         return self._radius
//!
//!     @radius.setter
//!     def radius(self, value):
//!         if value < 0:
//!             raise ValueError("Radius cannot be negative")
//!         self._radius = value
//!
//!     @property
//!     def area(self):
//!         return 3.14159 * self._radius ** 2
//! ```
//!
//! # Performance
//!
//! - Flags are cached for fast data descriptor checks
//! - Function pointers stored directly (no indirection)
//! - Inline caching can specialize property access

use super::{Descriptor, DescriptorFlags, DescriptorInvoker, DescriptorKind};
use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_core::{PrismError, PrismResult, Value};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

// =============================================================================
// Property Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags for property configuration.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct PropertyFlags: u8 {
        /// Property has a getter.
        const HAS_GETTER = 1 << 0;
        /// Property has a setter.
        const HAS_SETTER = 1 << 1;
        /// Property has a deleter.
        const HAS_DELETER = 1 << 2;
        /// Property has documentation.
        const HAS_DOC = 1 << 3;
        /// Documentation was copied from the getter and should be refreshed
        /// when `.getter()` creates a replacement descriptor.
        const GETTER_DOC = 1 << 4;
    }
}

impl Default for PropertyFlags {
    fn default() -> Self {
        Self::empty()
    }
}

// =============================================================================
// Property Descriptor
// =============================================================================

/// Property descriptor with optional getter, setter, and deleter.
///
/// # Memory Layout
///
/// Properties store function values directly for the getter, setter, and deleter.
/// Each is optional, allowing read-only, write-only, or delete-only properties
/// (though read-only is by far the most common).
///
#[derive(Debug)]
#[repr(C)]
pub struct PropertyDescriptor {
    /// Object header for heap storage and fast type dispatch.
    pub header: ObjectHeader,
    /// Property flags for fast checking.
    flags: AtomicU8,
    /// The getter function (fget).
    getter: Option<Value>,
    /// The setter function (fset).
    setter: Option<Value>,
    /// The deleter function (fdel).
    deleter: Option<Value>,
    /// Documentation string.
    doc: AtomicU64,
}

impl PropertyDescriptor {
    /// Create a new property with only a getter (read-only).
    pub fn new_getter(getter: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new(PropertyFlags::HAS_GETTER.bits()),
            getter: Some(getter),
            setter: None,
            deleter: None,
            doc: AtomicU64::new(Value::none().to_bits()),
        }
    }

    /// Create a new property with getter and setter.
    pub fn new_getter_setter(getter: Value, setter: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new((PropertyFlags::HAS_GETTER | PropertyFlags::HAS_SETTER).bits()),
            getter: Some(getter),
            setter: Some(setter),
            deleter: None,
            doc: AtomicU64::new(Value::none().to_bits()),
        }
    }

    /// Create a new property with all three functions.
    pub fn new_full(
        getter: Option<Value>,
        setter: Option<Value>,
        deleter: Option<Value>,
        doc: Option<Value>,
    ) -> Self {
        Self::new_full_with_doc_source(getter, setter, deleter, doc, false)
    }

    /// Create a new property and record whether its doc came from the getter.
    pub fn new_full_with_doc_source(
        getter: Option<Value>,
        setter: Option<Value>,
        deleter: Option<Value>,
        doc: Option<Value>,
        getter_doc: bool,
    ) -> Self {
        let mut flags = PropertyFlags::empty();
        if getter.is_some() {
            flags |= PropertyFlags::HAS_GETTER;
        }
        if setter.is_some() {
            flags |= PropertyFlags::HAS_SETTER;
        }
        if deleter.is_some() {
            flags |= PropertyFlags::HAS_DELETER;
        }
        if doc.is_some() {
            flags |= PropertyFlags::HAS_DOC;
        }
        if getter_doc {
            flags |= PropertyFlags::GETTER_DOC;
        }

        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new(flags.bits()),
            getter,
            setter,
            deleter,
            doc: AtomicU64::new(doc.unwrap_or_else(Value::none).to_bits()),
        }
    }

    /// Get property flags.
    #[inline]
    pub fn property_flags(&self) -> PropertyFlags {
        PropertyFlags::from_bits_truncate(self.flags.load(Ordering::Relaxed))
    }

    /// Check if property has a getter.
    #[inline]
    pub fn has_getter(&self) -> bool {
        self.property_flags().contains(PropertyFlags::HAS_GETTER)
    }

    /// Check if property has a setter.
    #[inline]
    pub fn has_setter(&self) -> bool {
        self.property_flags().contains(PropertyFlags::HAS_SETTER)
    }

    /// Check if property has a deleter.
    #[inline]
    pub fn has_deleter(&self) -> bool {
        self.property_flags().contains(PropertyFlags::HAS_DELETER)
    }

    /// Check if the property has documentation.
    #[inline]
    pub fn has_doc(&self) -> bool {
        self.property_flags().contains(PropertyFlags::HAS_DOC)
    }

    /// Check if the documentation was copied from the getter.
    #[inline]
    pub fn doc_from_getter(&self) -> bool {
        self.property_flags().contains(PropertyFlags::GETTER_DOC)
    }

    /// Get the getter function.
    pub fn getter(&self) -> Option<Value> {
        self.getter
    }

    /// Get the setter function.
    pub fn setter(&self) -> Option<Value> {
        self.setter
    }

    /// Get the deleter function.
    pub fn deleter(&self) -> Option<Value> {
        self.deleter
    }

    /// Get the documentation.
    pub fn doc(&self) -> Option<Value> {
        if self.has_doc() {
            Some(Value::from_bits(self.doc.load(Ordering::Acquire)))
        } else {
            None
        }
    }

    /// Replace the documentation exposed through `property.__doc__`.
    #[inline]
    pub fn set_doc(&self, doc: Value) {
        self.doc.store(doc.to_bits(), Ordering::Release);
        self.flags
            .fetch_or(PropertyFlags::HAS_DOC.bits(), Ordering::Release);
    }

    /// Create a new property with a different getter.
    pub fn with_getter(&self, getter: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new((self.property_flags() | PropertyFlags::HAS_GETTER).bits()),
            getter: Some(getter),
            setter: self.setter,
            deleter: self.deleter,
            doc: AtomicU64::new(self.doc.load(Ordering::Acquire)),
        }
    }

    /// Create a new property with a different setter.
    pub fn with_setter(&self, setter: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new((self.property_flags() | PropertyFlags::HAS_SETTER).bits()),
            getter: self.getter,
            setter: Some(setter),
            deleter: self.deleter,
            doc: AtomicU64::new(self.doc.load(Ordering::Acquire)),
        }
    }

    /// Create a new property with a different deleter.
    pub fn with_deleter(&self, deleter: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::PROPERTY),
            flags: AtomicU8::new((self.property_flags() | PropertyFlags::HAS_DELETER).bits()),
            getter: self.getter,
            setter: self.setter,
            deleter: Some(deleter),
            doc: AtomicU64::new(self.doc.load(Ordering::Acquire)),
        }
    }
}

impl Descriptor for PropertyDescriptor {
    fn kind(&self) -> DescriptorKind {
        DescriptorKind::Property
    }

    fn flags(&self) -> DescriptorFlags {
        let mut flags = DescriptorFlags::empty();

        if self.has_getter() {
            flags |= DescriptorFlags::HAS_GET;
        }
        if self.has_setter() {
            flags |= DescriptorFlags::HAS_SET;
        }
        if self.has_deleter() {
            flags |= DescriptorFlags::HAS_DELETE;
        }

        // Python properties are always data descriptors because assignment and
        // deletion route through the property object even when the underlying
        // accessor is absent and would raise at runtime.
        flags |= DescriptorFlags::DATA_DESCRIPTOR;

        flags
    }

    fn get(&self, obj: Option<Value>, objtype: Value) -> PrismResult<Value> {
        // If accessed through class (obj is None), return the property itself
        if obj.is_none() {
            let _ = objtype;
            return Ok(Value::object_ptr(self as *const Self as *const ()));
        }

        if self.getter.is_some() {
            let _ = objtype;
            Err(PrismError::internal(
                "property getter requires a DescriptorInvoker",
            ))
        } else {
            Err(PrismError::attribute("property has no getter"))
        }
    }

    fn get_with_invoker(
        &self,
        obj: Option<Value>,
        objtype: Value,
        invoker: &mut dyn DescriptorInvoker,
    ) -> PrismResult<Value> {
        if obj.is_none() {
            let _ = objtype;
            return Ok(Value::object_ptr(self as *const Self as *const ()));
        }

        let getter = self
            .getter
            .ok_or_else(|| PrismError::attribute("property has no getter"))?;
        invoker.call(getter, &[obj.expect("checked above")])
    }

    fn set(&self, obj: Value, value: Value) -> PrismResult<()> {
        if self.setter.is_some() {
            let _ = (obj, value);
            Err(PrismError::internal(
                "property setter requires a DescriptorInvoker",
            ))
        } else {
            Err(PrismError::attribute("property is read-only"))
        }
    }

    fn set_with_invoker(
        &self,
        obj: Value,
        value: Value,
        invoker: &mut dyn DescriptorInvoker,
    ) -> PrismResult<()> {
        let setter = self
            .setter
            .ok_or_else(|| PrismError::attribute("property is read-only"))?;
        invoker.call(setter, &[obj, value]).map(|_| ())
    }

    fn delete(&self, obj: Value) -> PrismResult<()> {
        if self.deleter.is_some() {
            let _ = obj;
            Err(PrismError::internal(
                "property deleter requires a DescriptorInvoker",
            ))
        } else {
            Err(PrismError::attribute("property does not support deletion"))
        }
    }

    fn delete_with_invoker(
        &self,
        obj: Value,
        invoker: &mut dyn DescriptorInvoker,
    ) -> PrismResult<()> {
        let deleter = self
            .deleter
            .ok_or_else(|| PrismError::attribute("property does not support deletion"))?;
        invoker.call(deleter, &[obj]).map(|_| ())
    }
}

impl PyObject for PropertyDescriptor {
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

    #[derive(Default)]
    struct RecordingInvoker {
        calls: Vec<(Value, Vec<Value>)>,
        returns: Vec<Value>,
    }

    impl DescriptorInvoker for RecordingInvoker {
        fn call(&mut self, callable: Value, args: &[Value]) -> PrismResult<Value> {
            self.calls.push((callable, args.to_vec()));
            Ok(self.returns.pop().unwrap_or_else(Value::none))
        }
    }

    #[test]
    fn property_get_uses_invoker() {
        let getter = Value::int_unchecked(100);
        let instance = Value::int_unchecked(7);
        let expected = Value::int_unchecked(42);
        let property = PropertyDescriptor::new_getter(getter);
        let mut invoker = RecordingInvoker {
            returns: vec![expected],
            ..RecordingInvoker::default()
        };

        assert_eq!(
            property
                .get_with_invoker(Some(instance), Value::none(), &mut invoker)
                .unwrap(),
            expected
        );
        assert_eq!(invoker.calls, vec![(getter, vec![instance])]);
        assert!(property.get(Some(instance), Value::none()).is_err());
    }

    #[test]
    fn property_set_and_delete_use_invoker() {
        let instance = Value::int_unchecked(7);
        let setter = Value::int_unchecked(101);
        let deleter = Value::int_unchecked(102);
        let assigned = Value::int_unchecked(9);
        let property = PropertyDescriptor::new_full(None, Some(setter), Some(deleter), None);
        let mut invoker = RecordingInvoker::default();

        property
            .set_with_invoker(instance, assigned, &mut invoker)
            .unwrap();
        property
            .delete_with_invoker(instance, &mut invoker)
            .unwrap();

        assert_eq!(
            invoker.calls,
            vec![
                (setter, vec![instance, assigned]),
                (deleter, vec![instance])
            ]
        );
        assert!(property.set(instance, assigned).is_err());
        assert!(property.delete(instance).is_err());
    }
}
