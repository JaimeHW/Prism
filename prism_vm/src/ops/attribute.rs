//! Descriptor-aware attribute resolution for Python semantics.
//!
//! This module implements Python's attribute access protocol with optimal performance.
//! It provides hooks for the full descriptor protocol while initially working with
//! ShapedObject-based instance attribute storage.
//!
//! # Python Attribute Resolution Order
//!
//! For `object.attr`:
//! 1. Check type's MRO for **data descriptor** (has `__set__` or `__delete__`)
//!    - If found, call `descriptor.__get__(object, type(object))`
//! 2. Check instance's `__dict__` (ShapedObject properties)
//!    - If found, return value directly
//! 3. Check type's MRO for **non-data descriptor** or class attribute
//!    - If descriptor, call `descriptor.__get__(object, type(object))`
//!    - If plain attribute, return it
//! 4. Raise `AttributeError`
//!
//! # Performance Design
//!
//! - **TypeId dispatch**: O(1) type identification via ObjectHeader
//! - **Shape-based lookup**: O(1) property access via hidden class slots
//! - **Inline cache ready**: Prepared for monomorphic/polymorphic IC integration
//! - **Descriptor flags**: Fast path to skip protocol dispatch for non-descriptors

use crate::error::RuntimeError;
use prism_core::Value;
use prism_core::intern::InternedString;
use prism_runtime::TypeId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;

// =============================================================================
// Attribute Resolution Result
// =============================================================================

/// Result of attribute lookup with descriptor classification.
#[derive(Debug, Clone)]
pub enum AttrLookupResult {
    /// Found in instance __dict__ (ShapedObject property).
    InstanceAttr(Value),
    /// Found as data descriptor in type hierarchy.
    DataDescriptor {
        descriptor: Value,
        owner_type: TypeId,
    },
    /// Found as non-data descriptor in type hierarchy.
    NonDataDescriptor {
        descriptor: Value,
        owner_type: TypeId,
    },
    /// Found as plain class attribute (not a descriptor).
    ClassAttr { value: Value, owner_type: TypeId },
    /// Attribute not found anywhere.
    NotFound,
}

impl AttrLookupResult {
    /// Check if attribute was found.
    #[inline]
    pub fn is_found(&self) -> bool {
        !matches!(self, AttrLookupResult::NotFound)
    }

    /// Check if this requires descriptor protocol invocation.
    #[inline]
    pub fn needs_descriptor_get(&self) -> bool {
        matches!(
            self,
            AttrLookupResult::DataDescriptor { .. } | AttrLookupResult::NonDataDescriptor { .. }
        )
    }

    /// Get the value if it's a simple case (instance attr or class attr).
    #[inline]
    pub fn simple_value(&self) -> Option<Value> {
        match self {
            AttrLookupResult::InstanceAttr(v) => Some(*v),
            AttrLookupResult::ClassAttr { value, .. } => Some(*value),
            _ => None,
        }
    }
}

// =============================================================================
// Descriptor Detection
// =============================================================================

/// Check if a type ID is a known data descriptor type.
///
/// A data descriptor defines `__set__` or `__delete__`.
/// Currently we have no built-in data descriptor types registered.
/// This will be extended when property descriptors are added.
#[inline]
pub fn is_data_descriptor_type(_type_id: TypeId) -> bool {
    // No built-in data descriptor types yet
    // Future: TypeId::PROPERTY when added
    false
}

/// Check if a type ID is a known non-data descriptor type.
///
/// A non-data descriptor defines `__get__` but not `__set__` or `__delete__`.
/// Functions and methods are non-data descriptors.
#[inline]
pub fn is_non_data_descriptor_type(type_id: TypeId) -> bool {
    matches!(type_id, TypeId::FUNCTION | TypeId::METHOD)
}

/// Check if a value is a data descriptor.
///
/// A data descriptor defines `__set__` or `__delete__`.
/// This is a fast path check using TypeId.
#[inline]
pub fn is_data_descriptor(value: Value) -> bool {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        is_data_descriptor_type(type_id)
    } else {
        false
    }
}

/// Check if a value is a non-data descriptor.
///
/// A non-data descriptor defines `__get__` but not `__set__` or `__delete__`.
#[inline]
pub fn is_non_data_descriptor(value: Value) -> bool {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        is_non_data_descriptor_type(type_id)
    } else {
        false
    }
}

/// Extract TypeId from an object pointer.
#[inline]
fn extract_type_id(ptr: *const ()) -> TypeId {
    use prism_runtime::ObjectHeader;
    unsafe { (*(ptr as *const ObjectHeader)).type_id }
}

/// Check if a TypeId represents a user-defined type.
#[inline]
pub fn is_user_defined_type(type_id: TypeId) -> bool {
    type_id.raw() >= TypeId::FIRST_USER_TYPE
}

// =============================================================================
// Instance Attribute Access
// =============================================================================

/// Get an attribute from a ShapedObject's instance dict.
///
/// This is the fast path for instance attribute access, bypassing
/// the full descriptor protocol.
#[inline]
pub fn get_instance_attr(obj: &ShapedObject, name: &str) -> Option<Value> {
    obj.get_property(name)
}

/// Set an attribute on a ShapedObject's instance dict.
///
/// Returns true if the property was successfully set.
#[inline]
pub fn set_instance_attr(obj: &mut ShapedObject, name: InternedString, value: Value) {
    let registry = shape_registry();
    obj.set_property(name, value, registry);
}

/// Delete an attribute from a ShapedObject's instance dict.
///
/// Returns true if the attribute existed and was deleted.
#[inline]
pub fn delete_instance_attr(obj: &mut ShapedObject, name: &str) -> bool {
    obj.delete_property(name)
}

// =============================================================================
// Full Attribute Resolution (with Descriptor Protocol)
// =============================================================================

/// Resolve attribute access following Python's full protocol.
///
/// # Arguments
///
/// * `obj` - The object to access the attribute on
/// * `obj_type_id` - The TypeId of the object
/// * `name` - The attribute name
///
/// # Returns
///
/// The attribute value or an error.
///
/// # Descriptor Protocol Integration
///
/// This function currently implements a simplified protocol:
/// 1. Check instance dict (ShapedObject properties)
/// 2. Fall back to AttributeError
///
/// Full MRO-based descriptor lookup will be added when class registry
/// infrastructure is complete. The design is ready for this extension.
pub fn resolve_get_attr(
    obj: Value,
    obj_type_id: TypeId,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    // Fast path: Check instance dict for ShapedObject types
    if let Some(ptr) = obj.as_object_ptr() {
        if obj_type_id == TypeId::OBJECT || is_user_defined_type(obj_type_id) {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };

            // Step 2 of descriptor protocol: Check instance __dict__
            if let Some(value) = shaped.get_property(&**name) {
                return Ok(value);
            }
        }
    }

    // TODO: Implement full MRO-based lookup when class registry is available
    // Step 1: Check for data descriptor in type's MRO
    // Step 3: Check for non-data descriptor or class attribute

    // Step 4: AttributeError
    Err(RuntimeError::attribute_error(
        obj_type_id.name(),
        name.as_str(),
    ))
}

/// Set an attribute following Python's protocol.
///
/// # Arguments
///
/// * `obj` - The object to set the attribute on
/// * `obj_type_id` - The TypeId of the object
/// * `name` - The attribute name
/// * `value` - The value to set
///
/// # Returns
///
/// Ok(()) on success, or an error.
pub fn resolve_set_attr(
    obj: Value,
    obj_type_id: TypeId,
    name: InternedString,
    value: Value,
) -> Result<(), RuntimeError> {
    // TODO: Check for data descriptor with __set__ in type's MRO

    // Set in instance dict for ShapedObject types
    if let Some(ptr) = obj.as_object_ptr() {
        if obj_type_id == TypeId::OBJECT || is_user_defined_type(obj_type_id) {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            let registry = shape_registry();
            shaped.set_property(name, value, registry);
            return Ok(());
        }
    }

    // Cannot set attribute on this type
    Err(RuntimeError::attribute_error(
        obj_type_id.name(),
        name.as_str(),
    ))
}

/// Delete an attribute following Python's protocol.
///
/// # Arguments
///
/// * `obj` - The object to delete the attribute from
/// * `obj_type_id` - The TypeId of the object
/// * `name` - The attribute name
///
/// # Returns
///
/// Ok(()) on success, or an error.
pub fn resolve_del_attr(
    obj: Value,
    obj_type_id: TypeId,
    name: &InternedString,
) -> Result<(), RuntimeError> {
    // TODO: Check for data descriptor with __delete__ in type's MRO

    // Delete from instance dict for ShapedObject types
    if let Some(ptr) = obj.as_object_ptr() {
        if obj_type_id == TypeId::OBJECT || is_user_defined_type(obj_type_id) {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            if shaped.delete_property(&**name) {
                return Ok(());
            }
        }
    }

    // Attribute not found
    Err(RuntimeError::attribute_error(
        obj_type_id.name(),
        name.as_str(),
    ))
}

// =============================================================================
// Inline Cache Support
// =============================================================================

/// Inline cache entry for attribute access.
///
/// Caches the shape ID and slot index for monomorphic attribute access.
/// This enables O(1) attribute lookup after the first access.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttrIC {
    /// Cached shape ID for guard check.
    pub shape_id: u64,
    /// Cached slot index for direct access.
    pub slot_index: u16,
    /// Property flags for descriptor handling.
    pub flags: u8,
}

impl AttrIC {
    /// Create a new empty IC entry.
    pub const fn new() -> Self {
        Self {
            shape_id: 0,
            slot_index: 0,
            flags: 0,
        }
    }

    /// Check if this IC entry matches the given shape.
    #[inline]
    pub fn matches(&self, shape_id: u64) -> bool {
        self.shape_id == shape_id && self.shape_id != 0
    }

    /// Update this IC entry with shape information.
    #[inline]
    pub fn update(&mut self, shape_id: u64, slot_index: u16) {
        self.shape_id = shape_id;
        self.slot_index = slot_index;
    }

    /// Invalidate this IC entry.
    #[inline]
    pub fn invalidate(&mut self) {
        self.shape_id = 0;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // AttrLookupResult Tests
    // =========================================================================

    #[test]
    fn test_lookup_result_not_found() {
        let result = AttrLookupResult::NotFound;
        assert!(!result.is_found());
        assert!(!result.needs_descriptor_get());
        assert!(result.simple_value().is_none());
    }

    #[test]
    fn test_lookup_result_instance_attr() {
        let value = Value::int(42).unwrap();
        let result = AttrLookupResult::InstanceAttr(value);
        assert!(result.is_found());
        assert!(!result.needs_descriptor_get());
        assert_eq!(result.simple_value(), Some(value));
    }

    #[test]
    fn test_lookup_result_class_attr() {
        let value = Value::bool(true);
        let result = AttrLookupResult::ClassAttr {
            value,
            owner_type: TypeId::OBJECT,
        };
        assert!(result.is_found());
        assert!(!result.needs_descriptor_get());
        assert_eq!(result.simple_value(), Some(value));
    }

    #[test]
    fn test_lookup_result_data_descriptor() {
        let result = AttrLookupResult::DataDescriptor {
            descriptor: Value::none(),
            owner_type: TypeId::OBJECT,
        };
        assert!(result.is_found());
        assert!(result.needs_descriptor_get());
        assert!(result.simple_value().is_none());
    }

    #[test]
    fn test_lookup_result_non_data_descriptor() {
        let result = AttrLookupResult::NonDataDescriptor {
            descriptor: Value::none(),
            owner_type: TypeId::FUNCTION,
        };
        assert!(result.is_found());
        assert!(result.needs_descriptor_get());
        assert!(result.simple_value().is_none());
    }

    // =========================================================================
    // AttrIC Tests
    // =========================================================================

    #[test]
    fn test_ic_new_is_invalid() {
        let ic = AttrIC::new();
        assert!(!ic.matches(1));
        assert!(!ic.matches(0));
    }

    #[test]
    fn test_ic_update_and_match() {
        let mut ic = AttrIC::new();
        ic.update(12345, 3);

        assert!(ic.matches(12345));
        assert!(!ic.matches(12346));
        assert_eq!(ic.slot_index, 3);
    }

    #[test]
    fn test_ic_invalidate() {
        let mut ic = AttrIC::new();
        ic.update(12345, 3);
        assert!(ic.matches(12345));

        ic.invalidate();
        assert!(!ic.matches(12345));
    }

    // =========================================================================
    // Descriptor Detection Tests
    // =========================================================================

    #[test]
    fn test_is_data_descriptor_primitives() {
        // Primitives are never descriptors
        assert!(!is_data_descriptor(Value::none()));
        assert!(!is_data_descriptor(Value::int(42).unwrap()));
        assert!(!is_data_descriptor(Value::bool(true)));
        assert!(!is_data_descriptor(Value::float(3.14)));
    }

    #[test]
    fn test_is_non_data_descriptor_primitives() {
        // Primitives are never descriptors
        assert!(!is_non_data_descriptor(Value::none()));
        assert!(!is_non_data_descriptor(Value::int(42).unwrap()));
        assert!(!is_non_data_descriptor(Value::bool(true)));
        assert!(!is_non_data_descriptor(Value::float(3.14)));
    }

    // =========================================================================
    // User-Defined Type Detection Tests
    // =========================================================================

    #[test]
    fn test_builtin_types_not_user_defined() {
        assert!(!is_user_defined_type(TypeId::OBJECT));
        assert!(!is_user_defined_type(TypeId::INT));
        assert!(!is_user_defined_type(TypeId::STR));
        assert!(!is_user_defined_type(TypeId::LIST));
        assert!(!is_user_defined_type(TypeId::DICT));
    }

    #[test]
    fn test_user_defined_types() {
        // First user type starts at 256
        let user_type = TypeId::from_raw(256);
        assert!(is_user_defined_type(user_type));

        let higher_user_type = TypeId::from_raw(1000);
        assert!(is_user_defined_type(higher_user_type));
    }

    // =========================================================================
    // Descriptor Type Detection Tests
    // =========================================================================

    #[test]
    fn test_function_is_non_data_descriptor() {
        assert!(is_non_data_descriptor_type(TypeId::FUNCTION));
        assert!(is_non_data_descriptor_type(TypeId::METHOD));
    }

    #[test]
    fn test_builtin_types_not_descriptors() {
        assert!(!is_data_descriptor_type(TypeId::INT));
        assert!(!is_data_descriptor_type(TypeId::STR));
        assert!(!is_non_data_descriptor_type(TypeId::INT));
        assert!(!is_non_data_descriptor_type(TypeId::STR));
    }
}
