//! Builtin function object type.
//!
//! Wraps native Rust function pointers as callable Python objects.

use prism_core::Value;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::{ObjectHeader, PyObject};
use std::sync::Arc;

use super::{BuiltinError, BuiltinFn};

// =============================================================================
// BuiltinFunctionObject
// =============================================================================

/// A callable builtin function object.
///
/// Wraps a native Rust function pointer with proper ObjectHeader for
/// type dispatch. When called, the VM extracts the function pointer
/// and invokes it directly with O(1) overhead.
///
/// # Memory Layout
///
/// Uses `#[repr(C)]` to ensure ObjectHeader is at offset 0 for
/// type extraction compatibility with JIT code.
#[repr(C)]
pub struct BuiltinFunctionObject {
    /// Object header with TypeId::BUILTIN_FUNCTION.
    pub header: ObjectHeader,
    /// The native function pointer.
    pub func: BuiltinFn,
    /// Function name for introspection.
    pub name: Arc<str>,
}

impl BuiltinFunctionObject {
    /// Create a new builtin function object.
    #[inline]
    pub fn new(name: impl Into<Arc<str>>, func: BuiltinFn) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            func,
            name: name.into(),
        }
    }

    /// Call this builtin function with arguments.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, BuiltinError> {
        (self.func)(args)
    }

    /// Get the function name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl PyObject for BuiltinFunctionObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

impl std::fmt::Debug for BuiltinFunctionObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<builtin function {}>", self.name)
    }
}

impl std::fmt::Display for BuiltinFunctionObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<builtin function {}>", self.name)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_builtin(_args: &[Value]) -> Result<Value, BuiltinError> {
        Ok(Value::int(42).unwrap())
    }

    #[test]
    fn test_builtin_function_object() {
        let builtin = BuiltinFunctionObject::new("test", test_builtin);

        assert_eq!(builtin.header.type_id, TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "test");

        let result = builtin.call(&[]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_type_id_extraction() {
        let builtin = BuiltinFunctionObject::new("abs", test_builtin);
        let ptr = &builtin as *const _ as *const ();

        // Verify ObjectHeader is at offset 0
        let header_ptr = ptr as *const ObjectHeader;
        let type_id = unsafe { (*header_ptr).type_id };

        assert_eq!(type_id, TypeId::BUILTIN_FUNCTION);
    }
}
