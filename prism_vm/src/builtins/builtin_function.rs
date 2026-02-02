//! Builtin function object type.
//!
//! Provides the `BuiltinFunctionObject` type which wraps native Rust functions
//! as callable Python objects with proper object headers for type dispatch.

use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use std::sync::Arc;

/// Type alias for builtin function pointers (mirrors mod.rs).
type BuiltinFnPtr = fn(&[Value]) -> Result<Value, super::BuiltinError>;

/// A builtin function object.
///
/// This wraps a native Rust function as a Python callable object.
/// It includes an ObjectHeader for proper type dispatch in the VM.
#[repr(C)]
pub struct BuiltinFunctionObject {
    /// Object header for type dispatch.
    pub header: ObjectHeader,
    /// Function name for display and debugging.
    pub name: Arc<str>,
    /// The actual function implementation.
    func: BuiltinFnPtr,
}

impl BuiltinFunctionObject {
    /// Create a new builtin function object.
    pub fn new(name: Arc<str>, func: BuiltinFnPtr) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func,
        }
    }

    /// Call the builtin function with arguments.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, super::BuiltinError> {
        (self.func)(args)
    }

    /// Get the function name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl std::fmt::Debug for BuiltinFunctionObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BuiltinFunctionObject")
            .field("name", &self.name)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_builtin(_args: &[Value]) -> Result<Value, super::super::BuiltinError> {
        Ok(Value::none())
    }

    #[test]
    fn test_builtin_function_object_creation() {
        let func = BuiltinFunctionObject::new(Arc::from("test"), dummy_builtin);
        assert_eq!(func.name(), "test");
        assert_eq!(func.header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_builtin_function_object_call() {
        let func = BuiltinFunctionObject::new(Arc::from("test"), dummy_builtin);
        let result = func.call(&[]);
        assert!(result.is_ok());
    }
}
