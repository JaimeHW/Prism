//! Function and closure objects.
//!
//! Implements Python function objects for the interpreter.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use prism_compiler::bytecode::CodeObject;
use prism_core::Value;
use std::sync::Arc;

// =============================================================================
// Closure Environment
// =============================================================================

/// Captured variable environment for closures.
///
/// Forms a chain of captured values from enclosing scopes.
/// Uses Arc for shared ownership between closure instances.
#[repr(C)]
pub struct ClosureEnv {
    /// Captured values from enclosing scope.
    values: Box<[Value]>,
    /// Parent closure environment (for nested closures).
    parent: Option<Arc<ClosureEnv>>,
}

impl ClosureEnv {
    /// Create a new closure environment.
    pub fn new(values: Box<[Value]>, parent: Option<Arc<ClosureEnv>>) -> Self {
        Self { values, parent }
    }

    /// Create an empty closure environment.
    pub fn empty() -> Self {
        Self {
            values: Box::new([]),
            parent: None,
        }
    }

    /// Get a captured value by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<Value> {
        self.values.get(index).copied()
    }

    /// Get a captured value by index, searching parent scopes.
    pub fn get_chain(&self, depth: usize, index: usize) -> Option<Value> {
        if depth == 0 {
            self.get(index)
        } else {
            self.parent.as_ref()?.get_chain(depth - 1, index)
        }
    }

    /// Set a captured value by index.
    #[inline]
    pub fn set(&mut self, index: usize, value: Value) -> bool {
        if index < self.values.len() {
            self.values[index] = value;
            true
        } else {
            false
        }
    }

    /// Get the number of captured values.
    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get parent environment.
    #[inline]
    pub fn parent(&self) -> Option<&Arc<ClosureEnv>> {
        self.parent.as_ref()
    }
}

// =============================================================================
// Function Object
// =============================================================================

/// Python function object.
///
/// Represents a compiled function with its code, defaults, and closure.
#[repr(C)]
pub struct FunctionObject {
    /// Object header.
    pub header: ObjectHeader,
    /// Compiled bytecode.
    pub code: Arc<CodeObject>,
    /// Function name.
    pub name: Arc<str>,
    /// Default argument values.
    pub defaults: Option<Box<[Value]>>,
    /// Keyword-only defaults (name -> value).
    pub kwdefaults: Option<Box<[(Arc<str>, Value)]>>,
    /// Closure environment (captured variables).
    pub closure: Option<Arc<ClosureEnv>>,
    /// Global scope reference (module globals).
    /// This is a raw pointer to avoid Arc overhead on every call.
    /// The globals must outlive the function.
    globals_ptr: *const (),
}

// Safety: FunctionObject is Send + Sync because:
// - All owned fields are Send + Sync (Arc, Box)
// - globals_ptr is only used for reading, and globals outlive functions
unsafe impl Send for FunctionObject {}
unsafe impl Sync for FunctionObject {}

impl FunctionObject {
    /// Create a new function object.
    pub fn new(
        code: Arc<CodeObject>,
        name: Arc<str>,
        defaults: Option<Box<[Value]>>,
        closure: Option<Arc<ClosureEnv>>,
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::FUNCTION),
            code,
            name,
            defaults,
            kwdefaults: None,
            closure,
            globals_ptr: std::ptr::null(),
        }
    }

    /// Create a function with a specific globals pointer.
    ///
    /// # Safety
    /// The globals pointer must be valid for the lifetime of the function.
    pub unsafe fn with_globals(
        code: Arc<CodeObject>,
        name: Arc<str>,
        globals_ptr: *const (),
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::FUNCTION),
            code,
            name,
            defaults: None,
            kwdefaults: None,
            closure: None,
            globals_ptr,
        }
    }

    /// Get the number of positional parameters.
    #[inline]
    pub fn arg_count(&self) -> u16 {
        self.code.arg_count
    }

    /// Get the number of keyword-only parameters.
    #[inline]
    pub fn kwonly_count(&self) -> u16 {
        self.code.kwonlyarg_count
    }

    /// Check if function takes *args.
    #[inline]
    pub fn has_varargs(&self) -> bool {
        self.code
            .flags
            .contains(prism_compiler::bytecode::CodeFlags::VARARGS)
    }

    /// Check if function takes **kwargs.
    #[inline]
    pub fn has_varkw(&self) -> bool {
        self.code
            .flags
            .contains(prism_compiler::bytecode::CodeFlags::VARKEYWORDS)
    }

    /// Get default value for parameter at index.
    pub fn get_default(&self, index: usize) -> Option<Value> {
        let defaults = self.defaults.as_ref()?;
        let num_required = (self.arg_count() as usize).saturating_sub(defaults.len());
        if index >= num_required {
            defaults.get(index - num_required).copied()
        } else {
            None
        }
    }

    /// Get the closure environment.
    #[inline]
    pub fn closure(&self) -> Option<&Arc<ClosureEnv>> {
        self.closure.as_ref()
    }

    /// Get a captured value from the closure.
    #[inline]
    pub fn get_closure_value(&self, index: usize) -> Option<Value> {
        self.closure.as_ref()?.get(index)
    }
}

impl PyObject for FunctionObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}

// =============================================================================
// Bound Method Object
// =============================================================================

/// Bound method - a function bound to an instance.
#[repr(C)]
pub struct BoundMethodObject {
    /// Object header.
    pub header: ObjectHeader,
    /// The underlying function.
    pub func: Arc<FunctionObject>,
    /// The bound instance (self).
    pub instance: Value,
}

impl BoundMethodObject {
    /// Create a new bound method.
    pub fn new(func: Arc<FunctionObject>, instance: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::METHOD),
            func,
            instance,
        }
    }
}

impl PyObject for BoundMethodObject {
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

    fn make_test_code() -> Arc<CodeObject> {
        let mut code = CodeObject::new("test", "test.py");
        code.arg_count = 2;
        code.register_count = 4;
        Arc::new(code)
    }

    #[test]
    fn test_function_creation() {
        let code = make_test_code();
        let func = FunctionObject::new(code, "my_func".into(), None, None);
        assert_eq!(func.arg_count(), 2);
        assert_eq!(&*func.name, "my_func");
    }

    #[test]
    fn test_closure_env() {
        let values: Box<[Value]> = vec![Value::int(42).unwrap(), Value::int(100).unwrap()].into();
        let env = ClosureEnv::new(values, None);
        assert_eq!(env.len(), 2);
        assert_eq!(env.get(0).unwrap().as_int(), Some(42));
        assert_eq!(env.get(1).unwrap().as_int(), Some(100));
    }

    #[test]
    fn test_closure_chain() {
        let outer: Box<[Value]> = vec![Value::int(1).unwrap()].into();
        let outer_env = Arc::new(ClosureEnv::new(outer, None));

        let inner: Box<[Value]> = vec![Value::int(2).unwrap()].into();
        let inner_env = ClosureEnv::new(inner, Some(outer_env));

        assert_eq!(inner_env.get_chain(0, 0).unwrap().as_int(), Some(2));
        assert_eq!(inner_env.get_chain(1, 0).unwrap().as_int(), Some(1));
    }
}
