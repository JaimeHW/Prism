//! Function and closure objects.
//!
//! Implements Python function objects for the interpreter.

use crate::object::type_obj::TypeId;
use crate::object::{ObjectHeader, PyObject};
use crate::types::dict::DictObject;
use prism_code::CodeObject;
use prism_core::Value;
use prism_core::intern::InternedString;
use rustc_hash::FxHashMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::RwLock;

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
#[derive(Default)]
struct FunctionAttrs {
    inline: FxHashMap<InternedString, Value>,
    dict_ptr: Option<NonNull<DictObject>>,
}

impl FunctionAttrs {
    #[inline]
    fn get(&self, name: &InternedString) -> Option<Value> {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.get(Value::string(name.clone())),
            None => self.inline.get(name).copied(),
        }
    }

    #[inline]
    fn set(&mut self, name: InternedString, value: Value) {
        match self.dict_ptr {
            Some(mut ptr) => unsafe { ptr.as_mut() }.set(Value::string(name), value),
            None => {
                self.inline.insert(name, value);
            }
        }
    }

    #[inline]
    fn remove(&mut self, name: &InternedString) -> Option<Value> {
        match self.dict_ptr {
            Some(mut ptr) => unsafe { ptr.as_mut() }.remove(Value::string(name.clone())),
            None => self.inline.remove(name),
        }
    }

    #[inline]
    fn contains(&self, name: &InternedString) -> bool {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.contains_key(Value::string(name.clone())),
            None => self.inline.contains_key(name),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        match self.dict_ptr {
            Some(ptr) => unsafe { ptr.as_ref() }.len(),
            None => self.inline.len(),
        }
    }

    #[inline]
    fn dict_ptr(&self) -> Option<*mut DictObject> {
        self.dict_ptr.map(NonNull::as_ptr)
    }

    fn materialize_dict<E, F>(&mut self, alloc: F) -> Result<*mut DictObject, E>
    where
        F: FnOnce(DictObject) -> Result<*mut DictObject, E>,
    {
        if let Some(ptr) = self.dict_ptr() {
            return Ok(ptr);
        }

        let mut dict = DictObject::with_capacity(self.inline.len());
        for (name, value) in &self.inline {
            dict.set(Value::string(name.clone()), *value);
        }

        let ptr = alloc(dict)?;
        let ptr = NonNull::new(ptr).expect("function attribute dict pointer must not be null");
        self.inline.clear();
        self.dict_ptr = Some(ptr);
        Ok(ptr.as_ptr())
    }

    fn for_each_value<F>(&self, mut f: F)
    where
        F: FnMut(Value),
    {
        match self.dict_ptr {
            Some(ptr) => {
                for value in unsafe { ptr.as_ref() }.values() {
                    f(value);
                }
            }
            None => {
                for value in self.inline.values() {
                    f(*value);
                }
            }
        }
    }
}

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
    /// Lazily populated custom function attributes and optional live __dict__.
    attrs: RwLock<FunctionAttrs>,
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
            attrs: RwLock::new(FunctionAttrs::default()),
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
            attrs: RwLock::new(FunctionAttrs::default()),
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
        self.code.flags.contains(prism_code::CodeFlags::VARARGS)
    }

    /// Check if function takes **kwargs.
    #[inline]
    pub fn has_varkw(&self) -> bool {
        self.code.flags.contains(prism_code::CodeFlags::VARKEYWORDS)
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

    /// Get the raw module globals pointer captured when the function was defined.
    #[inline]
    pub fn globals_ptr(&self) -> *const () {
        self.globals_ptr
    }

    /// Get a custom function attribute.
    #[inline]
    pub fn get_attr(&self, name: &InternedString) -> Option<Value> {
        self.attrs.read().unwrap().get(name)
    }

    /// Set a custom function attribute.
    #[inline]
    pub fn set_attr(&self, name: InternedString, value: Value) {
        self.attrs.write().unwrap().set(name, value);
    }

    /// Delete a custom function attribute.
    #[inline]
    pub fn del_attr(&self, name: &InternedString) -> Option<Value> {
        self.attrs.write().unwrap().remove(name)
    }

    /// Check whether a custom function attribute exists.
    #[inline]
    pub fn has_attr(&self, name: &InternedString) -> bool {
        self.attrs.read().unwrap().contains(name)
    }

    /// Return the live function attribute dictionary if it has been materialized.
    #[inline]
    pub fn attr_dict_ptr(&self) -> Option<*mut DictObject> {
        self.attrs.read().unwrap().dict_ptr()
    }

    /// Materialize and return the live function attribute dictionary.
    pub fn ensure_attr_dict<E, F>(&self, alloc: F) -> Result<*mut DictObject, E>
    where
        F: FnOnce(DictObject) -> Result<*mut DictObject, E>,
    {
        self.attrs.write().unwrap().materialize_dict(alloc)
    }

    /// Visit each custom function attribute value.
    pub fn for_each_attr_value<F>(&self, mut f: F)
    where
        F: FnMut(Value),
    {
        self.attrs.read().unwrap().for_each_value(|value| f(value));
    }

    /// Number of custom function attributes.
    #[inline]
    pub fn attr_len(&self) -> usize {
        self.attrs.read().unwrap().len()
    }

    /// Update the function's module globals pointer.
    ///
    /// # Safety
    /// The pointer must remain valid for the lifetime of the function object.
    #[inline]
    pub unsafe fn set_globals_ptr(&mut self, globals_ptr: *const ()) {
        self.globals_ptr = globals_ptr;
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
    use prism_core::intern::intern;

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

    #[test]
    fn test_function_attr_dict_materialization_preserves_existing_attrs() {
        let func = FunctionObject::new(make_test_code(), "dict_func".into(), None, None);
        func.set_attr(intern("copied"), Value::int(7).unwrap());

        let dict_ptr = func
            .ensure_attr_dict(|dict| Ok::<*mut DictObject, ()>(Box::into_raw(Box::new(dict))))
            .expect("dict allocation should succeed");
        let dict = unsafe { &*dict_ptr };

        assert_eq!(
            dict.get(Value::string(intern("copied"))).unwrap().as_int(),
            Some(7)
        );
        assert_eq!(func.get_attr(&intern("copied")).unwrap().as_int(), Some(7));
    }

    #[test]
    fn test_function_attr_reads_follow_materialized_dict_mutations() {
        let func = FunctionObject::new(make_test_code(), "dict_func".into(), None, None);
        let dict_ptr = func
            .ensure_attr_dict(|dict| Ok::<*mut DictObject, ()>(Box::into_raw(Box::new(dict))))
            .expect("dict allocation should succeed");

        unsafe { &mut *dict_ptr }.set(Value::string(intern("dynamic")), Value::int(11).unwrap());

        assert_eq!(
            func.get_attr(&intern("dynamic")).unwrap().as_int(),
            Some(11)
        );
        assert!(func.has_attr(&intern("dynamic")));
    }
}
