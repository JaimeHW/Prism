//! Builtin function object type.
//!
//! Provides the `BuiltinFunctionObject` type which wraps native Rust functions
//! as callable Python objects with proper object headers for type dispatch.

use crate::VirtualMachine;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use smallvec::SmallVec;
use std::sync::Arc;

/// Type alias for builtin function pointers (mirrors mod.rs).
type BuiltinFnPtr = fn(&[Value]) -> Result<Value, super::BuiltinError>;
type VmBuiltinFnPtr = fn(&mut VirtualMachine, &[Value]) -> Result<Value, super::BuiltinError>;

#[derive(Clone, Copy)]
enum BuiltinCallable {
    Stateless(BuiltinFnPtr),
    WithVm(VmBuiltinFnPtr),
}

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
    func: BuiltinCallable,
    /// Optional bound receiver for builtin methods accessed through instances.
    bound_self: Option<Value>,
}

impl BuiltinFunctionObject {
    /// Create a new builtin function object.
    pub fn new(name: Arc<str>, func: BuiltinFnPtr) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::Stateless(func),
            bound_self: None,
        }
    }

    /// Create a new builtin function object that needs VM context when called.
    pub fn new_vm(name: Arc<str>, func: VmBuiltinFnPtr) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithVm(func),
            bound_self: None,
        }
    }

    /// Create a builtin function object already bound to a receiver.
    pub fn new_bound(name: Arc<str>, func: BuiltinFnPtr, bound_self: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::Stateless(func),
            bound_self: Some(bound_self),
        }
    }

    /// Create a VM-aware builtin function object already bound to a receiver.
    pub fn new_bound_vm(name: Arc<str>, func: VmBuiltinFnPtr, bound_self: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithVm(func),
            bound_self: Some(bound_self),
        }
    }

    /// Bind this builtin function to a receiver.
    pub fn bind(&self, bound_self: Value) -> Self {
        match self.func {
            BuiltinCallable::Stateless(func) => {
                Self::new_bound(Arc::clone(&self.name), func, bound_self)
            }
            BuiltinCallable::WithVm(func) => {
                Self::new_bound_vm(Arc::clone(&self.name), func, bound_self)
            }
        }
    }

    /// Call the builtin function with arguments.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, super::BuiltinError> {
        self.call_impl(None, args)
    }

    /// Call the builtin function with VM context.
    #[inline]
    pub fn call_with_vm(
        &self,
        vm: &mut VirtualMachine,
        args: &[Value],
    ) -> Result<Value, super::BuiltinError> {
        self.call_impl(Some(vm), args)
    }

    #[inline]
    fn call_impl(
        &self,
        vm: Option<&mut VirtualMachine>,
        args: &[Value],
    ) -> Result<Value, super::BuiltinError> {
        if let Some(bound_self) = self.bound_self {
            let mut bound_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            bound_args.push(bound_self);
            bound_args.extend_from_slice(args);
            return self.invoke(vm, &bound_args);
        }

        self.invoke(vm, args)
    }

    #[inline]
    fn invoke(
        &self,
        vm: Option<&mut VirtualMachine>,
        args: &[Value],
    ) -> Result<Value, super::BuiltinError> {
        match self.func {
            BuiltinCallable::Stateless(func) => func(args),
            BuiltinCallable::WithVm(func) => {
                let Some(vm) = vm else {
                    return Err(super::BuiltinError::TypeError(format!(
                        "builtin '{}' requires VM context",
                        self.name
                    )));
                };
                func(vm, args)
            }
        }
    }

    /// Get the function name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the bound receiver for builtin methods, if present.
    #[inline]
    pub fn bound_self(&self) -> Option<Value> {
        self.bound_self
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
    use crate::VirtualMachine;

    fn dummy_builtin(_args: &[Value]) -> Result<Value, super::super::BuiltinError> {
        Ok(Value::none())
    }

    fn dummy_vm_builtin(
        _vm: &mut VirtualMachine,
        args: &[Value],
    ) -> Result<Value, super::super::BuiltinError> {
        Ok(Value::int(args.len() as i64).unwrap())
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

    #[test]
    fn test_vm_builtin_requires_vm_context() {
        let func = BuiltinFunctionObject::new_vm(Arc::from("test.vm"), dummy_vm_builtin);
        let err = func
            .call(&[])
            .expect_err("vm builtin should reject missing VM");
        assert!(err.to_string().contains("requires VM context"));
    }

    #[test]
    fn test_vm_builtin_function_object_call_with_vm() {
        let func = BuiltinFunctionObject::new_vm(Arc::from("test.vm"), dummy_vm_builtin);
        let mut vm = VirtualMachine::new();
        let result = func
            .call_with_vm(&mut vm, &[Value::int(1).unwrap(), Value::int(2).unwrap()])
            .expect("vm-aware builtin should execute");
        assert_eq!(result.as_int(), Some(2));
    }

    fn capture_args(args: &[Value]) -> Result<Value, super::super::BuiltinError> {
        Ok(Value::int(args.len() as i64).unwrap())
    }

    #[test]
    fn test_builtin_function_object_bind_prepends_receiver() {
        let func = BuiltinFunctionObject::new(Arc::from("capture"), capture_args);
        let bound = func.bind(Value::int(10).unwrap());
        let result = bound.call(&[Value::int(20).unwrap(), Value::int(30).unwrap()]);
        assert_eq!(result.unwrap().as_int(), Some(3));
        assert_eq!(bound.header.type_id, TypeId::BUILTIN_FUNCTION);
    }

    #[test]
    fn test_vm_builtin_bind_prepends_receiver() {
        let func = BuiltinFunctionObject::new_vm(Arc::from("capture.vm"), dummy_vm_builtin);
        let bound = func.bind(Value::int(10).unwrap());
        let mut vm = VirtualMachine::new();
        let result = bound
            .call_with_vm(&mut vm, &[Value::int(20).unwrap(), Value::int(30).unwrap()])
            .expect("vm-aware bound builtin should execute");
        assert_eq!(result.as_int(), Some(3));
    }
}
