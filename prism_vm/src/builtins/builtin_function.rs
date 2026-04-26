//! Builtin function object type.
//!
//! Provides the `BuiltinFunctionObject` type which wraps native Rust functions
//! as callable Python objects with proper object headers for type dispatch.

use crate::VirtualMachine;
use prism_core::Value;
use prism_gc::trace::{Trace, Tracer};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use smallvec::SmallVec;
use std::sync::Arc;

/// Type alias for builtin function pointers (mirrors mod.rs).
type BuiltinFnPtr = fn(&[Value]) -> Result<Value, super::BuiltinError>;
type VmBuiltinFnPtr = fn(&mut VirtualMachine, &[Value]) -> Result<Value, super::BuiltinError>;
type BuiltinKwFnPtr = fn(&[Value], &[(&str, Value)]) -> Result<Value, super::BuiltinError>;
type VmBuiltinKwFnPtr =
    fn(&mut VirtualMachine, &[Value], &[(&str, Value)]) -> Result<Value, super::BuiltinError>;

#[derive(Clone, Copy)]
enum BuiltinCallable {
    Stateless(BuiltinFnPtr),
    WithVm(VmBuiltinFnPtr),
    WithKeywords(BuiltinKwFnPtr),
    WithVmAndKeywords(VmBuiltinKwFnPtr),
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

    /// Create a builtin function object that accepts keyword arguments.
    pub fn new_kw(name: Arc<str>, func: BuiltinKwFnPtr) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithKeywords(func),
            bound_self: None,
        }
    }

    /// Create a VM-aware builtin function object that accepts keyword arguments.
    pub fn new_vm_kw(name: Arc<str>, func: VmBuiltinKwFnPtr) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithVmAndKeywords(func),
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

    /// Create a keyword-aware builtin function object already bound to a receiver.
    pub fn new_bound_kw(name: Arc<str>, func: BuiltinKwFnPtr, bound_self: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithKeywords(func),
            bound_self: Some(bound_self),
        }
    }

    /// Create a VM-aware keyword-aware builtin function object already bound to a receiver.
    pub fn new_bound_vm_kw(name: Arc<str>, func: VmBuiltinKwFnPtr, bound_self: Value) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::BUILTIN_FUNCTION),
            name,
            func: BuiltinCallable::WithVmAndKeywords(func),
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
            BuiltinCallable::WithKeywords(func) => {
                Self::new_bound_kw(Arc::clone(&self.name), func, bound_self)
            }
            BuiltinCallable::WithVmAndKeywords(func) => {
                Self::new_bound_vm_kw(Arc::clone(&self.name), func, bound_self)
            }
        }
    }

    /// Call the builtin function with arguments.
    #[inline]
    pub fn call(&self, args: &[Value]) -> Result<Value, super::BuiltinError> {
        self.call_impl(None, args, &[])
    }

    /// Call the builtin function with VM context.
    #[inline]
    pub fn call_with_vm(
        &self,
        vm: &mut VirtualMachine,
        args: &[Value],
    ) -> Result<Value, super::BuiltinError> {
        self.call_impl(Some(vm), args, &[])
    }

    /// Call the builtin function with keyword arguments.
    #[inline]
    pub fn call_with_keywords(
        &self,
        args: &[Value],
        keywords: &[(&str, Value)],
    ) -> Result<Value, super::BuiltinError> {
        self.call_impl(None, args, keywords)
    }

    /// Call the builtin function with VM context and keyword arguments.
    #[inline]
    pub fn call_with_vm_and_keywords(
        &self,
        vm: &mut VirtualMachine,
        args: &[Value],
        keywords: &[(&str, Value)],
    ) -> Result<Value, super::BuiltinError> {
        self.call_impl(Some(vm), args, keywords)
    }

    /// Return whether this builtin can accept keyword arguments directly.
    #[inline]
    pub fn accepts_keywords(&self) -> bool {
        matches!(
            self.func,
            BuiltinCallable::WithKeywords(_) | BuiltinCallable::WithVmAndKeywords(_)
        )
    }

    #[inline]
    fn call_impl(
        &self,
        vm: Option<&mut VirtualMachine>,
        args: &[Value],
        keywords: &[(&str, Value)],
    ) -> Result<Value, super::BuiltinError> {
        if let Some(bound_self) = self.bound_self {
            let mut bound_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            bound_args.push(bound_self);
            bound_args.extend_from_slice(args);
            return self.invoke(vm, &bound_args, keywords);
        }

        self.invoke(vm, args, keywords)
    }

    #[inline]
    fn invoke(
        &self,
        vm: Option<&mut VirtualMachine>,
        args: &[Value],
        keywords: &[(&str, Value)],
    ) -> Result<Value, super::BuiltinError> {
        match self.func {
            BuiltinCallable::Stateless(func) => {
                if keywords.is_empty() {
                    func(args)
                } else {
                    Err(super::BuiltinError::TypeError(format!(
                        "builtin '{}' does not accept keyword arguments",
                        self.name
                    )))
                }
            }
            BuiltinCallable::WithVm(func) => {
                if !keywords.is_empty() {
                    return Err(super::BuiltinError::TypeError(format!(
                        "builtin '{}' does not accept keyword arguments",
                        self.name
                    )));
                }
                let Some(vm) = vm else {
                    return Err(super::BuiltinError::TypeError(format!(
                        "builtin '{}' requires VM context",
                        self.name
                    )));
                };
                func(vm, args)
            }
            BuiltinCallable::WithKeywords(func) => func(args, keywords),
            BuiltinCallable::WithVmAndKeywords(func) => {
                let Some(vm) = vm else {
                    return Err(super::BuiltinError::TypeError(format!(
                        "builtin '{}' requires VM context",
                        self.name
                    )));
                };
                func(vm, args, keywords)
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

unsafe impl Trace for BuiltinFunctionObject {
    #[inline]
    fn trace(&self, tracer: &mut dyn Tracer) {
        if let Some(bound_self) = self.bound_self {
            tracer.trace_value(bound_self);
        }
    }

    #[inline]
    fn size_of(&self) -> usize {
        std::mem::size_of::<Self>() + self.name.len()
    }
}

#[cfg(test)]
mod tests;
