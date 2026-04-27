//! Native subset of CPython's `copy` module.
//!
//! This provides the identity-preserving fast paths needed by early regression
//! tests. Rich object graph copying belongs in the full stdlib implementation;
//! immutable runtime objects can use these native identity paths directly.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    runtime_error_to_builtin_error,
};
use crate::error::RuntimeErrorKind;
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id, get_attribute_value,
};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::tuple::TupleObject;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

static COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copy.copy"), copy_value));
static DEEPCOPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("copy.deepcopy"), deepcopy_value));

/// Native `copy` module descriptor.
#[derive(Debug, Clone)]
pub struct CopyModule {
    attrs: Vec<Arc<str>>,
}

impl CopyModule {
    /// Create a new `copy` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("copy"), Arc::from("deepcopy")],
        }
    }
}

impl Default for CopyModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CopyModule {
    fn name(&self) -> &str {
        "copy"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "copy" => Ok(builtin_value(&COPY_FUNCTION)),
            "deepcopy" => Ok(builtin_value(&DEEPCOPY_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'copy' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn copy_value(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    if let Some(copied) = call_copy_protocol(vm, args[0])? {
        return Ok(copied);
    }
    if let Some(copied) = shallow_copy_builtin(args[0]) {
        return Ok(copied);
    }
    Ok(args[0])
}

fn deepcopy_value(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "deepcopy() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let mut memo = HashMap::new();
    Ok(deepcopy_inner(args[0], &mut memo))
}

fn deepcopy_inner(value: Value, memo: &mut HashMap<usize, Value>) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };
    let key = ptr as usize;
    if let Some(copied) = memo.get(&key).copied() {
        return copied;
    }

    match extract_type_id(ptr) {
        TypeId::LIST => {
            let source = unsafe { &*(ptr as *const ListObject) };
            let copied = crate::alloc_managed_value(ListObject::with_capacity(source.len()));
            memo.insert(key, copied);
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new list values are object pointers");
            let target = unsafe { &mut *(copied_ptr as *mut ListObject) };
            for item in source.as_slice().iter().copied() {
                target.push(deepcopy_inner(item, memo));
            }
            copied
        }
        TypeId::TUPLE => {
            let source = unsafe { &*(ptr as *const TupleObject) };
            let items = source
                .as_slice()
                .iter()
                .copied()
                .map(|item| deepcopy_inner(item, memo))
                .collect::<Vec<_>>();
            let copied = crate::alloc_managed_value(TupleObject::from_vec(items));
            memo.insert(key, copied);
            copied
        }
        TypeId::DICT => {
            let source = unsafe { &*(ptr as *const DictObject) };
            let copied = crate::alloc_managed_value(DictObject::with_capacity(source.len()));
            memo.insert(key, copied);
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new dict values are object pointers");
            let target = unsafe { &mut *(copied_ptr as *mut DictObject) };
            for (key, item) in source.iter() {
                target.set(deepcopy_inner(key, memo), deepcopy_inner(item, memo));
            }
            copied
        }
        TypeId::SLICE => {
            let source = unsafe { &*(ptr as *const SliceObject) };
            let copied = crate::alloc_managed_value(SliceObject::new(
                deepcopy_inner(source.start_value(), memo),
                deepcopy_inner(source.stop_value(), memo),
                deepcopy_inner(source.step_value(), memo),
            ));
            memo.insert(key, copied);
            copied
        }
        type_id
            if type_id.raw() >= TypeId::FIRST_USER_TYPE
                && dict_storage_ref_from_ptr(ptr).is_some() =>
        {
            deepcopy_dict_backed_user(ptr, type_id, key, memo).unwrap_or(value)
        }
        _ => value,
    }
}

fn call_copy_protocol(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    match get_attribute_value(vm, value, &intern("__copy__")) {
        Ok(method) => invoke_callable_value(vm, method, &[])
            .map(Some)
            .map_err(runtime_error_to_builtin_error),
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn shallow_copy_builtin(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    match extract_type_id(ptr) {
        TypeId::LIST => {
            let source = unsafe { &*(ptr as *const ListObject) };
            Some(crate::alloc_managed_value(ListObject::from_slice(
                source.as_slice(),
            )))
        }
        TypeId::DICT => {
            let source = unsafe { &*(ptr as *const DictObject) };
            Some(crate::alloc_managed_value(shallow_copy_dict(source)))
        }
        _ => None,
    }
}

fn shallow_copy_dict(source: &DictObject) -> DictObject {
    let mut copied = DictObject::with_capacity(source.len());
    for (key, value) in source.iter() {
        if let Some(hash) = source.stored_hash(key) {
            copied.set_with_hash(key, value, hash);
        } else {
            copied.set(key, value);
        }
    }
    copied
}

fn deepcopy_dict_backed_user(
    ptr: *const (),
    type_id: TypeId,
    memo_key: usize,
    memo: &mut HashMap<usize, Value>,
) -> Option<Value> {
    let class = global_class(ClassId(type_id.raw()))?;
    let source = unsafe { &*(ptr as *const ShapedObject) };
    let dict_entries = dict_storage_ref_from_ptr(ptr)?
        .iter()
        .collect::<Vec<(Value, Value)>>();

    let copied = crate::alloc_managed_value(allocate_heap_instance_for_class(class.as_ref()));
    memo.insert(memo_key, copied);
    let copied_ptr = copied.as_object_ptr()?;

    {
        let target = unsafe { &mut *(copied_ptr as *mut ShapedObject) };
        for (name, item) in source.iter_properties() {
            target.set_property(name, deepcopy_inner(item, memo), shape_registry());
        }
    }

    if let Some(target_dict) = dict_storage_mut_from_ptr(copied_ptr) {
        for (key, item) in dict_entries {
            target_dict.set(deepcopy_inner(key, memo), deepcopy_inner(item, memo));
        }
    }

    Some(copied)
}
