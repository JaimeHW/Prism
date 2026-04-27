//! Native subset of CPython's `copy` module.
//!
//! The module keeps the common immutable and container paths native while using
//! Python's reducer protocol for extensible user-defined copying. That keeps
//! the steady-state fast path compact without turning compatibility behavior
//! into a source-level dependency.

use super::{_weakref, Module, ModuleError, ModuleResult, copyreg, weakref};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class,
    exception_type_value_for_id, runtime_error_to_builtin_error, value_type_object,
};
use crate::error::RuntimeErrorKind;
use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
use crate::ops::dict_access::{dict_get_item, dict_set_item};
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id, get_attribute_value,
    list_storage_mut_from_ptr, list_storage_ref_from_ptr, set_attribute_value,
    set_storage_mut_from_ptr, set_storage_ref_from_ptr, tuple_storage_ref_from_ptr,
};
use crate::ops::set_access::set_add_item;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::descriptor::BoundMethod;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::global_class;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

static COPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copy.copy"), copy_value));
static DEEPCOPY_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("copy.deepcopy"), deepcopy_value));

/// Native `copy` module descriptor.
#[derive(Debug, Clone)]
pub struct CopyModule {
    attrs: Vec<Arc<str>>,
}

impl CopyModule {
    /// Create a new `copy` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("Error"),
                Arc::from("copy"),
                Arc::from("deepcopy"),
                Arc::from("error"),
            ],
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
            "Error" | "error" => Ok(copy_error_type()),
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

#[inline]
fn copy_error_type() -> Value {
    exception_type_value_for_id(ExceptionTypeId::Exception as u16)
        .expect("Exception type is registered")
}

fn copy_value(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    if is_copy_atomic(args[0]) {
        return Ok(args[0]);
    }
    if let Some(copied) = shallow_copy_builtin(args[0]) {
        return Ok(copied);
    }
    if let Some(copied) = call_copy_protocol(vm, args[0])? {
        return Ok(copied);
    }
    if let Some(copied) = reduce_copy(vm, args[0])? {
        return Ok(copied);
    }
    Err(copy_error(args[0], "un(shallow)copyable"))
}

fn deepcopy_value(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "deepcopy() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }
    let memo_value = if args.len() == 2 && !args[1].is_none() {
        require_memo_dict(args[1])?;
        args[1]
    } else {
        crate::alloc_managed_value(DictObject::new())
    };
    let mut memo = DeepcopyMemo::new(memo_value);
    deepcopy_inner(vm, args[0], &mut memo)
}

fn deepcopy_inner(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Value, BuiltinError> {
    if is_deepcopy_atomic(value) {
        return Ok(value);
    }

    let memo_key = identity_key(value);
    if let Some(copied) = memo.lookup(vm, memo_key)? {
        return Ok(copied);
    }

    let copied = if let Some(copied) = deepcopy_builtin(vm, value, memo)? {
        copied
    } else if let Some(copied) = call_deepcopy_protocol(vm, value, memo.memo_value())? {
        copied
    } else if let Some(copied) = reduce_deepcopy(vm, value, memo)? {
        copied
    } else {
        return Err(copy_error(value, "un(deep)copyable"));
    };

    if !values_identical(copied, value) {
        memo.store(vm, memo_key, copied)?;
        memo.keep_alive(vm, value)?;
    }
    Ok(copied)
}

struct DeepcopyMemo {
    dict_value: Value,
    local: HashMap<usize, Value>,
}

impl DeepcopyMemo {
    fn new(dict_value: Value) -> Self {
        Self {
            dict_value,
            local: HashMap::new(),
        }
    }

    #[inline]
    fn memo_value(&self) -> Value {
        self.dict_value
    }

    fn lookup(
        &mut self,
        vm: &mut VirtualMachine,
        key: usize,
    ) -> Result<Option<Value>, BuiltinError> {
        if let Some(copied) = self.local.get(&key).copied() {
            return Ok(Some(copied));
        }
        let Some(value) = dict_get_item(vm, memo_dict_ref(self.dict_value)?, memo_key_value(key)?)
            .map_err(runtime_error_to_builtin_error)?
        else {
            return Ok(None);
        };
        self.local.insert(key, value);
        Ok(Some(value))
    }

    fn store(
        &mut self,
        vm: &mut VirtualMachine,
        key: usize,
        value: Value,
    ) -> Result<(), BuiltinError> {
        self.local.insert(key, value);
        dict_set_item(
            vm,
            memo_dict_mut(self.dict_value)?,
            memo_key_value(key)?,
            value,
        )
        .map(|_| ())
        .map_err(runtime_error_to_builtin_error)
    }

    fn keep_alive(&mut self, vm: &mut VirtualMachine, value: Value) -> Result<(), BuiltinError> {
        let key = identity_key(self.dict_value);
        let key_value = memo_key_value(key)?;
        let keepalive = dict_get_item(vm, memo_dict_ref(self.dict_value)?, key_value)
            .map_err(runtime_error_to_builtin_error)?;
        if let Some(keepalive) = keepalive {
            let keepalive_ptr = keepalive.as_object_ptr().ok_or_else(|| {
                BuiltinError::TypeError("deepcopy memo keep-alive entry must be a list".to_string())
            })?;
            let list = list_storage_mut_from_ptr(keepalive_ptr).ok_or_else(|| {
                BuiltinError::TypeError("deepcopy memo keep-alive entry must be a list".to_string())
            })?;
            list.push(value);
            return Ok(());
        }

        let list = crate::alloc_managed_value(ListObject::from_slice(&[value]));
        dict_set_item(vm, memo_dict_mut(self.dict_value)?, key_value, list)
            .map(|_| ())
            .map_err(runtime_error_to_builtin_error)
    }
}

fn require_memo_dict(value: Value) -> Result<(), BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("deepcopy memo must be a dict".to_string()))?;
    if dict_storage_ref_from_ptr(ptr).is_some() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(
            "deepcopy memo must be a dict".to_string(),
        ))
    }
}

fn memo_dict_ref(value: Value) -> Result<&'static DictObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("deepcopy memo must be a dict".to_string()))?;
    dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError("deepcopy memo must be a dict".to_string()))
}

fn memo_dict_mut(value: Value) -> Result<&'static mut DictObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("deepcopy memo must be a dict".to_string()))?;
    dict_storage_mut_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError("deepcopy memo must be a dict".to_string()))
}

fn memo_key_value(key: usize) -> Result<Value, BuiltinError> {
    Value::int(key as i64).ok_or_else(|| BuiltinError::OverflowError("id overflow".to_string()))
}

#[inline]
fn identity_key(value: Value) -> usize {
    if let Some(ptr) = value.as_object_ptr() {
        return ptr as usize;
    }
    if let Some(ptr) = value.as_string_object_ptr() {
        return ptr as usize;
    }
    value.raw_bits() as usize
}

#[inline]
fn values_identical(left: Value, right: Value) -> bool {
    left.raw_bits() == right.raw_bits()
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

fn call_deepcopy_protocol(
    vm: &mut VirtualMachine,
    value: Value,
    memo: Value,
) -> Result<Option<Value>, BuiltinError> {
    match get_attribute_value(vm, value, &intern("__deepcopy__")) {
        Ok(method) => invoke_callable_value(vm, method, &[memo])
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
        TypeId::SET => {
            let source = unsafe { &*(ptr as *const SetObject) };
            Some(crate::alloc_managed_value(source.clone()))
        }
        TypeId::BYTEARRAY => {
            let source = unsafe { &*(ptr as *const BytesObject) };
            Some(crate::alloc_managed_value(source.clone()))
        }
        _ => None,
    }
}

fn is_copy_atomic(value: Value) -> bool {
    if value.is_none()
        || value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_as_string_ref(value).is_some()
        || _weakref::is_reference_value(value)
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    matches!(
        extract_type_id(ptr),
        TypeId::INT
            | TypeId::BYTES
            | TypeId::CODE
            | TypeId::COMPLEX
            | TypeId::ELLIPSIS
            | TypeId::EXCEPTION_TYPE
            | TypeId::FROZENSET
            | TypeId::FUNCTION
            | TypeId::BUILTIN_FUNCTION
            | TypeId::NOT_IMPLEMENTED
            | TypeId::PROPERTY
            | TypeId::RANGE
            | TypeId::SLICE
            | TypeId::TYPE
            | TypeId::TUPLE
    )
}

fn is_deepcopy_atomic(value: Value) -> bool {
    if value.is_none()
        || value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
        || value_as_string_ref(value).is_some()
        || _weakref::is_reference_value(value)
    {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    matches!(
        extract_type_id(ptr),
        TypeId::INT
            | TypeId::BYTES
            | TypeId::CODE
            | TypeId::COMPLEX
            | TypeId::ELLIPSIS
            | TypeId::EXCEPTION_TYPE
            | TypeId::FUNCTION
            | TypeId::BUILTIN_FUNCTION
            | TypeId::NOT_IMPLEMENTED
            | TypeId::PROPERTY
            | TypeId::RANGE
            | TypeId::TYPE
    )
}

fn deepcopy_builtin(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Option<Value>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };

    match extract_type_id(ptr) {
        TypeId::LIST => {
            let source = unsafe { &*(ptr as *const ListObject) };
            let items = source.as_slice().to_vec();
            let copied = crate::alloc_managed_value(ListObject::with_capacity(items.len()));
            memo.store(vm, identity_key(value), copied)?;
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new list values are object pointers");
            for item in items {
                let item = deepcopy_inner(vm, item, memo)?;
                let target = unsafe { &mut *(copied_ptr as *mut ListObject) };
                target.push(item);
            }
            Ok(Some(copied))
        }
        TypeId::TUPLE => deepcopy_tuple(vm, value, ptr, memo).map(Some),
        TypeId::DICT => {
            let source = unsafe { &*(ptr as *const DictObject) };
            let entries = source.iter().collect::<Vec<(Value, Value)>>();
            let copied = crate::alloc_managed_value(DictObject::with_capacity(entries.len()));
            memo.store(vm, identity_key(value), copied)?;
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new dict values are object pointers");
            for (key, item) in entries {
                let key = deepcopy_inner(vm, key, memo)?;
                let item = deepcopy_inner(vm, item, memo)?;
                let target = unsafe { &mut *(copied_ptr as *mut DictObject) };
                dict_set_item(vm, target, key, item).map_err(runtime_error_to_builtin_error)?;
            }
            Ok(Some(copied))
        }
        TypeId::SET => {
            let source = unsafe { &*(ptr as *const SetObject) };
            let items = source.iter().collect::<Vec<_>>();
            let copied = crate::alloc_managed_value(SetObject::new());
            memo.store(vm, identity_key(value), copied)?;
            let copied_ptr = copied
                .as_object_ptr()
                .expect("new set values are object pointers");
            for item in items {
                let item = deepcopy_inner(vm, item, memo)?;
                let target = unsafe { &mut *(copied_ptr as *mut SetObject) };
                set_add_item(vm, target, item).map_err(runtime_error_to_builtin_error)?;
            }
            Ok(Some(copied))
        }
        TypeId::BYTEARRAY => {
            let source = unsafe { &*(ptr as *const BytesObject) };
            Ok(Some(crate::alloc_managed_value(source.clone())))
        }
        TypeId::METHOD => {
            let method = unsafe { &*(ptr as *const BoundMethod) };
            let instance = deepcopy_inner(vm, method.instance(), memo)?;
            Ok(Some(crate::alloc_managed_value(BoundMethod::new(
                method.function(),
                instance,
            ))))
        }
        TypeId::SLICE => {
            let source = unsafe { &*(ptr as *const SliceObject) };
            let copied = crate::alloc_managed_value(SliceObject::new(
                deepcopy_inner(vm, source.start_value(), memo)?,
                deepcopy_inner(vm, source.stop_value(), memo)?,
                deepcopy_inner(vm, source.step_value(), memo)?,
            ));
            Ok(Some(copied))
        }
        _ => Ok(None),
    }
}

fn deepcopy_tuple(
    vm: &mut VirtualMachine,
    value: Value,
    ptr: *const (),
    memo: &mut DeepcopyMemo,
) -> Result<Value, BuiltinError> {
    let source = unsafe { &*(ptr as *const TupleObject) };
    if source.is_empty() {
        return Ok(value);
    }

    let mut changed = false;
    let mut items = Vec::with_capacity(source.len());
    for item in source.as_slice().iter().copied() {
        let copied = deepcopy_inner(vm, item, memo)?;
        changed |= !values_identical(copied, item);
        items.push(copied);
    }

    if let Some(copied) = memo.lookup(vm, identity_key(value))? {
        return Ok(copied);
    }
    if !changed {
        return Ok(value);
    }
    Ok(crate::alloc_managed_value(TupleObject::from_vec(items)))
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

fn reduce_copy(vm: &mut VirtualMachine, value: Value) -> Result<Option<Value>, BuiltinError> {
    let type_value = value_type_object(value);
    if let Some(reducer) = copyreg::dispatch_reducer(vm, type_value)? {
        let reduction =
            invoke_callable_value(vm, reducer, &[value]).map_err(runtime_error_to_builtin_error)?;
        return reconstruct_reduce(vm, value, reduction).map(Some);
    }

    if let Some(reduction) = call_reduce_ex(vm, value)? {
        return reconstruct_reduce(vm, value, reduction).map(Some);
    }
    if let Some(reduction) = call_reduce(vm, value)? {
        return reconstruct_reduce(vm, value, reduction).map(Some);
    }
    if let Some(copied) = copy_from_newargs(vm, value)? {
        return Ok(Some(copied));
    }
    if let Some(copied) = copy_native_backed_user(vm, value)? {
        return Ok(Some(copied));
    }
    if let Some(copied) = copy_dict_backed_user(vm, value)? {
        return Ok(Some(copied));
    }
    Ok(None)
}

fn reduce_deepcopy(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Option<Value>, BuiltinError> {
    let type_value = value_type_object(value);
    if let Some(reducer) = copyreg::dispatch_reducer(vm, type_value)? {
        let reduction =
            invoke_callable_value(vm, reducer, &[value]).map_err(runtime_error_to_builtin_error)?;
        return reconstruct_reduce_deep(vm, value, reduction, memo).map(Some);
    }

    if let Some(reduction) = call_reduce_ex(vm, value)? {
        return reconstruct_reduce_deep(vm, value, reduction, memo).map(Some);
    }
    if let Some(reduction) = call_reduce(vm, value)? {
        return reconstruct_reduce_deep(vm, value, reduction, memo).map(Some);
    }
    if let Some(copied) = deepcopy_from_newargs(vm, value, memo)? {
        return Ok(Some(copied));
    }
    if let Some(copied) = deepcopy_native_backed_user(vm, value, memo)? {
        return Ok(Some(copied));
    }
    if let Some(copied) = deepcopy_plain_user(vm, value, memo)? {
        return Ok(Some(copied));
    }
    Ok(None)
}

fn call_reduce_ex(vm: &mut VirtualMachine, value: Value) -> Result<Option<Value>, BuiltinError> {
    let method = match get_attribute_value(vm, value, &intern("__reduce_ex__")) {
        Ok(method) => method,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };
    if is_builtin_function_named(method, "object.__reduce_ex__") {
        return Ok(None);
    }
    let protocol = Value::int(4).expect("copy protocol version fits tagged int");
    invoke_callable_value(vm, method, &[protocol])
        .map(Some)
        .map_err(runtime_error_to_builtin_error)
}

fn call_reduce(vm: &mut VirtualMachine, value: Value) -> Result<Option<Value>, BuiltinError> {
    let method = match get_attribute_value(vm, value, &intern("__reduce__")) {
        Ok(method) => method,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };
    invoke_callable_value(vm, method, &[])
        .map(Some)
        .map_err(runtime_error_to_builtin_error)
}

fn copy_from_newargs(vm: &mut VirtualMachine, value: Value) -> Result<Option<Value>, BuiltinError> {
    let class_value = value_type_object(value);
    if let Some(newargs_ex) = call_no_arg_method(vm, value, "__getnewargs_ex__")? {
        let pair = value_as_tuple_ref(newargs_ex).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs_ex__ must return a tuple".to_string())
        })?;
        if pair.len() != 2 {
            return Err(BuiltinError::TypeError(
                "__getnewargs_ex__ must return a 2-tuple".to_string(),
            ));
        }
        let args = value_as_tuple_ref(pair.get(0).expect("2-tuple has args")).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs_ex__ argument 1 must be a tuple".to_string())
        })?;
        let kwargs = keyword_entries_from_dict(pair.get(1).expect("2-tuple has kwargs"))?;
        let keyword_refs = kwargs
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect::<Vec<_>>();
        let copied = construct_with_new(vm, class_value, args.as_slice(), &keyword_refs)?;
        copy_user_attribute_state(vm, value, copied)?;
        return Ok(Some(copied));
    }

    if let Some(newargs) = call_no_arg_method(vm, value, "__getnewargs__")? {
        let args = value_as_tuple_ref(newargs).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs__ must return a tuple".to_string())
        })?;
        let copied = construct_with_new(vm, class_value, args.as_slice(), &[])?;
        copy_user_attribute_state(vm, value, copied)?;
        return Ok(Some(copied));
    }

    Ok(None)
}

fn deepcopy_from_newargs(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Option<Value>, BuiltinError> {
    let class_value = value_type_object(value);
    if let Some(newargs_ex) = call_no_arg_method(vm, value, "__getnewargs_ex__")? {
        let pair = value_as_tuple_ref(newargs_ex).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs_ex__ must return a tuple".to_string())
        })?;
        if pair.len() != 2 {
            return Err(BuiltinError::TypeError(
                "__getnewargs_ex__ must return a 2-tuple".to_string(),
            ));
        }
        let args = value_as_tuple_ref(pair.get(0).expect("2-tuple has args")).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs_ex__ argument 1 must be a tuple".to_string())
        })?;
        let copied_args = deepcopy_value_slice(vm, args.as_slice(), memo)?;
        let kwargs = keyword_entries_from_dict(pair.get(1).expect("2-tuple has kwargs"))?;
        let copied_kwargs = deepcopy_keyword_entries(vm, &kwargs, memo)?;
        let keyword_refs = copied_kwargs
            .iter()
            .map(|(name, value)| (name.as_str(), *value))
            .collect::<Vec<_>>();
        let copied = construct_with_new(vm, class_value, &copied_args, &keyword_refs)?;
        memo.store(vm, identity_key(value), copied)?;
        deepcopy_user_attribute_state(vm, value, copied, memo)?;
        return Ok(Some(copied));
    }

    if let Some(newargs) = call_no_arg_method(vm, value, "__getnewargs__")? {
        let args = value_as_tuple_ref(newargs).ok_or_else(|| {
            BuiltinError::TypeError("__getnewargs__ must return a tuple".to_string())
        })?;
        let copied_args = deepcopy_value_slice(vm, args.as_slice(), memo)?;
        let copied = construct_with_new(vm, class_value, &copied_args, &[])?;
        memo.store(vm, identity_key(value), copied)?;
        deepcopy_user_attribute_state(vm, value, copied, memo)?;
        return Ok(Some(copied));
    }

    Ok(None)
}

fn deepcopy_value_slice(
    vm: &mut VirtualMachine,
    values: &[Value],
    memo: &mut DeepcopyMemo,
) -> Result<Vec<Value>, BuiltinError> {
    let mut copied = Vec::with_capacity(values.len());
    for value in values {
        copied.push(deepcopy_inner(vm, *value, memo)?);
    }
    Ok(copied)
}

fn deepcopy_keyword_entries(
    vm: &mut VirtualMachine,
    values: &[(String, Value)],
    memo: &mut DeepcopyMemo,
) -> Result<Vec<(String, Value)>, BuiltinError> {
    let mut copied = Vec::with_capacity(values.len());
    for (name, value) in values {
        copied.push((name.clone(), deepcopy_inner(vm, *value, memo)?));
    }
    Ok(copied)
}

fn call_no_arg_method(
    vm: &mut VirtualMachine,
    value: Value,
    name: &'static str,
) -> Result<Option<Value>, BuiltinError> {
    match get_attribute_value(vm, value, &intern(name)) {
        Ok(method) => invoke_callable_value(vm, method, &[])
            .map(Some)
            .map_err(runtime_error_to_builtin_error),
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(None),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn construct_with_new(
    vm: &mut VirtualMachine,
    class_value: Value,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let new_method = get_attribute_value(vm, class_value, &intern("__new__"))
        .map_err(runtime_error_to_builtin_error)?;
    let mut call_args = Vec::with_capacity(args.len() + 1);
    call_args.push(class_value);
    call_args.extend_from_slice(args);
    if keywords.is_empty() {
        invoke_callable_value(vm, new_method, &call_args).map_err(runtime_error_to_builtin_error)
    } else {
        invoke_callable_value_with_keywords(vm, new_method, &call_args, keywords)
            .map_err(runtime_error_to_builtin_error)
    }
}

fn keyword_entries_from_dict(value: Value) -> Result<Vec<(String, Value)>, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("__getnewargs_ex__ argument 2 must be a dict".to_string())
    })?;
    let dict = dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError("__getnewargs_ex__ argument 2 must be a dict".to_string())
    })?;
    let mut keywords = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let Some(name) = value_as_string_ref(key) else {
            return Err(BuiltinError::TypeError(
                "__getnewargs_ex__ keyword names must be strings".to_string(),
            ));
        };
        keywords.push((name.as_str().to_string(), value));
    }
    Ok(keywords)
}

fn copy_native_backed_user(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    if let Some(kind) = weakref::weak_dict_kind(value) {
        return copy_weak_dict(vm, value, kind).map(Some);
    }
    let type_id = extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }
    if list_storage_ref_from_ptr(ptr).is_none()
        && tuple_storage_ref_from_ptr(ptr).is_none()
        && dict_storage_ref_from_ptr(ptr).is_none()
        && set_storage_ref_from_ptr(ptr).is_none()
    {
        return Ok(None);
    }

    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return Ok(None);
    };
    let copied = crate::alloc_managed_value(allocate_heap_instance_for_class(class.as_ref()));
    let copied_ptr = copied
        .as_object_ptr()
        .expect("new user instance values are object pointers");
    copy_user_attribute_state(vm, value, copied)?;

    if let Some(source) = list_storage_ref_from_ptr(ptr)
        && let Some(target) = list_storage_mut_from_ptr(copied_ptr)
    {
        target.clear();
        target.extend(source.as_slice().iter().copied());
    }
    if let Some(source) = tuple_storage_ref_from_ptr(ptr) {
        let target = unsafe { &mut *(copied_ptr as *mut ShapedObject) };
        target.set_tuple_backing(TupleObject::from_slice(source.as_slice()));
    }
    if let Some(source) = dict_storage_ref_from_ptr(ptr)
        && let Some(target) = dict_storage_mut_from_ptr(copied_ptr)
    {
        target.clear();
        for (key, item) in source.iter() {
            dict_set_item(vm, target, key, item).map_err(runtime_error_to_builtin_error)?;
        }
    }
    if let Some(source) = set_storage_ref_from_ptr(ptr)
        && let Some(target) = set_storage_mut_from_ptr(copied_ptr)
    {
        target.clear();
        for item in source.iter() {
            target.add(item);
        }
    }

    Ok(Some(copied))
}

fn copy_dict_backed_user(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Option<Value>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    let type_id = extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }
    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return Ok(None);
    };
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());
    if !class_uses_default_new(vm, class_value)? {
        return Ok(None);
    }

    let copied = crate::alloc_managed_value(allocate_heap_instance_for_class(class.as_ref()));
    copy_user_attribute_state(vm, value, copied)?;

    Ok(Some(copied))
}

fn deepcopy_native_backed_user(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Option<Value>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    if let Some(kind) = weakref::weak_dict_kind(value) {
        return deepcopy_weak_dict(vm, value, kind, memo).map(Some);
    }
    let type_id = extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }
    if list_storage_ref_from_ptr(ptr).is_none()
        && tuple_storage_ref_from_ptr(ptr).is_none()
        && dict_storage_ref_from_ptr(ptr).is_none()
        && set_storage_ref_from_ptr(ptr).is_none()
    {
        return Ok(None);
    }

    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return Ok(None);
    };
    let copied = crate::alloc_managed_value(allocate_heap_instance_for_class(class.as_ref()));
    let copied_ptr = copied
        .as_object_ptr()
        .expect("new user instance values are object pointers");
    memo.store(vm, identity_key(value), copied)?;

    if let Some(source) = list_storage_ref_from_ptr(ptr) {
        let items = source.as_slice().to_vec();
        for item in items {
            let item = deepcopy_inner(vm, item, memo)?;
            let target = list_storage_mut_from_ptr(copied_ptr)
                .expect("list-backed class should allocate list backing");
            target.push(item);
        }
    }
    if let Some(source) = tuple_storage_ref_from_ptr(ptr) {
        let items = deepcopy_value_slice(vm, source.as_slice(), memo)?;
        let target = unsafe { &mut *(copied_ptr as *mut ShapedObject) };
        target.set_tuple_backing(TupleObject::from_vec(items));
    }
    if let Some(source) = dict_storage_ref_from_ptr(ptr) {
        let entries = source.iter().collect::<Vec<(Value, Value)>>();
        for (key, item) in entries {
            let key = deepcopy_inner(vm, key, memo)?;
            let item = deepcopy_inner(vm, item, memo)?;
            let target = dict_storage_mut_from_ptr(copied_ptr)
                .expect("dict-backed class should allocate dict backing");
            dict_set_item(vm, target, key, item).map_err(runtime_error_to_builtin_error)?;
        }
    }
    if let Some(source) = set_storage_ref_from_ptr(ptr) {
        let items = source.iter().collect::<Vec<_>>();
        for item in items {
            let item = deepcopy_inner(vm, item, memo)?;
            let target = set_storage_mut_from_ptr(copied_ptr)
                .expect("set-backed class should allocate set backing");
            set_add_item(vm, target, item).map_err(runtime_error_to_builtin_error)?;
        }
    }

    deepcopy_user_attribute_state(vm, value, copied, memo)?;
    Ok(Some(copied))
}

fn copy_weak_dict(
    vm: &mut VirtualMachine,
    value: Value,
    kind: weakref::WeakDictKind,
) -> Result<Value, BuiltinError> {
    let source = dict_storage_for_copy(value, "weak dictionary")?;
    let entries = source.iter().collect::<Vec<_>>();
    let copied = weakref::new_weak_dict(kind);
    let copied_ptr = copied
        .as_object_ptr()
        .expect("new weak dictionary values are object pointers");
    let target = dict_storage_mut_from_ptr(copied_ptr)
        .expect("new weak dictionary values carry dict storage");
    for (key, item) in entries {
        dict_set_item(vm, target, key, item).map_err(runtime_error_to_builtin_error)?;
    }
    copy_user_attribute_state(vm, value, copied)?;
    Ok(copied)
}

fn deepcopy_weak_dict(
    vm: &mut VirtualMachine,
    value: Value,
    kind: weakref::WeakDictKind,
    memo: &mut DeepcopyMemo,
) -> Result<Value, BuiltinError> {
    let source = dict_storage_for_copy(value, "weak dictionary")?;
    let entries = source.iter().collect::<Vec<_>>();
    let copied = weakref::new_weak_dict(kind);
    memo.store(vm, identity_key(value), copied)?;
    let copied_ptr = copied
        .as_object_ptr()
        .expect("new weak dictionary values are object pointers");

    for (key, item) in entries {
        let key = match kind {
            weakref::WeakDictKind::Key => key,
            weakref::WeakDictKind::Value => deepcopy_inner(vm, key, memo)?,
        };
        let item = match kind {
            weakref::WeakDictKind::Key => deepcopy_inner(vm, item, memo)?,
            weakref::WeakDictKind::Value => item,
        };
        let target = dict_storage_mut_from_ptr(copied_ptr)
            .expect("new weak dictionary values carry dict storage");
        dict_set_item(vm, target, key, item).map_err(runtime_error_to_builtin_error)?;
    }

    deepcopy_user_attribute_state(vm, value, copied, memo)?;
    Ok(copied)
}

fn dict_storage_for_copy(
    value: Value,
    type_name: &'static str,
) -> Result<&'static DictObject, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("copy source is not a {type_name}")))?;
    dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError(format!("copy source is not a {type_name}")))
}

fn deepcopy_plain_user(
    vm: &mut VirtualMachine,
    value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Option<Value>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    let type_id = extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }
    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return Ok(None);
    };
    let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());
    if !class_uses_default_new(vm, class_value)? {
        return Ok(None);
    }

    let copied = crate::alloc_managed_value(allocate_heap_instance_for_class(class.as_ref()));
    memo.store(vm, identity_key(value), copied)?;
    deepcopy_user_attribute_state(vm, value, copied, memo)?;
    Ok(Some(copied))
}

fn copy_user_attribute_state(
    vm: &mut VirtualMachine,
    source_value: Value,
    target_value: Value,
) -> Result<(), BuiltinError> {
    if let Some(state) = instance_state_value(vm, source_value)? {
        restore_reduce_state(vm, target_value, state)?;
    }
    Ok(())
}

fn deepcopy_user_attribute_state(
    vm: &mut VirtualMachine,
    source_value: Value,
    target_value: Value,
    memo: &mut DeepcopyMemo,
) -> Result<(), BuiltinError> {
    if let Some(state) = instance_state_value(vm, source_value)? {
        let state = deepcopy_inner(vm, state, memo)?;
        restore_reduce_state(vm, target_value, state)?;
    }
    Ok(())
}

fn instance_state_value(
    vm: &mut VirtualMachine,
    source_value: Value,
) -> Result<Option<Value>, BuiltinError> {
    if let Some(state) = call_no_arg_method(vm, source_value, "__getstate__")? {
        return Ok(Some(state));
    }

    let Some(source_ptr) = source_value.as_object_ptr() else {
        return Ok(None);
    };
    let source = unsafe { &*(source_ptr as *const ShapedObject) };
    if let Some(dict_value) = source.instance_dict_value() {
        return Ok(Some(dict_value));
    }

    let property_count = source.property_count();
    if property_count == 0 {
        return Ok(None);
    }

    let mut state = DictObject::with_capacity(property_count);
    for (name, item) in source.iter_properties() {
        state.set(Value::string(name), item);
    }
    Ok(Some(crate::alloc_managed_value(state)))
}

fn class_uses_default_new(
    vm: &mut VirtualMachine,
    class_value: Value,
) -> Result<bool, BuiltinError> {
    match get_attribute_value(vm, class_value, &intern("__new__")) {
        Ok(method) => Ok(is_builtin_function_named(method, "object.__new__")),
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => Ok(true),
        Err(err) => Err(runtime_error_to_builtin_error(err)),
    }
}

fn is_builtin_function_named(value: Value, expected: &'static str) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    if extract_type_id(ptr) != TypeId::BUILTIN_FUNCTION {
        return false;
    }
    let function = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    function.name() == expected
}

fn reconstruct_reduce(
    vm: &mut VirtualMachine,
    original: Value,
    reduction: Value,
) -> Result<Value, BuiltinError> {
    if value_as_string_ref(reduction).is_some() {
        return Ok(original);
    }

    let tuple = value_as_tuple_ref(reduction).ok_or_else(|| {
        BuiltinError::TypeError("copy reducer must return a string or tuple".into())
    })?;
    if !(2..=5).contains(&tuple.len()) {
        return Err(BuiltinError::TypeError(
            "copy reducer tuple must contain 2 to 5 items".to_string(),
        ));
    }

    let callable = tuple
        .get(0)
        .ok_or_else(|| BuiltinError::TypeError("copy reducer is missing callable".into()))?;
    let args_value = tuple
        .get(1)
        .ok_or_else(|| BuiltinError::TypeError("copy reducer is missing argument tuple".into()))?;
    let args_tuple = value_as_tuple_ref(args_value).ok_or_else(|| {
        BuiltinError::TypeError("copy reducer arguments must be a tuple".to_string())
    })?;

    let result = invoke_callable_value(vm, callable, args_tuple.as_slice())
        .map_err(runtime_error_to_builtin_error)?;
    if let Some(state) = tuple.get(2)
        && !state.is_none()
    {
        restore_reduce_state(vm, result, state)?;
    }
    if let Some(listiter) = tuple.get(3)
        && !listiter.is_none()
    {
        restore_reduce_list_items(vm, result, listiter)?;
    }
    if let Some(dictiter) = tuple.get(4)
        && !dictiter.is_none()
    {
        restore_reduce_dict_items(vm, result, dictiter)?;
    }

    Ok(result)
}

fn reconstruct_reduce_deep(
    vm: &mut VirtualMachine,
    original: Value,
    reduction: Value,
    memo: &mut DeepcopyMemo,
) -> Result<Value, BuiltinError> {
    if value_as_string_ref(reduction).is_some() {
        return Ok(original);
    }

    let tuple = value_as_tuple_ref(reduction).ok_or_else(|| {
        BuiltinError::TypeError("copy reducer must return a string or tuple".into())
    })?;
    if !(2..=5).contains(&tuple.len()) {
        return Err(BuiltinError::TypeError(
            "copy reducer tuple must contain 2 to 5 items".to_string(),
        ));
    }

    let callable = tuple
        .get(0)
        .ok_or_else(|| BuiltinError::TypeError("copy reducer is missing callable".into()))?;
    let args_value = tuple
        .get(1)
        .ok_or_else(|| BuiltinError::TypeError("copy reducer is missing argument tuple".into()))?;
    let args_tuple = value_as_tuple_ref(args_value).ok_or_else(|| {
        BuiltinError::TypeError("copy reducer arguments must be a tuple".to_string())
    })?;
    let args = deepcopy_value_slice(vm, args_tuple.as_slice(), memo)?;

    let result =
        invoke_callable_value(vm, callable, &args).map_err(runtime_error_to_builtin_error)?;
    memo.store(vm, identity_key(original), result)?;
    if let Some(state) = tuple.get(2)
        && !state.is_none()
    {
        let state = deepcopy_inner(vm, state, memo)?;
        restore_reduce_state(vm, result, state)?;
    }
    if let Some(listiter) = tuple.get(3)
        && !listiter.is_none()
    {
        restore_reduce_list_items_deep(vm, result, listiter, memo)?;
    }
    if let Some(dictiter) = tuple.get(4)
        && !dictiter.is_none()
    {
        restore_reduce_dict_items_deep(vm, result, dictiter, memo)?;
    }

    Ok(result)
}

fn restore_reduce_state(
    vm: &mut VirtualMachine,
    object: Value,
    state: Value,
) -> Result<(), BuiltinError> {
    match get_attribute_value(vm, object, &intern("__setstate__")) {
        Ok(setstate) => {
            invoke_callable_value(vm, setstate, &[state])
                .map(|_| ())
                .map_err(runtime_error_to_builtin_error)?;
            return Ok(());
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {}
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    }

    let (state, slot_state) = if let Some(tuple) = value_as_tuple_ref(state) {
        if tuple.len() == 2 {
            (
                tuple.get(0).expect("2-tuple has state"),
                Some(tuple.get(1).expect("2-tuple has slot state")),
            )
        } else {
            (state, None)
        }
    } else {
        (state, None)
    };

    if !state.is_none() {
        restore_state_dict(vm, object, state)?;
    }
    if let Some(slot_state) = slot_state
        && !slot_state.is_none()
    {
        restore_slot_state(vm, object, slot_state)?;
    }
    Ok(())
}

fn restore_state_dict(
    vm: &mut VirtualMachine,
    object: Value,
    state: Value,
) -> Result<(), BuiltinError> {
    let state_ptr = state
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("copy state must be a dictionary".to_string()))?;
    let state_dict = dict_storage_ref_from_ptr(state_ptr)
        .ok_or_else(|| BuiltinError::TypeError("copy state must be a dictionary".to_string()))?;

    let object_ptr = object
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("copy state target is not an object".to_string()))?;
    if extract_type_id(object_ptr).raw() < TypeId::FIRST_USER_TYPE
        && let Some(target_dict) = dict_storage_mut_from_ptr(object_ptr)
    {
        for (key, value) in state_dict.iter() {
            dict_set_item(vm, target_dict, key, value).map_err(runtime_error_to_builtin_error)?;
        }
        return Ok(());
    }

    for (key, value) in state_dict.iter() {
        let Some(name) = value_as_string_ref(key) else {
            return Err(BuiltinError::TypeError(
                "copy state keys must be strings for attribute restore".to_string(),
            ));
        };
        set_attribute_value(vm, object, &intern(name.as_str()), value)
            .map_err(runtime_error_to_builtin_error)?;
    }
    Ok(())
}

fn restore_slot_state(
    vm: &mut VirtualMachine,
    object: Value,
    state: Value,
) -> Result<(), BuiltinError> {
    let state_ptr = state.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("copy slot state must be a dictionary".to_string())
    })?;
    let state_dict = dict_storage_ref_from_ptr(state_ptr).ok_or_else(|| {
        BuiltinError::TypeError("copy slot state must be a dictionary".to_string())
    })?;

    for (key, value) in state_dict.iter() {
        let Some(name) = value_as_string_ref(key) else {
            return Err(BuiltinError::TypeError(
                "copy slot state keys must be strings".to_string(),
            ));
        };
        set_attribute_value(vm, object, &intern(name.as_str()), value)
            .map_err(runtime_error_to_builtin_error)?;
    }
    Ok(())
}

fn restore_reduce_list_items(
    vm: &mut VirtualMachine,
    object: Value,
    iterable: Value,
) -> Result<(), BuiltinError> {
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    if let Some(ptr) = object.as_object_ptr()
        && let Some(list) = list_storage_mut_from_ptr(ptr)
    {
        loop {
            match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
                IterStep::Yielded(item) => list.push(item),
                IterStep::Exhausted => return Ok(()),
            }
        }
    }

    let append = get_attribute_value(vm, object, &intern("append"))
        .map_err(runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(item) => {
                invoke_callable_value(vm, append, &[item])
                    .map_err(runtime_error_to_builtin_error)?;
            }
            IterStep::Exhausted => return Ok(()),
        }
    }
}

fn restore_reduce_list_items_deep(
    vm: &mut VirtualMachine,
    object: Value,
    iterable: Value,
    memo: &mut DeepcopyMemo,
) -> Result<(), BuiltinError> {
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    if let Some(ptr) = object.as_object_ptr()
        && list_storage_mut_from_ptr(ptr).is_some()
    {
        loop {
            match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
                IterStep::Yielded(item) => {
                    let item = deepcopy_inner(vm, item, memo)?;
                    let list = list_storage_mut_from_ptr(ptr)
                        .expect("list storage existed before deepcopying item");
                    list.push(item);
                }
                IterStep::Exhausted => return Ok(()),
            }
        }
    }

    let append = get_attribute_value(vm, object, &intern("append"))
        .map_err(runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(item) => {
                let item = deepcopy_inner(vm, item, memo)?;
                invoke_callable_value(vm, append, &[item])
                    .map_err(runtime_error_to_builtin_error)?;
            }
            IterStep::Exhausted => return Ok(()),
        }
    }
}

fn restore_reduce_dict_items(
    vm: &mut VirtualMachine,
    object: Value,
    iterable: Value,
) -> Result<(), BuiltinError> {
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    if let Some(ptr) = object.as_object_ptr()
        && let Some(dict) = dict_storage_mut_from_ptr(ptr)
    {
        loop {
            match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
                IterStep::Yielded(pair) => {
                    let tuple = value_as_tuple_ref(pair).ok_or_else(|| {
                        BuiltinError::TypeError("dict item iterator must yield pairs".to_string())
                    })?;
                    if tuple.len() != 2 {
                        return Err(BuiltinError::ValueError(
                            "dict item iterator must yield 2-tuples".to_string(),
                        ));
                    }
                    dict_set_item(
                        vm,
                        dict,
                        tuple.get(0).expect("2-tuple has key"),
                        tuple.get(1).expect("2-tuple has value"),
                    )
                    .map_err(runtime_error_to_builtin_error)?;
                }
                IterStep::Exhausted => return Ok(()),
            }
        }
    }

    let setitem = get_attribute_value(vm, object, &intern("__setitem__"))
        .map_err(runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(pair) => {
                let tuple = value_as_tuple_ref(pair).ok_or_else(|| {
                    BuiltinError::TypeError("dict item iterator must yield pairs".to_string())
                })?;
                if tuple.len() != 2 {
                    return Err(BuiltinError::ValueError(
                        "dict item iterator must yield 2-tuples".to_string(),
                    ));
                }
                invoke_callable_value(
                    vm,
                    setitem,
                    &[
                        tuple.get(0).expect("2-tuple has key"),
                        tuple.get(1).expect("2-tuple has value"),
                    ],
                )
                .map_err(runtime_error_to_builtin_error)?;
            }
            IterStep::Exhausted => return Ok(()),
        }
    }
}

fn restore_reduce_dict_items_deep(
    vm: &mut VirtualMachine,
    object: Value,
    iterable: Value,
    memo: &mut DeepcopyMemo,
) -> Result<(), BuiltinError> {
    let iterator = ensure_iterator_value(vm, iterable).map_err(runtime_error_to_builtin_error)?;
    if let Some(ptr) = object.as_object_ptr()
        && dict_storage_mut_from_ptr(ptr).is_some()
    {
        loop {
            match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
                IterStep::Yielded(pair) => {
                    let (key, value) = reduce_dict_pair(pair)?;
                    let key = deepcopy_inner(vm, key, memo)?;
                    let value = deepcopy_inner(vm, value, memo)?;
                    let dict = dict_storage_mut_from_ptr(ptr)
                        .expect("dict storage existed before deepcopying item");
                    dict_set_item(vm, dict, key, value).map_err(runtime_error_to_builtin_error)?;
                }
                IterStep::Exhausted => return Ok(()),
            }
        }
    }

    let setitem = get_attribute_value(vm, object, &intern("__setitem__"))
        .map_err(runtime_error_to_builtin_error)?;
    loop {
        match next_step(vm, iterator).map_err(runtime_error_to_builtin_error)? {
            IterStep::Yielded(pair) => {
                let (key, value) = reduce_dict_pair(pair)?;
                let key = deepcopy_inner(vm, key, memo)?;
                let value = deepcopy_inner(vm, value, memo)?;
                invoke_callable_value(vm, setitem, &[key, value])
                    .map_err(runtime_error_to_builtin_error)?;
            }
            IterStep::Exhausted => return Ok(()),
        }
    }
}

fn reduce_dict_pair(pair: Value) -> Result<(Value, Value), BuiltinError> {
    let tuple = value_as_tuple_ref(pair).ok_or_else(|| {
        BuiltinError::TypeError("dict item iterator must yield pairs".to_string())
    })?;
    if tuple.len() != 2 {
        return Err(BuiltinError::ValueError(
            "dict item iterator must yield 2-tuples".to_string(),
        ));
    }
    Ok((
        tuple.get(0).expect("2-tuple has key"),
        tuple.get(1).expect("2-tuple has value"),
    ))
}

fn copy_error(value: Value, verb: &'static str) -> BuiltinError {
    BuiltinError::TypeError(format!("{verb} object of type '{}'", value.type_name()))
}
