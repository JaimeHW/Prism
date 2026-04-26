//! Type constructor builtins (int, float, str, bool, list, dict, etc.).

use super::BuiltinError;
use super::{
    builtin_bound_type_attribute_value_static, builtin_hash, builtin_instance_has_attribute,
    builtin_type_has_attribute, heap_type_attribute_value_static, heap_type_has_attribute,
};
use crate::VirtualMachine;
use crate::error::RuntimeErrorKind;
use crate::import::ModuleObject;
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::stdlib::collections::deque::{builtin_deque, builtin_deque_kw, builtin_deque_with_vm};
use num_bigint::{BigInt, Sign};
use num_traits::{ToPrimitive, Zero};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{ClassDict, PyClassObject};
use prism_runtime::object::descriptor::{
    BoundMethod, ClassMethodDescriptor, PropertyDescriptor, StaticMethodDescriptor,
};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::super_obj::SuperObject;
use prism_runtime::object::type_builtins::{
    global_class, global_class_bitmap, global_class_registry, register_global_class, type_new,
    type_new_with_metaclass, unregister_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{MappingProxyObject, UnionTypeObject};
use prism_runtime::types::bytes::{BytesObject, clone_bytes_value, value_as_bytes_ref};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::int::{bigint_to_value, value_as_heap_int, value_to_bigint};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::memoryview::{
    MemoryViewFormat, MemoryViewObject, value_as_memoryview_ref,
};
use prism_runtime::types::set::SetObject;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::string::{StringObject, clone_string_value};
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{LazyLock, Mutex};

#[repr(C)]
struct BuiltinTypeObject {
    header: ObjectHeader,
    represented: TypeId,
}

impl BuiltinTypeObject {
    fn new(represented: TypeId) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TYPE),
            represented,
        }
    }
}

static TYPE_OBJECTS_BY_TYPE_ID: LazyLock<Mutex<FxHashMap<u32, usize>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));
static TYPE_IDS_BY_TYPE_OBJECT_PTR: LazyLock<Mutex<FxHashMap<usize, TypeId>>> =
    LazyLock::new(|| Mutex::new(FxHashMap::default()));

#[inline]
pub(crate) fn builtin_type_object_type_id(ptr: *const ()) -> Option<TypeId> {
    TYPE_IDS_BY_TYPE_OBJECT_PTR
        .lock()
        .expect("type-object pointer cache lock poisoned")
        .get(&(ptr as usize))
        .copied()
}

#[inline]
fn to_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn to_frozenset_value(mut set: SetObject) -> Value {
    set.header.type_id = TypeId::FROZENSET;
    to_object_value(set)
}

#[inline]
fn validate_hashable_value(value: Value) -> Result<(), BuiltinError> {
    builtin_hash(&[value]).map(|_| ())
}

#[inline]
fn build_validated_set(values: Vec<Value>) -> Result<SetObject, BuiltinError> {
    let mut set = SetObject::with_capacity(values.len());
    for value in values {
        validate_hashable_value(value)?;
        set.add(value);
    }
    Ok(set)
}

#[inline]
fn value_to_owned_string(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return Some(interned_by_ptr(ptr as *const u8)?.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(string.as_str().to_string())
}

#[inline]
fn validate_unicode_scalar(value: i64, context: &str) -> Result<Value, BuiltinError> {
    let code_point = u32::try_from(value).map_err(|_| {
        BuiltinError::ValueError(format!(
            "maketrans() {} must be a valid Unicode code point",
            context
        ))
    })?;
    char::from_u32(code_point).ok_or_else(|| {
        BuiltinError::ValueError(format!(
            "maketrans() {} must be a valid Unicode code point",
            context
        ))
    })?;
    Ok(Value::int(value).expect("validated translation code point should remain representable"))
}

#[inline]
fn normalize_str_maketrans_key(key: Value) -> Result<Value, BuiltinError> {
    if let Some(code_point) = key.as_int() {
        return validate_unicode_scalar(code_point, "key");
    }

    let string = value_to_owned_string(key).ok_or_else(|| {
        BuiltinError::TypeError(
            "if you give only one argument to maketrans it must be a dict".to_string(),
        )
    })?;
    let code_point = super::string::ord_from_str(&string).map_err(|_| {
        BuiltinError::ValueError("string keys in translate table must be of length 1".to_string())
    })?;
    Ok(Value::int(code_point as i64).expect("Unicode scalar should fit in Prism integer range"))
}

#[inline]
fn normalize_str_maketrans_value(value: Value) -> Result<Value, BuiltinError> {
    if value.is_none() {
        return Ok(value);
    }
    if let Some(code_point) = value.as_int() {
        return validate_unicode_scalar(code_point, "value");
    }
    if value_to_owned_string(value).is_some() {
        return Ok(value);
    }
    Err(BuiltinError::TypeError(
        "translation table values must be integers, strings, or None".to_string(),
    ))
}

fn type_object_for_type_id(type_id: TypeId) -> Value {
    {
        let map = TYPE_OBJECTS_BY_TYPE_ID
            .lock()
            .expect("type-object cache lock poisoned");
        if let Some(&ptr) = map.get(&type_id.raw()) {
            return Value::object_ptr(ptr as *const ());
        }
    }

    let type_object = BuiltinTypeObject::new(type_id);
    let ptr = Box::leak(Box::new(type_object)) as *mut BuiltinTypeObject as *const ();

    {
        let mut by_id = TYPE_OBJECTS_BY_TYPE_ID
            .lock()
            .expect("type-object cache lock poisoned");
        by_id.insert(type_id.raw(), ptr as usize);
    }
    {
        let mut by_ptr = TYPE_IDS_BY_TYPE_OBJECT_PTR
            .lock()
            .expect("type-object pointer cache lock poisoned");
        by_ptr.insert(ptr as usize, type_id);
    }

    Value::object_ptr(ptr)
}

#[inline]
pub(crate) fn builtin_type_object_for_type_id(type_id: TypeId) -> Value {
    type_object_for_type_id(type_id)
}

#[inline]
fn value_type_id(value: Value) -> TypeId {
    if value.is_none() {
        TypeId::NONE
    } else if value.is_bool() {
        TypeId::BOOL
    } else if value.is_int() {
        TypeId::INT
    } else if value.is_float() {
        TypeId::FLOAT
    } else if value.is_string() {
        TypeId::STR
    } else if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::CELL_VIEW => TypeId::CELL,
            type_id => type_id,
        }
    } else {
        TypeId::OBJECT
    }
}

#[inline]
pub(crate) fn value_type_object(value: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return type_object_for_type_id(value_type_id(value));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::TYPE => {
            if builtin_type_object_type_id(ptr).is_some() {
                return type_object_for_type_id(TypeId::TYPE);
            }

            let class = unsafe { &*(ptr as *const PyClassObject) };
            if !class.metaclass().is_none() {
                return class.metaclass();
            }

            type_object_for_type_id(TypeId::TYPE)
        }
        TypeId::EXCEPTION => unsafe { crate::builtins::ExceptionValue::from_value(value) }
            .and_then(|exception| {
                crate::builtins::exception_type_value_for_id(exception.exception_type_id)
            })
            .unwrap_or_else(|| type_object_for_type_id(TypeId::EXCEPTION)),
        TypeId::EXCEPTION_TYPE => type_object_for_type_id(TypeId::TYPE),
        _ => {
            let type_id = value_type_id(value);
            if type_id.raw() >= TypeId::FIRST_USER_TYPE {
                return global_class(ClassId(type_id.raw()))
                    .map(|class| Value::object_ptr(std::sync::Arc::as_ptr(&class) as *const ()))
                    .unwrap_or_else(|| type_object_for_type_id(type_id));
            }

            type_object_for_type_id(type_id)
        }
    }
}

#[inline]
fn class_value_to_type_id(class_value: Value) -> Option<TypeId> {
    let ptr = class_value.as_object_ptr()?;
    let object_type = crate::ops::objects::extract_type_id(ptr);

    match object_type {
        TypeId::TYPE => Some(builtin_type_object_type_id(ptr).unwrap_or_else(|| {
            let class = unsafe { &*(ptr as *const PyClassObject) };
            class.class_type_id()
        })),
        TypeId::EXCEPTION_TYPE => crate::builtins::exception_proxy_class_id_from_ptr(ptr)
            .map(|class_id| TypeId::from_raw(class_id.0)),
        _ => None,
    }
}

#[inline]
fn class_object_from_ptr(ptr: *const ()) -> Option<&'static PyClassObject> {
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE {
        return None;
    }
    if builtin_type_object_type_id(ptr).is_some() {
        return None;
    }

    Some(unsafe { &*(ptr as *const PyClassObject) })
}

#[inline]
fn parse_type_name_arg(value: Value) -> Result<InternedString, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("type() argument 1 must be str".to_string()))?;
        return interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("type() argument 1 must be str".to_string()));
    }

    if let Some(ptr) = value.as_object_ptr() {
        if crate::ops::objects::extract_type_id(ptr) == TypeId::STR {
            let string = unsafe { &*(ptr as *const StringObject) };
            return Ok(intern(string.as_str()));
        }
    }

    Err(BuiltinError::TypeError(
        "type() argument 1 must be str".to_string(),
    ))
}

fn parse_type_bases_arg(value: Value) -> Result<Vec<ClassId>, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("type() argument 2 must be tuple".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TUPLE {
        return Err(BuiltinError::TypeError(
            "type() argument 2 must be tuple".to_string(),
        ));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let mut bases = Vec::with_capacity(tuple.len());

    for (index, item) in tuple.as_slice().iter().copied().enumerate() {
        if let Some(type_id) = class_value_to_type_id(item) {
            bases.push(ClassId(type_id.raw()));
            continue;
        }

        return Err(BuiltinError::TypeError(format!(
            "type() argument 2 item {} is not a type",
            index
        )));
    }

    Ok(bases)
}

fn parse_type_namespace_arg(value: Value) -> Result<ClassDict, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("type() argument 3 must be dict".to_string()))?;
    let dict = crate::ops::objects::dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError("type() argument 3 must be dict".to_string()))?;
    let namespace = ClassDict::new();
    for (key, attr_value) in dict.iter() {
        let name = attribute_name(key).map_err(|_| {
            BuiltinError::TypeError("type() argument 3 must be dict with string keys".to_string())
        })?;
        namespace.set(name, attr_value);
    }

    Ok(namespace)
}

fn parse_class_spec(value: Value, fn_name: &'static str) -> Result<Vec<TypeId>, BuiltinError> {
    if let Some(type_id) = class_value_to_type_id(value) {
        return Ok(vec![type_id]);
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| class_spec_error(fn_name).clone())?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::UNION => {
            let union = unsafe { &*(ptr as *const UnionTypeObject) };
            let members = union.members().to_vec();
            if members.is_empty() {
                return Err(class_spec_error(fn_name));
            }
            return Ok(members);
        }
        TypeId::TUPLE => {}
        _ => {
            return Err(class_spec_error(fn_name));
        }
    }

    if crate::ops::objects::extract_type_id(ptr) != TypeId::TUPLE {
        return Err(class_spec_error(fn_name));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let mut out = Vec::new();
    for item in tuple.as_slice() {
        if let Some(type_id) = class_value_to_type_id(*item) {
            out.push(type_id);
            continue;
        }

        if let Some(item_ptr) = item.as_object_ptr() {
            if crate::ops::objects::extract_type_id(item_ptr) == TypeId::TUPLE {
                out.extend(parse_class_spec(*item, fn_name)?);
                continue;
            }
        }

        return Err(class_spec_error(fn_name));
    }

    if out.is_empty() {
        return Err(class_spec_error(fn_name));
    }
    Ok(out)
}

#[inline]
fn class_value_for_type_id(type_id: TypeId) -> Option<Value> {
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Some(builtin_type_object_for_type_id(type_id));
    }

    if let Some(exception_type_id) =
        crate::builtins::exception_type_id_for_proxy_class_id(ClassId(type_id.raw()))
    {
        return crate::builtins::exception_type_value_for_id(exception_type_id);
    }

    global_class(ClassId(type_id.raw()))
        .map(|class| Value::object_ptr(std::sync::Arc::as_ptr(&class) as *const ()))
}

fn parse_class_spec_values(
    value: Value,
    fn_name: &'static str,
) -> Result<Vec<Value>, BuiltinError> {
    if class_value_to_type_id(value).is_some() {
        return Ok(vec![value]);
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| class_spec_error(fn_name).clone())?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::UNION => {
            let union = unsafe { &*(ptr as *const UnionTypeObject) };
            let mut members = Vec::with_capacity(union.members().len());
            for &member in union.members() {
                let class_value =
                    class_value_for_type_id(member).ok_or_else(|| class_spec_error(fn_name))?;
                members.push(class_value);
            }
            if members.is_empty() {
                return Err(class_spec_error(fn_name));
            }
            return Ok(members);
        }
        TypeId::TUPLE => {}
        _ => {
            return Err(class_spec_error(fn_name));
        }
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let mut out = Vec::new();
    for item in tuple.as_slice() {
        if class_value_to_type_id(*item).is_some() {
            out.push(*item);
            continue;
        }

        if let Some(item_ptr) = item.as_object_ptr()
            && crate::ops::objects::extract_type_id(item_ptr) == TypeId::TUPLE
        {
            out.extend(parse_class_spec_values(*item, fn_name)?);
            continue;
        }

        return Err(class_spec_error(fn_name));
    }

    if out.is_empty() {
        return Err(class_spec_error(fn_name));
    }
    Ok(out)
}

#[inline]
fn class_spec_error(fn_name: &'static str) -> BuiltinError {
    if fn_name == "isinstance" {
        BuiltinError::TypeError(
            "isinstance() arg 2 must be a type, a tuple of types, or a union".to_string(),
        )
    } else {
        BuiltinError::TypeError(
            "issubclass() arg 2 must be a class, a tuple of classes, or a union".to_string(),
        )
    }
}

#[inline]
fn is_subtype(actual: TypeId, target: TypeId) -> bool {
    if actual == target {
        return true;
    }
    if target == TypeId::OBJECT {
        return true;
    }
    if actual == TypeId::BOOL && target == TypeId::INT {
        return true;
    }
    if actual == TypeId::EXCEPTION_TYPE && target == TypeId::TYPE {
        return true;
    }

    global_class_bitmap(ClassId(actual.raw())).is_some_and(|bitmap| bitmap.is_subclass_of(target))
}

#[inline]
fn class_value_is_subtype(class_value: Value, target: TypeId) -> bool {
    let Some(class_type) = class_value_to_type_id(class_value) else {
        return false;
    };
    if is_subtype(class_type, target) {
        return true;
    }

    let Some(class_ptr) = class_value.as_object_ptr() else {
        return false;
    };
    let Some(class) = class_object_from_ptr(class_ptr) else {
        return false;
    };

    class.mro().iter().copied().any(|class_id| {
        if class_id == ClassId::OBJECT {
            return target == TypeId::OBJECT;
        }
        if class_id.0 < TypeId::FIRST_USER_TYPE {
            return TypeId::from_raw(class_id.0) == target;
        }
        class_id.0 == target.raw()
    })
}

#[inline]
fn raw_issubclass_value(subclass: Value, target: Value) -> Result<bool, BuiltinError> {
    let target_type = class_value_to_type_id(target)
        .ok_or_else(|| BuiltinError::TypeError("issubclass() arg 2 must be a class".to_string()))?;
    if class_value_to_type_id(subclass).is_none() {
        return Err(BuiltinError::TypeError(
            "issubclass() arg 1 must be a class".to_string(),
        ));
    }

    Ok(class_value_is_subtype(subclass, target_type))
}

#[inline]
fn raw_isinstance_value(instance: Value, target: Value) -> Result<bool, BuiltinError> {
    let actual_type = value_type_object(instance);
    raw_issubclass_value(actual_type, target)
}

#[inline]
fn exact_class_match(instance: Value, target: Value) -> bool {
    value_type_object(instance) == target
}

#[inline]
fn is_exact_type_target(target: Value) -> bool {
    let Some(ptr) = target.as_object_ptr() else {
        return false;
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::TYPE => builtin_type_object_type_id(ptr).is_some(),
        TypeId::EXCEPTION_TYPE => true,
        _ => false,
    }
}

#[inline]
fn lookup_metaclass_check(target: Value, method_name: &'static str) -> Option<Value> {
    let ptr = target.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE {
        return None;
    }

    let class = class_object_from_ptr(ptr)?;
    let method = crate::ops::objects::lookup_class_metaclass_attr(class, &intern(method_name))?;
    Some(crate::ops::objects::bind_instance_attribute(method, target))
}

#[inline]
fn invoke_metaclass_check(
    vm: &mut VirtualMachine,
    target: Value,
    method_name: &'static str,
    operand: Value,
) -> Result<Option<bool>, BuiltinError> {
    let Some(checker) = lookup_metaclass_check(target, method_name) else {
        return Ok(None);
    };

    let result =
        invoke_callable_value(vm, checker, &[operand]).map_err(runtime_error_to_builtin_error)?;
    let truthy =
        crate::truthiness::try_is_truthy(vm, result).map_err(runtime_error_to_builtin_error)?;
    Ok(Some(truthy))
}

fn iter_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iter) = super::iter_dispatch::get_iterator_mut(&value) {
        return Ok(iter.collect_remaining());
    }

    let mut iter = super::iter_dispatch::value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iter.collect_remaining())
}

fn iter_values_with_vm(vm: &mut VirtualMachine, value: Value) -> Result<Vec<Value>, BuiltinError> {
    crate::ops::iteration::collect_iterable_values(vm, value)
        .map_err(runtime_error_to_builtin_error)
}

#[inline]
fn invoke_zero_arg_special_method(
    vm: &mut VirtualMachine,
    receiver: Value,
    method_name: &'static str,
) -> Result<Option<Value>, BuiltinError> {
    let target = match resolve_special_method(receiver, method_name) {
        Ok(target) => target,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(runtime_error_to_builtin_error(err)),
    };

    let result = match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
    .map_err(runtime_error_to_builtin_error)?;

    if result == crate::builtins::builtin_not_implemented_value() {
        Ok(None)
    } else {
        Ok(Some(result))
    }
}

fn invoke_bound_method_with_arg(
    vm: &mut VirtualMachine,
    target: &BoundMethodTarget,
    arg: Value,
) -> Result<Value, BuiltinError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self, arg]),
        None => invoke_callable_value(vm, target.callable, &[arg]),
    }
    .map_err(runtime_error_to_builtin_error)
}

fn mapping_entries_with_vm(
    vm: &mut VirtualMachine,
    source: Value,
) -> Result<Option<Vec<(Value, Value)>>, BuiltinError> {
    let Some(keys_value) = invoke_zero_arg_special_method(vm, source, "keys")? else {
        return Ok(None);
    };
    let keys = iter_values_with_vm(vm, keys_value)?;
    let get_item =
        resolve_special_method(source, "__getitem__").map_err(runtime_error_to_builtin_error)?;
    let mut entries = Vec::with_capacity(keys.len());
    for key in keys {
        entries.push((key, invoke_bound_method_with_arg(vm, &get_item, key)?));
    }
    Ok(Some(entries))
}

#[inline]
fn runtime_error_to_builtin_error(err: crate::error::RuntimeError) -> BuiltinError {
    super::runtime_error_to_builtin_error(err)
}

fn dict_item_to_pair(item: Value, index: usize) -> Result<(Value, Value), BuiltinError> {
    if let Some(ptr) = item.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                let len = tuple.len();
                if len != 2 {
                    return Err(BuiltinError::TypeError(format!(
                        "dictionary update sequence element #{} has length {}; 2 is required",
                        index, len
                    )));
                }
                return Ok((tuple.get(0).unwrap(), tuple.get(1).unwrap()));
            }
            TypeId::LIST => {
                let list = unsafe { &*(ptr as *const ListObject) };
                let len = list.len();
                if len != 2 {
                    return Err(BuiltinError::TypeError(format!(
                        "dictionary update sequence element #{} has length {}; 2 is required",
                        index, len
                    )));
                }
                return Ok((list.get(0).unwrap(), list.get(1).unwrap()));
            }
            _ => {}
        }
    }

    let values = iter_values(item).map_err(|_| {
        BuiltinError::TypeError(format!(
            "cannot convert dictionary update sequence element #{} to a sequence",
            index
        ))
    })?;
    if values.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "dictionary update sequence element #{} has length {}; 2 is required",
            index,
            values.len()
        )));
    }
    Ok((values[0], values[1]))
}

fn attribute_name(name: Value) -> Result<InternedString, BuiltinError> {
    if name.is_string() {
        let ptr = name
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("attribute name must be string".to_string()))?;
        return interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("attribute name must be string".to_string()));
    }

    if let Some(ptr) = name.as_object_ptr() {
        if crate::ops::objects::extract_type_id(ptr) == TypeId::STR {
            let string_obj = unsafe { &*(ptr as *const StringObject) };
            return Ok(intern(string_obj.as_str()));
        }
    }

    Err(BuiltinError::TypeError(
        "attribute name must be string".to_string(),
    ))
}

#[inline]
fn value_to_slice_index(value: Value) -> Result<Option<i64>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }

    if let Some(index) = prism_runtime::types::int::value_to_i64(value) {
        return Ok(Some(index));
    }

    Err(BuiltinError::TypeError(
        "slice indices must be integers or None".to_string(),
    ))
}

/// Call a builtin type object using the same constructors that power the
/// corresponding Python-visible builtins.
pub(crate) fn call_builtin_type(type_id: TypeId, args: &[Value]) -> Result<Value, BuiltinError> {
    match type_id {
        TypeId::INT => builtin_int(args),
        TypeId::FLOAT => builtin_float(args),
        TypeId::COMPLEX => super::numeric::builtin_complex(args),
        TypeId::STR => builtin_str(args),
        TypeId::BOOL => builtin_bool(args),
        TypeId::BYTES => super::string::builtin_bytes(args),
        TypeId::LIST => builtin_list(args),
        TypeId::TUPLE => builtin_tuple(args),
        TypeId::DICT => builtin_dict(args),
        TypeId::SET => builtin_set(args),
        TypeId::FROZENSET => builtin_frozenset(args),
        TypeId::TYPE => builtin_type(args),
        TypeId::OBJECT => builtin_object(args),
        TypeId::SLICE => builtin_slice(args),
        TypeId::RANGE => super::itertools::builtin_range(args),
        TypeId::BYTEARRAY => super::string::builtin_bytearray(args),
        TypeId::MEMORYVIEW => builtin_memoryview(args),
        TypeId::MAPPING_PROXY => builtin_mappingproxy(args),
        TypeId::DEQUE => builtin_deque(args),
        TypeId::METHOD => builtin_methodtype(args),
        TypeId::MODULE => builtin_module(args),
        TypeId::CLASSMETHOD => builtin_classmethod(args),
        TypeId::STATICMETHOD => builtin_staticmethod(args),
        TypeId::PROPERTY => builtin_property(args),
        TypeId::SUPER => Err(BuiltinError::TypeError(
            "super() requires VM context".to_string(),
        )),
        _ => Err(BuiltinError::TypeError(format!(
            "'{}' is not a callable builtin type",
            type_id.name()
        ))),
    }
}

/// Call a builtin type object with VM-aware iterable consumption where needed.
pub(crate) fn call_builtin_type_with_vm(
    vm: &mut VirtualMachine,
    type_id: TypeId,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    match type_id {
        TypeId::INT => builtin_int_vm(vm, args),
        TypeId::TYPE => builtin_type_with_vm(vm, args),
        TypeId::SUPER => builtin_super(vm, args),
        TypeId::MAPPING_PROXY => builtin_mappingproxy(args),
        TypeId::DEQUE => builtin_deque_with_vm(vm, args),
        TypeId::METHOD => builtin_methodtype(args),
        TypeId::MODULE => builtin_module(args),
        TypeId::BOOL => builtin_bool_vm(vm, args),
        TypeId::FLOAT => builtin_float_vm(vm, args),
        TypeId::LIST => {
            if args.len() > 1 {
                return Err(BuiltinError::TypeError(format!(
                    "list() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if args.is_empty() {
                return Ok(to_object_value(ListObject::new()));
            }

            let values = iter_values_with_vm(vm, args[0])?;
            Ok(to_object_value(ListObject::from_iter(values)))
        }
        TypeId::TUPLE => {
            if args.len() > 1 {
                return Err(BuiltinError::TypeError(format!(
                    "tuple() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if args.is_empty() {
                return Ok(to_object_value(TupleObject::empty()));
            }

            let values = iter_values_with_vm(vm, args[0])?;
            Ok(to_object_value(TupleObject::from_vec(values)))
        }
        TypeId::DICT => {
            if args.len() > 1 {
                return Err(BuiltinError::TypeError(format!(
                    "dict() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if args.is_empty() {
                return Ok(to_object_value(DictObject::new()));
            }

            if let Some(ptr) = args[0].as_object_ptr() {
                if crate::ops::objects::extract_type_id(ptr) == TypeId::DICT {
                    let source = unsafe { &*(ptr as *const DictObject) };
                    let mut copy = DictObject::with_capacity(source.len());
                    for (key, value) in source.iter() {
                        copy.set(key, value);
                    }
                    return Ok(to_object_value(copy));
                }
            }

            if let Some(entries) = mapping_entries_with_vm(vm, args[0])? {
                let mut dict = DictObject::with_capacity(entries.len());
                for (key, value) in entries {
                    dict.set(key, value);
                }
                return Ok(to_object_value(dict));
            }

            let mut dict = DictObject::new();
            let mut sequence = iter_values_with_vm(vm, args[0])?;

            for (index, item) in sequence.drain(..).enumerate() {
                let (key, value) = dict_item_to_pair(item, index)?;
                dict.set(key, value);
            }

            Ok(to_object_value(dict))
        }
        TypeId::SET => {
            if args.len() > 1 {
                return Err(BuiltinError::TypeError(format!(
                    "set() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if args.is_empty() {
                return Ok(to_object_value(SetObject::new()));
            }

            let values = iter_values_with_vm(vm, args[0])?;
            Ok(to_object_value(build_validated_set(values)?))
        }
        TypeId::FROZENSET => {
            if args.len() > 1 {
                return Err(BuiltinError::TypeError(format!(
                    "frozenset() takes at most 1 argument ({} given)",
                    args.len()
                )));
            }
            if args.is_empty() {
                return Ok(to_frozenset_value(SetObject::new()));
            }

            if let Some(ptr) = args[0].as_object_ptr() {
                match crate::ops::objects::extract_type_id(ptr) {
                    TypeId::FROZENSET => return Ok(args[0]),
                    TypeId::SET => {
                        let source = unsafe { &*(ptr as *const SetObject) };
                        return Ok(to_frozenset_value(SetObject::from_iter(source.iter())));
                    }
                    _ => {}
                }
            }

            let values = iter_values_with_vm(vm, args[0])?;
            Ok(to_frozenset_value(build_validated_set(values)?))
        }
        _ => call_builtin_type(type_id, args),
    }
}

fn builtin_mappingproxy(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "mappingproxy() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let mapping = args[0];
    let Some(ptr) = mapping.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "mappingproxy() argument must be a mapping".to_string(),
        ));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::DICT => Ok(to_object_value(MappingProxyObject::for_mapping(mapping))),
        TypeId::MAPPING_PROXY => Ok(mapping),
        _ => Err(BuiltinError::TypeError(
            "mappingproxy() argument must be a mapping".to_string(),
        )),
    }
}

pub(crate) fn builtin_module(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "module() missing required argument 'name' (pos 1)".to_string(),
        ));
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "module() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = value_to_owned_string(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "module() argument 'name' must be str, not {}",
            args[0].type_name()
        ))
    })?;
    let module = ModuleObject::new(name);

    if let Some(doc) = args.get(1).copied() {
        module.set_attr("__doc__", doc);
    }

    Ok(to_object_value(module))
}

fn metaclass_value_for_type_base(base: Value) -> Value {
    let Some(ptr) = base.as_object_ptr() else {
        return Value::none();
    };
    if builtin_type_object_type_id(ptr).is_some() {
        return Value::none();
    }
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE {
        return Value::none();
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    class.metaclass()
}

fn metaclass_value_is_subclass(candidate: Value, target: Value) -> bool {
    if target.is_none() {
        return true;
    }
    if candidate.is_none() {
        return target.is_none();
    }
    if candidate == target {
        return true;
    }

    let Some(candidate_type) = class_value_to_type_id(candidate) else {
        return false;
    };
    let Some(target_type) = class_value_to_type_id(target) else {
        return false;
    };

    is_subtype(candidate_type, target_type)
}

fn choose_more_derived_metaclass_value(
    current: Value,
    candidate: Value,
) -> Result<Value, BuiltinError> {
    if metaclass_value_is_subclass(candidate, current) {
        return Ok(candidate);
    }
    if metaclass_value_is_subclass(current, candidate) {
        return Ok(current);
    }
    Err(BuiltinError::TypeError(
        "metaclass conflict: metaclass hierarchy is incompatible".to_string(),
    ))
}

fn resolve_type_call_metaclass(bases_value: Value) -> Result<Value, BuiltinError> {
    let ptr = bases_value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("type() argument 2 must be tuple".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TUPLE {
        return Err(BuiltinError::TypeError(
            "type() argument 2 must be tuple".to_string(),
        ));
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let mut winner = Value::none();
    for (index, base) in tuple.as_slice().iter().copied().enumerate() {
        if class_value_to_type_id(base).is_none() {
            return Err(BuiltinError::TypeError(format!(
                "type() argument 2 item {} is not a type",
                index
            )));
        }
        winner = choose_more_derived_metaclass_value(winner, metaclass_value_for_type_base(base))?;
    }
    Ok(winner)
}

fn builtin_type_kw_with_vm(
    vm: &mut VirtualMachine,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if positional.len() != 3 {
        if positional.len() == 1 {
            return Err(BuiltinError::TypeError(
                "type() does not accept keyword arguments in 1-argument form".to_string(),
            ));
        }
        return builtin_type(positional);
    }

    parse_type_name_arg(positional[0])?;
    parse_type_bases_arg(positional[1])?;
    parse_type_namespace_arg(positional[2])?;

    let metaclass = resolve_type_call_metaclass(positional[1])?;
    if metaclass.is_none() {
        return Err(BuiltinError::TypeError(
            "type() does not accept keyword arguments yet".to_string(),
        ));
    }

    let result =
        crate::ops::calls::invoke_callable_value_with_keywords(vm, metaclass, positional, keywords)
            .map_err(runtime_error_to_builtin_error)?;

    let Some(result_ptr) = result.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "type() did not return a type object".to_string(),
        ));
    };
    if crate::ops::objects::extract_type_id(result_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(format!(
            "type() returned non-type '{}'",
            crate::ops::objects::extract_type_id(result_ptr).name()
        )));
    }

    Ok(result)
}

fn resolve_super_type_arg(value: Value) -> Result<ClassId, BuiltinError> {
    let type_id = class_value_to_type_id(value)
        .ok_or_else(|| BuiltinError::TypeError("super() argument 1 must be a type".to_string()))?;
    Ok(ClassId(type_id.raw()))
}

fn resolve_super_class_cell_slot(code: &prism_code::CodeObject) -> Option<usize> {
    code.cellvars
        .iter()
        .position(|name| name.as_ref() == "__class__")
        .or_else(|| {
            code.freevars
                .iter()
                .position(|name| name.as_ref() == "__class__")
                .map(|idx| code.cellvars.len() + idx)
        })
}

fn resolve_implicit_super_context(vm: &VirtualMachine) -> Result<(ClassId, Value), BuiltinError> {
    let frame = vm.current_frame();
    if frame.code.arg_count == 0 {
        return Err(BuiltinError::TypeError("super(): no arguments".to_string()));
    }

    let closure = frame
        .closure
        .as_ref()
        .ok_or_else(|| BuiltinError::TypeError("super(): __class__ cell not found".to_string()))?;
    let slot = resolve_super_class_cell_slot(&frame.code)
        .ok_or_else(|| BuiltinError::TypeError("super(): __class__ cell not found".to_string()))?;
    if slot >= closure.len() {
        return Err(BuiltinError::TypeError(
            "super(): __class__ cell is invalid".to_string(),
        ));
    }

    let class_value = closure
        .get_cell(slot)
        .get()
        .ok_or_else(|| BuiltinError::TypeError("super(): empty __class__ cell".to_string()))?;
    let this_type = resolve_super_type_arg(class_value)?;
    Ok((this_type, frame.get_reg(0)))
}

fn bind_super_value(this_type: ClassId, bound: Value) -> Result<Value, BuiltinError> {
    if let Some(bound_type) = class_value_to_type_id(bound) {
        let this_type_id = TypeId::from_raw(this_type.0);
        if !is_subtype(bound_type, this_type_id) {
            return Err(BuiltinError::TypeError(
                "super(type, obj): obj must be an instance or subtype of type".to_string(),
            ));
        }
        return Ok(to_object_value(SuperObject::new_type(
            this_type,
            bound,
            ClassId(bound_type.raw()),
        )));
    }

    let obj_type = value_type_id(bound);
    let this_type_id = TypeId::from_raw(this_type.0);
    if !is_subtype(obj_type, this_type_id) {
        return Err(BuiltinError::TypeError(
            "super(type, obj): obj must be an instance or subtype of type".to_string(),
        ));
    }

    Ok(to_object_value(SuperObject::new_instance(
        this_type,
        bound,
        ClassId(obj_type.raw()),
    )))
}

fn builtin_super(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    match args.len() {
        0 => {
            let (this_type, bound) = resolve_implicit_super_context(vm)?;
            bind_super_value(this_type, bound)
        }
        1 => {
            let this_type = resolve_super_type_arg(args[0])?;
            Ok(to_object_value(SuperObject::new_unbound(this_type)))
        }
        2 => {
            let this_type = resolve_super_type_arg(args[0])?;
            bind_super_value(this_type, args[1])
        }
        _ => Err(BuiltinError::TypeError(format!(
            "super() takes at most 2 arguments ({} given)",
            args.len()
        ))),
    }
}

pub(crate) fn call_builtin_type_kw_with_vm(
    vm: &mut VirtualMachine,
    type_id: TypeId,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if keywords.is_empty() {
        return call_builtin_type_with_vm(vm, type_id, positional);
    }

    match type_id {
        TypeId::INT => builtin_int_kw_vm(vm, positional, keywords),
        TypeId::TYPE => builtin_type_kw_with_vm(vm, positional, keywords),
        TypeId::PROPERTY => builtin_property_kw(positional, keywords),
        TypeId::DICT => builtin_dict_kw(positional, keywords),
        TypeId::STR => builtin_str_kw(positional, keywords),
        TypeId::DEQUE => builtin_deque_kw(positional, keywords),
        _ => Err(BuiltinError::TypeError(format!(
            "{}() does not accept keyword arguments yet",
            type_id.name()
        ))),
    }
}

pub(crate) fn call_builtin_type_kw(
    type_id: TypeId,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if keywords.is_empty() {
        return call_builtin_type(type_id, positional);
    }

    match type_id {
        TypeId::INT => builtin_int_kw(positional, keywords),
        TypeId::PROPERTY => builtin_property_kw(positional, keywords),
        TypeId::DICT => builtin_dict_kw(positional, keywords),
        TypeId::STR => builtin_str_kw(positional, keywords),
        TypeId::DEQUE => builtin_deque_kw(positional, keywords),
        _ => Err(BuiltinError::TypeError(format!(
            "{}() does not accept keyword arguments yet",
            type_id.name()
        ))),
    }
}

pub fn builtin_classmethod(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "classmethod() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(to_object_value(ClassMethodDescriptor::new(args[0])))
}

pub fn builtin_methodtype(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "method expected 2 arguments, got {}",
            args.len()
        )));
    }

    let function = args[0];
    let instance = args[1];

    if !crate::ops::calls::value_supports_call_protocol(function) {
        return Err(BuiltinError::TypeError(
            "first argument must be callable".to_string(),
        ));
    }

    if instance.is_none() {
        return Err(BuiltinError::TypeError(
            "instance must not be None".to_string(),
        ));
    }

    Ok(to_object_value(BoundMethod::new(function, instance)))
}

pub fn builtin_staticmethod(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "staticmethod() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(to_object_value(StaticMethodDescriptor::new(args[0])))
}

pub fn builtin_property(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "property() takes at most 4 arguments ({} given)",
            args.len()
        )));
    }

    let normalize = |value: Option<Value>| value.filter(|value| !value.is_none());
    let getter = normalize(args.first().copied());
    let setter = normalize(args.get(1).copied());
    let deleter = normalize(args.get(2).copied());
    let doc = normalize(args.get(3).copied());

    Ok(to_object_value(PropertyDescriptor::new_full(
        getter, setter, deleter, doc,
    )))
}

fn builtin_property_kw(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if positional.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "property() takes at most 4 arguments ({} given)",
            positional.len()
        )));
    }

    let mut slots: [Option<Value>; 4] = [None, None, None, None];
    for (index, value) in positional.iter().copied().enumerate() {
        slots[index] = Some(value);
    }

    for &(name, value) in keywords {
        let index = match name {
            "fget" => 0,
            "fset" => 1,
            "fdel" => 2,
            "doc" => 3,
            _ => {
                return Err(BuiltinError::TypeError(format!(
                    "property() got an unexpected keyword argument '{}'",
                    name
                )));
            }
        };

        if index < positional.len() || slots[index].is_some() {
            return Err(BuiltinError::TypeError(format!(
                "property() got multiple values for argument '{}'",
                name
            )));
        }
        slots[index] = Some(value);
    }

    let normalize = |value: Option<Value>| value.filter(|value| !value.is_none());
    Ok(to_object_value(PropertyDescriptor::new_full(
        normalize(slots[0]),
        normalize(slots[1]),
        normalize(slots[2]),
        normalize(slots[3]),
    )))
}

/// Builtin int constructor.
pub fn builtin_int(args: &[Value]) -> Result<Value, BuiltinError> {
    let Some((arg, explicit_base)) = parse_builtin_int_args(args)? else {
        return Ok(Value::int(0).expect("zero should be representable"));
    };

    if let Some(value) = builtin_int_native(arg, explicit_base)? {
        return Ok(value);
    }

    if let Some(buffer_arg) = int_buffer_argument(arg) {
        return parse_int_text_argument(&buffer_arg, 10);
    }

    Err(builtin_int_unsupported_argument(arg))
}

pub fn builtin_int_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let Some((arg, explicit_base)) = parse_builtin_int_args(args)? else {
        return Ok(Value::int(0).expect("zero should be representable"));
    };

    if let Some(value) = builtin_int_native(arg, explicit_base)? {
        return Ok(value);
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__int__")? {
        return int_protocol_result(result, "__int__");
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__index__")? {
        return int_protocol_result(result, "__index__");
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__trunc__")? {
        return int_trunc_protocol_result(vm, result);
    }

    if let Some(buffer_arg) = int_buffer_argument(arg) {
        return parse_int_text_argument(&buffer_arg, 10);
    }

    Err(builtin_int_unsupported_argument(arg))
}

fn builtin_int_kw(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    let args = collect_builtin_int_keyword_args(positional, keywords)?;
    builtin_int(&args)
}

fn builtin_int_kw_vm(
    vm: &mut VirtualMachine,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let args = collect_builtin_int_keyword_args(positional, keywords)?;
    builtin_int_vm(vm, &args)
}

fn collect_builtin_int_keyword_args(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Vec<Value>, BuiltinError> {
    if positional.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            positional.len() + keywords.len()
        )));
    }

    let mut base = None;
    for &(name, value) in keywords {
        match name {
            "base" => {
                if positional.len() >= 2 {
                    return Err(BuiltinError::TypeError(format!(
                        "int() takes at most 2 arguments ({} given)",
                        positional.len() + 1
                    )));
                }
                if base.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "int() got multiple values for keyword argument 'base'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "int() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    match (positional, base) {
        ([], None) => Ok(Vec::new()),
        ([], Some(_)) => Err(BuiltinError::TypeError(
            "int() missing string argument".to_string(),
        )),
        ([arg], None) => Ok(vec![*arg]),
        ([arg], Some(base)) => Ok(vec![*arg, base]),
        ([arg, base], None) => Ok(vec![*arg, *base]),
        _ => unreachable!("positional argument count was validated"),
    }
}

#[inline]
fn parse_builtin_int_args(args: &[Value]) -> Result<Option<(Value, Option<u32>)>, BuiltinError> {
    if args.is_empty() {
        return Ok(None);
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    let explicit_base = if args.len() == 2 {
        Some(parse_int_base_argument(args[1])?)
    } else {
        None
    };
    Ok(Some((args[0], explicit_base)))
}

#[inline]
fn builtin_int_native(
    arg: Value,
    explicit_base: Option<u32>,
) -> Result<Option<Value>, BuiltinError> {
    if let Some(text_arg) = int_text_argument(arg) {
        return parse_int_text_argument(&text_arg, explicit_base.unwrap_or(10)).map(Some);
    }

    if explicit_base.is_some() {
        return Err(BuiltinError::TypeError(
            "int() can't convert non-string with explicit base".to_string(),
        ));
    }

    if arg.as_int().is_some() || value_as_heap_int(arg).is_some() {
        return Ok(Some(arg));
    }
    if let Some(integer) = value_to_bigint(arg) {
        return Ok(Some(bigint_to_value(integer)));
    }
    if let Some(f) = arg.as_float() {
        return Value::int(f as i64)
            .ok_or_else(|| BuiltinError::OverflowError("int too large".to_string()))
            .map(Some);
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Some(
            Value::int(if b { 1 } else { 0 }).expect("bool integer should be representable"),
        ));
    }

    Ok(None)
}

#[inline]
fn builtin_int_unsupported_argument(arg: Value) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "int() argument must be a string, a bytes-like object or a real number, not '{}'",
        arg.type_name()
    ))
}

#[inline]
fn int_protocol_result(result: Value, method_name: &'static str) -> Result<Value, BuiltinError> {
    if let Some(boolean) = result.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool integer should be representable"));
    }

    let integer = value_to_bigint(result).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{method_name} returned non-int (type {})",
            result.type_name()
        ))
    })?;
    Ok(bigint_to_value(integer))
}

fn int_trunc_protocol_result(
    vm: &mut VirtualMachine,
    result: Value,
) -> Result<Value, BuiltinError> {
    if let Some(boolean) = result.as_bool() {
        return Ok(Value::int(i64::from(boolean)).expect("bool integer should be representable"));
    }
    if let Some(integer) = value_to_bigint(result) {
        return Ok(bigint_to_value(integer));
    }

    if let Some(indexed) = invoke_zero_arg_special_method(vm, result, "__index__")? {
        return int_protocol_result(indexed, "__index__");
    }

    Err(BuiltinError::TypeError(format!(
        "__trunc__ returned non-Integral (type {})",
        result.type_name()
    )))
}

enum IntTextArgument {
    Str(String),
    Bytes(Vec<u8>),
}

impl IntTextArgument {
    #[inline]
    fn raw_bytes(&self) -> &[u8] {
        match self {
            Self::Str(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }

    fn invalid_literal(&self, base: u32) -> BuiltinError {
        match self {
            Self::Str(text) => BuiltinError::ValueError(format!(
                "invalid literal for int() with base {base}: {:?}",
                text
            )),
            Self::Bytes(bytes) => {
                let text = String::from_utf8_lossy(bytes);
                BuiltinError::ValueError(format!(
                    "invalid literal for int() with base {base}: b{:?}",
                    text
                ))
            }
        }
    }
}

#[inline]
fn int_text_argument(value: Value) -> Option<IntTextArgument> {
    if let Some(text) = value_to_owned_string(value) {
        return Some(IntTextArgument::Str(text));
    }

    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Some(IntTextArgument::Bytes(bytes.as_bytes().to_vec()))
        }
        _ => None,
    }
}

#[inline]
fn int_buffer_argument(value: Value) -> Option<IntTextArgument> {
    value_as_memoryview_ref(value).map(|view| IntTextArgument::Bytes(view.as_bytes().to_vec()))
}

#[inline]
fn parse_int_base_argument(value: Value) -> Result<u32, BuiltinError> {
    let base = if let Some(integer) = prism_runtime::types::int::value_to_i64(value) {
        integer
    } else if let Some(boolean) = value.as_bool() {
        if boolean { 1 } else { 0 }
    } else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        )));
    };

    if base == 0 || (2..=36).contains(&base) {
        Ok(base as u32)
    } else {
        Err(BuiltinError::ValueError(
            "int() base must be >= 2 and <= 36, or 0".to_string(),
        ))
    }
}

fn parse_int_text_argument(argument: &IntTextArgument, base: u32) -> Result<Value, BuiltinError> {
    let trimmed = trim_ascii_whitespace(argument.raw_bytes());
    if trimmed.is_empty() {
        return Err(argument.invalid_literal(base));
    }

    let (negative, digits) = match trimmed[0] {
        b'+' => (false, &trimmed[1..]),
        b'-' => (true, &trimmed[1..]),
        _ => (false, trimmed),
    };
    if digits.is_empty() {
        return Err(argument.invalid_literal(base));
    }

    let (resolved_base, digits, allow_leading_underscore) = resolve_int_parse_base(base, digits);
    let normalized = normalize_int_digits(digits, resolved_base, allow_leading_underscore)
        .ok_or_else(|| argument.invalid_literal(resolved_base))?;

    let mut value = BigInt::parse_bytes(&normalized, resolved_base)
        .ok_or_else(|| argument.invalid_literal(resolved_base))?;
    if negative {
        value = -value;
    }
    Ok(bigint_to_value(value))
}

#[inline]
fn trim_ascii_whitespace(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|byte| !byte.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    let end = bytes
        .iter()
        .rposition(|byte| !byte.is_ascii_whitespace())
        .map(|index| index + 1)
        .unwrap_or(start);
    &bytes[start..end]
}

#[inline]
fn resolve_int_parse_base(base: u32, digits: &[u8]) -> (u32, &[u8], bool) {
    if let Some((prefixed_base, prefix_len)) = int_base_prefix(digits) {
        if base == 0 {
            return (prefixed_base, &digits[prefix_len..], true);
        }
        if prefixed_base == base {
            return (base, &digits[prefix_len..], true);
        }
    }

    if base == 0 {
        (10, digits, false)
    } else {
        (base, digits, false)
    }
}

#[inline]
fn int_base_prefix(bytes: &[u8]) -> Option<(u32, usize)> {
    if bytes.len() < 2 || bytes[0] != b'0' {
        return None;
    }

    match bytes[1] {
        b'b' | b'B' => Some((2, 2)),
        b'o' | b'O' => Some((8, 2)),
        b'x' | b'X' => Some((16, 2)),
        _ => None,
    }
}

fn normalize_int_digits(
    digits: &[u8],
    base: u32,
    allow_leading_underscore: bool,
) -> Option<Vec<u8>> {
    let mut normalized = Vec::with_capacity(digits.len());
    let mut saw_digit = false;
    let mut previous_was_underscore = false;
    let mut leading_underscore_allowed = allow_leading_underscore;

    for &byte in digits {
        if byte == b'_' {
            if previous_was_underscore || (!saw_digit && !leading_underscore_allowed) {
                return None;
            }
            previous_was_underscore = true;
            leading_underscore_allowed = false;
            continue;
        }

        let digit = ascii_digit_value(byte)?;
        if digit >= base {
            return None;
        }

        normalized.push(byte.to_ascii_lowercase());
        saw_digit = true;
        previous_was_underscore = false;
        leading_underscore_allowed = false;
    }

    if !saw_digit || previous_was_underscore {
        return None;
    }

    Some(normalized)
}

#[inline]
fn ascii_digit_value(byte: u8) -> Option<u32> {
    match byte {
        b'0'..=b'9' => Some((byte - b'0') as u32),
        b'a'..=b'z' => Some((byte - b'a') as u32 + 10),
        b'A'..=b'Z' => Some((byte - b'A') as u32 + 10),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ByteOrder {
    Big,
    Little,
}

pub(crate) fn builtin_int_from_bytes(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    builtin_int_from_bytes_impl(None, args, keywords)
}

pub(crate) fn builtin_int_from_bytes_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    builtin_int_from_bytes_impl(Some(vm), args, keywords)
}

fn builtin_int_from_bytes_impl(
    vm: Option<&mut VirtualMachine>,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "from_bytes() descriptor requires a type receiver".to_string(),
        ));
    }

    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() takes at most 2 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver_type = int_from_bytes_receiver_type(args[0])?;
    let mut bytes_arg = args.get(1).copied();
    let mut byteorder_arg = args.get(2).copied();
    let mut signed_arg: Option<Value> = None;

    for &(name, value) in keywords {
        match name {
            "bytes" => assign_from_bytes_keyword(&mut bytes_arg, value, 1, args.len(), "bytes")?,
            "byteorder" => {
                assign_from_bytes_keyword(&mut byteorder_arg, value, 2, args.len(), "byteorder")?
            }
            "signed" => {
                if signed_arg.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "from_bytes() got multiple values for argument 'signed'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "from_bytes() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let bytes_arg = bytes_arg.ok_or_else(|| {
        BuiltinError::TypeError(
            "from_bytes() missing required argument 'bytes' (pos 1)".to_string(),
        )
    })?;
    let byteorder = match byteorder_arg {
        Some(value) => parse_from_bytes_byteorder(value)?,
        None => ByteOrder::Big,
    };
    let signed = signed_arg
        .map(crate::truthiness::is_truthy)
        .unwrap_or(false);

    let bytes = match vm {
        Some(vm) => value_to_byte_sequence_with_vm(vm, bytes_arg, "from_bytes() argument 1")?,
        None => value_to_byte_sequence(bytes_arg, "from_bytes() argument 1")?,
    };
    let value = decode_bigint_from_bytes(&bytes, byteorder, signed);

    match receiver_type {
        TypeId::BOOL => Ok(Value::bool(!value.is_zero())),
        TypeId::INT => Ok(bigint_to_value(value)),
        other => unreachable!("unexpected int.from_bytes receiver type: {other:?}"),
    }
}

pub(crate) fn builtin_int_to_bytes(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "to_bytes() descriptor requires an int receiver".to_string(),
        ));
    }

    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() takes at most 2 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let value = int_to_bytes_receiver_value(args[0])?;
    let mut length_arg = args.get(1).copied();
    let mut byteorder_arg = args.get(2).copied();
    let mut signed_arg: Option<Value> = None;

    for &(name, value) in keywords {
        match name {
            "length" => assign_to_bytes_keyword(&mut length_arg, value, 1, args.len(), "length")?,
            "byteorder" => {
                assign_to_bytes_keyword(&mut byteorder_arg, value, 2, args.len(), "byteorder")?
            }
            "signed" => {
                if signed_arg.replace(value).is_some() {
                    return Err(BuiltinError::TypeError(
                        "to_bytes() got multiple values for argument 'signed'".to_string(),
                    ));
                }
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "to_bytes() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let length = match length_arg {
        Some(value) => parse_to_bytes_length(value)?,
        None => 1,
    };
    let byteorder = match byteorder_arg {
        Some(value) => parse_to_bytes_byteorder(value)?,
        None => ByteOrder::Big,
    };
    let signed = signed_arg
        .map(crate::truthiness::is_truthy)
        .unwrap_or(false);

    let bytes = encode_bigint_to_bytes(&value, length, byteorder, signed)?;
    Ok(crate::alloc_managed_value(BytesObject::from_vec(bytes)))
}

fn int_from_bytes_receiver_type(receiver: Value) -> Result<TypeId, BuiltinError> {
    let class_type = class_value_to_type_id(receiver).ok_or_else(|| {
        BuiltinError::TypeError("from_bytes() descriptor requires a type receiver".to_string())
    })?;

    match class_type {
        TypeId::INT | TypeId::BOOL => Ok(class_type),
        _ if class_value_is_subtype(receiver, TypeId::INT) => Err(BuiltinError::NotImplemented(
            "from_bytes() for int subclasses is not implemented yet".to_string(),
        )),
        _ => Err(BuiltinError::TypeError(
            "from_bytes() requires the built-in int or bool type".to_string(),
        )),
    }
}

fn int_to_bytes_receiver_value(receiver: Value) -> Result<BigInt, BuiltinError> {
    if let Some(boolean) = receiver.as_bool() {
        return Ok(BigInt::from(u8::from(boolean)));
    }

    value_to_bigint(receiver).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor 'int.to_bytes' requires an 'int' object but received '{}'",
            receiver.type_name()
        ))
    })
}

fn assign_from_bytes_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    if slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "from_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    Ok(())
}

fn assign_to_bytes_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    if slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "to_bytes() got multiple values for argument '{}'",
            name
        )));
    }

    Ok(())
}

fn parse_from_bytes_byteorder(value: Value) -> Result<ByteOrder, BuiltinError> {
    parse_byteorder_arg(value, "from_bytes")
}

fn parse_to_bytes_byteorder(value: Value) -> Result<ByteOrder, BuiltinError> {
    parse_byteorder_arg(value, "to_bytes")
}

fn parse_byteorder_arg(value: Value, fn_name: &str) -> Result<ByteOrder, BuiltinError> {
    let Some(byteorder) = value_to_owned_string(value) else {
        return Err(BuiltinError::TypeError(format!(
            "{fn_name}() argument 'byteorder' must be str, not {}",
            value.type_name()
        )));
    };

    match byteorder.as_str() {
        "big" => Ok(ByteOrder::Big),
        "little" => Ok(ByteOrder::Little),
        _ => Err(BuiltinError::ValueError(
            "byteorder must be either 'little' or 'big'".to_string(),
        )),
    }
}

fn parse_to_bytes_length(value: Value) -> Result<usize, BuiltinError> {
    let Some(length) = value_to_bigint(value).or_else(|| {
        value
            .as_bool()
            .map(|boolean| BigInt::from(u8::from(boolean)))
    }) else {
        return Err(BuiltinError::TypeError(format!(
            "'{}' object cannot be interpreted as an integer",
            value.type_name()
        )));
    };

    if length.sign() == Sign::Minus {
        return Err(BuiltinError::ValueError(
            "length argument must be non-negative".to_string(),
        ));
    }

    usize::try_from(&length)
        .map_err(|_| BuiltinError::OverflowError("int too big to convert".to_string()))
}

fn value_to_byte_sequence(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                return Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec());
            }
            _ => {}
        }
    }

    let values = iter_values(value).map_err(|_| {
        BuiltinError::TypeError(format!(
            "{context} must be a bytes-like object or iterable of integers"
        ))
    })?;

    let mut bytes = Vec::with_capacity(values.len());
    for item in values {
        bytes.push(value_to_single_byte(item, context)?);
    }
    Ok(bytes)
}

fn value_to_byte_sequence_with_vm(
    vm: &mut VirtualMachine,
    value: Value,
    context: &str,
) -> Result<Vec<u8>, BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::BYTES | TypeId::BYTEARRAY => {
                return Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec());
            }
            _ => {}
        }
    }

    let values = iter_values_with_vm(vm, value)?;
    let mut bytes = Vec::with_capacity(values.len());
    for item in values {
        bytes.push(value_to_single_byte(item, context)?);
    }
    Ok(bytes)
}

fn value_to_single_byte(value: Value, context: &str) -> Result<u8, BuiltinError> {
    if let Some(number) = value_to_bigint(value) {
        if (BigInt::from(0_u8)..=BigInt::from(u8::MAX)).contains(&number) {
            return Ok(number
                .try_into()
                .expect("validated byte-sized bigint should convert to u8"));
        }
    } else if let Some(boolean) = value.as_bool() {
        return Ok(u8::from(boolean));
    }

    Err(BuiltinError::ValueError(format!(
        "{context} must yield integers in range(0, 256)"
    )))
}

fn decode_bigint_from_bytes(bytes: &[u8], byteorder: ByteOrder, signed: bool) -> BigInt {
    match (byteorder, signed) {
        (ByteOrder::Big, true) => BigInt::from_signed_bytes_be(bytes),
        (ByteOrder::Little, true) => BigInt::from_signed_bytes_le(bytes),
        (ByteOrder::Big, false) => BigInt::from_bytes_be(Sign::Plus, bytes),
        (ByteOrder::Little, false) => BigInt::from_bytes_le(Sign::Plus, bytes),
    }
}

fn encode_bigint_to_bytes(
    value: &BigInt,
    length: usize,
    byteorder: ByteOrder,
    signed: bool,
) -> Result<Vec<u8>, BuiltinError> {
    if value.is_zero() {
        return Ok(vec![0; length]);
    }

    if !signed && value.sign() == Sign::Minus {
        return Err(BuiltinError::OverflowError(
            "can't convert negative int to unsigned".to_string(),
        ));
    }

    let mut encoded = if signed {
        match byteorder {
            ByteOrder::Big => value.to_signed_bytes_be(),
            ByteOrder::Little => value.to_signed_bytes_le(),
        }
    } else {
        match byteorder {
            ByteOrder::Big => value.to_bytes_be().1,
            ByteOrder::Little => value.to_bytes_le().1,
        }
    };

    if encoded.len() > length {
        return Err(BuiltinError::OverflowError(
            "int too big to convert".to_string(),
        ));
    }

    let padding_len = length - encoded.len();
    if padding_len == 0 {
        return Ok(encoded);
    }

    let pad = if signed && value.sign() == Sign::Minus {
        0xFF
    } else {
        0x00
    };

    match byteorder {
        ByteOrder::Big => {
            let mut padded = vec![pad; padding_len];
            padded.extend_from_slice(&encoded);
            Ok(padded)
        }
        ByteOrder::Little => {
            encoded.resize(length, pad);
            Ok(encoded)
        }
    }
}

enum FloatTextArgument {
    Str(String),
    Bytes(Vec<u8>),
}

impl FloatTextArgument {
    #[inline]
    fn raw_bytes(&self) -> &[u8] {
        match self {
            Self::Str(text) => text.as_bytes(),
            Self::Bytes(bytes) => bytes,
        }
    }

    fn invalid_literal(&self) -> BuiltinError {
        match self {
            Self::Str(text) => {
                BuiltinError::ValueError(format!("could not convert string to float: {:?}", text))
            }
            Self::Bytes(bytes) => {
                let text = String::from_utf8_lossy(bytes);
                BuiltinError::ValueError(format!("could not convert string to float: b{:?}", text))
            }
        }
    }
}

#[inline]
fn float_text_argument(value: Value) -> Option<FloatTextArgument> {
    if let Some(text) = value_to_owned_string(value) {
        return Some(FloatTextArgument::Str(text));
    }

    let ptr = value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Some(FloatTextArgument::Bytes(bytes.as_bytes().to_vec()))
        }
        _ => None,
    }
}

#[inline]
fn normalize_float_special_value(bytes: &[u8]) -> Option<f64> {
    let (negative, rest) = match bytes.first().copied() {
        Some(b'+') => (false, &bytes[1..]),
        Some(b'-') => (true, &bytes[1..]),
        _ => (false, bytes),
    };

    if rest.eq_ignore_ascii_case(b"inf") || rest.eq_ignore_ascii_case(b"infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }

    if rest.eq_ignore_ascii_case(b"nan") {
        return Some(if negative { -f64::NAN } else { f64::NAN });
    }

    None
}

fn normalize_float_literal(bytes: &[u8]) -> Option<String> {
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Prev {
        Start,
        Sign,
        Digit,
        Underscore,
        Dot,
        Exp,
    }

    let mut normalized = String::with_capacity(bytes.len());
    let mut prev = Prev::Start;
    let mut saw_mantissa_digit = false;
    let mut saw_exponent_digit = false;
    let mut saw_dot = false;
    let mut in_exponent = false;

    for (index, &byte) in bytes.iter().enumerate() {
        match byte {
            b'0'..=b'9' => {
                normalized.push(byte as char);
                if in_exponent {
                    saw_exponent_digit = true;
                } else {
                    saw_mantissa_digit = true;
                }
                prev = Prev::Digit;
            }
            b'_' => {
                if prev != Prev::Digit {
                    return None;
                }
                match bytes.get(index + 1).copied() {
                    Some(b'0'..=b'9') => prev = Prev::Underscore,
                    _ => return None,
                }
            }
            b'.' => {
                if in_exponent || saw_dot {
                    return None;
                }
                normalized.push('.');
                saw_dot = true;
                prev = Prev::Dot;
            }
            b'e' | b'E' => {
                if in_exponent || !saw_mantissa_digit {
                    return None;
                }
                normalized.push('e');
                in_exponent = true;
                saw_exponent_digit = false;
                prev = Prev::Exp;
            }
            b'+' | b'-' => {
                if index == 0 || prev == Prev::Exp {
                    normalized.push(byte as char);
                    prev = Prev::Sign;
                } else {
                    return None;
                }
            }
            _ => return None,
        }
    }

    if !saw_mantissa_digit
        || matches!(
            prev,
            Prev::Start | Prev::Sign | Prev::Underscore | Prev::Exp
        )
    {
        return None;
    }
    if in_exponent && !saw_exponent_digit {
        return None;
    }

    Some(normalized)
}

fn parse_float_text_argument(argument: &FloatTextArgument) -> Result<Value, BuiltinError> {
    let trimmed = trim_ascii_whitespace(argument.raw_bytes());
    if trimmed.is_empty() {
        return Err(argument.invalid_literal());
    }

    if let Some(value) = normalize_float_special_value(trimmed) {
        return Ok(Value::float(value));
    }

    let normalized = normalize_float_literal(trimmed).ok_or_else(|| argument.invalid_literal())?;
    normalized
        .parse::<f64>()
        .map(Value::float)
        .map_err(|_| argument.invalid_literal())
}

#[inline]
fn builtin_float_unsupported_argument(arg: Value) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "float() argument must be a string or a real number, not '{}'",
        arg.type_name()
    ))
}

#[inline]
fn builtin_float_base(arg: Value) -> Result<Option<Value>, BuiltinError> {
    if let Some(text_arg) = float_text_argument(arg) {
        return parse_float_text_argument(&text_arg).map(Some);
    }

    if let Some(f) = arg.as_float() {
        return Ok(Some(Value::float(f)));
    }
    if let Some(i) = arg.as_int() {
        return Ok(Some(Value::float(i as f64)));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Some(Value::float(if b { 1.0 } else { 0.0 })));
    }

    Ok(None)
}

#[inline]
fn float_from_index_value(value: Value) -> Result<Value, BuiltinError> {
    if let Some(boolean) = value.as_bool() {
        return Ok(Value::float(if boolean { 1.0 } else { 0.0 }));
    }
    if let Some(integer) = value.as_int() {
        return Ok(Value::float(integer as f64));
    }

    let bigint = value_to_bigint(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "__index__ returned non-int (type {})",
            value.type_name()
        ))
    })?;
    let float = bigint.to_f64().ok_or_else(|| {
        BuiltinError::OverflowError("int too large to convert to float".to_string())
    })?;
    Ok(Value::float(float))
}

/// Builtin float constructor.
pub fn builtin_float(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    builtin_float_base(args[0])?.ok_or_else(|| builtin_float_unsupported_argument(args[0]))
}

pub fn builtin_float_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::float(0.0));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "float() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let arg = args[0];
    if let Some(value) = builtin_float_base(arg)? {
        return Ok(value);
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__float__")? {
        if let Some(float) = result.as_float() {
            return Ok(Value::float(float));
        }
        return Err(BuiltinError::TypeError(format!(
            "__float__ returned non-float (type {})",
            result.type_name()
        )));
    }

    if let Some(result) = invoke_zero_arg_special_method(vm, arg, "__index__")? {
        return float_from_index_value(result);
    }

    Err(builtin_float_unsupported_argument(arg))
}

#[inline]
fn native_float_format_description() -> &'static str {
    if cfg!(target_endian = "little") {
        "IEEE, little-endian"
    } else {
        "IEEE, big-endian"
    }
}

/// Builtin implementation backing `float.__getformat__`.
pub(crate) fn builtin_float_getformat(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "float.__getformat__() takes exactly 1 argument ({} given)",
            given
        )));
    }

    let receiver = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(
            "descriptor '__getformat__' for 'float' objects doesn't apply to a non-type object"
                .to_string(),
        )
    })?;
    if crate::ops::objects::extract_type_id(receiver) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "descriptor '__getformat__' for 'float' objects doesn't apply to a non-type object"
                .to_string(),
        ));
    }

    let Some(kind) = value_to_owned_string(args[1]) else {
        return Err(BuiltinError::TypeError(format!(
            "float.__getformat__() argument 1 must be str, not {}",
            args[1].type_name()
        )));
    };

    match kind.as_str() {
        "double" | "float" => Ok(Value::string(intern(native_float_format_description()))),
        _ => Err(BuiltinError::ValueError(
            "__getformat__() argument 1 must be 'double' or 'float'".to_string(),
        )),
    }
}

/// Builtin str constructor.
pub fn builtin_str(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::string(intern("")));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "str() takes at most 3 arguments ({} given)",
            args.len()
        )));
    }

    if args.len() > 1 {
        let encoding = str_decode_text_arg(args[1], "encoding")?;
        let errors = if args.len() == 3 {
            str_decode_text_arg(args[2], "errors")?
        } else {
            "strict".to_string()
        };
        return decode_value_to_str(args[0], &encoding, &errors);
    }

    let value = args[0];
    if value.is_string() {
        return Ok(value);
    }
    if let Some(text) = crate::builtins::exception_display_text_for_value(value) {
        if text.is_empty() {
            return Ok(Value::string(intern("")));
        }
        return Ok(to_object_value(StringObject::new(&text)));
    }
    if let Some(ptr) = value.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::STR => return Ok(value),
            _ => {}
        }
    }

    super::functions::builtin_repr(&[value])
}

fn builtin_str_kw(positional: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if positional.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "str() takes at most 3 arguments ({} given)",
            positional.len()
        )));
    }

    let mut object_arg = positional.first().copied();
    let mut encoding_arg = positional.get(1).copied();
    let mut errors_arg = positional.get(2).copied();

    for &(name, value) in keywords {
        match name {
            "object" => assign_str_keyword(&mut object_arg, value, 0, positional.len(), "object")?,
            "encoding" => {
                assign_str_keyword(&mut encoding_arg, value, 1, positional.len(), "encoding")?
            }
            "errors" => assign_str_keyword(&mut errors_arg, value, 2, positional.len(), "errors")?,
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "str() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let encoding = encoding_arg
        .map(|value| str_decode_text_arg(value, "encoding"))
        .transpose()?;
    let errors = errors_arg
        .map(|value| str_decode_text_arg(value, "errors"))
        .transpose()?;

    let Some(object) = object_arg else {
        return Ok(Value::string(intern("")));
    };

    if encoding.is_some() || errors.is_some() {
        return decode_value_to_str(
            object,
            encoding.as_deref().unwrap_or("utf-8"),
            errors.as_deref().unwrap_or("strict"),
        );
    }

    builtin_str(&[object])
}

#[inline]
fn assign_str_keyword(
    slot: &mut Option<Value>,
    value: Value,
    positional_index: usize,
    positional_len: usize,
    name: &'static str,
) -> Result<(), BuiltinError> {
    if positional_len > positional_index || slot.replace(value).is_some() {
        return Err(BuiltinError::TypeError(format!(
            "str() got multiple values for argument '{}'",
            name
        )));
    }
    Ok(())
}

#[inline]
fn str_decode_text_arg(value: Value, name: &'static str) -> Result<String, BuiltinError> {
    value_to_owned_string(value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "str() argument '{}' must be str, not {}",
            name,
            value.type_name()
        ))
    })
}

#[inline]
fn decode_value_to_str(value: Value, encoding: &str, errors: &str) -> Result<Value, BuiltinError> {
    if value.is_string() {
        return Err(BuiltinError::TypeError(
            "decoding str is not supported".to_string(),
        ));
    }

    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "decoding to str: need a bytes-like object, {} found",
            value.type_name()
        ))
    })?;

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            super::decode_bytes_to_value(bytes.as_bytes(), Some(encoding), Some(errors))
        }
        TypeId::STR => Err(BuiltinError::TypeError(
            "decoding str is not supported".to_string(),
        )),
        _ => Err(BuiltinError::TypeError(format!(
            "decoding to str: need a bytes-like object, {} found",
            value.type_name()
        ))),
    }
}

/// Builtin bool constructor.
pub fn builtin_bool(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::bool(false));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bool() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    Ok(Value::bool(crate::truthiness::is_truthy(args[0])))
}

pub fn builtin_bool_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::bool(false));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "bool() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    crate::truthiness::try_is_truthy(vm, args[0])
        .map(Value::bool)
        .map_err(super::runtime_error_to_builtin_error)
}

/// Builtin list constructor.
pub fn builtin_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "list() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    if args.is_empty() {
        return Ok(to_object_value(ListObject::new()));
    }

    let values = iter_values(args[0])?;
    Ok(to_object_value(ListObject::from_iter(values)))
}

/// Builtin tuple constructor.
pub fn builtin_tuple(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "tuple() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    if args.is_empty() {
        return Ok(to_object_value(TupleObject::empty()));
    }

    let values = iter_values(args[0])?;
    Ok(to_object_value(TupleObject::from_vec(values)))
}

/// Builtin dict constructor.
pub fn builtin_dict(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "dict() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    if args.is_empty() {
        return Ok(to_object_value(DictObject::new()));
    }

    if let Some(ptr) = args[0].as_object_ptr() {
        if crate::ops::objects::extract_type_id(ptr) == TypeId::DICT {
            let source = unsafe { &*(ptr as *const DictObject) };
            let mut copy = DictObject::with_capacity(source.len());
            for (key, value) in source.iter() {
                copy.set(key, value);
            }
            return Ok(to_object_value(copy));
        }
    }

    let mut dict = DictObject::new();
    let mut sequence = if let Some(iter) = super::iter_dispatch::get_iterator_mut(&args[0]) {
        iter.collect_remaining()
    } else {
        let mut iter =
            super::iter_dispatch::value_to_iterator(&args[0]).map_err(BuiltinError::from)?;
        iter.collect_remaining()
    };

    for (index, item) in sequence.drain(..).enumerate() {
        let (key, value) = dict_item_to_pair(item, index)?;
        dict.set(key, value);
    }

    Ok(to_object_value(dict))
}

fn builtin_dict_kw(
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let dict_value = builtin_dict(positional)?;
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("builtin_dict should return an object-backed dict");
    let dict = unsafe { &mut *(dict_ptr as *mut DictObject) };

    for (name, value) in keywords {
        dict.set(Value::string(intern(name)), *value);
    }

    Ok(dict_value)
}

/// Builtin implementation backing `dict.fromkeys`.
pub(crate) fn builtin_dict_fromkeys(args: &[Value]) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dict.fromkeys() takes 1 or 2 arguments ({} given)",
            given
        )));
    }

    let class = args[0];
    let class_ptr = class.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("dict.fromkeys() descriptor requires a type receiver".to_string())
    })?;
    if builtin_type_object_type_id(class_ptr) != Some(TypeId::DICT) {
        return Err(BuiltinError::TypeError(
            "dict.fromkeys() currently requires the built-in dict type".to_string(),
        ));
    }

    let keys = iter_values(args[1])?;
    let value = args.get(2).copied().unwrap_or(Value::none());
    let mut dict = DictObject::with_capacity(keys.len());
    for key in keys {
        dict.set(key, value);
    }

    Ok(to_object_value(dict))
}

/// Builtin implementation backing `str.maketrans`.
pub(crate) fn builtin_str_maketrans(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "maketrans() takes 1, 2, or 3 arguments ({} given)",
            args.len()
        )));
    }

    if args.len() == 1 {
        let mapping_ptr = args[0].as_object_ptr().ok_or_else(|| {
            BuiltinError::TypeError(
                "if you give only one argument to maketrans it must be a dict".to_string(),
            )
        })?;
        if crate::ops::objects::extract_type_id(mapping_ptr) != TypeId::DICT {
            return Err(BuiltinError::TypeError(
                "if you give only one argument to maketrans it must be a dict".to_string(),
            ));
        }

        let mapping = unsafe { &*(mapping_ptr as *const DictObject) };
        let mut normalized = DictObject::with_capacity(mapping.len());
        for (key, value) in mapping.iter() {
            normalized.set(
                normalize_str_maketrans_key(key)?,
                normalize_str_maketrans_value(value)?,
            );
        }
        return Ok(to_object_value(normalized));
    }

    let from = value_to_owned_string(args[0]).ok_or_else(|| {
        BuiltinError::TypeError("first maketrans argument must be a string".to_string())
    })?;
    let to = value_to_owned_string(args[1]).ok_or_else(|| {
        BuiltinError::TypeError("second maketrans argument must be a string".to_string())
    })?;

    let from_chars: Vec<char> = from.chars().collect();
    let to_chars: Vec<char> = to.chars().collect();
    if from_chars.len() != to_chars.len() {
        return Err(BuiltinError::ValueError(
            "the first two maketrans arguments must have equal length".to_string(),
        ));
    }

    let delete_chars = if args.len() == 3 {
        Some(value_to_owned_string(args[2]).ok_or_else(|| {
            BuiltinError::TypeError("third maketrans argument must be a string".to_string())
        })?)
    } else {
        None
    };

    let delete_len = delete_chars
        .as_ref()
        .map(|s| s.chars().count())
        .unwrap_or_default();
    let mut table = DictObject::with_capacity(from_chars.len() + delete_len);
    for (source, replacement) in from_chars.into_iter().zip(to_chars) {
        let source_key =
            Value::int(source as i64).expect("Unicode scalar should fit in Prism integer range");
        let replacement_value = Value::int(replacement as i64)
            .expect("Unicode scalar should fit in Prism integer range");
        table.set(source_key, replacement_value);
    }

    if let Some(delete_chars) = delete_chars {
        for ch in delete_chars.chars() {
            let key =
                Value::int(ch as i64).expect("Unicode scalar should fit in Prism integer range");
            table.set(key, Value::none());
        }
    }

    Ok(to_object_value(table))
}

/// Builtin implementation backing `bytes.maketrans` and `bytearray.maketrans`.
pub(crate) fn builtin_bytes_maketrans(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "maketrans() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let from = value_to_bytes_buffer(args[0], "maketrans() argument 1")?;
    let to = value_to_bytes_buffer(args[1], "maketrans() argument 2")?;
    if from.len() != to.len() {
        return Err(BuiltinError::ValueError(
            "maketrans arguments must have same length".to_string(),
        ));
    }

    let mut table = (0_u8..=u8::MAX).collect::<Vec<_>>();
    for (source, replacement) in from.into_iter().zip(to) {
        table[source as usize] = replacement;
    }
    Ok(to_object_value(BytesObject::from_vec(table)))
}

fn value_to_bytes_buffer(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(bytes.to_vec());
    }

    if let Some(view) = value_as_memoryview_ref(value) {
        if view.released() {
            return Err(BuiltinError::ValueError(
                "operation forbidden on released memoryview object".to_string(),
            ));
        }
        return Ok(view.to_vec());
    }

    if let Some(bytes) = crate::stdlib::array::value_as_array_bytes(value)? {
        return Ok(bytes);
    }

    Err(BuiltinError::TypeError(format!(
        "{context} must be a bytes-like object, not {}",
        value.type_name()
    )))
}

/// Builtin set constructor.
pub fn builtin_set(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "set() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    if args.is_empty() {
        return Ok(to_object_value(SetObject::new()));
    }

    let values = iter_values(args[0])?;
    Ok(to_object_value(build_validated_set(values)?))
}

/// Builtin frozenset constructor.
pub fn builtin_frozenset(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "frozenset() takes at most 1 argument ({} given)",
            args.len()
        )));
    }
    if args.is_empty() {
        return Ok(to_frozenset_value(SetObject::new()));
    }

    if let Some(ptr) = args[0].as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            // CPython returns the same object for frozenset(frozenset_obj).
            TypeId::FROZENSET => return Ok(args[0]),
            TypeId::SET => {
                let source = unsafe { &*(ptr as *const SetObject) };
                return Ok(to_frozenset_value(SetObject::from_iter(source.iter())));
            }
            _ => {}
        }
    }

    let values = iter_values(args[0])?;
    Ok(to_frozenset_value(build_validated_set(values)?))
}

/// Builtin slice constructor.
pub fn builtin_slice(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "slice expected at least 1 argument, got 0".to_string(),
        ));
    }
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "slice expected at most 3 arguments, got {}",
            args.len()
        )));
    }

    let (start, stop, step) = match args.len() {
        1 => (None, value_to_slice_index(args[0])?, None),
        2 => (
            value_to_slice_index(args[0])?,
            value_to_slice_index(args[1])?,
            None,
        ),
        3 => (
            value_to_slice_index(args[0])?,
            value_to_slice_index(args[1])?,
            value_to_slice_index(args[2])?,
        ),
        _ => unreachable!(),
    };

    if step == Some(0) {
        return Err(BuiltinError::ValueError(
            "slice step cannot be zero".to_string(),
        ));
    }

    Ok(to_object_value(SliceObject::new(start, stop, step)))
}

/// Builtin type function.
pub fn builtin_type(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "type() takes 1 or 3 arguments ({} given)",
            args.len()
        )));
    }

    if args.len() == 1 {
        return Ok(value_type_object(args[0]));
    }

    let name = parse_type_name_arg(args[0])?;
    let bases = parse_type_bases_arg(args[1])?;
    let namespace = parse_type_namespace_arg(args[2])?;

    let result = type_new(name, &bases, &namespace, global_class_registry())
        .map_err(|err| BuiltinError::TypeError(format!("type() could not build class: {}", err)))?;
    register_global_class(result.class.clone(), result.bitmap);

    Ok(Value::object_ptr(
        std::sync::Arc::into_raw(result.class) as *const ()
    ))
}

pub fn builtin_type_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "type() takes 1 or 3 arguments ({} given)",
            args.len()
        )));
    }

    if args.len() == 1 {
        return Ok(value_type_object(args[0]));
    }

    let name = parse_type_name_arg(args[0])?;
    let bases = parse_type_bases_arg(args[1])?;
    let namespace = parse_type_namespace_arg(args[2])?;

    let result = type_new(name, &bases, &namespace, global_class_registry())
        .map_err(|err| BuiltinError::TypeError(format!("type() could not build class: {}", err)))?;
    let class_value = Value::object_ptr(std::sync::Arc::as_ptr(&result.class) as *const ());
    let class_id = result.class.class_id();
    register_global_class(result.class.clone(), result.bitmap);
    if let Err(err) =
        crate::ops::class::invoke_descriptor_set_name_hooks(vm, class_value, &namespace)
    {
        unregister_global_class(class_id);
        return Err(runtime_error_to_builtin_error(err));
    }
    if let Err(err) = crate::ops::class::invoke_init_subclass_hook(vm, class_value, &[]) {
        unregister_global_class(class_id);
        return Err(runtime_error_to_builtin_error(err));
    }

    Ok(Value::object_ptr(
        std::sync::Arc::into_raw(result.class) as *const ()
    ))
}

pub(crate) fn builtin_type_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "type.__new__() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }

    let metaclass = args[0];
    let metaclass_ptr = metaclass
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("type.__new__(X): X must be a type".to_string()))?;
    if crate::ops::objects::extract_type_id(metaclass_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "type.__new__(X): X must be a type".to_string(),
        ));
    }

    let name = parse_type_name_arg(args[1])?;
    let bases = parse_type_bases_arg(args[2])?;
    let namespace = parse_type_namespace_arg(args[3])?;

    let result =
        type_new_with_metaclass(name, &bases, &namespace, metaclass, global_class_registry())
            .map_err(|err| {
                BuiltinError::TypeError(format!("type.__new__() could not build class: {}", err))
            })?;
    register_global_class(result.class.clone(), result.bitmap);

    Ok(Value::object_ptr(
        std::sync::Arc::into_raw(result.class) as *const ()
    ))
}

pub(crate) fn builtin_type_new_with_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "type.__new__() takes exactly 4 arguments ({} given)",
            args.len()
        )));
    }

    let metaclass = args[0];
    let metaclass_ptr = metaclass
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("type.__new__(X): X must be a type".to_string()))?;
    if crate::ops::objects::extract_type_id(metaclass_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "type.__new__(X): X must be a type".to_string(),
        ));
    }

    let name = parse_type_name_arg(args[1])?;
    let bases = parse_type_bases_arg(args[2])?;
    let namespace = parse_type_namespace_arg(args[3])?;

    let result =
        type_new_with_metaclass(name, &bases, &namespace, metaclass, global_class_registry())
            .map_err(|err| {
                BuiltinError::TypeError(format!("type.__new__() could not build class: {}", err))
            })?;
    let class_value = Value::object_ptr(std::sync::Arc::as_ptr(&result.class) as *const ());
    let class_id = result.class.class_id();
    register_global_class(result.class.clone(), result.bitmap);
    if let Err(err) =
        crate::ops::class::invoke_descriptor_set_name_hooks(vm, class_value, &namespace)
    {
        unregister_global_class(class_id);
        return Err(runtime_error_to_builtin_error(err));
    }
    if let Err(err) = crate::ops::class::invoke_init_subclass_hook(vm, class_value, &[]) {
        unregister_global_class(class_id);
        return Err(runtime_error_to_builtin_error(err));
    }

    Ok(Value::object_ptr(
        std::sync::Arc::into_raw(result.class) as *const ()
    ))
}

/// Builtin isinstance function.
pub fn builtin_isinstance(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "isinstance() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    let targets = parse_class_spec_values(args[1], "isinstance")?;
    for target in targets {
        if raw_isinstance_value(args[0], target)? {
            return Ok(Value::bool(true));
        }
    }
    Ok(Value::bool(false))
}

/// VM-aware isinstance builtin that honors metaclass __instancecheck__.
pub fn builtin_isinstance_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "isinstance() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let targets = parse_class_spec_values(args[1], "isinstance")?;
    for target in targets {
        if exact_class_match(args[0], target) {
            return Ok(Value::bool(true));
        }
        if let Some(result) = invoke_metaclass_check(vm, target, "__instancecheck__", args[0])? {
            if result {
                return Ok(Value::bool(true));
            }
            continue;
        }
        if raw_isinstance_value(args[0], target)? {
            return Ok(Value::bool(true));
        }
    }

    Ok(Value::bool(false))
}

/// Builtin issubclass function.
pub fn builtin_issubclass(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "issubclass() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    let targets = parse_class_spec_values(args[1], "issubclass")?;
    for target in targets {
        if raw_issubclass_value(args[0], target)? {
            return Ok(Value::bool(true));
        }
    }
    Ok(Value::bool(false))
}

/// VM-aware issubclass builtin that honors metaclass __subclasscheck__.
pub fn builtin_issubclass_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "issubclass() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    if class_value_to_type_id(args[0]).is_none() {
        return Err(BuiltinError::TypeError(
            "issubclass() arg 1 must be a class".to_string(),
        ));
    }

    let targets = parse_class_spec_values(args[1], "issubclass")?;
    for target in targets {
        if is_exact_type_target(target) && raw_issubclass_value(args[0], target)? {
            return Ok(Value::bool(true));
        }
        if let Some(result) = invoke_metaclass_check(vm, target, "__subclasscheck__", args[0])? {
            if result {
                return Ok(Value::bool(true));
            }
            continue;
        }
        if raw_issubclass_value(args[0], target)? {
            return Ok(Value::bool(true));
        }
    }

    Ok(Value::bool(false))
}

/// Builtin object constructor.
pub fn builtin_object(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "object() takes no arguments ({} given)",
            args.len()
        )));
    }

    let object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    Ok(to_object_value(object))
}

#[inline]
fn class_uses_native_storage(class: &PyClassObject, builtin_type: TypeId) -> bool {
    global_class_bitmap(class.class_id()).is_some_and(|bitmap| bitmap.is_subclass_of(builtin_type))
        || class
            .mro()
            .iter()
            .copied()
            .any(|class_id| class_id == ClassId(builtin_type.raw()))
}

#[inline]
fn class_uses_native_dict_storage(class: &PyClassObject) -> bool {
    class_uses_native_storage(class, TypeId::DICT)
}

#[inline]
fn class_uses_native_list_storage(class: &PyClassObject) -> bool {
    class_uses_native_storage(class, TypeId::LIST)
}

#[inline]
fn class_uses_native_tuple_storage(class: &PyClassObject) -> bool {
    class_uses_native_storage(class, TypeId::TUPLE)
}

#[inline]
fn class_uses_native_int_storage(class: &PyClassObject) -> bool {
    class_uses_native_storage(class, TypeId::INT)
}

#[inline]
fn class_uses_native_bytes_storage(class: &PyClassObject) -> bool {
    class_uses_native_storage(class, TypeId::BYTES)
        || class_uses_native_storage(class, TypeId::BYTEARRAY)
}

#[inline]
pub(crate) fn allocate_heap_instance_for_class(class: &PyClassObject) -> ShapedObject {
    if class_uses_native_dict_storage(class) {
        ShapedObject::new_dict_backed(class.class_type_id(), class.instance_shape().clone())
    } else if class_uses_native_list_storage(class) {
        ShapedObject::new_list_backed(class.class_type_id(), class.instance_shape().clone())
    } else if class_uses_native_tuple_storage(class) {
        ShapedObject::new_tuple_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
            TupleObject::empty(),
        )
    } else if class_uses_native_bytes_storage(class) {
        ShapedObject::new_bytes_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
            BytesObject::new(),
        )
    } else if class_uses_native_int_storage(class) {
        ShapedObject::new_int_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
            BigInt::zero(),
        )
    } else {
        ShapedObject::new(class.class_type_id(), class.instance_shape().clone())
    }
}

#[inline]
fn builtin_constructor_new(
    args: &[Value],
    target: TypeId,
    type_name: &'static str,
    constructor: fn(&[Value]) -> Result<Value, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{type_name}.__new__() takes at least 1 argument (0 given)"
        )));
    }

    let class_type = class_value_to_type_id(args[0]).ok_or_else(|| {
        BuiltinError::TypeError(format!("{type_name}.__new__(X): X must be a type"))
    })?;

    if class_type == target {
        return constructor(&args[1..]);
    }

    if class_value_is_subtype(args[0], target) {
        if let Some(err) = builtin_constructor_subtype_error(target, class_type) {
            return Err(err);
        }
        return Err(BuiltinError::NotImplemented(format!(
            "{type_name}.__new__() for {type_name} subclasses is not implemented yet"
        )));
    }

    Err(BuiltinError::TypeError(format!(
        "{type_name}.__new__(X): X is not a subtype of {type_name}"
    )))
}

#[inline]
fn builtin_constructor_subtype_error(target: TypeId, receiver: TypeId) -> Option<BuiltinError> {
    match (target, receiver) {
        (TypeId::INT, TypeId::BOOL) => Some(BuiltinError::TypeError(
            "int.__new__(bool) is not safe, use bool.__new__()".to_string(),
        )),
        _ => None,
    }
}

pub(crate) fn builtin_int_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_int_new_impl(None, args)
}

pub(crate) fn builtin_int_new_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    builtin_int_new_impl(Some(vm), args)
}

fn builtin_int_new_impl(
    vm: Option<&mut VirtualMachine>,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "int.__new__() takes at least 1 argument (0 given)".to_string(),
        ));
    }

    let class_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("int.__new__(X): X must be a type".to_string()))?;

    if class_type == TypeId::INT {
        return builtin_int(&args[1..]);
    }

    if !class_value_is_subtype(args[0], TypeId::INT) {
        return Err(BuiltinError::TypeError(
            "int.__new__(X): X is not a subtype of int".to_string(),
        ));
    }

    if let Some(err) = builtin_constructor_subtype_error(TypeId::INT, class_type) {
        return Err(err);
    }

    let class_ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("int.__new__(X): X must be a type".to_string()))?;
    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "int.__new__() for builtin int subclasses is unsupported".to_string(),
        )
    })?;

    let integer = if args.len() == 1 {
        BigInt::zero()
    } else {
        let value = match vm {
            Some(vm) => builtin_int_vm(vm, &args[1..])?,
            None => builtin_int(&args[1..])?,
        };
        value_to_bigint(value).ok_or_else(|| {
            BuiltinError::TypeError(
                "int.__new__() failed to materialize integer payload".to_string(),
            )
        })?
    };

    let instance = ShapedObject::new_int_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
        integer,
    );
    Ok(to_object_value(instance))
}

pub(crate) fn builtin_float_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::FLOAT, "float", builtin_float)
}

pub(crate) fn builtin_str_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "str.__new__() takes at least 1 argument (0 given)".to_string(),
        ));
    }

    let class_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("str.__new__(X): X must be a type".to_string()))?;

    if class_type == TypeId::STR {
        return builtin_str(&args[1..]);
    }

    if !class_value_is_subtype(args[0], TypeId::STR) {
        return Err(BuiltinError::TypeError(
            "str.__new__(X): X is not a subtype of str".to_string(),
        ));
    }

    let class_ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("str.__new__(X): X must be a type".to_string()))?;
    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "str.__new__() for builtin str subclasses is unsupported".to_string(),
        )
    })?;

    let string_value = if args.len() == 1 {
        Value::string(intern(""))
    } else {
        builtin_str(&args[1..])?
    };
    let string_object = clone_string_value(string_value).ok_or_else(|| {
        BuiltinError::TypeError("str.__new__() failed to materialize string payload".to_string())
    })?;

    let instance = ShapedObject::new_string_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
        string_object,
    );
    Ok(to_object_value(instance))
}

pub(crate) fn builtin_bool_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::BOOL, "bool", builtin_bool)
}

pub(crate) fn builtin_bytes_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_byte_sequence_new(args, TypeId::BYTES, "bytes", super::string::builtin_bytes)
}

pub(crate) fn builtin_bytearray_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_byte_sequence_new(
        args,
        TypeId::BYTEARRAY,
        "bytearray",
        super::string::builtin_bytearray,
    )
}

fn builtin_byte_sequence_new(
    args: &[Value],
    target: TypeId,
    type_name: &'static str,
    constructor: fn(&[Value]) -> Result<Value, BuiltinError>,
) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{type_name}.__new__() takes at least 1 argument (0 given)"
        )));
    }

    let receiver = args[0];
    let class_type = class_value_to_type_id(receiver).ok_or_else(|| {
        BuiltinError::TypeError(format!("{type_name}.__new__(X): X must be a type"))
    })?;

    if class_type == target {
        return constructor(&args[1..]);
    }

    if !class_value_is_subtype(receiver, target) {
        return Err(BuiltinError::TypeError(format!(
            "{type_name}.__new__(X): X is not a subtype of {type_name}"
        )));
    }

    let class_ptr = receiver.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!("{type_name}.__new__(X): X must be a type"))
    })?;
    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{type_name}.__new__() for builtin {type_name} subclasses is unsupported"
        ))
    })?;

    let bytes_value = constructor(&args[1..])?;
    let bytes_object = clone_bytes_value(bytes_value).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "{type_name}.__new__() failed to materialize byte payload"
        ))
    })?;

    let instance = ShapedObject::new_bytes_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
        bytes_object,
    );
    Ok(to_object_value(instance))
}

pub(crate) fn builtin_list_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::LIST, "list", builtin_list)
}

pub(crate) fn builtin_dict_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::DICT, "dict", builtin_dict)
}

pub(crate) fn builtin_set_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::SET, "set", builtin_set)
}

pub(crate) fn builtin_frozenset_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::FROZENSET, "frozenset", builtin_frozenset)
}

pub(crate) fn builtin_module_new(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_constructor_new(args, TypeId::MODULE, "module", builtin_module)
}

pub(crate) fn builtin_object_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "object.__new__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let class_value = args[0];
    let class_ptr = class_value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("object.__new__(X): X must be a type".to_string())
    })?;

    if let Some(type_id) = builtin_type_object_type_id(class_ptr) {
        return match type_id {
            TypeId::OBJECT => builtin_object(&[]),
            _ => Err(BuiltinError::NotImplemented(format!(
                "object.__new__() for builtin type {:?} is not implemented yet",
                type_id
            ))),
        };
    }

    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError("object.__new__(X): X must be a type".to_string())
    })?;
    let instance = allocate_heap_instance_for_class(class);
    Ok(to_object_value(instance))
}

pub(crate) fn builtin_object_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(
            "object.__init__() takes exactly one argument (the instance to initialize)".to_string(),
        ));
    }

    let self_ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires an object instance".to_string())
    })?;

    if crate::ops::objects::extract_type_id(self_ptr) == TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "object.__init__() is not valid for type instances".to_string(),
        ));
    }

    Ok(Value::none())
}

pub(crate) fn builtin_object_init_subclass(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(
            "object.__init_subclass__() takes no arguments".to_string(),
        ));
    }

    if class_value_to_type_id(args[0]).is_none() {
        return Err(BuiltinError::TypeError(
            "descriptor '__init_subclass__' requires a type object".to_string(),
        ));
    }

    if !keywords.is_empty() {
        return Err(BuiltinError::TypeError(
            "object.__init_subclass__() takes no keyword arguments".to_string(),
        ));
    }

    Ok(Value::none())
}

pub(crate) fn builtin_type_init(args: &[Value]) -> Result<Value, BuiltinError> {
    let explicit_argc = args.len().saturating_sub(1);
    if !matches!(args.len(), 1 | 4) {
        return Err(BuiltinError::TypeError(format!(
            "type.__init__() takes 1 or 3 arguments ({} given)",
            explicit_argc
        )));
    }

    let self_ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires a type object".to_string())
    })?;
    if crate::ops::objects::extract_type_id(self_ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "descriptor '__init__' requires a type object".to_string(),
        ));
    }

    if args.len() == 4 {
        parse_type_name_arg(args[1])?;
        parse_type_bases_arg(args[2])?;
        parse_type_namespace_arg(args[3])?;
    }

    Ok(Value::none())
}

pub(crate) fn builtin_tuple_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "tuple.__new__() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let class_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("tuple.__new__(X): X must be a type".to_string()))?;

    if class_type == TypeId::TUPLE {
        return builtin_tuple(&args[1..]);
    }

    if class_value_is_subtype(args[0], TypeId::TUPLE) {
        let class_ptr = args[0].as_object_ptr().ok_or_else(|| {
            BuiltinError::TypeError("tuple.__new__(X): X must be a type".to_string())
        })?;
        let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
            BuiltinError::TypeError(
                "tuple.__new__() for builtin tuple subclasses is unsupported".to_string(),
            )
        })?;
        let values = if args.len() == 1 {
            Vec::new()
        } else {
            iter_values(args[1])?
        };
        let instance = ShapedObject::new_tuple_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
            TupleObject::from_vec(values),
        );
        return Ok(to_object_value(instance));
    }

    Err(BuiltinError::TypeError(
        "tuple.__new__(X): X is not a subtype of tuple".to_string(),
    ))
}

pub fn builtin_memoryview(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "memoryview() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let (bytes, format, readonly, shape) = memoryview_export(args[0])?;
    Ok(to_object_value(MemoryViewObject::from_vec_with_shape(
        args[0], bytes, format, readonly, shape,
    )))
}

pub(crate) fn builtin_memoryview_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "memoryview.__new__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }
    let class_type = class_value_to_type_id(args[0]).ok_or_else(|| {
        BuiltinError::TypeError("memoryview.__new__(X): X must be a type".to_string())
    })?;
    if class_type != TypeId::MEMORYVIEW {
        return Err(BuiltinError::TypeError(
            "memoryview.__new__(X): X is not a subtype of memoryview".to_string(),
        ));
    }
    builtin_memoryview(&args[1..])
}

const ARRAY_BYTES_ATTR: &str = "__prism_array_bytes__";
const ARRAY_TYPECODE_ATTR: &str = "__prism_array_typecode__";

fn memoryview_export(
    value: Value,
) -> Result<(Vec<u8>, MemoryViewFormat, bool, Vec<usize>), BuiltinError> {
    if let Some(view) = value_as_memoryview_ref(value) {
        ensure_memoryview_not_released(view)?;
        return Ok((
            view.to_vec(),
            view.format(),
            view.readonly(),
            view.shape().to_vec(),
        ));
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        let readonly = !bytes.is_bytearray();
        return Ok((
            bytes.to_vec(),
            MemoryViewFormat::UnsignedByte,
            readonly,
            vec![bytes.len()],
        ));
    }

    if let Some((bytes, format)) = array_memoryview_export(value)? {
        let elements = bytes.len() / format.item_size();
        return Ok((bytes, format, false, vec![elements]));
    }

    Err(BuiltinError::TypeError(format!(
        "memoryview: a bytes-like object is required, not '{}'",
        value.type_name()
    )))
}

fn array_memoryview_export(
    value: Value,
) -> Result<Option<(Vec<u8>, MemoryViewFormat)>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };
    if crate::ops::objects::extract_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }

    let object = unsafe { &*(ptr as *const ShapedObject) };
    let Some(bytes_value) = object.get_property(ARRAY_BYTES_ATTR) else {
        return Ok(None);
    };
    let Some(bytes) = value_as_bytes_ref(bytes_value) else {
        return Err(BuiltinError::TypeError(
            "invalid array buffer storage".to_string(),
        ));
    };
    let format = object
        .get_property(ARRAY_TYPECODE_ATTR)
        .and_then(value_to_owned_string)
        .and_then(|typecode| MemoryViewFormat::parse(&typecode))
        .unwrap_or(MemoryViewFormat::UnsignedByte);
    Ok(Some((bytes.to_vec(), format)))
}

fn ensure_memoryview_not_released(view: &MemoryViewObject) -> Result<(), BuiltinError> {
    if view.released() {
        Err(BuiltinError::ValueError(
            "operation forbidden on released memoryview object".to_string(),
        ))
    } else {
        Ok(())
    }
}

/// Builtin getattr(object, name[, default]) function.
///
/// Returns the value of the named attribute of object.
/// If the named attribute does not exist, default is returned if provided,
/// otherwise AttributeError is raised.
///
/// # Python Semantics
/// - `getattr(x, 'name')` → x.name
/// - `getattr(x, 'name', default)` → x.name if exists, else default
/// - Raises AttributeError if attribute not found and no default
pub fn builtin_getattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "getattr() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;
    let default = args.get(2).copied();

    // Try to get the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::OBJECT => {
                let shaped = unsafe { &*(ptr as *const ShapedObject) };
                if let Some(value) = shaped.get_property_interned(&name) {
                    return Ok(value);
                }
            }
            type_id @ (TypeId::FUNCTION | TypeId::CLOSURE) => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                if let Some(value) = crate::ops::objects::function_attr_value(func, &name) {
                    return Ok(value);
                }
                if let Some(value) =
                    crate::ops::objects::builtin_instance_method_attr_value(obj, type_id, &name)
                {
                    return Ok(value);
                }
            }
            TypeId::BUILTIN_FUNCTION => {
                let builtin = unsafe { &*(ptr as *const crate::builtins::BuiltinFunctionObject) };
                if let Some(value) =
                    crate::ops::objects::builtin_function_attr_value(builtin, &name)
                {
                    return Ok(value);
                }
            }
            type_id if crate::ops::objects::builtin_instance_method_attr_exists(type_id, &name) => {
                if let Some(value) =
                    crate::ops::objects::builtin_instance_method_attr_value(obj, type_id, &name)
                {
                    return Ok(value);
                }
            }
            TypeId::TYPE => {
                if let Some(represented) = builtin_type_object_type_id(ptr) {
                    if let Some(value) =
                        builtin_bound_type_attribute_value_static(represented, obj, &name)
                            .map_err(|err| BuiltinError::AttributeError(err.to_string()))?
                    {
                        return Ok(value);
                    }
                }

                if let Some(class) = class_object_from_ptr(ptr) {
                    if let Some(value) =
                        heap_type_attribute_value_static(ptr as *const PyClassObject, &name)
                            .map_err(|err| BuiltinError::AttributeError(err.to_string()))?
                    {
                        return Ok(value);
                    }
                    if let Some(value) =
                        crate::ops::objects::lookup_class_metaclass_attr(class, &name)
                    {
                        return Ok(crate::ops::objects::bind_instance_attribute(value, obj));
                    }
                }
            }
            TypeId::SUPER => {
                if let Some(value) = crate::ops::objects::super_attribute_value_static(obj, &name)
                    .map_err(|err| BuiltinError::AttributeError(err.to_string()))?
                {
                    return Ok(value);
                }
            }
            TypeId::EXCEPTION_TYPE => {
                let exception_type =
                    unsafe { &*(ptr as *const crate::builtins::ExceptionTypeObject) };
                if let Some(value) =
                    crate::builtins::exception_type_attribute_value(exception_type, &name)
                {
                    return Ok(value);
                }
            }
            TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY => {
                if let Some(value) = crate::ops::objects::descriptor_attr_value(obj, &name) {
                    return Ok(value);
                }
            }
            _ => {}
        }
    }

    // Attribute not found - return default or raise error
    match default {
        Some(d) => Ok(d),
        None => Err(BuiltinError::AttributeError(format!(
            "'{}' object has no attribute '{}'",
            get_type_name(obj),
            name.as_str()
        ))),
    }
}

/// VM-aware getattr(object, name[, default]) implementation.
pub fn builtin_getattr_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "getattr() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;
    let default = args.get(2).copied();

    match crate::ops::objects::get_attribute_value(vm, obj, &name) {
        Ok(value) => Ok(value),
        Err(err) => match runtime_error_to_builtin_error(err) {
            BuiltinError::AttributeError(_) => default.ok_or_else(|| {
                BuiltinError::AttributeError(format!(
                    "'{}' object has no attribute '{}'",
                    get_type_name(obj),
                    name.as_str()
                ))
            }),
            other => Err(other),
        },
    }
}

/// Builtin setattr(object, name, value) function.
///
/// Sets the value of the named attribute of object.
///
/// # Python Semantics
/// - `setattr(x, 'name', value)` → x.name = value
/// - Raises TypeError if the object doesn't support attribute assignment
pub fn builtin_setattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "setattr() takes 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;
    let value = args[2];

    // Try to set the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = crate::ops::objects::extract_type_id(ptr);
        match type_id {
            TypeId::OBJECT => {
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                let registry = shape_registry();
                shaped.set_property(name.clone(), value, registry);
                return Ok(Value::none());
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                func.set_attr(name.clone(), value);
                return Ok(Value::none());
            }
            TypeId::TYPE => {
                if let Some(class) = class_object_from_ptr(ptr) {
                    class.set_attr(name.clone(), value);
                    return Ok(Value::none());
                }
            }
            _ => {}
        }

        return Err(BuiltinError::TypeError(format!(
            "'{}' object has no attribute '{}'",
            type_id.name(),
            name.as_str()
        )));
    }

    // Primitive types don't support setattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name.as_str()
    )))
}

/// VM-aware setattr(object, name, value) implementation.
pub fn builtin_setattr_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "setattr() takes 3 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;
    let value = args[2];

    crate::ops::objects::set_attribute_value(vm, obj, &name, value)
        .map(|_| Value::none())
        .map_err(runtime_error_to_builtin_error)
}

/// Builtin hasattr(object, name) function.
///
/// Returns True if the object has the named attribute, False otherwise.
///
/// # Python Semantics
/// - `hasattr(x, 'name')` → True if x.name exists
/// - Implemented by calling getattr and checking for AttributeError
pub fn builtin_hasattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "hasattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;

    // Check if the attribute exists
    if let Some(ptr) = obj.as_object_ptr() {
        match crate::ops::objects::extract_type_id(ptr) {
            TypeId::OBJECT => {
                let shaped = unsafe { &*(ptr as *const ShapedObject) };
                return Ok(Value::bool(
                    shaped.has_property_interned(&name)
                        || builtin_instance_has_attribute(TypeId::OBJECT, &name),
                ));
            }
            type_id @ (TypeId::FUNCTION | TypeId::CLOSURE) => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                return Ok(Value::bool(
                    crate::ops::objects::function_attr_exists(func, &name)
                        || crate::ops::objects::builtin_instance_method_attr_exists(type_id, &name),
                ));
            }
            type_id if crate::ops::objects::builtin_instance_method_attr_exists(type_id, &name) => {
                return Ok(Value::bool(true));
            }
            TypeId::TYPE => {
                if let Some(represented) = builtin_type_object_type_id(ptr) {
                    return Ok(Value::bool(builtin_type_has_attribute(represented, &name)));
                }
                if let Some(class) = class_object_from_ptr(ptr) {
                    return Ok(Value::bool(
                        heap_type_has_attribute(ptr as *const PyClassObject, &name)
                            || crate::ops::objects::lookup_class_metaclass_attr(class, &name)
                                .is_some(),
                    ));
                }
            }
            TypeId::SUPER => {
                return Ok(Value::bool(crate::ops::objects::super_attribute_exists(
                    obj, &name,
                )));
            }
            TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY => {
                return Ok(Value::bool(
                    crate::ops::objects::descriptor_attr_value(obj, &name).is_some(),
                ));
            }
            _ => {}
        }
    }

    // For other types, always return False (no custom attributes)
    Ok(Value::bool(false))
}

/// VM-aware hasattr(object, name) implementation.
pub fn builtin_hasattr_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "hasattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;

    match crate::ops::objects::get_attribute_value(vm, obj, &name) {
        Ok(_) => Ok(Value::bool(true)),
        Err(err) => match runtime_error_to_builtin_error(err) {
            BuiltinError::AttributeError(_) => Ok(Value::bool(false)),
            other => Err(other),
        },
    }
}

/// Builtin delattr(object, name) function.
///
/// Deletes the named attribute from the object.
///
/// # Python Semantics
/// - `delattr(x, 'name')` → del x.name
/// - Raises AttributeError if attribute doesn't exist
pub fn builtin_delattr(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "delattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;

    // Try to delete the attribute
    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = crate::ops::objects::extract_type_id(ptr);
        match type_id {
            TypeId::OBJECT => {
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                if shaped.delete_property_interned(&name) {
                    return Ok(Value::none());
                }

                return Err(BuiltinError::AttributeError(format!(
                    "'object' object has no attribute '{}'",
                    name.as_str()
                )));
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                if func.del_attr(&name).is_some() {
                    return Ok(Value::none());
                }

                return Err(BuiltinError::AttributeError(format!(
                    "'function' object has no attribute '{}'",
                    name.as_str()
                )));
            }
            TypeId::TYPE => {
                if let Some(class) = class_object_from_ptr(ptr) {
                    if class.del_attr(&name).is_some() {
                        return Ok(Value::none());
                    }

                    return Err(BuiltinError::AttributeError(format!(
                        "'type' object has no attribute '{}'",
                        name.as_str()
                    )));
                }
            }
            _ => {}
        }

        return Err(BuiltinError::TypeError(format!(
            "'{}' object has no attribute '{}'",
            type_id.name(),
            name.as_str()
        )));
    }

    // Primitive types don't support delattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name.as_str()
    )))
}

/// VM-aware delattr(object, name) implementation.
pub fn builtin_delattr_vm(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "delattr() takes 2 arguments ({} given)",
            args.len()
        )));
    }

    let obj = args[0];
    let name = attribute_name(args[1])?;

    crate::ops::objects::delete_attribute_value(vm, obj, &name)
        .map(|_| Value::none())
        .map_err(runtime_error_to_builtin_error)
}

/// Helper to get the type name of a value.
fn get_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.is_string() {
        "str"
    } else if let Some(ptr) = value.as_object_ptr() {
        use prism_runtime::object::ObjectHeader;
        let type_id = unsafe { (*(ptr as *const ObjectHeader)).type_id };
        type_id.name()
    } else {
        "unknown"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::itertools::{builtin_iter, builtin_next, builtin_range};
    use crate::stdlib::exceptions::ExceptionTypeId;
    use prism_core::intern::intern;
    use prism_core::intern::interned_by_ptr;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::type_builtins::builtin_class_mro;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::iter::IteratorObject;
    use prism_runtime::types::string::StringObject;
    use std::sync::Arc;

    fn value_to_string(value: Value) -> String {
        if value.is_string() {
            let ptr = value
                .as_string_object_ptr()
                .expect("tagged string should provide pointer");
            return interned_by_ptr(ptr as *const u8)
                .expect("interned pointer should resolve")
                .as_str()
                .to_string();
        }

        let ptr = value
            .as_object_ptr()
            .expect("string value should be object-backed");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::STR);
        let s = unsafe { &*(ptr as *const StringObject) };
        s.as_str().to_string()
    }

    fn value_to_bytes(value: Value) -> Vec<u8> {
        let ptr = value
            .as_object_ptr()
            .expect("bytes result should be heap allocated");
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        bytes.to_vec()
    }

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn class_value(class: PyClassObject) -> (Value, *const PyClassObject) {
        let class = Arc::new(class);
        let ptr = Arc::into_raw(class);
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_class(ptr: *const PyClassObject) {
        drop(unsafe { Arc::from_raw(ptr) });
    }

    fn namespace_builtin(
        namespace: &mut DictObject,
        name: &str,
        func: fn(&[Value]) -> Result<Value, BuiltinError>,
    ) -> *mut crate::builtins::BuiltinFunctionObject {
        let builtin = Box::new(crate::builtins::BuiltinFunctionObject::new(
            Arc::from(name),
            func,
        ));
        let ptr = Box::into_raw(builtin);
        namespace.set(
            Value::string(intern(name)),
            Value::object_ptr(ptr as *const ()),
        );
        ptr
    }

    fn heap_class(
        name: &str,
    ) -> (
        Value,
        *const PyClassObject,
        *mut TupleObject,
        *mut DictObject,
    ) {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let class_value =
            builtin_type(&[Value::string(intern(name)), bases_value, namespace_value])
                .expect("type() should build a heap class");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("heap class should be object-backed")
            as *const PyClassObject;
        (
            class_value,
            class_ptr,
            bases_ptr,
            namespace_ptr as *mut DictObject,
        )
    }

    fn heap_class_with_metaclass(
        metaclass: Value,
        name: &str,
    ) -> (
        Value,
        *const PyClassObject,
        *mut TupleObject,
        *mut DictObject,
    ) {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let class_value = builtin_type_new(&[
            metaclass,
            Value::string(intern(name)),
            bases_value,
            namespace_value,
        ])
        .expect("type.__new__ should build a heap class");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("heap class should be object-backed")
            as *const PyClassObject;
        (
            class_value,
            class_ptr,
            bases_ptr,
            namespace_ptr as *mut DictObject,
        )
    }

    fn heap_metaclass_with_hook(
        name: &str,
        hook_name: &str,
        hook: fn(&[Value]) -> Result<Value, BuiltinError>,
    ) -> (
        Value,
        *const PyClassObject,
        *mut TupleObject,
        *mut DictObject,
        *mut crate::builtins::BuiltinFunctionObject,
    ) {
        let type_type = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[type_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let namespace = unsafe { &mut *namespace_ptr };
        let hook_ptr = namespace_builtin(namespace, hook_name, hook);
        let metaclass = builtin_type(&[Value::string(intern(name)), bases_value, namespace_value])
            .expect("type() should build a heap metaclass");
        let metaclass_ptr = metaclass
            .as_object_ptr()
            .expect("heap metaclass should be object-backed")
            as *const PyClassObject;
        (
            metaclass,
            metaclass_ptr,
            bases_ptr,
            namespace_ptr as *mut DictObject,
            hook_ptr,
        )
    }

    #[test]
    fn test_object_new_allocates_native_dict_backing_for_heap_dict_subclass() {
        let class = PyClassObject::new(
            intern("DictSubclass"),
            &[ClassId(TypeId::DICT.raw())],
            |id| {
                (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                    builtin_class_mro(TypeId::from_raw(id.0))
                        .into_iter()
                        .collect()
                })
            },
        )
        .expect("dict subclass should build");
        let (class_value, class_ptr) = class_value(class);

        let result = builtin_object_new(&[class_value]).expect("object.__new__ should succeed");
        let result_ptr = result
            .as_object_ptr()
            .expect("object.__new__ should return a heap instance");
        let shaped = unsafe { &*(result_ptr as *const ShapedObject) };

        assert!(shaped.has_dict_backing());

        unsafe {
            drop_boxed(result_ptr as *mut ShapedObject);
            drop_class(class_ptr);
        }
    }

    #[test]
    fn test_object_new_allocates_native_list_backing_for_heap_list_subclass() {
        let class = PyClassObject::new(
            intern("ListSubclass"),
            &[ClassId(TypeId::LIST.raw())],
            |id| {
                (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                    builtin_class_mro(TypeId::from_raw(id.0))
                        .into_iter()
                        .collect()
                })
            },
        )
        .expect("list subclass should build");
        let (class_value, class_ptr) = class_value(class);

        let result = builtin_object_new(&[class_value]).expect("object.__new__ should succeed");
        let result_ptr = result
            .as_object_ptr()
            .expect("object.__new__ should return a heap instance");
        let shaped = unsafe { &*(result_ptr as *const ShapedObject) };

        assert!(shaped.has_list_backing());

        unsafe {
            drop_boxed(result_ptr as *mut ShapedObject);
            drop_class(class_ptr);
        }
    }

    #[test]
    fn test_int_from_int() {
        let result = builtin_int(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_int(), Some(42));
    }

    #[test]
    fn test_int_from_float() {
        let result = builtin_int(&[Value::float(3.9)]).unwrap();
        assert_eq!(result.as_int(), Some(3));
    }

    #[test]
    fn test_int_from_ascii_string_default_base() {
        let result = builtin_int(&[Value::string(intern("42"))]).unwrap();
        assert_eq!(result.as_int(), Some(42));

        let signed = builtin_int(&[Value::string(intern("  -17 "))]).unwrap();
        assert_eq!(signed.as_int(), Some(-17));
    }

    #[test]
    fn test_int_from_bytes_and_bytearray_default_base() {
        let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"01"));
        let bytes_result = builtin_int(&[bytes_value]).unwrap();
        assert_eq!(bytes_result.as_int(), Some(1));

        let (bytearray_value, bytearray_ptr) =
            boxed_value(BytesObject::bytearray_from_slice(b"255"));
        let bytearray_result = builtin_int(&[bytearray_value]).unwrap();
        assert_eq!(bytearray_result.as_int(), Some(255));

        unsafe {
            drop_boxed(bytes_ptr);
            drop_boxed(bytearray_ptr);
        }
    }

    #[test]
    fn test_int_with_explicit_base_parses_prefixed_text() {
        let value =
            builtin_int(&[Value::string(intern("0x_FF")), Value::int(16).unwrap()]).unwrap();
        assert_eq!(value.as_int(), Some(255));

        let (bytes_value, bytes_ptr) = boxed_value(BytesObject::from_slice(b"0b_1010"));
        let binary = builtin_int(&[bytes_value, Value::int(0).unwrap()]).unwrap();
        assert_eq!(binary.as_int(), Some(10));

        unsafe {
            drop_boxed(bytes_ptr);
        }
    }

    #[test]
    fn test_int_to_bytes_defaults_and_zero_length() {
        let default_bytes = builtin_int_to_bytes(&[Value::int(0).unwrap()], &[])
            .expect("0.to_bytes() should use CPython defaults");
        assert_eq!(value_to_bytes(default_bytes), vec![0]);

        let empty_bytes =
            builtin_int_to_bytes(&[Value::int(0).unwrap(), Value::int(0).unwrap()], &[])
                .expect("0.to_bytes(0) should succeed");
        assert_eq!(value_to_bytes(empty_bytes), Vec::<u8>::new());

        unsafe {
            drop_boxed(
                default_bytes
                    .as_object_ptr()
                    .expect("bytes result should be heap allocated")
                    as *mut BytesObject,
            );
            drop_boxed(
                empty_bytes
                    .as_object_ptr()
                    .expect("bytes result should be heap allocated")
                    as *mut BytesObject,
            );
        }
    }

    #[test]
    fn test_int_to_bytes_supports_signed_little_endian_and_bool_receivers() {
        let little_endian = builtin_int_to_bytes(
            &[
                Value::int(0x1234).unwrap(),
                Value::int(2).unwrap(),
                Value::string(intern("little")),
            ],
            &[],
        )
        .expect("little-endian encoding should succeed");
        assert_eq!(value_to_bytes(little_endian), vec![0x34, 0x12]);

        let signed_negative = builtin_int_to_bytes(
            &[
                Value::int(-1).unwrap(),
                Value::int(1).unwrap(),
                Value::string(intern("big")),
            ],
            &[("signed", Value::bool(true))],
        )
        .expect("signed encoding should support negative values");
        assert_eq!(value_to_bytes(signed_negative), vec![0xFF]);

        let boolean = builtin_int_to_bytes(&[Value::bool(true)], &[])
            .expect("bool should inherit int.to_bytes()");
        assert_eq!(value_to_bytes(boolean), vec![1]);

        unsafe {
            drop_boxed(
                little_endian
                    .as_object_ptr()
                    .expect("bytes result should be heap allocated")
                    as *mut BytesObject,
            );
            drop_boxed(
                signed_negative
                    .as_object_ptr()
                    .expect("bytes result should be heap allocated")
                    as *mut BytesObject,
            );
            drop_boxed(
                boolean
                    .as_object_ptr()
                    .expect("bytes result should be heap allocated")
                    as *mut BytesObject,
            );
        }
    }

    #[test]
    fn test_int_to_bytes_reports_cpython_compatible_argument_errors() {
        let byteorder_err = builtin_int_to_bytes(
            &[
                Value::int(1).unwrap(),
                Value::int(1).unwrap(),
                Value::int(5).unwrap(),
            ],
            &[],
        )
        .expect_err("non-string byteorder should fail");
        assert!(
            byteorder_err
                .to_string()
                .contains("to_bytes() argument 'byteorder' must be str, not int")
        );

        let length_err =
            builtin_int_to_bytes(&[Value::int(1).unwrap(), Value::string(intern("x"))], &[])
                .expect_err("non-integer length should fail");
        assert!(
            length_err
                .to_string()
                .contains("'str' object cannot be interpreted as an integer")
        );

        let unsigned_negative = builtin_int_to_bytes(
            &[
                Value::int(-1).unwrap(),
                Value::int(1).unwrap(),
                Value::string(intern("big")),
            ],
            &[],
        )
        .expect_err("negative unsigned encoding should fail");
        assert!(
            unsigned_negative
                .to_string()
                .contains("can't convert negative int to unsigned")
        );

        let signed_overflow = builtin_int_to_bytes(
            &[
                Value::int(128).unwrap(),
                Value::int(1).unwrap(),
                Value::string(intern("big")),
            ],
            &[("signed", Value::bool(true))],
        )
        .expect_err("signed one-byte overflow should fail");
        assert!(
            signed_overflow
                .to_string()
                .contains("int too big to convert")
        );
    }

    #[test]
    fn test_int_rejects_non_string_with_explicit_base() {
        let err = builtin_int(&[Value::int(12).unwrap(), Value::int(10).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("explicit base"));
    }

    #[test]
    fn test_float_from_int() {
        let result = builtin_float(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(result.as_float(), Some(42.0));
    }

    #[test]
    fn test_bool_truthy() {
        let result = builtin_bool(&[Value::int(1).unwrap()]).unwrap();
        assert!(result.is_truthy());

        let result = builtin_bool(&[Value::int(0).unwrap()]).unwrap();
        assert!(!result.is_truthy());
    }

    #[test]
    fn test_str_empty() {
        let result = builtin_str(&[]).unwrap();
        assert_eq!(value_to_string(result), "");
    }

    #[test]
    fn test_str_renders_exception_display_text() {
        let exc = crate::builtins::get_exception_type("ValueError")
            .expect("ValueError should exist")
            .construct(&[Value::string(intern("boom"))]);
        let rendered = builtin_str(&[exc]).expect("str() should render exceptions");

        assert_eq!(value_to_string(rendered), "boom");
    }

    #[test]
    fn test_str_identity_for_tagged_string() {
        let value = Value::string(intern("alpha"));
        let result = builtin_str(&[value]).unwrap();
        assert_eq!(result, value);
    }

    #[test]
    fn test_str_from_numeric_uses_text_form() {
        let int_result = builtin_str(&[Value::int(42).unwrap()]).unwrap();
        assert_eq!(value_to_string(int_result), "42");

        let float_result = builtin_str(&[Value::float(3.5)]).unwrap();
        assert_eq!(value_to_string(float_result), "3.5");
    }

    #[test]
    fn test_str_decodes_bytes_with_encoding_and_errors() {
        let encoded = to_object_value(BytesObject::from_slice("café".as_bytes()));
        let decoded = builtin_str(&[
            encoded,
            Value::string(intern("utf-8")),
            Value::string(intern("strict")),
        ])
        .expect("str(bytes, encoding, errors) should decode bytes");
        assert_eq!(value_to_string(decoded), "café");

        let bytearray = to_object_value(BytesObject::bytearray_from_slice(&[0x41, 0xFF, 0x42]));
        let escaped = builtin_str(&[
            bytearray,
            Value::string(intern("ascii")),
            Value::string(intern("backslashreplace")),
        ])
        .expect("str(bytearray, encoding, errors) should decode bytearray");
        assert_eq!(value_to_string(escaped), r"A\xffB");
    }

    #[test]
    fn test_str_decode_form_validates_argument_types_and_source_kind() {
        let bad_encoding = builtin_str(&[Value::int(1).unwrap(), Value::int(2).unwrap()])
            .expect_err("encoding must be a string");
        assert!(
            bad_encoding
                .to_string()
                .contains("str() argument 'encoding' must be str, not int")
        );

        let bad_errors = builtin_str(&[
            to_object_value(BytesObject::from_slice(b"x")),
            Value::string(intern("utf-8")),
            Value::int(2).unwrap(),
        ])
        .expect_err("errors must be a string");
        assert!(
            bad_errors
                .to_string()
                .contains("str() argument 'errors' must be str, not int")
        );

        let decoding_str =
            builtin_str(&[Value::string(intern("x")), Value::string(intern("utf-8"))])
                .expect_err("decoding a string should fail");
        assert!(
            decoding_str
                .to_string()
                .contains("decoding str is not supported")
        );

        let non_bytes = builtin_str(&[Value::int(1).unwrap(), Value::string(intern("utf-8"))])
            .expect_err("non-bytes decode source should fail");
        assert!(
            non_bytes
                .to_string()
                .contains("decoding to str: need a bytes-like object, int found")
        );
    }

    #[test]
    fn test_str_constructor_accepts_keyword_decode_arguments() {
        let result = call_builtin_type_kw(
            TypeId::STR,
            &[],
            &[
                ("object", to_object_value(BytesObject::from_slice(b"foo"))),
                ("errors", Value::string(intern("strict"))),
            ],
        )
        .expect("str(object=bytes, errors=...) should default to utf-8");
        assert_eq!(value_to_string(result), "foo");

        let empty = call_builtin_type_kw(
            TypeId::STR,
            &[],
            &[
                ("encoding", Value::string(intern("utf-8"))),
                ("errors", Value::string(intern("ignore"))),
            ],
        )
        .expect("str() should still default object to empty string");
        assert_eq!(value_to_string(empty), "");
    }

    #[test]
    fn test_str_constructor_keyword_validation_matches_cpython() {
        let duplicate = call_builtin_type_kw(
            TypeId::STR,
            &[to_object_value(BytesObject::from_slice(b"foo"))],
            &[("object", to_object_value(BytesObject::from_slice(b"bar")))],
        )
        .expect_err("duplicate object keyword should fail");
        assert!(
            duplicate
                .to_string()
                .contains("str() got multiple values for argument 'object'")
        );

        let unexpected =
            call_builtin_type_kw(TypeId::STR, &[], &[("bogus", Value::string(intern("x")))])
                .expect_err("unexpected keyword should fail");
        assert!(
            unexpected
                .to_string()
                .contains("str() got an unexpected keyword argument 'bogus'")
        );
    }

    #[test]
    fn test_property_constructor_accepts_doc_keyword() {
        let result = call_builtin_type_kw(
            TypeId::PROPERTY,
            &[],
            &[("doc", Value::string(intern("docs")))],
        )
        .expect("property(doc=...) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("property() should return descriptor object");
        let descriptor = unsafe { &*(ptr as *const PropertyDescriptor) };
        assert_eq!(descriptor.doc(), Some(Value::string(intern("docs"))));
        unsafe { drop_boxed(ptr as *mut PropertyDescriptor) };
    }

    #[test]
    fn test_property_constructor_rejects_duplicate_doc_argument() {
        let err = call_builtin_type_kw(
            TypeId::PROPERTY,
            &[
                Value::none(),
                Value::none(),
                Value::none(),
                Value::string(intern("positional")),
            ],
            &[("doc", Value::string(intern("keyword")))],
        )
        .expect_err("duplicate property doc should fail");
        assert!(
            err.to_string()
                .contains("property() got multiple values for argument 'doc'")
        );
    }

    #[test]
    fn test_dict_constructor_accepts_keyword_arguments() {
        let result = call_builtin_type_kw(
            TypeId::DICT,
            &[],
            &[
                ("alpha", Value::int(1).unwrap()),
                ("beta", Value::int(2).unwrap()),
            ],
        )
        .expect("dict(alpha=1, beta=2) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("dict() should return a dict object");
        let dict = unsafe { &*(ptr as *const DictObject) };

        assert_eq!(
            dict.get(Value::string(intern("alpha")))
                .and_then(|value| value.as_int()),
            Some(1)
        );
        assert_eq!(
            dict.get(Value::string(intern("beta")))
                .and_then(|value| value.as_int()),
            Some(2)
        );
    }

    #[test]
    fn test_dict_constructor_keywords_override_positional_mapping_entries() {
        let mut source = DictObject::new();
        source.set(Value::string(intern("alpha")), Value::int(1).unwrap());
        source.set(Value::string(intern("gamma")), Value::int(3).unwrap());
        let source_ptr = Box::into_raw(Box::new(source));

        let result = call_builtin_type_kw(
            TypeId::DICT,
            &[Value::object_ptr(source_ptr as *const ())],
            &[("alpha", Value::int(9).unwrap())],
        )
        .expect("dict(mapping, alpha=9) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("dict() should return a dict object");
        let dict = unsafe { &*(ptr as *const DictObject) };

        assert_eq!(
            dict.get(Value::string(intern("alpha")))
                .and_then(|value| value.as_int()),
            Some(9)
        );
        assert_eq!(
            dict.get(Value::string(intern("gamma")))
                .and_then(|value| value.as_int()),
            Some(3)
        );

        unsafe {
            drop(Box::from_raw(source_ptr));
        }
    }

    #[test]
    fn test_slice_stop_only_constructor() {
        let result = builtin_slice(&[Value::int(5).unwrap()]).expect("slice(stop) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("slice() should return object");
        let slice = unsafe { &*(ptr as *const SliceObject) };

        assert_eq!(slice.start(), None);
        assert_eq!(slice.stop(), Some(5));
        assert_eq!(slice.step(), None);

        unsafe { drop_boxed(ptr as *mut SliceObject) };
    }

    #[test]
    fn test_slice_full_constructor() {
        let result = builtin_slice(&[
            Value::int(1).unwrap(),
            Value::int(9).unwrap(),
            Value::int(2).unwrap(),
        ])
        .expect("slice(start, stop, step) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("slice() should return object");
        let slice = unsafe { &*(ptr as *const SliceObject) };

        assert_eq!(slice.start(), Some(1));
        assert_eq!(slice.stop(), Some(9));
        assert_eq!(slice.step(), Some(2));

        unsafe { drop_boxed(ptr as *mut SliceObject) };
    }

    #[test]
    fn test_slice_constructor_accepts_none_components() {
        let result = builtin_slice(&[Value::none(), Value::int(4).unwrap(), Value::none()])
            .expect("slice(None, 4, None) should succeed");
        let ptr = result
            .as_object_ptr()
            .expect("slice() should return object");
        let slice = unsafe { &*(ptr as *const SliceObject) };

        assert_eq!(slice.start(), None);
        assert_eq!(slice.stop(), Some(4));
        assert_eq!(slice.step(), None);

        unsafe { drop_boxed(ptr as *mut SliceObject) };
    }

    #[test]
    fn test_slice_constructor_rejects_zero_step() {
        let err = builtin_slice(&[
            Value::int(1).unwrap(),
            Value::int(5).unwrap(),
            Value::int(0).unwrap(),
        ])
        .expect_err("slice(..., step=0) should fail");
        assert!(err.to_string().contains("slice step cannot be zero"));
    }

    #[test]
    fn test_slice_constructor_rejects_non_integer_components() {
        let err = builtin_slice(&[Value::float(3.5)]).expect_err("slice(float) should fail");
        assert!(
            err.to_string()
                .contains("slice indices must be integers or None")
        );
    }

    #[test]
    fn test_list_empty() {
        let result = builtin_list(&[]).unwrap();
        let ptr = result.as_object_ptr().expect("list() should return object");
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 0);
        unsafe { drop_boxed(ptr as *mut ListObject) };
    }

    #[test]
    fn test_list_from_range() {
        let range = builtin_range(&[Value::int(0).unwrap(), Value::int(4).unwrap()]).unwrap();
        let list_value = builtin_list(&[range]).unwrap();
        let ptr = list_value.as_object_ptr().unwrap();
        let list = unsafe { &*(ptr as *const ListObject) };
        assert_eq!(list.len(), 4);
        assert_eq!(list.get(0).unwrap().as_int(), Some(0));
        assert_eq!(list.get(3).unwrap().as_int(), Some(3));
        unsafe { drop_boxed(ptr as *mut ListObject) };
    }

    #[test]
    fn test_list_consumes_iterator_state() {
        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let iterator = builtin_iter(&[list_value]).unwrap();

        let built = builtin_list(&[iterator]).unwrap();
        let built_ptr = built.as_object_ptr().unwrap();
        let built_list = unsafe { &*(built_ptr as *const ListObject) };
        assert_eq!(built_list.len(), 3);

        let next_result = builtin_next(&[iterator]);
        assert!(next_result.is_err());

        unsafe { drop_boxed(built_ptr as *mut ListObject) };
        unsafe { drop_boxed(list_ptr) };
        unsafe { drop_boxed(iterator.as_object_ptr().unwrap() as *mut IteratorObject) };
    }

    #[test]
    fn test_tuple_empty_and_from_iterable() {
        let empty = builtin_tuple(&[]).unwrap();
        let empty_ptr = empty.as_object_ptr().unwrap();
        let empty_tuple = unsafe { &*(empty_ptr as *const TupleObject) };
        assert_eq!(empty_tuple.len(), 0);
        unsafe { drop_boxed(empty_ptr as *mut TupleObject) };

        let source = ListObject::from_slice(&[Value::int(9).unwrap(), Value::int(8).unwrap()]);
        let (source_value, source_ptr) = boxed_value(source);
        let tuple_value = builtin_tuple(&[source_value]).unwrap();
        let tuple_ptr = tuple_value.as_object_ptr().unwrap();
        let tuple = unsafe { &*(tuple_ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(9));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(8));
        unsafe { drop_boxed(tuple_ptr as *mut TupleObject) };
        unsafe { drop_boxed(source_ptr) };
    }

    #[test]
    fn test_tuple_new_exposed_on_type_object_builds_tuple_instances() {
        let tuple_type = builtin_type_object_for_type_id(TypeId::TUPLE);
        let tuple_new = builtin_getattr(&[tuple_type, Value::string(intern("__new__"))])
            .expect("tuple.__new__ should resolve");
        let tuple_new_ptr = tuple_new
            .as_object_ptr()
            .expect("tuple.__new__ should be a builtin function object");
        let builtin = unsafe { &*(tuple_new_ptr as *const crate::builtins::BuiltinFunctionObject) };

        let (source_value, source_ptr) = boxed_value(ListObject::from_slice(&[
            Value::int(4).unwrap(),
            Value::int(5).unwrap(),
        ]));

        let result = builtin
            .call(&[tuple_type, source_value])
            .expect("tuple.__new__(tuple, iterable) should succeed");
        let result_ptr = result
            .as_object_ptr()
            .expect("tuple.__new__ should return a tuple object");
        let tuple = unsafe { &*(result_ptr as *const TupleObject) };

        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.get(0).unwrap().as_int(), Some(4));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(5));

        unsafe { drop_boxed(result_ptr as *mut TupleObject) };
        unsafe { drop_boxed(source_ptr) };
    }

    #[test]
    fn test_tuple_new_returns_empty_tuple_without_iterable() {
        let tuple_type = builtin_type_object_for_type_id(TypeId::TUPLE);

        let result = builtin_tuple_new(&[tuple_type]).expect("tuple.__new__(tuple) should succeed");
        let result_ptr = result
            .as_object_ptr()
            .expect("tuple.__new__(tuple) should return a tuple");
        let tuple = unsafe { &*(result_ptr as *const TupleObject) };

        assert_eq!(tuple.len(), 0);

        unsafe { drop_boxed(result_ptr as *mut TupleObject) };
    }

    #[test]
    fn test_tuple_new_builds_tuple_backed_subclass_instances() {
        let (tuple_instance, tuple_instance_ptr) = boxed_value(TupleObject::empty());
        let tuple_type = builtin_type(&[tuple_instance]).unwrap();
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[tuple_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

        let subclass = builtin_type(&[
            Value::string(intern("TupleChild")),
            bases_value,
            namespace_value,
        ])
        .expect("tuple subclass should be constructible");
        let (source_value, source_ptr) = boxed_value(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));

        let result = builtin_tuple_new(&[subclass, source_value])
            .expect("tuple.__new__ should allocate tuple-backed subclass instances");
        let result_ptr = result
            .as_object_ptr()
            .expect("tuple subclass instance should be heap allocated");
        assert_eq!(crate::ops::objects::extract_type_id(result_ptr), {
            let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
            class.class_type_id()
        });
        let shaped = unsafe { &*(result_ptr as *const ShapedObject) };
        let tuple = shaped
            .tuple_backing()
            .expect("tuple subclass should retain native tuple storage");
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.as_slice()[0].as_int(), Some(1));
        assert_eq!(tuple.as_slice()[1].as_int(), Some(2));

        let empty_result = builtin_object_new(&[subclass])
            .expect("object.__new__ should allocate tuple-backed subclass instances");
        let empty_ptr = empty_result
            .as_object_ptr()
            .expect("empty tuple subclass should be heap allocated");
        let empty_shaped = unsafe { &*(empty_ptr as *const ShapedObject) };
        assert_eq!(
            empty_shaped
                .tuple_backing()
                .expect("tuple subclass should have empty tuple backing")
                .len(),
            0
        );

        unsafe { drop_boxed(tuple_instance_ptr) };
        unsafe { drop_boxed(source_ptr) };
        unsafe { drop_boxed(result_ptr as *mut ShapedObject) };
        unsafe { drop_boxed(empty_ptr as *mut ShapedObject) };
        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_module_type_constructor_builds_module_objects() {
        let doc = Value::int(7).unwrap();
        let result = call_builtin_type(
            TypeId::MODULE,
            &[Value::string(intern("dynamic_module")), doc],
        )
        .expect("module(name, doc) should construct a module object");
        let ptr = result
            .as_object_ptr()
            .expect("module constructor should return a heap object");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::MODULE);

        let module = unsafe { &*(ptr as *const ModuleObject) };
        assert_eq!(module.name(), "dynamic_module");
        assert_eq!(
            module.get_attr("__name__").map(value_to_string).as_deref(),
            Some("dynamic_module")
        );
        assert_eq!(module.get_attr("__doc__"), Some(doc));
        assert!(module.get_attr("__loader__").unwrap().is_none());
        assert!(module.get_attr("__package__").unwrap().is_none());
        assert!(module.get_attr("__spec__").unwrap().is_none());
    }

    #[test]
    fn test_module_new_exposed_on_type_object_builds_module_objects() {
        let module_type = builtin_type_object_for_type_id(TypeId::MODULE);
        let module_new = builtin_getattr(&[module_type, Value::string(intern("__new__"))])
            .expect("module.__new__ should resolve");
        let module_new_ptr = module_new
            .as_object_ptr()
            .expect("module.__new__ should be a builtin function object");
        let builtin =
            unsafe { &*(module_new_ptr as *const crate::builtins::BuiltinFunctionObject) };

        let result = builtin
            .call(&[module_type, Value::string(intern("via_new"))])
            .expect("module.__new__(module, name) should construct a module");
        let ptr = result
            .as_object_ptr()
            .expect("module.__new__ should return a heap object");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::MODULE);

        let module = unsafe { &*(ptr as *const ModuleObject) };
        assert_eq!(module.name(), "via_new");
        assert!(module.get_attr("__doc__").unwrap().is_none());
    }

    #[test]
    fn test_core_builtin_new_wrappers_dispatch_to_runtime_constructors() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        assert_eq!(
            builtin_int_new(&[int_type, Value::int(7).unwrap()])
                .unwrap()
                .as_int(),
            Some(7)
        );

        let float_type = builtin_type(&[Value::float(0.0)]).unwrap();
        assert_eq!(
            builtin_float_new(&[float_type, Value::int(2).unwrap()])
                .unwrap()
                .as_float(),
            Some(2.0)
        );

        let str_type = builtin_type(&[Value::string(intern("seed"))]).unwrap();
        let str_value = builtin_str_new(&[str_type, Value::string(intern("seed"))]).unwrap();
        let str_ptr = str_value
            .as_string_object_ptr()
            .expect("str.__new__ should return an interned string");
        assert_eq!(
            interned_by_ptr(str_ptr as *const u8).unwrap().as_str(),
            "seed"
        );

        let bool_type = builtin_type(&[Value::bool(false)]).unwrap();
        assert_eq!(
            builtin_bool_new(&[bool_type, Value::int(1).unwrap()])
                .unwrap()
                .as_bool(),
            Some(true)
        );

        let list_type = builtin_type(&[builtin_list(&[]).unwrap()]).unwrap();
        let list_value = builtin_list_new(&[list_type]).unwrap();
        let list_ptr = list_value
            .as_object_ptr()
            .expect("list.__new__ should return a list object");
        assert_eq!(crate::ops::objects::extract_type_id(list_ptr), TypeId::LIST);

        let dict_type = builtin_type(&[builtin_dict(&[]).unwrap()]).unwrap();
        let dict_value = builtin_dict_new(&[dict_type]).unwrap();
        let dict_ptr = dict_value
            .as_object_ptr()
            .expect("dict.__new__ should return a dict object");
        assert_eq!(crate::ops::objects::extract_type_id(dict_ptr), TypeId::DICT);

        let set_type = builtin_type(&[builtin_set(&[]).unwrap()]).unwrap();
        let set_value = builtin_set_new(&[set_type]).unwrap();
        let set_ptr = set_value
            .as_object_ptr()
            .expect("set.__new__ should return a set object");
        assert_eq!(crate::ops::objects::extract_type_id(set_ptr), TypeId::SET);

        let frozenset_type = builtin_type(&[builtin_frozenset(&[]).unwrap()]).unwrap();
        let frozenset_value = builtin_frozenset_new(&[frozenset_type]).unwrap();
        let frozenset_ptr = frozenset_value
            .as_object_ptr()
            .expect("frozenset.__new__ should return a frozenset object");
        assert_eq!(
            crate::ops::objects::extract_type_id(frozenset_ptr),
            TypeId::FROZENSET
        );
    }

    #[test]
    fn test_builtin_float_getformat_reports_native_layout() {
        let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);
        let expected = Value::string(intern(native_float_format_description()));

        let double_format = builtin_float_getformat(&[float_type, Value::string(intern("double"))])
            .expect("float.__getformat__('double') should succeed");
        assert_eq!(double_format, expected);

        let float_format = builtin_float_getformat(&[float_type, Value::string(intern("float"))])
            .expect("float.__getformat__('float') should succeed");
        assert_eq!(float_format, expected);
    }

    #[test]
    fn test_builtin_float_getformat_rejects_invalid_arguments() {
        let float_type = builtin_type_object_for_type_id(TypeId::FLOAT);

        let invalid_kind =
            builtin_float_getformat(&[float_type, Value::string(intern("bogus"))]).unwrap_err();
        assert!(matches!(invalid_kind, BuiltinError::ValueError(_)));
        assert!(invalid_kind.to_string().contains("'double' or 'float'"));

        let invalid_type =
            builtin_float_getformat(&[float_type, Value::int(1).unwrap()]).unwrap_err();
        assert!(matches!(invalid_type, BuiltinError::TypeError(_)));

        let missing_arg = builtin_float_getformat(&[float_type]).unwrap_err();
        assert!(matches!(missing_arg, BuiltinError::TypeError(_)));
        assert!(missing_arg.to_string().contains("takes exactly 1 argument"));
    }

    #[test]
    fn test_str_new_builds_heap_subclass_instances_with_native_string_storage() {
        let str_type = builtin_type_object_for_type_id(TypeId::STR);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[str_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let subclass = builtin_type(&[
            Value::string(intern("StrChild")),
            bases_value,
            namespace_value,
        ])
        .expect("str subclass should be constructible");

        let value =
            builtin_str_new(&[subclass, Value::string(intern("seed"))]).expect("str subclass new");
        let ptr = value
            .as_object_ptr()
            .expect("str subclass instance should be heap allocated");
        let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
        assert_eq!(
            crate::ops::objects::extract_type_id(ptr),
            class.class_type_id()
        );

        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        assert_eq!(
            shaped
                .string_backing()
                .expect("string backing should exist")
                .as_str(),
            "seed"
        );

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_bytes_new_builds_heap_subclass_instances_with_native_byte_storage() {
        let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[bytes_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let subclass = builtin_type(&[
            Value::string(intern("BytesChild")),
            bases_value,
            namespace_value,
        ])
        .expect("bytes subclass should be constructible");

        let (source_value, source_ptr) = boxed_value(BytesObject::from_slice(b"seed"));
        let value = builtin_bytes_new(&[subclass, source_value]).expect("bytes subclass new");
        let ptr = value
            .as_object_ptr()
            .expect("bytes subclass instance should be heap allocated");
        let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
        assert_eq!(
            crate::ops::objects::extract_type_id(ptr),
            class.class_type_id()
        );

        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        assert_eq!(
            shaped
                .bytes_backing()
                .expect("bytes backing should exist")
                .as_bytes(),
            b"seed"
        );

        unsafe { drop_boxed(source_ptr) };
        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_int_new_builds_heap_subclass_instances_with_native_integer_storage() {
        let int_type = builtin_type_object_for_type_id(TypeId::INT);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[int_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let subclass = builtin_type(&[
            Value::string(intern("IntChild")),
            bases_value,
            namespace_value,
        ])
        .expect("int subclass should be constructible");

        let value =
            builtin_int_new(&[subclass, Value::string(intern("123"))]).expect("int subclass new");
        let ptr = value
            .as_object_ptr()
            .expect("int subclass instance should be heap allocated");
        let class = unsafe { &*(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
        assert_eq!(
            crate::ops::objects::extract_type_id(ptr),
            class.class_type_id()
        );

        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        assert_eq!(
            shaped.int_backing().expect("integer backing should exist"),
            &BigInt::from(123_i64)
        );
        assert_eq!(value_to_bigint(value), Some(BigInt::from(123_i64)));
        assert_eq!(builtin_int(&[value]).unwrap().as_int(), Some(123));

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_object_new_allocates_native_integer_storage_for_int_subclasses() {
        let int_type = builtin_type_object_for_type_id(TypeId::INT);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[int_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let subclass = builtin_type(&[
            Value::string(intern("ObjectNewIntChild")),
            bases_value,
            namespace_value,
        ])
        .expect("int subclass should be constructible");

        let value = builtin_object_new(&[subclass]).expect("object.__new__ should allocate");
        let ptr = value
            .as_object_ptr()
            .expect("int subclass instance should be heap allocated");
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        assert_eq!(
            shaped.int_backing().expect("integer backing should exist"),
            &BigInt::zero()
        );

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_object_new_allocates_native_byte_storage_for_bytes_subclasses() {
        let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::from_slice(&[bytes_type]));
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let subclass = builtin_type(&[
            Value::string(intern("ObjectNewBytesChild")),
            bases_value,
            namespace_value,
        ])
        .expect("bytes subclass should be constructible");

        let value = builtin_object_new(&[subclass]).expect("object.__new__ should allocate");
        let ptr = value
            .as_object_ptr()
            .expect("bytes subclass instance should be heap allocated");
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        assert!(shaped.has_bytes_backing());
        assert!(shaped.bytes_backing().unwrap().is_empty());

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
        unsafe { drop_class(subclass.as_object_ptr().unwrap() as *const PyClassObject) };
    }

    #[test]
    fn test_builtin_new_wrappers_validate_receiver_types() {
        let err = builtin_int_new(&[Value::int(1).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("int.__new__(X): X must be a type"));

        let bool_type = builtin_type_object_for_type_id(TypeId::BOOL);
        let err = builtin_int_new(&[bool_type, Value::int(0).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        match err {
            BuiltinError::TypeError(message) => {
                assert_eq!(message, "int.__new__(bool) is not safe, use bool.__new__()");
            }
            _ => panic!("expected TypeError for int.__new__(bool, 0)"),
        }

        let err = builtin_list_new(&[]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(
            err.to_string()
                .contains("list.__new__() takes at least 1 argument")
        );
    }

    #[test]
    fn test_object_init_accepts_single_receiver() {
        let instance = builtin_object(&[]).expect("object() should succeed");
        let result = builtin_object_init(&[instance]).expect("object.__init__ should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_object_init_rejects_extra_arguments() {
        let instance = builtin_object(&[]).expect("object() should succeed");
        let err = builtin_object_init(&[instance, Value::int(1).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("object.__init__()"));
    }

    #[test]
    fn test_object_init_subclass_accepts_single_class_receiver() {
        let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
        let result = builtin_object_init_subclass(&[object_type], &[])
            .expect("object.__init_subclass__ should succeed");
        assert!(result.is_none());
    }

    #[test]
    fn test_object_init_subclass_rejects_keyword_arguments() {
        let object_type = builtin_type_object_for_type_id(TypeId::OBJECT);
        let err =
            builtin_object_init_subclass(&[object_type], &[("token", Value::int(1).unwrap())])
                .unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(
            err.to_string()
                .contains("object.__init_subclass__() takes no keyword arguments")
        );
    }

    #[test]
    fn test_type_init_accepts_metaclass_construction_signature() {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());
        let type_value = builtin_type(&[
            Value::string(intern("MetaInitTarget")),
            bases_value,
            namespace_value,
        ])
        .expect("type() should construct a heap type");

        let result = builtin_type_init(&[
            type_value,
            Value::string(intern("MetaInitTarget")),
            bases_value,
            namespace_value,
        ])
        .expect("type.__init__ should accept the class creation signature");

        assert!(result.is_none());

        unsafe { drop_class(type_value.as_object_ptr().unwrap() as *const PyClassObject) };
        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
    }

    #[test]
    fn test_mappingproxy_type_wraps_dict_arguments() {
        let mut dict = DictObject::new();
        dict.set(Value::string(intern("token")), Value::int(7).unwrap());
        let dict_value = to_object_value(dict);

        let proxy = builtin_mappingproxy(&[dict_value]).expect("mappingproxy(dict) should succeed");
        let proxy_ptr = proxy
            .as_object_ptr()
            .expect("mappingproxy should allocate a proxy object");
        assert_eq!(
            crate::ops::objects::extract_type_id(proxy_ptr),
            TypeId::MAPPING_PROXY
        );

        let proxy = unsafe { &*(proxy_ptr as *const MappingProxyObject) };
        assert_eq!(
            crate::builtins::builtin_mapping_proxy_get_item_static(
                proxy,
                Value::string(intern("token"))
            )
            .expect("proxy lookup should succeed"),
            Some(Value::int(7).unwrap())
        );
    }

    #[test]
    fn test_mappingproxy_type_rejects_non_mappings() {
        let err = builtin_mappingproxy(&[Value::int(3).unwrap()])
            .expect_err("mappingproxy(int) should fail");
        match err {
            BuiltinError::TypeError(message) => {
                assert!(message.contains("must be a mapping"));
            }
            other => panic!("expected TypeError, got {other:?}"),
        }
    }

    #[test]
    fn test_type_init_rejects_invalid_argument_count() {
        let type_value = builtin_type_object_for_type_id(TypeId::TYPE);
        let err = builtin_type_init(&[type_value, Value::string(intern("oops"))]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("type.__init__()"));
    }

    #[test]
    fn test_set_empty_and_deduplicated() {
        let empty = builtin_set(&[]).unwrap();
        let empty_ptr = empty.as_object_ptr().unwrap();
        let empty_set = unsafe { &*(empty_ptr as *const SetObject) };
        assert_eq!(empty_set.len(), 0);
        unsafe { drop_boxed(empty_ptr as *mut SetObject) };

        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let set_value = builtin_set(&[list_value]).unwrap();
        let set_ptr = set_value.as_object_ptr().unwrap();
        let set = unsafe { &*(set_ptr as *const SetObject) };
        assert_eq!(set.len(), 2);
        assert!(set.contains(Value::int(1).unwrap()));
        assert!(set.contains(Value::int(2).unwrap()));
        unsafe { drop_boxed(set_ptr as *mut SetObject) };
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_frozenset_empty_and_deduplicated() {
        let empty = builtin_frozenset(&[]).unwrap();
        let empty_ptr = empty.as_object_ptr().unwrap();
        assert_eq!(
            crate::ops::objects::extract_type_id(empty_ptr),
            TypeId::FROZENSET
        );
        let empty_set = unsafe { &*(empty_ptr as *const SetObject) };
        assert_eq!(empty_set.len(), 0);
        unsafe { drop_boxed(empty_ptr as *mut SetObject) };

        let list = ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]);
        let (list_value, list_ptr) = boxed_value(list);
        let frozen = builtin_frozenset(&[list_value]).unwrap();
        let frozen_ptr = frozen.as_object_ptr().unwrap();
        assert_eq!(
            crate::ops::objects::extract_type_id(frozen_ptr),
            TypeId::FROZENSET
        );
        let frozen_set = unsafe { &*(frozen_ptr as *const SetObject) };
        assert_eq!(frozen_set.len(), 2);
        assert!(frozen_set.contains(Value::int(1).unwrap()));
        assert!(frozen_set.contains(Value::int(2).unwrap()));

        unsafe { drop_boxed(frozen_ptr as *mut SetObject) };
        unsafe { drop_boxed(list_ptr) };
    }

    #[test]
    fn test_frozenset_identity_for_existing_frozenset() {
        let source = builtin_frozenset(&[]).unwrap();
        let source_ptr = source.as_object_ptr().unwrap();

        let again = builtin_frozenset(&[source]).unwrap();
        assert_eq!(again, source);

        unsafe { drop_boxed(source_ptr as *mut SetObject) };
    }

    #[test]
    fn test_set_and_frozenset_reject_unhashable_elements() {
        let inner = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let (inner_value, inner_ptr) = boxed_value(inner);
        let outer = ListObject::from_slice(&[inner_value]);
        let (outer_value, outer_ptr) = boxed_value(outer);

        let set_err = builtin_set(&[outer_value]).expect_err("set should reject unhashable items");
        assert!(set_err.to_string().contains("unhashable type: 'list'"));

        let frozen_err = builtin_frozenset(&[outer_value])
            .expect_err("frozenset should reject unhashable items");
        assert!(frozen_err.to_string().contains("unhashable type: 'list'"));

        unsafe {
            drop_boxed(outer_ptr);
            drop_boxed(inner_ptr);
        }
    }

    #[test]
    fn test_dict_empty_and_copy() {
        let empty = builtin_dict(&[]).unwrap();
        let empty_ptr = empty.as_object_ptr().unwrap();
        let empty_dict = unsafe { &*(empty_ptr as *const DictObject) };
        assert_eq!(empty_dict.len(), 0);
        unsafe { drop_boxed(empty_ptr as *mut DictObject) };

        let mut source = DictObject::new();
        source.set(Value::int(1).unwrap(), Value::int(10).unwrap());
        let (source_value, source_ptr) = boxed_value(source);
        let copied = builtin_dict(&[source_value]).unwrap();
        let copied_ptr = copied.as_object_ptr().unwrap();
        let copied_dict = unsafe { &*(copied_ptr as *const DictObject) };
        assert_eq!(copied_dict.len(), 1);
        assert_eq!(
            copied_dict.get(Value::int(1).unwrap()).unwrap().as_int(),
            Some(10)
        );
        unsafe { drop_boxed(copied_ptr as *mut DictObject) };
        unsafe { drop_boxed(source_ptr) };
    }

    #[test]
    fn test_dict_from_pair_sequence() {
        let pair1 = TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(11).unwrap()]);
        let pair2 = TupleObject::from_slice(&[Value::int(2).unwrap(), Value::int(22).unwrap()]);
        let (pair1_value, pair1_ptr) = boxed_value(pair1);
        let (pair2_value, pair2_ptr) = boxed_value(pair2);

        let pairs = ListObject::from_slice(&[pair1_value, pair2_value]);
        let (pairs_value, pairs_ptr) = boxed_value(pairs);

        let dict_value = builtin_dict(&[pairs_value]).unwrap();
        let dict_ptr = dict_value.as_object_ptr().unwrap();
        let dict = unsafe { &*(dict_ptr as *const DictObject) };

        assert_eq!(dict.len(), 2);
        assert_eq!(dict.get(Value::int(1).unwrap()).unwrap().as_int(), Some(11));
        assert_eq!(dict.get(Value::int(2).unwrap()).unwrap().as_int(), Some(22));

        unsafe { drop_boxed(dict_ptr as *mut DictObject) };
        unsafe { drop_boxed(pairs_ptr) };
        unsafe { drop_boxed(pair1_ptr) };
        unsafe { drop_boxed(pair2_ptr) };
    }

    #[test]
    fn test_dict_invalid_pair_length_errors() {
        let bad_pair = TupleObject::from_slice(&[Value::int(1).unwrap()]);
        let (bad_pair_value, bad_pair_ptr) = boxed_value(bad_pair);
        let pairs = ListObject::from_slice(&[bad_pair_value]);
        let (pairs_value, pairs_ptr) = boxed_value(pairs);

        let err = builtin_dict(&[pairs_value]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("has length 1; 2 is required"));

        unsafe { drop_boxed(pairs_ptr) };
        unsafe { drop_boxed(bad_pair_ptr) };
    }

    #[test]
    fn test_dict_fromkeys_builds_mapping_with_default_none() {
        let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
        let (keys_value, keys_ptr) = boxed_value(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ]));

        let result = builtin_dict_fromkeys(&[dict_type, keys_value]).unwrap();
        let result_ptr = result.as_object_ptr().unwrap();
        let dict = unsafe { &*(result_ptr as *const DictObject) };

        assert_eq!(dict.len(), 2);
        assert!(dict.get(Value::int(1).unwrap()).unwrap().is_none());
        assert!(dict.get(Value::int(2).unwrap()).unwrap().is_none());

        unsafe { drop_boxed(result_ptr as *mut DictObject) };
        unsafe { drop_boxed(keys_ptr) };
    }

    #[test]
    fn test_dict_fromkeys_reuses_requested_value_for_each_key() {
        let dict_type = builtin_type_object_for_type_id(TypeId::DICT);
        let (keys_value, keys_ptr) = boxed_value(ListObject::from_slice(&[
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ]));

        let result =
            builtin_dict_fromkeys(&[dict_type, keys_value, Value::int(99).unwrap()]).unwrap();
        let result_ptr = result.as_object_ptr().unwrap();
        let dict = unsafe { &*(result_ptr as *const DictObject) };

        assert_eq!(dict.get(Value::int(3).unwrap()).unwrap().as_int(), Some(99));
        assert_eq!(dict.get(Value::int(4).unwrap()).unwrap().as_int(), Some(99));

        unsafe { drop_boxed(result_ptr as *mut DictObject) };
        unsafe { drop_boxed(keys_ptr) };
    }

    #[test]
    fn test_dict_fromkeys_rejects_non_dict_receivers() {
        let err =
            builtin_dict_fromkeys(&[builtin_type_object_for_type_id(TypeId::LIST), Value::none()])
                .unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("built-in dict type"));
    }

    #[test]
    fn test_str_maketrans_exposed_on_type_object_builds_mapping_from_dict() {
        let str_type = builtin_type_object_for_type_id(TypeId::STR);
        let method = builtin_getattr(&[str_type, Value::string(intern("maketrans"))])
            .expect("str.maketrans should resolve");
        let method_ptr = method
            .as_object_ptr()
            .expect("str.maketrans should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        let mut mapping = DictObject::new();
        mapping.set(Value::string(intern("a")), Value::string(intern("x")));
        mapping.set(
            Value::int('b' as i64).unwrap(),
            Value::int('y' as i64).unwrap(),
        );
        mapping.set(Value::string(intern("c")), Value::none());
        let mapping_ptr = Box::into_raw(Box::new(mapping));

        let result = builtin
            .call(&[Value::object_ptr(mapping_ptr as *const ())])
            .expect("str.maketrans(dict) should succeed");
        let result_ptr = result.as_object_ptr().expect("result should be a dict");
        let dict = unsafe { &*(result_ptr as *const DictObject) };

        assert_eq!(
            value_to_string(dict.get(Value::int('a' as i64).unwrap()).unwrap()),
            "x"
        );
        assert_eq!(
            dict.get(Value::int('b' as i64).unwrap()).unwrap().as_int(),
            Some('y' as i64)
        );
        assert!(dict.get(Value::int('c' as i64).unwrap()).unwrap().is_none());

        unsafe { drop_boxed(result_ptr as *mut DictObject) };
        unsafe { drop_boxed(mapping_ptr) };
    }

    #[test]
    fn test_str_maketrans_string_forms_build_expected_translation_table() {
        let str_type = builtin_type_object_for_type_id(TypeId::STR);
        let method = builtin_getattr(&[str_type, Value::string(intern("maketrans"))])
            .expect("str.maketrans should resolve");
        let method_ptr = method
            .as_object_ptr()
            .expect("str.maketrans should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        let result = builtin
            .call(&[
                Value::string(intern("ab")),
                Value::string(intern("xy")),
                Value::string(intern("z")),
            ])
            .expect("str.maketrans(string, string, delete) should succeed");
        let result_ptr = result.as_object_ptr().expect("result should be a dict");
        let dict = unsafe { &*(result_ptr as *const DictObject) };

        assert_eq!(
            dict.get(Value::int('a' as i64).unwrap()).unwrap().as_int(),
            Some('x' as i64)
        );
        assert_eq!(
            dict.get(Value::int('b' as i64).unwrap()).unwrap().as_int(),
            Some('y' as i64)
        );
        assert!(dict.get(Value::int('z' as i64).unwrap()).unwrap().is_none());

        unsafe { drop_boxed(result_ptr as *mut DictObject) };
    }

    #[test]
    fn test_bytes_maketrans_builds_256_byte_translation_table() {
        let bytes_type = builtin_type_object_for_type_id(TypeId::BYTES);
        let method = builtin_getattr(&[bytes_type, Value::string(intern("maketrans"))])
            .expect("bytes.maketrans should resolve");
        let method_ptr = method
            .as_object_ptr()
            .expect("bytes.maketrans should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        let (from_value, from_ptr) = boxed_value(BytesObject::from_slice(b"ab"));
        let (to_value, to_ptr) = boxed_value(BytesObject::bytearray_from_slice(b"xy"));
        let result = builtin
            .call(&[from_value, to_value])
            .expect("bytes.maketrans should accept bytes-like arguments");
        let table = value_to_bytes(result);

        assert_eq!(table.len(), 256);
        assert_eq!(table[b'a' as usize], b'x');
        assert_eq!(table[b'b' as usize], b'y');
        assert_eq!(table[b'z' as usize], b'z');

        unsafe { drop_boxed(result.as_object_ptr().unwrap() as *mut BytesObject) };

        let (memoryview_source, memoryview_source_ptr) = boxed_value(BytesObject::from_slice(b"c"));
        let memoryview_value =
            builtin_memoryview(&[memoryview_source]).expect("memoryview(bytes) should work");
        let (memoryview_target, memoryview_target_ptr) = boxed_value(BytesObject::from_slice(b"z"));
        let memoryview_result = builtin
            .call(&[memoryview_value, memoryview_target])
            .expect("bytes.maketrans should accept memoryview arguments");
        let memoryview_table = value_to_bytes(memoryview_result);
        assert_eq!(memoryview_table[b'c' as usize], b'z');

        unsafe {
            drop_boxed(memoryview_result.as_object_ptr().unwrap() as *mut BytesObject);
            drop_boxed(memoryview_value.as_object_ptr().unwrap() as *mut MemoryViewObject);
            drop_boxed(memoryview_source_ptr);
            drop_boxed(memoryview_target_ptr);
        }
        unsafe { drop_boxed(from_ptr) };
        unsafe { drop_boxed(to_ptr) };
    }

    #[test]
    fn test_bytes_maketrans_rejects_mismatched_lengths() {
        let (from_value, from_ptr) = boxed_value(BytesObject::from_slice(b"a"));
        let (to_value, to_ptr) = boxed_value(BytesObject::from_slice(b"xy"));
        let err = builtin_bytes_maketrans(&[from_value, to_value])
            .expect_err("mismatched translation tables should fail");
        assert!(matches!(err, BuiltinError::ValueError(_)));

        unsafe { drop_boxed(from_ptr) };
        unsafe { drop_boxed(to_ptr) };
    }

    #[test]
    fn test_object_constructor() {
        let value = builtin_object(&[]).unwrap();
        let ptr = value
            .as_object_ptr()
            .expect("object() should return object");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::OBJECT);
        unsafe { drop_boxed(ptr as *mut ShapedObject) };
    }

    #[test]
    fn test_object_constructor_arity_error() {
        let err = builtin_object(&[Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
    }

    #[test]
    fn test_type_builtin_returns_type_object_and_is_cached() {
        let t1 = builtin_type(&[Value::int(1).unwrap()]).unwrap();
        let t2 = builtin_type(&[Value::int(2).unwrap()]).unwrap();
        assert_eq!(t1, t2);

        let ptr = t1.as_object_ptr().expect("type() should return object");
        assert_eq!(crate::ops::objects::extract_type_id(ptr), TypeId::TYPE);
    }

    #[test]
    fn test_type_builtin_returns_exception_class_for_exception_instances() {
        let exc = crate::builtins::get_exception_type("ValueError")
            .expect("ValueError should exist")
            .construct(&[Value::string(intern("boom"))]);
        let exc_type = builtin_type(&[exc]).expect("type() should accept exception instances");
        let type_name = builtin_getattr(&[exc_type, Value::string(intern("__name__"))])
            .expect("__name__ should be readable");

        assert_eq!(type_name, Value::string(intern("ValueError")));
    }

    #[test]
    fn test_type_builtin_three_arg_form_builds_class_and_copies_namespace() {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

        let mut namespace = DictObject::new();
        namespace.set(Value::string(intern("answer")), Value::int(42).unwrap());
        let (namespace_value, namespace_ptr) = boxed_value(namespace);

        let class_value = builtin_type(&[
            Value::string(intern("Dynamic")),
            bases_value,
            namespace_value,
        ])
        .unwrap();
        let class_ptr = class_value
            .as_object_ptr()
            .expect("type() should return class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };

        assert_eq!(class.name().as_str(), "Dynamic");
        assert_eq!(
            class.get_attr(&intern("answer")).unwrap().as_int(),
            Some(42)
        );
        assert_eq!(class.mro(), &[class.class_id(), ClassId::OBJECT]);

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(class_ptr as *const PyClassObject) };
    }

    #[test]
    fn test_type_builtin_three_arg_form_accepts_dict_subclass_namespace() {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

        let mut namespace =
            ShapedObject::new_dict_backed(TypeId::from_raw(900), shape_registry().empty_shape());
        namespace
            .dict_backing_mut()
            .expect("dict subclass namespace should expose dict backing")
            .set(Value::string(intern("answer")), Value::int(42).unwrap());
        let (namespace_value, namespace_ptr) = boxed_value(namespace);

        let class_value = builtin_type(&[
            Value::string(intern("DynamicSubclassNamespace")),
            bases_value,
            namespace_value,
        ])
        .expect("type() should accept dict subclass namespaces");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("type() should return class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };

        assert_eq!(class.name().as_str(), "DynamicSubclassNamespace");
        assert_eq!(
            class.get_attr(&intern("answer")).unwrap().as_int(),
            Some(42)
        );

        unsafe { drop_boxed(namespace_ptr as *mut ShapedObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(class_ptr as *const PyClassObject) };
    }

    #[test]
    fn test_type_new_builtin_supports_explicit_metaclass_argument() {
        let metaclass = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());

        let mut namespace = DictObject::new();
        namespace.set(
            Value::string(intern("_member_names_")),
            Value::int(7).unwrap(),
        );
        namespace.set(Value::string(intern("answer")), Value::int(42).unwrap());
        let (namespace_value, namespace_ptr) = boxed_value(namespace);

        let class_value = builtin_type_new(&[
            metaclass,
            Value::string(intern("DynamicMeta")),
            bases_value,
            namespace_value,
        ])
        .expect("type.__new__ should build class");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("type.__new__ should return class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };

        assert_eq!(class.name().as_str(), "DynamicMeta");
        assert_eq!(
            class.get_attr(&intern("answer")).unwrap().as_int(),
            Some(42)
        );
        assert!(class.get_attr(&intern("_member_names_")).is_some());

        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(class_ptr as *const PyClassObject) };
    }

    #[test]
    fn test_type_builtin_three_arg_form_supports_builtin_base_types() {
        let (tuple_instance, tuple_instance_ptr) = boxed_value(TupleObject::empty());
        let tuple_type = builtin_type(&[tuple_instance]).unwrap();

        let bases = TupleObject::from_slice(&[tuple_type]);
        let (bases_value, bases_ptr) = boxed_value(bases);
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

        let class_value = builtin_type(&[
            Value::string(intern("TupleChild")),
            bases_value,
            namespace_value,
        ])
        .unwrap();
        let class_ptr = class_value
            .as_object_ptr()
            .expect("type() should return class object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };

        assert_eq!(class.bases(), &[ClassId(TypeId::TUPLE.raw())]);
        assert_eq!(
            class.mro(),
            &[
                class.class_id(),
                ClassId(TypeId::TUPLE.raw()),
                ClassId::OBJECT
            ]
        );

        unsafe { drop_boxed(tuple_instance_ptr) };
        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(class_ptr as *const PyClassObject) };
    }

    #[test]
    fn test_type_builtin_returns_heap_class_for_heap_instances() {
        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
        let (namespace_value, namespace_ptr) = boxed_value(DictObject::new());

        let class_value = builtin_type(&[
            Value::string(intern("HeapInstanceType")),
            bases_value,
            namespace_value,
        ])
        .expect("type() should build a heap class");
        let class_ptr = class_value
            .as_object_ptr()
            .expect("type() should return a heap class");

        let instance = builtin_object_new(&[class_value]).expect("object.__new__ should allocate");
        let instance_ptr = instance
            .as_object_ptr()
            .expect("object.__new__ should return a heap instance");

        assert_eq!(builtin_type(&[instance]).unwrap(), class_value);

        unsafe { drop_boxed(instance_ptr as *mut ShapedObject) };
        unsafe { drop_boxed(namespace_ptr as *mut DictObject) };
        unsafe { drop_boxed(bases_ptr) };
        unsafe { drop_class(class_ptr as *const PyClassObject) };
    }

    #[test]
    fn test_type_builtin_three_arg_form_validates_inputs() {
        let err =
            builtin_type(&[Value::int(1).unwrap(), Value::none(), Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("argument 1"));

        let err =
            builtin_type(&[Value::string(intern("C")), Value::none(), Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("argument 2"));

        let (bases_value, bases_ptr) = boxed_value(TupleObject::empty());
        let err =
            builtin_type(&[Value::string(intern("C")), bases_value, Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("argument 3"));

        unsafe { drop_boxed(bases_ptr) };
    }

    #[test]
    fn test_isinstance_true_and_false_cases() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        let float_type = builtin_type(&[Value::float(0.0)]).unwrap();
        let object_value = builtin_object(&[]).unwrap();
        let object_ptr = object_value.as_object_ptr().unwrap();
        let object_type = builtin_type(&[object_value]).unwrap();
        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };

        assert!(
            builtin_isinstance(&[Value::int(5).unwrap(), int_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            !builtin_isinstance(&[Value::int(5).unwrap(), float_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            builtin_isinstance(&[Value::int(5).unwrap(), object_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn test_isinstance_bool_is_subclass_of_int() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        assert!(
            builtin_isinstance(&[Value::bool(true), int_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn test_isinstance_tuple_of_types() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        let str_type = builtin_type(&[Value::string(intern("s"))]).unwrap();
        let tuple = TupleObject::from_slice(&[str_type, int_type]);
        let (tuple_value, tuple_ptr) = boxed_value(tuple);

        let result = builtin_isinstance(&[Value::int(9).unwrap(), tuple_value]).unwrap();
        assert!(result.as_bool().unwrap());
        unsafe { drop_boxed(tuple_ptr) };
    }

    #[test]
    fn test_isinstance_invalid_type_spec_error() {
        let err =
            builtin_isinstance(&[Value::int(1).unwrap(), Value::int(2).unwrap()]).unwrap_err();
        assert!(matches!(err, BuiltinError::TypeError(_)));
        assert!(err.to_string().contains("arg 2 must be a type"));
    }

    #[test]
    fn test_isinstance_uses_heap_metaclass_for_class_objects() {
        let type_type = crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE);

        let (metaclass_bases, metaclass_bases_ptr) =
            boxed_value(TupleObject::from_slice(&[type_type]));
        let (metaclass_namespace, metaclass_namespace_ptr) = boxed_value(DictObject::new());
        let metaclass = builtin_type(&[
            Value::string(intern("MetaEnumType")),
            metaclass_bases,
            metaclass_namespace,
        ])
        .expect("type() should build a heap metaclass");
        let metaclass_ptr = metaclass
            .as_object_ptr()
            .expect("heap metaclass should be object-backed");

        let (class_bases, class_bases_ptr) = boxed_value(TupleObject::empty());
        let (class_namespace, class_namespace_ptr) = boxed_value(DictObject::new());
        let enum_like = builtin_type_new(&[
            metaclass,
            Value::string(intern("EnumLike")),
            class_bases,
            class_namespace,
        ])
        .expect("type.__new__ should build a class with the custom metaclass");
        let enum_like_ptr = enum_like
            .as_object_ptr()
            .expect("heap class should be object-backed");

        assert_eq!(builtin_type(&[enum_like]).unwrap(), metaclass);
        assert!(
            builtin_isinstance(&[enum_like, metaclass])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            builtin_isinstance(&[enum_like, type_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        unsafe {
            drop_boxed(metaclass_bases_ptr);
            drop_boxed(metaclass_namespace_ptr as *mut DictObject);
            drop_boxed(class_bases_ptr);
            drop_boxed(class_namespace_ptr as *mut DictObject);
            drop_class(enum_like_ptr as *const PyClassObject);
            drop_class(metaclass_ptr as *const PyClassObject);
        }
    }

    #[test]
    fn test_issubclass_semantics() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        let bool_type = builtin_type(&[Value::bool(true)]).unwrap();
        let object_value = builtin_object(&[]).unwrap();
        let object_ptr = object_value.as_object_ptr().unwrap();
        let object_type = builtin_type(&[object_value]).unwrap();
        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
        let float_type = builtin_type(&[Value::float(0.0)]).unwrap();

        assert!(
            builtin_issubclass(&[int_type, object_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            builtin_issubclass(&[bool_type, int_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            !builtin_issubclass(&[float_type, int_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn test_issubclass_tuple_of_targets() {
        let int_type = builtin_type(&[Value::int(0).unwrap()]).unwrap();
        let str_type = builtin_type(&[Value::string(intern("x"))]).unwrap();
        let object_value = builtin_object(&[]).unwrap();
        let object_ptr = object_value.as_object_ptr().unwrap();
        let object_type = builtin_type(&[object_value]).unwrap();
        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
        let targets = TupleObject::from_slice(&[str_type, object_type]);
        let (targets_value, targets_ptr) = boxed_value(targets);

        let result = builtin_issubclass(&[int_type, targets_value]).unwrap();
        assert!(result.as_bool().unwrap());
        unsafe { drop_boxed(targets_ptr) };
    }

    #[test]
    fn test_issubclass_arg_validation() {
        let object_value = builtin_object(&[]).unwrap();
        let object_ptr = object_value.as_object_ptr().unwrap();
        let object_type = builtin_type(&[object_value]).unwrap();
        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
        let err1 = builtin_issubclass(&[Value::int(1).unwrap(), object_type]).unwrap_err();
        assert!(matches!(err1, BuiltinError::TypeError(_)));
        assert!(err1.to_string().contains("arg 1 must be a class"));

        let int_type = builtin_type(&[Value::int(1).unwrap()]).unwrap();
        let err2 = builtin_issubclass(&[int_type, Value::int(2).unwrap()]).unwrap_err();
        assert!(matches!(err2, BuiltinError::TypeError(_)));
        assert!(err2.to_string().contains("arg 2 must be a class"));
    }

    #[test]
    fn test_issubclass_accepts_exception_type_values_without_heap_class_casts() {
        let value_error = crate::builtins::exception_type_value_for_id(
            ExceptionTypeId::ValueError.as_u8() as u16,
        )
        .expect("ValueError type should exist");
        let exception =
            crate::builtins::exception_type_value_for_id(ExceptionTypeId::Exception.as_u8() as u16)
                .expect("Exception type should exist");
        let int_type = builtin_type_object_for_type_id(TypeId::INT);

        assert!(
            builtin_issubclass(&[value_error, exception])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            !builtin_issubclass(&[value_error, int_type])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        let mut vm = VirtualMachine::new();
        assert!(
            builtin_issubclass_vm(&mut vm, &[value_error, exception])
                .unwrap()
                .as_bool()
                .unwrap()
        );
    }

    #[test]
    fn test_issubclass_vm_honors_metaclass_subclasscheck() {
        fn subclasscheck(args: &[Value]) -> Result<Value, BuiltinError> {
            assert_eq!(args.len(), 1);
            Ok(Value::bool(true))
        }

        let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
            heap_metaclass_with_hook("MetaSubclassCheck", "__subclasscheck__", subclasscheck);
        let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
            heap_class_with_metaclass(metaclass, "VirtualBase");
        let (candidate, candidate_ptr, candidate_bases_ptr, candidate_namespace_ptr) =
            heap_class("VirtualCandidate");

        let mut vm = VirtualMachine::new();
        assert!(
            builtin_issubclass_vm(&mut vm, &[candidate, target])
                .unwrap()
                .as_bool()
                .unwrap()
        );
        assert!(
            !builtin_issubclass(&[candidate, target])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        unsafe {
            drop_boxed(hook_ptr);
            drop_boxed(target_namespace_ptr);
            drop_boxed(target_bases_ptr);
            drop_boxed(candidate_namespace_ptr);
            drop_boxed(candidate_bases_ptr);
            drop_boxed(metaclass_namespace_ptr);
            drop_boxed(metaclass_bases_ptr);
            drop_class(candidate_ptr);
            drop_class(target_ptr);
            drop_class(metaclass_ptr);
        }
    }

    #[test]
    fn test_isinstance_vm_honors_metaclass_instancecheck() {
        fn instancecheck(args: &[Value]) -> Result<Value, BuiltinError> {
            assert_eq!(args.len(), 1);
            Ok(Value::bool(true))
        }

        let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
            heap_metaclass_with_hook("MetaInstanceCheck", "__instancecheck__", instancecheck);
        let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
            heap_class_with_metaclass(metaclass, "VirtualTarget");
        let (candidate, candidate_ptr, candidate_bases_ptr, candidate_namespace_ptr) =
            heap_class("VirtualCandidate");
        let instance =
            builtin_object_new(&[candidate]).expect("object.__new__ should build a heap instance");
        let instance_ptr = instance
            .as_object_ptr()
            .expect("object.__new__ should return a heap instance");

        let mut vm = VirtualMachine::new();
        assert!(
            builtin_isinstance_vm(&mut vm, &[instance, target])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        unsafe {
            drop_boxed(instance_ptr as *mut ShapedObject);
            drop_boxed(hook_ptr);
            drop_boxed(target_namespace_ptr);
            drop_boxed(target_bases_ptr);
            drop_boxed(candidate_namespace_ptr);
            drop_boxed(candidate_bases_ptr);
            drop_boxed(metaclass_namespace_ptr);
            drop_boxed(metaclass_bases_ptr);
            drop_class(candidate_ptr);
            drop_class(target_ptr);
            drop_class(metaclass_ptr);
        }
    }

    #[test]
    fn test_isinstance_vm_preserves_exact_type_match_before_instancecheck_hook() {
        fn instancecheck(args: &[Value]) -> Result<Value, BuiltinError> {
            assert_eq!(args.len(), 1);
            Ok(Value::bool(false))
        }

        let (metaclass, metaclass_ptr, metaclass_bases_ptr, metaclass_namespace_ptr, hook_ptr) =
            heap_metaclass_with_hook("MetaExactInstance", "__instancecheck__", instancecheck);
        let (target, target_ptr, target_bases_ptr, target_namespace_ptr) =
            heap_class_with_metaclass(metaclass, "ExactTarget");
        let instance =
            builtin_object_new(&[target]).expect("object.__new__ should build a heap instance");
        let instance_ptr = instance
            .as_object_ptr()
            .expect("object.__new__ should return a heap instance");

        let mut vm = VirtualMachine::new();
        assert!(
            builtin_isinstance_vm(&mut vm, &[instance, target])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        unsafe {
            drop_boxed(instance_ptr as *mut ShapedObject);
            drop_boxed(hook_ptr);
            drop_boxed(target_namespace_ptr);
            drop_boxed(target_bases_ptr);
            drop_boxed(metaclass_namespace_ptr);
            drop_boxed(metaclass_bases_ptr);
            drop_class(target_ptr);
            drop_class(metaclass_ptr);
        }
    }

    #[test]
    fn test_attribute_builtins_roundtrip_with_tagged_name() {
        let object = builtin_object(&[]).unwrap();
        let object_ptr = object.as_object_ptr().unwrap();
        let name = Value::string(intern("field"));

        builtin_setattr(&[object, name, Value::int(42).unwrap()]).unwrap();
        assert_eq!(builtin_getattr(&[object, name]).unwrap().as_int(), Some(42));
        assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

        // Distinguish explicit None assignment from deletion.
        builtin_setattr(&[object, name, Value::none()]).unwrap();
        assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());
        assert!(builtin_getattr(&[object, name]).unwrap().is_none());

        builtin_delattr(&[object, name]).unwrap();
        assert!(!builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

        let err = builtin_getattr(&[object, name]).unwrap_err();
        assert!(matches!(err, BuiltinError::AttributeError(_)));

        let fallback = Value::int(7).unwrap();
        assert_eq!(
            builtin_getattr(&[object, name, fallback]).unwrap(),
            fallback
        );

        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
    }

    #[test]
    fn test_attribute_builtins_accept_heap_string_name() {
        let object = builtin_object(&[]).unwrap();
        let object_ptr = object.as_object_ptr().unwrap();
        let (name, name_ptr) = boxed_value(StringObject::new("heap_name"));

        builtin_setattr(&[object, name, Value::int(11).unwrap()]).unwrap();
        assert_eq!(builtin_getattr(&[object, name]).unwrap().as_int(), Some(11));
        assert!(builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());
        builtin_delattr(&[object, name]).unwrap();
        assert!(!builtin_hasattr(&[object, name]).unwrap().as_bool().unwrap());

        unsafe { drop_boxed(name_ptr) };
        unsafe { drop_boxed(object_ptr as *mut ShapedObject) };
    }

    #[test]
    fn test_getattr_vm_conversion_preserves_python_attribute_error_semantics() {
        let err = crate::error::RuntimeError::exception(
            ExceptionTypeId::AttributeError.as_u8() as u16,
            "_mock_methods",
        );
        assert!(matches!(
            runtime_error_to_builtin_error(err),
            BuiltinError::AttributeError(message) if message == "_mock_methods"
        ));
    }

    #[test]
    fn test_attribute_builtins_roundtrip_for_class_objects() {
        let (class_value, class_ptr) = class_value(PyClassObject::new_simple(intern("Example")));
        let name = Value::string(intern("field"));

        builtin_setattr(&[class_value, name, Value::int(42).unwrap()]).unwrap();
        assert_eq!(
            builtin_getattr(&[class_value, name]).unwrap().as_int(),
            Some(42)
        );
        assert!(
            builtin_hasattr(&[class_value, name])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        builtin_delattr(&[class_value, name]).unwrap();
        assert!(
            !builtin_hasattr(&[class_value, name])
                .unwrap()
                .as_bool()
                .unwrap()
        );

        unsafe { drop_class(class_ptr) };
    }

    #[test]
    fn test_builtin_getattr_reads_builtin_function_metadata() {
        fn metadata_builtin(args: &[Value]) -> Result<Value, BuiltinError> {
            Ok(Value::int(i64::try_from(args.len()).unwrap_or(i64::MAX))
                .expect("metadata builtin result should fit"))
        }

        let builtin = Box::new(crate::builtins::BuiltinFunctionObject::new(
            Arc::from("sys.gettrace"),
            metadata_builtin,
        ));
        let builtin_ptr = Box::into_raw(builtin);
        let builtin_value = Value::object_ptr(builtin_ptr as *const ());

        assert_eq!(
            builtin_getattr(&[builtin_value, Value::string(intern("__name__"))]).unwrap(),
            Value::string(intern("gettrace"))
        );
        assert_eq!(
            builtin_getattr(&[builtin_value, Value::string(intern("__qualname__"))]).unwrap(),
            Value::string(intern("sys.gettrace"))
        );
        assert!(
            builtin_getattr(&[builtin_value, Value::string(intern("__doc__"))])
                .unwrap()
                .is_none()
        );
        assert!(
            builtin_getattr(&[builtin_value, Value::string(intern("__self__"))])
                .unwrap()
                .is_none()
        );

        unsafe { drop_boxed(builtin_ptr) };
    }

    #[test]
    fn test_method_type_constructor_creates_bound_method() {
        fn callable_probe(_args: &[Value]) -> Result<Value, BuiltinError> {
            Ok(Value::none())
        }

        let (callable, callable_ptr) = boxed_value(crate::builtins::BuiltinFunctionObject::new(
            Arc::from("test.callable_probe"),
            callable_probe,
        ));
        let instance = Value::int(7).unwrap();

        let result =
            call_builtin_type(TypeId::METHOD, &[callable, instance]).expect("method() should bind");
        let result_ptr = result
            .as_object_ptr()
            .expect("method() should return a bound method object");
        assert_eq!(
            crate::ops::objects::extract_type_id(result_ptr),
            TypeId::METHOD
        );

        let bound = unsafe { &*(result_ptr as *const BoundMethod) };
        assert_eq!(bound.function(), callable);
        assert_eq!(bound.instance(), instance);

        unsafe {
            drop_boxed(result_ptr as *mut BoundMethod);
            drop_boxed(callable_ptr);
        }
    }

    #[test]
    fn test_method_type_constructor_rejects_none_instance() {
        fn callable_probe(_args: &[Value]) -> Result<Value, BuiltinError> {
            Ok(Value::none())
        }

        let (callable, callable_ptr) = boxed_value(crate::builtins::BuiltinFunctionObject::new(
            Arc::from("test.callable_probe"),
            callable_probe,
        ));

        let err = call_builtin_type(TypeId::METHOD, &[callable, Value::none()])
            .expect_err("method() should reject None instance");
        assert!(
            matches!(err, BuiltinError::TypeError(ref message) if message == "instance must not be None")
        );

        unsafe { drop_boxed(callable_ptr) };
    }

    #[test]
    fn test_method_type_constructor_rejects_non_callable_receiver() {
        let err = call_builtin_type(
            TypeId::METHOD,
            &[Value::int(42).unwrap(), Value::int(7).unwrap()],
        )
        .expect_err("method() should reject non-callable receiver");
        assert!(
            matches!(err, BuiltinError::TypeError(ref message) if message == "first argument must be callable")
        );
    }
}
