//! Type constructor builtins (int, float, str, bool, list, dict, etc.).

use super::BuiltinError;
use super::{
    builtin_bound_type_attribute_value_static, builtin_hash, builtin_instance_has_attribute,
    builtin_type_has_attribute, heap_type_attribute_value_static, heap_type_has_attribute,
};
use crate::VirtualMachine;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::import::ModuleObject;
use crate::ops::calls::{invoke_callable_value, value_supports_call_protocol};
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
    builtin_class_mro, class_id_to_type_id, global_class, global_class_bitmap,
    global_class_registry, register_global_class, type_new, type_new_with_metaclass,
    unregister_global_class,
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
use prism_runtime::types::string::{StringObject, clone_string_value, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::{LazyLock, Mutex};

mod numeric;

pub use numeric::{builtin_float, builtin_float_vm, builtin_int, builtin_int_vm};
pub(crate) use numeric::{
    builtin_float_getformat, builtin_int_from_bytes, builtin_int_from_bytes_vm,
    builtin_int_to_bytes, native_float_format_description,
};
use numeric::{builtin_int_kw, builtin_int_kw_vm};

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
        _ => crate::stdlib::typing::typing_marker_type_id(class_value),
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

    Ok(out)
}

const ABSTRACT_INSTANCE_CHECK_RECURSION_LIMIT: usize = 128;

#[inline]
fn recursion_limit_error(depth: usize) -> BuiltinError {
    BuiltinError::Raised(RuntimeError::recursion_error(depth))
}

#[inline]
fn issubclass_arg1_error() -> BuiltinError {
    BuiltinError::TypeError("issubclass() arg 1 must be a class".to_string())
}

#[inline]
fn tuple_items_vec(value: Value) -> Option<Vec<Value>> {
    let ptr = value.as_object_ptr()?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TUPLE {
        return None;
    }

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    Some(tuple.as_slice().to_vec())
}

#[inline]
fn class_value_for_class_id(class_id: ClassId) -> Option<Value> {
    if class_id.0 < TypeId::FIRST_USER_TYPE {
        return Some(builtin_type_object_for_type_id(class_id_to_type_id(
            class_id,
        )));
    }

    if let Some(exception_type_id) = crate::builtins::exception_type_id_for_proxy_class_id(class_id)
    {
        return crate::builtins::exception_type_value_for_id(exception_type_id);
    }

    global_class(class_id)
        .map(|class| Value::object_ptr(std::sync::Arc::as_ptr(&class) as *const ()))
}

fn real_class_direct_bases(value: Value) -> Option<Vec<Value>> {
    let type_id = class_value_to_type_id(value)?;
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return builtin_class_mro(type_id)
            .into_iter()
            .nth(1)
            .and_then(class_value_for_class_id)
            .map(|base| vec![base]);
    }

    let class = global_class(ClassId(type_id.raw()))?;
    let bases = if class.bases().is_empty() {
        vec![ClassId::OBJECT]
    } else {
        class.bases().to_vec()
    };
    bases.into_iter().map(class_value_for_class_id).collect()
}

fn lookup_class_like_bases_vm(
    vm: &mut VirtualMachine,
    value: Value,
    invalid_error: BuiltinError,
) -> Result<Option<Vec<Value>>, BuiltinError> {
    if let Some(bases) = real_class_direct_bases(value) {
        return Ok(Some(bases));
    }

    let bases_value =
        match crate::ops::objects::get_attribute_value(vm, value, &intern("__bases__")) {
            Ok(value) => value,
            Err(err) if err.is_attribute_error() => return Ok(None),
            Err(err) => return Err(runtime_error_to_builtin_error(err)),
        };

    tuple_items_vec(bases_value).map(Some).ok_or(invalid_error)
}

fn collect_class_spec_values_vm(
    vm: &mut VirtualMachine,
    value: Value,
    fn_name: &'static str,
    depth: usize,
    out: &mut Vec<Value>,
) -> Result<(), BuiltinError> {
    if depth >= ABSTRACT_INSTANCE_CHECK_RECURSION_LIMIT {
        return Err(recursion_limit_error(depth));
    }

    if class_value_to_type_id(value).is_some() {
        out.push(value);
        return Ok(());
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| class_spec_error(fn_name))?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::UNION => {
            let union = unsafe { &*(ptr as *const UnionTypeObject) };
            for &member in union.members() {
                out.push(class_value_for_type_id(member).ok_or_else(|| class_spec_error(fn_name))?);
            }
            Ok(())
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            for item in tuple.as_slice() {
                collect_class_spec_values_vm(vm, *item, fn_name, depth + 1, out)?;
            }
            Ok(())
        }
        _ => {
            if lookup_class_like_bases_vm(vm, value, class_spec_error(fn_name))?.is_some() {
                out.push(value);
                Ok(())
            } else {
                Err(class_spec_error(fn_name))
            }
        }
    }
}

fn parse_class_spec_values_vm(
    vm: &mut VirtualMachine,
    value: Value,
    fn_name: &'static str,
) -> Result<Vec<Value>, BuiltinError> {
    let mut out = Vec::new();
    collect_class_spec_values_vm(vm, value, fn_name, 0, &mut out)?;
    Ok(out)
}

fn validate_issubclass_arg_vm(vm: &mut VirtualMachine, value: Value) -> Result<(), BuiltinError> {
    if class_value_to_type_id(value).is_some() {
        return Ok(());
    }

    if lookup_class_like_bases_vm(vm, value, issubclass_arg1_error())?.is_some() {
        Ok(())
    } else {
        Err(issubclass_arg1_error())
    }
}

fn abstract_issubclass_value_vm(
    vm: &mut VirtualMachine,
    subclass: Value,
    target: Value,
    depth: usize,
    active: &mut FxHashSet<(u64, u64)>,
) -> Result<bool, BuiltinError> {
    if depth >= ABSTRACT_INSTANCE_CHECK_RECURSION_LIMIT {
        return Err(recursion_limit_error(depth));
    }
    if subclass.raw_bits() == target.raw_bits() {
        return Ok(true);
    }
    if let Some(target_type) = class_value_to_type_id(target)
        && class_value_is_subtype(subclass, target_type)
    {
        return Ok(true);
    }

    let key = (subclass.raw_bits(), target.raw_bits());
    if !active.insert(key) {
        return Err(recursion_limit_error(depth));
    }

    let Some(bases) = lookup_class_like_bases_vm(vm, subclass, issubclass_arg1_error())? else {
        active.remove(&key);
        return Ok(false);
    };

    for base in bases {
        if abstract_issubclass_value_vm(vm, base, target, depth + 1, active)? {
            active.remove(&key);
            return Ok(true);
        }
    }

    active.remove(&key);
    Ok(false)
}

fn abstract_isinstance_value_vm(
    vm: &mut VirtualMachine,
    instance: Value,
    target: Value,
) -> Result<bool, BuiltinError> {
    let class_value =
        match crate::ops::objects::get_attribute_value(vm, instance, &intern("__class__")) {
            Ok(value) => value,
            Err(err) if err.is_attribute_error() => return Ok(false),
            Err(err) => return Err(runtime_error_to_builtin_error(err)),
        };

    let mut active = FxHashSet::default();
    abstract_issubclass_value_vm(vm, class_value, target, 0, &mut active)
}

fn isinstance_single_target_vm(
    vm: &mut VirtualMachine,
    instance: Value,
    target: Value,
) -> Result<bool, BuiltinError> {
    if exact_class_match(instance, target) {
        return Ok(true);
    }
    if let Some(result) = collections_abc_isinstance_result(instance, target) {
        return Ok(result);
    }
    if let Some(result) = invoke_metaclass_check(vm, target, "__instancecheck__", instance)? {
        return Ok(result);
    }
    if class_value_to_type_id(target).is_some() && raw_isinstance_value(instance, target)? {
        return Ok(true);
    }
    abstract_isinstance_value_vm(vm, instance, target)
}

fn issubclass_single_target_vm(
    vm: &mut VirtualMachine,
    subclass: Value,
    target: Value,
) -> Result<bool, BuiltinError> {
    let subclass_is_real_class = class_value_to_type_id(subclass).is_some();
    if subclass_is_real_class {
        if is_exact_type_target(target) && raw_issubclass_value(subclass, target)? {
            return Ok(true);
        }
    }
    if let Some(result) = collections_abc_issubclass_result(subclass, target) {
        return Ok(result);
    }
    if let Some(result) = invoke_metaclass_check(vm, target, "__subclasscheck__", subclass)? {
        return Ok(result);
    }
    if subclass_is_real_class
        && class_value_to_type_id(target).is_some()
        && raw_issubclass_value(subclass, target)?
    {
        return Ok(true);
    }

    let mut active = FxHashSet::default();
    abstract_issubclass_value_vm(vm, subclass, target, 0, &mut active)
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

type CollectionsAbcKind = crate::stdlib::collections::abc::CollectionsAbcKind;

#[inline]
fn collections_abc_isinstance_result(instance: Value, target: Value) -> Option<bool> {
    let kind = crate::stdlib::collections::abc::abc_kind_for_class_value(target)?;
    Some(collections_abc_instance_matches(instance, target, kind))
}

#[inline]
fn collections_abc_issubclass_result(subclass: Value, target: Value) -> Option<bool> {
    let kind = crate::stdlib::collections::abc::abc_kind_for_class_value(target)?;
    if class_value_to_type_id(subclass).is_none() {
        return None;
    }
    Some(collections_abc_class_matches(subclass, target, kind))
}

fn collections_abc_instance_matches(
    instance: Value,
    target: Value,
    kind: CollectionsAbcKind,
) -> bool {
    if collections_abc_builtin_type_matches(value_type_id(instance), kind) {
        return true;
    }

    if kind == CollectionsAbcKind::Callable && value_supports_call_protocol(instance) {
        return true;
    }

    collections_abc_class_matches(value_type_object(instance), target, kind)
}

fn collections_abc_class_matches(
    class_value: Value,
    target: Value,
    kind: CollectionsAbcKind,
) -> bool {
    if let Some(target_type) = class_value_to_type_id(target)
        && class_value_is_subtype(class_value, target_type)
    {
        return true;
    }

    collections_abc_class_matches_kind(class_value, kind)
}

fn collections_abc_class_matches_kind(class_value: Value, kind: CollectionsAbcKind) -> bool {
    let Some(class_type) = class_value_to_type_id(class_value) else {
        return false;
    };

    if collections_abc_builtin_type_matches(class_type, kind) {
        return true;
    }

    if let Some(actual_kind) =
        crate::stdlib::collections::abc::abc_kind_for_class_value(class_value)
    {
        return collections_abc_kind_implies(actual_kind, kind);
    }

    match kind {
        CollectionsAbcKind::Awaitable => class_defines_all_specials(class_value, &["__await__"]),
        CollectionsAbcKind::Coroutine => {
            class_defines_all_specials(class_value, &["__await__", "send", "throw", "close"])
        }
        CollectionsAbcKind::AsyncIterable => {
            class_defines_all_specials(class_value, &["__aiter__"])
        }
        CollectionsAbcKind::AsyncIterator => {
            class_defines_all_specials(class_value, &["__aiter__", "__anext__"])
        }
        CollectionsAbcKind::AsyncGenerator => class_defines_all_specials(
            class_value,
            &["__aiter__", "__anext__", "asend", "athrow", "aclose"],
        ),
        CollectionsAbcKind::Hashable => class_hashable_for_collections_abc(class_value),
        CollectionsAbcKind::Iterable => class_defines_all_specials(class_value, &["__iter__"]),
        CollectionsAbcKind::Iterator => {
            class_defines_all_specials(class_value, &["__iter__", "__next__"])
        }
        CollectionsAbcKind::Generator => class_defines_all_specials(
            class_value,
            &["__iter__", "__next__", "send", "throw", "close"],
        ),
        CollectionsAbcKind::Reversible => {
            class_defines_all_specials(class_value, &["__reversed__"])
                || class_defines_all_specials(class_value, &["__len__", "__getitem__"])
        }
        CollectionsAbcKind::Sized => class_defines_all_specials(class_value, &["__len__"]),
        CollectionsAbcKind::Container => class_defines_all_specials(class_value, &["__contains__"]),
        CollectionsAbcKind::Callable => class_defines_all_specials(class_value, &["__call__"]),
        CollectionsAbcKind::Collection => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Sized)
                && collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Iterable)
                && collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Container)
        }
        CollectionsAbcKind::Set => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Collection)
                && class_defines_all_specials(class_value, &["__contains__"])
        }
        CollectionsAbcKind::MutableSet => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Set)
                && class_defines_all_specials(class_value, &["add", "discard"])
        }
        CollectionsAbcKind::Mapping => {
            class_defines_all_specials(class_value, &["__getitem__", "__iter__", "__len__"])
        }
        CollectionsAbcKind::MutableMapping => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Mapping)
                && class_defines_all_specials(class_value, &["__setitem__", "__delitem__"])
        }
        CollectionsAbcKind::MappingView => class_defines_all_specials(class_value, &["__len__"]),
        CollectionsAbcKind::KeysView | CollectionsAbcKind::ItemsView => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::MappingView)
                && collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Set)
        }
        CollectionsAbcKind::ValuesView => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::MappingView)
                && collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Collection)
        }
        CollectionsAbcKind::Sequence => {
            class_defines_all_specials(class_value, &["__len__", "__getitem__"])
        }
        CollectionsAbcKind::MutableSequence => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Sequence)
                && class_defines_all_specials(
                    class_value,
                    &["__setitem__", "__delitem__", "insert"],
                )
        }
        CollectionsAbcKind::ByteString => {
            collections_abc_class_matches_kind(class_value, CollectionsAbcKind::Sequence)
        }
        CollectionsAbcKind::Buffer => class_defines_all_specials(class_value, &["__buffer__"]),
    }
}

#[inline]
fn collections_abc_builtin_type_matches(type_id: TypeId, kind: CollectionsAbcKind) -> bool {
    use crate::stdlib::collections::abc::CollectionsAbcKind::*;

    match kind {
        Awaitable | Coroutine | AsyncIterable | AsyncIterator | AsyncGenerator => false,
        Hashable => !matches!(
            type_id,
            TypeId::LIST | TypeId::DICT | TypeId::SET | TypeId::BYTEARRAY
        ),
        Iterable => matches!(
            type_id,
            TypeId::LIST
                | TypeId::TUPLE
                | TypeId::STR
                | TypeId::RANGE
                | TypeId::DICT
                | TypeId::MAPPING_PROXY
                | TypeId::DICT_KEYS
                | TypeId::DICT_VALUES
                | TypeId::DICT_ITEMS
                | TypeId::SET
                | TypeId::FROZENSET
                | TypeId::BYTES
                | TypeId::BYTEARRAY
                | TypeId::MEMORYVIEW
                | TypeId::DEQUE
                | TypeId::ITERATOR
                | TypeId::ENUMERATE
                | TypeId::GENERATOR
        ),
        Iterator => matches!(
            type_id,
            TypeId::ITERATOR | TypeId::ENUMERATE | TypeId::GENERATOR
        ),
        Generator => type_id == TypeId::GENERATOR,
        Reversible => matches!(
            type_id,
            TypeId::LIST
                | TypeId::TUPLE
                | TypeId::STR
                | TypeId::RANGE
                | TypeId::BYTES
                | TypeId::BYTEARRAY
        ),
        Sized => matches!(
            type_id,
            TypeId::LIST
                | TypeId::TUPLE
                | TypeId::STR
                | TypeId::RANGE
                | TypeId::DICT
                | TypeId::MAPPING_PROXY
                | TypeId::DICT_KEYS
                | TypeId::DICT_VALUES
                | TypeId::DICT_ITEMS
                | TypeId::SET
                | TypeId::FROZENSET
                | TypeId::BYTES
                | TypeId::BYTEARRAY
                | TypeId::MEMORYVIEW
                | TypeId::DEQUE
        ),
        Container => matches!(
            type_id,
            TypeId::LIST
                | TypeId::TUPLE
                | TypeId::STR
                | TypeId::RANGE
                | TypeId::DICT
                | TypeId::MAPPING_PROXY
                | TypeId::DICT_KEYS
                | TypeId::DICT_VALUES
                | TypeId::DICT_ITEMS
                | TypeId::SET
                | TypeId::FROZENSET
                | TypeId::BYTES
                | TypeId::BYTEARRAY
                | TypeId::MEMORYVIEW
                | TypeId::DEQUE
        ),
        Callable => matches!(
            type_id,
            TypeId::FUNCTION
                | TypeId::METHOD
                | TypeId::CLOSURE
                | TypeId::STATICMETHOD
                | TypeId::TYPE
                | TypeId::BUILTIN_FUNCTION
                | TypeId::EXCEPTION_TYPE
                | TypeId::WRAPPER_DESCRIPTOR
                | TypeId::METHOD_WRAPPER
                | TypeId::METHOD_DESCRIPTOR
                | TypeId::CLASSMETHOD_DESCRIPTOR
        ),
        Collection => {
            collections_abc_builtin_type_matches(type_id, Sized)
                && collections_abc_builtin_type_matches(type_id, Iterable)
                && collections_abc_builtin_type_matches(type_id, Container)
        }
        Set => matches!(
            type_id,
            TypeId::SET | TypeId::FROZENSET | TypeId::DICT_KEYS | TypeId::DICT_ITEMS
        ),
        MutableSet => type_id == TypeId::SET,
        Mapping => matches!(type_id, TypeId::DICT | TypeId::MAPPING_PROXY),
        MutableMapping => type_id == TypeId::DICT,
        MappingView => matches!(
            type_id,
            TypeId::DICT_KEYS | TypeId::DICT_VALUES | TypeId::DICT_ITEMS
        ),
        KeysView => type_id == TypeId::DICT_KEYS,
        ItemsView => type_id == TypeId::DICT_ITEMS,
        ValuesView => type_id == TypeId::DICT_VALUES,
        Sequence => matches!(
            type_id,
            TypeId::LIST
                | TypeId::TUPLE
                | TypeId::STR
                | TypeId::RANGE
                | TypeId::BYTES
                | TypeId::BYTEARRAY
                | TypeId::MEMORYVIEW
        ),
        MutableSequence => matches!(type_id, TypeId::LIST | TypeId::BYTEARRAY),
        ByteString => matches!(type_id, TypeId::BYTES | TypeId::BYTEARRAY),
        Buffer => matches!(
            type_id,
            TypeId::BYTES | TypeId::BYTEARRAY | TypeId::MEMORYVIEW
        ),
    }
}

#[inline]
fn collections_abc_kind_implies(actual: CollectionsAbcKind, target: CollectionsAbcKind) -> bool {
    use crate::stdlib::collections::abc::CollectionsAbcKind::*;

    actual == target
        || matches!(
            (actual, target),
            (Coroutine, Awaitable)
                | (AsyncIterator, AsyncIterable)
                | (AsyncGenerator, AsyncIterator | AsyncIterable)
                | (Iterator, Iterable)
                | (Generator, Iterator | Iterable)
                | (Reversible, Iterable)
                | (Collection, Sized | Iterable | Container)
                | (Set, Collection | Sized | Iterable | Container)
                | (MutableSet, Set | Collection | Sized | Iterable | Container)
                | (Mapping, Collection | Sized | Iterable | Container)
                | (
                    MutableMapping,
                    Mapping | Collection | Sized | Iterable | Container
                )
                | (MappingView, Sized)
                | (
                    KeysView,
                    MappingView | Set | Collection | Sized | Iterable | Container
                )
                | (
                    ItemsView,
                    MappingView | Set | Collection | Sized | Iterable | Container
                )
                | (
                    ValuesView,
                    MappingView | Collection | Sized | Iterable | Container
                )
                | (
                    Sequence,
                    Reversible | Collection | Sized | Iterable | Container
                )
                | (
                    MutableSequence,
                    Sequence | Reversible | Collection | Sized | Iterable | Container
                )
                | (
                    ByteString,
                    Sequence | Reversible | Collection | Sized | Iterable | Container
                )
        )
}

fn class_defines_all_specials(class_value: Value, names: &[&str]) -> bool {
    names
        .iter()
        .all(|name| class_special_method_status(class_value, &intern(name)) == Some(true))
}

fn class_hashable_for_collections_abc(class_value: Value) -> bool {
    class_special_method_status(class_value, &intern("__hash__")) != Some(false)
}

fn class_special_method_status(class_value: Value, name: &InternedString) -> Option<bool> {
    let ptr = class_value.as_object_ptr()?;
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::TYPE => {
            if let Some(type_id) = builtin_type_object_type_id(ptr) {
                return builtin_special_method_status(type_id, name);
            }

            let class = class_object_from_ptr(ptr)?;
            for &class_id in class.mro() {
                if let Some(status) = class_id_special_method_status(class_id, name) {
                    return Some(status);
                }
            }
            None
        }
        TypeId::EXCEPTION_TYPE => builtin_special_method_status(TypeId::EXCEPTION, name),
        _ => None,
    }
}

fn class_id_special_method_status(class_id: ClassId, name: &InternedString) -> Option<bool> {
    if class_id == ClassId::OBJECT || class_id.0 < TypeId::FIRST_USER_TYPE {
        return builtin_special_method_status(class_id_to_type_id(class_id), name);
    }

    global_class(class_id)
        .and_then(|class| class.get_attr(name))
        .map(|value| !value.is_none())
}

#[inline]
fn builtin_special_method_status(type_id: TypeId, name: &InternedString) -> Option<bool> {
    if name.as_str() == "__hash__" {
        return Some(!matches!(
            type_id,
            TypeId::LIST | TypeId::DICT | TypeId::SET | TypeId::BYTEARRAY
        ));
    }

    if crate::ops::objects::builtin_instance_method_attr_exists(type_id, name)
        || builtin_instance_has_attribute(type_id, name)
    {
        Some(true)
    } else {
        None
    }
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
    if let Some(string) = value_as_string_ref(name) {
        return Ok(intern(string.as_str()));
    }

    Err(BuiltinError::TypeError(
        "attribute name must be string".to_string(),
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
        TypeId::ENUMERATE => super::itertools::builtin_enumerate(args),
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
        TypeId::RANGE => super::itertools::builtin_range_vm(vm, args),
        TypeId::ENUMERATE => super::itertools::builtin_enumerate_vm(vm, args),
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
            if args[0]
                .as_object_ptr()
                .is_some_and(|ptr| crate::ops::objects::extract_type_id(ptr) == TypeId::TUPLE)
            {
                return Ok(args[0]);
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
                        if let Some(hash) = source.stored_hash(key) {
                            copy.set_with_hash(key, value, hash);
                        } else {
                            copy.set(key, value);
                        }
                    }
                    return Ok(to_object_value(copy));
                }
            }

            if let Some(entries) = mapping_entries_with_vm(vm, args[0])? {
                let mut dict = DictObject::with_capacity(entries.len());
                for (key, value) in entries {
                    crate::ops::dict_access::dict_set_item(vm, &mut dict, key, value)
                        .map_err(runtime_error_to_builtin_error)?;
                }
                return Ok(to_object_value(dict));
            }

            let mut dict = DictObject::new();
            let mut sequence = iter_values_with_vm(vm, args[0])?;

            for (index, item) in sequence.drain(..).enumerate() {
                let (key, value) = dict_item_to_pair(item, index)?;
                crate::ops::dict_access::dict_set_item(vm, &mut dict, key, value)
                    .map_err(runtime_error_to_builtin_error)?;
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
        TypeId::DICT => builtin_dict_kw_vm(vm, positional, keywords),
        TypeId::STR => builtin_str_kw(positional, keywords),
        TypeId::DEQUE => builtin_deque_kw(positional, keywords),
        TypeId::ENUMERATE => super::itertools::builtin_enumerate_vm_kw(vm, positional, keywords),
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
        TypeId::ENUMERATE => super::itertools::builtin_enumerate_kw(positional, keywords),
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
    let explicit_doc = normalize(args.get(3).copied());
    let (doc, getter_doc) = property_doc_and_source(getter, explicit_doc);

    Ok(to_object_value(
        PropertyDescriptor::new_full_with_doc_source(getter, setter, deleter, doc, getter_doc),
    ))
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
    let getter = normalize(slots[0]);
    let setter = normalize(slots[1]);
    let deleter = normalize(slots[2]);
    let explicit_doc = normalize(slots[3]);
    let (doc, getter_doc) = property_doc_and_source(getter, explicit_doc);

    Ok(to_object_value(
        PropertyDescriptor::new_full_with_doc_source(getter, setter, deleter, doc, getter_doc),
    ))
}

#[inline]
fn property_doc_and_source(
    getter: Option<Value>,
    explicit_doc: Option<Value>,
) -> (Option<Value>, bool) {
    if explicit_doc.is_some() {
        return (explicit_doc, false);
    }

    let doc = getter.and_then(property_doc_from_accessor);
    (doc, doc.is_some())
}

#[inline]
pub(crate) fn property_doc_from_accessor(accessor: Value) -> Option<Value> {
    let ptr = accessor.as_object_ptr()?;
    let doc_name = intern("__doc__");

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let func = unsafe { &*(ptr as *const FunctionObject) };
            crate::ops::objects::function_attr_value(func, &doc_name)
                .filter(|value| !value.is_none())
        }
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(ptr as *const crate::builtins::BuiltinFunctionObject) };
            crate::ops::objects::builtin_function_attr_value(builtin, &doc_name)
                .filter(|value| !value.is_none())
        }
        _ => None,
    }
}

/// Builtin int constructor.
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
    if args[0]
        .as_object_ptr()
        .is_some_and(|ptr| crate::ops::objects::extract_type_id(ptr) == TypeId::TUPLE)
    {
        return Ok(args[0]);
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

pub(crate) fn builtin_dict_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    call_builtin_type_with_vm(vm, TypeId::DICT, args)
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

fn builtin_dict_kw_vm(
    vm: &mut VirtualMachine,
    positional: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let dict_value = call_builtin_type_with_vm(vm, TypeId::DICT, positional)?;
    let dict_ptr = dict_value
        .as_object_ptr()
        .expect("dict constructor should return an object-backed dict");
    let dict = unsafe { &mut *(dict_ptr as *mut DictObject) };

    for (name, value) in keywords {
        crate::ops::dict_access::dict_set_item(vm, dict, Value::string(intern(name)), *value)
            .map_err(runtime_error_to_builtin_error)?;
    }

    Ok(dict_value)
}

/// Builtin implementation backing `dict.fromkeys`.
pub(crate) fn builtin_dict_fromkeys_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let given = args.len().saturating_sub(1);
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dict.fromkeys() takes 1 or 2 arguments ({} given)",
            given
        )));
    }

    let class = dict_fromkeys_receiver_class(args[0])?;
    let value = args.get(2).copied().unwrap_or(Value::none());

    if class_value_to_type_id(class) == Some(TypeId::DICT) {
        let keys = iter_values_with_vm(vm, args[1])?;
        let mut dict = DictObject::with_capacity(keys.len());
        for key in keys {
            crate::ops::dict_access::dict_set_item(vm, &mut dict, key, value)
                .map_err(runtime_error_to_builtin_error)?;
        }
        return Ok(to_object_value(dict));
    }

    let result = invoke_callable_value(vm, class, &[]).map_err(runtime_error_to_builtin_error)?;
    let keys = iter_values_with_vm(vm, args[1])?;
    for key in keys {
        dict_fromkeys_set_item(vm, result, key, value)?;
    }

    Ok(result)
}

fn dict_fromkeys_receiver_class(receiver: Value) -> Result<Value, BuiltinError> {
    let class = if class_value_to_type_id(receiver).is_some() {
        receiver
    } else {
        value_type_object(receiver)
    };

    if class_value_is_subtype(class, TypeId::DICT) {
        return Ok(class);
    }

    Err(BuiltinError::TypeError(format!(
        "descriptor 'fromkeys' for 'dict' objects doesn't apply to a '{}' object",
        receiver.type_name()
    )))
}

fn dict_fromkeys_set_item(
    vm: &mut VirtualMachine,
    result: Value,
    key: Value,
    value: Value,
) -> Result<(), BuiltinError> {
    if let Some(ptr) = result.as_object_ptr()
        && crate::ops::objects::extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        return crate::ops::dict_access::dict_set_item(vm, dict, key, value)
            .map_err(runtime_error_to_builtin_error);
    }

    let setitem = crate::ops::objects::get_attribute_value(vm, result, &intern("__setitem__"))
        .map_err(runtime_error_to_builtin_error)?;
    invoke_callable_value(vm, setitem, &[key, value])
        .map(|_| ())
        .map_err(runtime_error_to_builtin_error)
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
        1 => (Value::none(), args[0], Value::none()),
        2 => (args[0], args[1], Value::none()),
        3 => (args[0], args[1], args[2]),
        _ => unreachable!(),
    };

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
    vm.record_published_class(class_id);

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
    builtin_type_new_with_keywords_vm(vm, args, &[])
}

pub(crate) fn builtin_type_new_with_keywords_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
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
    if let Err(err) = crate::ops::class::invoke_init_subclass_hook(vm, class_value, keywords) {
        unregister_global_class(class_id);
        return Err(runtime_error_to_builtin_error(err));
    }
    vm.record_published_class(class_id);

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
        if class_value_to_type_id(target).is_none() {
            continue;
        }
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

    let targets = parse_class_spec_values_vm(vm, args[1], "isinstance")?;
    for target in targets {
        if isinstance_single_target_vm(vm, args[0], target)? {
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
        if class_value_to_type_id(target).is_none() {
            continue;
        }
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
    validate_issubclass_arg_vm(vm, args[0])?;

    let targets = parse_class_spec_values_vm(vm, args[1], "issubclass")?;
    for target in targets {
        if issubclass_single_target_vm(vm, args[0], target)? {
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
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "list.__new__() takes at least 1 argument (0 given)".to_string(),
        ));
    }

    let class_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("list.__new__(X): X must be a type".to_string()))?;

    if class_type == TypeId::LIST {
        return Ok(to_object_value(ListObject::new()));
    }

    if !class_value_is_subtype(args[0], TypeId::LIST) {
        return Err(BuiltinError::TypeError(
            "list.__new__(X): X is not a subtype of list".to_string(),
        ));
    }

    let class_ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("list.__new__(X): X must be a type".to_string()))?;
    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "list.__new__() for builtin list subclasses is unsupported".to_string(),
        )
    })?;

    Ok(to_object_value(ShapedObject::new_list_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
    )))
}

pub(crate) fn builtin_list_init_vm(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "list.__init__() takes 1 or 2 arguments ({} given)",
            args.len()
        )));
    }

    let self_ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires a list object".to_string())
    })?;
    let list = crate::ops::objects::list_storage_mut_from_ptr(self_ptr).ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires a list object".to_string())
    })?;

    list.clear();
    if let Some(iterable) = args.get(1).copied() {
        let values = iter_values_with_vm(vm, iterable)?;
        list.extend(values);
    }

    Ok(Value::none())
}

pub(crate) fn builtin_dict_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(
            "dict.__new__() takes at least 1 argument (0 given)".to_string(),
        ));
    }

    let class_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("dict.__new__(X): X must be a type".to_string()))?;

    if class_type == TypeId::DICT {
        return Ok(to_object_value(DictObject::new()));
    }

    if !class_value_is_subtype(args[0], TypeId::DICT) {
        return Err(BuiltinError::TypeError(
            "dict.__new__(X): X is not a subtype of dict".to_string(),
        ));
    }

    let class_ptr = args[0]
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("dict.__new__(X): X must be a type".to_string()))?;
    let class = class_object_from_ptr(class_ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "dict.__new__() for builtin dict subclasses is unsupported".to_string(),
        )
    })?;

    Ok(to_object_value(ShapedObject::new_dict_backed(
        class.class_type_id(),
        class.instance_shape().clone(),
    )))
}

pub(crate) fn builtin_dict_init_vm_kw(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        let given = args.len().saturating_sub(1);
        return Err(BuiltinError::TypeError(format!(
            "dict.__init__() takes at most 1 argument ({given} given)"
        )));
    }

    let self_ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires a dict object".to_string())
    })?;
    let dict = crate::ops::objects::dict_storage_mut_from_ptr(self_ptr).ok_or_else(|| {
        BuiltinError::TypeError("descriptor '__init__' requires a dict object".to_string())
    })?;

    dict.clear();
    crate::ops::method_dispatch::dict_extend_with_vm_kw(
        vm,
        args[0],
        args.get(1).copied(),
        keywords,
        "__init__",
    )?;

    Ok(Value::none())
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
    let saved_exception_context =
        (vm.has_active_exception() || vm.has_exc_info()).then(|| vm.capture_exception_context());

    #[inline]
    fn restore_swallowed_exception_state(
        vm: &mut VirtualMachine,
        snapshot: Option<crate::vm::ExceptionContextSnapshot>,
    ) {
        if let Some(snapshot) = snapshot {
            vm.restore_exception_context(snapshot);
        } else if vm.has_active_exception() {
            vm.clear_active_exception();
            vm.clear_exception_state();
        }
    }

    match crate::ops::objects::get_attribute_value(vm, obj, &name) {
        Ok(_) => {
            restore_swallowed_exception_state(vm, saved_exception_context);
            Ok(Value::bool(true))
        }
        Err(err) => match runtime_error_to_builtin_error(err) {
            BuiltinError::AttributeError(_) => {
                restore_swallowed_exception_state(vm, saved_exception_context);
                Ok(Value::bool(false))
            }
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
