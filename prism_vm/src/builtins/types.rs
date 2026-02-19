//! Type constructor builtins (int, float, str, bool, list, dict, etc.).

use super::BuiltinError;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
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
fn to_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

#[inline]
fn to_frozenset_value(mut set: SetObject) -> Value {
    set.header.type_id = TypeId::FROZENSET;
    to_object_value(set)
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
        crate::ops::objects::extract_type_id(ptr)
    } else {
        TypeId::OBJECT
    }
}

#[inline]
fn class_value_to_type_id(class_value: Value) -> Option<TypeId> {
    let ptr = class_value.as_object_ptr()?;
    let object_type = crate::ops::objects::extract_type_id(ptr);

    match object_type {
        TypeId::TYPE => {
            let by_ptr = TYPE_IDS_BY_TYPE_OBJECT_PTR
                .lock()
                .expect("type-object pointer cache lock poisoned");
            Some(by_ptr.get(&(ptr as usize)).copied().unwrap_or(TypeId::TYPE))
        }
        TypeId::EXCEPTION_TYPE => Some(TypeId::EXCEPTION),
        _ => None,
    }
}

fn parse_class_spec(value: Value, fn_name: &'static str) -> Result<Vec<TypeId>, BuiltinError> {
    if let Some(type_id) = class_value_to_type_id(value) {
        return Ok(vec![type_id]);
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| class_spec_error(fn_name).clone())?;
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
fn class_spec_error(fn_name: &'static str) -> BuiltinError {
    if fn_name == "isinstance" {
        BuiltinError::TypeError(
            "isinstance() arg 2 must be a type, a tuple of types, or a union".to_string(),
        )
    } else {
        BuiltinError::TypeError(
            "issubclass() arg 2 must be a class, or tuple of classes".to_string(),
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
    false
}

fn iter_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iter) = super::iter_dispatch::get_iterator_mut(&value) {
        return Ok(iter.collect_remaining());
    }

    let mut iter = super::iter_dispatch::value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iter.collect_remaining())
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

/// Builtin int constructor.
pub fn builtin_int(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::int(0).unwrap());
    }
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "int() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }
    let arg = args[0];
    if let Some(i) = arg.as_int() {
        return Ok(arg);
    }
    if let Some(f) = arg.as_float() {
        return Value::int(f as i64)
            .ok_or_else(|| BuiltinError::OverflowError("int too large".to_string()));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::int(if b { 1 } else { 0 }).unwrap());
    }
    Err(BuiltinError::TypeError(
        "int() argument must be a string or number".to_string(),
    ))
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
    let arg = args[0];
    if let Some(f) = arg.as_float() {
        return Ok(arg);
    }
    if let Some(i) = arg.as_int() {
        return Ok(Value::float(i as f64));
    }
    if let Some(b) = arg.as_bool() {
        return Ok(Value::float(if b { 1.0 } else { 0.0 }));
    }
    Err(BuiltinError::TypeError(
        "float() argument must be a string or number".to_string(),
    ))
}

/// Builtin str constructor.
pub fn builtin_str(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() {
        return Ok(Value::string(intern("")));
    }
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "str() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let value = args[0];
    if value.is_string() {
        return Ok(value);
    }
    if let Some(ptr) = value.as_object_ptr() {
        if crate::ops::objects::extract_type_id(ptr) == TypeId::STR {
            return Ok(value);
        }
    }

    super::functions::builtin_repr(&[value])
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
    Ok(Value::bool(args[0].is_truthy()))
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
    Ok(to_object_value(SetObject::from_iter(values)))
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
    Ok(to_frozenset_value(SetObject::from_iter(values)))
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
        return Ok(type_object_for_type_id(value_type_id(args[0])));
    }

    Err(BuiltinError::NotImplemented(
        "type(name, bases, dict) is not yet implemented".to_string(),
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
    let actual = value_type_id(args[0]);
    let targets = parse_class_spec(args[1], "isinstance")?;
    Ok(Value::bool(
        targets.into_iter().any(|target| is_subtype(actual, target)),
    ))
}

/// Builtin issubclass function.
pub fn builtin_issubclass(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "issubclass() takes 2 arguments ({} given)",
            args.len()
        )));
    }
    let sub_type = class_value_to_type_id(args[0])
        .ok_or_else(|| BuiltinError::TypeError("issubclass() arg 1 must be a class".to_string()))?;
    let targets = parse_class_spec(args[1], "issubclass")?;

    Ok(Value::bool(
        targets
            .into_iter()
            .any(|target| is_subtype(sub_type, target)),
    ))
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
        if crate::ops::objects::extract_type_id(ptr) == TypeId::OBJECT {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            if let Some(value) = shaped.get_property_interned(&name) {
                return Ok(value);
            }
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
        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            let registry = shape_registry();
            shaped.set_property(name.clone(), value, registry);
            return Ok(Value::none());
        } else {
            return Err(BuiltinError::TypeError(format!(
                "'{}' object has no attribute '{}'",
                type_id.name(),
                name.as_str()
            )));
        }
    }

    // Primitive types don't support setattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name.as_str()
    )))
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
        if crate::ops::objects::extract_type_id(ptr) == TypeId::OBJECT {
            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            return Ok(Value::bool(shaped.has_property_interned(&name)));
        }
    }

    // For other types, always return False (no custom attributes)
    Ok(Value::bool(false))
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
        if type_id == TypeId::OBJECT {
            let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
            if shaped.delete_property_interned(&name) {
                return Ok(Value::none());
            } else {
                return Err(BuiltinError::AttributeError(format!(
                    "'object' object has no attribute '{}'",
                    name.as_str()
                )));
            }
        } else {
            return Err(BuiltinError::TypeError(format!(
                "'{}' object has no attribute '{}'",
                type_id.name(),
                name.as_str()
            )));
        }
    }

    // Primitive types don't support delattr
    Err(BuiltinError::TypeError(format!(
        "'{}' object has no attribute '{}'",
        get_type_name(obj),
        name.as_str()
    )))
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
    use prism_core::intern::interned_by_ptr;
    use prism_runtime::types::iter::IteratorObject;
    use prism_runtime::types::string::StringObject;

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

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
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
    fn test_type_builtin_three_arg_form_not_implemented() {
        let err =
            builtin_type(&[Value::string(intern("C")), Value::none(), Value::none()]).unwrap_err();
        assert!(matches!(err, BuiltinError::NotImplemented(_)));
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
}
