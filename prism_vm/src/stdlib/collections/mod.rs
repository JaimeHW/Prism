//! Python `collections` module implementation.
//!
//! Provides high-performance specialized container datatypes with optimal
//! algorithmic complexity and minimal memory overhead.
//!
//! # Data Structures
//!
//! | Type | Description | Key Operations |
//! |------|-------------|----------------|
//! | `deque` | Double-ended queue | O(1) append/pop both ends |
//! | `Counter` | Hashable element counter | O(1) increment/lookup |
//! | `defaultdict` | Dict with default factory | O(1) missing key handling |
//! | `OrderedDict` | Insertion-ordered dict | O(1) ops with order |
//! | `namedtuple` | Named field tuples | Immutable, memory-efficient |
//!
//! # Performance Characteristics
//!
//! ## deque
//!
//! - Implemented as a ring buffer with dynamic growth for cache efficiency
//! - O(1) `append()`, `appendleft()`, `pop()`, `popleft()`
//! - O(1) amortized `extend()`, `extendleft()`
//! - O(n) indexed access (use list for random access)
//!
//! ## Counter
//!
//! - Built on HashMap with optimized update path
//! - O(1) element counting and retrieval
//! - O(n) `most_common()` operation
//!
//! # Thread Safety
//!
//! All container types are **not** thread-safe by design (matching Python).
//! Use external synchronization if concurrent access is needed.

pub mod counter;
pub mod defaultdict;
pub mod deque;
pub mod ordereddict;

#[cfg(test)]
mod tests;

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject, get_iterator_mut, value_to_iterator};
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_parser::lexer::identifier::{is_id_continue, is_id_start};
use prism_parser::token::Keyword;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::Arc;
use std::sync::LazyLock;

// Re-export core types
pub use counter::Counter;
pub use deque::{Deque, DequeObject};

static COUNTER_CONSTRUCTOR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.Counter"), builtin_counter)
});
static NAMEDTUPLE_FACTORY: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.namedtuple"), builtin_namedtuple)
});
static ORDEREDDICT_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("collections.OrderedDict.__repr__"),
        ordered_dict_repr,
    )
});
static DEFAULTDICT_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("collections.defaultdict.__repr__"),
        defaultdict_repr,
    )
});
static CHAINMAP_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("collections.ChainMap.__repr__"), chainmap_repr)
});
static USERDICT_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("collections.UserDict.__repr__"), userdict_repr)
});
static USERLIST_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("collections.UserList.__repr__"), userlist_repr)
});
static USERSTRING_REPR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("collections.UserString.__repr__"),
        userstring_repr,
    )
});
static ORDEREDDICT_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "OrderedDict",
        &[ClassId(TypeId::DICT.raw())],
        &[("__repr__", builtin_value(&ORDEREDDICT_REPR))],
    )
});
static DEFAULTDICT_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "defaultdict",
        &[ClassId(TypeId::DICT.raw())],
        &[("__repr__", builtin_value(&DEFAULTDICT_REPR))],
    )
});
static CHAINMAP_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "ChainMap",
        &[],
        &[("__repr__", builtin_value(&CHAINMAP_REPR))],
    )
});
static USERDICT_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "UserDict",
        &[],
        &[("__repr__", builtin_value(&USERDICT_REPR))],
    )
});
static USERLIST_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "UserList",
        &[],
        &[("__repr__", builtin_value(&USERLIST_REPR))],
    )
});
static USERSTRING_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "UserString",
        &[],
        &[("__repr__", builtin_value(&USERSTRING_REPR))],
    )
});

// =============================================================================
// Collections Module
// =============================================================================

/// The collections module implementation.
pub struct CollectionsModule {
    attrs: Vec<Arc<str>>,
}

impl CollectionsModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("deque"),
                Arc::from("Counter"),
                Arc::from("defaultdict"),
                Arc::from("OrderedDict"),
                Arc::from("namedtuple"),
                Arc::from("ChainMap"),
                Arc::from("UserDict"),
                Arc::from("UserList"),
                Arc::from("UserString"),
            ],
        }
    }
}

impl Default for CollectionsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CollectionsModule {
    fn name(&self) -> &str {
        "collections"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "deque" => Ok(crate::builtins::builtin_type_object_for_type_id(
                TypeId::DEQUE,
            )),
            "Counter" => Ok(builtin_value(&COUNTER_CONSTRUCTOR)),
            "namedtuple" => Ok(builtin_value(&NAMEDTUPLE_FACTORY)),
            "ChainMap" => Ok(class_value(&CHAINMAP_CLASS)),
            "defaultdict" => Ok(class_value(&DEFAULTDICT_CLASS)),
            "OrderedDict" => Ok(class_value(&ORDEREDDICT_CLASS)),
            "UserDict" => Ok(class_value(&USERDICT_CLASS)),
            "UserList" => Ok(class_value(&USERLIST_CLASS)),
            "UserString" => Ok(class_value(&USERSTRING_CLASS)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'collections' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Create a new collections module instance.
pub fn create_module() -> CollectionsModule {
    CollectionsModule::new()
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::into_raw(Arc::clone(class)) as *const ())
}

fn build_native_collection_class(
    name: &str,
    bases: &[ClassId],
    attrs: &[(&str, Value)],
) -> Arc<PyClassObject> {
    let mut class = if bases.is_empty() {
        PyClassObject::new_simple(intern(name))
    } else {
        PyClassObject::new(intern(name), bases, |class_id| {
            (class_id == ClassId::OBJECT || class_id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(class_id_to_type_id(class_id))
                    .into_iter()
                    .collect()
            })
        })
        .unwrap_or_else(|err| panic!("failed to create collections.{name}: {err}"))
    };

    class.set_attr(intern("__module__"), Value::string(intern("collections")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    for &(attr_name, attr_value) in attrs {
        class.set_attr(intern(attr_name), attr_value);
    }
    class.add_flags(ClassFlags::INITIALIZED);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }

    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}

fn builtin_counter(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "Counter() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    if args.is_empty() {
        return Ok(leak_object_value(DictObject::new()));
    }

    if let Some(ptr) = args[0].as_object_ptr()
        && crate::ops::objects::extract_type_id(ptr) == TypeId::DICT
    {
        let source = unsafe { &*(ptr as *const DictObject) };
        let mut dict = DictObject::with_capacity(source.len());
        for (key, value) in source.iter() {
            dict.set(key, normalize_counter_value(value)?);
        }
        return Ok(leak_object_value(dict));
    }

    let values = collect_iterable_values_with_vm(vm, args[0])?;
    let counter = Counter::from_iter(values);
    let mut dict = DictObject::with_capacity(counter.len());
    for (element, count) in counter.iter() {
        dict.set(
            *element,
            normalize_counter_value(Value::int(*count).unwrap())?,
        );
    }
    Ok(leak_object_value(dict))
}

fn builtin_namedtuple(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "namedtuple() takes 2 to 5 arguments ({} given)",
            args.len()
        )));
    }

    let type_name = extract_string_arg(args[0], "typename")?;
    validate_namedtuple_type_name(&type_name)?;

    let rename = args
        .get(2)
        .copied()
        .is_some_and(crate::truthiness::is_truthy);
    let fields = normalize_namedtuple_fields(parse_field_names(vm, args[1])?, rename)?;
    if fields.is_empty() {
        return Err(BuiltinError::ValueError(
            "namedtuple() requires at least one field".to_string(),
        ));
    }

    let defaults_value = args.get(3).copied().filter(|value| !value.is_none());
    let module_name = if let Some(module) = args.get(4) {
        extract_string_arg(*module, "module")?
    } else {
        "collections".to_string()
    };

    let class = Arc::new(PyClassObject::new_simple(intern(&type_name)));
    let fields_tuple = tuple_value_from_interned(&fields);
    class.set_attr(intern("_fields"), fields_tuple);
    class.set_attr(intern("__match_args__"), fields_tuple);
    class.set_attr(intern("__module__"), Value::string(intern(&module_name)));
    class.set_attr(
        intern("_field_defaults"),
        build_field_defaults(vm, &fields, defaults_value)?,
    );

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    Ok(Value::object_ptr(Arc::into_raw(class) as *const ()))
}

fn collect_iterable_values(value: Value) -> Result<Vec<Value>, BuiltinError> {
    if let Some(iterator) = get_iterator_mut(&value) {
        return Ok(iterator.collect_remaining());
    }

    let mut iterator = value_to_iterator(&value).map_err(BuiltinError::from)?;
    Ok(iterator.collect_remaining())
}

fn collect_iterable_values_with_vm(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Vec<Value>, BuiltinError> {
    crate::ops::iteration::collect_iterable_values(vm, value)
        .map_err(|err| BuiltinError::TypeError(err.to_string()))
}

fn normalize_counter_value(value: Value) -> Result<Value, BuiltinError> {
    if let Some(i) = value.as_int() {
        return Value::int(i).ok_or_else(|| {
            BuiltinError::OverflowError("Counter value exceeds supported integer range".to_string())
        });
    }
    if let Some(b) = value.as_bool() {
        return Ok(Value::int(if b { 1 } else { 0 }).unwrap());
    }

    Err(BuiltinError::TypeError(
        "Counter mapping values must be integers".to_string(),
    ))
}

fn parse_field_names(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Vec<InternedString>, BuiltinError> {
    let text = extract_string_value(value);
    if let Ok(text) = text {
        let names = text
            .split(|ch: char| ch == ',' || ch.is_whitespace())
            .filter(|name| !name.is_empty())
            .map(intern)
            .collect::<Vec<_>>();
        return Ok(names);
    }

    let values = collect_iterable_values_with_vm(vm, value)?;
    let mut names = Vec::with_capacity(values.len());
    for value in values {
        names.push(intern(&extract_string_arg(value, "field name")?));
    }
    Ok(names)
}

fn tuple_value_from_interned(names: &[InternedString]) -> Value {
    let values = names
        .iter()
        .map(|name| Value::string(name.clone()))
        .collect::<Vec<_>>();
    leak_object_value(TupleObject::from_vec(values))
}

fn build_field_defaults(
    vm: &mut VirtualMachine,
    fields: &[InternedString],
    defaults_value: Option<Value>,
) -> Result<Value, BuiltinError> {
    let mut defaults = DictObject::new();
    let Some(defaults_value) = defaults_value else {
        return Ok(leak_object_value(defaults));
    };

    let values = collect_iterable_values_with_vm(vm, defaults_value)?;
    if values.len() > fields.len() {
        return Err(BuiltinError::TypeError(
            "namedtuple() defaults cannot exceed field count".to_string(),
        ));
    }

    let field_offset = fields.len() - values.len();
    for (field_name, default_value) in fields[field_offset..].iter().zip(values.into_iter()) {
        defaults.set(Value::string(field_name.clone()), default_value);
    }

    Ok(leak_object_value(defaults))
}

fn normalize_namedtuple_fields(
    fields: Vec<InternedString>,
    rename: bool,
) -> Result<Vec<InternedString>, BuiltinError> {
    let mut normalized = Vec::with_capacity(fields.len());
    let mut seen = std::collections::HashSet::with_capacity(fields.len());

    for (index, field) in fields.into_iter().enumerate() {
        let candidate = field.as_str();
        let invalid = candidate.starts_with('_')
            || !is_valid_namedtuple_identifier(candidate)
            || seen.contains(candidate);
        let field_name = if invalid {
            if rename {
                intern(&format!("_{index}"))
            } else if seen.contains(candidate) {
                return Err(BuiltinError::ValueError(format!(
                    "namedtuple() field name '{candidate}' is duplicated",
                )));
            } else if candidate.starts_with('_') {
                return Err(BuiltinError::ValueError(format!(
                    "namedtuple() field name '{candidate}' cannot start with an underscore",
                )));
            } else {
                return Err(BuiltinError::ValueError(format!(
                    "namedtuple() field name '{candidate}' is not a valid identifier",
                )));
            }
        } else {
            field
        };

        seen.insert(field_name.as_str().to_string());
        normalized.push(field_name);
    }

    Ok(normalized)
}

fn validate_namedtuple_type_name(type_name: &str) -> Result<(), BuiltinError> {
    if !is_valid_namedtuple_identifier(type_name) {
        return Err(BuiltinError::ValueError(format!(
            "namedtuple() typename '{type_name}' is not a valid identifier",
        )));
    }
    Ok(())
}

fn is_valid_namedtuple_identifier(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !is_id_start(first) {
        return false;
    }
    if !chars.all(is_id_continue) {
        return false;
    }
    Keyword::from_str(name).is_none()
}

fn extract_string_arg(value: Value, label: &str) -> Result<String, BuiltinError> {
    extract_string_value(value)
        .map_err(|_| BuiltinError::TypeError(format!("namedtuple() {label} must be a string")))
}

fn extract_string_value(value: Value) -> Result<String, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError("expected string".to_string()))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError("expected string".to_string()))?;
        return Ok(interned.as_str().to_string());
    }

    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("expected string".to_string()))?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError("expected string".to_string()));
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(string.as_str().to_string())
}

fn collection_repr_text(value: Value) -> Result<String, BuiltinError> {
    extract_string_value(crate::builtins::builtin_repr(&[value])?)
}

fn expect_collection_instance(
    args: &[Value],
    descriptor_name: &str,
) -> Result<*const (), BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "{descriptor_name}() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires an instance"
        ))
    })
}

fn expect_collection_dict(
    args: &[Value],
    descriptor_name: &str,
) -> Result<&'static DictObject, BuiltinError> {
    let ptr = expect_collection_instance(args, descriptor_name)?;
    crate::ops::objects::dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires a dict-backed collections object"
        ))
    })
}

fn dict_repr_body(dict: &DictObject) -> Result<String, BuiltinError> {
    let mut out = String::from("{");
    for (index, (key, value)) in dict.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&collection_repr_text(key)?);
        out.push_str(": ");
        out.push_str(&collection_repr_text(value)?);
    }
    out.push('}');
    Ok(out)
}

fn shaped_property_repr(ptr: *const (), name: &str) -> Result<Option<String>, BuiltinError> {
    if crate::ops::objects::extract_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
        return Ok(None);
    }

    let object = unsafe { &*(ptr as *const ShapedObject) };
    object
        .get_property(name)
        .map(collection_repr_text)
        .transpose()
}

fn ordered_dict_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let dict = expect_collection_dict(args, "__repr__")?;
    let mut out = String::from("OrderedDict(");
    if dict.is_empty() {
        out.push(')');
        return Ok(Value::string(intern(&out)));
    }

    out.push('[');
    for (index, (key, value)) in dict.iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push('(');
        out.push_str(&collection_repr_text(key)?);
        out.push_str(", ");
        out.push_str(&collection_repr_text(value)?);
        out.push(')');
    }
    out.push_str("])");
    Ok(Value::string(intern(&out)))
}

fn defaultdict_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let dict = crate::ops::objects::dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "descriptor '__repr__' requires a dict-backed collections object".to_string(),
        )
    })?;
    let default_factory =
        shaped_property_repr(ptr, "default_factory")?.unwrap_or_else(|| "None".to_string());
    let result = format!("defaultdict({default_factory}, {})", dict_repr_body(dict)?);
    Ok(Value::string(intern(&result)))
}

fn chainmap_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let maps_repr = shaped_property_repr(ptr, "maps")?.unwrap_or_else(|| "()".to_string());
    Ok(Value::string(intern(&format!("ChainMap({maps_repr})"))))
}

fn userdict_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let data_repr = shaped_property_repr(ptr, "data")?.unwrap_or_else(|| "{}".to_string());
    Ok(Value::string(intern(&format!("UserDict({data_repr})"))))
}

fn userlist_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let data_repr = shaped_property_repr(ptr, "data")?.unwrap_or_else(|| "[]".to_string());
    Ok(Value::string(intern(&format!("UserList({data_repr})"))))
}

fn userstring_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let data_repr = shaped_property_repr(ptr, "data")?.unwrap_or_else(|| "''".to_string());
    Ok(Value::string(intern(&format!("UserString({data_repr})"))))
}

#[cfg(test)]
mod module_tests {
    use super::*;
    use crate::ops::objects::extract_type_id;

    fn dict_from_value(value: Value) -> *mut DictObject {
        value
            .as_object_ptr()
            .expect("dict-backed value should be object") as *mut DictObject
    }

    #[test]
    fn test_get_attr_exposes_counter_namedtuple_and_collection_type_objects() {
        let module = CollectionsModule::new();

        assert!(
            module
                .get_attr("Counter")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
        assert!(
            module
                .get_attr("namedtuple")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );

        for name in [
            "ChainMap",
            "defaultdict",
            "OrderedDict",
            "UserDict",
            "UserList",
            "UserString",
        ] {
            let class_value = module
                .get_attr(name)
                .unwrap_or_else(|_| panic!("collections.{name} should resolve"));
            let class_ptr = class_value
                .as_object_ptr()
                .unwrap_or_else(|| panic!("collections.{name} should be a class"));
            assert_eq!(extract_type_id(class_ptr), TypeId::TYPE);

            unsafe {
                drop(Arc::from_raw(class_ptr as *const PyClassObject));
            }
        }
    }

    #[test]
    fn test_pprint_sensitive_collection_reprs_are_distinct_from_builtin_dispatch_keys() {
        let ordered = ORDEREDDICT_CLASS
            .get_attr(&intern("__repr__"))
            .expect("OrderedDict should define __repr__");
        let default_dict = DEFAULTDICT_CLASS
            .get_attr(&intern("__repr__"))
            .expect("defaultdict should define __repr__");
        let chainmap = CHAINMAP_CLASS
            .get_attr(&intern("__repr__"))
            .expect("ChainMap should define __repr__");
        let user_dict = USERDICT_CLASS
            .get_attr(&intern("__repr__"))
            .expect("UserDict should define __repr__");

        assert_ne!(ordered, default_dict);
        assert_ne!(ordered, chainmap);
        assert_ne!(default_dict, chainmap);
        assert_ne!(user_dict, chainmap);
    }

    #[test]
    fn test_counter_builtin_counts_iterables() {
        let mut vm = VirtualMachine::new();
        let value = builtin_counter(
            &mut vm,
            &[leak_object_value(TupleObject::from_slice(&[
                Value::int(1).unwrap(),
                Value::int(1).unwrap(),
                Value::int(2).unwrap(),
            ]))],
        )
        .expect("Counter should construct");

        let dict_ptr = dict_from_value(value);
        let dict = unsafe { &*dict_ptr };
        assert_eq!(dict.get(Value::int(1).unwrap()).unwrap().as_int(), Some(2));
        assert_eq!(dict.get(Value::int(2).unwrap()).unwrap().as_int(), Some(1));

        unsafe {
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_counter_builtin_accepts_mapping_input() {
        let mut vm = VirtualMachine::new();
        let mut mapping = DictObject::new();
        mapping.set(Value::string(intern("x")), Value::int(3).unwrap());
        let mapping_value = leak_object_value(mapping);

        let value =
            builtin_counter(&mut vm, &[mapping_value]).expect("Counter should copy mapping");
        let dict_ptr = dict_from_value(value);
        let dict = unsafe { &*dict_ptr };
        assert_eq!(
            dict.get(Value::string(intern("x"))).unwrap().as_int(),
            Some(3)
        );

        unsafe {
            drop(Box::from_raw(
                mapping_value.as_object_ptr().unwrap() as *mut DictObject
            ));
            drop(Box::from_raw(dict_ptr));
        }
    }

    #[test]
    fn test_namedtuple_factory_builds_type_with_declared_fields() {
        let mut vm = VirtualMachine::new();
        let class_value = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("Pair")),
                Value::string(intern("left right")),
            ],
        )
        .expect("namedtuple should construct class");

        let class_ptr = class_value
            .as_object_ptr()
            .expect("class value should be object");
        assert_eq!(
            crate::ops::objects::extract_type_id(class_ptr),
            TypeId::TYPE
        );

        let class = unsafe { &*(class_ptr as *const PyClassObject) };
        assert_eq!(class.name().as_str(), "Pair");

        let fields_value = class.get_attr(&intern("_fields")).expect("fields tuple");
        let fields_ptr = fields_value
            .as_object_ptr()
            .expect("_fields should be tuple object");
        let fields = unsafe { &*(fields_ptr as *const TupleObject) };
        assert_eq!(fields.len(), 2);

        unsafe {
            drop(Box::from_raw(fields_ptr as *mut TupleObject));
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
    }

    #[test]
    fn test_namedtuple_factory_records_module_and_defaults() {
        let mut vm = VirtualMachine::new();
        let defaults = leak_object_value(TupleObject::from_slice(&[Value::int(7).unwrap()]));
        let class_value = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("Point")),
                Value::string(intern("x y")),
                Value::bool(false),
                defaults,
                Value::string(intern("demo.module")),
            ],
        )
        .expect("namedtuple should construct class with defaults");

        let class_ptr = class_value
            .as_object_ptr()
            .expect("class value should be object");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };

        let module_value = class.get_attr(&intern("__module__")).expect("module attr");
        assert_eq!(extract_string_value(module_value).unwrap(), "demo.module");

        let field_defaults = class
            .get_attr(&intern("_field_defaults"))
            .expect("field defaults attr");
        let dict_ptr = dict_from_value(field_defaults);
        let dict = unsafe { &*dict_ptr };
        assert_eq!(
            dict.get(Value::string(intern("y"))).unwrap().as_int(),
            Some(7)
        );

        unsafe {
            drop(Box::from_raw(
                defaults.as_object_ptr().unwrap() as *mut TupleObject
            ));
            drop(Box::from_raw(dict_ptr));
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
    }
}
