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
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{
    dict_storage_mut_from_ptr, dict_storage_ref_from_ptr, extract_type_id, get_attribute_value,
    list_storage_ref_from_ptr, tuple_storage_ref_from_ptr,
};
use crate::stdlib::exceptions::types::ExceptionTypeId;
use crate::truthiness::try_is_truthy;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_parser::lexer::identifier::{is_id_continue, is_id_start};
use prism_parser::token::Keyword;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::{ClassMethodDescriptor, PropertyDescriptor};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, global_class, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
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
static NAMEDTUPLE_NEW: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("collections.namedtuple.__new__"),
        builtin_namedtuple_new,
    )
});
static NAMEDTUPLE_INIT: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(
        Arc::from("collections.namedtuple.__init__"),
        builtin_namedtuple_init,
    )
});
static NAMEDTUPLE_MAKE: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.namedtuple._make"),
        builtin_namedtuple_make,
    )
});
static NAMEDTUPLE_MAKE_DESCRIPTOR: LazyLock<Value> = LazyLock::new(|| {
    let descriptor = Box::new(ClassMethodDescriptor::new(builtin_value(&NAMEDTUPLE_MAKE)));
    Value::object_ptr(Box::into_raw(descriptor) as *const ())
});
static NAMEDTUPLE_ASDICT: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.namedtuple._asdict"),
        builtin_namedtuple_asdict,
    )
});
static NAMEDTUPLE_REPLACE: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("collections.namedtuple._replace"),
        builtin_namedtuple_replace,
    )
});
static NAMEDTUPLE_GETNEWARGS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.namedtuple.__getnewargs__"),
        builtin_namedtuple_getnewargs,
    )
});
static NAMEDTUPLE_FIELD_GET: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("collections._tuplegetter.__get__"),
        namedtuple_field_get,
    )
});
static NAMEDTUPLE_FIELD_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_collection_class(
        "_tuplegetter",
        &[],
        &[("__get__", builtin_value(&NAMEDTUPLE_FIELD_GET))],
    )
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
static CHAINMAP_INIT: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.__init__"), chainmap_init)
});
static CHAINMAP_MISSING: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("collections.ChainMap.__missing__"),
        chainmap_missing,
    )
});
static CHAINMAP_GETITEM: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.__getitem__"),
        chainmap_getitem,
    )
});
static CHAINMAP_GET: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.get"), chainmap_get)
});
static CHAINMAP_LEN: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.__len__"), chainmap_len)
});
static CHAINMAP_ITER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.__iter__"), chainmap_iter)
});
static CHAINMAP_CONTAINS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.__contains__"),
        chainmap_contains,
    )
});
static CHAINMAP_BOOL: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.__bool__"), chainmap_bool)
});
static CHAINMAP_KEYS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.keys"), chainmap_keys)
});
static CHAINMAP_ITEMS: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.items"), chainmap_items)
});
static CHAINMAP_VALUES: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.values"), chainmap_values)
});
static CHAINMAP_SETITEM: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.__setitem__"),
        chainmap_setitem,
    )
});
static CHAINMAP_DELITEM: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.__delitem__"),
        chainmap_delitem,
    )
});
static CHAINMAP_COPY: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.copy"), chainmap_copy)
});
static CHAINMAP_NEW_CHILD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("collections.ChainMap.new_child"),
        chainmap_new_child,
    )
});
static CHAINMAP_FROMKEYS_FN: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.fromkeys"),
        chainmap_fromkeys,
    )
});
static CHAINMAP_FROMKEYS: LazyLock<Value> = LazyLock::new(|| {
    let descriptor = Box::new(ClassMethodDescriptor::new(builtin_value(
        &CHAINMAP_FROMKEYS_FN,
    )));
    Value::object_ptr(Box::into_raw(descriptor) as *const ())
});
static CHAINMAP_PARENTS_GETTER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("collections.ChainMap.parents"),
        chainmap_parents_getter,
    )
});
static CHAINMAP_PARENTS: LazyLock<Value> = LazyLock::new(|| {
    let descriptor = Box::new(PropertyDescriptor::new_getter(builtin_value(
        &CHAINMAP_PARENTS_GETTER,
    )));
    Value::object_ptr(Box::into_raw(descriptor) as *const ())
});
static CHAINMAP_POP: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.pop"), chainmap_pop)
});
static CHAINMAP_POPITEM: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.popitem"), chainmap_popitem)
});
static CHAINMAP_CLEAR: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("collections.ChainMap.clear"), chainmap_clear)
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
        &[
            ("__init__", builtin_value(&CHAINMAP_INIT)),
            ("__missing__", builtin_value(&CHAINMAP_MISSING)),
            ("__getitem__", builtin_value(&CHAINMAP_GETITEM)),
            ("get", builtin_value(&CHAINMAP_GET)),
            ("__len__", builtin_value(&CHAINMAP_LEN)),
            ("__iter__", builtin_value(&CHAINMAP_ITER)),
            ("__contains__", builtin_value(&CHAINMAP_CONTAINS)),
            ("__bool__", builtin_value(&CHAINMAP_BOOL)),
            ("keys", builtin_value(&CHAINMAP_KEYS)),
            ("items", builtin_value(&CHAINMAP_ITEMS)),
            ("values", builtin_value(&CHAINMAP_VALUES)),
            ("__setitem__", builtin_value(&CHAINMAP_SETITEM)),
            ("__delitem__", builtin_value(&CHAINMAP_DELITEM)),
            ("__repr__", builtin_value(&CHAINMAP_REPR)),
            ("copy", builtin_value(&CHAINMAP_COPY)),
            ("__copy__", builtin_value(&CHAINMAP_COPY)),
            ("new_child", builtin_value(&CHAINMAP_NEW_CHILD)),
            ("fromkeys", *CHAINMAP_FROMKEYS),
            ("parents", *CHAINMAP_PARENTS),
            ("pop", builtin_value(&CHAINMAP_POP)),
            ("popitem", builtin_value(&CHAINMAP_POPITEM)),
            ("clear", builtin_value(&CHAINMAP_CLEAR)),
        ],
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
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

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

    let mut class = PyClassObject::new(intern(&type_name), &[ClassId(TypeId::TUPLE.raw())], |id| {
        (id == ClassId::OBJECT || id.0 < TypeId::FIRST_USER_TYPE).then(|| {
            builtin_class_mro(class_id_to_type_id(id))
                .into_iter()
                .collect()
        })
    })
    .map_err(|err| BuiltinError::TypeError(format!("failed to create namedtuple class: {err}")))?;
    class.add_flags(
        ClassFlags::INITIALIZED
            | ClassFlags::HAS_NEW
            | ClassFlags::HAS_INIT
            | ClassFlags::NATIVE_HEAPTYPE,
    );
    let class = Arc::new(class);
    let fields_tuple = tuple_value_from_interned(&fields);
    class.set_attr(intern("_fields"), fields_tuple);
    class.set_attr(intern("__match_args__"), fields_tuple);
    class.set_attr(intern("__module__"), Value::string(intern(&module_name)));
    class.set_attr(intern("__new__"), builtin_value(&NAMEDTUPLE_NEW));
    class.set_attr(intern("__init__"), builtin_value(&NAMEDTUPLE_INIT));
    class.set_attr(intern("_make"), *NAMEDTUPLE_MAKE_DESCRIPTOR);
    class.set_attr(intern("_asdict"), builtin_value(&NAMEDTUPLE_ASDICT));
    class.set_attr(intern("_replace"), builtin_value(&NAMEDTUPLE_REPLACE));
    class.set_attr(
        intern("__getnewargs__"),
        builtin_value(&NAMEDTUPLE_GETNEWARGS),
    );
    class.set_attr(
        intern("_field_defaults"),
        build_field_defaults(vm, &fields, defaults_value)?,
    );
    class.set_attr(
        intern("__doc__"),
        Value::string(intern(&namedtuple_class_doc(&type_name, &fields))),
    );

    for (index, field_name) in fields.iter().enumerate() {
        class.set_attr(
            field_name.clone(),
            new_namedtuple_field_descriptor(field_name, index),
        );
    }

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);

    Ok(Value::object_ptr(Arc::into_raw(class) as *const ()))
}

fn builtin_namedtuple_new(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let class = namedtuple_class_from_type_value(args.first().copied())?;
    let schema_class =
        namedtuple_schema_class_for_slot(class.as_ref(), "__new__", &NAMEDTUPLE_NEW)?;
    let fields = namedtuple_field_names(schema_class.as_ref())?;
    let values = bind_namedtuple_arguments(
        schema_class.name().as_str(),
        &fields,
        namedtuple_field_defaults(schema_class.as_ref()),
        &args[1..],
        keywords,
    )?;

    let instance = ShapedObject::new_tuple_backed(
        class.class_type_id(),
        Arc::clone(class.instance_shape()),
        TupleObject::from_vec(values),
    );
    Ok(leak_object_value(instance))
}

fn builtin_namedtuple_init(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let (instance_ptr, class) = namedtuple_instance_and_class(args.first().copied())?;
    let schema_class =
        namedtuple_schema_class_for_slot(class.as_ref(), "__init__", &NAMEDTUPLE_INIT)?;
    let fields = namedtuple_field_names(schema_class.as_ref())?;
    let instance = unsafe { &mut *instance_ptr };
    if let Some(existing) = instance.tuple_backing() {
        if existing.len() == fields.len() {
            return Ok(Value::none());
        }
    }

    let values = bind_namedtuple_arguments(
        schema_class.name().as_str(),
        &fields,
        namedtuple_field_defaults(schema_class.as_ref()),
        &args[1..],
        keywords,
    )?;

    instance.set_tuple_backing(TupleObject::from_vec(values));

    Ok(Value::none())
}

fn builtin_namedtuple_make(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "_make() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let class = namedtuple_class_from_type_value(args.first().copied())?;
    let schema_class =
        namedtuple_schema_class_for_slot(class.as_ref(), "__new__", &NAMEDTUPLE_NEW)?;
    let fields = namedtuple_field_names(schema_class.as_ref())?;
    let values = collect_iterable_values_runtime(vm, args[1])
        .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    if values.len() != fields.len() {
        return Err(BuiltinError::TypeError(format!(
            "Expected {} arguments, got {}",
            fields.len(),
            values.len()
        )));
    }

    let mut call_args = Vec::with_capacity(values.len() + 1);
    call_args.push(Value::object_ptr(Arc::into_raw(class) as *const ()));
    call_args.extend(values);
    builtin_namedtuple_new(&call_args, &[])
}

fn builtin_namedtuple_asdict(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_asdict() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let (_, class) = namedtuple_instance_and_class(args.first().copied())?;
    let fields = namedtuple_field_names(class.as_ref())?;
    let mut dict = DictObject::with_capacity(fields.len());
    for field in fields {
        let value = get_attribute_value(vm, args[0], &field)
            .map_err(crate::builtins::runtime_error_to_builtin_error)?;
        dict.set(Value::string(field), value);
    }
    Ok(leak_object_value(dict))
}

fn builtin_namedtuple_replace(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "_replace() takes exactly 0 positional arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let (_, class) = namedtuple_instance_and_class(args.first().copied())?;
    let schema_class =
        namedtuple_schema_class_for_slot(class.as_ref(), "_replace", &NAMEDTUPLE_REPLACE)?;
    let fields = namedtuple_field_names(schema_class.as_ref())?;
    let mut values = Vec::with_capacity(fields.len());
    for field in &fields {
        values.push(
            get_attribute_value(vm, args[0], field)
                .map_err(crate::builtins::runtime_error_to_builtin_error)?,
        );
    }

    for &(name, value) in keywords {
        let Some(index) = fields.iter().position(|field| field.as_str() == name) else {
            return Err(BuiltinError::ValueError(format!(
                "Got unexpected field names: ['{name}']"
            )));
        };
        values[index] = value;
    }

    let mut call_args = Vec::with_capacity(values.len() + 1);
    call_args.push(Value::object_ptr(Arc::into_raw(class) as *const ()));
    call_args.extend(values);
    builtin_namedtuple_new(&call_args, &[])
}

fn builtin_namedtuple_getnewargs(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__getnewargs__() takes exactly 0 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let values = collect_iterable_values_runtime(vm, args[0])
        .map_err(crate::builtins::runtime_error_to_builtin_error)?;
    Ok(leak_object_value(TupleObject::from_vec(values)))
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

fn collect_iterable_values_runtime(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<Vec<Value>, RuntimeError> {
    crate::ops::iteration::collect_iterable_values(vm, value)
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

fn namedtuple_instance_and_class(
    instance_value: Option<Value>,
) -> Result<(*mut ShapedObject, Arc<PyClassObject>), BuiltinError> {
    let instance = instance_value.ok_or_else(|| {
        BuiltinError::TypeError("namedtuple __init__() missing instance receiver".to_string())
    })?;
    let ptr = instance.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("namedtuple instances must be heap objects".into())
    })?;
    let type_id = crate::ops::objects::extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(
            "namedtuple instances must be user-defined heap objects".to_string(),
        ));
    }

    let class = global_class(ClassId(type_id.raw())).ok_or_else(|| {
        BuiltinError::TypeError("namedtuple instance type metadata is unavailable".to_string())
    })?;

    Ok((ptr as *mut ShapedObject, class))
}

fn namedtuple_class_from_type_value(
    class_value: Option<Value>,
) -> Result<Arc<PyClassObject>, BuiltinError> {
    let class_value = class_value.ok_or_else(|| {
        BuiltinError::TypeError("namedtuple __new__() missing class receiver".to_string())
    })?;
    let ptr = class_value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("namedtuple __new__() first argument must be a class".to_string())
    })?;
    if crate::ops::objects::extract_type_id(ptr) != TypeId::TYPE
        || crate::builtins::builtin_type_object_type_id(ptr).is_some()
    {
        return Err(BuiltinError::TypeError(
            "namedtuple __new__() first argument must be a heap class".to_string(),
        ));
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    global_class(class.class_id()).ok_or_else(|| {
        BuiltinError::TypeError("namedtuple class metadata is unavailable".to_string())
    })
}

fn namedtuple_schema_class_for_slot(
    runtime_class: &PyClassObject,
    attr_name: &str,
    sentinel: &'static BuiltinFunctionObject,
) -> Result<Arc<PyClassObject>, BuiltinError> {
    let attr = intern(attr_name);
    let sentinel_ptr = sentinel as *const BuiltinFunctionObject as *const ();

    for &class_id in runtime_class.mro() {
        if class_id.0 < TypeId::FIRST_USER_TYPE {
            continue;
        }

        let Some(candidate) = global_class(class_id) else {
            continue;
        };
        if candidate
            .get_attr(&attr)
            .and_then(|value| value.as_object_ptr())
            .is_some_and(|ptr| ptr == sentinel_ptr)
        {
            return Ok(candidate);
        }
    }

    Err(BuiltinError::TypeError(
        "namedtuple constructor metadata is unavailable".to_string(),
    ))
}

fn namedtuple_field_names(class: &PyClassObject) -> Result<Vec<InternedString>, BuiltinError> {
    let fields_value = class
        .get_attr(&intern("_fields"))
        .ok_or_else(|| BuiltinError::TypeError("namedtuple class is missing _fields".into()))?;
    let fields_ptr = fields_value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("namedtuple class _fields must be a tuple".to_string())
    })?;
    if crate::ops::objects::extract_type_id(fields_ptr) != TypeId::TUPLE {
        return Err(BuiltinError::TypeError(
            "namedtuple class _fields must be a tuple".to_string(),
        ));
    }

    let fields_tuple = unsafe { &*(fields_ptr as *const TupleObject) };
    let mut fields = Vec::with_capacity(fields_tuple.len());
    for index in 0..fields_tuple.len() {
        let field_value = fields_tuple.get(index as i64).ok_or_else(|| {
            BuiltinError::TypeError(
                "namedtuple class _fields contains an invalid entry".to_string(),
            )
        })?;
        fields.push(intern(&extract_string_value(field_value)?));
    }
    Ok(fields)
}

fn namedtuple_field_defaults(class: &PyClassObject) -> Option<*const DictObject> {
    let defaults = class.get_attr(&intern("_field_defaults"))?;
    let ptr = defaults.as_object_ptr()?;
    (crate::ops::objects::extract_type_id(ptr) == TypeId::DICT).then_some(ptr as *const DictObject)
}

fn bind_namedtuple_arguments(
    class_name: &str,
    fields: &[InternedString],
    defaults_ptr: Option<*const DictObject>,
    positional_args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Vec<Value>, BuiltinError> {
    if positional_args.len() > fields.len() {
        return Err(BuiltinError::TypeError(format!(
            "{class_name}() takes {expected} positional argument{s} but {given} {verb} given",
            expected = fields.len(),
            s = if fields.len() == 1 { "" } else { "s" },
            given = positional_args.len(),
            verb = if positional_args.len() == 1 {
                "was"
            } else {
                "were"
            },
        )));
    }

    let mut values = vec![Value::none(); fields.len()];
    let mut assigned = vec![false; fields.len()];

    for (index, value) in positional_args.iter().copied().enumerate() {
        values[index] = value;
        assigned[index] = true;
    }

    for &(name, value) in keywords {
        let Some(index) = fields.iter().position(|field| field.as_str() == name) else {
            return Err(BuiltinError::TypeError(format!(
                "{class_name}() got an unexpected keyword argument '{name}'",
            )));
        };
        if assigned[index] {
            return Err(BuiltinError::TypeError(format!(
                "{class_name}() got multiple values for argument '{name}'",
            )));
        }
        values[index] = value;
        assigned[index] = true;
    }

    let defaults = defaults_ptr.map(|ptr| unsafe { &*ptr });
    let mut missing = Vec::new();
    for (index, field_name) in fields.iter().enumerate() {
        if assigned[index] {
            continue;
        }

        let default = defaults.and_then(|dict| dict.get(Value::string(field_name.clone())));
        if let Some(default) = default {
            values[index] = default;
            assigned[index] = true;
            continue;
        }

        missing.push(field_name.as_str().to_string());
    }

    if !missing.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{class_name}() missing required field argument{s}: {names}",
            s = if missing.len() == 1 { "" } else { "s" },
            names = missing
                .into_iter()
                .map(|name| format!("'{name}'"))
                .collect::<Vec<_>>()
                .join(", "),
        )));
    }

    Ok(values)
}

fn namedtuple_class_doc(type_name: &str, fields: &[InternedString]) -> String {
    let args = fields
        .iter()
        .map(|name| name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{type_name}({args})")
}

fn new_namedtuple_field_descriptor(field_name: &InternedString, index: usize) -> Value {
    let class = &*NAMEDTUPLE_FIELD_CLASS;
    let mut descriptor =
        ShapedObject::new(class.class_type_id(), Arc::clone(class.instance_shape()));
    let registry = shape_registry();
    descriptor.set_property(intern("__doc__"), Value::none(), registry);
    descriptor.set_property(intern("name"), Value::string(field_name.clone()), registry);
    descriptor.set_property(
        intern("index"),
        Value::int(index as i64).expect("namedtuple field index should fit in i64"),
        registry,
    );
    leak_object_value(descriptor)
}

fn namedtuple_field_get(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "_tuplegetter.__get__() takes exactly 2 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    if args[1].is_none() {
        return Ok(args[0]);
    }

    let descriptor_ptr = args[0].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_tuplegetter.__get__() receiver must be a descriptor".to_string())
    })?;
    let descriptor = unsafe { &*(descriptor_ptr as *const ShapedObject) };
    let index = descriptor
        .get_property("index")
        .and_then(|value| value.as_int())
        .ok_or_else(|| {
            BuiltinError::TypeError(
                "_tuplegetter descriptor is missing its field index".to_string(),
            )
        })?;

    let instance_ptr = args[1].as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("_tuplegetter.__get__() instance must be tuple-backed".to_string())
    })?;
    let tuple = tuple_storage_ref_from_ptr(instance_ptr).ok_or_else(|| {
        BuiltinError::TypeError("_tuplegetter.__get__() instance must be tuple-backed".to_string())
    })?;

    tuple
        .get(index)
        .ok_or_else(|| BuiltinError::IndexError("tuple index out of range".to_string()))
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

fn expect_bound_collection_receiver(
    args: &[Value],
    descriptor_name: &str,
) -> Result<*const (), BuiltinError> {
    if args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "{descriptor_name}() missing required argument 'self'"
        )));
    }

    expect_collection_instance(&args[..1], descriptor_name)
}

fn expect_user_collection_from_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<&'static ShapedObject, BuiltinError> {
    if extract_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires a heap-allocated collections object"
        )));
    }

    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn expect_user_collection_from_ptr_mut(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    if extract_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires a heap-allocated collections object"
        )));
    }

    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn heap_class_for_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<Arc<PyClassObject>, BuiltinError> {
    let type_id = extract_type_id(ptr);
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return Err(BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires a heap-allocated collections object"
        )));
    }

    global_class(ClassId(type_id.raw())).ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{descriptor_name}' requires registered heap type metadata"
        ))
    })
}

fn heap_class_value_for_ptr(ptr: *const (), descriptor_name: &str) -> Result<Value, BuiltinError> {
    heap_class_for_ptr(ptr, descriptor_name).map(|class| class_value(&class))
}

fn heap_class_name_for_ptr(ptr: *const (), descriptor_name: &str) -> Result<String, BuiltinError> {
    heap_class_for_ptr(ptr, descriptor_name).map(|class| class.name().as_str().to_string())
}

fn chainmap_maps_value_from_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<Value, BuiltinError> {
    let instance = expect_user_collection_from_ptr(ptr, descriptor_name)?;
    instance.get_property("maps").ok_or_else(|| {
        BuiltinError::AttributeError(format!(
            "collections.ChainMap instance has no 'maps' attribute"
        ))
    })
}

fn chainmap_maps_list_from_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<&'static ListObject, BuiltinError> {
    let value = chainmap_maps_value_from_ptr(ptr, descriptor_name)?;
    let list_ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError("collections.ChainMap.maps must be a list".to_string())
    })?;
    list_storage_ref_from_ptr(list_ptr).ok_or_else(|| {
        BuiltinError::TypeError("collections.ChainMap.maps must be a list".to_string())
    })
}

fn chainmap_maps_values_from_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<Vec<Value>, BuiltinError> {
    Ok(chainmap_maps_list_from_ptr(ptr, descriptor_name)?
        .iter()
        .copied()
        .collect())
}

fn chainmap_primary_map_from_ptr(
    ptr: *const (),
    descriptor_name: &str,
) -> Result<Value, BuiltinError> {
    chainmap_maps_list_from_ptr(ptr, descriptor_name)?
        .get(0)
        .ok_or_else(|| BuiltinError::ValueError("collections.ChainMap.maps cannot be empty".into()))
}

fn mapping_iter_keys(vm: &mut VirtualMachine, mapping: Value) -> Result<Vec<Value>, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return Ok(dict.keys().collect());
    }

    collect_iterable_values_runtime(vm, mapping)
}

fn mapping_get_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<Value, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return dict
            .get(key)
            .ok_or_else(|| RuntimeError::key_error("key not found"));
    }

    let getitem = get_attribute_value(vm, mapping, &intern("__getitem__"))?;
    invoke_callable_value(vm, getitem, &[key])
}

fn mapping_contains_key(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<bool, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return Ok(dict.contains_key(key));
    }

    match get_attribute_value(vm, mapping, &intern("__contains__")) {
        Ok(contains) => {
            let result = invoke_callable_value(vm, contains, &[key])?;
            try_is_truthy(vm, result)
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            match mapping_get_item(vm, mapping, key) {
                Ok(_) => Ok(true),
                Err(err) if is_missing_mapping_key_error(&err) => Ok(false),
                Err(err) => Err(err),
            }
        }
        Err(err) => Err(err),
    }
}

fn mapping_set_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_mut_from_ptr(ptr)
    {
        dict.set(key, value);
        return Ok(());
    }

    let setitem = get_attribute_value(vm, mapping, &intern("__setitem__"))?;
    invoke_callable_value(vm, setitem, &[key, value]).map(|_| ())
}

fn mapping_delete_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_mut_from_ptr(ptr)
    {
        return dict
            .remove(key)
            .map(|_| ())
            .ok_or_else(|| RuntimeError::key_error("key not found"));
    }

    let delitem = get_attribute_value(vm, mapping, &intern("__delitem__"))?;
    invoke_callable_value(vm, delitem, &[key]).map(|_| ())
}

fn mapping_copy(vm: &mut VirtualMachine, mapping: Value) -> Result<Value, RuntimeError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        let mut copied = DictObject::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            copied.set(key, value);
        }
        return Ok(leak_object_value(copied));
    }

    let copy = get_attribute_value(vm, mapping, &intern("copy"))?;
    invoke_callable_value(vm, copy, &[])
}

fn is_missing_mapping_key_error(err: &RuntimeError) -> bool {
    matches!(err.kind, RuntimeErrorKind::KeyError { .. })
        || matches!(
            err.kind,
            RuntimeErrorKind::Exception { type_id, .. }
                if type_id == ExceptionTypeId::KeyError as u8 as u16
        )
        || matches!(
            &err.kind,
            RuntimeErrorKind::InternalError { message } if message.as_ref() == "key not found"
        )
}

fn first_mapping_missing_key_error(key: Value) -> Result<BuiltinError, BuiltinError> {
    Ok(BuiltinError::KeyError(format!(
        "Key not found in the first mapping: {}",
        collection_repr_text(key)?
    )))
}

fn shaped_property_repr(ptr: *const (), name: &str) -> Result<Option<String>, BuiltinError> {
    if extract_type_id(ptr).raw() < TypeId::FIRST_USER_TYPE {
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
    let dict = dict_storage_ref_from_ptr(ptr).ok_or_else(|| {
        BuiltinError::TypeError(
            "descriptor '__repr__' requires a dict-backed collections object".to_string(),
        )
    })?;
    let default_factory =
        shaped_property_repr(ptr, "default_factory")?.unwrap_or_else(|| "None".to_string());
    let result = format!("defaultdict({default_factory}, {})", dict_repr_body(dict)?);
    Ok(Value::string(intern(&result)))
}

fn chainmap_init(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_bound_collection_receiver(args, "__init__")?;
    let instance = expect_user_collection_from_ptr_mut(ptr, "__init__")?;
    let maps = if args.len() == 1 {
        vec![leak_object_value(DictObject::new())]
    } else {
        args[1..].to_vec()
    };
    let maps_value = leak_object_value(ListObject::from_iter(maps));
    instance.set_property(intern("maps"), maps_value, shape_registry());
    Ok(Value::none())
}

fn chainmap_missing(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__missing__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    Err(BuiltinError::KeyError(collection_repr_text(args[1])?))
}

fn chainmap_getitem(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__getitem__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__getitem__")?;
    for mapping in chainmap_maps_values_from_ptr(ptr, "__getitem__")? {
        match mapping_get_item(vm, mapping, args[1]) {
            Ok(value) => return Ok(value),
            Err(err) if is_missing_mapping_key_error(&err) => {}
            Err(err) => return Err(BuiltinError::Raised(err)),
        }
    }

    let missing =
        get_attribute_value(vm, args[0], &intern("__missing__")).map_err(BuiltinError::Raised)?;
    invoke_callable_value(vm, missing, &[args[1]]).map_err(BuiltinError::Raised)
}

fn chainmap_get(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "get() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "get")?;
    let default = args.get(2).copied().unwrap_or_else(Value::none);
    for mapping in chainmap_maps_values_from_ptr(ptr, "get")? {
        match mapping_contains_key(vm, mapping, args[1]) {
            Ok(true) => return chainmap_getitem(vm, &args[..2]),
            Ok(false) => {}
            Err(err) => return Err(BuiltinError::Raised(err)),
        }
    }
    Ok(default)
}

fn chainmap_len(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__len__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let keys = collect_iterable_values_runtime(vm, args[0]).map_err(BuiltinError::Raised)?;
    Value::int(keys.len() as i64)
        .ok_or_else(|| BuiltinError::OverflowError("collections.ChainMap is too large".into()))
}

fn chainmap_iter(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__iter__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__iter__")?;
    let mut keys = DictObject::new();
    for mapping in chainmap_maps_values_from_ptr(ptr, "__iter__")?
        .into_iter()
        .rev()
    {
        for key in mapping_iter_keys(vm, mapping).map_err(BuiltinError::Raised)? {
            if !keys.contains_key(key) {
                keys.set(key, Value::none());
            }
        }
    }

    let key_list = leak_object_value(ListObject::from_iter(keys.keys()));
    crate::builtins::builtin_iter_vm(vm, &[key_list])
}

fn chainmap_contains(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__contains__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__contains__")?;
    for mapping in chainmap_maps_values_from_ptr(ptr, "__contains__")? {
        match mapping_contains_key(vm, mapping, args[1]) {
            Ok(true) => return Ok(Value::bool(true)),
            Ok(false) => {}
            Err(err) => return Err(BuiltinError::Raised(err)),
        }
    }
    Ok(Value::bool(false))
}

fn chainmap_bool(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "__bool__() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__bool__")?;
    for mapping in chainmap_maps_values_from_ptr(ptr, "__bool__")? {
        if try_is_truthy(vm, mapping).map_err(BuiltinError::Raised)? {
            return Ok(Value::bool(true));
        }
    }
    Ok(Value::bool(false))
}

fn chainmap_keys(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "keys() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let keys = collect_iterable_values_runtime(vm, args[0]).map_err(BuiltinError::Raised)?;
    Ok(leak_object_value(ListObject::from_iter(keys)))
}

fn chainmap_items(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "items() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let keys = collect_iterable_values_runtime(vm, args[0]).map_err(BuiltinError::Raised)?;
    let getitem =
        get_attribute_value(vm, args[0], &intern("__getitem__")).map_err(BuiltinError::Raised)?;
    let mut items = Vec::with_capacity(keys.len());
    for key in keys {
        let value = invoke_callable_value(vm, getitem, &[key]).map_err(BuiltinError::Raised)?;
        items.push(leak_object_value(TupleObject::from_vec(vec![key, value])));
    }
    Ok(leak_object_value(ListObject::from_iter(items)))
}

fn chainmap_values(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "values() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let keys = collect_iterable_values_runtime(vm, args[0]).map_err(BuiltinError::Raised)?;
    let getitem =
        get_attribute_value(vm, args[0], &intern("__getitem__")).map_err(BuiltinError::Raised)?;
    let mut values = Vec::with_capacity(keys.len());
    for key in keys {
        values.push(invoke_callable_value(vm, getitem, &[key]).map_err(BuiltinError::Raised)?);
    }
    Ok(leak_object_value(ListObject::from_iter(values)))
}

fn chainmap_setitem(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 3 {
        return Err(BuiltinError::TypeError(format!(
            "__setitem__() takes exactly 3 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__setitem__")?;
    let first = chainmap_primary_map_from_ptr(ptr, "__setitem__")?;
    mapping_set_item(vm, first, args[1], args[2]).map_err(BuiltinError::Raised)?;
    Ok(Value::none())
}

fn chainmap_delitem(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "__delitem__() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "__delitem__")?;
    let first = chainmap_primary_map_from_ptr(ptr, "__delitem__")?;
    match mapping_delete_item(vm, first, args[1]) {
        Ok(()) => Ok(Value::none()),
        Err(err) if is_missing_mapping_key_error(&err) => Err(BuiltinError::KeyError(format!(
            "Key not found in the first mapping: {}",
            collection_repr_text(args[1])?
        ))),
        Err(err) => Err(BuiltinError::Raised(err)),
    }
}

fn chainmap_repr(args: &[Value]) -> Result<Value, BuiltinError> {
    let ptr = expect_collection_instance(args, "__repr__")?;
    let class_name =
        heap_class_name_for_ptr(ptr, "__repr__").unwrap_or_else(|_| "ChainMap".to_string());
    let maps = chainmap_maps_values_from_ptr(ptr, "__repr__")?;
    let mut out = String::new();
    for (index, mapping) in maps.into_iter().enumerate() {
        if index > 0 {
            out.push_str(", ");
        }
        out.push_str(&collection_repr_text(mapping)?);
    }
    Ok(Value::string(intern(&format!("{class_name}({out})"))))
}

fn chainmap_copy(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "copy() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "copy")?;
    let maps = chainmap_maps_values_from_ptr(ptr, "copy")?;
    let class_value = heap_class_value_for_ptr(ptr, "copy")?;
    let mut ctor_args = Vec::with_capacity(maps.len().max(1));
    if let Some((first, tail)) = maps.split_first() {
        ctor_args.push(mapping_copy(vm, *first).map_err(BuiltinError::Raised)?);
        ctor_args.extend_from_slice(tail);
    }
    invoke_callable_value(vm, class_value, &ctor_args).map_err(BuiltinError::Raised)
}

fn chainmap_new_child(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "new_child() takes at most 2 positional arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "new_child")?;
    let class_value = heap_class_value_for_ptr(ptr, "new_child")?;
    let current_maps = chainmap_maps_values_from_ptr(ptr, "new_child")?;

    let new_map = match args.get(1).copied() {
        Some(mapping) => {
            for &(name, value) in keywords {
                mapping_set_item(vm, mapping, Value::string(intern(name)), value)
                    .map_err(BuiltinError::Raised)?;
            }
            mapping
        }
        None => {
            let mut dict = DictObject::with_capacity(keywords.len());
            for &(name, value) in keywords {
                dict.set(Value::string(intern(name)), value);
            }
            leak_object_value(dict)
        }
    };

    let mut ctor_args = Vec::with_capacity(current_maps.len() + 1);
    ctor_args.push(new_map);
    ctor_args.extend(current_maps);
    invoke_callable_value(vm, class_value, &ctor_args).map_err(BuiltinError::Raised)
}

fn chainmap_fromkeys(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "fromkeys() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let default = args.get(2).copied().unwrap_or_else(Value::none);
    let keys = collect_iterable_values_runtime(vm, args[1]).map_err(BuiltinError::Raised)?;
    let mut dict = DictObject::with_capacity(keys.len());
    for key in keys {
        dict.set(key, default);
    }
    invoke_callable_value(vm, args[0], &[leak_object_value(dict)]).map_err(BuiltinError::Raised)
}

fn chainmap_parents_getter(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "parents() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "parents")?;
    let class_value = heap_class_value_for_ptr(ptr, "parents")?;
    let maps = chainmap_maps_values_from_ptr(ptr, "parents")?;
    let parent_maps = maps.get(1..).unwrap_or(&[]);
    invoke_callable_value(vm, class_value, parent_maps).map_err(BuiltinError::Raised)
}

fn chainmap_pop(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "pop() takes 2 or 3 arguments ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "pop")?;
    let first = chainmap_primary_map_from_ptr(ptr, "pop")?;
    if let Some(first_ptr) = first.as_object_ptr()
        && extract_type_id(first_ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_mut_from_ptr(first_ptr)
    {
        if let Some(value) = dict.remove(args[1]) {
            return Ok(value);
        }
        return args
            .get(2)
            .copied()
            .ok_or_else(|| first_mapping_missing_key_error(args[1]).unwrap_or_else(|err| err));
    }

    match get_attribute_value(vm, first, &intern("pop")) {
        Ok(pop) => match invoke_callable_value(vm, pop, &args[1..]) {
            Ok(value) => Ok(value),
            Err(err) if is_missing_mapping_key_error(&err) => args
                .get(2)
                .copied()
                .ok_or_else(|| first_mapping_missing_key_error(args[1]).unwrap_or_else(|err| err)),
            Err(err) => Err(BuiltinError::Raised(err)),
        },
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            match mapping_get_item(vm, first, args[1]) {
                Ok(value) => {
                    mapping_delete_item(vm, first, args[1]).map_err(BuiltinError::Raised)?;
                    Ok(value)
                }
                Err(err) if is_missing_mapping_key_error(&err) => {
                    args.get(2).copied().ok_or_else(|| {
                        first_mapping_missing_key_error(args[1]).unwrap_or_else(|err| err)
                    })
                }
                Err(err) => Err(BuiltinError::Raised(err)),
            }
        }
        Err(err) => Err(BuiltinError::Raised(err)),
    }
}

fn chainmap_popitem(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "popitem() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "popitem")?;
    let first = chainmap_primary_map_from_ptr(ptr, "popitem")?;
    if let Some(first_ptr) = first.as_object_ptr()
        && extract_type_id(first_ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_mut_from_ptr(first_ptr)
    {
        return dict
            .popitem()
            .map(|(key, value)| leak_object_value(TupleObject::from_vec(vec![key, value])))
            .ok_or_else(|| BuiltinError::KeyError("No keys found in the first mapping.".into()));
    }

    let popitem =
        get_attribute_value(vm, first, &intern("popitem")).map_err(BuiltinError::Raised)?;
    match invoke_callable_value(vm, popitem, &[]) {
        Ok(value) => Ok(value),
        Err(err) if is_missing_mapping_key_error(&err) => Err(BuiltinError::KeyError(
            "No keys found in the first mapping.".into(),
        )),
        Err(err) => Err(BuiltinError::Raised(err)),
    }
}

fn chainmap_clear(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "clear() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let ptr = expect_bound_collection_receiver(args, "clear")?;
    let first = chainmap_primary_map_from_ptr(ptr, "clear")?;
    if let Some(first_ptr) = first.as_object_ptr()
        && extract_type_id(first_ptr) == TypeId::DICT
        && let Some(dict) = dict_storage_mut_from_ptr(first_ptr)
    {
        dict.clear();
        return Ok(Value::none());
    }

    let clear = get_attribute_value(vm, first, &intern("clear")).map_err(BuiltinError::Raised)?;
    invoke_callable_value(vm, clear, &[]).map_err(BuiltinError::Raised)?;
    Ok(Value::none())
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
    use crate::builtins::builtin_type;
    use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
    use crate::ops::objects::{extract_type_id, get_attribute_value, list_storage_ref_from_ptr};

    fn dict_from_value(value: Value) -> *mut DictObject {
        value
            .as_object_ptr()
            .expect("dict-backed value should be object") as *mut DictObject
    }

    fn list_from_value(value: Value) -> &'static ListObject {
        let ptr = value
            .as_object_ptr()
            .expect("list-backed value should be object");
        list_storage_ref_from_ptr(ptr).expect("value should be list-backed")
    }

    fn tuple_from_value(value: Value) -> &'static TupleObject {
        let ptr = value
            .as_object_ptr()
            .expect("tuple-backed value should be object");
        unsafe { &*(ptr as *const TupleObject) }
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
        assert!(class.mro().contains(&ClassId(TypeId::TUPLE.raw())));

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

    #[test]
    fn test_namedtuple_instances_bind_positional_fields_and_publish_class_placeholders() {
        let mut vm = VirtualMachine::new();
        let class_value = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("Pair")),
                Value::string(intern("left right")),
            ],
        )
        .expect("namedtuple should construct class");

        let instance = invoke_callable_value(
            &mut vm,
            class_value,
            &[Value::int(10).unwrap(), Value::int(20).unwrap()],
        )
        .expect("namedtuple class should accept positional construction");

        let instance_ptr = instance
            .as_object_ptr()
            .expect("namedtuple instance should be heap allocated");
        let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
        let tuple = shaped
            .tuple_backing()
            .expect("namedtuple instances should use native tuple storage");
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.as_slice()[0].as_int(), Some(10));
        assert_eq!(tuple.as_slice()[1].as_int(), Some(20));
        assert_eq!(
            get_attribute_value(&mut vm, instance, &intern("left"))
                .expect("left descriptor should read tuple storage")
                .as_int(),
            Some(10)
        );
        assert_eq!(
            get_attribute_value(&mut vm, instance, &intern("right"))
                .expect("right descriptor should read tuple storage")
                .as_int(),
            Some(20)
        );

        let class_ptr = class_value
            .as_object_ptr()
            .expect("namedtuple class should be object-backed");
        let class = unsafe { &*(class_ptr as *const PyClassObject) };
        assert!(class.get_attr(&intern("left")).is_some());
        assert!(class.get_attr(&intern("right")).is_some());

        unsafe {
            drop(Box::from_raw(instance_ptr as *mut ShapedObject));
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
    }

    #[test]
    fn test_namedtuple_instances_apply_defaults_and_keywords() {
        let mut vm = VirtualMachine::new();
        let defaults = leak_object_value(TupleObject::from_slice(&[Value::int(7).unwrap()]));
        let class_value = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("Point")),
                Value::string(intern("x y")),
                Value::bool(false),
                defaults,
                Value::string(intern("demo.point")),
            ],
        )
        .expect("namedtuple should construct class");

        let instance = invoke_callable_value_with_keywords(
            &mut vm,
            class_value,
            &[Value::int(3).unwrap()],
            &[("y", Value::int(11).unwrap())],
        )
        .expect("namedtuple class should accept keyword overrides");

        let instance_ptr = instance
            .as_object_ptr()
            .expect("namedtuple instance should be heap allocated");
        let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
        let tuple = shaped
            .tuple_backing()
            .expect("namedtuple instances should use native tuple storage");
        assert_eq!(tuple.as_slice()[0].as_int(), Some(3));
        assert_eq!(tuple.as_slice()[1].as_int(), Some(11));

        let defaulted = invoke_callable_value(&mut vm, class_value, &[Value::int(5).unwrap()])
            .expect("namedtuple class should fill omitted defaults");
        let defaulted_ptr = defaulted
            .as_object_ptr()
            .expect("defaulted namedtuple instance should be heap allocated");
        let defaulted_shaped = unsafe { &*(defaulted_ptr as *const ShapedObject) };
        let defaulted_tuple = defaulted_shaped
            .tuple_backing()
            .expect("defaulted namedtuple should use native tuple storage");
        assert_eq!(defaulted_tuple.as_slice()[0].as_int(), Some(5));
        assert_eq!(defaulted_tuple.as_slice()[1].as_int(), Some(7));
        assert_eq!(
            get_attribute_value(&mut vm, defaulted, &intern("x"))
                .expect("x descriptor should read tuple storage")
                .as_int(),
            Some(5)
        );
        assert_eq!(
            get_attribute_value(&mut vm, defaulted, &intern("y"))
                .expect("y descriptor should read tuple storage")
                .as_int(),
            Some(7)
        );

        unsafe {
            drop(Box::from_raw(
                defaults.as_object_ptr().unwrap() as *mut TupleObject
            ));
            drop(Box::from_raw(instance_ptr as *mut ShapedObject));
            drop(Box::from_raw(defaulted_ptr as *mut ShapedObject));
            drop(Arc::from_raw(
                class_value.as_object_ptr().unwrap() as *const PyClassObject
            ));
        }
    }

    #[test]
    fn test_namedtuple_subclasses_construct_from_base_field_schema() {
        let mut vm = VirtualMachine::new();
        let base_class = builtin_namedtuple(
            &mut vm,
            &[
                Value::string(intern("BasePair")),
                Value::string(intern("left right")),
            ],
        )
        .expect("namedtuple should construct base class");
        let bases = leak_object_value(TupleObject::from_slice(&[base_class]));
        let widened_fields = leak_object_value(TupleObject::from_slice(&[
            Value::string(intern("left")),
            Value::string(intern("right")),
            Value::string(intern("extra")),
        ]));
        let mut namespace = DictObject::new();
        namespace.set(Value::string(intern("_fields")), widened_fields);
        let namespace_value = leak_object_value(namespace);

        let subclass =
            builtin_type(&[Value::string(intern("WidenedPair")), bases, namespace_value])
                .expect("namedtuple subclass should be constructible");
        let instance = invoke_callable_value(
            &mut vm,
            subclass,
            &[Value::int(10).unwrap(), Value::int(20).unwrap()],
        )
        .expect("subclass should use the base namedtuple constructor schema");

        let instance_ptr = instance
            .as_object_ptr()
            .expect("subclass instance should be heap allocated");
        let shaped = unsafe { &*(instance_ptr as *const ShapedObject) };
        let tuple = shaped
            .tuple_backing()
            .expect("namedtuple subclass should retain native tuple storage");
        assert_eq!(tuple.len(), 2);
        assert_eq!(tuple.as_slice()[0].as_int(), Some(10));
        assert_eq!(tuple.as_slice()[1].as_int(), Some(20));
        assert_eq!(
            get_attribute_value(&mut vm, instance, &intern("left"))
                .expect("inherited descriptor should read tuple storage")
                .as_int(),
            Some(10)
        );

        unsafe {
            drop(Box::from_raw(instance_ptr as *mut ShapedObject));
            drop(Box::from_raw(
                widened_fields.as_object_ptr().unwrap() as *mut TupleObject
            ));
            drop(Box::from_raw(
                namespace_value.as_object_ptr().unwrap() as *mut DictObject
            ));
            drop(Box::from_raw(
                bases.as_object_ptr().unwrap() as *mut TupleObject
            ));
            drop(Arc::from_raw(
                subclass.as_object_ptr().unwrap() as *const PyClassObject
            ));
            drop(Arc::from_raw(
                base_class.as_object_ptr().unwrap() as *const PyClassObject
            ));
        }
    }

    #[test]
    fn test_chainmap_supports_items_new_child_parents_and_fromkeys() {
        let mut vm = VirtualMachine::new();
        let class_value = class_value(&CHAINMAP_CLASS);
        let class_ptr = class_value
            .as_object_ptr()
            .expect("ChainMap class should be object-backed");

        let mut base = DictObject::new();
        base.set(
            Value::string(intern("scope")),
            Value::string(intern("outer")),
        );
        let mut front = DictObject::new();
        front.set(
            Value::string(intern("scope")),
            Value::string(intern("inner")),
        );

        let chainmap = invoke_callable_value(
            &mut vm,
            class_value,
            &[leak_object_value(front), leak_object_value(base)],
        )
        .expect("ChainMap should construct");

        let getitem = get_attribute_value(&mut vm, chainmap, &intern("__getitem__"))
            .expect("__getitem__ should resolve");
        let scope = invoke_callable_value(&mut vm, getitem, &[Value::string(intern("scope"))])
            .expect("lookup should succeed");
        assert_eq!(extract_string_value(scope).unwrap(), "inner");

        let items =
            get_attribute_value(&mut vm, chainmap, &intern("items")).expect("items should exist");
        let item_values =
            invoke_callable_value(&mut vm, items, &[]).expect("items() should succeed");
        let items_list = list_from_value(item_values);
        assert_eq!(items_list.len(), 1);
        let item = tuple_from_value(
            items_list
                .get(0)
                .expect("items list should contain a tuple"),
        );
        assert_eq!(
            extract_string_value(item.get(0).expect("tuple key")).unwrap(),
            "scope"
        );
        assert_eq!(
            extract_string_value(item.get(1).expect("tuple value")).unwrap(),
            "inner"
        );

        let new_child = get_attribute_value(&mut vm, chainmap, &intern("new_child"))
            .expect("new_child should exist");
        let child = invoke_callable_value_with_keywords(
            &mut vm,
            new_child,
            &[],
            &[("phase", Value::string(intern("leaf")))],
        )
        .expect("new_child() should succeed");

        let child_items = get_attribute_value(&mut vm, child, &intern("items"))
            .expect("child items should exist");
        let child_item_values =
            invoke_callable_value(&mut vm, child_items, &[]).expect("child items() should work");
        let child_items_list = list_from_value(child_item_values);
        assert_eq!(child_items_list.len(), 2);

        let first_child_item = tuple_from_value(
            child_items_list
                .get(0)
                .expect("child items should contain a first tuple"),
        );
        let second_child_item = tuple_from_value(
            child_items_list
                .get(1)
                .expect("child items should contain a second tuple"),
        );
        assert_eq!(
            extract_string_value(first_child_item.get(0).unwrap()).unwrap(),
            "scope"
        );
        assert_eq!(
            extract_string_value(second_child_item.get(0).unwrap()).unwrap(),
            "phase"
        );

        let parents =
            get_attribute_value(&mut vm, child, &intern("parents")).expect("parents should exist");
        let parent_items = get_attribute_value(&mut vm, parents, &intern("items"))
            .expect("parent items should exist");
        let parent_item_values =
            invoke_callable_value(&mut vm, parent_items, &[]).expect("parent items() should work");
        let parent_items_list = list_from_value(parent_item_values);
        assert_eq!(parent_items_list.len(), 1);
        let parent_item = tuple_from_value(
            parent_items_list
                .get(0)
                .expect("parents items should contain a tuple"),
        );
        assert_eq!(
            extract_string_value(parent_item.get(0).unwrap()).unwrap(),
            "scope"
        );
        assert_eq!(
            extract_string_value(parent_item.get(1).unwrap()).unwrap(),
            "inner"
        );

        let fromkeys = get_attribute_value(&mut vm, class_value, &intern("fromkeys"))
            .expect("fromkeys should resolve");
        let fromkeys_result = invoke_callable_value(
            &mut vm,
            fromkeys,
            &[
                leak_object_value(ListObject::from_iter([
                    Value::string(intern("a")),
                    Value::string(intern("b")),
                ])),
                Value::int(5).unwrap(),
            ],
        )
        .expect("fromkeys() should succeed");
        let fromkeys_items = get_attribute_value(&mut vm, fromkeys_result, &intern("items"))
            .expect("fromkeys items should exist");
        let fromkeys_item_values =
            invoke_callable_value(&mut vm, fromkeys_items, &[]).expect("items() should work");
        let fromkeys_items_list = list_from_value(fromkeys_item_values);
        assert_eq!(fromkeys_items_list.len(), 2);
        let first_pair = tuple_from_value(
            fromkeys_items_list
                .get(0)
                .expect("fromkeys items should contain first tuple"),
        );
        let second_pair = tuple_from_value(
            fromkeys_items_list
                .get(1)
                .expect("fromkeys items should contain second tuple"),
        );
        assert_eq!(
            extract_string_value(first_pair.get(0).unwrap()).unwrap(),
            "a"
        );
        assert_eq!(first_pair.get(1).unwrap().as_int(), Some(5));
        assert_eq!(
            extract_string_value(second_pair.get(0).unwrap()).unwrap(),
            "b"
        );
        assert_eq!(second_pair.get(1).unwrap().as_int(), Some(5));

        unsafe {
            drop(Arc::from_raw(class_ptr as *const PyClassObject));
        }
    }
}
