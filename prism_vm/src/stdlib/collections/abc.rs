//! Native `collections.abc` module.
//!
//! The public ABC class objects are represented as real heap types so normal
//! MRO and subclass bitmap checks remain fast. Structural checks for builtin
//! containers are handled by the `isinstance`/`issubclass` fast path in
//! `builtins::types`, avoiding Python-level ABC dispatch in hot code.

use crate::stdlib::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, builtin_class_mro, class_id_to_type_id, global_class, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const ABC_NAMES: &[&str] = &[
    "Awaitable",
    "Coroutine",
    "AsyncIterable",
    "AsyncIterator",
    "AsyncGenerator",
    "Hashable",
    "Iterable",
    "Iterator",
    "Generator",
    "Reversible",
    "Sized",
    "Container",
    "Callable",
    "Collection",
    "Set",
    "MutableSet",
    "Mapping",
    "MutableMapping",
    "MappingView",
    "KeysView",
    "ItemsView",
    "ValuesView",
    "Sequence",
    "MutableSequence",
    "ByteString",
    "Buffer",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CollectionsAbcKind {
    Awaitable,
    Coroutine,
    AsyncIterable,
    AsyncIterator,
    AsyncGenerator,
    Hashable,
    Iterable,
    Iterator,
    Generator,
    Reversible,
    Sized,
    Container,
    Callable,
    Collection,
    Set,
    MutableSet,
    Mapping,
    MutableMapping,
    MappingView,
    KeysView,
    ItemsView,
    ValuesView,
    Sequence,
    MutableSequence,
    ByteString,
    Buffer,
}

pub struct CollectionsAbcModule {
    attrs: Vec<Arc<str>>,
}

impl CollectionsAbcModule {
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(ABC_NAMES.len() + 1);
        attrs.push(Arc::from("__all__"));
        attrs.extend(ABC_NAMES.iter().map(|name| Arc::from(*name)));
        Self { attrs }
    }
}

impl Default for CollectionsAbcModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CollectionsAbcModule {
    fn name(&self) -> &str {
        "collections.abc"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(all_value()),
            "Awaitable" => Ok(class_value(&AWAITABLE_CLASS)),
            "Coroutine" => Ok(class_value(&COROUTINE_CLASS)),
            "AsyncIterable" => Ok(class_value(&ASYNC_ITERABLE_CLASS)),
            "AsyncIterator" => Ok(class_value(&ASYNC_ITERATOR_CLASS)),
            "AsyncGenerator" => Ok(class_value(&ASYNC_GENERATOR_CLASS)),
            "Hashable" => Ok(class_value(&HASHABLE_CLASS)),
            "Iterable" => Ok(class_value(&ITERABLE_CLASS)),
            "Iterator" => Ok(class_value(&ITERATOR_CLASS)),
            "Generator" => Ok(class_value(&GENERATOR_CLASS)),
            "Reversible" => Ok(class_value(&REVERSIBLE_CLASS)),
            "Sized" => Ok(class_value(&SIZED_CLASS)),
            "Container" => Ok(class_value(&CONTAINER_CLASS)),
            "Callable" => Ok(class_value(&CALLABLE_CLASS)),
            "Collection" => Ok(class_value(&COLLECTION_CLASS)),
            "Set" => Ok(class_value(&SET_CLASS)),
            "MutableSet" => Ok(class_value(&MUTABLE_SET_CLASS)),
            "Mapping" => Ok(class_value(&MAPPING_CLASS)),
            "MutableMapping" => Ok(class_value(&MUTABLE_MAPPING_CLASS)),
            "MappingView" => Ok(class_value(&MAPPING_VIEW_CLASS)),
            "KeysView" => Ok(class_value(&KEYS_VIEW_CLASS)),
            "ItemsView" => Ok(class_value(&ITEMS_VIEW_CLASS)),
            "ValuesView" => Ok(class_value(&VALUES_VIEW_CLASS)),
            "Sequence" => Ok(class_value(&SEQUENCE_CLASS)),
            "MutableSequence" => Ok(class_value(&MUTABLE_SEQUENCE_CLASS)),
            "ByteString" => Ok(class_value(&BYTE_STRING_CLASS)),
            "Buffer" => Ok(class_value(&BUFFER_CLASS)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'collections.abc' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

static AWAITABLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Awaitable", &[]));
static COROUTINE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Coroutine", &[AWAITABLE_CLASS.class_id()]));
static ASYNC_ITERABLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("AsyncIterable", &[]));
static ASYNC_ITERATOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("AsyncIterator", &[ASYNC_ITERABLE_CLASS.class_id()]));
static ASYNC_GENERATOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("AsyncGenerator", &[ASYNC_ITERATOR_CLASS.class_id()]));
static HASHABLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Hashable", &[]));
static ITERABLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Iterable", &[]));
static ITERATOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Iterator", &[ITERABLE_CLASS.class_id()]));
static GENERATOR_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Generator", &[ITERATOR_CLASS.class_id()]));
static REVERSIBLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Reversible", &[ITERABLE_CLASS.class_id()]));
static SIZED_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| build_abc_class("Sized", &[]));
static CONTAINER_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Container", &[]));
static CALLABLE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Callable", &[]));
static COLLECTION_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_abc_class(
        "Collection",
        &[
            SIZED_CLASS.class_id(),
            ITERABLE_CLASS.class_id(),
            CONTAINER_CLASS.class_id(),
        ],
    )
});
static SET_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Set", &[COLLECTION_CLASS.class_id()]));
static MUTABLE_SET_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("MutableSet", &[SET_CLASS.class_id()]));
static MAPPING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Mapping", &[COLLECTION_CLASS.class_id()]));
static MUTABLE_MAPPING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("MutableMapping", &[MAPPING_CLASS.class_id()]));
static MAPPING_VIEW_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("MappingView", &[SIZED_CLASS.class_id()]));
static KEYS_VIEW_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_abc_class(
        "KeysView",
        &[MAPPING_VIEW_CLASS.class_id(), SET_CLASS.class_id()],
    )
});
static ITEMS_VIEW_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_abc_class(
        "ItemsView",
        &[MAPPING_VIEW_CLASS.class_id(), SET_CLASS.class_id()],
    )
});
static VALUES_VIEW_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_abc_class(
        "ValuesView",
        &[MAPPING_VIEW_CLASS.class_id(), COLLECTION_CLASS.class_id()],
    )
});
static SEQUENCE_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_abc_class(
        "Sequence",
        &[REVERSIBLE_CLASS.class_id(), COLLECTION_CLASS.class_id()],
    )
});
static MUTABLE_SEQUENCE_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("MutableSequence", &[SEQUENCE_CLASS.class_id()]));
static BYTE_STRING_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("ByteString", &[SEQUENCE_CLASS.class_id()]));
static BUFFER_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_abc_class("Buffer", &[]));

pub(crate) fn abc_kind_for_class_value(value: Value) -> Option<CollectionsAbcKind> {
    let ptr = value.as_object_ptr()?;
    if ptr == class_ptr(&AWAITABLE_CLASS) {
        Some(CollectionsAbcKind::Awaitable)
    } else if ptr == class_ptr(&COROUTINE_CLASS) {
        Some(CollectionsAbcKind::Coroutine)
    } else if ptr == class_ptr(&ASYNC_ITERABLE_CLASS) {
        Some(CollectionsAbcKind::AsyncIterable)
    } else if ptr == class_ptr(&ASYNC_ITERATOR_CLASS) {
        Some(CollectionsAbcKind::AsyncIterator)
    } else if ptr == class_ptr(&ASYNC_GENERATOR_CLASS) {
        Some(CollectionsAbcKind::AsyncGenerator)
    } else if ptr == class_ptr(&HASHABLE_CLASS) {
        Some(CollectionsAbcKind::Hashable)
    } else if ptr == class_ptr(&ITERABLE_CLASS) {
        Some(CollectionsAbcKind::Iterable)
    } else if ptr == class_ptr(&ITERATOR_CLASS) {
        Some(CollectionsAbcKind::Iterator)
    } else if ptr == class_ptr(&GENERATOR_CLASS) {
        Some(CollectionsAbcKind::Generator)
    } else if ptr == class_ptr(&REVERSIBLE_CLASS) {
        Some(CollectionsAbcKind::Reversible)
    } else if ptr == class_ptr(&SIZED_CLASS) {
        Some(CollectionsAbcKind::Sized)
    } else if ptr == class_ptr(&CONTAINER_CLASS) {
        Some(CollectionsAbcKind::Container)
    } else if ptr == class_ptr(&CALLABLE_CLASS) {
        Some(CollectionsAbcKind::Callable)
    } else if ptr == class_ptr(&COLLECTION_CLASS) {
        Some(CollectionsAbcKind::Collection)
    } else if ptr == class_ptr(&SET_CLASS) {
        Some(CollectionsAbcKind::Set)
    } else if ptr == class_ptr(&MUTABLE_SET_CLASS) {
        Some(CollectionsAbcKind::MutableSet)
    } else if ptr == class_ptr(&MAPPING_CLASS) {
        Some(CollectionsAbcKind::Mapping)
    } else if ptr == class_ptr(&MUTABLE_MAPPING_CLASS) {
        Some(CollectionsAbcKind::MutableMapping)
    } else if ptr == class_ptr(&MAPPING_VIEW_CLASS) {
        Some(CollectionsAbcKind::MappingView)
    } else if ptr == class_ptr(&KEYS_VIEW_CLASS) {
        Some(CollectionsAbcKind::KeysView)
    } else if ptr == class_ptr(&ITEMS_VIEW_CLASS) {
        Some(CollectionsAbcKind::ItemsView)
    } else if ptr == class_ptr(&VALUES_VIEW_CLASS) {
        Some(CollectionsAbcKind::ValuesView)
    } else if ptr == class_ptr(&SEQUENCE_CLASS) {
        Some(CollectionsAbcKind::Sequence)
    } else if ptr == class_ptr(&MUTABLE_SEQUENCE_CLASS) {
        Some(CollectionsAbcKind::MutableSequence)
    } else if ptr == class_ptr(&BYTE_STRING_CLASS) {
        Some(CollectionsAbcKind::ByteString)
    } else if ptr == class_ptr(&BUFFER_CLASS) {
        Some(CollectionsAbcKind::Buffer)
    } else {
        None
    }
}

#[inline]
fn class_ptr(class: &Arc<PyClassObject>) -> *const () {
    Arc::as_ptr(class) as *const ()
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(class_ptr(class))
}

fn build_abc_class(name: &str, bases: &[ClassId]) -> Arc<PyClassObject> {
    let mut class = if bases.is_empty() {
        PyClassObject::new_simple(intern(name))
    } else {
        PyClassObject::new(intern(name), bases, |class_id| {
            if class_id == ClassId::OBJECT || class_id.0 < TypeId::FIRST_USER_TYPE {
                return Some(
                    builtin_class_mro(class_id_to_type_id(class_id))
                        .into_iter()
                        .collect(),
                );
            }
            global_class(class_id).map(|class| class.mro().iter().copied().collect())
        })
        .unwrap_or_else(|err| panic!("failed to create collections.abc.{name}: {err}"))
    };

    class.set_attr(
        intern("__module__"),
        Value::string(intern("collections.abc")),
    );
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    class.set_attr(intern("__slots__"), empty_tuple_value());
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }

    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn all_value() -> Value {
    let names: Vec<Value> = ABC_NAMES
        .iter()
        .map(|name| Value::string(intern(name)))
        .collect();
    leak_object_value(ListObject::from_iter(names))
}

fn empty_tuple_value() -> Value {
    leak_object_value(TupleObject::from_vec(Vec::new()))
}

#[inline]
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::leak(Box::new(object)) as *mut T as *const ();
    Value::object_ptr(ptr)
}
