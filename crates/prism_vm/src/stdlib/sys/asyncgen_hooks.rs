//! Async generator hook support for `sys`.
//!
//! Event loops use these hooks to track async generators and schedule cleanup
//! when an async generator becomes unreachable before it is explicitly closed.

use super::builtin_value;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::value_supports_call_protocol;
use crate::python_numeric::int_like_value;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::slice::SliceObject;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::sync::{Arc, LazyLock, Mutex};

const FIRSTITER_FIELD: &str = "firstiter";
const FINALIZER_FIELD: &str = "finalizer";

static GET_ASYNCGEN_HOOKS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.get_asyncgen_hooks"), sys_get_asyncgen_hooks)
});
static SET_ASYNCGEN_HOOKS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_kw(Arc::from("sys.set_asyncgen_hooks"), sys_set_asyncgen_hooks)
});

static ASYNCGEN_HOOKS_LEN_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.asyncgen_hooks.__len__"), asyncgen_hooks_len)
});
static ASYNCGEN_HOOKS_GETITEM_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("sys.asyncgen_hooks.__getitem__"),
        asyncgen_hooks_getitem,
    )
});
static ASYNCGEN_HOOKS_COUNT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.asyncgen_hooks.count"), asyncgen_hooks_count)
});
static ASYNCGEN_HOOKS_INDEX_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("sys.asyncgen_hooks.index"), asyncgen_hooks_index)
});

static CURRENT_ASYNCGEN_HOOKS: LazyLock<Mutex<AsyncgenHooks>> =
    LazyLock::new(|| Mutex::new(AsyncgenHooks::default()));

#[derive(Clone, Copy)]
struct AsyncgenHooks {
    firstiter: Value,
    finalizer: Value,
}

impl Default for AsyncgenHooks {
    fn default() -> Self {
        Self {
            firstiter: Value::none(),
            finalizer: Value::none(),
        }
    }
}

pub(super) fn get_asyncgen_hooks_function_value() -> Value {
    builtin_value(&GET_ASYNCGEN_HOOKS_FUNCTION)
}

pub(super) fn set_asyncgen_hooks_function_value() -> Value {
    builtin_value(&SET_ASYNCGEN_HOOKS_FUNCTION)
}

pub(crate) fn asyncgen_firstiter_hook() -> Option<Value> {
    non_none_hook(
        CURRENT_ASYNCGEN_HOOKS
            .lock()
            .expect("sys async generator hooks lock poisoned")
            .firstiter,
    )
}

pub(crate) fn asyncgen_finalizer_hook() -> Option<Value> {
    non_none_hook(
        CURRENT_ASYNCGEN_HOOKS
            .lock()
            .expect("sys async generator hooks lock poisoned")
            .finalizer,
    )
}

#[inline]
fn non_none_hook(value: Value) -> Option<Value> {
    (!value.is_none()).then_some(value)
}

fn sys_get_asyncgen_hooks(args: &[Value]) -> Result<Value, BuiltinError> {
    if !args.is_empty() {
        return Err(BuiltinError::TypeError(format!(
            "get_asyncgen_hooks() takes no arguments ({} given)",
            args.len()
        )));
    }

    let hooks = *CURRENT_ASYNCGEN_HOOKS
        .lock()
        .expect("sys async generator hooks lock poisoned");
    Ok(asyncgen_hooks_record(hooks))
}

fn sys_set_asyncgen_hooks(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "set_asyncgen_hooks() takes at most 2 positional arguments ({} given)",
            args.len()
        )));
    }

    let mut firstiter = args.first().copied();
    let mut finalizer = args.get(1).copied();
    let mut firstiter_seen = firstiter.is_some();
    let mut finalizer_seen = finalizer.is_some();

    for &(name, value) in keywords {
        match name {
            FIRSTITER_FIELD => {
                if firstiter_seen {
                    return Err(multiple_values_error(FIRSTITER_FIELD));
                }
                firstiter_seen = true;
                firstiter = Some(value);
            }
            FINALIZER_FIELD => {
                if finalizer_seen {
                    return Err(multiple_values_error(FINALIZER_FIELD));
                }
                finalizer_seen = true;
                finalizer = Some(value);
            }
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "set_asyncgen_hooks() got an unexpected keyword argument '{other}'"
                )));
            }
        }
    }

    if let Some(value) = finalizer {
        validate_asyncgen_hook(value, FINALIZER_FIELD)?;
    }
    if let Some(value) = firstiter {
        validate_asyncgen_hook(value, FIRSTITER_FIELD)?;
    }

    let mut hooks = CURRENT_ASYNCGEN_HOOKS
        .lock()
        .expect("sys async generator hooks lock poisoned");
    if let Some(value) = firstiter {
        hooks.firstiter = value;
    }
    if let Some(value) = finalizer {
        hooks.finalizer = value;
    }

    Ok(Value::none())
}

#[inline]
fn multiple_values_error(name: &'static str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "set_asyncgen_hooks() got multiple values for argument '{name}'"
    ))
}

fn validate_asyncgen_hook(value: Value, name: &'static str) -> Result<(), BuiltinError> {
    if value.is_none() || value_supports_call_protocol(value) {
        return Ok(());
    }

    Err(BuiltinError::TypeError(format!(
        "callable {name} expected, got {}",
        value.type_name()
    )))
}

fn asyncgen_hooks_record(hooks: AsyncgenHooks) -> Value {
    let registry = shape_registry();
    let value = crate::alloc_managed_value(ShapedObject::new_tuple_backed(
        TypeId::OBJECT,
        registry.empty_shape(),
        TupleObject::from_slice(&[hooks.firstiter, hooks.finalizer]),
    ));
    let ptr = value
        .as_object_ptr()
        .expect("managed shaped object allocation returns an object pointer")
        as *mut ShapedObject;
    let record = unsafe { &mut *ptr };

    record.set_property(intern(FIRSTITER_FIELD), hooks.firstiter, registry);
    record.set_property(intern(FINALIZER_FIELD), hooks.finalizer, registry);
    for (name, method) in [
        ("__len__", &*ASYNCGEN_HOOKS_LEN_FUNCTION),
        ("__getitem__", &*ASYNCGEN_HOOKS_GETITEM_FUNCTION),
        ("count", &*ASYNCGEN_HOOKS_COUNT_FUNCTION),
        ("index", &*ASYNCGEN_HOOKS_INDEX_FUNCTION),
    ] {
        record.set_property(intern(name), bound_builtin_value(method, value), registry);
    }

    value
}

#[inline]
fn bound_builtin_value(function: &'static BuiltinFunctionObject, receiver: Value) -> Value {
    crate::alloc_managed_value(function.bind(receiver))
}

fn asyncgen_hooks_tuple(value: Value) -> Result<&'static TupleObject, BuiltinError> {
    value_as_tuple_ref(value).ok_or_else(|| {
        BuiltinError::TypeError("descriptor requires a sys.asyncgen_hooks object".to_string())
    })
}

fn asyncgen_hooks_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "sys.asyncgen_hooks.__len__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = asyncgen_hooks_tuple(args[0])?;
    Ok(int_value(tuple.len() as i64))
}

fn asyncgen_hooks_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "sys.asyncgen_hooks.__getitem__() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = asyncgen_hooks_tuple(args[0])?;
    if let Some(index) = int_like_value(args[1]) {
        return tuple.get(index).ok_or_else(|| {
            BuiltinError::IndexError(format!(
                "index {index} out of range for length {}",
                tuple.len()
            ))
        });
    }

    if let Some(slice) = slice_from_value(args[1]) {
        return Ok(crate::alloc_managed_value(tuple_slice(tuple, slice)));
    }

    Err(BuiltinError::TypeError(format!(
        "tuple indices must be integers or slices, not {}",
        args[1].type_name()
    )))
}

fn asyncgen_hooks_count(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "sys.asyncgen_hooks.count() takes exactly one argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = asyncgen_hooks_tuple(args[0])?;
    let count = tuple
        .iter()
        .copied()
        .filter(|item| crate::ops::comparison::values_equal(*item, args[1]))
        .count();
    Ok(int_value(count as i64))
}

fn asyncgen_hooks_index(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "sys.asyncgen_hooks.index() takes from 1 to 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let tuple = asyncgen_hooks_tuple(args[0])?;
    let start = tuple_index_bound(args.get(2).copied(), 0, tuple.len(), "start")?;
    let stop = tuple_index_bound(
        args.get(3).copied(),
        tuple.len() as i64,
        tuple.len(),
        "stop",
    )?;

    for index in start..stop.max(start) {
        let item = tuple
            .get(index as i64)
            .expect("normalized tuple index should be in bounds");
        if crate::ops::comparison::values_equal(item, args[1]) {
            return Ok(int_value(index as i64));
        }
    }

    Err(BuiltinError::ValueError(
        "tuple.index(x): x not in tuple".to_string(),
    ))
}

fn tuple_index_bound(
    value: Option<Value>,
    default: i64,
    len: usize,
    name: &'static str,
) -> Result<usize, BuiltinError> {
    let raw = match value {
        Some(value) => int_like_value(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "slice indices must be integers or have an __index__ method ({name})"
            ))
        })?,
        None => default,
    };

    let len_i64 = len as i64;
    let normalized = if raw < 0 { len_i64 + raw } else { raw };
    Ok(normalized.clamp(0, len_i64) as usize)
}

fn slice_from_value(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

fn tuple_slice(tuple: &TupleObject, slice: &SliceObject) -> TupleObject {
    let indices = slice.indices(tuple.len());
    let mut items = Vec::with_capacity(indices.length);
    for index in indices.iter() {
        if index < tuple.len() {
            items.push(
                tuple
                    .get(index as i64)
                    .expect("slice index should be in bounds"),
            );
        }
    }
    TupleObject::from_vec(items)
}

#[inline]
fn int_value(value: i64) -> Value {
    Value::int(value).expect("small sys.asyncgen_hooks integer should fit")
}
