//! Native `gc` module compatibility surface.
//!
//! CPython's regression suite and `test.support` helpers import `gc` directly
//! for manual collections and basic tracking queries. Prism exposes the most
//! commonly exercised API natively so those flows can run without depending on
//! extension-backed CPython internals.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::python_numeric::int_like_value;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashSet;
use std::sync::{Arc, LazyLock, RwLock};

static COLLECT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("gc.collect"), builtin_collect));
static DISABLE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("gc.disable"), builtin_disable));
static ENABLE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("gc.enable"), builtin_enable));
static ISENABLED_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("gc.isenabled"), builtin_isenabled));
static ISTRACKED_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("gc.is_tracked"), builtin_is_tracked));
static GETCOUNT_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("gc.get_count"), builtin_get_count));
static GETTHRESHOLD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("gc.get_threshold"), builtin_get_threshold)
});
static SETTHRESHOLD_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("gc.set_threshold"), builtin_set_threshold)
});
static GC_STATE: LazyLock<RwLock<GcState>> = LazyLock::new(|| RwLock::new(GcState::default()));

const DEFAULT_THRESHOLDS: [i64; 3] = [700, 10, 10];

#[derive(Debug, Clone, Copy)]
struct GcState {
    enabled: bool,
    thresholds: [i64; 3],
}

impl Default for GcState {
    fn default() -> Self {
        Self {
            enabled: true,
            thresholds: DEFAULT_THRESHOLDS,
        }
    }
}

/// Native `gc` module descriptor.
pub struct GcModule {
    attrs: Vec<Arc<str>>,
    callbacks_value: Value,
    garbage_value: Value,
}

impl GcModule {
    /// Create a new `gc` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("callbacks"),
                Arc::from("collect"),
                Arc::from("disable"),
                Arc::from("enable"),
                Arc::from("garbage"),
                Arc::from("get_count"),
                Arc::from("get_threshold"),
                Arc::from("is_tracked"),
                Arc::from("isenabled"),
                Arc::from("set_threshold"),
            ],
            callbacks_value: leak_object_value(ListObject::new()),
            garbage_value: leak_object_value(ListObject::new()),
        }
    }
}

impl Default for GcModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GcModule {
    fn name(&self) -> &str {
        "gc"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "callbacks" => Ok(self.callbacks_value),
            "collect" => Ok(builtin_value(&COLLECT_FUNCTION)),
            "disable" => Ok(builtin_value(&DISABLE_FUNCTION)),
            "enable" => Ok(builtin_value(&ENABLE_FUNCTION)),
            "garbage" => Ok(self.garbage_value),
            "get_count" => Ok(builtin_value(&GETCOUNT_FUNCTION)),
            "get_threshold" => Ok(builtin_value(&GETTHRESHOLD_FUNCTION)),
            "is_tracked" => Ok(builtin_value(&ISTRACKED_FUNCTION)),
            "isenabled" => Ok(builtin_value(&ISENABLED_FUNCTION)),
            "set_threshold" => Ok(builtin_value(&SETTHRESHOLD_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'gc' has no attribute '{}'",
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
fn leak_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

#[inline]
fn tuple_value(values: &[Value]) -> Value {
    leak_object_value(TupleObject::from_slice(values))
}

fn builtin_collect(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 1 {
        return Err(BuiltinError::TypeError(format!(
            "collect() takes at most 1 argument ({} given)",
            args.len()
        )));
    }

    let generation = args
        .first()
        .copied()
        .map(|value| {
            int_like_value(value).ok_or_else(|| {
                BuiltinError::TypeError("collect() argument must be an integer".to_string())
            })
        })
        .transpose()?
        .unwrap_or(2);

    if !(0..=2).contains(&generation) {
        return Err(BuiltinError::ValueError("invalid generation".to_string()));
    }

    super::_weakref::clear_unreachable_weakrefs(vm);

    // The heap collector is currently conservative at the Python boundary:
    // weakrefs are finalized above, while live VM objects are left untouched.
    Ok(Value::int(0).expect("gc.collect result should fit"))
}

fn builtin_disable(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("disable", args)?;
    GC_STATE.write().unwrap().enabled = false;
    Ok(Value::none())
}

fn builtin_enable(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("enable", args)?;
    GC_STATE.write().unwrap().enabled = true;
    Ok(Value::none())
}

fn builtin_isenabled(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("isenabled", args)?;
    Ok(Value::bool(GC_STATE.read().unwrap().enabled))
}

fn builtin_is_tracked(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "is_tracked() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let mut seen = FxHashSet::default();
    Ok(Value::bool(is_tracked_value(args[0], &mut seen)))
}

fn builtin_get_count(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("get_count", args)?;
    Ok(tuple_value(&[
        Value::int(0).unwrap(),
        Value::int(0).unwrap(),
        Value::int(0).unwrap(),
    ]))
}

fn builtin_get_threshold(args: &[Value]) -> Result<Value, BuiltinError> {
    expect_no_args("get_threshold", args)?;
    let thresholds = GC_STATE.read().unwrap().thresholds;
    Ok(tuple_value(&[
        Value::int(thresholds[0]).unwrap(),
        Value::int(thresholds[1]).unwrap(),
        Value::int(thresholds[2]).unwrap(),
    ]))
}

fn builtin_set_threshold(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "set_threshold() takes from 1 to 3 positional arguments ({} given)",
            args.len()
        )));
    }

    let mut state = GC_STATE.write().unwrap();
    for (index, value) in args.iter().copied().enumerate() {
        state.thresholds[index] = int_like_value(value).ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "an integer is required (got type {})",
                value.type_name()
            ))
        })?;
    }
    if args.len() == 1 {
        state.thresholds[1] = DEFAULT_THRESHOLDS[1];
        state.thresholds[2] = DEFAULT_THRESHOLDS[2];
    } else if args.len() == 2 {
        state.thresholds[2] = DEFAULT_THRESHOLDS[2];
    }
    Ok(Value::none())
}

#[inline]
fn expect_no_args(name: &str, args: &[Value]) -> Result<(), BuiltinError> {
    if args.is_empty() {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "{name}() takes no arguments ({} given)",
            args.len()
        )))
    }
}

fn is_tracked_value(value: Value, seen: &mut FxHashSet<usize>) -> bool {
    if value.is_none()
        || value.as_bool().is_some()
        || value.as_int().is_some()
        || value.as_float().is_some()
    {
        return false;
    }
    if value.is_string() {
        return false;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    let ptr_key = ptr as usize;
    if !seen.insert(ptr_key) {
        return true;
    }

    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::STR
        | TypeId::BYTES
        | TypeId::BYTEARRAY
        | TypeId::RANGE
        | TypeId::SLICE
        | TypeId::COMPLEX
        | TypeId::ELLIPSIS
        | TypeId::NOT_IMPLEMENTED => false,
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple
                .as_slice()
                .iter()
                .copied()
                .any(|item| is_tracked_value(item, seen))
        }
        TypeId::DICT => {
            let dict = unsafe { &*(ptr as *const DictObject) };
            dict.iter()
                .any(|(key, value)| is_tracked_value(key, seen) || is_tracked_value(value, seen))
        }
        TypeId::LIST | TypeId::SET | TypeId::FROZENSET | TypeId::DEQUE => true,
        _ => true,
    }
}

fn gc_thresholds() -> [i64; 3] {
    GC_STATE.read().unwrap().thresholds
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value
            .as_object_ptr()
            .expect("expected builtin function object");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    fn reset_state() {
        *GC_STATE.write().unwrap() = GcState::default();
    }

    #[test]
    fn test_module_exposes_expected_attributes() {
        let module = GcModule::new();
        assert_eq!(module.name(), "gc");
        assert!(
            module
                .get_attr("collect")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
        assert!(
            module
                .get_attr("garbage")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
        assert!(
            module
                .get_attr("callbacks")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
    }

    #[test]
    fn test_enable_disable_and_isenabled_roundtrip() {
        reset_state();
        let module = GcModule::new();
        let disable = builtin_from_value(module.get_attr("disable").unwrap());
        let enable = builtin_from_value(module.get_attr("enable").unwrap());
        let isenabled = builtin_from_value(module.get_attr("isenabled").unwrap());

        assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(true));
        disable.call(&[]).unwrap();
        assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(false));
        enable.call(&[]).unwrap();
        assert_eq!(isenabled.call(&[]).unwrap(), Value::bool(true));
    }

    #[test]
    fn test_collect_accepts_optional_generation() {
        reset_state();
        let module = GcModule::new();
        let collect = builtin_from_value(module.get_attr("collect").unwrap());
        let mut vm = VirtualMachine::new();

        assert_eq!(
            collect.call_with_vm(&mut vm, &[]).unwrap().as_int(),
            Some(0)
        );
        assert_eq!(
            collect
                .call_with_vm(&mut vm, &[Value::int(0).unwrap()])
                .unwrap()
                .as_int(),
            Some(0)
        );

        let err = collect
            .call_with_vm(&mut vm, &[Value::int(3).unwrap()])
            .expect_err("invalid generation should fail");
        assert!(err.to_string().contains("invalid generation"));
    }

    #[test]
    fn test_get_and_set_threshold_roundtrip() {
        reset_state();
        let module = GcModule::new();
        let get_threshold = builtin_from_value(module.get_attr("get_threshold").unwrap());
        let set_threshold = builtin_from_value(module.get_attr("set_threshold").unwrap());

        let initial = get_threshold.call(&[]).unwrap();
        let initial_ptr = initial
            .as_object_ptr()
            .expect("thresholds should be a tuple");
        let initial_tuple = unsafe { &*(initial_ptr as *const TupleObject) };
        assert_eq!(
            initial_tuple
                .as_slice()
                .iter()
                .map(|value| value.as_int().unwrap())
                .collect::<Vec<_>>(),
            DEFAULT_THRESHOLDS
        );

        set_threshold
            .call(&[
                Value::int(100).unwrap(),
                Value::int(20).unwrap(),
                Value::int(5).unwrap(),
            ])
            .unwrap();
        assert_eq!(gc_thresholds(), [100, 20, 5]);
    }

    #[test]
    fn test_is_tracked_matches_container_shape() {
        let atomic_tuple =
            TupleObject::from_slice(&[Value::int(1).unwrap(), Value::int(2).unwrap()]);
        let atomic_tuple_ptr = Box::into_raw(Box::new(atomic_tuple));
        let atomic_tuple_value = Value::object_ptr(atomic_tuple_ptr as *const ());

        let tracked_list = ListObject::from_slice(&[Value::int(1).unwrap()]);
        let tracked_list_ptr = Box::into_raw(Box::new(tracked_list));
        let tracked_list_value = Value::object_ptr(tracked_list_ptr as *const ());

        let tracked_tuple = TupleObject::from_slice(&[tracked_list_value]);
        let tracked_tuple_ptr = Box::into_raw(Box::new(tracked_tuple));
        let tracked_tuple_value = Value::object_ptr(tracked_tuple_ptr as *const ());

        let mut atomic_dict = DictObject::new();
        atomic_dict.set(Value::string(intern("a")), Value::int(1).unwrap());
        let atomic_dict_ptr = Box::into_raw(Box::new(atomic_dict));
        let atomic_dict_value = Value::object_ptr(atomic_dict_ptr as *const ());

        let mut nested_dict = DictObject::new();
        nested_dict.set(Value::string(intern("a")), tracked_list_value);
        let nested_dict_ptr = Box::into_raw(Box::new(nested_dict));
        let nested_dict_value = Value::object_ptr(nested_dict_ptr as *const ());

        let set = SetObject::from_slice(&[Value::int(1).unwrap()]);
        let set_ptr = Box::into_raw(Box::new(set));
        let set_value = Value::object_ptr(set_ptr as *const ());

        assert!(!is_tracked_value(
            Value::int(1).unwrap(),
            &mut FxHashSet::default()
        ));
        assert!(!is_tracked_value(
            Value::string(intern("abc")),
            &mut FxHashSet::default()
        ));
        assert!(!is_tracked_value(
            atomic_tuple_value,
            &mut FxHashSet::default()
        ));
        assert!(is_tracked_value(
            tracked_tuple_value,
            &mut FxHashSet::default()
        ));
        assert!(!is_tracked_value(
            atomic_dict_value,
            &mut FxHashSet::default()
        ));
        assert!(is_tracked_value(
            nested_dict_value,
            &mut FxHashSet::default()
        ));
        assert!(is_tracked_value(set_value, &mut FxHashSet::default()));

        unsafe {
            drop(Box::from_raw(set_ptr));
            drop(Box::from_raw(nested_dict_ptr));
            drop(Box::from_raw(atomic_dict_ptr));
            drop(Box::from_raw(tracked_tuple_ptr));
            drop(Box::from_raw(tracked_list_ptr));
            drop(Box::from_raw(atomic_tuple_ptr));
        }
    }

    #[test]
    fn test_is_tracked_treats_byte_sequences_as_untracked() {
        let bytes = BytesObject::from_slice(b"abc");
        let bytes_ptr = Box::into_raw(Box::new(bytes));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());

        let bytearray = BytesObject::bytearray_from_slice(b"abc");
        let bytearray_ptr = Box::into_raw(Box::new(bytearray));
        let bytearray_value = Value::object_ptr(bytearray_ptr as *const ());

        assert!(!is_tracked_value(bytes_value, &mut FxHashSet::default()));
        assert!(!is_tracked_value(
            bytearray_value,
            &mut FxHashSet::default()
        ));

        unsafe {
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(bytes_ptr));
        }
    }
}
