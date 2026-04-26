//! Native `pickle` bootstrap subset.
//!
//! The full pickle protocol is intentionally broad. This native module starts
//! with the protocol-stable scalar opcodes that CPython's bool regression tests
//! require, keeping the implementation table-driven and easy to extend without
//! penalizing ordinary runtime startup.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use num_traits::ToPrimitive;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::int::value_to_bigint;
use prism_runtime::types::list::ListObject;
use std::sync::{Arc, LazyLock};

const HIGHEST_PROTOCOL: i64 = 5;
const DEFAULT_PROTOCOL: i64 = 4;
const TRUE_PROTO0: &[u8] = b"I01\n.";
const FALSE_PROTO0: &[u8] = b"I00\n.";
const PROTO: u8 = 0x80;
const NEWTRUE: u8 = 0x88;
const NEWFALSE: u8 = 0x89;
const STOP: u8 = b'.';

static DUMPS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_kw(Arc::from("pickle.dumps"), pickle_dumps));
static LOADS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("pickle.loads"), pickle_loads));

/// Native `pickle` module descriptor.
#[derive(Debug, Clone)]
pub struct PickleModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl PickleModule {
    /// Create a native `pickle` module.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("DEFAULT_PROTOCOL"),
                Arc::from("HIGHEST_PROTOCOL"),
                Arc::from("__all__"),
                Arc::from("dumps"),
                Arc::from("loads"),
            ],
            all: string_list_value(&["dumps", "loads"]),
        }
    }
}

impl Default for PickleModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for PickleModule {
    fn name(&self) -> &str {
        "pickle"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "DEFAULT_PROTOCOL" => Ok(Value::int(DEFAULT_PROTOCOL).expect("protocol fits")),
            "HIGHEST_PROTOCOL" => Ok(Value::int(HIGHEST_PROTOCOL).expect("protocol fits")),
            "__all__" => Ok(self.all),
            "dumps" => Ok(builtin_value(&DUMPS_FUNCTION)),
            "loads" => Ok(builtin_value(&LOADS_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'pickle' has no attribute '{}'",
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
fn bytes_value(bytes: &[u8]) -> Value {
    crate::alloc_managed_value(BytesObject::from_slice(bytes))
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn pickle_dumps(args: &[Value], keywords: &[(&str, Value)]) -> Result<Value, BuiltinError> {
    if !(1..=2).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "dumps() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut protocol = args.get(1).copied();
    for (name, value) in keywords {
        match *name {
            "protocol" => assign_keyword(&mut protocol, *value, "protocol")?,
            "fix_imports" | "buffer_callback" => {}
            other => {
                return Err(BuiltinError::TypeError(format!(
                    "dumps() got an unexpected keyword argument '{}'",
                    other
                )));
            }
        }
    }

    let protocol = protocol
        .filter(|value| !value.is_none())
        .map(protocol_number)
        .transpose()?
        .unwrap_or(DEFAULT_PROTOCOL);
    let protocol = normalize_protocol(protocol)?;

    let Some(boolean) = args[0].as_bool() else {
        return Err(BuiltinError::TypeError(format!(
            "cannot pickle '{}' objects yet",
            args[0].type_name()
        )));
    };

    Ok(bytes_value(&encode_bool(boolean, protocol)))
}

fn pickle_loads(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "loads() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let bytes = bytes_from_value(args[0], "loads() argument")?;
    decode_bool_pickle(&bytes)
        .ok_or_else(|| BuiltinError::ValueError("unsupported or invalid pickle stream".to_string()))
}

fn assign_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &'static str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "dumps() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn protocol_number(value: Value) -> Result<i64, BuiltinError> {
    if let Some(flag) = value.as_bool() {
        return Ok(i64::from(flag));
    }
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    value_to_bigint(value)
        .and_then(|integer| integer.to_i64())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "pickle protocol must be int, not {}",
                value.type_name()
            ))
        })
}

fn normalize_protocol(protocol: i64) -> Result<i64, BuiltinError> {
    if protocol == -1 {
        return Ok(HIGHEST_PROTOCOL);
    }
    if (0..=HIGHEST_PROTOCOL).contains(&protocol) {
        Ok(protocol)
    } else {
        Err(BuiltinError::ValueError(format!(
            "pickle protocol must be <= {HIGHEST_PROTOCOL}"
        )))
    }
}

fn encode_bool(value: bool, protocol: i64) -> Vec<u8> {
    if protocol < 2 {
        return if value {
            TRUE_PROTO0.to_vec()
        } else {
            FALSE_PROTO0.to_vec()
        };
    }

    vec![
        PROTO,
        u8::try_from(protocol).expect("protocol is normalized to a byte"),
        if value { NEWTRUE } else { NEWFALSE },
        STOP,
    ]
}

fn bytes_from_value(value: Value, context: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        )));
    };

    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => Ok(unsafe { &*(ptr as *const BytesObject) }.to_vec()),
        _ => Err(BuiltinError::TypeError(format!(
            "{context} must be bytes-like, not {}",
            value.type_name()
        ))),
    }
}

fn decode_bool_pickle(bytes: &[u8]) -> Option<Value> {
    match bytes {
        TRUE_PROTO0 => Some(Value::bool(true)),
        FALSE_PROTO0 => Some(Value::bool(false)),
        [PROTO, protocol, NEWTRUE, STOP] if i64::from(*protocol) <= HIGHEST_PROTOCOL => {
            Some(Value::bool(true))
        }
        [PROTO, protocol, NEWFALSE, STOP] if i64::from(*protocol) <= HIGHEST_PROTOCOL => {
            Some(Value::bool(false))
        }
        _ => None,
    }
}
