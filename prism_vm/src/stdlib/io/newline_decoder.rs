use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::get_attribute_value;
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

const DECODER_ATTR: &str = "_prism_decoder";
const TRANSLATE_ATTR: &str = "_prism_translate";
const ERRORS_ATTR: &str = "_prism_errors";
const PENDING_CR_ATTR: &str = "_prism_pendingcr";
const SEEN_NL_ATTR: &str = "_prism_seennl";

const LF_MASK: u8 = 1;
const CR_MASK: u8 = 2;
const CRLF_MASK: u8 = 4;

static INCREMENTAL_NEWLINE_DECODER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("io.IncrementalNewlineDecoder"),
        incremental_newline_decoder_new,
    )
});
static INCREMENTAL_NEWLINE_DECODE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("io.IncrementalNewlineDecoder.decode"),
        incremental_newline_decoder_decode,
    )
});
static INCREMENTAL_NEWLINE_GETSTATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("io.IncrementalNewlineDecoder.getstate"),
        incremental_newline_decoder_getstate,
    )
});
static INCREMENTAL_NEWLINE_SETSTATE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("io.IncrementalNewlineDecoder.setstate"),
        incremental_newline_decoder_setstate,
    )
});
static INCREMENTAL_NEWLINE_RESET_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("io.IncrementalNewlineDecoder.reset"),
        incremental_newline_decoder_reset,
    )
});

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct IncrementalNewlineState {
    translate: bool,
    pending_cr: bool,
    seen_nl: u8,
}

impl IncrementalNewlineState {
    fn decode_text(&mut self, input: &str, final_chunk: bool) -> String {
        let mut output = String::new();

        if self.pending_cr && (!input.is_empty() || final_chunk) {
            output.push('\r');
            self.pending_cr = false;
        }

        output.push_str(input);

        if output.ends_with('\r') && !final_chunk {
            output.pop();
            self.pending_cr = true;
        }

        let crlf = output.matches("\r\n").count();
        let cr = output.matches('\r').count().saturating_sub(crlf);
        let lf = output.matches('\n').count().saturating_sub(crlf);

        if lf > 0 {
            self.seen_nl |= LF_MASK;
        }
        if cr > 0 {
            self.seen_nl |= CR_MASK;
        }
        if crlf > 0 {
            self.seen_nl |= CRLF_MASK;
        }

        if self.translate {
            if crlf > 0 {
                output = output.replace("\r\n", "\n");
            }
            if cr > 0 {
                output = output.replace('\r', "\n");
            }
        }

        output
    }
}

pub(super) fn incremental_newline_decoder_constructor_value() -> Value {
    builtin_value(&INCREMENTAL_NEWLINE_DECODER)
}

fn incremental_newline_decoder_new(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "IncrementalNewlineDecoder() takes from 2 to 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let decoder = args[0];
    let translate = crate::truthiness::is_truthy(args[1]);
    let errors = if let Some(value) = args.get(2).copied() {
        string_from_value(value, "errors")?
    } else {
        "strict".to_string()
    };

    Ok(new_incremental_newline_decoder_object(
        decoder, translate, &errors,
    ))
}

fn incremental_newline_decoder_decode(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if !(2..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "decode() takes from 1 to 2 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let decoder = decoder_value(receiver)?;
    let final_chunk = args
        .get(2)
        .copied()
        .is_some_and(crate::truthiness::is_truthy);
    let input = if decoder.is_none() {
        string_from_value(args[1], "input")?
    } else {
        let decoded =
            invoke_decoder_method(vm, decoder, "decode", &[args[1], Value::bool(final_chunk)])?;
        string_from_value(decoded, "input")?
    };

    let mut state = load_state(receiver)?;
    let output = state.decode_text(&input, final_chunk);
    store_state(receiver, state)?;
    Ok(string_value(&output))
}

fn incremental_newline_decoder_getstate(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arity(args, "getstate", 0)?;

    let receiver = args[0];
    let decoder = decoder_value(receiver)?;
    let state = load_state(receiver)?;

    let (buffer, flag) = if decoder.is_none() {
        (bytes_value(&[]), 0_i64)
    } else {
        let value = invoke_decoder_method(vm, decoder, "getstate", &[])?;
        let (buffer, flag) = tuple_pair(value, "getstate() result")?;
        (buffer, int_from_value(flag, "decoder state flag")?)
    };

    let combined_flag = (flag << 1) | i64::from(state.pending_cr);
    tuple_value(&[
        buffer,
        Value::int(combined_flag).unwrap_or_else(Value::none),
    ])
}

fn incremental_newline_decoder_setstate(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arity(args, "setstate", 1)?;

    let receiver = args[0];
    let (buffer, flag_value) = tuple_pair(args[1], "state")?;
    let flag = int_from_value(flag_value, "state flag")?;
    let decoder = decoder_value(receiver)?;
    if !decoder.is_none() {
        let forwarded_flag = Value::int(flag >> 1).unwrap_or_else(Value::none);
        let forwarded_state = tuple_value(&[buffer, forwarded_flag])?;
        let _ = invoke_decoder_method(vm, decoder, "setstate", &[forwarded_state])?;
    }

    let mut state = load_state(receiver)?;
    state.pending_cr = (flag & 1) != 0;
    store_state(receiver, state)?;
    Ok(Value::none())
}

fn incremental_newline_decoder_reset(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_method_arity(args, "reset", 0)?;

    let receiver = args[0];
    let decoder = decoder_value(receiver)?;
    if !decoder.is_none() {
        let _ = invoke_decoder_method(vm, decoder, "reset", &[])?;
    }

    let mut state = load_state(receiver)?;
    state.pending_cr = false;
    state.seen_nl = 0;
    store_state(receiver, state)?;
    Ok(Value::none())
}

fn new_incremental_newline_decoder_object(decoder: Value, translate: bool, errors: &str) -> Value {
    let mut object = ShapedObject::with_empty_shape(shape_registry().empty_shape());
    object.set_property(intern(DECODER_ATTR), decoder, shape_registry());
    object.set_property(
        intern(TRANSLATE_ATTR),
        Value::bool(translate),
        shape_registry(),
    );
    object.set_property(intern(ERRORS_ATTR), string_value(errors), shape_registry());
    object.set_property(
        intern(PENDING_CR_ATTR),
        Value::bool(false),
        shape_registry(),
    );
    object.set_property(
        intern(SEEN_NL_ATTR),
        Value::int(0).unwrap(),
        shape_registry(),
    );
    object.set_property(intern("newlines"), Value::none(), shape_registry());
    object.set_property(
        intern("decode"),
        builtin_value(&INCREMENTAL_NEWLINE_DECODE_METHOD),
        shape_registry(),
    );
    object.set_property(
        intern("getstate"),
        builtin_value(&INCREMENTAL_NEWLINE_GETSTATE_METHOD),
        shape_registry(),
    );
    object.set_property(
        intern("setstate"),
        builtin_value(&INCREMENTAL_NEWLINE_SETSTATE_METHOD),
        shape_registry(),
    );
    object.set_property(
        intern("reset"),
        builtin_value(&INCREMENTAL_NEWLINE_RESET_METHOD),
        shape_registry(),
    );
    finalize_incremental_newline_decoder_object(object)
}

fn finalize_incremental_newline_decoder_object(object: ShapedObject) -> Value {
    let value = crate::alloc_managed_value(object);
    let shaped = unsafe { &mut *(value.as_object_ptr().unwrap() as *mut ShapedObject) };
    for interned in builtins_method_names() {
        let Some(method_value) = shaped.get_property_interned(&interned) else {
            continue;
        };
        let method_ptr = method_value
            .as_object_ptr()
            .expect("newline decoder helper methods should be builtin functions");
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        shaped.set_property(
            interned,
            crate::alloc_managed_value(builtin.bind(value)),
            shape_registry(),
        );
    }
    value
}

fn invoke_decoder_method(
    vm: &mut VirtualMachine,
    decoder: Value,
    method_name: &str,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    let callable =
        get_attribute_value(vm, decoder, &intern(method_name)).map_err(BuiltinError::Raised)?;
    invoke_callable_value(vm, callable, args).map_err(BuiltinError::Raised)
}

fn load_state(receiver: Value) -> Result<IncrementalNewlineState, BuiltinError> {
    let shaped = shaped_object_ref(receiver)?;
    Ok(IncrementalNewlineState {
        translate: bool_from_property(shaped, TRANSLATE_ATTR)?,
        pending_cr: bool_from_property(shaped, PENDING_CR_ATTR)?,
        seen_nl: int_from_property(shaped, SEEN_NL_ATTR)? as u8,
    })
}

fn store_state(receiver: Value, state: IncrementalNewlineState) -> Result<(), BuiltinError> {
    let shaped = shaped_object_mut(receiver)?;
    shaped.set_property(
        intern(PENDING_CR_ATTR),
        Value::bool(state.pending_cr),
        shape_registry(),
    );
    shaped.set_property(
        intern(SEEN_NL_ATTR),
        Value::int(state.seen_nl as i64).unwrap_or_else(Value::none),
        shape_registry(),
    );
    shaped.set_property(
        intern("newlines"),
        newlines_value(state.seen_nl)?,
        shape_registry(),
    );
    Ok(())
}

fn newlines_value(seen_nl: u8) -> Result<Value, BuiltinError> {
    match seen_nl {
        0 => Ok(Value::none()),
        LF_MASK => Ok(Value::string(intern("\n"))),
        CR_MASK => Ok(Value::string(intern("\r"))),
        3 => tuple_value(&[Value::string(intern("\r")), Value::string(intern("\n"))]),
        CRLF_MASK => Ok(Value::string(intern("\r\n"))),
        5 => tuple_value(&[Value::string(intern("\n")), Value::string(intern("\r\n"))]),
        6 => tuple_value(&[Value::string(intern("\r")), Value::string(intern("\r\n"))]),
        7 => tuple_value(&[
            Value::string(intern("\r")),
            Value::string(intern("\n")),
            Value::string(intern("\r\n")),
        ]),
        _ => Err(BuiltinError::ValueError(format!(
            "unsupported newline mask {seen_nl}"
        ))),
    }
}

fn expect_method_arity(args: &[Value], name: &str, expected: usize) -> Result<(), BuiltinError> {
    let received = args.len().saturating_sub(1);
    if received != expected {
        return Err(BuiltinError::TypeError(format!(
            "{name}() takes {} positional argument{} but {} were given",
            expected,
            if expected == 1 { "" } else { "s" },
            received
        )));
    }
    Ok(())
}

fn tuple_pair(value: Value, context: &str) -> Result<(Value, Value), BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a 2-tuple")))?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    if header.type_id != TypeId::TUPLE {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be a 2-tuple"
        )));
    }
    let tuple = unsafe { &*(ptr as *const TupleObject) };
    if tuple.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "{context} must be a 2-tuple"
        )));
    }
    Ok((tuple.as_slice()[0], tuple.as_slice()[1]))
}

fn tuple_value(items: &[Value]) -> Result<Value, BuiltinError> {
    Ok(crate::alloc_managed_value(TupleObject::from_slice(items)))
}

fn decoder_value(receiver: Value) -> Result<Value, BuiltinError> {
    property_value(receiver, DECODER_ATTR)
}

fn property_value(receiver: Value, name: &str) -> Result<Value, BuiltinError> {
    shaped_object_ref(receiver)?
        .get_property(name)
        .ok_or_else(|| BuiltinError::AttributeError(format!("missing {name} property")))
}

fn bool_from_property(shaped: &ShapedObject, name: &str) -> Result<bool, BuiltinError> {
    shaped
        .get_property(name)
        .and_then(|value| value.as_bool())
        .ok_or_else(|| BuiltinError::AttributeError(format!("missing boolean {name} property")))
}

fn int_from_property(shaped: &ShapedObject, name: &str) -> Result<i64, BuiltinError> {
    let value = shaped
        .get_property(name)
        .ok_or_else(|| BuiltinError::AttributeError(format!("missing integer {name} property")))?;
    int_from_value(value, name)
}

fn int_from_value(value: Value, context: &str) -> Result<i64, BuiltinError> {
    value.as_int().ok_or_else(|| {
        BuiltinError::TypeError(format!("{context} must be int, not {}", value.type_name()))
    })
}

fn string_from_value(value: Value, context: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|value| value.as_str().to_string())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!("{context} must be str, not {}", value.type_name()))
        })
}

fn string_value(value: &str) -> Value {
    if value.is_empty() {
        Value::string(intern(""))
    } else {
        crate::alloc_managed_value(StringObject::new(value))
    }
}

fn bytes_value(value: &[u8]) -> Value {
    crate::alloc_managed_value(BytesObject::from_slice(value))
}

fn builtins_method_names() -> [InternedString; 4] {
    [
        intern("decode"),
        intern("getstate"),
        intern("setstate"),
        intern("reset"),
    ]
}

fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn shaped_object_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "expected object-backed newline decoder receiver, got {}",
            value.type_name()
        ))
    })?;
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn shaped_object_mut(value: Value) -> Result<&'static mut ShapedObject, BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "expected object-backed newline decoder receiver, got {}",
            value.type_name()
        ))
    })?;
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

#[cfg(test)]
mod tests;
