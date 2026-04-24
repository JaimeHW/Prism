//! Native `_codecs` bootstrap module.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use crate::builtins::{decode_bytes_to_text, encode_text_to_data};
use crate::ops::calls::invoke_callable_value;
use prism_core::Value;
use prism_core::intern::{InternedString, intern, interned_by_ptr};
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use rustc_hash::FxHashMap;
use std::sync::{Arc, LazyLock, RwLock};

static REGISTER_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.register"), builtin_register));
static LOOKUP_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("_codecs.lookup"), builtin_lookup));
static ENCODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.encode"), builtin_encode));
static DECODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.decode"), builtin_decode));
static REGISTER_ERROR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.register_error"), builtin_register_error)
});
static LOOKUP_ERROR_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.lookup_error"), builtin_lookup_error)
});

static ASCII_ENCODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.ascii_encode"), ascii_encode));
static ASCII_DECODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.ascii_decode"), ascii_decode));
static LATIN1_ENCODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.latin_1_encode"), latin_1_encode)
});
static LATIN1_DECODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.latin_1_decode"), latin_1_decode)
});
static UTF8_ENCODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.utf_8_encode"), utf_8_encode));
static UTF8_DECODE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("_codecs.utf_8_decode"), utf_8_decode));
static RAW_UNICODE_ESCAPE_ENCODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.raw_unicode_escape_encode"),
        raw_unicode_escape_encode,
    )
});
static RAW_UNICODE_ESCAPE_DECODE_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.raw_unicode_escape_decode"),
        raw_unicode_escape_decode,
    )
});

static STRICT_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.strict_errors"), builtin_strict_errors)
});
static IGNORE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.ignore_errors"), builtin_ignore_errors)
});
static REPLACE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("_codecs.replace_errors"), builtin_replace_errors)
});
static SURROGATEESCAPE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.surrogateescape_errors"),
        builtin_surrogateescape_errors,
    )
});
static SURROGATEPASS_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.surrogatepass_errors"),
        builtin_surrogatepass_errors,
    )
});
static XMLCHARREFREPLACE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.xmlcharrefreplace_errors"),
        builtin_passthrough_error_handler,
    )
});
static BACKSLASHREPLACE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.backslashreplace_errors"),
        builtin_passthrough_error_handler,
    )
});
static NAMEREPLACE_ERRORS_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("_codecs.namereplace_errors"),
        builtin_passthrough_error_handler,
    )
});

static ASCII_CODEC_INFO: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_codec_info(CodecKind::Ascii));
static LATIN1_CODEC_INFO: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_codec_info(CodecKind::Latin1));
static UTF8_CODEC_INFO: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_codec_info(CodecKind::Utf8));
static UTF8_SIG_CODEC_INFO: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_codec_info(CodecKind::Utf8Sig));
static RAW_UNICODE_ESCAPE_CODEC_INFO: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(|| build_codec_info(CodecKind::RawUnicodeEscape));

static ASCII_CODEC_ENCODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_encode(CodecKind::Ascii));
static ASCII_CODEC_DECODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_decode(CodecKind::Ascii));
static LATIN1_CODEC_ENCODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_encode(CodecKind::Latin1));
static LATIN1_CODEC_DECODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_decode(CodecKind::Latin1));
static UTF8_CODEC_ENCODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_encode(CodecKind::Utf8));
static UTF8_CODEC_DECODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_decode(CodecKind::Utf8));
static UTF8_SIG_CODEC_ENCODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_encode(CodecKind::Utf8Sig));
static UTF8_SIG_CODEC_DECODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_decode(CodecKind::Utf8Sig));
static RAW_UNICODE_ESCAPE_CODEC_ENCODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_encode(CodecKind::RawUnicodeEscape));
static RAW_UNICODE_ESCAPE_CODEC_DECODE: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| codec_method_decode(CodecKind::RawUnicodeEscape));

static REGISTRY: LazyLock<RwLock<CodecRegistry>> =
    LazyLock::new(|| RwLock::new(CodecRegistry::new()));

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CodecKind {
    Ascii,
    Latin1,
    Utf8,
    Utf8Sig,
    RawUnicodeEscape,
}

impl CodecKind {
    fn canonical_name(self) -> &'static str {
        match self {
            Self::Ascii => "ascii",
            Self::Latin1 => "latin-1",
            Self::Utf8 => "utf-8",
            Self::Utf8Sig => "utf-8-sig",
            Self::RawUnicodeEscape => "raw-unicode-escape",
        }
    }

    fn class_name(self) -> &'static str {
        match self {
            Self::Ascii => "ascii_codec_info",
            Self::Latin1 => "latin_1_codec_info",
            Self::Utf8 => "utf_8_codec_info",
            Self::Utf8Sig => "utf_8_sig_codec_info",
            Self::RawUnicodeEscape => "raw_unicode_escape_codec_info",
        }
    }

    fn codec_info_value(self) -> Value {
        let class = match self {
            Self::Ascii => &*ASCII_CODEC_INFO,
            Self::Latin1 => &*LATIN1_CODEC_INFO,
            Self::Utf8 => &*UTF8_CODEC_INFO,
            Self::Utf8Sig => &*UTF8_SIG_CODEC_INFO,
            Self::RawUnicodeEscape => &*RAW_UNICODE_ESCAPE_CODEC_INFO,
        };
        Value::object_ptr(Arc::as_ptr(class) as *const ())
    }

    fn encode_method(self) -> &'static BuiltinFunctionObject {
        match self {
            Self::Ascii => &ASCII_CODEC_ENCODE,
            Self::Latin1 => &LATIN1_CODEC_ENCODE,
            Self::Utf8 => &UTF8_CODEC_ENCODE,
            Self::Utf8Sig => &UTF8_SIG_CODEC_ENCODE,
            Self::RawUnicodeEscape => &RAW_UNICODE_ESCAPE_CODEC_ENCODE,
        }
    }

    fn decode_method(self) -> &'static BuiltinFunctionObject {
        match self {
            Self::Ascii => &ASCII_CODEC_DECODE,
            Self::Latin1 => &LATIN1_CODEC_DECODE,
            Self::Utf8 => &UTF8_CODEC_DECODE,
            Self::Utf8Sig => &UTF8_SIG_CODEC_DECODE,
            Self::RawUnicodeEscape => &RAW_UNICODE_ESCAPE_CODEC_DECODE,
        }
    }
}

struct CodecRegistry {
    search_functions: Vec<Value>,
    error_handlers: FxHashMap<InternedString, Value>,
}

impl CodecRegistry {
    fn new() -> Self {
        let mut error_handlers = FxHashMap::default();
        error_handlers.insert(intern("strict"), builtin_value(&STRICT_ERRORS_FUNCTION));
        error_handlers.insert(intern("ignore"), builtin_value(&IGNORE_ERRORS_FUNCTION));
        error_handlers.insert(intern("replace"), builtin_value(&REPLACE_ERRORS_FUNCTION));
        error_handlers.insert(
            intern("surrogateescape"),
            builtin_value(&SURROGATEESCAPE_ERRORS_FUNCTION),
        );
        error_handlers.insert(
            intern("surrogatepass"),
            builtin_value(&SURROGATEPASS_ERRORS_FUNCTION),
        );
        error_handlers.insert(
            intern("xmlcharrefreplace"),
            builtin_value(&XMLCHARREFREPLACE_ERRORS_FUNCTION),
        );
        error_handlers.insert(
            intern("backslashreplace"),
            builtin_value(&BACKSLASHREPLACE_ERRORS_FUNCTION),
        );
        error_handlers.insert(
            intern("namereplace"),
            builtin_value(&NAMEREPLACE_ERRORS_FUNCTION),
        );
        Self {
            search_functions: Vec::new(),
            error_handlers,
        }
    }
}

pub struct CodecsModule {
    attrs: Vec<Arc<str>>,
    all_value: Value,
}

impl CodecsModule {
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("__all__"),
                Arc::from("register"),
                Arc::from("lookup"),
                Arc::from("encode"),
                Arc::from("decode"),
                Arc::from("register_error"),
                Arc::from("lookup_error"),
                Arc::from("ascii_encode"),
                Arc::from("ascii_decode"),
                Arc::from("latin_1_encode"),
                Arc::from("latin_1_decode"),
                Arc::from("utf_8_encode"),
                Arc::from("utf_8_decode"),
                Arc::from("raw_unicode_escape_encode"),
                Arc::from("raw_unicode_escape_decode"),
            ],
            all_value: export_names_value(),
        }
    }
}

impl Default for CodecsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for CodecsModule {
    fn name(&self) -> &str {
        "_codecs"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all_value),
            "register" => Ok(builtin_value(&REGISTER_FUNCTION)),
            "lookup" => Ok(builtin_value(&LOOKUP_FUNCTION)),
            "encode" => Ok(builtin_value(&ENCODE_FUNCTION)),
            "decode" => Ok(builtin_value(&DECODE_FUNCTION)),
            "register_error" => Ok(builtin_value(&REGISTER_ERROR_FUNCTION)),
            "lookup_error" => Ok(builtin_value(&LOOKUP_ERROR_FUNCTION)),
            "ascii_encode" => Ok(builtin_value(&ASCII_ENCODE_FUNCTION)),
            "ascii_decode" => Ok(builtin_value(&ASCII_DECODE_FUNCTION)),
            "latin_1_encode" => Ok(builtin_value(&LATIN1_ENCODE_FUNCTION)),
            "latin_1_decode" => Ok(builtin_value(&LATIN1_DECODE_FUNCTION)),
            "utf_8_encode" => Ok(builtin_value(&UTF8_ENCODE_FUNCTION)),
            "utf_8_decode" => Ok(builtin_value(&UTF8_DECODE_FUNCTION)),
            "raw_unicode_escape_encode" => Ok(builtin_value(&RAW_UNICODE_ESCAPE_ENCODE_FUNCTION)),
            "raw_unicode_escape_decode" => Ok(builtin_value(&RAW_UNICODE_ESCAPE_DECODE_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module '_codecs' has no attribute '{}'",
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
fn leak_object_value<T>(object: T) -> Value {
    let ptr = Box::into_raw(Box::new(object)) as *const ();
    Value::object_ptr(ptr)
}

fn export_names_value() -> Value {
    leak_object_value(TupleObject::from_vec(
        [
            "register",
            "lookup",
            "encode",
            "decode",
            "register_error",
            "lookup_error",
            "ascii_encode",
            "ascii_decode",
            "latin_1_encode",
            "latin_1_decode",
            "utf_8_encode",
            "utf_8_decode",
            "raw_unicode_escape_encode",
            "raw_unicode_escape_decode",
        ]
        .into_iter()
        .map(|name| Value::string(intern(name)))
        .collect(),
    ))
}

fn build_codec_info(kind: CodecKind) -> Arc<PyClassObject> {
    let class = Arc::new(PyClassObject::new_simple(intern(kind.class_name())));
    class.set_attr(intern("__module__"), Value::string(intern("_codecs")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern(kind.class_name())),
    );
    class.set_attr(intern("name"), Value::string(intern(kind.canonical_name())));
    class.set_attr(intern("encode"), builtin_value(kind.encode_method()));
    class.set_attr(intern("decode"), builtin_value(kind.decode_method()));
    class.set_attr(intern("incrementalencoder"), Value::none());
    class.set_attr(intern("incrementaldecoder"), Value::none());
    class.set_attr(intern("streamreader"), Value::none());
    class.set_attr(intern("streamwriter"), Value::none());
    class.set_attr(intern("_is_text_encoding"), Value::bool(true));

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn codec_method_encode(kind: CodecKind) -> BuiltinFunctionObject {
    BuiltinFunctionObject::new_bound(
        Arc::from(format!("_codecs.{}.encode", kind.class_name())),
        builtin_codec_method_encode,
        Value::string(intern(kind.canonical_name())),
    )
}

fn codec_method_decode(kind: CodecKind) -> BuiltinFunctionObject {
    BuiltinFunctionObject::new_bound(
        Arc::from(format!("_codecs.{}.decode", kind.class_name())),
        builtin_codec_method_decode,
        Value::string(intern(kind.canonical_name())),
    )
}

fn builtin_register(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "register() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    if !crate::ops::calls::value_supports_call_protocol(args[0]) {
        return Err(BuiltinError::TypeError(format!(
            "argument must be callable, not {}",
            args[0].type_name()
        )));
    }
    REGISTRY.write().unwrap().search_functions.push(args[0]);
    Ok(Value::none())
}

fn builtin_lookup(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lookup() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let encoding = expect_string(args[0], "lookup() argument must be str")?;
    if let Some(kind) = codec_kind_for_label(&encoding) {
        return Ok(kind.codec_info_value());
    }

    let search_functions = REGISTRY.read().unwrap().search_functions.clone();
    let name_arg = Value::string(intern(&encoding));
    for search in search_functions {
        let value = invoke_callable_value(vm, search, &[name_arg])
            .map_err(|err| BuiltinError::TypeError(err.to_string()))?;
        if !value.is_none() {
            return Ok(value);
        }
    }

    Err(BuiltinError::KeyError(format!(
        "unknown encoding: {}",
        encoding
    )))
}

fn builtin_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    let (input, encoding, errors) = parse_transform_args(args, "encode")?;
    encode_value(input, lookup_codec(&encoding)?, &errors)
}

fn builtin_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    let (input, encoding, errors) = parse_transform_args(args, "decode")?;
    decode_value(input, lookup_codec(&encoding)?, &errors)
}

fn builtin_register_error(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "register_error() takes exactly 2 arguments ({} given)",
            args.len()
        )));
    }

    let name = expect_string(args[0], "register_error() argument 1 must be str")?;
    if !crate::ops::calls::value_supports_call_protocol(args[1]) {
        return Err(BuiltinError::TypeError(format!(
            "handler must be callable, not {}",
            args[1].type_name()
        )));
    }

    REGISTRY
        .write()
        .unwrap()
        .error_handlers
        .insert(intern(&name), args[1]);
    Ok(Value::none())
}

fn builtin_lookup_error(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "lookup_error() takes exactly one argument ({} given)",
            args.len()
        )));
    }

    let name = expect_string(args[0], "lookup_error() argument must be str")?;
    REGISTRY
        .read()
        .unwrap()
        .error_handlers
        .get(&intern(&name))
        .copied()
        .ok_or_else(|| BuiltinError::KeyError(format!("unknown error handler name '{}'", name)))
}

fn ascii_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_encode(args, CodecKind::Ascii, "ascii_encode")
}

fn ascii_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_decode(args, CodecKind::Ascii, "ascii_decode")
}

fn latin_1_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_encode(args, CodecKind::Latin1, "latin_1_encode")
}

fn latin_1_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_decode(args, CodecKind::Latin1, "latin_1_decode")
}

fn utf_8_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_encode(args, CodecKind::Utf8, "utf_8_encode")
}

fn utf_8_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_decode(args, CodecKind::Utf8, "utf_8_decode")
}

fn raw_unicode_escape_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_encode(
        args,
        CodecKind::RawUnicodeEscape,
        "raw_unicode_escape_encode",
    )
}

fn raw_unicode_escape_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_tuple_decode(
        args,
        CodecKind::RawUnicodeEscape,
        "raw_unicode_escape_decode",
    )
}

fn builtin_codec_method_encode(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "encode() takes 2 or 3 positional arguments ({} given)",
            args.len()
        )));
    }

    let encoding = expect_string(args[0], "codec binding must carry encoding")?;
    let errors = if args.len() == 3 {
        expect_string(args[2], "encode() argument 3 must be str")?
    } else {
        "strict".to_string()
    };
    let out = encode_value(args[1], lookup_codec(&encoding)?, &errors)?;
    Ok(tuple2(out, consumed_length_for_encode(args[1])?))
}

fn builtin_codec_method_decode(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() < 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "decode() takes from 2 to 4 positional arguments ({} given)",
            args.len()
        )));
    }

    let encoding = expect_string(args[0], "codec binding must carry encoding")?;
    let kind = lookup_codec(&encoding)?;
    let errors = if args.len() >= 3 {
        expect_string(args[2], "decode() argument 3 must be str")?
    } else {
        "strict".to_string()
    };
    let out = decode_value(args[1], kind, &errors)?;
    Ok(tuple2(out, consumed_length_for_decode(args[1], kind)?))
}

fn builtin_strict_errors(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "strict_errors() takes exactly one argument ({} given)",
            args.len()
        )));
    }
    Err(BuiltinError::ValueError(
        "codec error handler invoked".to_string(),
    ))
}

fn builtin_ignore_errors(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_passthrough_error_handler(args)
}

fn builtin_replace_errors(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_passthrough_error_handler(args)
}

fn builtin_surrogateescape_errors(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_passthrough_error_handler(args)
}

fn builtin_surrogatepass_errors(args: &[Value]) -> Result<Value, BuiltinError> {
    builtin_passthrough_error_handler(args)
}

fn builtin_passthrough_error_handler(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "error handler takes exactly one argument ({} given)",
            args.len()
        )));
    }

    Ok(leak_object_value(TupleObject::from_vec(vec![
        Value::string(intern("")),
        Value::int(0).expect("zero fits"),
    ])))
}

fn parse_transform_args(
    args: &[Value],
    fn_name: &str,
) -> Result<(Value, String, String), BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes from 1 to 3 positional arguments but {} were given",
            fn_name,
            args.len()
        )));
    }
    let encoding = if args.len() >= 2 {
        expect_string(args[1], &format!("{fn_name}() argument 2 must be str"))?
    } else {
        "utf-8".to_string()
    };
    let errors = if args.len() == 3 {
        expect_string(args[2], &format!("{fn_name}() argument 3 must be str"))?
    } else {
        "strict".to_string()
    };
    Ok((args[0], encoding, errors))
}

fn builtin_tuple_encode(
    args: &[Value],
    kind: CodecKind,
    fn_name: &str,
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes 1 or 2 positional arguments ({} given)",
            fn_name,
            args.len()
        )));
    }
    let errors = if args.len() == 2 {
        expect_string(args[1], &format!("{fn_name}() argument 2 must be str"))?
    } else {
        "strict".to_string()
    };
    let out = encode_value(args[0], kind, &errors)?;
    Ok(tuple2(out, consumed_length_for_encode(args[0])?))
}

fn builtin_tuple_decode(
    args: &[Value],
    kind: CodecKind,
    fn_name: &str,
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "{}() takes from 1 to 3 positional arguments ({} given)",
            fn_name,
            args.len()
        )));
    }
    let errors = if args.len() >= 2 {
        expect_string(args[1], &format!("{fn_name}() argument 2 must be str"))?
    } else {
        "strict".to_string()
    };
    let out = decode_value(args[0], kind, &errors)?;
    Ok(tuple2(out, consumed_length_for_decode(args[0], kind)?))
}

fn tuple2(first: Value, second: Value) -> Value {
    leak_object_value(TupleObject::from_vec(vec![first, second]))
}

fn lookup_codec(label: &str) -> Result<CodecKind, BuiltinError> {
    codec_kind_for_label(label)
        .ok_or_else(|| BuiltinError::KeyError(format!("unknown encoding: {}", label)))
}

fn codec_kind_for_label(label: &str) -> Option<CodecKind> {
    match normalize_encoding_name(label).as_str() {
        "ascii" | "us-ascii" => Some(CodecKind::Ascii),
        "latin1" | "latin-1" | "iso-8859-1" => Some(CodecKind::Latin1),
        "utf8" | "utf-8" => Some(CodecKind::Utf8),
        "utf-8-sig" | "utf8-sig" => Some(CodecKind::Utf8Sig),
        "raw-unicode-escape" | "rawunicodeescape" => Some(CodecKind::RawUnicodeEscape),
        _ => None,
    }
}

fn normalize_encoding_name(label: &str) -> String {
    label
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
        .replace(' ', "-")
}

fn encode_value(value: Value, kind: CodecKind, errors: &str) -> Result<Value, BuiltinError> {
    let input = expect_string(
        value,
        &format!("{} encoder expects str input", kind.canonical_name()),
    )?;
    let bytes = encode_with_kind(&input, kind, errors)?;
    Ok(leak_object_value(BytesObject::from_vec_with_type(
        bytes,
        TypeId::BYTES,
    )))
}

fn decode_value(value: Value, kind: CodecKind, errors: &str) -> Result<Value, BuiltinError> {
    let input = expect_bytes(
        value,
        &format!("{} decoder expects bytes input", kind.canonical_name()),
    )?;
    Ok(string_value(decode_with_kind(&input, kind, errors)?))
}

fn string_value(text: String) -> Value {
    if text.is_ascii() {
        return Value::string(intern(&text));
    }
    leak_object_value(StringObject::from_string(text))
}

fn expect_string(value: Value, message: &str) -> Result<String, BuiltinError> {
    if value.is_string() {
        let ptr = value
            .as_string_object_ptr()
            .ok_or_else(|| BuiltinError::TypeError(message.to_string()))?;
        let interned = interned_by_ptr(ptr as *const u8)
            .ok_or_else(|| BuiltinError::TypeError(message.to_string()))?;
        return Ok(interned.as_str().to_string());
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(message.to_string()));
    };
    if crate::ops::objects::extract_type_id(ptr) != TypeId::STR {
        return Err(BuiltinError::TypeError(message.to_string()));
    }
    let string = unsafe { &*(ptr as *const StringObject) };
    Ok(string.as_str().to_string())
}

fn expect_bytes(value: Value, message: &str) -> Result<Vec<u8>, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(message.to_string()));
    };
    match crate::ops::objects::extract_type_id(ptr) {
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            Ok(bytes.as_bytes().to_vec())
        }
        _ => Err(BuiltinError::TypeError(message.to_string())),
    }
}

fn consumed_length_for_encode(value: Value) -> Result<Value, BuiltinError> {
    let count = expect_string(value, "encoder expects str input")?
        .chars()
        .count() as i64;
    Ok(Value::int(count).expect("consumed length fits"))
}

fn consumed_length_for_decode(value: Value, _kind: CodecKind) -> Result<Value, BuiltinError> {
    let count = expect_bytes(value, "decoder expects bytes input")?.len() as i64;
    Ok(Value::int(count).expect("consumed length fits"))
}

fn encode_with_kind(input: &str, kind: CodecKind, errors: &str) -> Result<Vec<u8>, BuiltinError> {
    let normalized_errors = validate_error_policy(errors)?;
    match kind {
        CodecKind::Ascii => {
            encode_text_to_data(input, Some("ascii"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Latin1 => {
            encode_text_to_data(input, Some("latin-1"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Utf8 => {
            encode_text_to_data(input, Some("utf-8"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Utf8Sig => {
            let payload =
                encode_text_to_data(input, Some("utf-8"), Some(normalized_errors.as_str()))
                    .map_err(remap_unknown_error_handler)?;
            let mut out = Vec::with_capacity(payload.len() + 3);
            out.extend_from_slice(&[0xEF, 0xBB, 0xBF]);
            out.extend_from_slice(&payload);
            Ok(out)
        }
        CodecKind::RawUnicodeEscape => encode_text_to_data(
            input,
            Some("raw-unicode-escape"),
            Some(normalized_errors.as_str()),
        )
        .map_err(remap_unknown_error_handler),
    }
}

fn decode_with_kind(input: &[u8], kind: CodecKind, errors: &str) -> Result<String, BuiltinError> {
    let normalized_errors = validate_error_policy(errors)?;
    match kind {
        CodecKind::Ascii => {
            decode_bytes_to_text(input, Some("ascii"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Latin1 => {
            decode_bytes_to_text(input, Some("latin-1"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Utf8 => {
            decode_bytes_to_text(input, Some("utf-8"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::Utf8Sig => {
            let payload = if input.starts_with(&[0xEF, 0xBB, 0xBF]) {
                &input[3..]
            } else {
                input
            };
            decode_bytes_to_text(payload, Some("utf-8"), Some(normalized_errors.as_str()))
                .map_err(remap_unknown_error_handler)
        }
        CodecKind::RawUnicodeEscape => decode_bytes_to_text(
            input,
            Some("raw-unicode-escape"),
            Some(normalized_errors.as_str()),
        )
        .map_err(remap_unknown_error_handler),
    }
}

fn validate_error_policy(errors: &str) -> Result<String, BuiltinError> {
    let normalized = errors.to_ascii_lowercase();
    match normalized.as_str() {
        "strict" | "ignore" | "replace" | "backslashreplace" | "surrogateescape"
        | "surrogatepass" => Ok(normalized),
        _ => Err(BuiltinError::KeyError(format!(
            "unknown error handler name '{}'",
            errors
        ))),
    }
}

fn remap_unknown_error_handler(err: BuiltinError) -> BuiltinError {
    match err {
        BuiltinError::ValueError(message) if message.starts_with("unknown error handler name ") => {
            BuiltinError::KeyError(message)
        }
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::python_unicode::encode_python_code_point;

    fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
        let ptr = value.as_object_ptr().expect("expected builtin");
        unsafe { &*(ptr as *const BuiltinFunctionObject) }
    }

    #[test]
    fn test_module_exposes_bootstrap_surface() {
        let module = CodecsModule::new();
        assert!(module.get_attr("lookup").unwrap().as_object_ptr().is_some());
        assert!(
            module
                .get_attr("utf_8_encode")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
        assert!(
            module
                .get_attr("raw_unicode_escape_encode")
                .unwrap()
                .as_object_ptr()
                .is_some()
        );
    }

    #[test]
    fn test_lookup_known_and_unknown_codec() {
        let lookup = builtin_from_value(CodecsModule::new().get_attr("lookup").unwrap());
        let mut vm = VirtualMachine::new();
        assert!(
            lookup
                .call_with_vm(&mut vm, &[Value::string(intern("utf-8"))])
                .is_ok()
        );
        assert!(
            lookup
                .call_with_vm(&mut vm, &[Value::string(intern("raw-unicode-escape"))])
                .is_ok()
        );
        let err = lookup
            .call_with_vm(&mut vm, &[Value::string(intern("nope-codec"))])
            .expect_err("unknown codec should fail");
        assert!(matches!(err, BuiltinError::KeyError(_)));
    }

    #[test]
    fn test_register_error_roundtrip() {
        static HANDLER: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
            BuiltinFunctionObject::new(Arc::from("tests.handler"), |_args| Ok(Value::none()))
        });

        let module = CodecsModule::new();
        builtin_from_value(module.get_attr("register_error").unwrap())
            .call(&[
                Value::string(intern("codex-handler")),
                builtin_value(&HANDLER),
            ])
            .expect("register_error should succeed");

        let value = builtin_from_value(module.get_attr("lookup_error").unwrap())
            .call(&[Value::string(intern("codex-handler"))])
            .expect("lookup_error should succeed");
        assert_eq!(
            value.as_object_ptr().expect("handler should be object"),
            &*HANDLER as *const BuiltinFunctionObject as *const ()
        );
    }

    #[test]
    fn test_lookup_error_exposes_surrogate_handlers() {
        let lookup_error =
            builtin_from_value(CodecsModule::new().get_attr("lookup_error").unwrap());

        for handler_name in ["surrogateescape", "surrogatepass", "backslashreplace"] {
            let value = lookup_error
                .call(&[Value::string(intern(handler_name))])
                .expect("built-in surrogate handler should resolve");
            assert!(
                value.as_object_ptr().is_some(),
                "lookup_error should return callable object for {handler_name}"
            );
        }
    }

    #[test]
    fn test_utf8_encode_returns_tuple() {
        let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_encode").unwrap())
            .call(&[Value::string(intern("hello"))])
            .expect("utf_8_encode should succeed");
        let ptr = value.as_object_ptr().expect("tuple should be object");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        let bytes_ptr = tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("bytes should be object");
        let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
        assert_eq!(bytes.as_bytes(), b"hello");
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(5));
    }

    #[test]
    fn test_utf8_encode_supports_surrogatepass() {
        let surrogate = encode_python_code_point(0xDC80).expect("surrogate carrier should encode");
        let text = format!("A{surrogate}");

        let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_encode").unwrap())
            .call(&[
                leak_object_value(StringObject::from_string(text)),
                Value::string(intern("surrogatepass")),
            ])
            .expect("utf_8_encode should accept surrogatepass");

        let ptr = value.as_object_ptr().expect("tuple should be object");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        let bytes_ptr = tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("bytes should be object");
        let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
        assert_eq!(bytes.as_bytes(), &[0x41, 0xED, 0xB2, 0x80]);
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(2));
    }

    #[test]
    fn test_raw_unicode_escape_encode_decode_returns_codec_tuples() {
        let encoded =
            builtin_from_value(CodecsModule::new().get_attr("raw_unicode_escape_encode").unwrap())
                .call(&[Value::string(intern("\u{0100}"))])
                .expect("raw_unicode_escape_encode should succeed");
        let encoded_ptr = encoded.as_object_ptr().expect("tuple should be object");
        let encoded_tuple = unsafe { &*(encoded_ptr as *const TupleObject) };
        let bytes_ptr = encoded_tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("bytes should be object");
        let bytes = unsafe { &*(bytes_ptr as *const BytesObject) };
        assert_eq!(bytes.as_bytes(), b"\\u0100");
        assert_eq!(encoded_tuple.get(1).unwrap().as_int(), Some(1));

        let decoded =
            builtin_from_value(CodecsModule::new().get_attr("raw_unicode_escape_decode").unwrap())
                .call(&[leak_object_value(BytesObject::from_vec_with_type(
                    b"\\u0100".to_vec(),
                    TypeId::BYTES,
                ))])
                .expect("raw_unicode_escape_decode should succeed");
        let decoded_ptr = decoded.as_object_ptr().expect("tuple should be object");
        let decoded_tuple = unsafe { &*(decoded_ptr as *const TupleObject) };
        let text_ptr = decoded_tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("decoded text should be object");
        let text = unsafe { &*(text_ptr as *const StringObject) };
        assert_eq!(text.as_str(), "\u{0100}");
        assert_eq!(decoded_tuple.get(1).unwrap().as_int(), Some(6));
    }

    #[test]
    fn test_utf8_decode_supports_surrogateescape() {
        let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_decode").unwrap())
            .call(&[
                leak_object_value(BytesObject::from_vec_with_type(
                    vec![0x41, 0xFF, 0x42],
                    TypeId::BYTES,
                )),
                Value::string(intern("surrogateescape")),
            ])
            .expect("utf_8_decode should accept surrogateescape");

        let ptr = value.as_object_ptr().expect("tuple should be object");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        let string_ptr = tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("decoded text should be object");
        let decoded = unsafe { &*(string_ptr as *const StringObject) };
        let escaped = encode_python_code_point(0xDCFF).expect("surrogate carrier should encode");
        assert_eq!(decoded.as_str(), format!("A{escaped}B"));
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(3));
    }

    #[test]
    fn test_utf8_decode_supports_backslashreplace() {
        let value = builtin_from_value(CodecsModule::new().get_attr("utf_8_decode").unwrap())
            .call(&[
                leak_object_value(BytesObject::from_vec_with_type(
                    vec![0x41, 0xFF, 0x42],
                    TypeId::BYTES,
                )),
                Value::string(intern("backslashreplace")),
            ])
            .expect("utf_8_decode should accept backslashreplace");

        let ptr = value.as_object_ptr().expect("tuple should be object");
        let tuple = unsafe { &*(ptr as *const TupleObject) };
        assert_eq!(tuple.get(1).unwrap().as_int(), Some(3));

        let string_ptr = tuple
            .get(0)
            .unwrap()
            .as_object_ptr()
            .expect("decoded text should be object");
        let decoded = unsafe { &*(string_ptr as *const StringObject) };
        assert_eq!(decoded.as_str(), r"A\xffB");
    }
}
