//! Native public `string` module.
//!
//! The hot constants and the small class surface used by the standard library
//! are implemented natively so Prism keeps bootstrap fast without delegating
//! public behavior to CPython's source module.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, allocate_heap_instance_for_class, builtin_str_vm,
    runtime_error_to_builtin_error,
};
use crate::ops::calls::invoke_callable_value;
use crate::ops::dict_access::dict_get_item;
use crate::ops::objects::{
    dict_storage_ref_from_ptr, extract_type_id, get_attribute_value, list_storage_ref_from_ptr,
    tuple_storage_ref_from_ptr,
};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::int::value_to_i64;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use std::sync::{Arc, LazyLock};

const ASCII_LOWERCASE: &str = "abcdefghijklmnopqrstuvwxyz";
const ASCII_UPPERCASE: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const ASCII_LETTERS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const DIGITS: &str = "0123456789";
const HEXDIGITS: &str = "0123456789abcdefABCDEF";
const OCTDIGITS: &str = "01234567";
const PUNCTUATION: &str = r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##;
const WHITESPACE: &str = " \t\n\r\x0b\x0c";
const PRINTABLE: &str = concat!(
    "0123456789",
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    r##"!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"##,
    " \t\n\r\x0b\x0c"
);

const TEMPLATE_DELIMITER: &str = "$";
const TEMPLATE_ID_PATTERN: &str = r"(?a:[_a-z][_a-z0-9]*)";

const EXPORTS: &[&str] = &[
    "ascii_lowercase",
    "ascii_uppercase",
    "ascii_letters",
    "digits",
    "hexdigits",
    "octdigits",
    "punctuation",
    "whitespace",
    "printable",
    "capwords",
    "Template",
    "Formatter",
];

static CAPWORDS_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("string.capwords"), capwords));

static FORMATTER_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("string.Formatter.__init__"), formatter_init)
});
static FORMATTER_PARSE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("string.Formatter.parse"), formatter_parse)
});
static FORMATTER_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("string.Formatter.format"), formatter_format)
});
static FORMATTER_VFORMAT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("string.Formatter.vformat"), formatter_vformat)
});

static TEMPLATE_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("string.Template.__init__"), template_init)
});
static TEMPLATE_SUBSTITUTE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("string.Template.substitute"), template_substitute)
});
static TEMPLATE_SAFE_SUBSTITUTE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("string.Template.safe_substitute"),
        template_safe_substitute,
    )
});
static TEMPLATE_IS_VALID_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(Arc::from("string.Template.is_valid"), template_is_valid)
});
static TEMPLATE_GET_IDENTIFIERS_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("string.Template.get_identifiers"),
        template_get_identifiers,
    )
});

static TEMPLATE_PATTERN_FINDITER_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("string.Template.pattern.finditer"),
        template_pattern_finditer,
    )
});
static TEMPLATE_MATCH_GROUPDICT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("string.Template.pattern.match.groupdict"),
        template_match_groupdict,
    )
});
static TEMPLATE_MATCH_GROUP_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("string.Template.pattern.match.group"),
        template_match_group,
    )
});

static FORMATTER_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_string_class(
        "Formatter",
        &[
            ("__init__", builtin_value(&FORMATTER_INIT_METHOD)),
            ("format", builtin_value(&FORMATTER_FORMAT_METHOD)),
            ("vformat", builtin_value(&FORMATTER_VFORMAT_METHOD)),
            ("parse", builtin_value(&FORMATTER_PARSE_METHOD)),
        ],
    )
});

static TEMPLATE_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_string_class(
        "Template",
        &[
            ("delimiter", string_value(TEMPLATE_DELIMITER)),
            ("idpattern", string_value(TEMPLATE_ID_PATTERN)),
            ("braceidpattern", Value::none()),
            ("flags", Value::int_unchecked(2)),
            ("pattern", *TEMPLATE_PATTERN_VALUE),
            ("__init__", builtin_value(&TEMPLATE_INIT_METHOD)),
            ("substitute", builtin_value(&TEMPLATE_SUBSTITUTE_METHOD)),
            (
                "safe_substitute",
                builtin_value(&TEMPLATE_SAFE_SUBSTITUTE_METHOD),
            ),
            ("is_valid", builtin_value(&TEMPLATE_IS_VALID_METHOD)),
            (
                "get_identifiers",
                builtin_value(&TEMPLATE_GET_IDENTIFIERS_METHOD),
            ),
        ],
    )
});

static TEMPLATE_PATTERN_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_string_class(
        "_TemplatePattern",
        &[("finditer", builtin_value(&TEMPLATE_PATTERN_FINDITER_METHOD))],
    )
});

static TEMPLATE_MATCH_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(|| {
    build_native_string_class(
        "_TemplateMatch",
        &[
            ("groupdict", builtin_value(&TEMPLATE_MATCH_GROUPDICT_METHOD)),
            ("group", builtin_value(&TEMPLATE_MATCH_GROUP_METHOD)),
        ],
    )
});

static TEMPLATE_PATTERN_VALUE: LazyLock<Value> = LazyLock::new(|| {
    let instance = allocate_heap_instance_for_class(TEMPLATE_PATTERN_CLASS.as_ref());
    let ptr = Box::leak(Box::new(instance)) as *mut ShapedObject as *const ();
    Value::object_ptr(ptr)
});

/// Native `string` module descriptor.
#[derive(Debug, Clone)]
pub struct StringPublicModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl StringPublicModule {
    /// Create a native public `string` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS
                .iter()
                .copied()
                .chain(["__all__"])
                .map(Arc::from)
                .collect(),
            all: string_list_value(EXPORTS),
        }
    }
}

impl Default for StringPublicModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for StringPublicModule {
    fn name(&self) -> &str {
        "string"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "ascii_lowercase" => Ok(string_value(ASCII_LOWERCASE)),
            "ascii_uppercase" => Ok(string_value(ASCII_UPPERCASE)),
            "ascii_letters" => Ok(string_value(ASCII_LETTERS)),
            "digits" => Ok(string_value(DIGITS)),
            "hexdigits" => Ok(string_value(HEXDIGITS)),
            "octdigits" => Ok(string_value(OCTDIGITS)),
            "punctuation" => Ok(string_value(PUNCTUATION)),
            "whitespace" => Ok(string_value(WHITESPACE)),
            "printable" => Ok(string_value(PRINTABLE)),
            "capwords" => Ok(builtin_value(&CAPWORDS_FUNCTION)),
            "Template" => Ok(class_value(&TEMPLATE_CLASS)),
            "Formatter" => Ok(class_value(&FORMATTER_CLASS)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'string' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[derive(Debug, Clone)]
struct TemplateMatchData {
    start: usize,
    end: usize,
    full: String,
    named: Option<String>,
    braced: Option<String>,
    escaped: Option<String>,
    invalid: Option<String>,
    invalid_index: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubstituteMode {
    Strict,
    Safe,
}

#[inline]
fn string_value(value: &str) -> Value {
    Value::string(intern(value))
}

#[inline]
fn owned_string_value(value: String) -> Value {
    crate::alloc_managed_value(StringObject::from_string(value))
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn class_value(class: &Arc<PyClassObject>) -> Value {
    Value::object_ptr(Arc::as_ptr(class) as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn list_value(values: Vec<Value>) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn build_native_string_class(name: &str, attrs: &[(&str, Value)]) -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern(name));
    class.set_attr(intern("__module__"), Value::string(intern("string")));
    class.set_attr(intern("__qualname__"), Value::string(intern(name)));
    for &(attr_name, attr_value) in attrs {
        class.set_attr(intern(attr_name), attr_value);
    }
    let mut flags = ClassFlags::INITIALIZED | ClassFlags::NATIVE_HEAPTYPE;
    if attrs.iter().any(|(attr_name, _)| *attr_name == "__init__") {
        flags |= ClassFlags::HAS_INIT;
    }
    class.add_flags(flags);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }

    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn value_to_string(value: Value, context: &str) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|string| string.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a str")))
}

fn capwords(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "capwords() takes from 1 to 2 arguments ({} given)",
            args.len()
        )));
    }

    let input = value_to_string(args[0], "s")?;
    let separator = match args.get(1).copied() {
        None => None,
        Some(value) if value.is_none() => None,
        Some(value) => Some(value_to_string(value, "sep")?),
    };

    let output = if let Some(separator) = separator {
        input
            .split(&separator)
            .map(capitalize_word)
            .collect::<Vec<_>>()
            .join(&separator)
    } else {
        input
            .split_whitespace()
            .map(capitalize_word)
            .collect::<Vec<_>>()
            .join(" ")
    };
    Ok(owned_string_value(output))
}

fn capitalize_word(word: &str) -> String {
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return String::new();
    };
    let mut out = String::new();
    out.extend(first.to_uppercase());
    out.extend(chars.flat_map(char::to_lowercase));
    out
}

fn formatter_init(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "Formatter() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(Value::none())
}

fn formatter_parse(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "parse() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    super::_string::formatter_parser_builtin(&[args[1]])
}

fn formatter_format(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if args.len() < 2 {
        return Err(BuiltinError::TypeError(
            "format() missing required argument 'format_string'".to_string(),
        ));
    }
    super::_string::builtin_str_format_method(vm, &args[1..], keywords)
}

fn formatter_vformat(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 4 {
        return Err(BuiltinError::TypeError(format!(
            "vformat() takes exactly 3 arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let positional = sequence_values(args[2], "args")?;
    let keyword_storage = dict_keyword_values(args[3], "kwargs")?;
    let keyword_refs = keyword_storage
        .iter()
        .map(|(name, value)| (name.as_str(), *value))
        .collect::<Vec<_>>();
    let mut format_args = Vec::with_capacity(positional.len() + 1);
    format_args.push(args[1]);
    format_args.extend(positional);
    super::_string::builtin_str_format_method(vm, &format_args, &keyword_refs)
}

fn sequence_values(value: Value, context: &str) -> Result<Vec<Value>, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a sequence")))?;
    if let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
        return Ok(tuple.iter().copied().collect());
    }
    if let Some(list) = list_storage_ref_from_ptr(ptr) {
        return Ok(list.iter().copied().collect());
    }
    Err(BuiltinError::TypeError(format!(
        "{context} must be a sequence"
    )))
}

fn dict_keyword_values(value: Value, context: &str) -> Result<Vec<(String, Value)>, BuiltinError> {
    let ptr = value
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))?;
    let dict = dict_storage_ref_from_ptr(ptr)
        .ok_or_else(|| BuiltinError::TypeError(format!("{context} must be a dict")))?;
    let mut values = Vec::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        values.push((value_to_string(key, "keyword key")?, value));
    }
    Ok(values)
}

fn template_init(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "Template() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let ptr = expect_template_instance(args[0], "__init__")?;
    let template = value_to_string(args[1], "template")?;
    let instance = unsafe { &mut *(ptr as *mut ShapedObject) };
    instance.set_property(
        intern("template"),
        owned_string_value(template),
        shape_registry(),
    );
    Ok(Value::none())
}

fn template_substitute(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    template_substitute_impl(vm, args, keywords, SubstituteMode::Strict)
}

fn template_safe_substitute(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    template_substitute_impl(vm, args, keywords, SubstituteMode::Safe)
}

fn template_substitute_impl(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
    mode: SubstituteMode,
) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "substitute() takes from 1 to 2 positional arguments ({} given)",
            args.len()
        )));
    }

    let template = template_text(args[0])?;
    let delimiter = template_delimiter(vm, args[0])?;
    let mapping = args.get(1).copied();
    let matches = scan_template(&template, &delimiter)?;
    let mut rendered = String::with_capacity(template.len());
    let mut cursor = 0;

    for match_data in matches {
        rendered.push_str(&template[cursor..match_data.start]);
        cursor = match_data.end;

        if match_data.escaped.is_some() {
            rendered.push_str(&delimiter);
            continue;
        }

        if match_data.invalid.is_some() {
            if mode == SubstituteMode::Safe {
                rendered.push_str(&match_data.full);
                continue;
            }
            return Err(invalid_placeholder_error(&template, &match_data));
        }

        let name = match_data
            .named
            .as_deref()
            .or(match_data.braced.as_deref())
            .expect("valid template match should have a name");
        match lookup_template_value(vm, mapping, keywords, name) {
            Ok(value) => rendered.push_str(&stringify_template_value(vm, value)?),
            Err(err) if mode == SubstituteMode::Safe && is_builtin_key_error(&err) => {
                rendered.push_str(&match_data.full);
            }
            Err(err) => return Err(err),
        }
    }

    rendered.push_str(&template[cursor..]);
    Ok(owned_string_value(rendered))
}

fn template_is_valid(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "is_valid() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    let template = template_text(args[0])?;
    let delimiter = template_delimiter(vm, args[0])?;
    Ok(Value::bool(
        scan_template(&template, &delimiter)?
            .iter()
            .all(|match_data| match_data.invalid.is_none()),
    ))
}

fn template_get_identifiers(
    vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "get_identifiers() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }
    let template = template_text(args[0])?;
    let delimiter = template_delimiter(vm, args[0])?;
    let mut names = Vec::<String>::new();
    for match_data in scan_template(&template, &delimiter)? {
        if match_data.invalid.is_some() {
            continue;
        }
        let Some(name) = match_data.named.or(match_data.braced) else {
            continue;
        };
        if !names.iter().any(|existing| existing == &name) {
            names.push(name);
        }
    }
    Ok(list_value(
        names
            .into_iter()
            .map(|name| Value::string(intern(&name)))
            .collect(),
    ))
}

fn template_pattern_finditer(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "finditer() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let template = value_to_string(args[1], "string")?;
    let values = scan_template(&template, TEMPLATE_DELIMITER)?
        .into_iter()
        .map(template_match_value)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(list_value(values))
}

fn template_match_groupdict(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "groupdict() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let ptr = expect_template_match_instance(args[0], "groupdict")?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    let mut dict = DictObject::with_capacity(4);
    for name in ["escaped", "named", "braced", "invalid"] {
        let value = shaped.get_property(name).unwrap_or_else(Value::none);
        dict.set(Value::string(intern(name)), value);
    }
    Ok(crate::alloc_managed_value(dict))
}

fn template_match_group(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "group() takes at most 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let ptr = expect_template_match_instance(args[0], "group")?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    let selector = args
        .get(1)
        .copied()
        .unwrap_or_else(|| Value::int_unchecked(0));

    if value_to_i64(selector) == Some(0) {
        return shaped.get_property("full").ok_or_else(|| {
            BuiltinError::AttributeError("match object has no full group".to_string())
        });
    }

    let name = value_to_string(selector, "group name")?;
    match name.as_str() {
        "escaped" | "named" | "braced" | "invalid" => {
            Ok(shaped.get_property(&name).unwrap_or_else(Value::none))
        }
        _ => Err(BuiltinError::IndexError("no such group".to_string())),
    }
}

fn template_match_value(match_data: TemplateMatchData) -> Result<Value, BuiltinError> {
    let mut instance = allocate_heap_instance_for_class(TEMPLATE_MATCH_CLASS.as_ref());
    set_match_property(&mut instance, "full", Some(match_data.full));
    set_match_property(&mut instance, "escaped", match_data.escaped);
    set_match_property(&mut instance, "named", match_data.named);
    set_match_property(&mut instance, "braced", match_data.braced);
    set_match_property(&mut instance, "invalid", match_data.invalid);
    Ok(crate::alloc_managed_value(instance))
}

fn set_match_property(instance: &mut ShapedObject, name: &str, value: Option<String>) {
    let value = value
        .map(|value| Value::string(intern(&value)))
        .unwrap_or_else(Value::none);
    instance.set_property(intern(name), value, shape_registry());
}

fn template_text(value: Value) -> Result<String, BuiltinError> {
    let ptr = expect_template_instance(value, "template")?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    let template = shaped.get_property("template").ok_or_else(|| {
        BuiltinError::AttributeError("Template instance has no 'template' attribute".to_string())
    })?;
    value_to_string(template, "template")
}

fn template_delimiter(vm: &mut VirtualMachine, receiver: Value) -> Result<String, BuiltinError> {
    let delimiter = get_attribute_value(vm, receiver, &intern("delimiter"))
        .map_err(runtime_error_to_builtin_error)?;
    let delimiter = value_to_string(delimiter, "delimiter")?;
    if delimiter.is_empty() {
        return Err(BuiltinError::ValueError(
            "delimiter must not be empty".to_string(),
        ));
    }
    Ok(delimiter)
}

fn expect_template_instance(value: Value, method: &str) -> Result<*const (), BuiltinError> {
    expect_instance_of(
        value,
        TEMPLATE_CLASS.class_type_id(),
        "string.Template",
        method,
    )
}

fn expect_template_match_instance(value: Value, method: &str) -> Result<*const (), BuiltinError> {
    expect_instance_of(
        value,
        TEMPLATE_MATCH_CLASS.class_type_id(),
        "string._TemplateMatch",
        method,
    )
}

fn expect_instance_of(
    value: Value,
    class_type_id: TypeId,
    class_name: &str,
    method: &str,
) -> Result<*const (), BuiltinError> {
    let ptr = value.as_object_ptr().ok_or_else(|| {
        BuiltinError::TypeError(format!(
            "descriptor '{method}' requires a {class_name} object"
        ))
    })?;
    let type_id = extract_type_id(ptr);
    let is_instance = type_id == class_type_id
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(class_type_id)));
    if is_instance {
        Ok(ptr)
    } else {
        Err(BuiltinError::TypeError(format!(
            "descriptor '{method}' requires a {class_name} object"
        )))
    }
}

fn scan_template(template: &str, delimiter: &str) -> Result<Vec<TemplateMatchData>, BuiltinError> {
    let mut matches = Vec::new();
    let mut cursor = 0;

    while cursor <= template.len() {
        let Some(relative) = template[cursor..].find(delimiter) else {
            break;
        };
        let start = cursor + relative;
        let after_delimiter = start + delimiter.len();

        if template[after_delimiter..].starts_with(delimiter) {
            let end = after_delimiter + delimiter.len();
            matches.push(TemplateMatchData {
                start,
                end,
                full: template[start..end].to_string(),
                named: None,
                braced: None,
                escaped: Some(delimiter.to_string()),
                invalid: None,
                invalid_index: None,
            });
            cursor = end;
            continue;
        }

        if template[after_delimiter..].starts_with('{') {
            let name_start = after_delimiter + 1;
            match template[name_start..].find('}') {
                Some(relative_end) => {
                    let name_end = name_start + relative_end;
                    let end = name_end + 1;
                    let name = &template[name_start..name_end];
                    if is_template_identifier(name) {
                        matches.push(TemplateMatchData {
                            start,
                            end,
                            full: template[start..end].to_string(),
                            named: None,
                            braced: Some(name.to_string()),
                            escaped: None,
                            invalid: None,
                            invalid_index: None,
                        });
                    } else {
                        matches.push(invalid_template_match(
                            template,
                            start,
                            end,
                            after_delimiter,
                        ));
                    }
                    cursor = end;
                }
                None => {
                    matches.push(invalid_template_match(
                        template,
                        start,
                        template.len(),
                        after_delimiter,
                    ));
                    break;
                }
            }
            continue;
        }

        let name_end = identifier_end(template, after_delimiter);
        if name_end > after_delimiter {
            matches.push(TemplateMatchData {
                start,
                end: name_end,
                full: template[start..name_end].to_string(),
                named: Some(template[after_delimiter..name_end].to_string()),
                braced: None,
                escaped: None,
                invalid: None,
                invalid_index: None,
            });
            cursor = name_end;
        } else {
            matches.push(invalid_template_match(
                template,
                start,
                after_delimiter,
                after_delimiter,
            ));
            cursor = after_delimiter;
        }
    }

    Ok(matches)
}

fn invalid_template_match(
    template: &str,
    start: usize,
    end: usize,
    invalid_index: usize,
) -> TemplateMatchData {
    TemplateMatchData {
        start,
        end,
        full: template[start..end].to_string(),
        named: None,
        braced: None,
        escaped: None,
        invalid: Some(String::new()),
        invalid_index: Some(invalid_index),
    }
}

fn identifier_end(input: &str, start: usize) -> usize {
    let mut iter = input[start..].char_indices();
    let Some((_, first)) = iter.next() else {
        return start;
    };
    if !is_template_identifier_start(first) {
        return start;
    }

    let mut end = start + first.len_utf8();
    for (offset, ch) in iter {
        if !is_template_identifier_continue(ch) {
            break;
        }
        end = start + offset + ch.len_utf8();
    }
    end
}

fn is_template_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    is_template_identifier_start(first) && chars.all(is_template_identifier_continue)
}

#[inline]
fn is_template_identifier_start(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphabetic()
}

#[inline]
fn is_template_identifier_continue(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

fn invalid_placeholder_error(template: &str, match_data: &TemplateMatchData) -> BuiltinError {
    let invalid_index = match_data.invalid_index.unwrap_or(match_data.start);
    let prefix = &template[..invalid_index.min(template.len())];
    let line = prefix.bytes().filter(|byte| *byte == b'\n').count() + 1;
    let column = prefix
        .rsplit_once('\n')
        .map(|(_, tail)| tail.chars().count() + 1)
        .unwrap_or_else(|| prefix.chars().count() + 1);
    BuiltinError::ValueError(format!(
        "Invalid placeholder in string: line {line}, col {column}"
    ))
}

fn lookup_template_value(
    vm: &mut VirtualMachine,
    mapping: Option<Value>,
    keywords: &[(&str, Value)],
    name: &str,
) -> Result<Value, BuiltinError> {
    if let Some(value) = keywords
        .iter()
        .find_map(|(keyword, value)| (*keyword == name).then_some(*value))
    {
        return Ok(value);
    }

    let Some(mapping) = mapping else {
        return Err(BuiltinError::KeyError(name.to_string()));
    };
    mapping_get_item(vm, mapping, Value::string(intern(name)))
}

fn mapping_get_item(
    vm: &mut VirtualMachine,
    mapping: Value,
    key: Value,
) -> Result<Value, BuiltinError> {
    if let Some(ptr) = mapping.as_object_ptr()
        && let Some(dict) = dict_storage_ref_from_ptr(ptr)
    {
        return dict_get_item(vm, dict, key)
            .map_err(runtime_error_to_builtin_error)?
            .ok_or_else(|| {
                BuiltinError::KeyError(value_to_string(key, "key").unwrap_or_default())
            });
    }

    let getitem = get_attribute_value(vm, mapping, &intern("__getitem__"))
        .map_err(runtime_error_to_builtin_error)?;
    invoke_callable_value(vm, getitem, &[key]).map_err(runtime_error_to_builtin_error)
}

fn stringify_template_value(vm: &mut VirtualMachine, value: Value) -> Result<String, BuiltinError> {
    let string = builtin_str_vm(vm, &[value])?;
    value_to_string(string, "substitution")
}

#[inline]
fn is_builtin_key_error(err: &BuiltinError) -> bool {
    matches!(err, BuiltinError::KeyError(_))
}
