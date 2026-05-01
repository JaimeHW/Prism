//! Native `traceback` module subset.
//!
//! Traceback collection stays on Prism's native frame/traceback objects. Source
//! lines and column spans are resolved lazily by `extract_tb`, so normal
//! exception creation remains allocation-light and does not touch the filesystem.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, ExceptionTypeObject, ExceptionValue, builtin_str_vm,
    exception_display_text_for_value, runtime_error_to_builtin_error,
};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{extract_type_id, get_attribute_value};
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
use prism_runtime::object::views::{FrameViewObject, TracebackViewObject};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use std::sync::{Arc, LazyLock};

static EXTRACT_TB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.extract_tb"), extract_tb));
static CLEAR_FRAMES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.clear_frames"), clear_frames));
static FORMAT_TB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.format_tb"), format_tb));
static FORMAT_EXCEPTION_ONLY_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.format_exception_only"),
        format_exception_only,
    )
});
static FORMAT_EXCEPTION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.format_exception"), format_exception)
});
static FORMAT_EXC_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.format_exc"), format_exc)
});
static PRINT_EXCEPTION_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.print_exception"), print_exception)
});
static TRACEBACK_EXCEPTION_CLASS: LazyLock<Arc<PyClassObject>> =
    LazyLock::new(build_traceback_exception_class);
static TRACEBACK_EXCEPTION_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.TracebackException.__init__"),
        traceback_exception_init,
    )
});
static TRACEBACK_EXCEPTION_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.TracebackException.format"),
        traceback_exception_format,
    )
});
static TRACEBACK_EXCEPTION_FORMAT_ONLY_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| {
        BuiltinFunctionObject::new_vm_kw(
            Arc::from("traceback.TracebackException.format_exception_only"),
            traceback_exception_format_exception_only,
        )
    });
static TRACEBACK_EXCEPTION_STR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("traceback.TracebackException.__str__"),
        traceback_exception_str,
    )
});

/// Native `traceback` module descriptor.
#[derive(Debug, Clone)]
pub struct TracebackModule {
    attrs: Vec<Arc<str>>,
}

impl TracebackModule {
    /// Create a new `traceback` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![
                Arc::from("TracebackException"),
                Arc::from("clear_frames"),
                Arc::from("extract_tb"),
                Arc::from("format_exc"),
                Arc::from("format_exception"),
                Arc::from("format_exception_only"),
                Arc::from("format_tb"),
                Arc::from("print_exception"),
            ],
        }
    }
}

impl Default for TracebackModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for TracebackModule {
    fn name(&self) -> &str {
        "traceback"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "TracebackException" => Ok(traceback_exception_class_value()),
            "clear_frames" => Ok(builtin_value(&CLEAR_FRAMES_FUNCTION)),
            "extract_tb" => Ok(builtin_value(&EXTRACT_TB_FUNCTION)),
            "format_exc" => Ok(builtin_value(&FORMAT_EXC_FUNCTION)),
            "format_exception" => Ok(builtin_value(&FORMAT_EXCEPTION_FUNCTION)),
            "format_exception_only" => Ok(builtin_value(&FORMAT_EXCEPTION_ONLY_FUNCTION)),
            "format_tb" => Ok(builtin_value(&FORMAT_TB_FUNCTION)),
            "print_exception" => Ok(builtin_value(&PRINT_EXCEPTION_FUNCTION)),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'traceback' has no attribute '{}'",
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
fn traceback_exception_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&TRACEBACK_EXCEPTION_CLASS) as *const ())
}

fn build_traceback_exception_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("TracebackException"));
    class.set_attr(intern("__module__"), Value::string(intern("traceback")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("TracebackException")),
    );
    class.set_attr(
        intern("__doc__"),
        Value::string(intern("Captured exception state ready for formatting.")),
    );
    class.set_attr(
        intern("__init__"),
        builtin_value(&TRACEBACK_EXCEPTION_INIT_METHOD),
    );
    class.set_attr(
        intern("format"),
        builtin_value(&TRACEBACK_EXCEPTION_FORMAT_METHOD),
    );
    class.set_attr(
        intern("format_exception_only"),
        builtin_value(&TRACEBACK_EXCEPTION_FORMAT_ONLY_METHOD),
    );
    class.set_attr(
        intern("__str__"),
        builtin_value(&TRACEBACK_EXCEPTION_STR_METHOD),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    let mut bitmap = SubclassBitmap::new();
    for &class_id in class.mro() {
        bitmap.set_bit(class_id_to_type_id(class_id));
    }
    let class = Arc::new(class);
    register_global_class(Arc::clone(&class), bitmap);
    class
}

fn extract_tb(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "extract_tb() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    if args[0].is_none() {
        return Ok(list_value(Vec::new()));
    }

    let limit = parse_limit(args.get(1).copied())?;
    if matches!(limit, TracebackLimit::First(0) | TracebackLimit::Last(0)) {
        return Ok(list_value(Vec::new()));
    }

    let mut summaries = Vec::new();
    let mut cursor = Some(args[0]);
    let mut seen = Vec::new();

    while let Some(traceback_value) = cursor {
        let Some(ptr) = traceback_value.as_object_ptr() else {
            return Err(traceback_type_error());
        };
        if extract_type_id(ptr) != TypeId::TRACEBACK {
            return Err(traceback_type_error());
        }
        if seen.contains(&ptr) {
            break;
        }
        seen.push(ptr);

        let traceback = unsafe { &*(ptr as *const TracebackViewObject) };
        summaries.push(frame_summary_value(traceback)?);

        if limit.reached_first(summaries.len()) {
            break;
        }

        cursor = traceback.next().filter(|value| !value.is_none());
    }

    if let TracebackLimit::Last(count) = limit {
        let len = summaries.len();
        if count < len {
            summaries.drain(0..len - count);
        }
    }

    Ok(list_value(summaries))
}

fn clear_frames(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "clear_frames() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    validate_traceback_chain(args[0])?;
    Ok(Value::none())
}

fn format_tb(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "format_tb() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let extracted = extract_tb(args)?;
    let lines = format_stack_value(extracted)?;
    Ok(string_list_value(lines))
}

fn format_exception_only(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if !keywords.is_empty() {
        return Err(unexpected_keyword("format_exception_only", keywords[0].0));
    }
    let (exc_type, exc_value) = match args {
        [exc] => (exception_class_value(vm, *exc)?, *exc),
        [exc_type, exc_value] => (*exc_type, *exc_value),
        _ => {
            return Err(BuiltinError::TypeError(format!(
                "format_exception_only() takes 1 or 2 positional arguments but {} were given",
                args.len()
            )));
        }
    };

    let message = exception_message_text(vm, exc_value)?;
    Ok(string_list_value(vec![format_exception_only_line(
        exc_type, exc_value, &message,
    )]))
}

fn format_exception(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_format_exception_args(vm, args, keywords, "format_exception")?;
    let lines = format_exception_values(
        vm,
        bound.exc_type,
        bound.exc_value,
        bound.exc_traceback,
        bound.limit,
    )?;
    Ok(string_list_value(lines))
}

fn format_exc(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_format_exc_args(args, keywords)?;
    let exc_info = crate::ops::exception::build_exc_info(vm);
    let (exc_type, exc_value, exc_traceback) = exc_info.to_tuple();
    let lines = format_exception_values(vm, exc_type, exc_value, exc_traceback, bound.limit)?;
    Ok(owned_string_value(lines.concat()))
}

fn print_exception(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_print_exception_args(vm, args, keywords)?;
    let lines = format_exception_values(
        vm,
        bound.exc_type,
        bound.exc_value,
        bound.exc_traceback,
        bound.limit,
    )?;
    write_traceback_lines(vm, bound.file, &lines)?;
    Ok(Value::none())
}

fn traceback_exception_init(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_traceback_exception_init_args(args, keywords)?;
    let object = traceback_exception_object_mut(bound.self_value, "__init__")?;
    let message = exception_message_text(vm, bound.exc_value)?;
    let stack = stack_value(bound.exc_traceback, bound.limit)?;

    set_attr(object, "max_group_width", bound.max_group_width);
    set_attr(object, "max_group_depth", bound.max_group_depth);
    set_attr(object, "stack", stack);
    set_attr(object, "exc_type", bound.exc_type);
    set_attr(object, "_str", owned_string_value(message));
    set_attr(object, "__notes__", Value::none());
    set_attr(object, "__cause__", Value::none());
    set_attr(object, "__context__", Value::none());
    set_attr(object, "exceptions", Value::none());
    set_attr(object, "__suppress_context__", Value::bool(false));

    Ok(Value::none())
}

fn traceback_exception_format(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    expect_traceback_exception_self_arg("format", args)?;
    parse_chain_keyword("TracebackException.format", keywords)?;
    let object = traceback_exception_object(args[0], "format")?;
    let lines = traceback_exception_lines(vm, object)?;
    Ok(string_list_value(lines))
}

fn traceback_exception_format_exception_only(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    expect_traceback_exception_self_arg("format_exception_only", args)?;
    parse_chain_keyword("TracebackException.format_exception_only", keywords)?;
    let object = traceback_exception_object(args[0], "format_exception_only")?;
    let exc_type = object.get_property("exc_type").unwrap_or_else(Value::none);
    let message = traceback_exception_message(object);
    Ok(string_list_value(vec![format_exception_only_line(
        exc_type,
        Value::none(),
        &message,
    )]))
}

fn traceback_exception_str(
    _vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    expect_traceback_exception_self_arg("__str__", args)?;
    let object = traceback_exception_object(args[0], "__str__")?;
    Ok(owned_string_value(traceback_exception_message(object)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TracebackLimit {
    All,
    First(usize),
    Last(usize),
}

impl TracebackLimit {
    #[inline]
    fn reached_first(self, len: usize) -> bool {
        matches!(self, Self::First(limit) if len >= limit)
    }
}

#[derive(Clone, Copy)]
struct FormatExceptionArgs {
    exc_type: Value,
    exc_value: Value,
    exc_traceback: Value,
    limit: Option<Value>,
}

#[derive(Clone, Copy)]
struct FormatExcArgs {
    limit: Option<Value>,
}

#[derive(Clone, Copy)]
struct PrintExceptionArgs {
    exc_type: Value,
    exc_value: Value,
    exc_traceback: Value,
    limit: Option<Value>,
    file: Option<Value>,
}

#[derive(Clone, Copy)]
struct TracebackExceptionInitArgs {
    self_value: Value,
    exc_type: Value,
    exc_value: Value,
    exc_traceback: Value,
    limit: Option<Value>,
    max_group_width: Value,
    max_group_depth: Value,
}

fn parse_limit(value: Option<Value>) -> Result<TracebackLimit, BuiltinError> {
    let Some(value) = value else {
        return Ok(TracebackLimit::All);
    };
    if value.is_none() {
        return Ok(TracebackLimit::All);
    }
    let Some(limit) = value.as_int() else {
        return Err(BuiltinError::TypeError(
            "limit must be an integer or None".to_string(),
        ));
    };
    if limit >= 0 {
        Ok(TracebackLimit::First(limit as usize))
    } else {
        Ok(TracebackLimit::Last(limit.unsigned_abs() as usize))
    }
}

fn bind_traceback_exception_init_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<TracebackExceptionInitArgs, BuiltinError> {
    if args.len() < 4 || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "TracebackException() takes 3 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let mut limit = args.get(4).copied();
    let mut max_group_width = int_value(15);
    let mut max_group_depth = int_value(10);

    for &(name, value) in keywords {
        match name {
            "limit" => assign_optional_keyword(
                &mut limit,
                value,
                "TracebackException",
                "limit",
                args.len() > 4,
            )?,
            "lookup_lines" | "capture_locals" | "compact" | "_seen" => {}
            "max_group_width" => max_group_width = value,
            "max_group_depth" => max_group_depth = value,
            other => return Err(unexpected_keyword("TracebackException", other)),
        }
    }

    Ok(TracebackExceptionInitArgs {
        self_value: args[0],
        exc_type: args[1],
        exc_value: args[2],
        exc_traceback: args[3],
        limit,
        max_group_width,
        max_group_depth,
    })
}

fn bind_format_exception_args(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
    function_name: &'static str,
) -> Result<FormatExceptionArgs, BuiltinError> {
    if args.is_empty() || args.len() == 2 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "{function_name}() takes 1 or 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut limit = args.get(3).copied();
    for &(name, value) in keywords {
        match name {
            "limit" => {
                assign_optional_keyword(&mut limit, value, function_name, "limit", args.len() > 3)?
            }
            "chain" => {}
            other => return Err(unexpected_keyword(function_name, other)),
        }
    }

    if args.len() == 1 {
        let exc_value = args[0];
        return Ok(FormatExceptionArgs {
            exc_type: exception_class_value(vm, exc_value)?,
            exc_value,
            exc_traceback: exception_traceback_value(vm, exc_value)?,
            limit,
        });
    }

    Ok(FormatExceptionArgs {
        exc_type: args[0],
        exc_value: args[1],
        exc_traceback: args[2],
        limit,
    })
}

fn bind_format_exc_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<FormatExcArgs, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "format_exc() takes at most 2 arguments ({} given)",
            args.len()
        )));
    }

    let mut limit = args.first().copied();
    for &(name, value) in keywords {
        match name {
            "limit" => {
                assign_optional_keyword(&mut limit, value, "format_exc", "limit", !args.is_empty())?
            }
            "chain" => {}
            other => return Err(unexpected_keyword("format_exc", other)),
        }
    }

    Ok(FormatExcArgs { limit })
}

fn bind_print_exception_args(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<PrintExceptionArgs, BuiltinError> {
    if args.is_empty() || args.len() == 2 || args.len() > 5 {
        return Err(BuiltinError::TypeError(format!(
            "print_exception() takes 1 or 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut limit = args.get(3).copied();
    let mut file = args.get(4).copied();
    for &(name, value) in keywords {
        match name {
            "limit" => assign_optional_keyword(
                &mut limit,
                value,
                "print_exception",
                "limit",
                args.len() > 3,
            )?,
            "file" => assign_optional_keyword(
                &mut file,
                value,
                "print_exception",
                "file",
                args.len() > 4,
            )?,
            "chain" => {}
            other => return Err(unexpected_keyword("print_exception", other)),
        }
    }

    if args.len() == 1 {
        let exc_value = args[0];
        return Ok(PrintExceptionArgs {
            exc_type: exception_class_value(vm, exc_value)?,
            exc_value,
            exc_traceback: exception_traceback_value(vm, exc_value)?,
            limit,
            file,
        });
    }

    Ok(PrintExceptionArgs {
        exc_type: args[0],
        exc_value: args[1],
        exc_traceback: args[2],
        limit,
        file,
    })
}

fn assign_optional_keyword(
    slot: &mut Option<Value>,
    value: Value,
    function_name: &'static str,
    keyword: &'static str,
    duplicate_from_positional: bool,
) -> Result<(), BuiltinError> {
    if duplicate_from_positional || slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "{function_name}() got multiple values for argument '{keyword}'"
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn unexpected_keyword(function_name: &str, keyword: &str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "{function_name}() got an unexpected keyword argument '{keyword}'"
    ))
}

fn parse_chain_keyword(
    function_name: &str,
    keywords: &[(&str, Value)],
) -> Result<(), BuiltinError> {
    for &(name, _) in keywords {
        if name != "chain" {
            return Err(unexpected_keyword(function_name, name));
        }
    }
    Ok(())
}

fn stack_value(traceback: Value, limit: Option<Value>) -> Result<Value, BuiltinError> {
    if let Some(limit) = limit {
        extract_tb(&[traceback, limit])
    } else {
        extract_tb(&[traceback])
    }
}

fn format_exception_values(
    vm: &mut VirtualMachine,
    exc_type: Value,
    exc_value: Value,
    exc_traceback: Value,
    limit: Option<Value>,
) -> Result<Vec<String>, BuiltinError> {
    let mut lines = format_stack_value(stack_value(exc_traceback, limit)?)?;
    let message = exception_message_text(vm, exc_value)?;
    lines.push(format_exception_only_line(exc_type, exc_value, &message));
    Ok(lines)
}

fn write_traceback_lines(
    vm: &mut VirtualMachine,
    file: Option<Value>,
    lines: &[String],
) -> Result<(), BuiltinError> {
    let Some(file) = file.filter(|value| !value.is_none()) else {
        eprint!("{}", lines.concat());
        return Ok(());
    };

    let write =
        get_attribute_value(vm, file, &intern("write")).map_err(runtime_error_to_builtin_error)?;
    for line in lines {
        invoke_callable_value(vm, write, &[owned_string_value(line.clone())])
            .map_err(runtime_error_to_builtin_error)?;
    }
    Ok(())
}

fn traceback_exception_lines(
    _vm: &mut VirtualMachine,
    object: &'static ShapedObject,
) -> Result<Vec<String>, BuiltinError> {
    let mut lines = format_stack_value(
        object
            .get_property("stack")
            .unwrap_or_else(|| list_value(Vec::new())),
    )?;
    let exc_type = object.get_property("exc_type").unwrap_or_else(Value::none);
    let message = traceback_exception_message(object);
    lines.push(format_exception_only_line(
        exc_type,
        Value::none(),
        &message,
    ));
    Ok(lines)
}

fn format_stack_value(stack: Value) -> Result<Vec<String>, BuiltinError> {
    let stack = list_ref(stack)?;
    let mut lines = Vec::with_capacity(stack.len().saturating_mul(2).saturating_add(1));
    if stack.is_empty() {
        return Ok(lines);
    }

    lines.push("Traceback (most recent call last):\n".to_string());
    for frame in stack.iter().copied() {
        lines.extend(format_frame_summary(frame)?);
    }
    Ok(lines)
}

fn format_frame_summary(frame: Value) -> Result<Vec<String>, BuiltinError> {
    let frame = frame_summary_ref(frame)?;
    let filename = string_attr(frame, "filename").unwrap_or_else(|| "<unknown>".to_string());
    let name = string_attr(frame, "name").unwrap_or_else(|| "<unknown>".to_string());
    let line_number = int_attr(frame, "lineno").unwrap_or(0);
    let source = string_attr(frame, "line").unwrap_or_default();

    let mut lines = vec![format!(
        "  File \"{}\", line {}, in {}\n",
        filename, line_number, name
    )];
    if !source.is_empty() {
        lines.push(format!("    {}\n", source.trim()));
    }
    Ok(lines)
}

fn format_exception_only_line(exc_type: Value, exc_value: Value, message: &str) -> String {
    let type_name = exception_type_name(exc_type, exc_value);
    if message.is_empty() {
        format!("{type_name}\n")
    } else {
        format!("{type_name}: {message}\n")
    }
}

fn exception_message_text(
    vm: &mut VirtualMachine,
    exc_value: Value,
) -> Result<String, BuiltinError> {
    if exc_value.is_none() {
        return Ok("None".to_string());
    }
    if let Some(text) = exception_display_text_for_value(exc_value) {
        return Ok(text);
    }

    let rendered = builtin_str_vm(vm, &[exc_value])?;
    let Some(text) = value_as_string_ref(rendered) else {
        return Err(BuiltinError::TypeError(
            "str() returned non-string".to_string(),
        ));
    };
    Ok(text.as_str().to_string())
}

fn exception_type_name(exc_type: Value, exc_value: Value) -> String {
    if let Some(exception) = unsafe { ExceptionValue::from_value(exc_value) } {
        return exception.type_name().to_string();
    }

    if exc_type.is_none() {
        return "NoneType".to_string();
    }

    let Some(ptr) = exc_type.as_object_ptr() else {
        return exc_type.type_name().to_string();
    };
    match extract_type_id(ptr) {
        TypeId::EXCEPTION_TYPE => {
            let exception_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
            exception_type.name().to_string()
        }
        TypeId::TYPE => class_type_name(ptr),
        type_id => type_id.name().to_string(),
    }
}

fn class_type_name(ptr: *const ()) -> String {
    if let Some(type_id) = crate::builtins::builtin_type_object_type_id(ptr) {
        return type_id.name().to_string();
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    class.name().as_str().to_string()
}

fn exception_class_value(vm: &mut VirtualMachine, exc_value: Value) -> Result<Value, BuiltinError> {
    if exc_value.is_none() {
        return Ok(Value::none());
    }
    get_attribute_value(vm, exc_value, &intern("__class__")).map_err(runtime_error_to_builtin_error)
}

fn exception_traceback_value(
    vm: &mut VirtualMachine,
    exc_value: Value,
) -> Result<Value, BuiltinError> {
    if exc_value.is_none() {
        return Ok(Value::none());
    }
    get_attribute_value(vm, exc_value, &intern("__traceback__"))
        .map_err(runtime_error_to_builtin_error)
}

fn validate_traceback_chain(traceback: Value) -> Result<(), BuiltinError> {
    if traceback.is_none() {
        return Ok(());
    }

    let mut cursor = Some(traceback);
    let mut seen = Vec::new();
    while let Some(value) = cursor {
        if value.is_none() {
            return Ok(());
        }
        let Some(ptr) = value.as_object_ptr() else {
            return Err(traceback_type_error());
        };
        if extract_type_id(ptr) != TypeId::TRACEBACK {
            return Err(traceback_type_error());
        }
        if seen.contains(&ptr) {
            return Ok(());
        }
        seen.push(ptr);
        let traceback = unsafe { &*(ptr as *const TracebackViewObject) };
        cursor = traceback.next().filter(|value| !value.is_none());
    }
    Ok(())
}

fn traceback_exception_message(object: &'static ShapedObject) -> String {
    object
        .get_property("_str")
        .and_then(string_from_value)
        .unwrap_or_default()
}

fn expect_traceback_exception_self_arg(
    method: &'static str,
    args: &[Value],
) -> Result<(), BuiltinError> {
    if args.len() == 1 {
        Ok(())
    } else {
        Err(BuiltinError::TypeError(format!(
            "TracebackException.{method}() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )))
    }
}

fn traceback_exception_object(
    value: Value,
    context: &'static str,
) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(traceback_exception_type_error(context));
    };
    if !is_traceback_exception_type(extract_type_id(ptr)) {
        return Err(traceback_exception_type_error(context));
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn traceback_exception_object_mut(
    value: Value,
    context: &'static str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(traceback_exception_type_error(context));
    };
    if !is_traceback_exception_type(extract_type_id(ptr)) {
        return Err(traceback_exception_type_error(context));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn traceback_exception_type_error(context: &'static str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "TracebackException.{context} requires a TracebackException object"
    ))
}

#[inline]
fn is_traceback_exception_type(type_id: TypeId) -> bool {
    type_id == traceback_exception_type_id()
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(traceback_exception_type_id())))
}

#[inline]
fn traceback_exception_type_id() -> TypeId {
    TRACEBACK_EXCEPTION_CLASS.class_type_id()
}

fn list_ref(value: Value) -> Result<&'static ListObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError("expected a list".to_string()));
    };
    if extract_type_id(ptr) != TypeId::LIST {
        return Err(BuiltinError::TypeError("expected a list".to_string()));
    }
    Ok(unsafe { &*(ptr as *const ListObject) })
}

fn frame_summary_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "traceback frame summary is not an object".to_string(),
        ));
    };
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn string_attr(object: &'static ShapedObject, name: &str) -> Option<String> {
    object.get_property(name).and_then(string_from_value)
}

fn int_attr(object: &'static ShapedObject, name: &str) -> Option<i64> {
    object.get_property(name).and_then(|value| value.as_int())
}

fn string_from_value(value: Value) -> Option<String> {
    value_as_string_ref(value).map(|text| text.as_str().to_string())
}

fn frame_summary_value(traceback: &TracebackViewObject) -> Result<Value, BuiltinError> {
    let line_number = traceback.line_number();
    let lasti = traceback.lasti();
    let frame = frame_view(traceback.frame())?;
    let code = frame.code();
    let filename = code.map_or("<unknown>", |code| code.filename.as_ref());
    let name = code.map_or("<unknown>", |code| code.name.as_ref());
    let source = source_line(filename, line_number);
    let (colno, end_colno) = source
        .as_ref()
        .and_then(|line| expression_columns(line.raw.as_str()))
        .unwrap_or_else(|| fallback_columns(source.as_ref().map(|line| line.raw.as_str())));

    let mut summary = ShapedObject::new(TypeId::OBJECT, shape_registry().empty_shape());
    set_attr(&mut summary, "filename", string_value(filename));
    set_attr(&mut summary, "lineno", int_value(line_number as i64));
    set_attr(&mut summary, "end_lineno", int_value(line_number as i64));
    set_attr(&mut summary, "name", string_value(name));
    set_attr(
        &mut summary,
        "line",
        owned_string_value(source.map_or_else(String::new, |line| line.trimmed)),
    );
    set_attr(&mut summary, "colno", int_value(colno as i64));
    set_attr(&mut summary, "end_colno", int_value(end_colno as i64));
    set_attr(&mut summary, "lasti", int_value(lasti as i64));

    Ok(crate::alloc_managed_value(summary))
}

fn frame_view(value: Value) -> Result<&'static FrameViewObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "traceback frame is not a frame object".to_string(),
        ));
    };
    if extract_type_id(ptr) != TypeId::FRAME {
        return Err(BuiltinError::TypeError(
            "traceback frame is not a frame object".to_string(),
        ));
    }
    Ok(unsafe { &*(ptr as *const FrameViewObject) })
}

#[derive(Debug, Clone)]
struct SourceLine {
    raw: String,
    trimmed: String,
}

fn source_line(filename: &str, line_number: u32) -> Option<SourceLine> {
    if line_number == 0 || filename.starts_with('<') {
        return None;
    }

    let source = std::fs::read_to_string(filename).ok()?;
    let raw = source.lines().nth(line_number.saturating_sub(1) as usize)?;
    let trimmed = raw.trim().to_string();
    Some(SourceLine {
        raw: raw.to_string(),
        trimmed,
    })
}

fn expression_columns(raw: &str) -> Option<(usize, usize)> {
    if let Some(start) = raw.find("BrokenIter(") {
        let rest = &raw[start..];
        let end = rest
            .find(')')
            .map_or(raw.len(), |offset| start + offset + ')'.len_utf8());
        return Some((start, end));
    }

    None
}

fn fallback_columns(raw: Option<&str>) -> (usize, usize) {
    let Some(raw) = raw else {
        return (0, 0);
    };
    let start = raw
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_whitespace()).then_some(index))
        .unwrap_or(0);
    let end = raw.trim_end().len().max(start);
    (start, end)
}

#[inline]
fn set_attr(object: &mut ShapedObject, name: &str, value: Value) {
    object.set_property(intern(name), value, shape_registry());
}

#[inline]
fn list_value(values: Vec<Value>) -> Value {
    crate::alloc_managed_value(ListObject::from_iter(values))
}

#[inline]
fn string_list_value(values: Vec<String>) -> Value {
    list_value(values.into_iter().map(owned_string_value).collect())
}

#[inline]
fn int_value(value: i64) -> Value {
    Value::int(value).unwrap_or_else(Value::none)
}

#[inline]
fn string_value(value: &str) -> Value {
    if value.len() <= 64 {
        Value::string(intern(value))
    } else {
        owned_string_value(value.to_string())
    }
}

#[inline]
fn owned_string_value(value: String) -> Value {
    if value.len() <= 64 {
        Value::string(intern(&value))
    } else {
        crate::alloc_managed_value(StringObject::from_string(value))
    }
}

#[inline]
fn traceback_type_error() -> BuiltinError {
    BuiltinError::TypeError("extract_tb() argument must be a traceback object or None".to_string())
}
