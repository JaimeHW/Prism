//! Native `traceback` module subset.
//!
//! Traceback collection stays on Prism's native frame/traceback objects. Source
//! lines and column spans are resolved lazily by `extract_tb`, so normal
//! exception creation remains allocation-light and does not touch the filesystem.

use super::{Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{
    BuiltinError, BuiltinFunctionObject, ExceptionTypeObject, ExceptionValue, builtin_repr_vm,
    builtin_str_vm, exception_display_text_for_value, iterator_to_value,
    runtime_error_to_builtin_error,
};
use crate::ops::calls::invoke_callable_value;
use crate::ops::iteration::collect_iterable_values;
use crate::ops::objects::{
    extract_type_id, get_attribute_value, snapshot_frame_globals_dict, snapshot_frame_locals_dict,
};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::class::{ClassFlags, PyClassObject};
use prism_runtime::object::descriptor::ClassMethodDescriptor;
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    SubclassBitmap, class_id_to_type_id, global_class_bitmap, register_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{FrameViewObject, TracebackViewObject};
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
use std::sync::{Arc, LazyLock};

static EXTRACT_TB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.extract_tb"), extract_tb));
static CLEAR_FRAMES_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.clear_frames"), clear_frames));
static FORMAT_TB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.format_tb"), format_tb));
static FORMAT_LIST_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("traceback.format_list"), format_list));
static PRINT_LIST_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.print_list"), print_list)
});
static WALK_STACK_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm(Arc::from("traceback.walk_stack"), walk_stack));
static EXTRACT_STACK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.extract_stack"), extract_stack)
});
static FORMAT_STACK_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(Arc::from("traceback.format_stack"), format_stack)
});
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
static FRAME_SUMMARY_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_frame_summary_class);
static STACK_SUMMARY_CLASS: LazyLock<Arc<PyClassObject>> = LazyLock::new(build_stack_summary_class);
static FRAME_SUMMARY_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.FrameSummary.__init__"),
        frame_summary_init,
    )
});
static FRAME_SUMMARY_REPR_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("traceback.FrameSummary.__repr__"),
        frame_summary_repr,
    )
});
static STACK_SUMMARY_INIT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.StackSummary.__init__"),
        stack_summary_init,
    )
});
static STACK_SUMMARY_EXTRACT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("traceback.StackSummary.extract"),
        stack_summary_extract,
    )
});
static STACK_SUMMARY_FROM_LIST_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("traceback.StackSummary.from_list"),
        stack_summary_from_list,
    )
});
static STACK_SUMMARY_FORMAT_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("traceback.StackSummary.format"),
        stack_summary_format,
    )
});
static STACK_SUMMARY_FORMAT_FRAME_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("traceback.StackSummary.format_frame_summary"),
        stack_summary_format_frame_summary,
    )
});
static STACK_SUMMARY_REVERSE_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("traceback.StackSummary.reverse"),
        stack_summary_reverse,
    )
});
static STACK_SUMMARY_APPEND_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("traceback.StackSummary.append"),
        stack_summary_append,
    )
});
static STACK_SUMMARY_ITER_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("traceback.StackSummary.__iter__"),
        stack_summary_iter,
    )
});
static STACK_SUMMARY_LEN_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("traceback.StackSummary.__len__"),
        stack_summary_len,
    )
});
static STACK_SUMMARY_GETITEM_METHOD: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("traceback.StackSummary.__getitem__"),
        stack_summary_getitem,
    )
});
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
                Arc::from("FrameSummary"),
                Arc::from("StackSummary"),
                Arc::from("TracebackException"),
                Arc::from("clear_frames"),
                Arc::from("extract_stack"),
                Arc::from("extract_tb"),
                Arc::from("format_exc"),
                Arc::from("format_exception"),
                Arc::from("format_exception_only"),
                Arc::from("format_list"),
                Arc::from("format_stack"),
                Arc::from("format_tb"),
                Arc::from("print_list"),
                Arc::from("print_exception"),
                Arc::from("walk_stack"),
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
            "FrameSummary" => Ok(frame_summary_class_value()),
            "StackSummary" => Ok(stack_summary_class_value()),
            "TracebackException" => Ok(traceback_exception_class_value()),
            "clear_frames" => Ok(builtin_value(&CLEAR_FRAMES_FUNCTION)),
            "extract_stack" => Ok(builtin_value(&EXTRACT_STACK_FUNCTION)),
            "extract_tb" => Ok(builtin_value(&EXTRACT_TB_FUNCTION)),
            "format_exc" => Ok(builtin_value(&FORMAT_EXC_FUNCTION)),
            "format_exception" => Ok(builtin_value(&FORMAT_EXCEPTION_FUNCTION)),
            "format_exception_only" => Ok(builtin_value(&FORMAT_EXCEPTION_ONLY_FUNCTION)),
            "format_list" => Ok(builtin_value(&FORMAT_LIST_FUNCTION)),
            "format_stack" => Ok(builtin_value(&FORMAT_STACK_FUNCTION)),
            "format_tb" => Ok(builtin_value(&FORMAT_TB_FUNCTION)),
            "print_list" => Ok(builtin_value(&PRINT_LIST_FUNCTION)),
            "print_exception" => Ok(builtin_value(&PRINT_EXCEPTION_FUNCTION)),
            "walk_stack" => Ok(builtin_value(&WALK_STACK_FUNCTION)),
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

#[inline]
fn frame_summary_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&FRAME_SUMMARY_CLASS) as *const ())
}

#[inline]
fn stack_summary_class_value() -> Value {
    Value::object_ptr(Arc::as_ptr(&STACK_SUMMARY_CLASS) as *const ())
}

#[inline]
fn classmethod_value(function: &'static BuiltinFunctionObject) -> Value {
    crate::alloc_managed_value(ClassMethodDescriptor::new(builtin_value(function)))
}

fn build_frame_summary_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("FrameSummary"));
    class.set_attr(intern("__module__"), Value::string(intern("traceback")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("FrameSummary")),
    );
    class.set_attr(
        intern("__doc__"),
        Value::string(intern("Information about a single frame from a traceback.")),
    );
    class.set_attr(
        intern("__init__"),
        builtin_value(&FRAME_SUMMARY_INIT_METHOD),
    );
    class.set_attr(
        intern("__repr__"),
        builtin_value(&FRAME_SUMMARY_REPR_METHOD),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    register_traceback_class(class)
}

fn build_stack_summary_class() -> Arc<PyClassObject> {
    let mut class = PyClassObject::new_simple(intern("StackSummary"));
    class.set_attr(intern("__module__"), Value::string(intern("traceback")));
    class.set_attr(
        intern("__qualname__"),
        Value::string(intern("StackSummary")),
    );
    class.set_attr(
        intern("__doc__"),
        Value::string(intern("A list-like summary of traceback frames.")),
    );
    class.set_attr(
        intern("__init__"),
        builtin_value(&STACK_SUMMARY_INIT_METHOD),
    );
    class.set_attr(
        intern("extract"),
        classmethod_value(&STACK_SUMMARY_EXTRACT_METHOD),
    );
    class.set_attr(
        intern("from_list"),
        classmethod_value(&STACK_SUMMARY_FROM_LIST_METHOD),
    );
    class.set_attr(
        intern("format"),
        builtin_value(&STACK_SUMMARY_FORMAT_METHOD),
    );
    class.set_attr(
        intern("format_frame_summary"),
        builtin_value(&STACK_SUMMARY_FORMAT_FRAME_METHOD),
    );
    class.set_attr(
        intern("reverse"),
        builtin_value(&STACK_SUMMARY_REVERSE_METHOD),
    );
    class.set_attr(
        intern("append"),
        builtin_value(&STACK_SUMMARY_APPEND_METHOD),
    );
    class.set_attr(
        intern("__iter__"),
        builtin_value(&STACK_SUMMARY_ITER_METHOD),
    );
    class.set_attr(intern("__len__"), builtin_value(&STACK_SUMMARY_LEN_METHOD));
    class.set_attr(
        intern("__getitem__"),
        builtin_value(&STACK_SUMMARY_GETITEM_METHOD),
    );
    class.add_flags(ClassFlags::INITIALIZED | ClassFlags::HAS_INIT | ClassFlags::NATIVE_HEAPTYPE);

    register_traceback_class(class)
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

fn register_traceback_class(class: PyClassObject) -> Arc<PyClassObject> {
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

    Ok(stack_summary_value(stack_summary_type_id(), summaries))
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
    let lines = format_stack_entries_value(extracted)?;
    Ok(string_list_value(lines))
}

fn format_list(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "format_list() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    Ok(string_list_value(format_stack_entries_value(args[0])?))
}

fn print_list(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let (stack, file) = bind_print_list_args(args, keywords)?;
    let lines = format_stack_entries_value(stack)?;
    write_traceback_lines(vm, file, &lines)?;
    Ok(Value::none())
}

fn walk_stack(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "walk_stack() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let frame_value = if args[0].is_none() {
        build_frame_view_at_depth(vm, 0)
            .ok_or_else(|| BuiltinError::ValueError("call stack is not deep enough".to_string()))?
    } else {
        args[0]
    };

    let mut frames = Vec::new();
    let mut cursor = Some(frame_value);
    while let Some(value) = cursor {
        if value.is_none() {
            break;
        }
        let frame = frame_view(value)?;
        let line = int_value(i64::from(frame.line_number()));
        frames.push(tuple_value(&[value, line]));
        cursor = frame.back();
    }

    Ok(list_value(frames))
}

fn extract_stack(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_extract_stack_args(vm, args, keywords)?;
    let walk = walk_stack(vm, &[bound.frame])?;
    let stack =
        stack_summary_from_frame_gen(vm, stack_summary_type_id(), walk, bound.limit, true, false)?;
    reverse_stack_summary_value(stack)?;
    Ok(stack)
}

fn format_stack(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let stack = extract_stack(vm, args, keywords)?;
    Ok(string_list_value(format_stack_entries_value(stack)?))
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

fn frame_summary_init(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_frame_summary_init_args(vm, args, keywords)?;
    let object = frame_summary_object_mut(bound.self_value, "__init__")?;
    populate_frame_summary(
        object,
        bound.filename,
        bound.lineno,
        bound.name,
        bound.line,
        bound.lookup_line,
        bound.locals,
        bound.end_lineno,
        bound.colno,
        bound.end_colno,
        0,
    )?;
    Ok(Value::none())
}

fn frame_summary_repr(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "FrameSummary.__repr__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let frame = frame_summary_ref(args[0])?;
    let filename = string_attr(frame, "filename").unwrap_or_else(|| "<unknown>".to_string());
    let name = string_attr(frame, "name").unwrap_or_else(|| "<unknown>".to_string());
    let line_number = int_attr(frame, "lineno").unwrap_or(0);
    let line_repr = frame
        .get_property("line")
        .map(|value| builtin_repr_vm(vm, &[value]))
        .transpose()?
        .and_then(|value| value_as_string_ref(value).map(|text| text.as_str().to_string()))
        .unwrap_or_else(|| "None".to_string());

    Ok(owned_string_value(format!(
        "<FrameSummary file {filename}, line {line_number} in {name}, line={line_repr}>"
    )))
}

fn stack_summary_init(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    if !keywords.is_empty() {
        return Err(unexpected_keyword("StackSummary", keywords[0].0));
    }
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary() takes at most 1 positional argument but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let object = stack_summary_object_mut(args[0], "__init__")?;
    let frames = if let Some(iterable) = args.get(1).copied() {
        collect_iterable_values(vm, iterable).map_err(runtime_error_to_builtin_error)?
    } else {
        Vec::new()
    };
    set_attr(object, stack_summary_frames_attr(), list_value(frames));
    Ok(Value::none())
}

fn stack_summary_extract(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let bound = bind_stack_summary_extract_args(vm, args, keywords)?;
    stack_summary_from_frame_gen(
        vm,
        bound.stack_type,
        bound.frame_gen,
        bound.limit,
        bound.lookup_lines,
        bound.capture_locals,
    )
}

fn stack_summary_from_list(vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.from_list() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let stack_type = stack_summary_type_from_class(args[0])?;
    let items = collect_stack_summary_items(vm, args[1])?;
    Ok(stack_summary_value(stack_type, items))
}

fn stack_summary_format(_vm: &mut VirtualMachine, args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.format() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(string_list_value(format_stack_entries_value(args[0])?))
}

fn stack_summary_format_frame_summary(
    _vm: &mut VirtualMachine,
    args: &[Value],
) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.format_frame_summary() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let lines = format_frame_summary(args[1])?;
    Ok(owned_string_value(lines.concat()))
}

fn stack_summary_reverse(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.reverse() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    reverse_stack_summary_value(args[0])?;
    Ok(Value::none())
}

fn stack_summary_append(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.append() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    stack_summary_frames_mut(args[0], "append")?.push(args[1]);
    Ok(Value::none())
}

fn stack_summary_iter(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.__iter__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let values = stack_summary_frames_ref(args[0], "__iter__")?
        .as_slice()
        .to_vec();
    Ok(iterator_to_value(IteratorObject::from_values(values)))
}

fn stack_summary_len(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.__len__() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    Ok(int_value(
        stack_summary_frames_ref(args[0], "__len__")?.len() as i64,
    ))
}

fn stack_summary_getitem(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.__getitem__() takes exactly 1 argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }
    let index = args[1].as_int().or_else(|| {
        args[1]
            .as_bool()
            .map(|flag| if flag { 1_i64 } else { 0_i64 })
    });
    let Some(index) = index else {
        return Err(BuiltinError::TypeError(format!(
            "list indices must be integers, not {}",
            args[1].type_name()
        )));
    };
    stack_summary_frames_ref(args[0], "__getitem__")?
        .get(index)
        .ok_or_else(|| BuiltinError::IndexError("list index out of range".to_string()))
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

#[derive(Clone, Copy)]
struct FrameSummaryInitArgs {
    self_value: Value,
    filename: Value,
    lineno: i64,
    name: Value,
    lookup_line: bool,
    locals: Value,
    line: Option<Value>,
    end_lineno: Value,
    colno: Value,
    end_colno: Value,
}

#[derive(Clone, Copy)]
struct StackSummaryExtractArgs {
    stack_type: TypeId,
    frame_gen: Value,
    limit: Option<Value>,
    lookup_lines: bool,
    capture_locals: bool,
}

#[derive(Clone, Copy)]
struct ExtractStackArgs {
    frame: Value,
    limit: Option<Value>,
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

fn bind_frame_summary_init_args(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<FrameSummaryInitArgs, BuiltinError> {
    if args.len() < 4 || args.len() > 4 {
        return Err(BuiltinError::TypeError(format!(
            "FrameSummary() takes 3 positional arguments but {} were given",
            args.len().saturating_sub(1)
        )));
    }

    let mut lookup_line = Value::bool(true);
    let mut locals = Value::none();
    let mut line = None;
    let mut end_lineno = Value::none();
    let mut colno = Value::none();
    let mut end_colno = Value::none();

    for &(name, value) in keywords {
        match name {
            "lookup_line" => lookup_line = value,
            "locals" => locals = value,
            "line" => line = Some(value),
            "end_lineno" => end_lineno = value,
            "colno" => colno = value,
            "end_colno" => end_colno = value,
            other => return Err(unexpected_keyword("FrameSummary", other)),
        }
    }

    Ok(FrameSummaryInitArgs {
        self_value: args[0],
        filename: args[1],
        lineno: int_arg(args[2], "lineno")?,
        name: args[3],
        lookup_line: crate::truthiness::try_is_truthy(vm, lookup_line)
            .map_err(runtime_error_to_builtin_error)?,
        locals,
        line,
        end_lineno,
        colno,
        end_colno,
    })
}

fn bind_stack_summary_extract_args(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<StackSummaryExtractArgs, BuiltinError> {
    if args.len() != 2 {
        return Err(BuiltinError::TypeError(format!(
            "StackSummary.extract() takes exactly 1 positional argument ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let mut limit = None;
    let mut lookup_lines = Value::bool(true);
    let mut capture_locals = Value::bool(false);

    for &(name, value) in keywords {
        match name {
            "limit" => limit = Some(value),
            "lookup_lines" => lookup_lines = value,
            "capture_locals" => capture_locals = value,
            other => return Err(unexpected_keyword("StackSummary.extract", other)),
        }
    }

    Ok(StackSummaryExtractArgs {
        stack_type: stack_summary_type_from_class(args[0])?,
        frame_gen: args[1],
        limit,
        lookup_lines: crate::truthiness::try_is_truthy(vm, lookup_lines)
            .map_err(runtime_error_to_builtin_error)?,
        capture_locals: crate::truthiness::try_is_truthy(vm, capture_locals)
            .map_err(runtime_error_to_builtin_error)?,
    })
}

fn bind_extract_stack_args(
    vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<ExtractStackArgs, BuiltinError> {
    if args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "extract_stack() takes from 0 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut frame = args.first().copied();
    let mut limit = args.get(1).copied();
    for &(name, value) in keywords {
        match name {
            "f" => {
                assign_optional_keyword(&mut frame, value, "extract_stack", "f", !args.is_empty())?
            }
            "limit" => assign_optional_keyword(
                &mut limit,
                value,
                "extract_stack",
                "limit",
                args.len() > 1,
            )?,
            other => return Err(unexpected_keyword("extract_stack", other)),
        }
    }

    let frame = match frame {
        Some(value) if !value.is_none() => value,
        _ => build_frame_view_at_depth(vm, 0)
            .ok_or_else(|| BuiltinError::ValueError("call stack is not deep enough".to_string()))?,
    };

    Ok(ExtractStackArgs { frame, limit })
}

fn bind_print_list_args(
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<(Value, Option<Value>), BuiltinError> {
    if args.is_empty() || args.len() > 2 {
        return Err(BuiltinError::TypeError(format!(
            "print_list() takes from 1 to 2 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut file = args.get(1).copied();
    for &(name, value) in keywords {
        match name {
            "file" => {
                assign_optional_keyword(&mut file, value, "print_list", "file", args.len() > 1)?
            }
            other => return Err(unexpected_keyword("print_list", other)),
        }
    }

    Ok((args[0], file))
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
    let entries = stack_entries_ref(stack)?;
    let mut lines = Vec::with_capacity(entries.len().saturating_mul(2).saturating_add(1));
    if entries.is_empty() {
        return Ok(lines);
    }

    lines.push("Traceback (most recent call last):\n".to_string());
    for frame in entries.iter().copied() {
        lines.extend(format_frame_summary(frame)?);
    }
    Ok(lines)
}

fn format_stack_entries_value(stack: Value) -> Result<Vec<String>, BuiltinError> {
    let entries = stack_entries_ref(stack)?;
    let mut lines = Vec::with_capacity(entries.len().saturating_mul(2));
    for frame in entries.iter().copied() {
        lines.extend(format_frame_summary(frame)?);
    }
    Ok(lines)
}

fn format_frame_summary(frame: Value) -> Result<Vec<String>, BuiltinError> {
    let frame = frame_summary_ref(frame)?;
    let filename = string_attr(frame, "filename").unwrap_or_else(|| "<unknown>".to_string());
    let name = string_attr(frame, "name").unwrap_or_else(|| "<unknown>".to_string());
    let line_number = int_attr(frame, "lineno").unwrap_or(0);
    let source = string_attr(frame, "line")
        .filter(|line| !line.is_empty())
        .or_else(|| source_line(&filename, line_number.max(0) as u32).map(|line| line.trimmed))
        .unwrap_or_default();

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

fn list_mut(value: Value) -> Result<&'static mut ListObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError("expected a list".to_string()));
    };
    if extract_type_id(ptr) != TypeId::LIST {
        return Err(BuiltinError::TypeError("expected a list".to_string()));
    }
    Ok(unsafe { &mut *(ptr as *mut ListObject) })
}

fn stack_entries_ref(value: Value) -> Result<&'static [Value], BuiltinError> {
    if let Some(ptr) = value.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        if type_id == TypeId::LIST {
            return Ok(unsafe { &*(ptr as *const ListObject) }.as_slice());
        }
        if is_stack_summary_type(type_id) {
            return Ok(stack_summary_frames_ref(value, "format")?.as_slice());
        }
    }

    Err(BuiltinError::TypeError(
        "expected a StackSummary or list".to_string(),
    ))
}

fn frame_summary_ref(value: Value) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "traceback frame summary is not an object".to_string(),
        ));
    };
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn frame_summary_object_mut(
    value: Value,
    context: &'static str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(frame_summary_type_error(context));
    };
    if !is_frame_summary_type(extract_type_id(ptr)) {
        return Err(frame_summary_type_error(context));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn stack_summary_object_mut(
    value: Value,
    context: &'static str,
) -> Result<&'static mut ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(stack_summary_type_error(context));
    };
    if !is_stack_summary_type(extract_type_id(ptr)) {
        return Err(stack_summary_type_error(context));
    }
    Ok(unsafe { &mut *(ptr as *mut ShapedObject) })
}

fn stack_summary_object(
    value: Value,
    context: &'static str,
) -> Result<&'static ShapedObject, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(stack_summary_type_error(context));
    };
    if !is_stack_summary_type(extract_type_id(ptr)) {
        return Err(stack_summary_type_error(context));
    }
    Ok(unsafe { &*(ptr as *const ShapedObject) })
}

fn stack_summary_frames_ref(
    value: Value,
    context: &'static str,
) -> Result<&'static ListObject, BuiltinError> {
    let object = stack_summary_object(value, context)?;
    let frames = object
        .get_property(stack_summary_frames_attr())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "StackSummary.{context} requires initialized frame storage"
            ))
        })?;
    list_ref(frames)
}

fn stack_summary_frames_mut(
    value: Value,
    context: &'static str,
) -> Result<&'static mut ListObject, BuiltinError> {
    let object = stack_summary_object(value, context)?;
    let frames = object
        .get_property(stack_summary_frames_attr())
        .ok_or_else(|| {
            BuiltinError::TypeError(format!(
                "StackSummary.{context} requires initialized frame storage"
            ))
        })?;
    list_mut(frames)
}

fn reverse_stack_summary_value(value: Value) -> Result<(), BuiltinError> {
    stack_summary_frames_mut(value, "reverse")?.reverse();
    Ok(())
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
    frame_summary_from_frame_view(
        frame,
        line_number,
        Value::none(),
        Value::none(),
        Value::none(),
        true,
        false,
        lasti,
    )
}

fn frame_summary_from_frame_view(
    frame: &FrameViewObject,
    line_number: u32,
    end_lineno: Value,
    colno_value: Value,
    end_colno_value: Value,
    lookup_line: bool,
    capture_locals: bool,
    lasti: u32,
) -> Result<Value, BuiltinError> {
    let code = frame.code();
    let filename = string_value(code.map_or("<unknown>", |code| code.filename.as_ref()));
    let name = string_value(code.map_or("<unknown>", |code| code.name.as_ref()));
    let locals = if capture_locals {
        frame.locals()
    } else {
        Value::none()
    };

    let mut summary = ShapedObject::new(frame_summary_type_id(), shape_registry().empty_shape());
    populate_frame_summary(
        &mut summary,
        filename,
        i64::from(line_number),
        name,
        None,
        lookup_line,
        locals,
        end_lineno,
        colno_value,
        end_colno_value,
        lasti,
    )?;
    Ok(crate::alloc_managed_value(summary))
}

fn populate_frame_summary(
    summary: &mut ShapedObject,
    filename_value: Value,
    line_number: i64,
    name_value: Value,
    line_value: Option<Value>,
    lookup_line: bool,
    locals: Value,
    end_lineno: Value,
    colno_value: Value,
    end_colno_value: Value,
    lasti: u32,
) -> Result<(), BuiltinError> {
    let filename = string_from_value(filename_value)
        .ok_or_else(|| BuiltinError::TypeError("filename must be a string".to_string()))?;
    let name = string_from_value(name_value)
        .ok_or_else(|| BuiltinError::TypeError("name must be a string".to_string()))?;
    let source = line_value
        .and_then(string_from_value)
        .map(|line| SourceLine {
            raw: line.clone(),
            trimmed: line.trim().to_string(),
        })
        .or_else(|| {
            lookup_line
                .then(|| source_line(&filename, line_number.max(0) as u32))
                .flatten()
        });
    let (colno, end_colno) = source
        .as_ref()
        .and_then(|line| expression_columns(line.raw.as_str()))
        .unwrap_or_else(|| fallback_columns(source.as_ref().map(|line| line.raw.as_str())));

    set_attr(summary, "filename", string_value(&filename));
    set_attr(summary, "lineno", int_value(line_number));
    set_attr(
        summary,
        "end_lineno",
        normalize_optional_int(end_lineno)?.unwrap_or_else(|| int_value(line_number)),
    );
    set_attr(summary, "name", string_value(&name));
    set_attr(
        summary,
        "line",
        source
            .as_ref()
            .map(|line| owned_string_value(line.trimmed.clone()))
            .unwrap_or_else(Value::none),
    );
    let stored_line = summary.get_property("line").unwrap_or_else(Value::none);
    set_attr(summary, "_line", stored_line);
    set_attr(
        summary,
        "colno",
        normalize_optional_int(colno_value)?.unwrap_or_else(|| int_value(colno as i64)),
    );
    set_attr(
        summary,
        "end_colno",
        normalize_optional_int(end_colno_value)?.unwrap_or_else(|| int_value(end_colno as i64)),
    );
    set_attr(summary, "locals", locals);
    set_attr(summary, "lasti", int_value(lasti as i64));

    Ok(())
}

fn stack_summary_from_frame_gen(
    vm: &mut VirtualMachine,
    stack_type: TypeId,
    frame_gen: Value,
    limit: Option<Value>,
    lookup_lines: bool,
    capture_locals: bool,
) -> Result<Value, BuiltinError> {
    let mut values =
        collect_iterable_values(vm, frame_gen).map_err(runtime_error_to_builtin_error)?;
    apply_frame_limit(&mut values, parse_limit(limit)?);

    let mut summaries = Vec::with_capacity(values.len());
    for value in values {
        let (frame_value, lineno, end_lineno, colno, end_colno) = frame_entry_parts(value)?;
        let frame = frame_view(frame_value)?;
        summaries.push(frame_summary_from_frame_view(
            frame,
            lineno,
            end_lineno,
            colno,
            end_colno,
            lookup_lines,
            capture_locals,
            frame.lasti(),
        )?);
    }

    Ok(stack_summary_value(stack_type, summaries))
}

fn apply_frame_limit(values: &mut Vec<Value>, limit: TracebackLimit) {
    match limit {
        TracebackLimit::All => {}
        TracebackLimit::First(count) => {
            values.truncate(count);
        }
        TracebackLimit::Last(count) => {
            let len = values.len();
            if count < len {
                values.drain(0..len - count);
            }
        }
    }
}

fn frame_entry_parts(value: Value) -> Result<(Value, u32, Value, Value, Value), BuiltinError> {
    let items = sequence_items(value).ok_or_else(|| {
        BuiltinError::TypeError(
            "StackSummary.extract() frame generator must yield tuples".to_string(),
        )
    })?;
    if items.len() != 2 {
        return Err(BuiltinError::ValueError(
            "StackSummary.extract() frame generator must yield (frame, lineno) pairs".to_string(),
        ));
    }

    let frame = items[0];
    let positions = sequence_items(items[1]);
    if let Some(positions) = positions {
        if positions.is_empty() {
            return Err(BuiltinError::ValueError(
                "frame position tuple must contain a line number".to_string(),
            ));
        }
        return Ok((
            frame,
            u32_arg(positions[0], "lineno")?,
            positions.get(1).copied().unwrap_or_else(Value::none),
            positions.get(2).copied().unwrap_or_else(Value::none),
            positions.get(3).copied().unwrap_or_else(Value::none),
        ));
    }

    Ok((
        frame,
        u32_arg(items[1], "lineno")?,
        Value::none(),
        Value::none(),
        Value::none(),
    ))
}

fn collect_stack_summary_items(
    vm: &mut VirtualMachine,
    source: Value,
) -> Result<Vec<Value>, BuiltinError> {
    let values = if let Some(ptr) = source.as_object_ptr() {
        if is_stack_summary_type(extract_type_id(ptr)) {
            return Ok(stack_summary_frames_ref(source, "from_list")?
                .as_slice()
                .to_vec());
        }
        collect_iterable_values(vm, source).map_err(runtime_error_to_builtin_error)?
    } else {
        collect_iterable_values(vm, source).map_err(runtime_error_to_builtin_error)?
    };

    values
        .into_iter()
        .map(frame_summary_from_list_item)
        .collect()
}

fn frame_summary_from_list_item(value: Value) -> Result<Value, BuiltinError> {
    if value
        .as_object_ptr()
        .is_some_and(|ptr| is_frame_summary_type(extract_type_id(ptr)))
    {
        return Ok(value);
    }

    let items = sequence_items(value).ok_or_else(|| {
        BuiltinError::TypeError(
            "StackSummary.from_list() entries must be frame summaries or tuples".to_string(),
        )
    })?;
    if items.len() < 4 {
        return Err(BuiltinError::ValueError(
            "StackSummary.from_list() entries must have at least four fields".to_string(),
        ));
    }

    let mut summary = ShapedObject::new(frame_summary_type_id(), shape_registry().empty_shape());
    populate_frame_summary(
        &mut summary,
        items[0],
        int_arg(items[1], "lineno")?,
        items[2],
        Some(items[3]),
        false,
        Value::none(),
        Value::none(),
        Value::none(),
        Value::none(),
        0,
    )?;
    Ok(crate::alloc_managed_value(summary))
}

fn stack_summary_value(type_id: TypeId, frames: Vec<Value>) -> Value {
    let mut summary = ShapedObject::new(type_id, shape_registry().empty_shape());
    set_attr(
        &mut summary,
        stack_summary_frames_attr(),
        list_value(frames),
    );
    crate::alloc_managed_value(summary)
}

fn stack_summary_type_from_class(value: Value) -> Result<TypeId, BuiltinError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Err(BuiltinError::TypeError(
            "StackSummary class method requires a type".to_string(),
        ));
    };
    if extract_type_id(ptr) != TypeId::TYPE {
        return Err(BuiltinError::TypeError(
            "StackSummary class method requires a type".to_string(),
        ));
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    let type_id = class.class_type_id();
    if is_stack_summary_type(type_id) {
        Ok(type_id)
    } else {
        Err(BuiltinError::TypeError(
            "StackSummary class method requires a StackSummary subclass".to_string(),
        ))
    }
}

fn build_frame_view_at_depth(vm: &VirtualMachine, depth: usize) -> Option<Value> {
    let frame_index = vm.frame_index_at_depth(depth)?;
    build_frame_view_from_index(vm, frame_index)
}

fn build_frame_view_from_index(vm: &VirtualMachine, frame_index: usize) -> Option<Value> {
    let frame = vm.frames.get(frame_index)?;
    let globals = crate::alloc_managed_value(snapshot_frame_globals_dict(vm, frame));
    let locals = crate::alloc_managed_value(snapshot_frame_locals_dict(frame));
    let back = frame
        .return_frame
        .and_then(|back_index| build_frame_view_from_index(vm, back_index as usize));
    Some(crate::alloc_managed_value(FrameViewObject::new(
        Some(Arc::clone(&frame.code)),
        globals,
        locals,
        frame_line_number(frame),
        frame.ip,
        back,
    )))
}

#[inline]
fn frame_line_number(frame: &crate::frame::Frame) -> u32 {
    frame
        .code
        .line_for_pc(frame.ip)
        .or_else(|| {
            frame
                .ip
                .checked_sub(1)
                .and_then(|pc| frame.code.line_for_pc(pc))
        })
        .unwrap_or(frame.code.first_lineno)
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

fn sequence_items(value: Value) -> Option<Vec<Value>> {
    if let Some(list) = value.as_object_ptr().and_then(|ptr| {
        (extract_type_id(ptr) == TypeId::LIST).then(|| unsafe { &*(ptr as *const ListObject) })
    }) {
        return Some(list.as_slice().to_vec());
    }
    value_as_tuple_ref(value).map(|tuple| tuple.as_slice().to_vec())
}

fn int_arg(value: Value, name: &'static str) -> Result<i64, BuiltinError> {
    if let Some(integer) = value.as_int() {
        return Ok(integer);
    }
    if let Some(flag) = value.as_bool() {
        return Ok(if flag { 1 } else { 0 });
    }
    Err(BuiltinError::TypeError(format!(
        "{name} must be an integer"
    )))
}

fn u32_arg(value: Value, name: &'static str) -> Result<u32, BuiltinError> {
    let value = int_arg(value, name)?;
    if value < 0 {
        return Err(BuiltinError::ValueError(format!(
            "{name} must be greater than or equal to zero"
        )));
    }
    u32::try_from(value).map_err(|_| BuiltinError::OverflowError(format!("{name} is too large")))
}

fn normalize_optional_int(value: Value) -> Result<Option<Value>, BuiltinError> {
    if value.is_none() {
        return Ok(None);
    }
    if let Some(integer) = value.as_int() {
        return Ok(Some(int_value(integer)));
    }
    if let Some(flag) = value.as_bool() {
        return Ok(Some(int_value(if flag { 1 } else { 0 })));
    }
    Err(BuiltinError::TypeError(
        "optional traceback position fields must be integers or None".to_string(),
    ))
}

#[inline]
fn frame_summary_type_id() -> TypeId {
    FRAME_SUMMARY_CLASS.class_type_id()
}

#[inline]
fn stack_summary_type_id() -> TypeId {
    STACK_SUMMARY_CLASS.class_type_id()
}

#[inline]
fn is_frame_summary_type(type_id: TypeId) -> bool {
    type_id == frame_summary_type_id()
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(frame_summary_type_id())))
}

#[inline]
fn is_stack_summary_type(type_id: TypeId) -> bool {
    type_id == stack_summary_type_id()
        || (type_id.raw() >= TypeId::FIRST_USER_TYPE
            && global_class_bitmap(ClassId(type_id.raw()))
                .is_some_and(|bitmap| bitmap.is_subclass_of(stack_summary_type_id())))
}

#[inline]
fn stack_summary_frames_attr() -> &'static str {
    "__prism_stack_summary_frames__"
}

fn frame_summary_type_error(context: &'static str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "FrameSummary.{context} requires a FrameSummary object"
    ))
}

fn stack_summary_type_error(context: &'static str) -> BuiltinError {
    BuiltinError::TypeError(format!(
        "StackSummary.{context} requires a StackSummary object"
    ))
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
fn tuple_value(values: &[Value]) -> Value {
    crate::alloc_managed_value(TupleObject::from_slice(values))
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
