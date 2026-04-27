//! Function call opcode handlers.
//!
//! Handles function calls, closures, and tail calls.
//!
//! # Performance Notes
//!
//! - Function objects allocate through the active VM heap with stable fallback storage
//! - Call dispatch uses O(1) type discrimination via ObjectHeader
//! - Arguments are passed via register file, avoiding heap allocation

use crate::VirtualMachine;
use crate::builtins::BuiltinFunctionObject;
use crate::builtins::{EXCEPTION_TYPE_ID, ExceptionTypeObject};
use crate::builtins::{
    builtin_type_object_type_id, call_builtin_type_kw_with_vm, call_builtin_type_with_vm,
};
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::frame::{ClosureEnv, RegisterSnapshot};
use crate::ops::kw_binding::{ArgumentBinder, BoundArguments};
use crate::stdlib::generators::{GeneratorObject, LivenessMap};
use crate::vm::NestedTargetFrameOutcome;
use prism_code::{CodeFlags, CodeObject, Instruction};
use prism_core::Value;
use prism_core::intern::{intern, interned_by_ptr};
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{BoundMethod, StaticMethodDescriptor};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::DescriptorViewObject;
use prism_runtime::types::Cell;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::sync::{Arc, LazyLock};

use super::iteration::{IterStep, next_step};

mod function_creation;
mod function_defaults;

use function_creation::materialize_function_invocation_closure;
pub(crate) use function_creation::{
    capture_closure_environment, initialize_closure_cellvars_from_locals,
};
pub use function_creation::{make_closure, make_function};
pub use function_defaults::set_function_defaults;

static OBJECT_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("object.__new__"),
        crate::builtins::builtin_object_new,
    )
});
static OBJECT_INIT_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("object.__init__"),
        crate::builtins::builtin_object_init,
    )
});
static TYPE_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("type.__new__"),
        crate::builtins::builtin_type_new_with_vm,
    )
});
static TYPE_INIT_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("type.__init__"),
        crate::builtins::builtin_type_init,
    )
});
static TUPLE_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("tuple.__new__"),
        crate::builtins::builtin_tuple_new,
    )
});
static LIST_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("list.__new__"), crate::builtins::builtin_list_new)
});
static LIST_INIT_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("list.__init__"),
        crate::builtins::builtin_list_init_vm,
    )
});
static DICT_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("dict.__new__"), crate::builtins::builtin_dict_new)
});
static DICT_INIT_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("dict.__init__"),
        crate::builtins::builtin_dict_init_vm_kw,
    )
});
static INT_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm(
        Arc::from("int.__new__"),
        crate::builtins::builtin_int_new_vm,
    )
});
static FLOAT_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("float.__new__"),
        crate::builtins::builtin_float_new,
    )
});
static STR_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("str.__new__"), crate::builtins::builtin_str_new)
});
static BOOL_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(Arc::from("bool.__new__"), crate::builtins::builtin_bool_new)
});
static BYTES_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("bytes.__new__"),
        crate::builtins::builtin_bytes_new,
    )
});
static BYTEARRAY_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new(
        Arc::from("bytearray.__new__"),
        crate::builtins::builtin_bytearray_new,
    )
});
static ENUMERATE_NEW_SLOT_FUNCTION: LazyLock<BuiltinFunctionObject> = LazyLock::new(|| {
    BuiltinFunctionObject::new_vm_kw(
        Arc::from("enumerate.__new__"),
        crate::builtins::builtin_enumerate_new_vm_kw,
    )
});

const DIRECT_CALL_RETURN_REG: u8 = u8::MAX;

#[derive(Clone, Copy, Debug)]
pub(crate) enum InvokeCallableOutcome {
    Returned(Value),
    ControlTransferred,
}

#[inline(always)]
fn restore_direct_call_caller_state(
    vm: &mut VirtualMachine,
    stop_depth: usize,
    saved_register: RegisterSnapshot,
    saved_exception_context: crate::vm::ExceptionContextSnapshot,
) {
    if vm.call_depth() == stop_depth {
        vm.current_frame_mut().restore_register(saved_register);
        vm.restore_exception_context(saved_exception_context);
    }
}

#[inline(always)]
fn should_restore_direct_call_caller_state(
    vm: &VirtualMachine,
    stop_depth: usize,
    result: &Result<InvokeCallableOutcome, RuntimeError>,
) -> bool {
    if vm.call_depth() != stop_depth {
        return false;
    }

    match result {
        Ok(InvokeCallableOutcome::ControlTransferred) => false,
        Err(err) if err.is_control_transferred() => false,
        _ => true,
    }
}

#[inline(always)]
fn should_restore_direct_call_caller_state_for_value_result(
    vm: &VirtualMachine,
    stop_depth: usize,
    result: &Result<Value, RuntimeError>,
) -> bool {
    if vm.call_depth() != stop_depth {
        return false;
    }

    !matches!(result, Err(err) if err.is_control_transferred())
}

// =============================================================================
// Type ID Extraction Helper
// =============================================================================

/// Extract TypeId from an object pointer.
///
/// SAFETY: Relies on ObjectHeader being at offset 0 of all PyObject types.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header_ptr = ptr as *const ObjectHeader;
    unsafe { (*header_ptr).type_id }
}

#[derive(Clone, Copy, Debug)]
struct ResolvedCallableTarget {
    callable: Value,
    implicit_self: Value,
}

fn resolve_dunder_call_target(value: Value) -> Option<ResolvedCallableTarget> {
    let ptr = value.as_object_ptr()?;
    let type_id = extract_type_id(ptr);

    if type_id == TypeId::OBJECT || type_id.raw() >= TypeId::FIRST_USER_TYPE {
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        if let Some(callable) = shaped.get_property("__call__") {
            return Some(ResolvedCallableTarget {
                callable,
                implicit_self: value,
            });
        }
    }

    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        let class = global_class(ClassId(type_id.raw()))?;
        let name = intern("__call__");
        if let Some(slot) = class.lookup_method_published(&name) {
            return Some(ResolvedCallableTarget {
                callable: slot.value,
                implicit_self: value,
            });
        }
    }

    None
}

pub(crate) fn value_supports_call_protocol(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION
        | TypeId::METHOD
        | TypeId::CLOSURE
        | TypeId::STATICMETHOD
        | TypeId::TYPE
        | TypeId::BUILTIN_FUNCTION
        | TypeId::EXCEPTION_TYPE => true,
        TypeId::WRAPPER_DESCRIPTOR | TypeId::METHOD_DESCRIPTOR | TypeId::CLASSMETHOD_DESCRIPTOR => {
            reflected_descriptor_callable_target(value).is_some()
        }
        _ => resolve_dunder_call_target(value).is_some(),
    }
}

#[inline]
fn reflected_descriptor_callable_target(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    let descriptor_type = extract_type_id(ptr);
    if !matches!(
        descriptor_type,
        TypeId::WRAPPER_DESCRIPTOR | TypeId::METHOD_DESCRIPTOR | TypeId::CLASSMETHOD_DESCRIPTOR
    ) {
        return None;
    }

    let descriptor = unsafe { &*(ptr as *const DescriptorViewObject) };
    crate::builtins::reflected_descriptor_callable_value(
        descriptor_type,
        descriptor.owner(),
        descriptor.name(),
    )
}

fn invoke_resolved_callable(
    vm: &mut VirtualMachine,
    resolved: ResolvedCallableTarget,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<Value, RuntimeError> {
    let Some(callable_ptr) = resolved.callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            resolved.callable.type_name()
        )));
    };

    match extract_type_id(callable_ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => invoke_user_function_with_implicit_self(
            vm,
            callable_ptr,
            resolved.implicit_self,
            dst_reg,
            posargc,
            kwargc,
            kwnames_idx,
        ),
        TypeId::BUILTIN_FUNCTION => invoke_builtin_with_implicit_self_from_frame(
            vm,
            callable_ptr,
            resolved.implicit_self,
            dst_reg,
            posargc,
            kwargc,
            kwnames_idx,
        ),
        TypeId::METHOD => {
            let bound = unsafe { &*(callable_ptr as *const BoundMethod) };
            let bound_callable = bound.function();
            let bound_self = bound.instance();
            let Some(bound_ptr) = bound_callable.as_object_ptr() else {
                return Err(RuntimeError::type_error(
                    "bound __call__ method has invalid function",
                ));
            };

            match extract_type_id(bound_ptr) {
                TypeId::FUNCTION | TypeId::CLOSURE => invoke_user_function_with_implicit_self(
                    vm,
                    bound_ptr,
                    bound_self,
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                ),
                TypeId::BUILTIN_FUNCTION => invoke_builtin_with_implicit_self_from_frame(
                    vm,
                    bound_ptr,
                    bound_self,
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                ),
                type_id => Err(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    type_id.name()
                ))),
            }
        }
        type_id => Err(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            type_id.name()
        ))),
    }
}

#[inline]
pub(crate) fn function_module_ptr(vm: &VirtualMachine, func: &FunctionObject) -> *const () {
    if func.globals_ptr().is_null() {
        vm.current_module_cloned()
            .map(|module| Arc::as_ptr(&module) as *const ())
            .unwrap_or(std::ptr::null())
    } else {
        func.globals_ptr()
    }
}

#[inline]
pub(crate) fn resolve_function_module(
    vm: &VirtualMachine,
    func: &FunctionObject,
) -> Option<Arc<crate::import::ModuleObject>> {
    vm.module_from_globals_ptr(function_module_ptr(vm, func))
        .or_else(|| vm.current_module_cloned())
}

#[inline]
fn class_object_from_type_ptr(ptr: *const ()) -> Option<&'static PyClassObject> {
    if builtin_type_object_type_id(ptr).is_some() {
        return None;
    }

    Some(unsafe { &*(ptr as *const PyClassObject) })
}

#[inline]
fn function_vectorcall_override_result(func: &FunctionObject) -> Option<Value> {
    func.has_test_vectorcall_override()
        .then(|| Value::string(intern("overridden")))
}

#[inline]
fn alloc_heap_value<T>(
    _vm: &mut VirtualMachine,
    object: T,
    _context: &'static str,
) -> Result<Value, RuntimeError>
where
    T: prism_runtime::Trace + 'static,
{
    Ok(alloc_value_in_current_heap_or_box(object))
}

fn snapshot_module_dict(module: &crate::import::ModuleObject) -> DictObject {
    let attrs = module.all_attrs();
    let mut dict = DictObject::with_capacity(attrs.len());
    for (name, value) in attrs {
        dict.set(Value::string(name), value);
    }
    dict
}

fn snapshot_globals_dict(vm: &VirtualMachine) -> DictObject {
    if let Some(module) = vm.current_module_cloned() {
        return snapshot_module_dict(module.as_ref());
    }

    let mut dict = DictObject::with_capacity(vm.globals.len());
    for (name, value) in vm.globals.iter() {
        dict.set(Value::string(intern(name.as_ref())), *value);
    }
    dict
}

fn snapshot_frame_locals_dict(frame: &crate::frame::Frame) -> DictObject {
    let limit = frame.code.locals.len().min(u8::MAX as usize + 1);
    let mut dict = DictObject::with_capacity(limit);
    for idx in 0..limit {
        let reg = idx as u8;
        if !frame.reg_is_written(reg) {
            continue;
        }
        dict.set(
            Value::string(intern(frame.code.locals[idx].as_ref())),
            frame.get_reg(reg),
        );
    }
    dict
}

fn current_globals_value(vm: &mut VirtualMachine) -> Result<Value, RuntimeError> {
    if let Some(module) = vm.current_module_cloned() {
        return Ok(module.dict_value());
    }

    let dict = snapshot_globals_dict(vm);
    alloc_heap_value(vm, dict, "globals dict snapshot")
}

fn current_locals_value(vm: &mut VirtualMachine) -> Result<Value, RuntimeError> {
    let is_module_frame = {
        let frame = vm.current_frame();
        frame.return_frame.is_none() && frame.module.is_some()
    };
    if is_module_frame && let Some(module) = vm.current_module_cloned() {
        return Ok(module.dict_value());
    }

    let dict = if is_module_frame {
        snapshot_globals_dict(vm)
    } else {
        let frame = vm.current_frame();
        snapshot_frame_locals_dict(frame)
    };
    alloc_heap_value(vm, dict, "locals dict snapshot")
}

#[inline]
fn load_code_constant(
    frame: &crate::frame::Frame,
    code_idx: u16,
    context: &'static str,
) -> Result<Arc<CodeObject>, RuntimeError> {
    let code_val = frame.get_const(code_idx);
    let Some(code_ptr) = code_val.as_object_ptr() else {
        return Err(RuntimeError::internal(format!(
            "invalid code object in constant pool for {context}",
        )));
    };

    let code_raw = code_ptr as *const CodeObject;
    let code = unsafe { Arc::from_raw(code_raw) };
    let code_clone = Arc::clone(&code);
    std::mem::forget(code);
    Ok(code_clone)
}

#[inline]
fn current_function_module_name(vm: &VirtualMachine) -> prism_core::intern::InternedString {
    vm.current_module_cloned()
        .map(|module| intern(module.name()))
        .unwrap_or_else(|| intern("__main__"))
}

#[inline]
fn current_function_globals_ptr(vm: &VirtualMachine) -> *const () {
    vm.current_module_cloned()
        .map(|module| Arc::as_ptr(&module) as *const ())
        .unwrap_or(std::ptr::null())
}

fn new_function_object(
    vm: &VirtualMachine,
    code: Arc<CodeObject>,
    closure: Option<Arc<ClosureEnv>>,
) -> FunctionObject {
    let function_name = Arc::clone(&code.name);
    let qualname_value = Value::string(intern(code.qualname.as_ref()));
    let module_name_value = Value::string(current_function_module_name(vm));

    let mut func = FunctionObject::new(code, function_name, None, closure);
    unsafe {
        func.set_globals_ptr(current_function_globals_ptr(vm));
    }
    func.set_attr(intern("__qualname__"), qualname_value);
    func.set_attr(intern("__module__"), module_name_value);
    func
}

#[inline(always)]
fn builtin_slot_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

#[inline]
fn builtin_instantiation_slot_value(owner: TypeId, name: &str) -> Option<Value> {
    match (owner, name) {
        (TypeId::OBJECT, "__new__") => Some(builtin_slot_value(&OBJECT_NEW_SLOT_FUNCTION)),
        (TypeId::OBJECT, "__init__") => Some(builtin_slot_value(&OBJECT_INIT_SLOT_FUNCTION)),
        (TypeId::TYPE, "__new__") => Some(builtin_slot_value(&TYPE_NEW_SLOT_FUNCTION)),
        (TypeId::TYPE, "__init__") => Some(builtin_slot_value(&TYPE_INIT_SLOT_FUNCTION)),
        (TypeId::TUPLE, "__new__") => Some(builtin_slot_value(&TUPLE_NEW_SLOT_FUNCTION)),
        (TypeId::LIST, "__new__") => Some(builtin_slot_value(&LIST_NEW_SLOT_FUNCTION)),
        (TypeId::LIST, "__init__") => Some(builtin_slot_value(&LIST_INIT_SLOT_FUNCTION)),
        (TypeId::DICT, "__new__") => Some(builtin_slot_value(&DICT_NEW_SLOT_FUNCTION)),
        (TypeId::DICT, "__init__") => Some(builtin_slot_value(&DICT_INIT_SLOT_FUNCTION)),
        (TypeId::INT, "__new__") => Some(builtin_slot_value(&INT_NEW_SLOT_FUNCTION)),
        (TypeId::FLOAT, "__new__") => Some(builtin_slot_value(&FLOAT_NEW_SLOT_FUNCTION)),
        (TypeId::STR, "__new__") => Some(builtin_slot_value(&STR_NEW_SLOT_FUNCTION)),
        (TypeId::BOOL, "__new__") => Some(builtin_slot_value(&BOOL_NEW_SLOT_FUNCTION)),
        (TypeId::BYTES, "__new__") => Some(builtin_slot_value(&BYTES_NEW_SLOT_FUNCTION)),
        (TypeId::BYTEARRAY, "__new__") => Some(builtin_slot_value(&BYTEARRAY_NEW_SLOT_FUNCTION)),
        (TypeId::ENUMERATE, "__new__") => Some(builtin_slot_value(&ENUMERATE_NEW_SLOT_FUNCTION)),
        _ => None,
    }
}

fn resolve_instantiation_slot(class: &PyClassObject, name: &str) -> Option<Value> {
    let name = intern(name);

    for &class_id in class.mro() {
        if class_id == class.class_id() {
            if let Some(value) = class.get_attr(&name) {
                return Some(value);
            }
            continue;
        }

        if class_id.0 < TypeId::FIRST_USER_TYPE {
            if let Some(value) =
                builtin_instantiation_slot_value(class_id_to_type_id(class_id), name.as_str())
            {
                return Some(value);
            }
            continue;
        }

        if let Some(parent) = global_class(class_id) {
            if let Some(value) = parent.get_attr(&name) {
                return Some(value);
            }
        }
    }

    if class
        .flags()
        .contains(prism_runtime::object::class::ClassFlags::METACLASS)
    {
        return builtin_instantiation_slot_value(TypeId::TYPE, name.as_str());
    }

    None
}

fn instantiate_user_defined_class(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<Value, RuntimeError> {
    reject_abstract_class_instantiation(class)?;

    let class_value = Value::object_ptr(class as *const PyClassObject as *const ());
    let new_result = if let Some(new_callable) = resolve_instantiation_slot(class, "__new__") {
        let new_kwargc = if should_route_keywords_to_init_only(class, new_callable, kwargc != 0) {
            0
        } else {
            kwargc
        };
        invoke_class_new(
            vm,
            class_value,
            new_callable,
            dst_reg,
            posargc,
            new_kwargc,
            kwnames_idx,
        )?
    } else {
        allocate_default_instance(vm, class)?
    };

    let should_run_init = should_run_init_for_new_result(class, new_result);
    let Some(init_callable) =
        resolve_instantiation_slot_for_new_result(class, new_result, "__init__")
    else {
        if !should_run_init {
            return Ok(new_result);
        }
        if posargc != 0 || kwargc != 0 {
            return Err(RuntimeError::type_error(format!(
                "{}() takes no arguments",
                class.name()
            )));
        }
        return Ok(new_result);
    };

    if should_run_init {
        let init_arg_policy = init_arg_policy_for_new_result(class, new_result, init_callable);
        invoke_class_init(
            vm,
            new_result,
            init_callable,
            dst_reg,
            posargc,
            kwargc,
            kwnames_idx,
            init_arg_policy,
        )?;
    }

    Ok(new_result)
}

fn instantiate_user_defined_class_from_values(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    reject_abstract_class_instantiation(class)?;

    let class_value = Value::object_ptr(class as *const PyClassObject as *const ());
    let new_result = if let Some(new_callable) = resolve_instantiation_slot(class, "__new__") {
        invoke_class_new_direct(vm, class_value, new_callable, args)?
    } else {
        allocate_default_instance(vm, class)?
    };

    let should_run_init = should_run_init_for_new_result(class, new_result);
    let Some(init_callable) =
        resolve_instantiation_slot_for_new_result(class, new_result, "__init__")
    else {
        if !should_run_init {
            return Ok(new_result);
        }
        if !args.is_empty() {
            return Err(RuntimeError::type_error(format!(
                "{}() takes no arguments",
                class.name()
            )));
        }
        return Ok(new_result);
    };

    if should_run_init {
        let init_arg_policy = init_arg_policy_for_new_result(class, new_result, init_callable);
        invoke_class_init_direct(vm, new_result, init_callable, args, init_arg_policy)?;
    }

    Ok(new_result)
}

fn instantiate_user_defined_class_from_values_with_keywords(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    reject_abstract_class_instantiation(class)?;

    let class_value = Value::object_ptr(class as *const PyClassObject as *const ());
    let new_result = if let Some(new_callable) = resolve_instantiation_slot(class, "__new__") {
        let new_keywords: &[(&str, Value)] =
            if should_route_keywords_to_init_only(class, new_callable, !keywords.is_empty()) {
                &[]
            } else {
                keywords
            };
        invoke_class_new_direct_with_keywords(vm, class_value, new_callable, args, new_keywords)?
    } else {
        allocate_default_instance(vm, class)?
    };

    let should_run_init = should_run_init_for_new_result(class, new_result);
    let Some(init_callable) =
        resolve_instantiation_slot_for_new_result(class, new_result, "__init__")
    else {
        if !should_run_init {
            return Ok(new_result);
        }
        if !args.is_empty() || !keywords.is_empty() {
            return Err(RuntimeError::type_error(format!(
                "{}() takes no arguments",
                class.name()
            )));
        }
        return Ok(new_result);
    };

    if should_run_init {
        let init_arg_policy = init_arg_policy_for_new_result(class, new_result, init_callable);
        invoke_class_init_direct_with_keywords(
            vm,
            new_result,
            init_callable,
            args,
            keywords,
            init_arg_policy,
        )?;
    }

    Ok(new_result)
}

fn reject_abstract_class_instantiation(class: &PyClassObject) -> Result<(), RuntimeError> {
    let Some(abstracts) = class.get_attr(&intern("__abstractmethods__")) else {
        return Ok(());
    };

    let mut names = abstract_method_names(abstracts);
    if names.is_empty() {
        return Ok(());
    }

    names.sort_unstable();
    let method_word = if names.len() == 1 {
        "method"
    } else {
        "methods"
    };
    let rendered_names = names
        .into_iter()
        .map(|name| format!("'{name}'"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(RuntimeError::type_error(format!(
        "Can't instantiate abstract class {} without an implementation for abstract {method_word} {rendered_names}",
        class.name()
    )))
}

fn abstract_method_names(value: Value) -> Vec<String> {
    let Some(ptr) = value.as_object_ptr() else {
        return Vec::new();
    };

    match extract_type_id(ptr) {
        TypeId::SET | TypeId::FROZENSET => {
            let set = unsafe { &*(ptr as *const SetObject) };
            set.iter().filter_map(abstract_method_name).collect()
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            tuple
                .iter()
                .copied()
                .filter_map(abstract_method_name)
                .collect()
        }
        _ => Vec::new(),
    }
}

fn abstract_method_name(value: Value) -> Option<String> {
    if value.is_string() {
        let ptr = value.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8).map(|name| name.as_str().to_string());
    }

    let ptr = value.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    Some(
        unsafe { &*(ptr as *const StringObject) }
            .as_str()
            .to_string(),
    )
}

#[inline]
fn allocate_default_instance(
    vm: &mut VirtualMachine,
    class: &PyClassObject,
) -> Result<Value, RuntimeError> {
    let instance = crate::builtins::allocate_heap_instance_for_class(class);
    alloc_heap_value(vm, instance, "heap type instance")
}

#[inline]
fn should_run_init_for_new_result(class: &PyClassObject, new_result: Value) -> bool {
    let Some(ptr) = new_result.as_object_ptr() else {
        return false;
    };

    let type_id = extract_type_id(ptr);
    if type_id == class.class_type_id() {
        return true;
    }

    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        return prism_runtime::object::type_builtins::global_class_bitmap(ClassId(type_id.raw()))
            .is_some_and(|bitmap| bitmap.is_subclass_of(class.class_type_id()));
    }

    class
        .flags()
        .contains(prism_runtime::object::class::ClassFlags::METACLASS)
        && type_id == TypeId::TYPE
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum InitArgPolicy {
    ForwardConstructorArgs,
    IgnoreConstructorArgs,
}

fn resolve_instantiation_slot_for_new_result(
    class: &PyClassObject,
    new_result: Value,
    name: &str,
) -> Option<Value> {
    let ptr = new_result.as_object_ptr()?;
    let type_id = extract_type_id(ptr);

    if type_id == class.class_type_id() {
        return resolve_instantiation_slot(class, name);
    }

    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        let runtime_class = global_class(ClassId(type_id.raw()))?;
        return resolve_instantiation_slot(runtime_class.as_ref(), name);
    }

    if class
        .flags()
        .contains(prism_runtime::object::class::ClassFlags::METACLASS)
        && type_id == TypeId::TYPE
    {
        let result_class = unsafe { &*(ptr as *const PyClassObject) };
        let Some(meta_ptr) = result_class.metaclass().as_object_ptr() else {
            return builtin_instantiation_slot_value(TypeId::TYPE, name);
        };

        if builtin_type_object_type_id(meta_ptr) == Some(TypeId::TYPE) {
            return builtin_instantiation_slot_value(TypeId::TYPE, name);
        }

        let meta_class = unsafe { &*(meta_ptr as *const PyClassObject) };
        return resolve_instantiation_slot(meta_class, name)
            .or_else(|| builtin_instantiation_slot_value(TypeId::TYPE, name));
    }

    builtin_instantiation_slot_value(type_id, name)
}

#[inline]
fn init_arg_policy_for_new_result(
    class: &PyClassObject,
    new_result: Value,
    init_callable: Value,
) -> InitArgPolicy {
    if !slot_callable_matches_builtin_name(init_callable, "object.__init__") {
        return InitArgPolicy::ForwardConstructorArgs;
    }

    let Some(runtime_new_callable) =
        resolve_instantiation_slot_for_new_result(class, new_result, "__new__")
    else {
        return InitArgPolicy::ForwardConstructorArgs;
    };

    if slot_callable_matches_builtin_name(runtime_new_callable, "object.__new__") {
        InitArgPolicy::ForwardConstructorArgs
    } else {
        InitArgPolicy::IgnoreConstructorArgs
    }
}

#[inline]
fn should_route_keywords_to_init_only(
    class: &PyClassObject,
    new_callable: Value,
    has_keywords: bool,
) -> bool {
    has_keywords
        && (slot_callable_matches_builtin_name(new_callable, "tuple.__new__")
            || slot_callable_matches_builtin_name(new_callable, "dict.__new__"))
        && resolve_instantiation_slot(class, "__init__")
            .is_some_and(|init| !slot_callable_matches_builtin_name(init, "object.__init__"))
}

#[inline]
fn slot_callable_matches_builtin_name(value: Value, expected: &str) -> bool {
    let Some(callable) = unwrap_slot_callable(value) else {
        return false;
    };
    let Some(ptr) = callable.as_object_ptr() else {
        return false;
    };
    if extract_type_id(ptr) != TypeId::BUILTIN_FUNCTION {
        return false;
    }

    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    builtin.name() == expected
}

#[inline]
fn unwrap_slot_callable(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    match extract_type_id(ptr) {
        TypeId::STATICMETHOD => {
            let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Some(descriptor.function())
        }
        _ => Some(value),
    }
}

fn invoke_class_new(
    vm: &mut VirtualMachine,
    class_value: Value,
    new_callable: Value,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<Value, RuntimeError> {
    let callable =
        crate::ops::objects::resolve_class_attribute_in_vm(vm, new_callable, class_value)?;
    invoke_callable_with_implicit_self(
        vm,
        callable,
        class_value,
        dst_reg,
        posargc,
        kwargc,
        kwnames_idx,
        "__new__",
    )
}

fn invoke_class_new_direct(
    vm: &mut VirtualMachine,
    class_value: Value,
    new_callable: Value,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    let callable =
        crate::ops::objects::resolve_class_attribute_in_vm(vm, new_callable, class_value)?;
    invoke_callable_with_implicit_self_values(vm, callable, class_value, args, "__new__")
}

fn invoke_class_new_direct_with_keywords(
    vm: &mut VirtualMachine,
    class_value: Value,
    new_callable: Value,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    let callable =
        crate::ops::objects::resolve_class_attribute_in_vm(vm, new_callable, class_value)?;
    invoke_callable_with_implicit_self_values_with_keywords(
        vm,
        callable,
        class_value,
        args,
        keywords,
        "__new__",
    )
}

fn invoke_class_init(
    vm: &mut VirtualMachine,
    instance: Value,
    init_callable: Value,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
    init_arg_policy: InitArgPolicy,
) -> Result<(), RuntimeError> {
    let bound_init =
        crate::ops::objects::bind_instance_attribute_in_vm(vm, init_callable, instance)?;
    let init_result = match init_arg_policy {
        InitArgPolicy::ForwardConstructorArgs => invoke_callable_with_implicit_self(
            vm,
            bound_init,
            instance,
            dst_reg,
            posargc,
            kwargc,
            kwnames_idx,
            "__init__",
        )?,
        InitArgPolicy::IgnoreConstructorArgs => {
            invoke_callable_with_implicit_self_values(vm, bound_init, instance, &[], "__init__")?
        }
    };

    if !init_result.is_none() {
        return Err(RuntimeError::type_error(format!(
            "__init__() should return None, not '{}'",
            init_result.type_name()
        )));
    }

    Ok(())
}

fn invoke_class_init_direct(
    vm: &mut VirtualMachine,
    instance: Value,
    init_callable: Value,
    args: &[Value],
    init_arg_policy: InitArgPolicy,
) -> Result<(), RuntimeError> {
    let bound_init =
        crate::ops::objects::bind_instance_attribute_in_vm(vm, init_callable, instance)?;
    let init_result = match init_arg_policy {
        InitArgPolicy::ForwardConstructorArgs => {
            invoke_callable_with_implicit_self_values(vm, bound_init, instance, args, "__init__")?
        }
        InitArgPolicy::IgnoreConstructorArgs => {
            invoke_callable_with_implicit_self_values(vm, bound_init, instance, &[], "__init__")?
        }
    };

    if !init_result.is_none() {
        return Err(RuntimeError::type_error(format!(
            "__init__() should return None, not '{}'",
            init_result.type_name()
        )));
    }

    Ok(())
}

fn invoke_class_init_direct_with_keywords(
    vm: &mut VirtualMachine,
    instance: Value,
    init_callable: Value,
    args: &[Value],
    keywords: &[(&str, Value)],
    init_arg_policy: InitArgPolicy,
) -> Result<(), RuntimeError> {
    let bound_init =
        crate::ops::objects::bind_instance_attribute_in_vm(vm, init_callable, instance)?;
    let init_result = match init_arg_policy {
        InitArgPolicy::ForwardConstructorArgs => {
            invoke_callable_with_implicit_self_values_with_keywords(
                vm, bound_init, instance, args, keywords, "__init__",
            )?
        }
        InitArgPolicy::IgnoreConstructorArgs => {
            invoke_callable_with_implicit_self_values_with_keywords(
                vm,
                bound_init,
                instance,
                &[],
                &[],
                "__init__",
            )?
        }
    };

    if !init_result.is_none() {
        return Err(RuntimeError::type_error(format!(
            "__init__() should return None, not '{}'",
            init_result.type_name()
        )));
    }

    Ok(())
}

fn invoke_callable_with_implicit_self(
    vm: &mut VirtualMachine,
    callable: Value,
    implicit_self: Value,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
    slot_name: &'static str,
) -> Result<Value, RuntimeError> {
    let Some(callable_ptr) = callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "{slot_name} attribute is not callable",
        )));
    };

    match extract_type_id(callable_ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => invoke_user_function_with_implicit_self(
            vm,
            callable_ptr,
            implicit_self,
            dst_reg,
            posargc,
            kwargc,
            kwnames_idx,
        ),
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(callable_ptr as *const BuiltinFunctionObject) };
            if should_ignore_constructor_args_for_builtin_new(builtin, slot_name) {
                return invoke_builtin_with_implicit_self_only(vm, builtin, implicit_self);
            }
            invoke_builtin_with_implicit_self_from_frame(
                vm,
                callable_ptr,
                implicit_self,
                dst_reg,
                posargc,
                kwargc,
                kwnames_idx,
            )
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(callable_ptr as *const BoundMethod) };
            invoke_callable_with_implicit_self(
                vm,
                bound.function(),
                bound.instance(),
                dst_reg,
                posargc,
                kwargc,
                kwnames_idx,
                slot_name,
            )
        }
        _ if value_supports_call_protocol(callable) => {
            let (args, keyword_args) = {
                let caller_frame = &vm.frames[vm.call_depth() - 1];
                let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(posargc + 1);
                args.push(implicit_self);
                for i in 0..posargc {
                    args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
                }

                let keyword_args = if kwargc == 0 {
                    SmallVec::new()
                } else {
                    collect_call_keyword_args(caller_frame, dst_reg, posargc, kwargc, kwnames_idx)?
                };
                (args, keyword_args)
            };

            if keyword_args.is_empty() {
                invoke_callable_value(vm, callable, &args)
            } else {
                let keyword_refs: SmallVec<[(&str, Value); 4]> = keyword_args
                    .iter()
                    .map(|(name, value)| (name.as_ref(), *value))
                    .collect();
                invoke_callable_value_with_keywords(vm, callable, &args, &keyword_refs)
            }
        }
        type_id => Err(RuntimeError::type_error(format!(
            "{slot_name} is not callable (got '{}')",
            type_id.name()
        ))),
    }
}

fn invoke_callable_with_implicit_self_values(
    vm: &mut VirtualMachine,
    callable: Value,
    implicit_self: Value,
    args: &[Value],
    slot_name: &'static str,
) -> Result<Value, RuntimeError> {
    let Some(callable_ptr) = callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "{slot_name} attribute is not callable",
        )));
    };

    match extract_type_id(callable_ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value(vm, callable, &all_args)
        }
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(callable_ptr as *const BuiltinFunctionObject) };
            if should_ignore_constructor_args_for_builtin_new(builtin, slot_name) {
                return invoke_builtin_with_implicit_self_only(vm, builtin, implicit_self);
            }
            let mut all_args: SmallVec<[Value; 8]> =
                SmallVec::with_capacity(args.len() + usize::from(builtin.bound_self().is_none()));
            if builtin.bound_self().is_none() {
                all_args.push(implicit_self);
            }
            all_args.extend_from_slice(args);
            invoke_builtin(vm, builtin, &all_args)
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(callable_ptr as *const BoundMethod) };
            invoke_callable_with_implicit_self_values(
                vm,
                bound.function(),
                bound.instance(),
                args,
                slot_name,
            )
        }
        _ if value_supports_call_protocol(callable) => {
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value(vm, callable, &all_args)
        }
        type_id => Err(RuntimeError::type_error(format!(
            "{slot_name} is not callable (got '{}')",
            type_id.name()
        ))),
    }
}

fn invoke_callable_with_implicit_self_values_with_keywords(
    vm: &mut VirtualMachine,
    callable: Value,
    implicit_self: Value,
    args: &[Value],
    keywords: &[(&str, Value)],
    slot_name: &'static str,
) -> Result<Value, RuntimeError> {
    if keywords.is_empty() {
        return invoke_callable_with_implicit_self_values(
            vm,
            callable,
            implicit_self,
            args,
            slot_name,
        );
    }

    let Some(callable_ptr) = callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "{slot_name} attribute is not callable",
        )));
    };

    match extract_type_id(callable_ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value_with_keywords(vm, callable, &all_args, keywords)
        }
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(callable_ptr as *const BuiltinFunctionObject) };
            if should_ignore_constructor_args_for_builtin_new(builtin, slot_name) {
                return invoke_builtin_with_implicit_self_only(vm, builtin, implicit_self);
            }
            let mut all_args: SmallVec<[Value; 8]> =
                SmallVec::with_capacity(args.len() + usize::from(builtin.bound_self().is_none()));
            if builtin.bound_self().is_none() {
                all_args.push(implicit_self);
            }
            all_args.extend_from_slice(args);
            invoke_builtin_with_keywords(vm, builtin, &all_args, keywords)
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(callable_ptr as *const BoundMethod) };
            invoke_callable_with_implicit_self_values_with_keywords(
                vm,
                bound.function(),
                bound.instance(),
                args,
                keywords,
                slot_name,
            )
        }
        _ if value_supports_call_protocol(callable) => {
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value_with_keywords(vm, callable, &all_args, keywords)
        }
        type_id => Err(RuntimeError::type_error(format!(
            "{slot_name} is not callable (got '{}')",
            type_id.name()
        ))),
    }
}

#[inline]
fn should_ignore_constructor_args_for_builtin_new(
    builtin: &BuiltinFunctionObject,
    slot_name: &str,
) -> bool {
    slot_name == "__new__" && builtin.name() == "object.__new__"
}

#[inline]
fn invoke_builtin_with_implicit_self_only(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    implicit_self: Value,
) -> Result<Value, RuntimeError> {
    if builtin.bound_self().is_none() {
        let args = [implicit_self];
        invoke_builtin(vm, builtin, &args)
    } else {
        invoke_builtin(vm, builtin, &[])
    }
}

fn invoke_builtin_with_implicit_self_from_frame(
    vm: &mut VirtualMachine,
    init_ptr: *const (),
    instance: Value,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<Value, RuntimeError> {
    let builtin = unsafe { &*(init_ptr as *const BuiltinFunctionObject) };
    let (args, keyword_args) = {
        let caller_frame = vm.current_frame();
        let mut args: SmallVec<[Value; 8]> =
            SmallVec::with_capacity(posargc + usize::from(builtin.bound_self().is_none()));
        if builtin.bound_self().is_none() {
            args.push(instance);
        }
        for i in 0..posargc {
            args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
        }
        let keyword_args = if kwargc == 0 {
            SmallVec::new()
        } else {
            collect_call_keyword_args(caller_frame, dst_reg, posargc, kwargc, kwnames_idx)?
        };
        (args, keyword_args)
    };

    if keyword_args.is_empty() {
        return invoke_builtin(vm, builtin, &args);
    }

    let keyword_refs: SmallVec<[(&str, Value); 4]> = keyword_args
        .iter()
        .map(|(name, value)| (name.as_ref(), *value))
        .collect();
    invoke_builtin_with_keywords(vm, builtin, &args, &keyword_refs)
}

fn invoke_user_function_with_implicit_self(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    implicit_self: Value,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<Value, RuntimeError> {
    let func = unsafe { &*(func_ptr as *const FunctionObject) };
    if let Some(result) = function_vectorcall_override_result(func) {
        return Ok(result);
    }
    let code = Arc::clone(&func.code);
    let closure = materialize_function_invocation_closure(func, &code)?;
    let module = resolve_function_module(vm, func);

    let mut bound = {
        let caller_frame = vm.current_frame();
        let positional_args = std::iter::once(implicit_self)
            .chain((0..posargc).map(|i| caller_frame.get_reg(dst_reg + 1 + i as u8)));

        if kwargc == 0 {
            ArgumentBinder::bind(func, positional_args, std::iter::empty())
        } else {
            let kwnames_val = caller_frame.get_const(kwnames_idx);
            let Some(kwnames_ptr) = kwnames_val.as_object_ptr() else {
                return Err(RuntimeError::internal(
                    "Invalid keyword names in constant pool",
                ));
            };
            let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };
            let mut keyword_args: SmallVec<[(&str, Value); 4]> = SmallVec::with_capacity(kwargc);
            for i in 0..kwargc {
                let kw_name = kwnames
                    .get(i)
                    .ok_or_else(|| RuntimeError::internal("Invalid keyword names tuple"))?;
                let kw_val = caller_frame.get_reg(dst_reg + 1 + posargc as u8 + i as u8);
                keyword_args.push((kw_name, kw_val));
            }
            ArgumentBinder::bind(func, positional_args, keyword_args.into_iter())
        }
    }
    .map_err(|err| RuntimeError::type_error(err.to_error_message()))?;

    let (varargs_value, varkw_value) = allocate_bound_variadics(vm, &mut bound)?;

    if is_generator_code(&code) {
        return generator_value_from_bound_arguments(
            vm,
            &code,
            function_module_ptr(vm, func),
            closure,
            &bound,
            varargs_value,
            varkw_value,
        );
    }

    let stop_depth = vm.call_depth();
    vm.push_frame_with_closure_and_module(Arc::clone(&code), dst_reg, closure, module)?;

    let initialized_local_count = {
        let new_frame = vm.current_frame_mut();
        write_bound_arguments_to_frame(
            new_frame,
            &bound,
            code.arg_count as usize,
            varargs_value,
            varkw_value,
        )
    };
    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), initialized_local_count);
    if vm.dispatch_prepared_current_frame_via_jit()? {
        return Ok(vm.current_frame().get_reg(dst_reg));
    }
    let target_frame_id = vm.current_frame_id();
    match vm.execute_until_target_frame_returns_with_outcome(stop_depth, target_frame_id)? {
        NestedTargetFrameOutcome::Returned(value) => Ok(value),
        NestedTargetFrameOutcome::ControlTransferred => Err(RuntimeError::control_transferred()),
    }
}

fn allocate_bound_variadics(
    vm: &mut VirtualMachine,
    bound: &mut BoundArguments,
) -> Result<(Option<Value>, Option<Value>), RuntimeError> {
    let varargs_value = match bound.varargs.take() {
        Some(tuple) => Some(alloc_heap_value(vm, *tuple, "bound varargs tuple")?),
        None => None,
    };
    let varkw_value = match bound.varkw.take() {
        Some(dict) => Some(alloc_heap_value(vm, *dict, "bound kwargs dict")?),
        None => None,
    };
    Ok((varargs_value, varkw_value))
}

fn write_bound_arguments_to_frame(
    frame: &mut crate::frame::Frame,
    bound: &BoundArguments,
    arg_count: usize,
    varargs_value: Option<Value>,
    varkw_value: Option<Value>,
) -> usize {
    let mut local_idx = 0usize;

    for i in 0..arg_count {
        frame.set_local(local_idx as u16, bound.parameters[i]);
        local_idx += 1;
    }

    if let Some(tuple_val) = varargs_value {
        frame.set_local(local_idx as u16, tuple_val);
        local_idx += 1;
    }

    for i in arg_count..bound.parameters.len() {
        frame.set_local(local_idx as u16, bound.parameters[i]);
        local_idx += 1;
    }

    if let Some(dict_val) = varkw_value {
        frame.set_local(local_idx as u16, dict_val);
        local_idx += 1;
    }

    local_idx
}

fn create_generator_from_bound_arguments(
    vm: &mut VirtualMachine,
    code: &Arc<CodeObject>,
    dst_reg: u8,
    module_ptr: *const (),
    closure: Option<Arc<ClosureEnv>>,
    bound: &BoundArguments,
    varargs_value: Option<Value>,
    varkw_value: Option<Value>,
) -> ControlFlow {
    let value = match generator_value_from_bound_arguments(
        vm,
        code,
        module_ptr,
        closure,
        bound,
        varargs_value,
        varkw_value,
    ) {
        Ok(value) => value,
        Err(err) => return ControlFlow::Error(err),
    };
    vm.current_frame_mut().set_reg(dst_reg, value);
    ControlFlow::Continue
}

fn generator_value_from_bound_arguments(
    vm: &mut VirtualMachine,
    code: &Arc<CodeObject>,
    module_ptr: *const (),
    closure: Option<Arc<ClosureEnv>>,
    bound: &BoundArguments,
    varargs_value: Option<Value>,
    varkw_value: Option<Value>,
) -> Result<Value, RuntimeError> {
    let arg_count = code.arg_count as usize;
    let mut locals = [Value::none(); 256];
    let mut local_idx = 0usize;

    for i in 0..arg_count {
        locals[local_idx] = bound.parameters[i];
        local_idx += 1;
    }

    if let Some(tuple_val) = varargs_value {
        locals[local_idx] = tuple_val;
        local_idx += 1;
    }

    for i in arg_count..bound.parameters.len() {
        locals[local_idx] = bound.parameters[i];
        local_idx += 1;
    }

    if let Some(dict_val) = varkw_value {
        locals[local_idx] = dict_val;
        local_idx += 1;
    }

    let mut generator = GeneratorObject::from_code(Arc::clone(code));
    generator.set_module_ptr(module_ptr);
    if let Some(closure) = closure {
        generator.set_closure(closure);
    }
    generator.seed_locals(&locals, prefix_liveness(local_idx));
    alloc_heap_value(vm, generator, "generator object")
}

fn create_generator_from_values(
    vm: &mut VirtualMachine,
    code: &Arc<CodeObject>,
    dst_reg: u8,
    module_ptr: *const (),
    closure: Option<Arc<ClosureEnv>>,
    args: &[Value],
) -> ControlFlow {
    let mut locals = [Value::none(); 256];
    for (i, arg) in args.iter().copied().enumerate() {
        locals[i] = arg;
    }

    let mut generator = GeneratorObject::from_code(Arc::clone(code));
    generator.set_module_ptr(module_ptr);
    if let Some(closure) = closure {
        generator.set_closure(closure);
    }
    generator.seed_locals(&locals, prefix_liveness(args.len()));
    let value = match alloc_heap_value(vm, generator, "generator object") {
        Ok(value) => value,
        Err(err) => return ControlFlow::Error(err),
    };
    vm.current_frame_mut().set_reg(dst_reg, value);
    ControlFlow::Continue
}

pub(crate) fn call_user_function_from_values(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    dst_reg: u8,
    positional_args: &[Value],
    keyword_args: &[(&str, Value)],
) -> ControlFlow {
    let func = unsafe { &*(func_ptr as *const FunctionObject) };
    if let Some(result) = function_vectorcall_override_result(func) {
        vm.current_frame_mut().set_reg(dst_reg, result);
        return ControlFlow::Continue;
    }
    let code = Arc::clone(&func.code);
    let module_ptr = function_module_ptr(vm, func);
    let closure = match materialize_function_invocation_closure(func, &code) {
        Ok(closure) => closure,
        Err(err) => return ControlFlow::Error(err),
    };
    let is_generator_function = is_generator_code(&code);
    let defaults_empty = func.defaults.as_ref().is_none_or(|d| d.is_empty());
    let kwdefaults_empty = func.kwdefaults.as_ref().is_none_or(|d| d.is_empty());
    let simple_positional = keyword_args.is_empty()
        && !code.flags.contains(CodeFlags::VARARGS)
        && !code.flags.contains(CodeFlags::VARKEYWORDS)
        && code.kwonlyarg_count == 0
        && defaults_empty
        && kwdefaults_empty
        && positional_args.len() == code.arg_count as usize;

    if is_generator_function && simple_positional {
        return create_generator_from_values(
            vm,
            &code,
            dst_reg,
            module_ptr,
            closure.clone(),
            positional_args,
        );
    }

    if simple_positional {
        let module = resolve_function_module(vm, func);
        if let Err(err) =
            vm.push_frame_with_closure_and_module(Arc::clone(&code), dst_reg, closure, module)
        {
            return ControlFlow::Error(err);
        }

        for (i, arg) in positional_args.iter().copied().enumerate() {
            vm.current_frame_mut().set_local(i as u16, arg);
        }

        initialize_closure_cellvars_from_locals(vm.current_frame_mut(), positional_args.len());
        if let Err(err) = vm.dispatch_prepared_current_frame_via_jit() {
            return ControlFlow::Error(err);
        }
        return ControlFlow::Continue;
    }

    let mut bound = match ArgumentBinder::bind(
        func,
        positional_args.iter().copied(),
        keyword_args.iter().copied(),
    ) {
        Ok(bound) => bound,
        Err(err) => {
            return ControlFlow::Error(RuntimeError::type_error(err.to_error_message()));
        }
    };

    let (varargs_value, varkw_value) = match allocate_bound_variadics(vm, &mut bound) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    if is_generator_function {
        return create_generator_from_bound_arguments(
            vm,
            &code,
            dst_reg,
            module_ptr,
            closure.clone(),
            &bound,
            varargs_value,
            varkw_value,
        );
    }

    let module = resolve_function_module(vm, func);
    if let Err(err) =
        vm.push_frame_with_closure_and_module(Arc::clone(&code), dst_reg, closure, module)
    {
        return ControlFlow::Error(err);
    }

    let initialized_local_count = {
        let frame = vm.current_frame_mut();
        write_bound_arguments_to_frame(
            frame,
            &bound,
            code.arg_count as usize,
            varargs_value,
            varkw_value,
        )
    };
    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), initialized_local_count);
    if let Err(err) = vm.dispatch_prepared_current_frame_via_jit() {
        return ControlFlow::Error(err);
    }
    ControlFlow::Continue
}

fn invoke_user_function_direct(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    args: &[Value],
) -> Result<Value, RuntimeError> {
    match invoke_user_function_direct_impl(vm, func_ptr, args, false)? {
        InvokeCallableOutcome::Returned(value) => Ok(value),
        InvokeCallableOutcome::ControlTransferred => Err(RuntimeError::control_transferred()),
    }
}

fn invoke_user_function_direct_impl(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    args: &[Value],
    allow_control_transfer: bool,
) -> Result<InvokeCallableOutcome, RuntimeError> {
    vm.with_native_recursion_guard(|vm| {
        invoke_user_function_direct_impl_guarded(vm, func_ptr, args, allow_control_transfer)
    })
}

fn invoke_user_function_direct_impl_guarded(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    args: &[Value],
    allow_control_transfer: bool,
) -> Result<InvokeCallableOutcome, RuntimeError> {
    let func = unsafe { &*(func_ptr as *const FunctionObject) };
    if let Some(result) = function_vectorcall_override_result(func) {
        return Ok(InvokeCallableOutcome::Returned(result));
    }
    let code = Arc::clone(&func.code);
    let closure = materialize_function_invocation_closure(func, &code)?;
    let module = resolve_function_module(vm, func);

    let mut bound = ArgumentBinder::bind(func, args.iter().copied(), std::iter::empty())
        .map_err(|err| RuntimeError::type_error(err.to_error_message()))?;

    if vm.call_depth() == 0 {
        return Err(RuntimeError::internal(
            "direct callable invocation requires an active caller frame",
        ));
    }

    let saved_caller_register = vm.current_frame().snapshot_register(DIRECT_CALL_RETURN_REG);
    let saved_exception_context = vm.capture_exception_context();
    let arg_count = code.arg_count as usize;
    let (varargs_value, varkw_value) = allocate_bound_variadics(vm, &mut bound)?;

    if is_generator_code(&code) {
        let result = match create_generator_from_bound_arguments(
            vm,
            &code,
            DIRECT_CALL_RETURN_REG,
            function_module_ptr(vm, func),
            closure.clone(),
            &bound,
            varargs_value,
            varkw_value,
        ) {
            ControlFlow::Continue => Ok(InvokeCallableOutcome::Returned(
                vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG),
            )),
            ControlFlow::Error(err) => Err(err),
            ControlFlow::Jump(_) => Err(RuntimeError::internal(
                "generator direct call produced unexpected jump control flow",
            )),
            _ => Err(RuntimeError::internal(
                "generator direct call produced unexpected jump control flow",
            )),
        };
        restore_direct_call_caller_state(
            vm,
            vm.call_depth(),
            saved_caller_register,
            saved_exception_context,
        );
        return result;
    }

    let stop_depth = vm.call_depth();
    vm.push_frame_with_closure_and_module(
        Arc::clone(&code),
        DIRECT_CALL_RETURN_REG,
        closure,
        module,
    )?;

    let initialized_local_count = {
        let new_frame = vm.current_frame_mut();
        write_bound_arguments_to_frame(new_frame, &bound, arg_count, varargs_value, varkw_value)
    };

    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), initialized_local_count);
    vm.dispatch_prepared_current_frame_via_jit()?;
    let result = if vm.call_depth() == stop_depth {
        Ok(InvokeCallableOutcome::Returned(
            vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG),
        ))
    } else {
        let target_frame_id = vm.current_frame_id();
        match vm.execute_until_target_frame_returns_with_outcome(stop_depth, target_frame_id)? {
            NestedTargetFrameOutcome::Returned(value) => Ok(InvokeCallableOutcome::Returned(value)),
            NestedTargetFrameOutcome::ControlTransferred if allow_control_transfer => {
                Ok(InvokeCallableOutcome::ControlTransferred)
            }
            NestedTargetFrameOutcome::ControlTransferred => {
                Err(RuntimeError::control_transferred())
            }
        }
    };
    if should_restore_direct_call_caller_state(vm, stop_depth, &result) {
        restore_direct_call_caller_state(
            vm,
            stop_depth,
            saved_caller_register,
            saved_exception_context,
        );
    }

    result
}

fn invoke_user_function_direct_with_keywords(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    vm.with_native_recursion_guard(|vm| {
        invoke_user_function_direct_with_keywords_guarded(vm, func_ptr, args, keywords)
    })
}

fn invoke_user_function_direct_with_keywords_guarded(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    let func = unsafe { &*(func_ptr as *const FunctionObject) };
    if let Some(result) = function_vectorcall_override_result(func) {
        return Ok(result);
    }
    let code = Arc::clone(&func.code);
    let closure = materialize_function_invocation_closure(func, &code)?;
    let module = resolve_function_module(vm, func);

    let mut bound = ArgumentBinder::bind(func, args.iter().copied(), keywords.iter().copied())
        .map_err(|err| RuntimeError::type_error(err.to_error_message()))?;

    if vm.call_depth() == 0 {
        return Err(RuntimeError::internal(
            "direct callable invocation requires an active caller frame",
        ));
    }

    let saved_caller_register = vm.current_frame().snapshot_register(DIRECT_CALL_RETURN_REG);
    let saved_exception_context = vm.capture_exception_context();
    let arg_count = code.arg_count as usize;
    let (varargs_value, varkw_value) = allocate_bound_variadics(vm, &mut bound)?;

    if is_generator_code(&code) {
        let result = match create_generator_from_bound_arguments(
            vm,
            &code,
            DIRECT_CALL_RETURN_REG,
            function_module_ptr(vm, func),
            closure.clone(),
            &bound,
            varargs_value,
            varkw_value,
        ) {
            ControlFlow::Continue => Ok(vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG)),
            ControlFlow::Error(err) => Err(err),
            ControlFlow::Jump(_) => Err(RuntimeError::internal(
                "generator direct call produced unexpected jump control flow",
            )),
            _ => Err(RuntimeError::internal(
                "generator direct call produced unexpected jump control flow",
            )),
        };
        restore_direct_call_caller_state(
            vm,
            vm.call_depth(),
            saved_caller_register,
            saved_exception_context,
        );
        return result;
    }

    let stop_depth = vm.call_depth();
    vm.push_frame_with_closure_and_module(
        Arc::clone(&code),
        DIRECT_CALL_RETURN_REG,
        closure,
        module,
    )?;

    let initialized_local_count = {
        let new_frame = vm.current_frame_mut();
        write_bound_arguments_to_frame(new_frame, &bound, arg_count, varargs_value, varkw_value)
    };

    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), initialized_local_count);
    vm.dispatch_prepared_current_frame_via_jit()?;
    let result = if vm.call_depth() == stop_depth {
        Ok(vm.current_frame().get_reg(DIRECT_CALL_RETURN_REG))
    } else {
        let target_frame_id = vm.current_frame_id();
        vm.execute_until_target_frame_returns(stop_depth, target_frame_id)
    };
    if should_restore_direct_call_caller_state_for_value_result(vm, stop_depth, &result) {
        restore_direct_call_caller_state(
            vm,
            stop_depth,
            saved_caller_register,
            saved_exception_context,
        );
    }

    result
}

fn invoke_callable_value_impl(
    vm: &mut VirtualMachine,
    callable: Value,
    args: &[Value],
    allow_control_transfer: bool,
) -> Result<InvokeCallableOutcome, RuntimeError> {
    let Some(ptr) = callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            callable.type_name()
        )));
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            invoke_builtin(vm, builtin, args).map(InvokeCallableOutcome::Returned)
        }
        _ if extract_type_id(ptr) == EXCEPTION_TYPE_ID => {
            let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
            exc_type
                .call_in_vm(vm, args)
                .map(InvokeCallableOutcome::Returned)
        }
        TypeId::TYPE => {
            if let Some(represented_type) = builtin_type_object_type_id(ptr) {
                call_builtin_type_with_vm(vm, represented_type, args)
                    .map(InvokeCallableOutcome::Returned)
                    .map_err(RuntimeError::from)
            } else if let Some(class) = class_object_from_type_ptr(ptr) {
                instantiate_user_defined_class_from_values(vm, class, args)
                    .map(InvokeCallableOutcome::Returned)
            } else {
                Err(RuntimeError::type_error("type object is not callable"))
            }
        }
        TypeId::FUNCTION | TypeId::CLOSURE => {
            invoke_user_function_direct_impl(vm, ptr, args, allow_control_transfer)
        }
        TypeId::STATICMETHOD => {
            let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            invoke_callable_value_impl(vm, descriptor.function(), args, allow_control_transfer)
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(ptr as *const BoundMethod) };
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(bound.instance());
            all_args.extend_from_slice(args);
            invoke_callable_value_impl(vm, bound.function(), &all_args, allow_control_transfer)
        }
        TypeId::WRAPPER_DESCRIPTOR | TypeId::METHOD_DESCRIPTOR | TypeId::CLASSMETHOD_DESCRIPTOR => {
            let Some(target) = reflected_descriptor_callable_target(callable) else {
                return Err(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    extract_type_id(ptr).name()
                )));
            };
            invoke_callable_value_impl(vm, target, args, allow_control_transfer)
        }
        _ => {
            let Some(resolved) = resolve_dunder_call_target(callable) else {
                return Err(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    extract_type_id(ptr).name()
                )));
            };

            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(resolved.implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value_impl(vm, resolved.callable, &all_args, allow_control_transfer)
        }
    }
}

pub(crate) fn invoke_callable_value(
    vm: &mut VirtualMachine,
    callable: Value,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    match invoke_callable_value_impl(vm, callable, args, false)? {
        InvokeCallableOutcome::Returned(value) => Ok(value),
        InvokeCallableOutcome::ControlTransferred => Err(RuntimeError::control_transferred()),
    }
}

pub(crate) fn invoke_callable_value_with_control_transfer(
    vm: &mut VirtualMachine,
    callable: Value,
    args: &[Value],
) -> Result<InvokeCallableOutcome, RuntimeError> {
    invoke_callable_value_impl(vm, callable, args, true)
}

pub(crate) fn invoke_callable_value_with_keywords(
    vm: &mut VirtualMachine,
    callable: Value,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if keywords.is_empty() {
        return invoke_callable_value(vm, callable, args);
    }

    let Some(ptr) = callable.as_object_ptr() else {
        return Err(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            callable.type_name()
        )));
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            invoke_builtin_with_keywords(vm, builtin, args, keywords)
        }
        _ if extract_type_id(ptr) == EXCEPTION_TYPE_ID => Err(RuntimeError::type_error(
            "keyword arguments for exception constructors via **kwargs are not implemented yet",
        )),
        TypeId::TYPE => {
            if let Some(represented_type) = builtin_type_object_type_id(ptr) {
                call_builtin_type_kw_with_vm(vm, represented_type, args, keywords)
                    .map_err(RuntimeError::from)
            } else if let Some(class) = class_object_from_type_ptr(ptr) {
                instantiate_user_defined_class_from_values_with_keywords(vm, class, args, keywords)
            } else {
                Err(RuntimeError::type_error(
                    "keyword arguments for unpacked heap-type calls are not implemented yet",
                ))
            }
        }
        TypeId::FUNCTION | TypeId::CLOSURE => {
            invoke_user_function_direct_with_keywords(vm, ptr, args, keywords)
        }
        TypeId::STATICMETHOD => {
            let descriptor = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            invoke_callable_value_with_keywords(vm, descriptor.function(), args, keywords)
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(ptr as *const BoundMethod) };
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(bound.instance());
            all_args.extend_from_slice(args);
            invoke_callable_value_with_keywords(vm, bound.function(), &all_args, keywords)
        }
        TypeId::WRAPPER_DESCRIPTOR | TypeId::METHOD_DESCRIPTOR | TypeId::CLASSMETHOD_DESCRIPTOR => {
            let Some(target) = reflected_descriptor_callable_target(callable) else {
                return Err(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    extract_type_id(ptr).name()
                )));
            };
            invoke_callable_value_with_keywords(vm, target, args, keywords)
        }
        _ => {
            let Some(resolved) = resolve_dunder_call_target(callable) else {
                return Err(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    extract_type_id(ptr).name()
                )));
            };

            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(resolved.implicit_self);
            all_args.extend_from_slice(args);
            invoke_callable_value_with_keywords(vm, resolved.callable, &all_args, keywords)
        }
    }
}

pub(crate) fn call_callable_value_with_keywords_from_values(
    vm: &mut VirtualMachine,
    callable: Value,
    dst_reg: u8,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> ControlFlow {
    let Some(ptr) = callable.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::type_error(format!(
            "'{}' object is not callable",
            callable.type_name()
        )));
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            call_user_function_from_values(vm, ptr, dst_reg, args, keywords)
        }
        TypeId::METHOD => {
            let bound = unsafe { &*(ptr as *const BoundMethod) };
            let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
            all_args.push(bound.instance());
            all_args.extend_from_slice(args);
            call_callable_value_with_keywords_from_values(
                vm,
                bound.function(),
                dst_reg,
                &all_args,
                keywords,
            )
        }
        _ => {
            if let Some(resolved) = resolve_dunder_call_target(callable) {
                let mut all_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(args.len() + 1);
                all_args.push(resolved.implicit_self);
                all_args.extend_from_slice(args);
                return call_callable_value_with_keywords_from_values(
                    vm,
                    resolved.callable,
                    dst_reg,
                    &all_args,
                    keywords,
                );
            }

            match invoke_callable_value_with_keywords(vm, callable, args, keywords) {
                Ok(result) => {
                    vm.current_frame_mut().set_reg(dst_reg, result);
                    ControlFlow::Continue
                }
                Err(err) => ControlFlow::Error(err),
            }
        }
    }
}

pub(crate) fn invoke_builtin(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
) -> Result<Value, RuntimeError> {
    if builtin.name() == "next" {
        if args.is_empty() || args.len() > 2 {
            return Err(RuntimeError::type_error(format!(
                "next() expected 1 or 2 arguments, got {}",
                args.len()
            )));
        }

        let default = args.get(1).copied();
        return match next_step(vm, args[0])? {
            IterStep::Yielded(value) => Ok(value),
            IterStep::Exhausted => default.ok_or_else(RuntimeError::stop_iteration),
        };
    }

    if builtin.name() == "globals" {
        if !args.is_empty() {
            return Err(RuntimeError::type_error(format!(
                "globals() takes no arguments ({} given)",
                args.len()
            )));
        }
        return current_globals_value(vm);
    }

    if builtin.name() == "locals" {
        if !args.is_empty() {
            return Err(RuntimeError::type_error(format!(
                "locals() takes no arguments ({} given)",
                args.len()
            )));
        }
        return current_locals_value(vm);
    }

    builtin.call_with_vm(vm, args).map_err(RuntimeError::from)
}

fn collect_call_keyword_args(
    caller_frame: &crate::frame::Frame,
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<SmallVec<[(Arc<str>, Value); 4]>, RuntimeError> {
    let kwnames_val = caller_frame.get_const(kwnames_idx);
    let Some(kwnames_ptr) = kwnames_val.as_object_ptr() else {
        return Err(RuntimeError::internal(
            "Invalid keyword names in constant pool",
        ));
    };
    let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };

    let mut keyword_args: SmallVec<[(Arc<str>, Value); 4]> = SmallVec::with_capacity(kwargc);
    for i in 0..kwargc {
        let kw_name = kwnames
            .get(i)
            .ok_or_else(|| RuntimeError::internal("Invalid keyword names tuple"))?;
        let kw_val = caller_frame.get_reg(dst_reg + 1 + posargc as u8 + i as u8);
        keyword_args.push((Arc::clone(kw_name), kw_val));
    }

    Ok(keyword_args)
}

fn invoke_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if keywords.is_empty() {
        return invoke_builtin(vm, builtin, args);
    }

    match builtin.name() {
        "sorted" => invoke_sorted_builtin_with_keywords(vm, builtin, args, keywords),
        "__import__" => invoke_import_builtin_with_keywords(vm, builtin, args, keywords),
        "collections.namedtuple" => {
            invoke_namedtuple_builtin_with_keywords(vm, builtin, args, keywords)
        }
        "list.__init__" => invoke_list_init_builtin_with_keywords(vm, builtin, args, keywords),
        "list.__new__" => invoke_builtin(vm, builtin, args),
        "tuple.__new__" => Err(RuntimeError::type_error(
            "tuple() takes no keyword arguments",
        )),
        "type.__init__" => invoke_type_init_builtin_with_keywords(builtin, args, keywords),
        "type.__new__" => invoke_type_new_builtin_with_keywords(vm, builtin, args, keywords),
        _ if builtin.accepts_keywords() => builtin
            .call_with_vm_and_keywords(vm, args, keywords)
            .map_err(RuntimeError::from),
        name => Err(RuntimeError::type_error(format!(
            "keyword arguments for builtin '{name}' are not implemented yet",
        ))),
    }
}

fn invoke_sorted_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if args.is_empty() || args.len() > 3 {
        return invoke_builtin(vm, builtin, args);
    }

    let mut key_arg = args.get(1).copied();
    let mut reverse_arg = args.get(2).copied();

    for &(name, value) in keywords {
        match name {
            "key" => {
                if key_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "sorted() got multiple values for argument 'key'",
                    ));
                }
                key_arg = Some(value);
            }
            "reverse" => {
                if reverse_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "sorted() got multiple values for argument 'reverse'",
                    ));
                }
                reverse_arg = Some(value);
            }
            _ => {
                return Err(RuntimeError::type_error(format!(
                    "sorted() got an unexpected keyword argument '{name}'",
                )));
            }
        }
    }

    let mut call_args: SmallVec<[Value; 3]> = SmallVec::with_capacity(3);
    call_args.push(args[0]);
    if key_arg.is_some() || reverse_arg.is_some() {
        call_args.push(key_arg.unwrap_or(Value::none()));
    }
    if let Some(reverse_arg) = reverse_arg {
        call_args.push(reverse_arg);
    }

    invoke_builtin(vm, builtin, &call_args)
}

fn invoke_import_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if args.len() > 5 {
        return invoke_builtin(vm, builtin, args);
    }

    let mut name_arg = args.first().copied();
    let mut globals_arg = args.get(1).copied();
    let mut locals_arg = args.get(2).copied();
    let mut fromlist_arg = args.get(3).copied();
    let mut level_arg = args.get(4).copied();

    for &(name, value) in keywords {
        match name {
            "name" => {
                if name_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "__import__() got multiple values for argument 'name'",
                    ));
                }
                name_arg = Some(value);
            }
            "globals" => {
                if globals_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "__import__() got multiple values for argument 'globals'",
                    ));
                }
                globals_arg = Some(value);
            }
            "locals" => {
                if locals_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "__import__() got multiple values for argument 'locals'",
                    ));
                }
                locals_arg = Some(value);
            }
            "fromlist" => {
                if fromlist_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "__import__() got multiple values for argument 'fromlist'",
                    ));
                }
                fromlist_arg = Some(value);
            }
            "level" => {
                if level_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "__import__() got multiple values for argument 'level'",
                    ));
                }
                level_arg = Some(value);
            }
            _ => {
                return Err(RuntimeError::type_error(format!(
                    "__import__() got an unexpected keyword argument '{name}'",
                )));
            }
        }
    }

    let mut call_args: SmallVec<[Value; 5]> = SmallVec::with_capacity(5);
    if let Some(name_arg) = name_arg {
        call_args.push(name_arg);
    }
    if globals_arg.is_some()
        || locals_arg.is_some()
        || fromlist_arg.is_some()
        || level_arg.is_some()
    {
        call_args.push(globals_arg.unwrap_or(Value::none()));
    }
    if locals_arg.is_some() || fromlist_arg.is_some() || level_arg.is_some() {
        call_args.push(locals_arg.unwrap_or(Value::none()));
    }
    if fromlist_arg.is_some() || level_arg.is_some() {
        call_args.push(fromlist_arg.unwrap_or(Value::none()));
    }
    if let Some(level_arg) = level_arg {
        call_args.push(level_arg);
    }

    invoke_builtin(vm, builtin, &call_args)
}

fn invoke_list_init_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    _keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if list_init_rejects_keywords(args.first().copied()) {
        return Err(RuntimeError::type_error(
            "list() takes no keyword arguments",
        ));
    }

    invoke_builtin(vm, builtin, args)
}

fn list_init_rejects_keywords(receiver: Option<Value>) -> bool {
    let Some(receiver) = receiver else {
        return false;
    };
    let Some(ptr) = receiver.as_object_ptr() else {
        return false;
    };

    let type_id = extract_type_id(ptr);
    if type_id == TypeId::LIST {
        return true;
    }
    if type_id.raw() < TypeId::FIRST_USER_TYPE {
        return true;
    }

    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return true;
    };
    if !class
        .mro()
        .iter()
        .any(|&class_id| class_id_to_type_id(class_id) == TypeId::LIST)
    {
        return true;
    }

    resolve_instantiation_slot(class.as_ref(), "__new__")
        .is_none_or(|new| slot_callable_matches_builtin_name(new, "list.__new__"))
}

fn invoke_namedtuple_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if args.len() < 2 || args.len() > 5 {
        return invoke_builtin(vm, builtin, args);
    }

    let mut rename_arg = args.get(2).copied();
    let mut defaults_arg = args.get(3).copied();
    let mut module_arg = args.get(4).copied();

    for &(name, value) in keywords {
        match name {
            "rename" => {
                if rename_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "namedtuple() got multiple values for argument 'rename'",
                    ));
                }
                rename_arg = Some(value);
            }
            "defaults" => {
                if defaults_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "namedtuple() got multiple values for argument 'defaults'",
                    ));
                }
                defaults_arg = Some(value);
            }
            "module" => {
                if module_arg.is_some() {
                    return Err(RuntimeError::type_error(
                        "namedtuple() got multiple values for argument 'module'",
                    ));
                }
                module_arg = Some(value);
            }
            _ => {
                return Err(RuntimeError::type_error(format!(
                    "namedtuple() got an unexpected keyword argument '{name}'",
                )));
            }
        }
    }

    let mut call_args: SmallVec<[Value; 5]> = SmallVec::with_capacity(5);
    call_args.push(args[0]);
    call_args.push(args[1]);

    if rename_arg.is_some() || defaults_arg.is_some() || module_arg.is_some() {
        call_args.push(rename_arg.unwrap_or(Value::bool(false)));
    }
    if defaults_arg.is_some() || module_arg.is_some() {
        call_args.push(defaults_arg.unwrap_or(Value::none()));
    }
    if let Some(module_arg) = module_arg {
        call_args.push(module_arg);
    }

    invoke_builtin(vm, builtin, &call_args)
}

fn invoke_type_init_builtin_with_keywords(
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if keywords.is_empty() {
        return builtin.call(args).map_err(RuntimeError::from);
    }

    let full_args = prepend_bound_self_if_present(builtin, args);
    if full_args.len() == 1 {
        return Err(RuntimeError::type_error(
            "type.__init__() takes no keyword arguments",
        ));
    }

    crate::builtins::builtin_type_init(&full_args).map_err(RuntimeError::from)
}

fn invoke_type_new_builtin_with_keywords(
    vm: &mut VirtualMachine,
    builtin: &BuiltinFunctionObject,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, RuntimeError> {
    if keywords.is_empty() {
        return builtin.call_with_vm(vm, args).map_err(RuntimeError::from);
    }

    let full_args = prepend_bound_self_if_present(builtin, args);
    crate::builtins::builtin_type_new_with_keywords_vm(vm, &full_args, keywords)
        .map_err(RuntimeError::from)
}

#[inline]
fn prepend_bound_self_if_present(
    builtin: &BuiltinFunctionObject,
    args: &[Value],
) -> SmallVec<[Value; 8]> {
    let mut full_args =
        SmallVec::with_capacity(args.len() + usize::from(builtin.bound_self().is_some()));
    if let Some(bound_self) = builtin.bound_self() {
        full_args.push(bound_self);
    }
    full_args.extend_from_slice(args);
    full_args
}

// =============================================================================
// Function Calls
// =============================================================================

/// Call: dst = func(args...)
/// src1 = function, src2 = argc, args in r(dst+1)..r(dst+argc)
///
/// Dispatches to the appropriate call handler based on the function type.
/// Uses O(1) type discrimination via ObjectHeader for fast dispatch.
#[inline(always)]
pub fn call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let func_val = vm.current_frame().get_reg(inst.src1().0);
    let argc = inst.src2().0 as usize;
    let dst_reg = inst.dst().0;

    // Check if this is a callable object
    if let Some(ptr) = func_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::BUILTIN_FUNCTION => {
                // Fast path: builtin function - call directly without frame push
                let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

                // Collect arguments from registers
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                // Call the builtin function
                match invoke_builtin(vm, builtin, &args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            _ if type_id == EXCEPTION_TYPE_ID => {
                // Exception type object - call to construct an exception instance
                let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };

                // Collect arguments from registers
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                // Call the exception type's call method
                match exc_type.call(&args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(RuntimeError::type_error(e.to_string())),
                }
            }
            TypeId::TYPE => {
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                let result = if let Some(represented_type) = builtin_type_object_type_id(ptr) {
                    call_builtin_type_with_vm(vm, represented_type, &args)
                        .map_err(RuntimeError::from)
                } else if let Some(class) = class_object_from_type_ptr(ptr) {
                    instantiate_user_defined_class(vm, class, dst_reg, argc, 0, 0)
                } else {
                    Err(RuntimeError::type_error("type object is not callable"))
                };

                match result {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                if let Some(result) = function_vectorcall_override_result(func) {
                    vm.current_frame_mut().set_reg(dst_reg, result);
                    return ControlFlow::Continue;
                }
                let code = &func.code;
                let closure = match materialize_function_invocation_closure(func, code) {
                    Ok(closure) => closure,
                    Err(err) => return ControlFlow::Error(err),
                };
                let is_generator_function = is_generator_code(code);

                // Fast path for exact-arity positional calls without advanced binding.
                let defaults_empty = func.defaults.as_ref().is_none_or(|d| d.is_empty());
                let kwdefaults_empty = func.kwdefaults.as_ref().is_none_or(|d| d.is_empty());
                let simple_positional = !code.flags.contains(CodeFlags::VARARGS)
                    && !code.flags.contains(CodeFlags::VARKEYWORDS)
                    && code.kwonlyarg_count == 0
                    && defaults_empty
                    && kwdefaults_empty
                    && argc == code.arg_count as usize;

                if is_generator_function && simple_positional {
                    create_generator_from_simple_call(
                        vm,
                        code,
                        dst_reg,
                        argc,
                        function_module_ptr(vm, func),
                        closure.clone(),
                    )
                } else if simple_positional {
                    let module = resolve_function_module(vm, func);
                    let caller_frame_idx = vm.call_depth() - 1;

                    if let Err(e) = vm.push_frame_with_closure_and_module(
                        Arc::clone(code),
                        dst_reg,
                        closure,
                        module,
                    ) {
                        return ControlFlow::Error(e);
                    }

                    for i in 0..argc {
                        let arg = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
                        vm.current_frame_mut().set_local(i as u16, arg);
                    }

                    initialize_closure_cellvars_from_locals(vm.current_frame_mut(), argc);
                    if let Err(err) = vm.dispatch_prepared_current_frame_via_jit() {
                        return ControlFlow::Error(err);
                    }
                    ControlFlow::Continue
                } else {
                    // Use full binder for defaults, *args/**kwargs, and kw-only semantics.
                    call_kw_user_function(vm, ptr, dst_reg, argc, 0, 0)
                }
            }
            TypeId::STATICMETHOD => {
                let frame = vm.current_frame();
                let args: Vec<Value> = (0..argc)
                    .map(|i| frame.get_reg(dst_reg + 1 + i as u8))
                    .collect();

                match invoke_callable_value(vm, func_val, &args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            TypeId::METHOD => {
                let bound = unsafe { &*(ptr as *const BoundMethod) };
                match invoke_callable_with_implicit_self(
                    vm,
                    bound.function(),
                    bound.instance(),
                    dst_reg,
                    argc,
                    0,
                    0,
                    "__call__",
                ) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            TypeId::WRAPPER_DESCRIPTOR
            | TypeId::METHOD_DESCRIPTOR
            | TypeId::CLASSMETHOD_DESCRIPTOR => {
                let Some(target) = reflected_descriptor_callable_target(func_val) else {
                    return ControlFlow::Error(RuntimeError::type_error(format!(
                        "'{}' object is not callable",
                        type_id.name()
                    )));
                };
                let frame = vm.current_frame();
                let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(argc);
                for i in 0..argc {
                    args.push(frame.get_reg(dst_reg + 1 + i as u8));
                }

                match invoke_callable_value(vm, target, &args) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            _ => match resolve_dunder_call_target(func_val) {
                Some(resolved) => match invoke_resolved_callable(vm, resolved, dst_reg, argc, 0, 0)
                {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                },
                None => ControlFlow::Error(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    type_id.name()
                ))),
            },
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not callable"))
    }
}

/// CallKw: call with keyword arguments
///
/// This is a two-instruction sequence:
/// - Instruction 1 (CallKw): [opcode][dst][func][posargc]
/// - Instruction 2 (CallKwEx): [kwargc][kwnames_idx_lo][kwnames_idx_hi]
///
/// Arguments layout in registers:
/// - dst+1 .. dst+posargc: positional argument values
/// - dst+posargc+1 .. dst+posargc+kwargc: keyword argument values
///
/// Keyword names are stored in the constant pool as a KwNamesTuple.
///
/// # Performance Optimizations
///
/// - Uses SmallVec<[Value; 8]> to avoid heap allocation for typical calls
/// - Uses u64 bitmap for tracking bound parameters (supports up to 64 params)
/// - Single-pass binding algorithm with O(P + K) complexity
/// - Pre-allocated varargs/varkw containers with exact capacity
#[inline(always)]
pub fn call_kw(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let func_reg = inst.src1().0;
    let posargc = inst.src2().0 as usize;

    // Read the extension instruction (CallKwEx) for keyword metadata
    let frame = vm.current_frame_mut();
    let ip = frame.ip as usize;

    // The CallKwEx instruction should be at ip (current instruction is already consumed)
    let ext_inst = frame.code.instructions[ip];
    frame.ip = (ip + 1) as u32; // Skip past the extension instruction

    let kwargc = ext_inst.dst().0 as usize;
    let kwnames_idx = (ext_inst.src1().0 as u16) | ((ext_inst.src2().0 as u16) << 8);

    // Get function object
    let func_val = vm.frames[vm.call_depth() - 1].get_reg(func_reg);

    if let Some(ptr) = func_val.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        match type_id {
            TypeId::BUILTIN_FUNCTION => {
                call_kw_builtin(vm, ptr, dst_reg, posargc, kwargc, kwnames_idx)
            }
            TypeId::TYPE => {
                let caller_frame = &vm.frames[vm.call_depth() - 1];
                let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(posargc);
                for i in 0..posargc {
                    args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
                }

                let result = if let Some(represented_type) = builtin_type_object_type_id(ptr) {
                    if kwargc == 0 {
                        call_builtin_type_with_vm(vm, represented_type, &args)
                            .map_err(RuntimeError::from)
                    } else {
                        let kwnames_val = caller_frame.get_const(kwnames_idx);
                        let Some(kwnames_ptr) = kwnames_val.as_object_ptr() else {
                            return ControlFlow::Error(RuntimeError::internal(
                                "Invalid keyword names in constant pool",
                            ));
                        };
                        let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };
                        let mut keyword_args: SmallVec<[(&str, Value); 4]> =
                            SmallVec::with_capacity(kwargc);
                        for i in 0..kwargc {
                            let kw_name = match kwnames.get(i) {
                                Some(name) => name,
                                None => {
                                    return ControlFlow::Error(RuntimeError::internal(
                                        "Invalid keyword names tuple",
                                    ));
                                }
                            };
                            let kw_val =
                                caller_frame.get_reg(dst_reg + 1 + posargc as u8 + i as u8);
                            keyword_args.push((kw_name, kw_val));
                        }
                        call_builtin_type_kw_with_vm(vm, represented_type, &args, &keyword_args)
                            .map_err(RuntimeError::from)
                    }
                } else if let Some(class) = class_object_from_type_ptr(ptr) {
                    instantiate_user_defined_class(vm, class, dst_reg, posargc, kwargc, kwnames_idx)
                } else {
                    Err(RuntimeError::type_error("type object is not callable"))
                };

                match result {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            TypeId::FUNCTION | TypeId::CLOSURE => {
                call_kw_user_function(vm, ptr, dst_reg, posargc, kwargc, kwnames_idx)
            }
            TypeId::STATICMETHOD => {
                let caller_frame = &vm.frames[vm.call_depth() - 1];
                let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(posargc);
                for i in 0..posargc {
                    args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
                }
                let keyword_args = match collect_call_keyword_args(
                    caller_frame,
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                ) {
                    Ok(keyword_args) => keyword_args,
                    Err(err) => return ControlFlow::Error(err),
                };
                let keyword_refs: SmallVec<[(&str, Value); 4]> = keyword_args
                    .iter()
                    .map(|(name, value)| (name.as_ref(), *value))
                    .collect();

                match invoke_callable_value_with_keywords(vm, func_val, &args, &keyword_refs) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(err) => ControlFlow::Error(err),
                }
            }
            TypeId::METHOD => {
                let bound = unsafe { &*(ptr as *const BoundMethod) };
                match invoke_callable_with_implicit_self(
                    vm,
                    bound.function(),
                    bound.instance(),
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                    "__call__",
                ) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            TypeId::WRAPPER_DESCRIPTOR
            | TypeId::METHOD_DESCRIPTOR
            | TypeId::CLASSMETHOD_DESCRIPTOR => {
                let Some(target) = reflected_descriptor_callable_target(func_val) else {
                    return ControlFlow::Error(RuntimeError::type_error(format!(
                        "'{}' object is not callable",
                        type_id.name()
                    )));
                };

                let caller_frame = &vm.frames[vm.call_depth() - 1];
                let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(posargc);
                for i in 0..posargc {
                    args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
                }
                let keyword_args = match collect_call_keyword_args(
                    caller_frame,
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                ) {
                    Ok(keyword_args) => keyword_args,
                    Err(err) => return ControlFlow::Error(err),
                };
                let keyword_refs: SmallVec<[(&str, Value); 4]> = keyword_args
                    .iter()
                    .map(|(name, value)| (name.as_ref(), *value))
                    .collect();

                match invoke_callable_value_with_keywords(vm, target, &args, &keyword_refs) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                }
            }
            _ => match resolve_dunder_call_target(func_val) {
                Some(resolved) => match invoke_resolved_callable(
                    vm,
                    resolved,
                    dst_reg,
                    posargc,
                    kwargc,
                    kwnames_idx,
                ) {
                    Ok(result) => {
                        vm.current_frame_mut().set_reg(dst_reg, result);
                        ControlFlow::Continue
                    }
                    Err(e) => ControlFlow::Error(e),
                },
                None => ControlFlow::Error(RuntimeError::type_error(format!(
                    "'{}' object is not callable",
                    type_id.name()
                ))),
            },
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not callable"))
    }
}

/// Handle CallKw for builtin functions.
///
/// Builtins receive all arguments as a flat vector; keyword semantics are
/// handled by the builtin implementation itself.
#[inline]
fn call_kw_builtin(
    vm: &mut VirtualMachine,
    ptr: *const (),
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> ControlFlow {
    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };

    let (args, keyword_args) = {
        let caller_frame = &vm.frames[vm.call_depth() - 1];
        let mut args: SmallVec<[Value; 8]> = SmallVec::with_capacity(posargc);
        for i in 0..posargc {
            args.push(caller_frame.get_reg(dst_reg + 1 + i as u8));
        }
        let keyword_args = if kwargc == 0 {
            SmallVec::new()
        } else {
            match collect_call_keyword_args(caller_frame, dst_reg, posargc, kwargc, kwnames_idx) {
                Ok(keyword_args) => keyword_args,
                Err(err) => return ControlFlow::Error(err),
            }
        };
        (args, keyword_args)
    };

    let result = if keyword_args.is_empty() {
        invoke_builtin(vm, builtin, &args)
    } else {
        let keyword_refs: SmallVec<[(&str, Value); 4]> = keyword_args
            .iter()
            .map(|(name, value)| (name.as_ref(), *value))
            .collect();
        invoke_builtin_with_keywords(vm, builtin, &args, &keyword_refs)
    };

    match result {
        Ok(result) => {
            vm.current_frame_mut().set_reg(dst_reg, result);
            ControlFlow::Continue
        }
        Err(e) => ControlFlow::Error(e),
    }
}

/// Handle CallKw for user-defined functions.
///
/// This implements full Python 3.12 argument binding semantics:
/// 1. Bind positional arguments to parameters
/// 2. Collect excess positional into *args tuple (if function accepts varargs)
/// 3. Bind keyword arguments by name
/// 4. Collect unrecognized keywords into **kwargs dict (if function accepts varkw)
/// 5. Fill unbound parameters with defaults
/// 6. Verify all required parameters are bound
/// 7. Push new frame and populate locals
///
/// # Optimizations
///
/// - SmallVec<[Value; 8]> avoids heap allocation for typical functions
/// - u64 bitmap tracks bound parameters without allocation (supports ≤64 params)
/// - Varargs tuple allocated with exact capacity
/// - Varkw dict allocated only when needed
#[inline]
fn call_kw_user_function(
    vm: &mut VirtualMachine,
    ptr: *const (),
    dst_reg: u8,
    posargc: usize,
    kwargc: usize,
    kwnames_idx: u16,
) -> ControlFlow {
    let func = unsafe { &*(ptr as *const FunctionObject) };
    if let Some(result) = function_vectorcall_override_result(func) {
        vm.current_frame_mut().set_reg(dst_reg, result);
        return ControlFlow::Continue;
    }
    let code = Arc::clone(&func.code);
    let closure = match materialize_function_invocation_closure(func, &code) {
        Ok(closure) => closure,
        Err(err) => return ControlFlow::Error(err),
    };
    let module = resolve_function_module(vm, func);

    // Extract function signature metadata
    let arg_count = code.arg_count as usize;
    let kwonly_count = code.kwonlyarg_count as usize;
    let total_params = arg_count + kwonly_count;
    let has_varargs = code.flags.contains(CodeFlags::VARARGS);
    let has_varkw = code.flags.contains(CodeFlags::VARKEYWORDS);

    // Optimization: use u64 bitmap for tracking bound parameters (supports ≤64 params)
    // Falls back to linear check for functions with >64 params (extremely rare)
    // Fast-path for common signatures with a zero-allocation bitmap.
    // Large signatures transparently fall back to heap flags.
    let mut bound = BoundArgs::new(total_params);

    // Pre-allocate bound arguments with SmallVec optimization
    // SmallVec<[Value; 8]> stores up to 8 Values inline, avoiding heap allocation
    let mut bound_args: SmallVec<[Value; 8]> = SmallVec::with_capacity(total_params);
    bound_args.resize(total_params, Value::none());

    // =========================================================================
    // Phase 1: Bind positional arguments
    // =========================================================================

    let caller_frame_idx = vm.call_depth() - 1;

    // Bind positional args that fit in regular parameter slots
    let bound_positional = posargc.min(arg_count);
    for i in 0..bound_positional {
        let arg_val = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
        bound_args[i] = arg_val;
        bound.set_bound(i);
    }

    // Handle excess positional arguments
    // We defer allocation to Phase 5 when we have access to the allocator
    let varargs_values: Option<SmallVec<[Value; 8]>> = if posargc > arg_count {
        if has_varargs {
            // Collect excess for later allocation into *args tuple
            let excess_count = posargc - arg_count;
            let mut excess: SmallVec<[Value; 8]> = SmallVec::with_capacity(excess_count);
            for i in arg_count..posargc {
                let arg_val = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
                excess.push(arg_val);
            }
            Some(excess)
        } else {
            // Too many positional arguments - error
            return ControlFlow::Error(RuntimeError::type_error(format!(
                "{}() takes {} positional argument{} but {} {} given",
                code.name,
                arg_count,
                if arg_count == 1 { "" } else { "s" },
                posargc,
                if posargc == 1 { "was" } else { "were" }
            )));
        }
    } else if has_varargs {
        // Empty *args tuple (will create empty tuple during Phase 5)
        Some(SmallVec::new())
    } else {
        None
    };

    // =========================================================================
    // Phase 2: Bind keyword arguments
    // =========================================================================

    // Prepare varkw entries - defer dict allocation to Phase 5
    let mut varkw_entries: Option<SmallVec<[(Value, Value); 4]>> = if has_varkw {
        Some(SmallVec::new())
    } else {
        None
    };

    if kwargc > 0 {
        // Get keyword names from constant pool
        let kwnames_val = vm.frames[caller_frame_idx].get_const(kwnames_idx);

        if let Some(kwnames_ptr) = kwnames_val.as_object_ptr() {
            let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };

            for i in 0..kwargc {
                let kw_name = match kwnames.get(i) {
                    Some(name) => name,
                    None => {
                        return ControlFlow::Error(RuntimeError::internal(
                            "Invalid keyword names tuple",
                        ));
                    }
                };
                let kw_val =
                    vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + posargc as u8 + i as u8);

                // Find parameter index by name
                // Account for varargs offset: if VARARGS is set, kwonly params
                // are offset by 1 in locals array
                if let Some(param_idx) = find_param_index_with_varargs(&code, kw_name, has_varargs)
                {
                    // Check for duplicate assignment using bitmap
                    if bound.is_bound(param_idx) {
                        return ControlFlow::Error(RuntimeError::type_error(format!(
                            "{}() got multiple values for argument '{}'",
                            code.name, kw_name
                        )));
                    }
                    bound_args[param_idx] = kw_val;
                    bound.set_bound(param_idx);
                } else if let Some(ref mut varkw_list) = varkw_entries {
                    // Store entry for later allocation into **kwargs dict
                    let key = create_string_key(kw_name);
                    varkw_list.push((key, kw_val));
                } else {
                    // Unexpected keyword argument - error
                    return ControlFlow::Error(RuntimeError::type_error(format!(
                        "{}() got an unexpected keyword argument '{}'",
                        code.name, kw_name
                    )));
                }
            }
        } else {
            return ControlFlow::Error(RuntimeError::internal(
                "Invalid keyword names in constant pool",
            ));
        }
    }

    // =========================================================================
    // Phase 3: Fill missing positional parameters with defaults
    // =========================================================================

    for i in 0..arg_count {
        if !bound.is_bound(i) {
            // Try to get default value
            if let Some(default_val) = func.get_default(i) {
                bound_args[i] = default_val;
                bound.set_bound(i);
            } else {
                // Missing required positional argument
                let param_name = code.locals.get(i).map(|s| s.as_ref()).unwrap_or("?");
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "{}() missing {} required positional argument: '{}'",
                    code.name,
                    1, // Could count total missing for better message
                    param_name
                )));
            }
        }
    }

    // =========================================================================
    // Phase 4: Fill missing keyword-only parameters with kwdefaults
    // =========================================================================

    for i in arg_count..total_params {
        if !bound.is_bound(i) {
            // Calculate correct locals index accounting for varargs slot
            let locals_idx = if has_varargs { i + 1 } else { i };
            let param_name = code
                .locals
                .get(locals_idx)
                .map(|s| s.as_ref())
                .unwrap_or("?");

            // Check kwdefaults for this parameter
            let found_default = func.kwdefaults.as_ref().and_then(|kwdefaults| {
                kwdefaults
                    .iter()
                    .find(|(n, _)| n.as_ref() == param_name)
                    .map(|(_, val)| *val)
            });

            if let Some(default_val) = found_default {
                bound_args[i] = default_val;
                bound.set_bound(i);
            } else {
                // Missing required keyword-only argument
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "{}() missing {} required keyword-only argument: '{}'",
                    code.name, 1, param_name
                )));
            }
        }
    }

    // =========================================================================
    // Phase 5: Push new frame and populate locals
    // =========================================================================

    // Allocate *args tuple on GC heap if needed
    let varargs_value = if has_varargs {
        let tuple = match &varargs_values {
            Some(vals) if !vals.is_empty() => TupleObject::from_slice(&vals),
            _ => TupleObject::empty(),
        };
        Some(alloc_value_in_current_heap_or_box(tuple))
    } else {
        None
    };

    // Allocate **kwargs dict on GC heap if needed
    let varkw_value = if has_varkw {
        let mut dict = DictObject::new();
        if let Some(entries) = &varkw_entries {
            for (key, val) in entries {
                dict.set(*key, *val);
            }
        }
        Some(alloc_value_in_current_heap_or_box(dict))
    } else {
        None
    };

    if is_generator_code(&code) {
        let mut locals = [Value::none(); 256];
        let mut local_idx = 0usize;

        for i in 0..arg_count {
            locals[local_idx] = bound_args[i];
            local_idx += 1;
        }

        if let Some(tuple_val) = varargs_value {
            locals[local_idx] = tuple_val;
            local_idx += 1;
        }

        for i in arg_count..total_params {
            locals[local_idx] = bound_args[i];
            local_idx += 1;
        }

        if let Some(dict_val) = varkw_value {
            locals[local_idx] = dict_val;
            local_idx += 1;
        }

        let mut generator = GeneratorObject::from_code(Arc::clone(&code));
        generator.set_module_ptr(function_module_ptr(vm, func));
        generator.seed_locals(&locals, prefix_liveness(local_idx));
        let value = match alloc_heap_value(vm, generator, "generator object") {
            Ok(value) => value,
            Err(err) => return ControlFlow::Error(err),
        };
        vm.current_frame_mut().set_reg(dst_reg, value);
        return ControlFlow::Continue;
    }

    if let Err(e) =
        vm.push_frame_with_closure_and_module(Arc::clone(&code), dst_reg, closure, module)
    {
        return ControlFlow::Error(e);
    }

    // Populate bound parameters in new frame
    // Locals layout:
    // - [0..arg_count): positional parameters
    // - [arg_count]: *args (if VARARGS)
    // - [arg_count + varargs_offset..]: keyword-only parameters
    // - [after kwonly]: **kwargs (if VARKEYWORDS)

    {
        let new_frame = vm.current_frame_mut();
        let mut local_idx = 0usize;

        // Set positional parameters
        for i in 0..arg_count {
            new_frame.set_local(local_idx as u16, bound_args[i]);
            local_idx += 1;
        }

        // Set *args tuple if present
        if let Some(tuple_val) = varargs_value {
            new_frame.set_local(local_idx as u16, tuple_val);
            local_idx += 1;
        }

        // Set keyword-only parameters
        for i in arg_count..total_params {
            new_frame.set_local(local_idx as u16, bound_args[i]);
            local_idx += 1;
        }

        // Set **kwargs dict if present
        if let Some(dict_val) = varkw_value {
            new_frame.set_local(local_idx as u16, dict_val);
        }

        let initialized_local_count = local_idx as usize + usize::from(varkw_value.is_some());
        initialize_closure_cellvars_from_locals(new_frame, initialized_local_count);
    }
    if let Err(err) = vm.dispatch_prepared_current_frame_via_jit() {
        return ControlFlow::Error(err);
    }
    ControlFlow::Continue
}

#[inline(always)]
fn is_generator_code(code: &CodeObject) -> bool {
    code.is_generator() || code.is_coroutine() || code.is_async_generator()
}

#[inline]
fn create_generator_from_simple_call(
    vm: &mut VirtualMachine,
    code: &Arc<CodeObject>,
    dst_reg: u8,
    argc: usize,
    module_ptr: *const (),
    closure: Option<Arc<ClosureEnv>>,
) -> ControlFlow {
    let caller_frame_idx = vm.call_depth() - 1;
    let mut locals = [Value::none(); 256];
    for i in 0..argc {
        locals[i] = vm.frames[caller_frame_idx].get_reg(dst_reg + 1 + i as u8);
    }

    let mut generator = GeneratorObject::from_code(Arc::clone(code));
    generator.set_module_ptr(module_ptr);
    if let Some(closure) = closure {
        generator.set_closure(closure);
    }
    generator.seed_locals(&locals, prefix_liveness(argc));
    let value = match alloc_heap_value(vm, generator, "generator object") {
        Ok(value) => value,
        Err(err) => return ControlFlow::Error(err),
    };
    vm.current_frame_mut().set_reg(dst_reg, value);
    ControlFlow::Continue
}

#[inline(always)]
fn prefix_liveness(count: usize) -> LivenessMap {
    if count == 0 {
        LivenessMap::empty()
    } else if count >= 64 {
        LivenessMap::ALL
    } else {
        LivenessMap::from_bits((1u64 << count) - 1)
    }
}

/// Tracks which parameters have been bound during argument binding.
///
/// Uses a `u64` bitset for common small signatures and a heap-allocated
/// boolean array for large signatures.
enum BoundArgs {
    Inline(u64),
    Heap(Box<[bool]>),
}

impl BoundArgs {
    #[inline]
    fn new(total_params: usize) -> Self {
        if total_params <= u64::BITS as usize {
            Self::Inline(0)
        } else {
            Self::Heap(vec![false; total_params].into_boxed_slice())
        }
    }

    #[inline]
    fn is_bound(&self, index: usize) -> bool {
        match self {
            Self::Inline(mask) => {
                debug_assert!(index < u64::BITS as usize);
                (mask & (1u64 << index)) != 0
            }
            Self::Heap(flags) => *flags.get(index).unwrap_or(&false),
        }
    }

    #[inline]
    fn set_bound(&mut self, index: usize) {
        match self {
            Self::Inline(mask) => {
                debug_assert!(index < u64::BITS as usize);
                *mask |= 1u64 << index;
            }
            Self::Heap(flags) => {
                if let Some(slot) = flags.get_mut(index) {
                    *slot = true;
                }
            }
        }
    }
}

/// Find parameter index by name, accounting for varargs slot offset.
///
/// When a function has *args, the locals array layout is:
/// - [0..arg_count): positional params
/// - [arg_count]: *args slot
/// - [arg_count+1..): keyword-only params
///
/// This function returns the *parameter index* (0 to total_params-1),
/// not the locals index.
///
/// # Performance
///
/// Uses linear search which is optimal for typical parameter counts (<10).
/// For hot paths, the JIT's inline caching handles optimization.
#[inline]
fn find_param_index_with_varargs(
    code: &CodeObject,
    name: &str,
    has_varargs: bool,
) -> Option<usize> {
    let arg_count = code.arg_count as usize;
    let kwonly_count = code.kwonlyarg_count as usize;

    // Search positional parameters (no offset)
    for i in 0..arg_count {
        if let Some(param_name) = code.locals.get(i) {
            if param_name.as_ref() == name {
                return Some(i);
            }
        }
    }

    // Search keyword-only parameters (with varargs offset if applicable)
    let kwonly_locals_start = if has_varargs {
        arg_count + 1
    } else {
        arg_count
    };
    for i in 0..kwonly_count {
        let locals_idx = kwonly_locals_start + i;
        if let Some(param_name) = code.locals.get(locals_idx) {
            if param_name.as_ref() == name {
                return Some(arg_count + i); // Return parameter index, not locals index
            }
        }
    }

    None
}

/// Create a string key Value for use in **kwargs dict.
#[inline]
fn create_string_key(name: &str) -> Value {
    Value::string(intern(name))
}

/// CallMethod: dst = obj.method(args...)
#[inline(always)]
pub fn call_method(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // Delegate to the optimized method-dispatch implementation used by the opcode table.
    crate::ops::method_dispatch::call_method(vm, inst)
}

/// CallKwEx: extension instruction for CallKw.
///
/// This instruction should never be executed directly - it is consumed by CallKw.
/// If we reach this opcode, it indicates bytecode corruption.
#[inline(always)]
pub fn call_kw_ex(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::internal(
        "CallKwEx executed directly (bytecode corruption)",
    ))
}

/// TailCall: call reusing current frame
///
/// Optimizes tail-recursive calls by reusing the current frame.
#[inline(always)]
pub fn tail_call(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // For now, fall back to regular call
    // TODO: Implement true tail call optimization
    call(vm, inst)
}
