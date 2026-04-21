//! Object operation handlers.
//!
//! Handles attribute access, item access, and iteration with inline caching.
//! All operations use TypeId-based dispatch for type safety and JIT compatibility.

use crate::VirtualMachine;
use crate::builtins::BuiltinError;
use crate::builtins::BuiltinFunctionObject;
use crate::builtins::{ExceptionFlags, ExceptionTypeObject, ExceptionValue};
use crate::builtins::{
    builtin_bound_type_attribute_value, builtin_bound_type_attribute_value_static,
    builtin_instance_attribute_value, exception_proxy_class_id_from_ptr,
    exception_type_attribute_value, heap_type_attribute_value,
};
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::frame::{Frame, REGISTER_COUNT};
use crate::ops::attribute::is_user_defined_type;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::method_cache::method_cache;
use crate::ops::method_dispatch::resolve_builtin_instance_method;
use crate::stdlib::collections::deque::DequeObject;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{
    BoundMethod, ClassMethodDescriptor, PropertyDescriptor, StaticMethodDescriptor,
};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::super_obj::{SuperBinding, SuperObject};
use prism_runtime::object::type_builtins::{builtin_class_mro, class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{
    CellViewObject, CodeObjectView, FrameViewObject, TracebackViewObject,
};
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::iter::IteratorObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use std::sync::{Arc, LazyLock};

// =============================================================================
// Type Extraction
// =============================================================================

/// Extract TypeId from an object pointer.
///
/// # Safety
/// The pointer must point to a valid object with ObjectHeader at offset 0.
/// All Prism objects use #[repr(C)] layout with ObjectHeader as first field.
///
/// # Performance
/// This is O(1) - a single memory read. JIT code can inline this as:
/// ```asm
/// mov eax, [rdi]  ; Load TypeId (first 4 bytes of object)
/// ```
#[inline(always)]
pub fn extract_type_id(ptr: *const ()) -> TypeId {
    // SAFETY: All objects have ObjectHeader at offset 0 due to #[repr(C)]
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

#[inline]
pub(crate) fn dict_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static DictObject> {
    match extract_type_id(ptr) {
        TypeId::DICT => Some(unsafe { &*(ptr as *const DictObject) }),
        type_id if is_user_defined_type(type_id) => {
            unsafe { &*(ptr as *const ShapedObject) }.dict_backing()
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn dict_storage_mut_from_ptr(ptr: *const ()) -> Option<&'static mut DictObject> {
    match extract_type_id(ptr) {
        TypeId::DICT => Some(unsafe { &mut *(ptr as *mut DictObject) }),
        type_id if is_user_defined_type(type_id) => {
            unsafe { &mut *(ptr as *mut ShapedObject) }.dict_backing_mut()
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn list_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static ListObject> {
    prism_runtime::types::list::object_ptr_as_list_ref(ptr)
}

#[inline]
pub(crate) fn list_storage_mut_from_ptr(ptr: *const ()) -> Option<&'static mut ListObject> {
    prism_runtime::types::list::object_ptr_as_list_mut(ptr as *mut ())
}

#[inline]
pub(crate) fn string_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static StringObject> {
    prism_runtime::types::string::object_ptr_as_string_ref(ptr).and_then(|value| match value {
        prism_runtime::types::string::StringValueRef::Heap(string) => Some(string),
        prism_runtime::types::string::StringValueRef::Interned(_) => None,
    })
}

#[inline]
fn class_object_from_type_ptr(ptr: *const ()) -> Option<&'static PyClassObject> {
    if crate::builtins::builtin_type_object_type_id(ptr).is_some() {
        return None;
    }

    Some(unsafe { &*(ptr as *const PyClassObject) })
}

#[inline]
fn lookup_user_class_attr(class: &PyClassObject, name: &InternedString) -> Option<Value> {
    class
        .lookup_method(name, global_class)
        .map(|slot| slot.value)
}

#[inline]
pub(crate) fn lookup_class_metaclass_attr(
    class: &PyClassObject,
    name: &InternedString,
) -> Option<Value> {
    let metaclass = class.metaclass();
    let ptr = metaclass.as_object_ptr()?;
    let metaclass_class = class_object_from_type_ptr(ptr)?;
    metaclass_class
        .lookup_method(name, global_class)
        .map(|slot| slot.value)
}

#[inline]
fn lookup_instance_class_attr(type_id: TypeId, name: &InternedString) -> Option<Value> {
    let class = global_class(ClassId(type_id.raw()))?;
    lookup_user_class_attr(class.as_ref(), name)
}

fn lookup_builtin_base_instance_attr(
    obj: Value,
    type_id: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(class) = global_class(ClassId(type_id.raw())) else {
        return Ok(None);
    };

    for &class_id in class.mro().iter().skip(1) {
        if class_id.0 >= TypeId::FIRST_USER_TYPE {
            continue;
        }

        let owner = class_id_to_type_id(class_id);
        if let Some(cached) = resolve_builtin_instance_method(owner, name.as_str()) {
            return Ok(Some(bind_instance_attribute(cached.method, obj)));
        }

        if let Some(value) = builtin_bound_type_attribute_value_static(owner, obj, name)? {
            return Ok(Some(value));
        }
    }

    Ok(None)
}

#[inline]
fn class_id_to_value(class_id: ClassId) -> Option<Value> {
    if class_id == ClassId::NONE {
        return None;
    }

    if class_id == ClassId::OBJECT {
        return Some(crate::builtins::builtin_type_object_for_type_id(
            TypeId::OBJECT,
        ));
    }

    if class_id.0 < TypeId::FIRST_USER_TYPE {
        return Some(crate::builtins::builtin_type_object_for_type_id(
            class_id_to_type_id(class_id),
        ));
    }

    global_class(class_id).map(|class| Value::object_ptr(Arc::as_ptr(&class) as *const ()))
}

#[inline]
fn bind_super_lookup_value(value: Value, super_obj: &SuperObject, name: &InternedString) -> Value {
    match super_obj.binding() {
        SuperBinding::Unbound => value,
        SuperBinding::Instance => bind_instance_attribute(value, super_obj.obj()),
        SuperBinding::Type => {
            if name.as_str() == "__new__" {
                resolve_class_attribute(value, super_obj.obj())
            } else {
                bind_instance_attribute(value, super_obj.obj())
            }
        }
    }
}

#[inline]
fn is_property_descriptor_value(value: Value) -> bool {
    value
        .as_object_ptr()
        .is_some_and(|ptr| extract_type_id(ptr) == TypeId::PROPERTY)
}

#[inline]
fn property_descriptor_from_value(value: Value) -> Option<&'static PropertyDescriptor> {
    let ptr = value.as_object_ptr()?;
    (extract_type_id(ptr) == TypeId::PROPERTY)
        .then(|| unsafe { &*(ptr as *const PropertyDescriptor) })
}

#[inline]
fn property_descriptor_error(kind: &'static str) -> RuntimeError {
    RuntimeError::from(BuiltinError::AttributeError(format!(
        "property has no {kind}"
    )))
}

fn invoke_property_getter(
    vm: &mut VirtualMachine,
    descriptor: &PropertyDescriptor,
    instance: Value,
) -> Result<Value, RuntimeError> {
    let getter = descriptor
        .getter()
        .ok_or_else(|| property_descriptor_error("getter"))?;
    crate::ops::calls::invoke_callable_value(vm, getter, &[instance])
}

fn invoke_property_setter(
    vm: &mut VirtualMachine,
    descriptor: &PropertyDescriptor,
    instance: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    let setter = descriptor
        .setter()
        .ok_or_else(|| property_descriptor_error("setter"))?;
    crate::ops::calls::invoke_callable_value(vm, setter, &[instance, value]).map(|_| ())
}

fn invoke_property_deleter(
    vm: &mut VirtualMachine,
    descriptor: &PropertyDescriptor,
    instance: Value,
) -> Result<(), RuntimeError> {
    let deleter = descriptor
        .deleter()
        .ok_or_else(|| property_descriptor_error("deleter"))?;
    crate::ops::calls::invoke_callable_value(vm, deleter, &[instance]).map(|_| ())
}

pub(crate) fn super_attribute_value_static(
    super_value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let ptr = super_value
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::attribute_error("super", name.as_str()))?;
    if extract_type_id(ptr) != TypeId::SUPER {
        return Ok(None);
    }

    let super_obj = unsafe { &*(ptr as *const SuperObject) };
    match name.as_str() {
        "__self__" => {
            return Ok(Some(match super_obj.binding() {
                SuperBinding::Unbound => Value::none(),
                SuperBinding::Instance | SuperBinding::Type => super_obj.obj(),
            }));
        }
        "__self_class__" => {
            return Ok(Some(
                class_id_to_value(super_obj.obj_type()).unwrap_or_else(Value::none),
            ));
        }
        "__thisclass__" => {
            return Ok(Some(
                class_id_to_value(super_obj.this_type()).unwrap_or_else(Value::none),
            ));
        }
        _ => {}
    }

    let mro = if super_obj.obj_type().0 < TypeId::FIRST_USER_TYPE {
        builtin_class_mro(class_id_to_type_id(super_obj.obj_type()))
    } else {
        let Some(class) = global_class(super_obj.obj_type()) else {
            return Ok(None);
        };
        class.mro().to_vec()
    };

    let start = mro
        .iter()
        .position(|&class_id| class_id == super_obj.this_type())
        .map_or(0, |index| index + 1);

    for &class_id in &mro[start..] {
        if class_id.0 < TypeId::FIRST_USER_TYPE {
            let owner = class_id_to_type_id(class_id);
            if super_obj.binding() != SuperBinding::Unbound
                && let Some(cached) = resolve_builtin_instance_method(owner, name.as_str())
            {
                return Ok(Some(bind_instance_attribute(
                    cached.method,
                    super_obj.obj(),
                )));
            }
            if let Some(value) =
                builtin_bound_type_attribute_value_static(owner, super_obj.obj(), name)?
            {
                return Ok(Some(value));
            }
            continue;
        }

        if let Some(class) = global_class(class_id)
            && let Some(value) = class.get_attr(name)
        {
            return Ok(Some(bind_super_lookup_value(value, super_obj, name)));
        }
    }

    Ok(None)
}

#[inline]
pub(crate) fn super_attribute_exists(super_value: Value, name: &InternedString) -> bool {
    super_attribute_value_static(super_value, name)
        .ok()
        .flatten()
        .is_some()
}

#[inline]
fn alloc_heap_value<T>(
    vm: &mut VirtualMachine,
    object: T,
    context: &'static str,
) -> Result<Value, RuntimeError>
where
    T: prism_runtime::Trace,
{
    vm.allocator()
        .alloc(object)
        .map(|ptr| Value::object_ptr(ptr as *const ()))
        .ok_or_else(|| {
            RuntimeError::internal(format!("out of memory: failed to allocate {context}"))
        })
}

static CODE_CO_POSITIONS_METHOD: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("code.co_positions"), code_co_positions));

#[inline]
fn optional_u32_to_value(value: Option<u32>) -> Value {
    value
        .and_then(|value| Value::int(value as i64))
        .unwrap_or_else(Value::none)
}

fn code_co_positions(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "code.co_positions() takes no arguments ({} given)",
            args.len().saturating_sub(1)
        )));
    }

    let receiver = args[0];
    let receiver_ptr = receiver
        .as_object_ptr()
        .ok_or_else(|| BuiltinError::TypeError("descriptor requires a 'code' receiver".into()))?;
    if extract_type_id(receiver_ptr) != TypeId::CODE {
        return Err(BuiltinError::TypeError(
            "descriptor requires a 'code' receiver".into(),
        ));
    }

    let code = unsafe { &*(receiver_ptr as *const CodeObjectView) }.code();
    let positions = code
        .positions()
        .map(
            |(lineno, end_lineno, col_offset, end_col_offset)| -> Value {
                let tuple = TupleObject::from_slice(&[
                    optional_u32_to_value(lineno),
                    optional_u32_to_value(end_lineno),
                    optional_u32_to_value(col_offset),
                    optional_u32_to_value(end_col_offset),
                ]);
                Value::object_ptr(Box::into_raw(Box::new(tuple)) as *const ())
            },
        )
        .collect();

    Ok(crate::builtins::iterator_to_value(
        IteratorObject::from_values(positions),
    ))
}

pub(crate) fn snapshot_module_dict(module: &crate::import::ModuleObject) -> DictObject {
    let attrs = module.all_attrs();
    let mut dict = DictObject::with_capacity(attrs.len());
    for (name, value) in attrs {
        dict.set(Value::string(name), value);
    }
    dict
}

pub(crate) fn snapshot_current_globals_dict(vm: &VirtualMachine) -> DictObject {
    if let Some(module) = vm.current_module_cloned() {
        return snapshot_module_dict(module.as_ref());
    }

    let mut dict = DictObject::with_capacity(vm.globals.len());
    for (name, value) in vm.globals.iter() {
        dict.set(Value::string(intern(name.as_ref())), *value);
    }
    dict
}

#[inline]
fn snapshot_dict_storage(storage: &DictObject) -> DictObject {
    let mut snapshot = DictObject::with_capacity(storage.len());
    snapshot.update(storage);
    snapshot
}

pub(crate) fn snapshot_frame_globals_dict(vm: &VirtualMachine, frame: &Frame) -> DictObject {
    if let Some(module) = frame.module.as_ref() {
        return snapshot_module_dict(module.as_ref());
    }

    snapshot_current_globals_dict(vm)
}

pub(crate) fn snapshot_frame_locals_dict(frame: &Frame) -> DictObject {
    if let Some(mapping) = frame.locals_mapping()
        && let Some(ptr) = mapping.as_object_ptr()
        && let Some(storage) = dict_storage_ref_from_ptr(ptr)
    {
        return snapshot_dict_storage(storage);
    }

    let mut dict = DictObject::with_capacity(frame.code.locals.len());
    for (slot, name) in frame.code.locals.iter().enumerate() {
        if slot >= REGISTER_COUNT || !frame.reg_is_written(slot as u8) {
            continue;
        }
        dict.set(
            Value::string(intern(name.as_ref())),
            frame.get_reg(slot as u8),
        );
    }
    dict
}

fn function_attr_dict_value(
    vm: &mut VirtualMachine,
    func: &FunctionObject,
) -> Result<Value, RuntimeError> {
    let dict_ptr = func.ensure_attr_dict(|dict| {
        vm.allocator()
            .alloc(dict)
            .map(|ptr| ptr as *mut DictObject)
            .ok_or_else(|| {
                RuntimeError::internal(
                    "out of memory: failed to allocate function attribute dict".to_string(),
                )
            })
    })?;
    Ok(Value::object_ptr(dict_ptr as *const ()))
}

fn function_module(
    vm: &VirtualMachine,
    func: &FunctionObject,
) -> Option<Arc<crate::import::ModuleObject>> {
    if func.globals_ptr().is_null() {
        vm.current_module_cloned()
    } else {
        vm.module_from_globals_ptr(func.globals_ptr())
            .or_else(|| vm.current_module_cloned())
    }
}

fn function_attr_value_in_vm(
    vm: &mut VirtualMachine,
    func_ptr: *const (),
    func: &FunctionObject,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if let Some(value) = function_attr_value(func, name) {
        return Ok(Some(value));
    }

    match name.as_str() {
        "__doc__" => Ok(Some(Value::none())),
        "__dict__" => Ok(Some(function_attr_dict_value(vm, func)?)),
        "__module__" => Ok(Some(
            function_module(vm, func)
                .map(|module| Value::string(intern(module.name())))
                .unwrap_or_else(|| Value::string(intern("__main__"))),
        )),
        "__defaults__" => match func
            .defaults
            .as_ref()
            .filter(|defaults| !defaults.is_empty())
        {
            Some(defaults) => {
                let tuple = TupleObject::from_slice(defaults);
                Ok(Some(alloc_heap_value(
                    vm,
                    tuple,
                    "function defaults tuple",
                )?))
            }
            None => Ok(Some(Value::none())),
        },
        "__code__" => Ok(Some(alloc_heap_value(
            vm,
            CodeObjectView::new(Arc::clone(&func.code)),
            "code object view",
        )?)),
        "__globals__" => {
            let dict = function_module(vm, func)
                .map(|module| snapshot_module_dict(module.as_ref()))
                .unwrap_or_else(|| snapshot_current_globals_dict(vm));
            Ok(Some(alloc_heap_value(vm, dict, "globals dict snapshot")?))
        }
        "__closure__" => {
            let Some(closure) = vm.lookup_function_closure(func_ptr) else {
                return Ok(Some(Value::none()));
            };

            let mut items = Vec::with_capacity(closure.len());
            for idx in 0..closure.len() {
                items.push(alloc_heap_value(
                    vm,
                    CellViewObject::new(Arc::clone(closure.get_cell(idx))),
                    "closure cell view",
                )?);
            }

            let tuple = TupleObject::from_vec(items);
            Ok(Some(alloc_heap_value(vm, tuple, "closure tuple")?))
        }
        _ => Ok(None),
    }
}

fn method_attr_value_in_vm(
    vm: &mut VirtualMachine,
    method: &BoundMethod,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    match name.as_str() {
        "__self__" => Ok(Some(method.instance())),
        "__func__" | "__wrapped__" => Ok(Some(method.function())),
        _ => {
            let Some(func_ptr) = method.function().as_object_ptr() else {
                return Ok(None);
            };

            match extract_type_id(func_ptr) {
                TypeId::FUNCTION | TypeId::CLOSURE => {
                    let func = unsafe { &*(func_ptr as *const FunctionObject) };
                    function_attr_value_in_vm(vm, func_ptr, func, name)
                }
                _ => Ok(None),
            }
        }
    }
}

fn code_string_tuple_value(
    vm: &mut VirtualMachine,
    values: &[Arc<str>],
    context: &'static str,
) -> Result<Value, RuntimeError> {
    let items = values
        .iter()
        .map(|value| Value::string(intern(value.as_ref())))
        .collect();
    alloc_heap_value(vm, TupleObject::from_vec(items), context)
}

fn code_attr_value_in_vm(
    vm: &mut VirtualMachine,
    code_value: Value,
    code_view: &CodeObjectView,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let code = code_view.code();
    match name.as_str() {
        "co_name" => Ok(Some(Value::string(intern(code.name.as_ref())))),
        "co_qualname" => Ok(Some(Value::string(intern(code.qualname.as_ref())))),
        "co_filename" => Ok(Some(Value::string(intern(code.filename.as_ref())))),
        "co_firstlineno" => Ok(Some(
            Value::int(code.first_lineno as i64).unwrap_or_else(Value::none),
        )),
        "co_argcount" => Ok(Some(
            Value::int(code.arg_count as i64).unwrap_or_else(Value::none),
        )),
        "co_posonlyargcount" => Ok(Some(
            Value::int(code.posonlyarg_count as i64).unwrap_or_else(Value::none),
        )),
        "co_kwonlyargcount" => Ok(Some(
            Value::int(code.kwonlyarg_count as i64).unwrap_or_else(Value::none),
        )),
        "co_nlocals" => Ok(Some(
            Value::int(code.locals.len() as i64).unwrap_or_else(Value::none),
        )),
        "co_flags" => Ok(Some(
            Value::int(code.flags.bits() as i64).unwrap_or_else(Value::none),
        )),
        "co_consts" => Ok(Some(alloc_heap_value(
            vm,
            TupleObject::from_vec(code.constants.to_vec()),
            "code constants tuple",
        )?)),
        "co_varnames" => Ok(Some(code_string_tuple_value(
            vm,
            &code.locals,
            "code varnames tuple",
        )?)),
        "co_names" => Ok(Some(code_string_tuple_value(
            vm,
            &code.names,
            "code names tuple",
        )?)),
        "co_freevars" => Ok(Some(code_string_tuple_value(
            vm,
            &code.freevars,
            "code freevars tuple",
        )?)),
        "co_cellvars" => Ok(Some(code_string_tuple_value(
            vm,
            &code.cellvars,
            "code cellvars tuple",
        )?)),
        "co_positions" => {
            let bound = Box::leak(Box::new(CODE_CO_POSITIONS_METHOD.bind(code_value)));
            Ok(Some(Value::object_ptr(
                bound as *mut BuiltinFunctionObject as *const (),
            )))
        }
        _ => Ok(None),
    }
}

#[inline]
pub(crate) fn function_attr_value(func: &FunctionObject, name: &InternedString) -> Option<Value> {
    if let Some(value) = func.get_attr(name) {
        return Some(value);
    }

    match name.as_str() {
        "__name__" => Some(Value::string(intern(func.name.as_ref()))),
        "__qualname__" => Some(Value::string(intern(func.code.qualname.as_ref()))),
        "__doc__" => Some(Value::none()),
        _ => None,
    }
}

#[inline]
pub(crate) fn function_attr_exists(func: &FunctionObject, name: &InternedString) -> bool {
    function_attr_value(func, name).is_some()
        || matches!(
            name.as_str(),
            "__doc__"
                | "__dict__"
                | "__module__"
                | "__defaults__"
                | "__code__"
                | "__globals__"
                | "__closure__"
        )
}

#[inline]
pub(crate) fn descriptor_is_abstract(value: Value) -> bool {
    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let func = unsafe { &*(ptr as *const FunctionObject) };
            func.get_attr(&intern("__isabstractmethod__"))
                .and_then(|value| value.as_bool())
                .unwrap_or(false)
        }
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            descriptor_is_abstract(desc.function())
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            descriptor_is_abstract(desc.function())
        }
        TypeId::PROPERTY => {
            let desc = unsafe { &*(ptr as *const PropertyDescriptor) };
            desc.getter().is_some_and(descriptor_is_abstract)
                || desc.setter().is_some_and(descriptor_is_abstract)
                || desc.deleter().is_some_and(descriptor_is_abstract)
        }
        _ => false,
    }
}

#[inline]
pub(crate) fn descriptor_attr_value(value: Value, name: &InternedString) -> Option<Value> {
    let ptr = value.as_object_ptr()?;

    match extract_type_id(ptr) {
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            match name.as_str() {
                "__func__" | "__wrapped__" => Some(desc.function()),
                "__isabstractmethod__" => {
                    Some(Value::bool(descriptor_is_abstract(desc.function())))
                }
                _ => None,
            }
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            match name.as_str() {
                "__func__" | "__wrapped__" => Some(desc.function()),
                "__isabstractmethod__" => {
                    Some(Value::bool(descriptor_is_abstract(desc.function())))
                }
                _ => None,
            }
        }
        TypeId::PROPERTY => {
            let desc = unsafe { &*(ptr as *const PropertyDescriptor) };
            match name.as_str() {
                "fget" => Some(desc.getter().unwrap_or(Value::none())),
                "fset" => Some(desc.setter().unwrap_or(Value::none())),
                "fdel" => Some(desc.deleter().unwrap_or(Value::none())),
                "__doc__" => Some(desc.doc().unwrap_or(Value::none())),
                "__isabstractmethod__" => Some(Value::bool(descriptor_is_abstract(value))),
                _ => None,
            }
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn resolve_class_attribute(value: Value, owner: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    match extract_type_id(ptr) {
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            desc.bind_value(owner)
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            desc.function()
        }
        _ => value,
    }
}

#[inline]
pub(crate) fn bind_instance_attribute(value: Value, instance: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            let bound = Box::leak(Box::new(builtin.bind(instance)));
            Value::object_ptr(bound as *mut BuiltinFunctionObject as *const ())
        }
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let bound = Box::leak(Box::new(BoundMethod::new(value, instance)));
            Value::object_ptr(bound as *mut BoundMethod as *const ())
        }
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            let owner = if let Some(instance_ptr) = instance.as_object_ptr() {
                if let Some(class) = global_class(ClassId(extract_type_id(instance_ptr).raw())) {
                    Value::object_ptr(Arc::as_ptr(&class) as *const ())
                } else {
                    instance
                }
            } else {
                instance
            };
            desc.bind_value(owner)
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            desc.function()
        }
        _ => value,
    }
}

#[inline]
pub(crate) fn builtin_instance_method_attr_value(
    obj: Value,
    type_id: TypeId,
    name: &InternedString,
) -> Option<Value> {
    resolve_builtin_instance_method(type_id, name.as_str())
        .map(|cached| bind_instance_attribute(cached.method, obj))
}

#[inline]
pub(crate) fn builtin_instance_method_attr_exists(type_id: TypeId, name: &InternedString) -> bool {
    resolve_builtin_instance_method(type_id, name.as_str()).is_some()
}

// =============================================================================
// Attribute Access (with Inline Caching)
// =============================================================================

#[inline]
fn primitive_attribute_type_name(value: Value) -> &'static str {
    if value.is_none() {
        "NoneType"
    } else if value.is_bool() {
        "bool"
    } else if value.is_int() {
        "int"
    } else if value.is_float() {
        "float"
    } else if value.is_string() {
        "str"
    } else {
        "unknown"
    }
}

#[inline]
fn primitive_owner_type_id(value: Value) -> Option<TypeId> {
    if value.is_none() {
        Some(TypeId::NONE)
    } else if value.is_bool() {
        Some(TypeId::BOOL)
    } else if value.is_int() {
        Some(TypeId::INT)
    } else if value.is_float() {
        Some(TypeId::FLOAT)
    } else if value.is_string() {
        Some(TypeId::STR)
    } else {
        None
    }
}

fn lookup_builtin_primitive_attr(
    vm: &mut VirtualMachine,
    obj: Value,
    owner: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if name.as_str() == "__class__" {
        return Ok(Some(crate::builtins::builtin_type_object_for_type_id(
            owner,
        )));
    }

    for class_id in builtin_class_mro(owner) {
        let builtin_owner = class_id_to_type_id(class_id);

        if let Some(value) = builtin_instance_method_attr_value(obj, builtin_owner, name) {
            return Ok(Some(value));
        }

        if let Some(value) = builtin_instance_attribute_value(vm, builtin_owner, obj, name)? {
            return Ok(Some(value));
        }

        if let Some(value) = builtin_bound_type_attribute_value_static(builtin_owner, obj, name)? {
            // `builtin_bound_type_attribute_value_static()` already applies the
            // descriptor semantics appropriate for type-level attributes such as
            // `__new__`, `maketrans`, and classmethods. Primitive instance
            // lookup must not bind those results a second time or we turn
            // stable type descriptors into fresh bound builtins on every access.
            return Ok(Some(value));
        }
    }

    Ok(None)
}

#[inline]
fn user_defined_instance_type_name(type_id: TypeId) -> Arc<str> {
    global_class(ClassId(type_id.raw()))
        .map(|class| Arc::<str>::from(class.name().as_str()))
        .unwrap_or_else(|| Arc::<str>::from(type_id.name()))
}

#[inline]
fn user_defined_attribute_error(type_id: TypeId, name: &InternedString) -> RuntimeError {
    RuntimeError::attribute_error(user_defined_instance_type_name(type_id), name.as_str())
}

fn lookup_user_defined_instance_attribute_default(
    vm: &mut VirtualMachine,
    obj: Value,
    ptr: *const (),
    type_id: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if name.as_str() == "__class__" {
        return Ok(class_id_to_value(ClassId(type_id.raw())));
    }

    let class_attr = lookup_instance_class_attr(type_id, name);
    if let Some(descriptor) = class_attr.and_then(property_descriptor_from_value) {
        return invoke_property_getter(vm, descriptor, obj).map(Some);
    }

    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    if let Some(value) = shaped.get_property_interned(name) {
        return Ok(Some(value));
    }

    if let Some(value) = class_attr {
        return Ok(Some(bind_instance_attribute(value, obj)));
    }

    if let Some(value) = lookup_builtin_base_instance_attr(obj, type_id, name)? {
        return Ok(Some(value));
    }

    Ok(None)
}

fn invoke_user_defined_getattr(
    vm: &mut VirtualMachine,
    obj: Value,
    type_id: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if name.as_str() == "__getattr__" {
        return Ok(None);
    }

    let Some(getattr_value) = lookup_instance_class_attr(type_id, &intern("__getattr__")) else {
        return Ok(None);
    };

    crate::ops::calls::invoke_callable_value(
        vm,
        bind_instance_attribute(getattr_value, obj),
        &[Value::string(name.clone())],
    )
    .map(Some)
}

fn lookup_user_defined_instance_attribute(
    vm: &mut VirtualMachine,
    obj: Value,
    ptr: *const (),
    type_id: TypeId,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    if let Some(getattribute_value) =
        lookup_instance_class_attr(type_id, &intern("__getattribute__"))
    {
        let getattribute = bind_instance_attribute(getattribute_value, obj);
        return match crate::ops::calls::invoke_callable_value(
            vm,
            getattribute,
            &[Value::string(name.clone())],
        ) {
            Ok(value) => Ok(value),
            Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
                if let Some(value) = invoke_user_defined_getattr(vm, obj, type_id, name)? {
                    return Ok(value);
                }
                Err(err)
            }
            Err(err) => Err(err),
        };
    }

    match lookup_user_defined_instance_attribute_default(vm, obj, ptr, type_id, name) {
        Ok(Some(value)) => Ok(value),
        Ok(None) => {
            if let Some(value) = invoke_user_defined_getattr(vm, obj, type_id, name)? {
                return Ok(value);
            }
            Err(user_defined_attribute_error(type_id, name))
        }
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            if let Some(value) = invoke_user_defined_getattr(vm, obj, type_id, name)? {
                return Ok(value);
            }
            Err(err)
        }
        Err(err) => Err(err),
    }
}

pub(crate) fn object_getattribute_default(
    vm: &mut VirtualMachine,
    obj: Value,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        if type_id == TypeId::OBJECT {
            if name.as_str() == "__class__" {
                return Ok(crate::builtins::builtin_type_object_for_type_id(
                    TypeId::OBJECT,
                ));
            }

            let shaped = unsafe { &*(ptr as *const ShapedObject) };
            if let Some(value) = shaped.get_property_interned(name) {
                return Ok(value);
            }

            if let Some(value) = builtin_instance_attribute_value(vm, TypeId::OBJECT, obj, name)? {
                return Ok(value);
            }

            return Err(RuntimeError::attribute_error("object", name.as_str()));
        }

        if is_user_defined_type(type_id) {
            return lookup_user_defined_instance_attribute_default(vm, obj, ptr, type_id, name)?
                .ok_or_else(|| user_defined_attribute_error(type_id, name));
        }
    }

    get_attribute_value(vm, obj, name)
}

pub(crate) fn get_attribute_value(
    vm: &mut VirtualMachine,
    obj: Value,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    if let Some(ptr) = obj.as_object_ptr() {
        if let Some(module) = vm.import_resolver.module_from_ptr(ptr) {
            return module
                .get_attr(name.as_str())
                .ok_or_else(|| RuntimeError::attribute_error("module", name.as_str()));
        }

        let type_id = extract_type_id(ptr);

        if let Some(value) = builtin_instance_method_attr_value(obj, type_id, name) {
            return Ok(value);
        }

        return match type_id {
            TypeId::OBJECT => {
                if name.as_str() == "__class__" {
                    return Ok(crate::builtins::builtin_type_object_for_type_id(
                        TypeId::OBJECT,
                    ));
                }

                let shaped = unsafe { &*(ptr as *const ShapedObject) };
                if let Some(value) = shaped.get_property_interned(name) {
                    return Ok(value);
                }

                if let Some(value) =
                    builtin_instance_attribute_value(vm, TypeId::OBJECT, obj, name)?
                {
                    return Ok(value);
                }

                Err(RuntimeError::attribute_error("object", name.as_str()))
            }
            TypeId::DEQUE => Err(RuntimeError::attribute_error("deque", name.as_str())),
            TypeId::REGEX_PATTERN => crate::stdlib::re::pattern_attr_value(vm, obj, name)?
                .ok_or_else(|| RuntimeError::attribute_error("Pattern", name.as_str())),
            TypeId::REGEX_MATCH => crate::stdlib::re::match_attr_value(vm, obj, name)?
                .ok_or_else(|| RuntimeError::attribute_error("Match", name.as_str())),
            TypeId::DICT => Err(RuntimeError::attribute_error("dict", name.as_str())),
            TypeId::LIST => Err(RuntimeError::attribute_error("list", name.as_str())),
            TypeId::TUPLE => Err(RuntimeError::attribute_error("tuple", name.as_str())),
            TypeId::SET => Err(RuntimeError::attribute_error("set", name.as_str())),
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                function_attr_value_in_vm(vm, ptr, func, name)?
                    .ok_or_else(|| RuntimeError::attribute_error("function", name.as_str()))
            }
            TypeId::METHOD => {
                let method = unsafe { &*(ptr as *const BoundMethod) };
                method_attr_value_in_vm(vm, method, name)?
                    .ok_or_else(|| RuntimeError::attribute_error("method", name.as_str()))
            }
            TypeId::CODE => {
                let code = unsafe { &*(ptr as *const CodeObjectView) };
                code_attr_value_in_vm(vm, obj, code, name)?
                    .ok_or_else(|| RuntimeError::attribute_error("code", name.as_str()))
            }
            TypeId::EXCEPTION => {
                let exc = unsafe { &*(ptr as *const ExceptionValue) };
                match name.as_str() {
                    "__class__" => Ok(crate::builtins::exception_type_value_for_id(
                        exc.exception_type_id,
                    )
                    .unwrap_or_else(|| {
                        crate::builtins::builtin_type_object_for_type_id(TypeId::EXCEPTION)
                    })),
                    "args" => {
                        let items = if let Some(args) = exc.args.as_deref() {
                            args.to_vec()
                        } else if let Some(message) = exc.message() {
                            vec![Value::string(intern(message))]
                        } else {
                            Vec::new()
                        };
                        alloc_heap_value(vm, TupleObject::from_vec(items), "exception args tuple")
                    }
                    "msg" => Ok(exc
                        .message()
                        .map(|message| Value::string(intern(message)))
                        .unwrap_or_else(Value::none)),
                    "name" => Ok(exc
                        .import_name
                        .as_deref()
                        .map(|import_name| Value::string(intern(import_name)))
                        .unwrap_or_else(Value::none)),
                    "path" => Ok(exc
                        .import_path
                        .as_deref()
                        .map(|import_path| Value::string(intern(import_path)))
                        .unwrap_or_else(Value::none)),
                    "__traceback__" => Ok(exc.traceback().unwrap_or_else(Value::none)),
                    "__cause__" => Ok(exc
                        .cause
                        .map(|cause| Value::object_ptr(cause as *const ()))
                        .unwrap_or_else(Value::none)),
                    "__context__" => Ok(exc
                        .context
                        .map(|context| Value::object_ptr(context as *const ()))
                        .unwrap_or_else(Value::none)),
                    "__suppress_context__" => {
                        Ok(Value::bool(exc.flags.has(ExceptionFlags::SUPPRESS_CONTEXT)))
                    }
                    _ => Err(RuntimeError::attribute_error(
                        exc.type_name(),
                        name.as_str(),
                    )),
                }
            }
            TypeId::EXCEPTION_TYPE => {
                let exc_type = unsafe { &*(ptr as *const ExceptionTypeObject) };
                if let Some(value) = exception_type_attribute_value(exc_type, name) {
                    return Ok(value);
                }

                if let Some(proxy_class_id) = exception_proxy_class_id_from_ptr(ptr)
                    && let Some(proxy_class) = global_class(proxy_class_id)
                    && let Some(value) =
                        heap_type_attribute_value(vm, Arc::as_ptr(&proxy_class), name)?
                {
                    return Ok(value);
                }

                Err(RuntimeError::attribute_error(
                    exc_type.name(),
                    name.as_str(),
                ))
            }
            TypeId::TRACEBACK => {
                let traceback = unsafe { &*(ptr as *const TracebackViewObject) };
                match name.as_str() {
                    "tb_frame" => Ok(traceback.frame()),
                    "tb_next" => Ok(traceback.next().unwrap_or_else(Value::none)),
                    "tb_lineno" => {
                        Ok(Value::int(traceback.line_number() as i64).unwrap_or_else(Value::none))
                    }
                    "tb_lasti" => {
                        Ok(Value::int(traceback.lasti() as i64).unwrap_or_else(Value::none))
                    }
                    _ => Err(RuntimeError::attribute_error("traceback", name.as_str())),
                }
            }
            TypeId::FRAME => {
                let frame = unsafe { &*(ptr as *const FrameViewObject) };
                match name.as_str() {
                    "f_code" => {
                        let Some(code) = frame.code() else {
                            return Ok(Value::none());
                        };
                        alloc_heap_value(
                            vm,
                            CodeObjectView::new(Arc::clone(code)),
                            "frame code view",
                        )
                    }
                    "f_lineno" => {
                        Ok(Value::int(frame.line_number() as i64).unwrap_or_else(Value::none))
                    }
                    "f_lasti" => Ok(Value::int(frame.lasti() as i64).unwrap_or_else(Value::none)),
                    "f_globals" => Ok(frame.globals()),
                    "f_locals" => Ok(frame.locals()),
                    "f_back" => Ok(Value::none()),
                    _ => Err(RuntimeError::attribute_error("frame", name.as_str())),
                }
            }
            TypeId::CELL_VIEW => {
                let cell = unsafe { &*(ptr as *const CellViewObject) }.cell();
                match name.as_str() {
                    "cell_contents" => cell
                        .get()
                        .ok_or_else(|| RuntimeError::value_error("Cell is empty")),
                    _ => Err(RuntimeError::attribute_error("cell", name.as_str())),
                }
            }
            TypeId::SUPER => super_attribute_value_static(obj, name)?
                .ok_or_else(|| RuntimeError::attribute_error("super", name.as_str())),
            TypeId::TYPE => {
                if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr)
                    && let Some(value) =
                        builtin_bound_type_attribute_value(vm, represented, obj, name)?
                {
                    return Ok(value);
                }

                if let Some(class) = class_object_from_type_ptr(ptr) {
                    if let Some(value) =
                        heap_type_attribute_value(vm, ptr as *const PyClassObject, name)?
                    {
                        return Ok(value);
                    }

                    if let Some(value) = lookup_class_metaclass_attr(class, name) {
                        return Ok(bind_instance_attribute(value, obj));
                    }
                }

                if let Some(value) =
                    builtin_bound_type_attribute_value(vm, TypeId::TYPE, obj, name)?
                {
                    return Ok(value);
                }

                Err(RuntimeError::attribute_error("type", name.as_str()))
            }
            _ => {
                if let Some(value) = descriptor_attr_value(obj, name) {
                    return Ok(value);
                }

                if is_user_defined_type(type_id) {
                    return lookup_user_defined_instance_attribute(vm, obj, ptr, type_id, name);
                }

                Err(RuntimeError::attribute_error(type_id.name(), name.as_str()))
            }
        };
    }

    if let Some(owner) = primitive_owner_type_id(obj) {
        return lookup_builtin_primitive_attr(vm, obj, owner, name)?.ok_or_else(|| {
            RuntimeError::attribute_error(primitive_attribute_type_name(obj), name.as_str())
        });
    }

    Err(RuntimeError::attribute_error(
        primitive_attribute_type_name(obj),
        name.as_str(),
    ))
}

pub(crate) fn set_attribute_value(
    vm: &mut VirtualMachine,
    obj: Value,
    name: &InternedString,
    value: Value,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = obj.as_object_ptr() {
        if let Some(module) = vm.import_resolver.module_from_ptr(ptr) {
            module.set_attr(name.as_str(), value);
            return Ok(());
        }

        let type_id = extract_type_id(ptr);
        return match type_id {
            TypeId::OBJECT => {
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                shaped.set_property(name.clone(), value, shape_registry());
                Ok(())
            }
            TypeId::DEQUE
            | TypeId::REGEX_PATTERN
            | TypeId::REGEX_MATCH
            | TypeId::DICT
            | TypeId::LIST
            | TypeId::TUPLE
            | TypeId::SET
            | TypeId::METHOD => Err(RuntimeError::attribute_error(
                type_id.name(),
                format!(
                    "'{}' object attribute '{}' is read-only",
                    type_id.name(),
                    name
                ),
            )),
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                func.set_attr(name.clone(), value);
                Ok(())
            }
            TypeId::EXCEPTION => {
                let exc = unsafe {
                    ExceptionValue::from_value_mut(obj)
                        .ok_or_else(|| RuntimeError::internal("expected exception instance"))?
                };
                match name.as_str() {
                    "__traceback__" => exc
                        .replace_traceback(value)
                        .map_err(RuntimeError::type_error),
                    _ => Err(RuntimeError::attribute_error(
                        exc.type_name(),
                        name.as_str(),
                    )),
                }
            }
            TypeId::TYPE => {
                let Some(class) = class_object_from_type_ptr(ptr) else {
                    return Err(RuntimeError::attribute_error(
                        "type",
                        format!("'type' object has no attribute '{}'", name),
                    ));
                };
                class.set_attr(name.clone(), value);
                method_cache().invalidate_type_hierarchy(class.class_type_id());
                Ok(())
            }
            _ => {
                if matches!(
                    type_id,
                    TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY
                ) {
                    return Err(RuntimeError::attribute_error(
                        type_id.name(),
                        format!(
                            "'{}' object attribute '{}' is read-only",
                            type_id.name(),
                            name
                        ),
                    ));
                }

                if is_user_defined_type(type_id) {
                    if let Some(descriptor) = lookup_instance_class_attr(type_id, name)
                        .filter(|descriptor| is_property_descriptor_value(*descriptor))
                        .and_then(property_descriptor_from_value)
                    {
                        invoke_property_setter(vm, descriptor, obj, value)?;
                        return Ok(());
                    }

                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    shaped.set_property(name.clone(), value, shape_registry());
                    return Ok(());
                }

                Err(RuntimeError::attribute_error(
                    type_id.name(),
                    format!("'{}' object has no attribute '{}'", type_id.name(), name),
                ))
            }
        };
    }

    let type_name = primitive_attribute_type_name(obj);
    Err(RuntimeError::attribute_error(
        type_name,
        format!("'{}' object has no attribute '{}'", type_name, name),
    ))
}

pub(crate) fn delete_attribute_value(
    vm: &mut VirtualMachine,
    obj: Value,
    name: &InternedString,
) -> Result<(), RuntimeError> {
    if let Some(ptr) = obj.as_object_ptr() {
        if let Some(module) = vm.import_resolver.module_from_ptr(ptr) {
            return if module.del_attr(name.as_str()) {
                Ok(())
            } else {
                Err(RuntimeError::attribute_error("module", name.as_str()))
            };
        }

        let type_id = extract_type_id(ptr);
        return match type_id {
            TypeId::OBJECT => {
                let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                if shaped.delete_property_interned(name) {
                    Ok(())
                } else {
                    Err(RuntimeError::attribute_error(
                        "object",
                        format!("'object' object has no attribute '{}'", name),
                    ))
                }
            }
            TypeId::DEQUE
            | TypeId::REGEX_PATTERN
            | TypeId::REGEX_MATCH
            | TypeId::DICT
            | TypeId::LIST
            | TypeId::TUPLE
            | TypeId::SET
            | TypeId::METHOD => Err(RuntimeError::attribute_error(
                type_id.name(),
                format!(
                    "cannot delete attribute '{}' of '{}' object",
                    name,
                    type_id.name()
                ),
            )),
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                if func.del_attr(name).is_some() {
                    Ok(())
                } else {
                    Err(RuntimeError::attribute_error("function", name.as_str()))
                }
            }
            TypeId::TYPE => {
                let Some(class) = class_object_from_type_ptr(ptr) else {
                    return Err(RuntimeError::attribute_error("type", name.as_str()));
                };
                if class.del_attr(name).is_some() {
                    method_cache().invalidate_type_hierarchy(class.class_type_id());
                    Ok(())
                } else {
                    Err(RuntimeError::attribute_error("type", name.as_str()))
                }
            }
            _ => {
                if matches!(
                    type_id,
                    TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY
                ) {
                    return Err(RuntimeError::attribute_error(type_id.name(), name.as_str()));
                }

                if is_user_defined_type(type_id) {
                    if let Some(descriptor) = lookup_instance_class_attr(type_id, name)
                        .filter(|descriptor| is_property_descriptor_value(*descriptor))
                        .and_then(property_descriptor_from_value)
                    {
                        invoke_property_deleter(vm, descriptor, obj)?;
                        return Ok(());
                    }

                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    if shaped.delete_property_interned(name) {
                        return Ok(());
                    }
                }

                Err(RuntimeError::attribute_error(
                    type_id.name(),
                    format!("'{}' object has no attribute '{}'", type_id.name(), name),
                ))
            }
        };
    }

    let type_name = primitive_attribute_type_name(obj);
    Err(RuntimeError::attribute_error(
        type_name,
        format!("'{}' object has no attribute '{}'", type_name, name),
    ))
}

/// GetAttr: dst = src.attr[name_idx]
///
/// Attribute lookup follows Python's descriptor protocol with Shape optimization:
/// 1. Check instance Shape for property (O(1) via hidden class)
/// 2. Fall back to type lookup for methods/class attributes
/// 3. Raise AttributeError if not found
///
/// Supports:
/// - OBJECT: ShapedObject with hidden class optimization
/// - List/Dict/Tuple: Built-in method dispatch (future)
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn get_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (obj, name) = {
        let frame = vm.current_frame();
        let obj = frame.get_reg(inst.src1().0);
        let name_idx = inst.src2().0 as u16;
        let name = frame.get_name(name_idx).clone();
        (obj, name)
    };
    match get_attribute_value(vm, obj, &intern(&*name)) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// SetAttr: src1.attr[name_idx] = src2
///
/// Sets an attribute on an object. This may cause a Shape transition
/// if the property is new.
///
/// Supports:
/// - OBJECT: ShapedObject with Shape transition support
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn set_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (obj, value, name) = {
        let frame = vm.current_frame();
        let obj = frame.get_reg(inst.dst().0);
        let value = frame.get_reg(inst.src2().0);
        let name_idx = inst.src1().0 as u16;
        let name = frame.get_name(name_idx).clone();
        (obj, value, name)
    };
    match set_attribute_value(vm, obj, &intern(&*name), value) {
        Ok(()) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

/// DelAttr: del src.attr[name_idx]
///
/// Deletes an attribute from an object.
///
/// Note: This doesn't change the object's Shape - the slot is just set to None.
/// A more sophisticated implementation could use "delete shapes" like V8 does
/// for objects that frequently have properties deleted.
///
/// Supports:
/// - OBJECT: ShapedObject with slot-based deletion
/// - Custom types: User-defined objects with __dict__
#[inline(always)]
pub fn del_attr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (obj, name) = {
        let frame = vm.current_frame();
        let obj = frame.get_reg(inst.src1().0);
        let name_idx = inst.src2().0 as u16;
        let name = frame.get_name(name_idx).clone();
        (obj, name)
    };
    match delete_attribute_value(vm, obj, &intern(&*name)) {
        Ok(()) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Item Access (Type-Discriminated)
// =============================================================================

/// GetItem: dst = src1[src2]
///
/// Supports list/tuple (integer index) and dict (any hashable key).
/// Uses TypeId dispatch for correct type handling.
#[inline(always)]
pub fn get_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);
    let dst = inst.dst().0;

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        if let Some(list) = list_storage_ref_from_ptr(ptr) {
            return if let Some(idx) = key.as_int() {
                if let Some(val) = list.get(idx) {
                    frame.set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                }
            } else {
                ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
            };
        }

        match type_id {
            TypeId::TUPLE => {
                let tuple = unsafe { &*(ptr as *const TupleObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = tuple.get(idx) {
                        frame.set_reg(dst, val);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, tuple.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("tuple indices must be integers"))
                }
            }
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                if let Some(val) = dict.get(key) {
                    frame.set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_ref_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object is not subscriptable",
                    ));
                };
                if let Some(val) = dict.get(key) {
                    frame.set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = range.get_value(idx) {
                        frame.set_reg(dst, val);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(
                            idx,
                            range.try_len().unwrap_or(usize::MAX),
                        ))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("range indices must be integers"))
                }
            }
            _ => ControlFlow::Error(RuntimeError::type_error("object is not subscriptable")),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error("object is not subscriptable"))
    }
}

/// SetItem: src1[dst] = src2 (dst is key register)
///
/// Sets items in mutable containers (list, dict).
#[inline(always)]
pub fn set_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        if let Some(list) = list_storage_mut_from_ptr(ptr) {
            return if let Some(idx) = key.as_int() {
                if list.set(idx, value) {
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                }
            } else {
                ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
            };
        }

        match type_id {
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                dict.set(key, value);
                ControlFlow::Continue
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_mut_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object does not support item assignment",
                    ));
                };
                dict.set(key, value);
                ControlFlow::Continue
            }
            TypeId::TUPLE => ControlFlow::Error(RuntimeError::type_error(
                "'tuple' object does not support item assignment",
            )),
            _ => ControlFlow::Error(RuntimeError::type_error(
                "object does not support item assignment",
            )),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error(
            "object does not support item assignment",
        ))
    }
}

/// DelItem: del src1[src2]
#[inline(always)]
pub fn del_item(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let container = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.src2().0);

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        if let Some(list) = list_storage_mut_from_ptr(ptr) {
            return if let Some(idx) = key.as_int() {
                if list.remove(idx).is_some() {
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                }
            } else {
                ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
            };
        }

        match type_id {
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                if dict.remove(key).is_some() {
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_mut_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object does not support item deletion",
                    ));
                };
                if dict.remove(key).is_some() {
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::key_error("key not found"))
                }
            }
            TypeId::TUPLE => ControlFlow::Error(RuntimeError::type_error(
                "'tuple' object does not support item deletion",
            )),
            _ => ControlFlow::Error(RuntimeError::type_error(
                "object does not support item deletion",
            )),
        }
    } else {
        ControlFlow::Error(RuntimeError::type_error(
            "object does not support item deletion",
        ))
    }
}

// =============================================================================
// Iteration (Type-Discriminated)
// =============================================================================

/// GetIter: dst = iter(src)
///
/// Creates an iterator for the given object.
/// Uses TypeId dispatch for type-specific optimized iterators.
#[inline(always)]
pub fn get_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let obj = vm.current_frame().get_reg(inst.src1().0);
    let dst = inst.dst().0;

    match ensure_iterator_value(vm, obj) {
        Ok(iterator) => {
            vm.current_frame_mut().set_reg(dst, iterator);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// ForIter: dst = next(src), jump if StopIteration
///
/// Advances the iterator and jumps to offset if exhausted.
#[inline(always)]
pub fn for_iter(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let Some(iter_reg) = dst.checked_sub(1) else {
        return ControlFlow::Error(RuntimeError::internal(
            "ForIter destination register must follow its iterator register",
        ));
    };
    let iter_val = vm.current_frame().get_reg(iter_reg);
    let offset = inst.imm16() as i16;

    match next_step(vm, iter_val) {
        Ok(IterStep::Yielded(value)) => {
            vm.current_frame_mut().set_reg(dst, value);
            ControlFlow::Continue
        }
        Ok(IterStep::Exhausted) => ControlFlow::Jump(offset),
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Utilities
// =============================================================================

/// Len: dst = len(src)
///
/// Returns the length of a container.
#[inline(always)]
pub fn len(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);
    let dst = inst.dst().0;

    if let Some(string) = prism_runtime::types::string::value_as_string_ref(obj) {
        let value = Value::int(string.len() as i64).unwrap_or_else(Value::none);
        frame.set_reg(dst, value);
        ControlFlow::Continue
    } else if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        let len_val = if let Some(list) = list_storage_ref_from_ptr(ptr) {
            list.len() as i64
        } else {
            match type_id {
                TypeId::DEQUE => {
                    let deque = unsafe { &*(ptr as *const DequeObject) };
                    deque.len() as i64
                }
                TypeId::TUPLE => {
                    let tuple = unsafe { &*(ptr as *const TupleObject) };
                    tuple.len() as i64
                }
                TypeId::DICT => {
                    let dict = unsafe { &*(ptr as *const DictObject) };
                    dict.len() as i64
                }
                TypeId::SET => {
                    let set = unsafe { &*(ptr as *const SetObject) };
                    set.len() as i64
                }
                TypeId::RANGE => {
                    let range = unsafe { &*(ptr as *const RangeObject) };
                    let Some(len) = range.try_len() else {
                        return ControlFlow::Error(RuntimeError::new(
                            crate::error::RuntimeErrorKind::OverflowError {
                                message: "range length overflow".into(),
                            },
                        ));
                    };
                    len as i64
                }
                _ => {
                    return ControlFlow::Error(RuntimeError::type_error(format!(
                        "object of type '{}' has no len()",
                        type_id.name()
                    )));
                }
            }
        };

        let value = Value::int(len_val).unwrap_or_else(Value::none);
        frame.set_reg(dst, value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("object has no len()"))
    }
}

/// IsCallable: dst = callable(src)
#[inline(always)]
pub fn is_callable(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let obj = frame.get_reg(inst.src1().0);

    let is_callable = crate::ops::calls::value_supports_call_protocol(obj);

    frame.set_reg(inst.dst().0, Value::bool(is_callable));
    ControlFlow::Continue
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::builtins::BuiltinFunctionObject;
    use crate::frame::ClosureEnv;
    use crate::import::ModuleObject;
    use prism_compiler::bytecode::{
        CodeFlags, CodeObject, Instruction, LineTableEntry, Opcode, Register,
    };
    use prism_core::Value;
    use prism_core::intern::intern;
    use prism_runtime::object::ObjectHeader;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::descriptor::{BoundMethod, PropertyDescriptor};
    use prism_runtime::object::mro::ClassId;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
    };
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::object::views::{CodeObjectView, FrameViewObject};
    use prism_runtime::types::Cell;
    use prism_runtime::types::dict::DictObject;
    use prism_runtime::types::function::FunctionObject;
    use prism_runtime::types::set::SetObject;
    use prism_runtime::types::string::StringObject;
    use std::sync::Arc;

    fn vm_with_frame() -> VirtualMachine {
        let mut vm = VirtualMachine::new();
        let code = Arc::new(CodeObject::new("test_len", "<test>"));
        vm.push_frame(code, 0).expect("frame push failed");
        vm
    }

    fn boxed_value<T>(obj: T) -> (Value, *mut T) {
        let ptr = Box::into_raw(Box::new(obj));
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_boxed<T>(ptr: *mut T) {
        drop(unsafe { Box::from_raw(ptr) });
    }

    fn vm_with_names(names: &[&str]) -> VirtualMachine {
        let mut code = CodeObject::new("test_attrs", "<test>");
        code.names = names
            .iter()
            .map(|name| Arc::<str>::from(*name))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0).expect("frame push failed");
        vm
    }

    fn class_value(class: PyClassObject) -> (Value, *const PyClassObject) {
        let class = Arc::new(class);
        let ptr = Arc::into_raw(class);
        (Value::object_ptr(ptr as *const ()), ptr)
    }

    unsafe fn drop_class(ptr: *const PyClassObject) {
        drop(unsafe { Arc::from_raw(ptr) });
    }

    fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
        let mut bitmap = SubclassBitmap::new();
        for &class_id in class.mro() {
            bitmap.set_bit(prism_runtime::object::type_obj::TypeId::from_raw(
                class_id.0,
            ));
        }

        let class = Arc::new(class);
        register_global_class(class.clone(), bitmap);
        class
    }

    fn make_test_function_value(name: &str) -> (*mut FunctionObject, Value) {
        let mut code = CodeObject::new(name, "<test>");
        code.register_count = 8;
        let func = Box::new(FunctionObject::new(
            Arc::new(code),
            Arc::from(name),
            None,
            None,
        ));
        let ptr = Box::into_raw(func);
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    fn instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
        let ptr = Box::into_raw(Box::new(ShapedObject::new(
            class.class_type_id(),
            class.instance_shape().clone(),
        )));
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    fn dict_backed_instance_value(class: &Arc<PyClassObject>) -> (*mut ShapedObject, Value) {
        let ptr = Box::into_raw(Box::new(ShapedObject::new_dict_backed(
            class.class_type_id(),
            class.instance_shape().clone(),
        )));
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
        let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
            if id.0 < TypeId::FIRST_USER_TYPE {
                Some(
                    builtin_class_mro(class_id_to_type_id(id))
                        .into_iter()
                        .collect(),
                )
            } else {
                None
            }
        })
        .expect("dict subclass should build");
        register_test_class(class)
    }

    fn property_storage_getter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
        if args.len() != 1 {
            return Err(crate::builtins::BuiltinError::TypeError(format!(
                "getter expected 1 argument, got {}",
                args.len()
            )));
        }
        let ptr = args[0].as_object_ptr().ok_or_else(|| {
            crate::builtins::BuiltinError::TypeError("getter requires object receiver".to_string())
        })?;
        let shaped = unsafe { &*(ptr as *const ShapedObject) };
        shaped.get_property("_value").ok_or_else(|| {
            crate::builtins::BuiltinError::AttributeError("_value missing".to_string())
        })
    }

    fn property_storage_setter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
        if args.len() != 2 {
            return Err(crate::builtins::BuiltinError::TypeError(format!(
                "setter expected 2 arguments, got {}",
                args.len()
            )));
        }
        let ptr = args[0].as_object_ptr().ok_or_else(|| {
            crate::builtins::BuiltinError::TypeError("setter requires object receiver".to_string())
        })?;
        let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
        shaped.set_property(intern("_value"), args[1], shape_registry());
        Ok(Value::none())
    }

    fn property_storage_deleter(args: &[Value]) -> Result<Value, crate::builtins::BuiltinError> {
        if args.len() != 1 {
            return Err(crate::builtins::BuiltinError::TypeError(format!(
                "deleter expected 1 argument, got {}",
                args.len()
            )));
        }
        let ptr = args[0].as_object_ptr().ok_or_else(|| {
            crate::builtins::BuiltinError::TypeError("deleter requires object receiver".to_string())
        })?;
        let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
        shaped.delete_property("_value");
        Ok(Value::none())
    }

    fn builtin_function_value(
        name: &str,
        func: fn(&[Value]) -> Result<Value, crate::builtins::BuiltinError>,
    ) -> (*mut BuiltinFunctionObject, Value) {
        let builtin = Box::new(BuiltinFunctionObject::new(Arc::from(name), func));
        let ptr = Box::into_raw(builtin);
        (ptr, Value::object_ptr(ptr as *const ()))
    }

    #[test]
    fn test_extract_type_id() {
        // Create a list and verify TypeId extraction
        let list = Box::new(ListObject::new());
        let ptr = Box::into_raw(list) as *const ();

        let type_id = extract_type_id(ptr);
        assert_eq!(type_id, TypeId::LIST);

        // Clean up
        unsafe {
            drop(Box::from_raw(ptr as *mut ListObject));
        }
    }

    #[test]
    fn test_type_id_layout() {
        // Verify ObjectHeader layout is correct for JIT compatibility
        assert_eq!(std::mem::offset_of!(ObjectHeader, type_id), 0);
        assert_eq!(std::mem::size_of::<TypeId>(), 4);
        assert_eq!(std::mem::size_of::<ObjectHeader>(), 16);
    }

    #[test]
    fn test_len_opcode_tagged_string() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("hello")));

        let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
        assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(5));
    }

    #[test]
    fn test_len_opcode_set_object() {
        let mut vm = vm_with_frame();
        let set = SetObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
            Value::int(2).unwrap(),
            Value::int(3).unwrap(),
        ]);
        let (set_value, ptr) = boxed_value(set);
        vm.current_frame_mut().set_reg(1, set_value);

        let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
        assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(3));

        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_opcode_string_object() {
        let mut vm = vm_with_frame();
        let (string_value, ptr) = boxed_value(StringObject::new("runtime"));
        vm.current_frame_mut().set_reg(1, string_value);

        let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
        assert!(matches!(len(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(7));

        unsafe { drop_boxed(ptr) };
    }

    #[test]
    fn test_len_opcode_type_error_for_int() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::int(42).unwrap());

        let inst = Instruction::op_ds(Opcode::Len, Register::new(2), Register::new(1));
        let flow = len(&mut vm, inst);
        assert!(matches!(flow, ControlFlow::Error(_)));
    }

    #[test]
    fn test_get_attr_reads_class_attributes() {
        let mut vm = vm_with_names(&["field"]);
        let class = PyClassObject::new_simple(intern("Example"));
        class.set_attr(intern("field"), Value::int(99).unwrap());
        let (class_value, class_ptr) = class_value(class);
        vm.current_frame_mut().set_reg(1, class_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(99));

        unsafe { drop_class(class_ptr) };
    }

    #[test]
    fn test_get_attr_reads_inherited_class_attributes() {
        let mut parent = PyClassObject::new_simple(intern("Parent"));
        parent.set_attr(intern("field"), Value::int(123).unwrap());
        let parent = register_test_class(parent);

        let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
            (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
        })
        .expect("child class should build");
        let child = register_test_class(child);

        let mut vm = vm_with_names(&["field"]);
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(Arc::as_ptr(&child) as *const ()));

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(123));
    }

    #[test]
    fn test_get_attr_binds_inherited_instance_methods() {
        let (func_ptr, func_value) = make_test_function_value("method");

        let mut parent = PyClassObject::new_simple(intern("Parent"));
        parent.set_attr(intern("method"), func_value);
        let parent = register_test_class(parent);

        let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
            (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
        })
        .expect("child class should build");
        let child = register_test_class(child);

        let (instance_ptr, instance_value) = instance_value(&child);

        let mut vm = vm_with_names(&["method"]);
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let bound_value = vm.current_frame().get_reg(2);
        let bound_ptr = bound_value
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        assert_eq!(
            extract_type_id(bound_ptr),
            prism_runtime::object::type_obj::TypeId::METHOD
        );

        let bound = unsafe { &*(bound_ptr as *const BoundMethod) };
        assert_eq!(bound.function(), func_value);
        assert_eq!(bound.instance(), instance_value);

        unsafe {
            drop_boxed(instance_ptr);
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attr_binds_builtin_list_methods_as_builtin_functions() {
        let mut vm = vm_with_names(&["append"]);
        let (list_value, list_ptr) = boxed_value(ListObject::new());
        vm.current_frame_mut().set_reg(1, list_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("builtin method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "list.append");
        let result = builtin
            .call(&[Value::int(9).unwrap()])
            .expect("append should call");
        assert!(result.is_none());

        let list = unsafe { &*list_ptr.cast::<ListObject>() };
        assert_eq!(list.len(), 1);
        assert_eq!(list.as_slice()[0].as_int(), Some(9));
    }

    #[test]
    fn test_get_attr_binds_builtin_frozenset_contains_method() {
        let mut vm = vm_with_names(&["__contains__"]);
        let mut frozenset =
            SetObject::from_slice(&[Value::int(3).unwrap(), Value::int(5).unwrap()]);
        frozenset.header.type_id = TypeId::FROZENSET;
        let (frozenset_value, frozenset_ptr) = boxed_value(frozenset);
        vm.current_frame_mut().set_reg(1, frozenset_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("builtin method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "frozenset.__contains__");
        let present = builtin
            .call(&[Value::int(5).unwrap()])
            .expect("membership call should succeed");
        let missing = builtin
            .call(&[Value::int(9).unwrap()])
            .expect("membership call should succeed");
        assert_eq!(present.as_bool(), Some(true));
        assert_eq!(missing.as_bool(), Some(false));

        unsafe {
            drop_boxed(frozenset_ptr);
        }
    }

    #[test]
    fn test_get_attribute_value_exposes_function_get_descriptor() {
        let mut vm = vm_with_names(&[]);
        let (func_ptr, func_value) = make_test_function_value("descriptor");

        let method = get_attribute_value(&mut vm, func_value, &intern("__get__"))
            .expect("function objects should expose __get__");
        let method_ptr = method
            .as_object_ptr()
            .expect("function.__get__ should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "function.__get__");
        assert_eq!(builtin.bound_self(), Some(func_value));

        assert_eq!(
            builtin
                .call(&[Value::none()])
                .expect("__get__(None) should return the underlying function"),
            func_value
        );

        let bound_method = builtin
            .call(&[Value::int(7).unwrap()])
            .expect("__get__(instance) should create a bound method");
        let bound_ptr = bound_method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        assert_eq!(extract_type_id(bound_ptr), TypeId::METHOD);

        let bound = unsafe { &*(bound_ptr as *const BoundMethod) };
        assert_eq!(bound.function(), func_value);
        assert_eq!(bound.instance(), Value::int(7).unwrap());

        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attribute_value_exposes_bound_method_metadata() {
        let mut vm = vm_with_names(&[]);
        let (func_ptr, func_value) = make_test_function_value("metadata_method");
        let instance = Value::int(7).unwrap();
        let method_value = bind_instance_attribute(func_value, instance);

        assert_eq!(
            get_attribute_value(&mut vm, method_value, &intern("__self__"))
                .expect("__self__ should resolve"),
            instance
        );
        assert_eq!(
            get_attribute_value(&mut vm, method_value, &intern("__func__"))
                .expect("__func__ should resolve"),
            func_value
        );
        assert_eq!(
            get_attribute_value(&mut vm, method_value, &intern("__name__"))
                .expect("__name__ should resolve"),
            Value::string(intern("metadata_method"))
        );
        assert!(
            get_attribute_value(&mut vm, method_value, &intern("__doc__"))
                .expect("__doc__ should resolve")
                .is_none()
        );

        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attr_binds_builtin_dict_items_method_for_dict_subclass() {
        let mut vm = vm_with_names(&["items"]);
        let class = register_dict_subclass("DictSubclassAttr");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
        unsafe { &mut *instance_ptr }
            .dict_backing_mut()
            .expect("dict backing should exist")
            .set(Value::string(intern("alpha")), Value::int(1).unwrap());
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("builtin method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "dict.items");
        let result = builtin.call(&[]).expect("dict.items should call");
        let result_ptr = result.as_object_ptr().expect("items should return a view");
        assert_eq!(extract_type_id(result_ptr), TypeId::DICT_ITEMS);

        unsafe {
            drop_boxed(instance_ptr);
        }
    }

    #[test]
    fn test_get_attribute_value_binds_builtin_dict_setitem_for_dict_subclass() {
        let mut vm = vm_with_names(&[]);
        let class = register_dict_subclass("DictSubclassSetitem");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

        let method = get_attribute_value(&mut vm, instance_value, &intern("__setitem__"))
            .expect("dict subclass should inherit dict.__setitem__");
        let method_ptr = method
            .as_object_ptr()
            .expect("bound builtin method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "dict.__setitem__");
        builtin
            .call(&[Value::string(intern("beta")), Value::int(7).unwrap()])
            .expect("bound dict.__setitem__ should accept dict subclasses");

        let backing = unsafe { &*instance_ptr }
            .dict_backing()
            .expect("dict backing should exist");
        assert_eq!(
            backing.get(Value::string(intern("beta"))),
            Some(Value::int(7).unwrap())
        );

        unsafe {
            drop_boxed(instance_ptr);
        }
    }

    #[test]
    fn test_get_attribute_value_binds_python_setitem_on_dict_subclass() {
        let mut vm = vm_with_names(&[]);
        let (func_ptr, func_value) = make_test_function_value("__setitem__");

        let mut class = PyClassObject::new(
            intern("PreparedNamespace"),
            &[ClassId(TypeId::DICT.raw())],
            |id| {
                (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                    builtin_class_mro(class_id_to_type_id(id))
                        .into_iter()
                        .collect()
                })
            },
        )
        .expect("dict subclass should build");
        class.set_attr(intern("__setitem__"), func_value);
        let class = register_test_class(class);

        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);

        let method = get_attribute_value(&mut vm, instance_value, &intern("__setitem__"))
            .expect("dict subclass should expose class-defined __setitem__");
        let method_ptr = method
            .as_object_ptr()
            .expect("bound method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::METHOD);

        let bound = unsafe { &*(method_ptr as *const BoundMethod) };
        assert_eq!(bound.function(), func_value);
        assert_eq!(bound.instance(), instance_value);

        unsafe {
            drop_boxed(instance_ptr);
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attribute_value_resolves_object_new_for_none_primitive() {
        let mut vm = vm_with_names(&[]);

        let method = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
            .expect("None should inherit object.__new__");
        let method_ptr = method
            .as_object_ptr()
            .expect("bound builtin method should be heap allocated");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "object.__new__");
    }

    #[test]
    fn test_get_attribute_value_reuses_object_new_for_none_primitive() {
        let mut vm = vm_with_names(&[]);

        let first = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
            .expect("first None.__new__ lookup should succeed");
        let second = get_attribute_value(&mut vm, Value::none(), &intern("__new__"))
            .expect("second None.__new__ lookup should succeed");

        assert_eq!(first, second, "None.__new__ should remain stable");
        assert_eq!(
            first.as_object_ptr(),
            second.as_object_ptr(),
            "None.__new__ should reuse the shared builtin callable",
        );
    }

    #[test]
    fn test_get_attribute_value_reuses_str_maketrans_for_primitive_instance() {
        let mut vm = vm_with_names(&[]);
        let owner = Value::string(intern("seed"));

        let first = get_attribute_value(&mut vm, owner, &intern("maketrans"))
            .expect("first str.maketrans lookup should succeed");
        let second = get_attribute_value(&mut vm, owner, &intern("maketrans"))
            .expect("second str.maketrans lookup should succeed");

        let first_ptr = first
            .as_object_ptr()
            .expect("str.maketrans should be heap allocated");
        let second_ptr = second
            .as_object_ptr()
            .expect("str.maketrans should be heap allocated");
        assert_eq!(first_ptr, second_ptr, "str.maketrans should stay unbound");

        let builtin = unsafe { &*(first_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "str.maketrans");
        assert!(builtin.bound_self().is_none());
    }

    #[test]
    fn test_set_and_get_item_use_dict_backing_for_heap_dict_subclass() {
        let mut vm = vm_with_frame();
        let class = register_dict_subclass("DictSubclassItems");
        let (instance_ptr, instance_value) = dict_backed_instance_value(&class);
        let key = Value::string(intern("key"));
        let value = Value::int(42).unwrap();

        vm.current_frame_mut().set_reg(1, instance_value);
        vm.current_frame_mut().set_reg(2, value);
        vm.current_frame_mut().set_reg(3, key);

        let set_inst = Instruction::op_dss(
            Opcode::SetItem,
            Register::new(3),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(set_item(&mut vm, set_inst), ControlFlow::Continue));

        let get_inst = Instruction::op_dss(
            Opcode::GetItem,
            Register::new(4),
            Register::new(1),
            Register::new(3),
        );
        assert!(matches!(get_item(&mut vm, get_inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(4), value);

        let backing = unsafe { &*instance_ptr }
            .dict_backing()
            .expect("dict backing should exist");
        assert_eq!(backing.get(key), Some(value));

        unsafe {
            drop_boxed(instance_ptr);
        }
    }

    #[test]
    fn test_get_attr_reads_imported_module_attributes_from_registry() {
        let mut vm = vm_with_names(&["iskeyword"]);
        let module = Arc::new(ModuleObject::new("keyword"));
        module.set_attr("iskeyword", Value::bool(true));
        vm.import_resolver.insert_module("keyword", module.clone());
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_bool(), Some(true));
    }

    #[test]
    fn test_get_attr_exposes_import_exception_metadata() {
        let mut vm = vm_with_names(&[]);
        let exc = crate::builtins::create_exception_with_import_details(
            crate::stdlib::exceptions::ExceptionTypeId::ModuleNotFoundError,
            Some(Arc::from("No module named 'pkg.missing'")),
            Some(Arc::from("pkg.missing")),
            None,
        );

        let args_value = get_attribute_value(&mut vm, exc, &intern("args"))
            .expect("exception args should be readable");
        let args_ptr = args_value.as_object_ptr().expect("args should be a tuple");
        let args = unsafe { &*(args_ptr as *const TupleObject) };
        assert_eq!(args.len(), 1);
        assert_eq!(
            args.get(0),
            Some(Value::string(intern("No module named 'pkg.missing'")))
        );

        let class_value = get_attribute_value(&mut vm, exc, &intern("__class__"))
            .expect("__class__ should be readable");
        let class_name = get_attribute_value(&mut vm, class_value, &intern("__name__"))
            .expect("__class__.__name__ should be readable");
        assert_eq!(class_name, Value::string(intern("ModuleNotFoundError")));

        let name_value =
            get_attribute_value(&mut vm, exc, &intern("name")).expect("name should be readable");
        assert_eq!(name_value, Value::string(intern("pkg.missing")));

        let path_value =
            get_attribute_value(&mut vm, exc, &intern("path")).expect("path should be readable");
        assert!(path_value.is_none());
    }

    #[test]
    fn test_get_attr_reads_function_code_view() {
        let (func_ptr, func_value) = make_test_function_value("callable");
        let mut vm = vm_with_names(&["__code__"]);
        vm.current_frame_mut().set_reg(1, func_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let code_ptr = vm.current_frame().get_reg(2).as_object_ptr().unwrap();
        assert_eq!(extract_type_id(code_ptr), TypeId::CODE);

        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attr_exposes_code_object_metadata_used_by_warnings() {
        let mut code = CodeObject::new("warnsite", "warning_probe.py");
        code.qualname = Arc::from("WarningProbe.warnsite");
        code.first_lineno = 27;
        code.arg_count = 2;
        code.posonlyarg_count = 1;
        code.kwonlyarg_count = 1;
        code.flags = CodeFlags::MODULE;
        code.locals = vec![Arc::from("alpha"), Arc::from("beta")].into_boxed_slice();
        code.names = vec![Arc::from("__warningregistry__")].into_boxed_slice();
        code.freevars = vec![Arc::from("captured")].into_boxed_slice();
        code.cellvars = vec![Arc::from("cell")].into_boxed_slice();
        code.constants = vec![Value::none(), Value::int(7).unwrap()].into_boxed_slice();

        let (code_value, code_ptr) = boxed_value(CodeObjectView::new(Arc::new(code)));
        let mut vm = vm_with_frame();

        assert_eq!(
            get_attribute_value(&mut vm, code_value, &intern("co_filename"))
                .expect("co_filename should be readable"),
            Value::string(intern("warning_probe.py"))
        );
        assert_eq!(
            get_attribute_value(&mut vm, code_value, &intern("co_name"))
                .expect("co_name should be readable"),
            Value::string(intern("warnsite"))
        );
        assert_eq!(
            get_attribute_value(&mut vm, code_value, &intern("co_qualname"))
                .expect("co_qualname should be readable"),
            Value::string(intern("WarningProbe.warnsite"))
        );
        assert_eq!(
            get_attribute_value(&mut vm, code_value, &intern("co_firstlineno"))
                .expect("co_firstlineno should be readable")
                .as_int(),
            Some(27)
        );

        let varnames = get_attribute_value(&mut vm, code_value, &intern("co_varnames"))
            .expect("co_varnames should be readable");
        let varnames_ptr = varnames
            .as_object_ptr()
            .expect("co_varnames should be a tuple");
        let varnames = unsafe { &*(varnames_ptr as *const TupleObject) };
        assert_eq!(
            varnames.as_slice(),
            &[
                Value::string(intern("alpha")),
                Value::string(intern("beta"))
            ]
        );

        let consts = get_attribute_value(&mut vm, code_value, &intern("co_consts"))
            .expect("co_consts should be readable");
        let consts_ptr = consts.as_object_ptr().expect("co_consts should be a tuple");
        let consts = unsafe { &*(consts_ptr as *const TupleObject) };
        assert_eq!(consts.as_slice(), &[Value::none(), Value::int(7).unwrap()]);

        unsafe {
            drop_boxed(code_ptr);
        }
    }

    #[test]
    fn test_get_attr_exposes_code_positions_iterator_used_by_traceback() {
        let mut code = CodeObject::new("warnsite", "warning_probe.py");
        code.instructions = vec![
            Instruction::op(Opcode::Nop),
            Instruction::op(Opcode::Nop),
            Instruction::op(Opcode::Nop),
        ]
        .into_boxed_slice();
        code.line_table = vec![
            LineTableEntry {
                start_pc: 0,
                end_pc: 1,
                line: 27,
            },
            LineTableEntry {
                start_pc: 1,
                end_pc: 3,
                line: 31,
            },
        ]
        .into_boxed_slice();

        let (code_value, code_ptr) = boxed_value(CodeObjectView::new(Arc::new(code)));
        let mut vm = vm_with_frame();

        let method = get_attribute_value(&mut vm, code_value, &intern("co_positions"))
            .expect("co_positions should be readable");
        let method_ptr = method
            .as_object_ptr()
            .expect("co_positions should bind a builtin");
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);

        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "code.co_positions");

        let iter_value = builtin
            .call(&[])
            .expect("co_positions() should return an iterator");
        let iter = crate::builtins::get_iterator_mut(&iter_value)
            .expect("co_positions() result should be iterable");

        let first = unsafe {
            &*(iter
                .next()
                .expect("first position should exist")
                .as_object_ptr()
                .expect("position entries should be tuples") as *const TupleObject)
        };
        assert_eq!(
            first.as_slice(),
            &[
                Value::int(27).unwrap(),
                Value::int(27).unwrap(),
                Value::none(),
                Value::none(),
            ]
        );

        let second = unsafe {
            &*(iter
                .next()
                .expect("second position should exist")
                .as_object_ptr()
                .expect("position entries should be tuples") as *const TupleObject)
        };
        assert_eq!(
            second.as_slice(),
            &[
                Value::int(31).unwrap(),
                Value::int(31).unwrap(),
                Value::none(),
                Value::none(),
            ]
        );

        let third = unsafe {
            &*(iter
                .next()
                .expect("third position should exist")
                .as_object_ptr()
                .expect("position entries should be tuples") as *const TupleObject)
        };
        assert_eq!(
            third.as_slice(),
            &[
                Value::int(31).unwrap(),
                Value::int(31).unwrap(),
                Value::none(),
                Value::none(),
            ]
        );
        assert!(iter.next().is_none(), "iterator should be exhausted");

        unsafe {
            drop_boxed(code_ptr);
        }
    }

    #[test]
    fn test_get_attr_exposes_frame_globals_and_locals_snapshots() {
        let mut globals = DictObject::new();
        globals.set(
            Value::string(intern("__name__")),
            Value::string(intern("warning_probe")),
        );
        let (globals_value, globals_ptr) = boxed_value(globals);

        let mut locals = DictObject::new();
        locals.set(Value::string(intern("flag")), Value::bool(true));
        let (locals_value, locals_ptr) = boxed_value(locals);

        let (frame_value, frame_ptr) = boxed_value(FrameViewObject::new(
            None,
            globals_value,
            locals_value,
            19,
            5,
        ));
        let mut vm = vm_with_frame();

        assert_eq!(
            get_attribute_value(&mut vm, frame_value, &intern("f_globals"))
                .expect("f_globals should be readable"),
            globals_value
        );
        assert_eq!(
            get_attribute_value(&mut vm, frame_value, &intern("f_locals"))
                .expect("f_locals should be readable"),
            locals_value
        );
        assert_eq!(
            get_attribute_value(&mut vm, frame_value, &intern("f_lineno"))
                .expect("f_lineno should be readable")
                .as_int(),
            Some(19)
        );
        assert_eq!(
            get_attribute_value(&mut vm, frame_value, &intern("f_lasti"))
                .expect("f_lasti should be readable")
                .as_int(),
            Some(5)
        );

        unsafe {
            drop_boxed(frame_ptr);
            drop_boxed(globals_ptr);
            drop_boxed(locals_ptr);
        }
    }

    #[test]
    fn test_get_attr_reads_closure_and_cell_contents() {
        let (func_ptr, func_value) = make_test_function_value("inner");
        let mut vm = vm_with_names(&["__closure__", "cell_contents"]);
        let closure = Arc::new(ClosureEnv::new(vec![Arc::new(Cell::new(
            Value::int(41).unwrap(),
        ))]));
        vm.register_function_closure(func_ptr as *const (), closure);
        vm.current_frame_mut().set_reg(1, func_value);

        let closure_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(
            get_attr(&mut vm, closure_inst),
            ControlFlow::Continue
        ));

        let closure_value = vm.current_frame().get_reg(2);
        let closure_ptr = closure_value.as_object_ptr().unwrap();
        let tuple = unsafe { &*(closure_ptr as *const TupleObject) };
        assert_eq!(tuple.len(), 1);

        let cell_value = tuple.get(0).unwrap();
        let cell_ptr = cell_value.as_object_ptr().unwrap();
        assert_eq!(extract_type_id(cell_ptr), TypeId::CELL_VIEW);
        vm.current_frame_mut().set_reg(3, cell_value);

        let cell_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(4),
            Register::new(3),
            Register::new(1),
        );
        assert!(matches!(
            get_attr(&mut vm, cell_inst),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(4).as_int(), Some(41));

        unsafe {
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_get_attr_exposes_builtin_type_reflection_objects() {
        let mut vm = vm_with_names(&["__dict__", "__init__", "join", "__code__", "__globals__"]);
        vm.current_frame_mut().set_reg(
            1,
            crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE),
        );
        vm.current_frame_mut().set_reg(
            2,
            crate::builtins::builtin_type_object_for_type_id(TypeId::OBJECT),
        );
        vm.current_frame_mut().set_reg(
            3,
            crate::builtins::builtin_type_object_for_type_id(TypeId::STR),
        );
        vm.current_frame_mut().set_reg(
            4,
            crate::builtins::builtin_type_object_for_type_id(TypeId::FUNCTION),
        );

        let type_dict_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(10),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(
            get_attr(&mut vm, type_dict_inst),
            ControlFlow::Continue
        ));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(10).as_object_ptr().unwrap()),
            TypeId::MAPPING_PROXY
        );

        let object_init_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(11),
            Register::new(2),
            Register::new(1),
        );
        assert!(matches!(
            get_attr(&mut vm, object_init_inst),
            ControlFlow::Continue
        ));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(11).as_object_ptr().unwrap()),
            TypeId::WRAPPER_DESCRIPTOR
        );

        let str_join_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(12),
            Register::new(3),
            Register::new(2),
        );
        assert!(matches!(
            get_attr(&mut vm, str_join_inst),
            ControlFlow::Continue
        ));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(12).as_object_ptr().unwrap()),
            TypeId::METHOD_DESCRIPTOR
        );

        let function_code_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(13),
            Register::new(4),
            Register::new(3),
        );
        assert!(matches!(
            get_attr(&mut vm, function_code_inst),
            ControlFlow::Continue
        ));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(13).as_object_ptr().unwrap()),
            TypeId::GETSET_DESCRIPTOR
        );

        let function_globals_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(14),
            Register::new(4),
            Register::new(4),
        );
        assert!(matches!(
            get_attr(&mut vm, function_globals_inst),
            ControlFlow::Continue
        ));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(14).as_object_ptr().unwrap()),
            TypeId::MEMBER_DESCRIPTOR
        );
    }

    #[test]
    fn test_get_attr_exposes___class___for_user_defined_instances() {
        let class = register_test_class(PyClassObject::new_simple(intern("HeapCarrier")));
        let (instance_ptr, instance_value) = instance_value(&class);
        let mut vm = vm_with_names(&[]);

        let class_value = get_attribute_value(&mut vm, instance_value, &intern("__class__"))
            .expect("__class__ should be readable on heap instances");
        assert_eq!(
            class_value.as_object_ptr(),
            Some(Arc::as_ptr(&class) as *const ()),
        );

        unsafe {
            drop_boxed(instance_ptr);
        }
    }

    #[test]
    fn test_get_attr_exposes_object_method_wrapper_view() {
        let mut vm = vm_with_names(&["__str__"]);
        let object = crate::builtins::builtin_object(&[]).expect("object() should succeed");
        vm.current_frame_mut().set_reg(1, object);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            extract_type_id(vm.current_frame().get_reg(2).as_object_ptr().unwrap()),
            TypeId::METHOD_WRAPPER
        );
    }

    #[test]
    fn test_get_attr_binds_property_setter_builtin_method() {
        let property = Box::new(PropertyDescriptor::new_getter(Value::int(1).unwrap()));
        let property_ptr = Box::into_raw(property);
        let property_value = Value::object_ptr(property_ptr as *const ());

        let mut vm = vm_with_names(&["setter"]);
        vm.current_frame_mut().set_reg(1, property_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));

        let method_ptr = vm.current_frame().get_reg(2).as_object_ptr().unwrap();
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        let builtin = unsafe { &*(method_ptr as *const BuiltinFunctionObject) };
        assert_eq!(builtin.name(), "property.setter");

        let copied = builtin
            .call(&[Value::int(2).unwrap()])
            .expect("bound property setter should call");
        let copied_ptr = copied.as_object_ptr().unwrap();
        let copied_desc = unsafe { &*(copied_ptr as *const PropertyDescriptor) };
        assert_eq!(copied_desc.getter(), Some(Value::int(1).unwrap()));
        assert_eq!(copied_desc.setter(), Some(Value::int(2).unwrap()));

        unsafe {
            drop(Box::from_raw(copied_ptr as *mut PropertyDescriptor));
            drop(Box::from_raw(property_ptr));
        }
    }

    #[test]
    fn test_get_attr_invokes_property_data_descriptor_before_instance_attr() {
        let (getter_ptr, getter_value) =
            builtin_function_value("property_test.getter", property_storage_getter);
        let property = Box::new(PropertyDescriptor::new_getter(getter_value));
        let property_ptr = Box::into_raw(property);
        let property_value = Value::object_ptr(property_ptr as *const ());

        let mut class = PyClassObject::new_simple(intern("Managed"));
        class.set_attr(intern("managed"), property_value);
        let class = register_test_class(class);
        let (instance_ptr, instance_value) = instance_value(&class);
        unsafe {
            (*instance_ptr).set_property(
                intern("_value"),
                Value::int(41).unwrap(),
                shape_registry(),
            );
            (*instance_ptr).set_property(
                intern("managed"),
                Value::int(99).unwrap(),
                shape_registry(),
            );
        }

        let mut vm = vm_with_names(&["managed"]);
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2).as_int(), Some(41));

        unsafe {
            drop_boxed(instance_ptr);
            drop(Box::from_raw(property_ptr));
            drop(Box::from_raw(getter_ptr));
        }
    }

    #[test]
    fn test_set_attr_invokes_property_setter_on_user_instance() {
        let (getter_ptr, getter_value) =
            builtin_function_value("property_test.getter", property_storage_getter);
        let (setter_ptr, setter_value) =
            builtin_function_value("property_test.setter", property_storage_setter);
        let property = Box::new(PropertyDescriptor::new_full(
            Some(getter_value),
            Some(setter_value),
            None,
            None,
        ));
        let property_ptr = Box::into_raw(property);
        let property_value = Value::object_ptr(property_ptr as *const ());

        let mut class = PyClassObject::new_simple(intern("Managed"));
        class.set_attr(intern("managed"), property_value);
        let class = register_test_class(class);
        let (instance_ptr, instance_value) = instance_value(&class);

        let mut vm = vm_with_names(&["managed"]);
        vm.current_frame_mut().set_reg(1, instance_value);
        vm.current_frame_mut().set_reg(2, Value::int(73).unwrap());

        let inst = Instruction::op_dss(
            Opcode::SetAttr,
            Register::new(1),
            Register::new(0),
            Register::new(2),
        );
        assert!(matches!(set_attr(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            unsafe { &*instance_ptr }
                .get_property("_value")
                .and_then(|value| value.as_int()),
            Some(73)
        );
        assert!(unsafe { &*instance_ptr }.get_property("managed").is_none());

        unsafe {
            drop_boxed(instance_ptr);
            drop(Box::from_raw(property_ptr));
            drop(Box::from_raw(setter_ptr));
            drop(Box::from_raw(getter_ptr));
        }
    }

    #[test]
    fn test_del_attr_invokes_property_deleter_on_user_instance() {
        let (getter_ptr, getter_value) =
            builtin_function_value("property_test.getter", property_storage_getter);
        let (deleter_ptr, deleter_value) =
            builtin_function_value("property_test.deleter", property_storage_deleter);
        let property = Box::new(PropertyDescriptor::new_full(
            Some(getter_value),
            None,
            Some(deleter_value),
            None,
        ));
        let property_ptr = Box::into_raw(property);
        let property_value = Value::object_ptr(property_ptr as *const ());

        let mut class = PyClassObject::new_simple(intern("Managed"));
        class.set_attr(intern("managed"), property_value);
        let class = register_test_class(class);
        let (instance_ptr, instance_value) = instance_value(&class);
        unsafe {
            (*instance_ptr).set_property(
                intern("_value"),
                Value::int(11).unwrap(),
                shape_registry(),
            );
        }

        let mut vm = vm_with_names(&["managed"]);
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::DelAttr,
            Register::new(0),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(del_attr(&mut vm, inst), ControlFlow::Continue));
        assert!(unsafe { &*instance_ptr }.get_property("_value").is_none());

        unsafe {
            drop_boxed(instance_ptr);
            drop(Box::from_raw(property_ptr));
            drop(Box::from_raw(deleter_ptr));
            drop(Box::from_raw(getter_ptr));
        }
    }

    #[test]
    fn test_set_and_del_attr_operate_on_class_objects() {
        let mut vm = vm_with_names(&["field"]);
        let (class_value, class_ptr) = class_value(PyClassObject::new_simple(intern("Example")));
        vm.current_frame_mut().set_reg(1, class_value);
        vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());

        let set_inst = Instruction::op_dss(
            Opcode::SetAttr,
            Register::new(1),
            Register::new(0),
            Register::new(2),
        );
        assert!(matches!(set_attr(&mut vm, set_inst), ControlFlow::Continue));

        let get_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(3),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(7));

        let del_inst = Instruction::op_dss(
            Opcode::DelAttr,
            Register::new(0),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(del_attr(&mut vm, del_inst), ControlFlow::Continue));
        assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Error(_)));

        unsafe { drop_class(class_ptr) };
    }

    #[test]
    fn test_set_and_del_attr_operate_on_imported_modules() {
        let mut vm = vm_with_names(&["field"]);
        let module = Arc::new(ModuleObject::new("modprobe"));
        vm.import_resolver.insert_module("modprobe", module.clone());
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(7).unwrap());

        let set_inst = Instruction::op_dss(
            Opcode::SetAttr,
            Register::new(1),
            Register::new(0),
            Register::new(2),
        );
        assert!(matches!(set_attr(&mut vm, set_inst), ControlFlow::Continue));
        assert_eq!(
            module.get_attr("field").and_then(|value| value.as_int()),
            Some(7)
        );

        let get_inst = Instruction::op_dss(
            Opcode::GetAttr,
            Register::new(3),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(3).as_int(), Some(7));

        let del_inst = Instruction::op_dss(
            Opcode::DelAttr,
            Register::new(0),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(del_attr(&mut vm, del_inst), ControlFlow::Continue));
        assert!(module.get_attr("field").is_none());
        assert!(matches!(get_attr(&mut vm, get_inst), ControlFlow::Error(_)));
    }
}
