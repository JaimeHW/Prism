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
    builtin_instance_attribute_value, builtin_type_class_or_static_attribute_value_static,
    builtin_type_object_attribute_value, exception_proxy_class_id_from_ptr,
    exception_type_attribute_value, heap_type_attribute_value,
};
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::frame::Frame;
use crate::ops::attribute::is_user_defined_type;
use crate::ops::iteration::{IterStep, ensure_iterator_value, next_step};
use crate::ops::method_dispatch::method_cache::method_cache;
use crate::ops::method_dispatch::resolve_builtin_instance_method;
use crate::stdlib::collections::deque::DequeObject;
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::{Constant, Instruction, Opcode};
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{MethodSlot, PyClassObject};
use prism_runtime::object::descriptor::{
    BoundMethod, ClassMethodDescriptor, Descriptor, PropertyDescriptor, SlotDescriptor,
    StaticMethodDescriptor,
};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shape::shape_registry;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::super_obj::{SuperBinding, SuperObject};
use prism_runtime::object::type_builtins::{builtin_class_mro, class_id_to_type_id, global_class};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::object::views::{
    CellViewObject, CodeObjectView, DictViewObject, FrameViewObject, MappingProxyObject,
    TracebackViewObject,
};
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::function::FunctionObject;
use prism_runtime::types::int::{bigint_to_value, value_to_saturated_i64};
use prism_runtime::types::iter::{IteratorObject, is_native_iterator_type_id};
use prism_runtime::types::list::ListObject;
use prism_runtime::types::range::RangeObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::slice::SliceObject;
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
fn uses_shaped_user_instance_layout(type_id: TypeId) -> bool {
    is_user_defined_type(type_id) && !is_native_iterator_type_id(type_id)
}

#[inline]
pub(crate) fn dict_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static DictObject> {
    match extract_type_id(ptr) {
        TypeId::DICT => Some(unsafe { &*(ptr as *const DictObject) }),
        type_id if uses_shaped_user_instance_layout(type_id) => {
            unsafe { &*(ptr as *const ShapedObject) }.dict_backing()
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn dict_storage_mut_from_ptr(ptr: *const ()) -> Option<&'static mut DictObject> {
    match extract_type_id(ptr) {
        TypeId::DICT => Some(unsafe { &mut *(ptr as *mut DictObject) }),
        type_id if uses_shaped_user_instance_layout(type_id) => {
            unsafe { &mut *(ptr as *mut ShapedObject) }.dict_backing_mut()
        }
        _ => None,
    }
}

#[inline]
fn dict_storage_ref_from_value(value: Value) -> Option<&'static DictObject> {
    value.as_object_ptr().and_then(dict_storage_ref_from_ptr)
}

#[inline]
fn dict_storage_mut_from_value(value: Value) -> Option<&'static mut DictObject> {
    value.as_object_ptr().and_then(dict_storage_mut_from_ptr)
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
pub(crate) fn set_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static SetObject> {
    match extract_type_id(ptr) {
        TypeId::SET | TypeId::FROZENSET => Some(unsafe { &*(ptr as *const SetObject) }),
        type_id if uses_shaped_user_instance_layout(type_id) => {
            unsafe { &*(ptr as *const ShapedObject) }.set_backing()
        }
        _ => None,
    }
}

#[inline]
pub(crate) fn set_storage_mut_from_ptr(ptr: *const ()) -> Option<&'static mut SetObject> {
    match extract_type_id(ptr) {
        TypeId::SET | TypeId::FROZENSET => Some(unsafe { &mut *(ptr as *mut SetObject) }),
        type_id if uses_shaped_user_instance_layout(type_id) => {
            unsafe { &mut *(ptr as *mut ShapedObject) }.set_backing_mut()
        }
        _ => None,
    }
}

pub(crate) fn set_list_item_value(
    vm: &mut VirtualMachine,
    ptr: *const (),
    key: Value,
    value: Value,
) -> Result<(), RuntimeError> {
    if let Some(slice) = slice_object_from_value(key) {
        if slice.step() == Some(0) {
            return Err(RuntimeError::value_error("slice step cannot be zero"));
        }

        let slice = copy_slice_object(slice);
        let replacement = crate::ops::iteration::collect_iterable_values(vm, value)?;
        let list = list_storage_mut_from_ptr(ptr).ok_or_else(|| {
            RuntimeError::type_error("descriptor 'list.__setitem__' requires a list object")
        })?;
        list.assign_slice(&slice, replacement)
            .map_err(|err| RuntimeError::value_error(err.to_string()))?;
        return Ok(());
    }

    let idx = list_index_from_key(key)?;
    let list = list_storage_mut_from_ptr(ptr).ok_or_else(|| {
        RuntimeError::type_error("descriptor 'list.__setitem__' requires a list object")
    })?;
    if list.set(idx, value) {
        Ok(())
    } else {
        Err(RuntimeError::index_error(idx, list.len()))
    }
}

pub(crate) fn delete_list_item_value(ptr: *const (), key: Value) -> Result<(), RuntimeError> {
    if let Some(slice) = slice_object_from_value(key) {
        if slice.step() == Some(0) {
            return Err(RuntimeError::value_error("slice step cannot be zero"));
        }

        let list = list_storage_mut_from_ptr(ptr).ok_or_else(|| {
            RuntimeError::type_error("descriptor 'list.__delitem__' requires a list object")
        })?;
        list.delete_slice(slice);
        return Ok(());
    }

    let idx = list_index_from_key(key)?;
    let list = list_storage_mut_from_ptr(ptr).ok_or_else(|| {
        RuntimeError::type_error("descriptor 'list.__delitem__' requires a list object")
    })?;
    if list.remove(idx).is_some() {
        Ok(())
    } else {
        Err(RuntimeError::index_error(idx, list.len()))
    }
}

#[inline]
fn list_index_from_key(key: Value) -> Result<i64, RuntimeError> {
    if let Some(index) = key.as_bool().map(i64::from) {
        return Ok(index);
    }
    if let Some(index) = value_to_saturated_i64(key) {
        return Ok(index);
    }

    Err(RuntimeError::type_error(
        "list indices must be integers or slices",
    ))
}

#[inline]
fn slice_object_from_value(value: Value) -> Option<&'static SliceObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    (header.type_id == TypeId::SLICE).then(|| unsafe { &*(ptr as *const SliceObject) })
}

#[inline]
fn copy_slice_object(slice: &SliceObject) -> SliceObject {
    SliceObject::new(slice.start_value(), slice.stop_value(), slice.step_value())
}

#[inline]
pub(crate) fn tuple_storage_ref_from_ptr(ptr: *const ()) -> Option<&'static TupleObject> {
    prism_runtime::types::tuple::object_ptr_as_tuple_ref(ptr)
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
fn lookup_user_class_slot(class: &PyClassObject, name: &InternedString) -> Option<MethodSlot> {
    class.lookup_method_published(name)
}

#[inline]
fn lookup_user_class_attr(class: &PyClassObject, name: &InternedString) -> Option<Value> {
    lookup_user_class_slot(class, name).map(|slot| slot.value)
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
        .lookup_method_published(name)
        .map(|slot| slot.value)
}

#[inline]
fn lookup_instance_class_attr(type_id: TypeId, name: &InternedString) -> Option<Value> {
    lookup_instance_class_slot(type_id, name).map(|slot| slot.value)
}

#[inline]
fn lookup_instance_class_slot(type_id: TypeId, name: &InternedString) -> Option<MethodSlot> {
    let class = global_class(ClassId(type_id.raw()))?;
    lookup_user_class_slot(class.as_ref(), name)
}

#[inline]
fn defining_class_binds_builtin_instance_attributes(defining_class: ClassId) -> bool {
    defining_class.0 >= TypeId::FIRST_USER_TYPE
        && global_class(defining_class).is_some_and(|class| class.is_native_heaptype())
}

#[inline]
fn bind_user_class_attribute_value(
    value: Value,
    defining_class: ClassId,
    instance: Value,
) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };
    if extract_type_id(ptr) != TypeId::BUILTIN_FUNCTION
        || !defining_class_binds_builtin_instance_attributes(defining_class)
    {
        return bind_instance_attribute(value, instance);
    }

    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    if builtin.bound_self().is_some() {
        value
    } else {
        let bound = Box::leak(Box::new(builtin.bind(instance)));
        Value::object_ptr(bound as *mut BuiltinFunctionObject as *const ())
    }
}

#[inline]
fn bind_user_class_attribute_value_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    defining_class: ClassId,
    instance: Value,
) -> Result<Value, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(value);
    };
    if extract_type_id(ptr) != TypeId::BUILTIN_FUNCTION
        || !defining_class_binds_builtin_instance_attributes(defining_class)
    {
        return bind_instance_attribute_in_vm(vm, value, instance);
    }

    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    if builtin.bound_self().is_some() {
        Ok(value)
    } else {
        let bound = Box::leak(Box::new(builtin.bind(instance)));
        Ok(Value::object_ptr(
            bound as *mut BuiltinFunctionObject as *const (),
        ))
    }
}

#[inline]
fn bind_type_attribute_value(value: Value, defining_class: ClassId, type_instance: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let bound = Box::leak(Box::new(BoundMethod::new(value, type_instance)));
            Value::object_ptr(bound as *mut BoundMethod as *const ())
        }
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            desc.bind_value(type_instance)
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            desc.function()
        }
        TypeId::BUILTIN_FUNCTION
            if defining_class_binds_builtin_instance_attributes(defining_class) =>
        {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            if builtin.bound_self().is_some() {
                value
            } else {
                let bound = Box::leak(Box::new(builtin.bind(type_instance)));
                Value::object_ptr(bound as *mut BuiltinFunctionObject as *const ())
            }
        }
        TypeId::BUILTIN_FUNCTION => value,
        _ => value,
    }
}

#[inline]
fn bind_type_attribute_value_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    defining_class: ClassId,
    type_instance: Value,
) -> Result<Value, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(value);
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let bound = Box::leak(Box::new(BoundMethod::new(value, type_instance)));
            Ok(Value::object_ptr(bound as *mut BoundMethod as *const ()))
        }
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            bind_wrapped_classmethod_value(vm, desc.function(), type_instance)
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Ok(desc.function())
        }
        TypeId::BUILTIN_FUNCTION
            if defining_class_binds_builtin_instance_attributes(defining_class) =>
        {
            let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
            if builtin.bound_self().is_some() {
                Ok(value)
            } else {
                let bound = Box::leak(Box::new(builtin.bind(type_instance)));
                Ok(Value::object_ptr(
                    bound as *mut BuiltinFunctionObject as *const (),
                ))
            }
        }
        TypeId::BUILTIN_FUNCTION => Ok(value),
        _ => Ok(
            invoke_descriptor_get_in_vm(vm, value, type_instance, type_instance)?.unwrap_or(value),
        ),
    }
}

#[inline]
fn bind_user_class_slot(slot: MethodSlot, instance: Value) -> Value {
    bind_user_class_attribute_value(slot.value, slot.defining_class, instance)
}

fn lookup_builtin_base_instance_attr(
    vm: &mut VirtualMachine,
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
            return Ok(Some(bind_cached_builtin_method(cached.method, obj)));
        }

        if let Some(value) = builtin_instance_attribute_value(vm, owner, obj, name)? {
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
fn bind_super_lookup_value(
    value: Value,
    defining_class: ClassId,
    super_obj: &SuperObject,
    name: &InternedString,
) -> Value {
    match super_obj.binding() {
        SuperBinding::Unbound => value,
        SuperBinding::Instance => {
            bind_user_class_attribute_value(value, defining_class, super_obj.obj())
        }
        SuperBinding::Type => {
            if name.as_str() == "__new__" {
                resolve_class_attribute(value, super_obj.obj())
            } else {
                bind_type_attribute_value(value, defining_class, super_obj.obj())
            }
        }
    }
}

#[inline]
fn bind_super_lookup_value_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    defining_class: ClassId,
    super_obj: &SuperObject,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    match super_obj.binding() {
        SuperBinding::Unbound => Ok(value),
        SuperBinding::Instance => {
            bind_user_class_attribute_value_in_vm(vm, value, defining_class, super_obj.obj())
        }
        SuperBinding::Type => {
            if name.as_str() == "__new__" {
                resolve_class_attribute_in_vm(vm, value, super_obj.obj())
            } else {
                bind_type_attribute_value_in_vm(vm, value, defining_class, super_obj.obj())
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
fn code_constant_to_value(constant: &Constant) -> Value {
    match constant {
        Constant::Value(value) => *value,
        Constant::BigInt(value) => bigint_to_value(value.clone()),
    }
}

#[inline]
fn property_descriptor_from_value(value: Value) -> Option<&'static PropertyDescriptor> {
    let ptr = value.as_object_ptr()?;
    (extract_type_id(ptr) == TypeId::PROPERTY)
        .then(|| unsafe { &*(ptr as *const PropertyDescriptor) })
}

#[inline]
fn slot_descriptor_from_value(value: Value) -> Option<&'static SlotDescriptor> {
    let ptr = value.as_object_ptr()?;
    (extract_type_id(ptr) == TypeId::MEMBER_DESCRIPTOR)
        .then(|| unsafe { &*(ptr as *const SlotDescriptor) })
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
                return Ok(Some(bind_cached_builtin_method(
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
            return Ok(Some(bind_super_lookup_value(
                value, class_id, super_obj, name,
            )));
        }
    }

    Ok(None)
}

pub(crate) fn super_attribute_value_in_vm(
    vm: &mut VirtualMachine,
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
                return Ok(Some(bind_cached_builtin_method(
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
            return bind_super_lookup_value_in_vm(vm, value, class_id, super_obj, name).map(Some);
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
pub(crate) fn alloc_heap_value<T>(
    _vm: &mut VirtualMachine,
    object: T,
    _context: &'static str,
) -> Result<Value, RuntimeError>
where
    T: prism_runtime::Trace + 'static,
{
    Ok(alloc_value_in_current_heap_or_box(object))
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
                crate::alloc_managed_value(tuple)
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

fn module_attribute_value(
    _vm: &mut VirtualMachine,
    module: &crate::import::ModuleObject,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    if name.as_str() == "__dict__" {
        return Ok(module.dict_value());
    }

    module
        .get_attr(name.as_str())
        .ok_or_else(|| RuntimeError::attribute_error("module", name.as_str()))
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
        if !frame.local_is_written(slot as u16) {
            continue;
        }
        dict.set(
            Value::string(intern(name.as_ref())),
            frame.get_local(slot as u16),
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

fn function_annotations_value(
    vm: &mut VirtualMachine,
    func: &FunctionObject,
) -> Result<Value, RuntimeError> {
    let annotations_name = intern("__annotations__");
    if let Some(value) = func.get_attr(&annotations_name) {
        return Ok(value);
    }

    let value = alloc_heap_value(vm, DictObject::new(), "function annotations dict")?;
    func.set_attr(annotations_name, value);
    Ok(value)
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
    if name.as_str() == "__class__" {
        return Ok(Some(crate::builtins::builtin_type_object_for_type_id(
            TypeId::FUNCTION,
        )));
    }

    if let Some(value) = function_attr_value(func, name) {
        return Ok(Some(value));
    }

    if name.as_str() == "__get__" {
        return Ok(
            crate::ops::method_dispatch::resolve_builtin_instance_method(
                TypeId::FUNCTION,
                "__get__",
            )
            .map(|cached| bind_cached_builtin_method(cached.method, Value::object_ptr(func_ptr))),
        );
    }

    match name.as_str() {
        "__doc__" => Ok(Some(Value::none())),
        "__dict__" => Ok(Some(function_attr_dict_value(vm, func)?)),
        "__module__" => Ok(Some(
            function_module(vm, func)
                .map(|module| Value::string(intern(module.name())))
                .unwrap_or_else(|| Value::string(intern("__main__"))),
        )),
        "__annotations__" => Ok(Some(function_annotations_value(vm, func)?)),
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
            if let Some(module) = function_module(vm, func) {
                Ok(Some(module.dict_value()))
            } else {
                let dict = snapshot_current_globals_dict(vm);
                Ok(Some(alloc_heap_value(vm, dict, "globals dict snapshot")?))
            }
        }
        "__closure__" => {
            let Some(closure) = func.closure() else {
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
        "__class__" => Ok(Some(crate::builtins::builtin_type_object_for_type_id(
            TypeId::METHOD,
        ))),
        "__self__" => Ok(Some(method.instance())),
        "__func__" | "__wrapped__" => Ok(Some(method.function())),
        "__get__" => Ok(None),
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
        "__class__" => Ok(Some(crate::builtins::builtin_type_object_for_type_id(
            TypeId::CODE,
        ))),
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
            TupleObject::from_vec(code.constants.iter().map(code_constant_to_value).collect()),
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
fn builtin_function_display_name(name: &str) -> &str {
    name.rsplit('.').next().unwrap_or(name)
}

#[inline]
fn builtin_function_module_name(name: &str) -> &str {
    name.rsplit_once('.')
        .map(|(module, _)| module)
        .unwrap_or("builtins")
}

#[inline]
pub(crate) fn builtin_function_attr_value(
    builtin: &BuiltinFunctionObject,
    name: &InternedString,
) -> Option<Value> {
    match name.as_str() {
        "__name__" => Some(Value::string(intern(builtin_function_display_name(
            builtin.name(),
        )))),
        "__qualname__" => Some(Value::string(intern(builtin.name()))),
        "__module__" => Some(Value::string(intern(builtin_function_module_name(
            builtin.name(),
        )))),
        "__doc__" => Some(Value::none()),
        "__self__" => Some(builtin.bound_self().unwrap_or_else(Value::none)),
        _ => None,
    }
}

#[inline]
pub(crate) fn function_attr_exists(func: &FunctionObject, name: &InternedString) -> bool {
    function_attr_value(func, name).is_some()
        || matches!(
            name.as_str(),
            "__doc__"
                | "__class__"
                | "__dict__"
                | "__get__"
                | "__module__"
                | "__annotations__"
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
fn descriptor_wrapped_value(value: Value) -> Option<Value> {
    let ptr = value.as_object_ptr()?;
    match extract_type_id(ptr) {
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            Some(desc.function())
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Some(desc.function())
        }
        _ => None,
    }
}

pub(crate) fn descriptor_attr_value_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    if name.as_str() == "__isabstractmethod__"
        && value.as_object_ptr().is_some_and(|ptr| {
            matches!(
                extract_type_id(ptr),
                TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY
            )
        })
    {
        return descriptor_is_abstract_in_vm(vm, value)
            .map(|abstract_method| Some(Value::bool(abstract_method)));
    }

    if let Some(value) = descriptor_attr_value(value, name) {
        return Ok(Some(value));
    }

    if matches!(
        name.as_str(),
        "__get__" | "__set__" | "__delete__" | "__set_name__"
    ) {
        let Some(ptr) = value.as_object_ptr() else {
            return Ok(None);
        };
        let owner = match extract_type_id(ptr) {
            TypeId::CLASSMETHOD => TypeId::CLASSMETHOD,
            TypeId::STATICMETHOD => TypeId::STATICMETHOD,
            TypeId::PROPERTY => TypeId::PROPERTY,
            _ => return Ok(None),
        };

        return Ok(
            crate::ops::method_dispatch::resolve_builtin_instance_method(owner, name.as_str())
                .map(|cached| bind_cached_builtin_method(cached.method, value)),
        );
    }

    let Some(wrapped) = descriptor_wrapped_value(value) else {
        return Ok(None);
    };

    match name.as_str() {
        "__module__" | "__qualname__" | "__name__" | "__doc__" | "__annotations__" => {
            get_attribute_value(vm, wrapped, name).map(Some)
        }
        _ => Ok(None),
    }
}

fn descriptor_is_abstract_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
) -> Result<bool, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(false);
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let func = unsafe { &*(ptr as *const FunctionObject) };
            match func.get_attr(&intern("__isabstractmethod__")) {
                Some(value) => crate::truthiness::try_is_truthy(vm, value),
                None => Ok(false),
            }
        }
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            descriptor_is_abstract_in_vm(vm, desc.function())
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            descriptor_is_abstract_in_vm(vm, desc.function())
        }
        TypeId::PROPERTY => {
            let desc = unsafe { &*(ptr as *const PropertyDescriptor) };
            if let Some(getter) = desc.getter()
                && descriptor_is_abstract_in_vm(vm, getter)?
            {
                return Ok(true);
            }
            if let Some(setter) = desc.setter()
                && descriptor_is_abstract_in_vm(vm, setter)?
            {
                return Ok(true);
            }
            if let Some(deleter) = desc.deleter()
                && descriptor_is_abstract_in_vm(vm, deleter)?
            {
                return Ok(true);
            }
            Ok(false)
        }
        _ => Ok(false),
    }
}

#[inline]
fn classmethod_owner_from_instance(instance: Value) -> Value {
    crate::builtins::value_type_object(instance)
}

fn descriptor_special_method_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let Some(ptr) = descriptor.as_object_ptr() else {
        return Ok(None);
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let func = unsafe { &*(ptr as *const FunctionObject) };
            function_attr_value_in_vm(vm, ptr, func, name)
        }
        TypeId::METHOD => {
            let method = unsafe { &*(ptr as *const BoundMethod) };
            method_attr_value_in_vm(vm, method, name)
        }
        TypeId::CLASSMETHOD | TypeId::STATICMETHOD | TypeId::PROPERTY => {
            descriptor_attr_value_in_vm(vm, descriptor, name)
        }
        type_id => {
            if let Some(method) = builtin_instance_method_attr_value(descriptor, type_id, name) {
                return Ok(Some(method));
            }

            if is_user_defined_type(type_id)
                && let Some(slot) = lookup_instance_class_slot(type_id, name)
            {
                return bind_user_class_attribute_value_in_vm(
                    vm,
                    slot.value,
                    slot.defining_class,
                    descriptor,
                )
                .map(Some);
            }

            Ok(None)
        }
    }
}

fn invoke_descriptor_get_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
    instance: Value,
    owner: Value,
) -> Result<Option<Value>, RuntimeError> {
    let Some(getter) = descriptor_special_method_in_vm(vm, descriptor, &intern("__get__"))? else {
        return Ok(None);
    };

    crate::ops::calls::invoke_callable_value(vm, getter, &[instance, owner]).map(Some)
}

fn descriptor_has_special_method_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
    name: &str,
) -> Result<bool, RuntimeError> {
    descriptor_special_method_in_vm(vm, descriptor, &intern(name)).map(|value| value.is_some())
}

fn descriptor_is_data_descriptor_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
) -> Result<bool, RuntimeError> {
    Ok(
        descriptor_has_special_method_in_vm(vm, descriptor, "__set__")?
            || descriptor_has_special_method_in_vm(vm, descriptor, "__delete__")?,
    )
}

fn invoke_descriptor_set_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
    instance: Value,
    value: Value,
) -> Result<bool, RuntimeError> {
    let Some(setter) = descriptor_special_method_in_vm(vm, descriptor, &intern("__set__"))? else {
        return Ok(false);
    };

    crate::ops::calls::invoke_callable_value(vm, setter, &[instance, value])?;
    Ok(true)
}

fn invoke_descriptor_delete_in_vm(
    vm: &mut VirtualMachine,
    descriptor: Value,
    instance: Value,
) -> Result<bool, RuntimeError> {
    let Some(deleter) = descriptor_special_method_in_vm(vm, descriptor, &intern("__delete__"))?
    else {
        return Ok(false);
    };

    crate::ops::calls::invoke_callable_value(vm, deleter, &[instance])?;
    Ok(true)
}

#[inline]
fn bind_fallback_callable(value: Value, owner: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        let bound = Box::leak(Box::new(BoundMethod::new(value, owner)));
        return Value::object_ptr(bound as *mut BoundMethod as *const ());
    };

    if extract_type_id(ptr) == TypeId::BUILTIN_FUNCTION {
        return bind_cached_builtin_method(value, owner);
    }

    let bound = Box::leak(Box::new(BoundMethod::new(value, owner)));
    Value::object_ptr(bound as *mut BoundMethod as *const ())
}

pub(crate) fn bind_wrapped_classmethod_value(
    vm: &mut VirtualMachine,
    wrapped: Value,
    owner: Value,
) -> Result<Value, RuntimeError> {
    match invoke_descriptor_get_in_vm(vm, wrapped, owner, owner)? {
        Some(bound) => Ok(bound),
        None => Ok(bind_fallback_callable(wrapped, owner)),
    }
}

#[inline]
pub(crate) fn resolve_class_attribute_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    owner: Value,
) -> Result<Value, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(value);
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => Ok(value),
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            bind_wrapped_classmethod_value(vm, desc.function(), owner)
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Ok(desc.function())
        }
        _ => Ok(invoke_descriptor_get_in_vm(vm, value, Value::none(), owner)?.unwrap_or(value)),
    }
}

#[inline]
pub(crate) fn bind_instance_attribute_in_vm(
    vm: &mut VirtualMachine,
    value: Value,
    instance: Value,
) -> Result<Value, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(value);
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let bound = Box::leak(Box::new(BoundMethod::new(value, instance)));
            Ok(Value::object_ptr(bound as *mut BoundMethod as *const ()))
        }
        TypeId::BUILTIN_FUNCTION => Ok(value),
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            bind_wrapped_classmethod_value(
                vm,
                desc.function(),
                classmethod_owner_from_instance(instance),
            )
        }
        TypeId::STATICMETHOD => {
            let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
            Ok(desc.function())
        }
        _ => Ok(invoke_descriptor_get_in_vm(
            vm,
            value,
            instance,
            classmethod_owner_from_instance(instance),
        )?
        .unwrap_or(value)),
    }
}

#[inline]
pub(crate) fn resolve_class_attribute(value: Value, owner: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    match extract_type_id(ptr) {
        TypeId::BUILTIN_FUNCTION => value,
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
        TypeId::FUNCTION | TypeId::CLOSURE => {
            let bound = Box::leak(Box::new(BoundMethod::new(value, instance)));
            Value::object_ptr(bound as *mut BoundMethod as *const ())
        }
        TypeId::BUILTIN_FUNCTION => value,
        TypeId::CLASSMETHOD => {
            let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
            let owner = classmethod_owner_from_instance(instance);
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
pub(crate) fn bind_cached_builtin_method(value: Value, instance: Value) -> Value {
    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };
    if extract_type_id(ptr) != TypeId::BUILTIN_FUNCTION {
        return bind_instance_attribute(value, instance);
    }

    let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
    let bound = Box::leak(Box::new(builtin.bind(instance)));
    Value::object_ptr(bound as *mut BuiltinFunctionObject as *const ())
}

#[inline]
pub(crate) fn builtin_instance_method_attr_value(
    obj: Value,
    type_id: TypeId,
    name: &InternedString,
) -> Option<Value> {
    resolve_builtin_instance_method(type_id, name.as_str())
        .map(|cached| bind_cached_builtin_method(cached.method, obj))
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

        if let Some(value) = builtin_instance_attribute_value(vm, builtin_owner, obj, name)? {
            return Ok(Some(value));
        }

        if let Some(value) = builtin_instance_method_attr_value(obj, builtin_owner, name) {
            return Ok(Some(value));
        }

        let owner_value = crate::builtins::builtin_type_object_for_type_id(builtin_owner);
        if let Some(value) =
            builtin_type_class_or_static_attribute_value_static(builtin_owner, owner_value, name)
        {
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

fn ensure_user_defined_instance_dict_value(ptr: *const ()) -> Value {
    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
    shaped.ensure_instance_dict_value()
}

fn set_user_defined_instance_dict_value(ptr: *const (), value: Value) -> Result<(), RuntimeError> {
    if dict_storage_ref_from_value(value).is_none() {
        return Err(RuntimeError::type_error(
            "__dict__ must be set to a dictionary",
        ));
    }

    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
    shaped.set_instance_dict_value(value);
    Ok(())
}

fn reset_user_defined_instance_dict_value(ptr: *const ()) {
    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
    shaped.reset_instance_dict();
}

fn lookup_user_defined_instance_attribute_default(
    vm: &mut VirtualMachine,
    obj: Value,
    ptr: *const (),
    type_id: TypeId,
    name: &InternedString,
) -> Result<Option<Value>, RuntimeError> {
    let class_slot = lookup_instance_class_slot(type_id, name);
    if let Some(descriptor) = class_slot
        .as_ref()
        .map(|slot| slot.value)
        .and_then(property_descriptor_from_value)
    {
        return invoke_property_getter(vm, descriptor, obj).map(Some);
    }

    if let Some(descriptor) = class_slot
        .as_ref()
        .map(|slot| slot.value)
        .and_then(slot_descriptor_from_value)
    {
        let owner = class_id_to_value(ClassId(type_id.raw())).unwrap_or_else(Value::none);
        return descriptor
            .get(Some(obj), owner)
            .map(Some)
            .map_err(RuntimeError::from);
    }

    if let Some(ref slot) = class_slot
        && descriptor_is_data_descriptor_in_vm(vm, slot.value)?
    {
        return bind_user_class_attribute_value_in_vm(vm, slot.value, slot.defining_class, obj)
            .map(Some);
    }

    if name.as_str() == "__class__" {
        return Ok(class_id_to_value(ClassId(type_id.raw())));
    }

    if !uses_shaped_user_instance_layout(type_id) {
        if let Some(slot) = class_slot {
            return bind_user_class_attribute_value_in_vm(vm, slot.value, slot.defining_class, obj)
                .map(Some);
        }

        if let Some(value) = lookup_builtin_base_instance_attr(vm, obj, type_id, name)? {
            return Ok(Some(value));
        }

        return Ok(None);
    }

    if name.as_str() == "__dict__" {
        return Ok(Some(ensure_user_defined_instance_dict_value(ptr)));
    }

    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    if let Some(dict_value) = shaped.instance_dict_value() {
        let dict = dict_storage_ref_from_value(dict_value).ok_or_else(|| {
            RuntimeError::type_error("instance __dict__ storage is not a dictionary")
        })?;
        if let Some(value) = dict.get(Value::string(name.clone())) {
            return Ok(Some(value));
        }
    } else if let Some(value) = shaped.get_property_interned(name) {
        return Ok(Some(value));
    }

    if let Some(slot) = class_slot {
        return bind_user_class_attribute_value_in_vm(vm, slot.value, slot.defining_class, obj)
            .map(Some);
    }

    if let Some(value) = lookup_builtin_base_instance_attr(vm, obj, type_id, name)? {
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

    let Some(getattr_slot) = lookup_instance_class_slot(type_id, &intern("__getattr__")) else {
        return Ok(None);
    };

    let getattr = bind_user_class_attribute_value_in_vm(
        vm,
        getattr_slot.value,
        getattr_slot.defining_class,
        obj,
    )?;
    crate::ops::calls::invoke_callable_value(vm, getattr, &[Value::string(name.clone())]).map(Some)
}

fn lookup_user_defined_instance_attribute(
    vm: &mut VirtualMachine,
    obj: Value,
    ptr: *const (),
    type_id: TypeId,
    name: &InternedString,
) -> Result<Value, RuntimeError> {
    if let Some(getattribute_slot) =
        lookup_instance_class_slot(type_id, &intern("__getattribute__"))
    {
        let getattribute = bind_user_class_attribute_value_in_vm(
            vm,
            getattribute_slot.value,
            getattribute_slot.defining_class,
            obj,
        )?;
        return match crate::ops::calls::invoke_callable_value(
            vm,
            getattribute,
            &[Value::string(name.clone())],
        ) {
            Ok(value) => Ok(value),
            Err(err) if err.is_attribute_error() => {
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
        Err(err) if err.is_attribute_error() => {
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
            return module_attribute_value(vm, module.as_ref(), name);
        }

        let type_id = extract_type_id(ptr);
        if type_id == TypeId::MODULE {
            let module = unsafe { &*(ptr as *const crate::import::ModuleObject) };
            return module_attribute_value(vm, module, name);
        }

        if type_id == TypeId::EXCEPTION && name.as_str() == "__class__" {
            let exc = unsafe { &*(ptr as *const ExceptionValue) };
            return Ok(
                crate::builtins::exception_type_value_for_id(exc.exception_type_id).unwrap_or_else(
                    || crate::builtins::builtin_type_object_for_type_id(TypeId::EXCEPTION),
                ),
            );
        }

        if type_id == TypeId::TYPE
            && name.as_str() == "__class__"
            && let Some(class) = class_object_from_type_ptr(ptr)
        {
            let metaclass = class.metaclass();
            return Ok(if metaclass.is_none() {
                crate::builtins::builtin_type_object_for_type_id(TypeId::TYPE)
            } else {
                metaclass
            });
        }

        if !matches!(type_id, TypeId::EXCEPTION_TYPE | TypeId::TYPE)
            && let Some(value) = builtin_instance_attribute_value(vm, type_id, obj, name)?
        {
            return Ok(value);
        }

        if !matches!(type_id, TypeId::EXCEPTION_TYPE | TypeId::TYPE)
            && let Some(value) = builtin_instance_method_attr_value(obj, type_id, name)
        {
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
            TypeId::COMPLEX => {
                let complex = unsafe { &*(ptr as *const ComplexObject) };
                match name.as_str() {
                    "real" => Ok(Value::float(complex.real())),
                    "imag" => Ok(Value::float(complex.imag())),
                    _ => Err(RuntimeError::attribute_error("complex", name.as_str())),
                }
            }
            TypeId::DICT => Err(RuntimeError::attribute_error("dict", name.as_str())),
            TypeId::LIST => Err(RuntimeError::attribute_error("list", name.as_str())),
            TypeId::TUPLE => Err(RuntimeError::attribute_error("tuple", name.as_str())),
            TypeId::SET => Err(RuntimeError::attribute_error("set", name.as_str())),
            TypeId::FUNCTION | TypeId::CLOSURE => {
                let func = unsafe { &*(ptr as *const FunctionObject) };
                function_attr_value_in_vm(vm, ptr, func, name)?
                    .ok_or_else(|| RuntimeError::attribute_error("function", name.as_str()))
            }
            TypeId::BUILTIN_FUNCTION => {
                let builtin = unsafe { &*(ptr as *const BuiltinFunctionObject) };
                builtin_function_attr_value(builtin, name)
                    .ok_or_else(|| RuntimeError::attribute_error("builtin_function", name.as_str()))
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
                    "errno" if exc.is_subclass_of(ExceptionTypeId::OSError) => {
                        Ok(exception_arg_or_none(exc, 0))
                    }
                    "strerror" if exc.is_subclass_of(ExceptionTypeId::OSError) => {
                        Ok(exception_arg_or_message(exc, 1))
                    }
                    "winerror" if exc.is_subclass_of(ExceptionTypeId::OSError) => Ok(Value::none()),
                    "name" => Ok(exc
                        .import_name()
                        .map(|import_name| Value::string(intern(import_name)))
                        .unwrap_or_else(Value::none)),
                    "path" => Ok(exc
                        .import_path()
                        .map(|import_path| Value::string(intern(import_path)))
                        .unwrap_or_else(Value::none)),
                    "filename" if exc.is_subclass_of(ExceptionTypeId::OSError) => {
                        Ok(exception_arg_or_none(exc, 2))
                    }
                    "filename2" if exc.is_subclass_of(ExceptionTypeId::OSError) => {
                        Ok(exception_arg_or_none(exc, 4))
                    }
                    "filename" => Ok(exc
                        .syntax_filename()
                        .map(|filename| Value::string(intern(filename)))
                        .unwrap_or_else(Value::none)),
                    "lineno" => Ok(optional_u32_to_value(exc.syntax_lineno())),
                    "offset" => Ok(optional_u32_to_value(exc.syntax_offset())),
                    "text" => Ok(exc
                        .syntax_text()
                        .map(|text| Value::string(intern(text)))
                        .unwrap_or_else(Value::none)),
                    "end_lineno" => Ok(optional_u32_to_value(exc.syntax_end_lineno())),
                    "end_offset" => Ok(optional_u32_to_value(exc.syntax_end_offset())),
                    "print_file_and_line" => Ok(Value::none()),
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
                    "f_back" => Ok(frame.back().unwrap_or_else(Value::none)),
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
            TypeId::DICT_KEYS | TypeId::DICT_VALUES | TypeId::DICT_ITEMS => {
                let view = unsafe { &*(ptr as *const DictViewObject) };
                match name.as_str() {
                    "mapping" => alloc_heap_value(
                        vm,
                        MappingProxyObject::for_mapping(view.dict()),
                        "dictionary view mapping proxy",
                    ),
                    _ => Err(RuntimeError::attribute_error(type_id.name(), name.as_str())),
                }
            }
            TypeId::SUPER => super_attribute_value_in_vm(vm, obj, name)?
                .ok_or_else(|| RuntimeError::attribute_error("super", name.as_str())),
            TypeId::TYPE => {
                if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr)
                    && let Some(value) =
                        builtin_type_object_attribute_value(vm, represented, obj, name)?
                {
                    return Ok(value);
                }

                if let Some(class) = class_object_from_type_ptr(ptr) {
                    let metaclass_attr = lookup_class_metaclass_attr(class, name);
                    if let Some(value) = metaclass_attr
                        && descriptor_is_data_descriptor_in_vm(vm, value)?
                    {
                        return bind_instance_attribute_in_vm(vm, value, obj);
                    }

                    if let Some(value) =
                        heap_type_attribute_value(vm, ptr as *const PyClassObject, name)?
                    {
                        return Ok(value);
                    }

                    if let Some(value) = metaclass_attr {
                        return bind_instance_attribute_in_vm(vm, value, obj);
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
                if let Some(value) = descriptor_attr_value_in_vm(vm, obj, name)? {
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

#[inline]
fn exception_arg_or_none(exception: &ExceptionValue, index: usize) -> Value {
    exception
        .args
        .as_deref()
        .and_then(|args| args.get(index).copied())
        .unwrap_or_else(Value::none)
}

#[inline]
fn exception_arg_or_message(exception: &ExceptionValue, index: usize) -> Value {
    if let Some(value) = exception
        .args
        .as_deref()
        .and_then(|args| args.get(index).copied())
    {
        return value;
    }

    exception
        .message()
        .map(|message| Value::string(intern(message)))
        .unwrap_or_else(Value::none)
}

#[inline]
fn exception_link_ptr(
    value: Value,
    attr_name: &'static str,
) -> Result<Option<*const ExceptionValue>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    unsafe { ExceptionValue::from_value(value) }
        .map(|exception| Some(exception as *const ExceptionValue))
        .ok_or_else(|| {
            RuntimeError::type_error(format!("{attr_name} must be an exception or None"))
        })
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
        if type_id == TypeId::MODULE {
            let module = unsafe { &*(ptr as *const crate::import::ModuleObject) };
            module.set_attr(name.as_str(), value);
            return Ok(());
        }

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
                    "__cause__" => {
                        exc.cause = exception_link_ptr(value, "__cause__")?;
                        exc.flags = if exc.cause.is_some() {
                            exc.flags.with(ExceptionFlags::HAS_CAUSE)
                        } else {
                            exc.flags.without(ExceptionFlags::HAS_CAUSE)
                        }
                        .with(ExceptionFlags::SUPPRESS_CONTEXT);
                        Ok(())
                    }
                    "__context__" => {
                        exc.context = exception_link_ptr(value, "__context__")?;
                        Ok(())
                    }
                    "__suppress_context__" => {
                        let Some(suppress) = value.as_bool() else {
                            return Err(RuntimeError::type_error(
                                "__suppress_context__ must be a bool",
                            ));
                        };
                        exc.flags = if suppress {
                            exc.flags.with(ExceptionFlags::SUPPRESS_CONTEXT)
                        } else {
                            exc.flags.without(ExceptionFlags::SUPPRESS_CONTEXT)
                        };
                        Ok(())
                    }
                    _ => Err(RuntimeError::attribute_error(
                        exc.type_name(),
                        name.as_str(),
                    )),
                }
            }
            TypeId::TRACEBACK => {
                let next = match name.as_str() {
                    "tb_next" => normalize_traceback_next_assignment(ptr, value)?,
                    _ => return Err(RuntimeError::attribute_error("traceback", name.as_str())),
                };
                let traceback = unsafe { &mut *(ptr as *mut TracebackViewObject) };
                traceback.set_next(next);
                Ok(())
            }
            TypeId::PROPERTY if name.as_str() == "__doc__" => {
                let descriptor = unsafe { &*(ptr as *const PropertyDescriptor) };
                descriptor.set_doc(value);
                Ok(())
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
                    if let Some(descriptor) = lookup_instance_class_slot(type_id, name)
                        .map(|slot| slot.value)
                        .filter(|descriptor| is_property_descriptor_value(*descriptor))
                        .and_then(property_descriptor_from_value)
                    {
                        invoke_property_setter(vm, descriptor, obj, value)?;
                        return Ok(());
                    }

                    if let Some(descriptor) =
                        lookup_instance_class_slot(type_id, name).map(|slot| slot.value)
                        && let Some(descriptor) = slot_descriptor_from_value(descriptor)
                    {
                        descriptor.set(obj, value).map_err(RuntimeError::from)?;
                        return Ok(());
                    }

                    if let Some(descriptor) =
                        lookup_instance_class_slot(type_id, name).map(|slot| slot.value)
                        && invoke_descriptor_set_in_vm(vm, descriptor, obj, value)?
                    {
                        return Ok(());
                    }

                    if !uses_shaped_user_instance_layout(type_id) {
                        let type_name = user_defined_instance_type_name(type_id);
                        return Err(RuntimeError::attribute_error(
                            Arc::clone(&type_name),
                            format!("'{}' object has no attribute '{}'", type_name, name),
                        ));
                    }

                    if name.as_str() == "__dict__" {
                        return set_user_defined_instance_dict_value(ptr, value);
                    }

                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    let instance_dict = shaped.instance_dict_value();
                    shaped.set_property(name.clone(), value, shape_registry());
                    if let Some(dict_value) = instance_dict {
                        let dict = dict_storage_mut_from_value(dict_value).ok_or_else(|| {
                            RuntimeError::type_error(
                                "instance __dict__ storage is not a dictionary",
                            )
                        })?;
                        dict.set(Value::string(name.clone()), value);
                    }
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

fn normalize_traceback_next_assignment(
    target_ptr: *const (),
    value: Value,
) -> Result<Option<Value>, RuntimeError> {
    if value.is_none() {
        return Ok(None);
    }

    let Some(candidate_ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "tb_next must be a traceback object or None",
        ));
    };
    if extract_type_id(candidate_ptr) != TypeId::TRACEBACK {
        return Err(RuntimeError::type_error(
            "tb_next must be a traceback object or None",
        ));
    }

    let mut cursor = Some(value);
    let mut seen = Vec::new();
    while let Some(traceback_value) = cursor {
        let Some(traceback_ptr) = traceback_value.as_object_ptr() else {
            break;
        };
        if traceback_ptr == target_ptr {
            return Err(RuntimeError::value_error("traceback loop detected"));
        }
        if seen.contains(&traceback_ptr) {
            break;
        }
        seen.push(traceback_ptr);

        let traceback = unsafe { &*(traceback_ptr as *const TracebackViewObject) };
        cursor = traceback.next().filter(|next| !next.is_none());
    }

    Ok(Some(value))
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
        if type_id == TypeId::MODULE {
            let module = unsafe { &*(ptr as *const crate::import::ModuleObject) };
            return if module.del_attr(name.as_str()) {
                Ok(())
            } else {
                Err(RuntimeError::attribute_error("module", name.as_str()))
            };
        }

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
                    if let Some(descriptor) = lookup_instance_class_slot(type_id, name)
                        .map(|slot| slot.value)
                        .filter(|descriptor| is_property_descriptor_value(*descriptor))
                        .and_then(property_descriptor_from_value)
                    {
                        invoke_property_deleter(vm, descriptor, obj)?;
                        return Ok(());
                    }

                    if let Some(descriptor) =
                        lookup_instance_class_slot(type_id, name).map(|slot| slot.value)
                        && let Some(descriptor) = slot_descriptor_from_value(descriptor)
                    {
                        descriptor.delete(obj).map_err(RuntimeError::from)?;
                        return Ok(());
                    }

                    if let Some(descriptor) =
                        lookup_instance_class_slot(type_id, name).map(|slot| slot.value)
                        && invoke_descriptor_delete_in_vm(vm, descriptor, obj)?
                    {
                        return Ok(());
                    }

                    if !uses_shaped_user_instance_layout(type_id) {
                        let type_name = user_defined_instance_type_name(type_id);
                        return Err(RuntimeError::attribute_error(
                            Arc::clone(&type_name),
                            format!("'{}' object has no attribute '{}'", type_name, name),
                        ));
                    }

                    if name.as_str() == "__dict__" {
                        reset_user_defined_instance_dict_value(ptr);
                        return Ok(());
                    }

                    let shaped = unsafe { &mut *(ptr as *mut ShapedObject) };
                    if let Some(dict_value) = shaped.instance_dict_value() {
                        let dict = dict_storage_mut_from_value(dict_value).ok_or_else(|| {
                            RuntimeError::type_error(
                                "instance __dict__ storage is not a dictionary",
                            )
                        })?;
                        let removed = dict.remove(Value::string(name.clone())).is_some();
                        shaped.delete_property_interned(name);
                        if removed {
                            return Ok(());
                        }
                    } else if shaped.delete_property_interned(name) {
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

const EXTENDED_ATTR_NAME_SENTINEL: u8 = u8::MAX;

#[inline]
pub(crate) fn read_attr_name(
    vm: &mut VirtualMachine,
    inline_name: u8,
) -> Result<Arc<str>, RuntimeError> {
    let name_idx = if inline_name != EXTENDED_ATTR_NAME_SENTINEL {
        inline_name as u16
    } else {
        let frame = vm.current_frame_mut();
        let ip = frame.ip as usize;
        let ext_inst = frame
            .code
            .instructions
            .get(ip)
            .copied()
            .ok_or_else(|| RuntimeError::internal("missing AttrName extension"))?;

        if ext_inst.opcode() != Opcode::AttrName as u8 {
            return Err(RuntimeError::internal(
                "extended attribute opcode missing AttrName extension",
            ));
        }

        frame.ip = (ip + 1) as u32;
        ext_inst.imm16()
    };

    vm.current_frame()
        .code
        .names
        .get(name_idx as usize)
        .cloned()
        .ok_or_else(|| RuntimeError::internal("attribute name index out of bounds"))
}

/// AttrName is metadata consumed by the preceding attribute opcode.
#[inline(always)]
pub fn attr_name(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::internal(
        "AttrName executed without a preceding attribute opcode",
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
    let obj = {
        let frame = vm.current_frame();
        frame.get_reg(inst.src1().0)
    };
    let name = match read_attr_name(vm, inst.src2().0) {
        Ok(name) => name,
        Err(err) => return ControlFlow::Error(err),
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
    let (obj, value) = {
        let frame = vm.current_frame();
        let obj = frame.get_reg(inst.dst().0);
        let value = frame.get_reg(inst.src2().0);
        (obj, value)
    };
    let name = match read_attr_name(vm, inst.src1().0) {
        Ok(name) => name,
        Err(err) => return ControlFlow::Error(err),
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
    let obj = {
        let frame = vm.current_frame();
        frame.get_reg(inst.src1().0)
    };
    let name = match read_attr_name(vm, inst.src2().0) {
        Ok(name) => name,
        Err(err) => return ControlFlow::Error(err),
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
    let dst = inst.dst().0;
    let (container, key) = {
        let frame = vm.current_frame();
        (frame.get_reg(inst.src1().0), frame.get_reg(inst.src2().0))
    };

    if let Some(ptr) = container.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        if let Some(list) = list_storage_ref_from_ptr(ptr) {
            return if let Some(idx) = key.as_int() {
                if let Some(val) = list.get(idx) {
                    vm.current_frame_mut().set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::index_error(idx, list.len()))
                }
            } else {
                ControlFlow::Error(RuntimeError::type_error("list indices must be integers"))
            };
        }

        if let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
            return if let Some(idx) = key.as_int() {
                if let Some(val) = tuple.get(idx) {
                    vm.current_frame_mut().set_reg(dst, val);
                    ControlFlow::Continue
                } else {
                    ControlFlow::Error(RuntimeError::index_error(idx, tuple.len()))
                }
            } else {
                ControlFlow::Error(RuntimeError::type_error("tuple indices must be integers"))
            };
        }

        match type_id {
            TypeId::DEQUE => {
                let deque = unsafe { &*(ptr as *const DequeObject) };
                if let Some(idx) = key.as_int() {
                    let Some(index) = isize::try_from(idx).ok() else {
                        return ControlFlow::Error(RuntimeError::index_error(idx, deque.len()));
                    };
                    if let Some(value) = deque.deque().get(index) {
                        vm.current_frame_mut().set_reg(dst, *value);
                        ControlFlow::Continue
                    } else {
                        ControlFlow::Error(RuntimeError::index_error(idx, deque.len()))
                    }
                } else {
                    ControlFlow::Error(RuntimeError::type_error("deque indices must be integers"))
                }
            }
            TypeId::DICT => {
                let dict = unsafe { &*(ptr as *const DictObject) };
                match crate::ops::dict_access::dict_get_item(vm, dict, key) {
                    Ok(Some(val)) => {
                        vm.current_frame_mut().set_reg(dst, val);
                        ControlFlow::Continue
                    }
                    Ok(None) => ControlFlow::Error(RuntimeError::key_error("key not found")),
                    Err(err) => ControlFlow::Error(err),
                }
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_ref_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object is not subscriptable",
                    ));
                };
                match crate::ops::dict_access::dict_get_item(vm, dict, key) {
                    Ok(Some(val)) => {
                        vm.current_frame_mut().set_reg(dst, val);
                        ControlFlow::Continue
                    }
                    Ok(None) => ControlFlow::Error(RuntimeError::key_error("key not found")),
                    Err(err) => ControlFlow::Error(err),
                }
            }
            TypeId::RANGE => {
                let range = unsafe { &*(ptr as *const RangeObject) };
                if let Some(idx) = key.as_int() {
                    if let Some(val) = range.get_value(idx) {
                        vm.current_frame_mut().set_reg(dst, val);
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

        if list_storage_ref_from_ptr(ptr).is_some() {
            return match set_list_item_value(vm, ptr, key, value) {
                Ok(()) => ControlFlow::Continue,
                Err(err) => ControlFlow::Error(err),
            };
        }

        match type_id {
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                match crate::ops::dict_access::dict_set_item(vm, dict, key, value) {
                    Ok(()) => ControlFlow::Continue,
                    Err(err) => ControlFlow::Error(err),
                }
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_mut_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object does not support item assignment",
                    ));
                };
                match crate::ops::dict_access::dict_set_item(vm, dict, key, value) {
                    Ok(()) => ControlFlow::Continue,
                    Err(err) => ControlFlow::Error(err),
                }
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

        if list_storage_ref_from_ptr(ptr).is_some() {
            return match delete_list_item_value(ptr, key) {
                Ok(()) => ControlFlow::Continue,
                Err(err) => ControlFlow::Error(err),
            };
        }

        match type_id {
            TypeId::DICT => {
                let dict = unsafe { &mut *(ptr as *mut DictObject) };
                match crate::ops::dict_access::dict_remove_item(vm, dict, key) {
                    Ok(Some(_)) => ControlFlow::Continue,
                    Ok(None) => ControlFlow::Error(RuntimeError::key_error("key not found")),
                    Err(err) => ControlFlow::Error(err),
                }
            }
            type_id if is_user_defined_type(type_id) => {
                let Some(dict) = dict_storage_mut_from_ptr(ptr) else {
                    return ControlFlow::Error(RuntimeError::type_error(
                        "object does not support item deletion",
                    ));
                };
                match crate::ops::dict_access::dict_remove_item(vm, dict, key) {
                    Ok(Some(_)) => ControlFlow::Continue,
                    Ok(None) => ControlFlow::Error(RuntimeError::key_error("key not found")),
                    Err(err) => ControlFlow::Error(err),
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
        let value = Value::int(string.char_count() as i64).unwrap_or_else(Value::none);
        frame.set_reg(dst, value);
        ControlFlow::Continue
    } else if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);

        let len_val = if let Some(list) = list_storage_ref_from_ptr(ptr) {
            list.len() as i64
        } else if let Some(tuple) = tuple_storage_ref_from_ptr(ptr) {
            tuple.len() as i64
        } else {
            match type_id {
                TypeId::DEQUE => {
                    let deque = unsafe { &*(ptr as *const DequeObject) };
                    deque.len() as i64
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
