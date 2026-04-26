//! LoadMethod opcode: Optimized method lookup with tiered caching.
//!
//! # Encoding
//!
//! `LoadMethod dst, obj, name_idx`
//! - `dst`: receives the method (or function)
//! - `dst+1`: receives self (or marker for unbound)
//! - `obj`: register containing the object
//! - `name_idx`: constant pool index of method name
//!
//! # Three-Tier Caching
//!
//! 1. **Inline Cache (IC)**: Per call-site, fastest (~3 cycles)
//! 2. **Method Cache**: Global type+name cache (~10 cycles)
//! 3. **Full MRO Traversal**: Slow path (~100+ cycles), populates caches
//!
//! # Register Layout
//!
//! After successful LoadMethod:
//! ```text
//! [dst]   = method/function
//! [dst+1] = self (for method calls) or None marker (for descriptors)
//! ```

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::objects::{
    get_attribute_value, lookup_class_metaclass_attr, read_attr_name, super_attribute_value_static,
};
use prism_code::Instruction;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::{MethodSlot, PyClassObject};
use prism_runtime::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{
    builtin_class_mro, class_id_to_type_id, global_class, global_class_version,
};
use prism_runtime::object::type_obj::TypeId;
use std::sync::Arc;

use super::builtin_methods;
use super::method_cache::{CachedMethod, method_cache};

#[derive(Clone, Copy, Debug)]
pub(crate) struct BoundMethodTarget {
    pub callable: Value,
    pub implicit_self: Option<Value>,
}

// =============================================================================
// LoadMethod Handler
// =============================================================================

/// LoadMethod: Optimized method lookup with caching.
///
/// Replaces separate GetAttr + Call with a single LoadMethod + CallMethod pair
/// that avoids creating intermediate BoundMethod objects for O(1) dispatch.
///
/// # Performance
///
/// - IC hit: ~3 cycles (type check + register write)
/// - Cache hit: ~10 cycles (hash lookup + register write)
/// - Cache miss: ~100+ cycles (MRO traversal, cache population)
#[inline(always)]
pub fn load_method(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst().0;
    let obj_reg = inst.src1().0;

    let obj = vm.current_frame().get_reg(obj_reg);

    // Get the name string for error messages and slow path
    let name = match read_attr_name(vm, inst.src2().0) {
        Ok(name) => name,
        Err(err) => return ControlFlow::Error(err),
    };

    // Check if object is heap-allocated
    if let Some(ptr) = obj.as_object_ptr() {
        if let Some(module) = vm.import_resolver.module_from_ptr(ptr) {
            return match module.get_attr(&name) {
                Some(value) => {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(dst, value);
                    frame.set_reg(dst + 1, Value::none());
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::attribute_error("module", name)),
            };
        }

        let type_id = extract_type_id(ptr);

        if matches!(type_id, TypeId::TYPE | TypeId::EXCEPTION_TYPE) {
            return load_runtime_attribute(vm, dst, obj, &name);
        }

        if type_id == TypeId::OBJECT {
            return match resolve_object_instance_method(obj, &name) {
                Ok(cached) => apply_cached_method(vm, dst, obj, cached),
                Err(e) => ControlFlow::Error(e),
            };
        }

        if type_id == TypeId::SUPER {
            return match resolve_super_method(obj, &name) {
                Ok(cached) => apply_cached_method(vm, dst, obj, cached),
                Err(e) => ControlFlow::Error(e),
            };
        }

        if type_id.raw() >= TypeId::FIRST_USER_TYPE
            && let Some(cached) = resolve_user_defined_instance_method(obj, &name)
        {
            return apply_cached_method(vm, dst, obj, cached);
        }

        // Check method cache first (shared across call sites)
        let name_ptr = name.as_ptr() as u64;
        let cache_version = method_cache_version(type_id);
        if let Some(cached) = method_cache().get(type_id, name_ptr, cache_version) {
            if type_id.raw() >= TypeId::FIRST_USER_TYPE && cached.is_descriptor {
                return load_runtime_attribute(vm, dst, obj, &name);
            }
            return apply_cached_method(vm, dst, obj, cached);
        }

        if type_id.raw() >= TypeId::FIRST_USER_TYPE {
            return match resolve_user_defined_method(obj, type_id, &name) {
                Ok(cached) => {
                    method_cache().insert(type_id, name_ptr, cache_version, cached);
                    if cached.is_descriptor {
                        load_runtime_attribute(vm, dst, obj, &name)
                    } else {
                        apply_cached_method(vm, dst, obj, cached)
                    }
                }
                Err(err)
                    if matches!(
                        err.kind,
                        crate::error::RuntimeErrorKind::AttributeError { .. }
                    ) =>
                {
                    match get_attribute_value(vm, obj, &intern(&name)) {
                        Ok(value) => apply_bound_method_target(
                            vm,
                            dst,
                            bind_cached_method_target(
                                obj,
                                cached_method_from_instance_value(value),
                            ),
                        ),
                        Err(fallback) => ControlFlow::Error(fallback),
                    }
                }
                Err(err) => ControlFlow::Error(err),
            };
        }

        // Slow path: resolve method through type system
        match resolve_method(obj, type_id, &name) {
            Ok(cached) => {
                // Populate cache for future calls
                method_cache().insert(type_id, name_ptr, cache_version, cached);
                apply_cached_method(vm, dst, obj, cached)
            }
            Err(err)
                if matches!(
                    err.kind,
                    crate::error::RuntimeErrorKind::AttributeError { .. }
                ) =>
            {
                match get_attribute_value(vm, obj, &intern(&name)) {
                    Ok(value) => apply_bound_method_target(
                        vm,
                        dst,
                        bind_cached_method_target(obj, cached_method_from_instance_value(value)),
                    ),
                    Err(fallback) => ControlFlow::Error(fallback),
                }
            }
            Err(err) => ControlFlow::Error(err),
        }
    } else {
        // Primitive type - get type from value and look up method
        let prim_type_id = get_primitive_type_id(obj);
        let name_ptr = name.as_ptr() as u64;
        let cache_version = method_cache_version(prim_type_id);

        if let Some(cached) = method_cache().get(prim_type_id, name_ptr, cache_version) {
            return apply_cached_method(vm, dst, obj, cached);
        }

        // Resolve method on primitive type
        match resolve_primitive_method(obj, prim_type_id, &name) {
            Ok(cached) => {
                method_cache().insert(prim_type_id, name_ptr, cache_version, cached);
                apply_cached_method(vm, dst, obj, cached)
            }
            Err(e) => ControlFlow::Error(e),
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Extract TypeId from an object pointer.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}

/// Get TypeId for primitive (non-heap) values.
#[inline]
fn get_primitive_type_id(val: Value) -> TypeId {
    if val.is_none() {
        TypeId::NONE
    } else if val.is_bool() {
        TypeId::BOOL
    } else if val.is_int() {
        TypeId::INT
    } else if val.is_float() {
        TypeId::FLOAT
    } else if val.is_string() {
        TypeId::STR
    } else {
        TypeId::OBJECT // Fallback
    }
}

#[inline]
fn method_cache_version(type_id: TypeId) -> u64 {
    if type_id.raw() >= TypeId::FIRST_USER_TYPE {
        global_class_version(ClassId(type_id.raw())).unwrap_or(0)
    } else {
        0
    }
}

/// Apply a cached method resolution to registers.
#[inline]
fn apply_cached_method(
    vm: &mut VirtualMachine,
    dst: u8,
    obj: Value,
    cached: CachedMethod,
) -> ControlFlow {
    let bound = bind_cached_method_target(obj, cached);
    let frame = vm.current_frame_mut();
    frame.set_reg(dst, bound.callable);
    frame.set_reg(dst + 1, bound.implicit_self.unwrap_or_else(Value::none));

    ControlFlow::Continue
}

#[inline]
fn apply_bound_method_target(
    vm: &mut VirtualMachine,
    dst: u8,
    bound: BoundMethodTarget,
) -> ControlFlow {
    let frame = vm.current_frame_mut();
    frame.set_reg(dst, bound.callable);
    frame.set_reg(dst + 1, bound.implicit_self.unwrap_or_else(Value::none));
    ControlFlow::Continue
}

#[inline]
fn load_runtime_attribute(vm: &mut VirtualMachine, dst: u8, obj: Value, name: &str) -> ControlFlow {
    match get_attribute_value(vm, obj, &intern(name)) {
        Ok(value) => apply_bound_method_target(
            vm,
            dst,
            BoundMethodTarget {
                callable: value,
                implicit_self: None,
            },
        ),
        Err(err) => ControlFlow::Error(err),
    }
}

pub(crate) fn resolve_special_method(
    obj: Value,
    name: &str,
) -> Result<BoundMethodTarget, RuntimeError> {
    if let Some(ptr) = obj.as_object_ptr() {
        let type_id = extract_type_id(ptr);
        let cached = match type_id {
            TypeId::TYPE | TypeId::EXCEPTION_TYPE => resolve_type_object_special_method(obj, name)?,
            TypeId::SUPER => resolve_super_method(obj, name)?,
            _ if type_id.raw() >= TypeId::FIRST_USER_TYPE => {
                resolve_user_defined_method(obj, type_id, name)?
            }
            _ => resolve_method(obj, type_id, name)?,
        };

        return Ok(bind_cached_method_target(obj, cached));
    }

    let type_id = get_primitive_type_id(obj);
    let cached = resolve_primitive_method(obj, type_id, name)?;
    Ok(bind_cached_method_target(obj, cached))
}

fn resolve_type_object_special_method(
    obj: Value,
    name: &str,
) -> Result<CachedMethod, RuntimeError> {
    let ptr = obj
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::attribute_error("type", name))?;
    let interned_name = prism_core::intern::intern(name);

    if let Some(class) = class_object_from_type_ptr(ptr)
        && let Some(value) = lookup_class_metaclass_attr(class, &interned_name)
    {
        return Ok(cached_method_from_value(value));
    }

    crate::builtins::builtin_bound_type_attribute_value_static(TypeId::TYPE, obj, &interned_name)?
        .map(cached_method_from_value)
        .ok_or_else(|| RuntimeError::attribute_error("type", name))
}

pub(crate) fn bind_cached_method_target(obj: Value, cached: CachedMethod) -> BoundMethodTarget {
    if cached.is_descriptor
        && let Some(ptr) = cached.method.as_object_ptr()
    {
        match extract_type_id(ptr) {
            TypeId::CLASSMETHOD => {
                let desc = unsafe { &*(ptr as *const ClassMethodDescriptor) };
                return BoundMethodTarget {
                    callable: desc.function(),
                    implicit_self: descriptor_owner_value(obj),
                };
            }
            TypeId::STATICMETHOD => {
                let desc = unsafe { &*(ptr as *const StaticMethodDescriptor) };
                return BoundMethodTarget {
                    callable: desc.function(),
                    implicit_self: None,
                };
            }
            _ => {}
        }
    }

    BoundMethodTarget {
        callable: cached.method,
        implicit_self: (!cached.is_descriptor).then_some(obj),
    }
}

#[inline]
fn class_object_from_type_ptr(ptr: *const ()) -> Option<&'static PyClassObject> {
    if crate::builtins::builtin_type_object_type_id(ptr).is_some() {
        return None;
    }

    Some(unsafe { &*(ptr as *const PyClassObject) })
}

#[inline]
fn descriptor_owner_value(obj: Value) -> Option<Value> {
    let ptr = obj.as_object_ptr()?;
    let type_id = extract_type_id(ptr);
    if type_id == TypeId::TYPE {
        return Some(obj);
    }

    let class = global_class(ClassId(type_id.raw()))?;
    Some(Value::object_ptr(Arc::as_ptr(&class) as *const ()))
}

#[inline]
fn cached_method_from_value(value: Value) -> CachedMethod {
    let Some(ptr) = value.as_object_ptr() else {
        return CachedMethod::descriptor(value);
    };

    match extract_type_id(ptr) {
        // Only Python function objects participate in method binding when
        // stored directly in a Python class dictionary. Plain builtin function
        // objects must stay unbound here so callbacks like
        // `re.compile(...).match` and class attributes such as `helper = len`
        // keep their original call signatures.
        TypeId::FUNCTION | TypeId::CLOSURE => CachedMethod::simple(value),
        _ => CachedMethod::descriptor(value),
    }
}

#[inline]
fn cached_method_from_user_class_slot(slot: MethodSlot) -> CachedMethod {
    let Some(ptr) = slot.value.as_object_ptr() else {
        return CachedMethod::descriptor(slot.value);
    };

    match extract_type_id(ptr) {
        TypeId::FUNCTION | TypeId::CLOSURE => CachedMethod::simple(slot.value),
        TypeId::BUILTIN_FUNCTION
            if slot.defining_class.0 >= TypeId::FIRST_USER_TYPE
                && global_class(slot.defining_class)
                    .is_some_and(|class| class.is_native_heaptype()) =>
        {
            CachedMethod::simple(slot.value)
        }
        _ => CachedMethod::descriptor(slot.value),
    }
}

#[inline]
fn cached_method_from_instance_value(value: Value) -> CachedMethod {
    // Instance attributes are ordinary values. Even when the stored value is a
    // function object, Python returns it unbound and does not inject `self`.
    CachedMethod::descriptor(value)
}

fn resolve_object_instance_method(obj: Value, name: &str) -> Result<CachedMethod, RuntimeError> {
    let ptr = obj
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::attribute_error("object", name))?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };

    if let Some(value) = shaped.get_property(name) {
        return Ok(cached_method_from_instance_value(value));
    }

    Err(RuntimeError::attribute_error("object", name))
}

fn resolve_super_method(obj: Value, name: &str) -> Result<CachedMethod, RuntimeError> {
    let interned_name = prism_core::intern::intern(name);
    let value = super_attribute_value_static(obj, &interned_name)?
        .ok_or_else(|| RuntimeError::attribute_error("super", name))?;
    Ok(CachedMethod::descriptor(value))
}

fn resolve_user_defined_instance_method(obj: Value, name: &str) -> Option<CachedMethod> {
    let ptr = obj.as_object_ptr()?;
    let shaped = unsafe { &*(ptr as *const ShapedObject) };
    shaped
        .get_property(name)
        .map(cached_method_from_instance_value)
}

/// Resolve a method on a heap object by traversing its type's MRO.
fn resolve_method(obj: Value, type_id: TypeId, name: &str) -> Result<CachedMethod, RuntimeError> {
    // For dict objects, check __dict__ first
    // TODO: Implement full MRO traversal when type system is complete

    // Check for builtin methods based on type
    match type_id {
        TypeId::OBJECT => super::resolve_builtin_instance_method(TypeId::OBJECT, name)
            .ok_or_else(|| RuntimeError::attribute_error("object", name)),
        TypeId::DEQUE => resolve_deque_method(name),
        TypeId::REGEX_PATTERN => resolve_regex_pattern_method(name),
        TypeId::REGEX_MATCH => resolve_regex_match_method(name),
        TypeId::LIST => resolve_list_method(name),
        TypeId::DICT => resolve_dict_method(name),
        TypeId::MAPPING_PROXY => resolve_mapping_proxy_method(name),
        TypeId::BYTES => resolve_bytes_method(name),
        TypeId::BYTEARRAY => resolve_bytearray_method(name),
        TypeId::MEMORYVIEW => builtin_methods::resolve_memoryview_method(name)
            .ok_or_else(|| RuntimeError::attribute_error(type_id.name(), name)),
        TypeId::PROPERTY => resolve_property_method(name),
        TypeId::ITERATOR => builtin_methods::resolve_iterator_method(name)
            .ok_or_else(|| RuntimeError::attribute_error(type_id.name(), name)),
        TypeId::EXCEPTION => resolve_exception_method(obj, name),
        TypeId::TUPLE => resolve_tuple_method(name),
        TypeId::STR => resolve_str_method(name),
        TypeId::SET | TypeId::FROZENSET => resolve_set_method(type_id, name),
        TypeId::INT => resolve_int_method(name),
        TypeId::FLOAT => resolve_float_method(name),
        TypeId::GENERATOR => resolve_generator_method(name),
        TypeId::FUNCTION | TypeId::CLOSURE => resolve_function_method(name),
        TypeId::CLASSMETHOD | TypeId::STATICMETHOD => {
            super::resolve_builtin_instance_method(type_id, name)
                .ok_or_else(|| RuntimeError::attribute_error(type_id.name(), name))
        }
        TypeId::TYPE => resolve_type_object_method(obj, name),
        _ if type_id.raw() >= TypeId::FIRST_USER_TYPE => {
            resolve_user_defined_method(obj, type_id, name)
        }
        _ => {
            // Check if object has instance __dict__
            // For now, return AttributeError
            Err(RuntimeError::attribute_error(type_id.name(), name))
        }
    }
}

fn resolve_type_object_method(obj: Value, name: &str) -> Result<CachedMethod, RuntimeError> {
    let ptr = obj
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::attribute_error("type", name))?;

    if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr) {
        let interned_name = prism_core::intern::intern(name);
        return crate::builtins::builtin_bound_type_attribute_value_static(
            represented,
            obj,
            &interned_name,
        )?
        .map(CachedMethod::descriptor)
        .ok_or_else(|| RuntimeError::attribute_error("type", name));
    }

    let Some(class) = class_object_from_type_ptr(ptr) else {
        return Err(RuntimeError::attribute_error("type", name));
    };

    let interned_name = prism_core::intern::intern(name);
    if let Some(value) = crate::builtins::heap_type_attribute_value_static(
        ptr as *const PyClassObject,
        &interned_name,
    )? {
        // Attributes resolved from the class hierarchy have already had their
        // descriptor semantics materialized by `heap_type_attribute_value_static`.
        // Treat them as descriptor results here so `LoadMethod` does not inject
        // the class object as an implicit first argument for plain functions or
        // staticmethods looked up on the class itself.
        return Ok(CachedMethod::descriptor(value));
    }

    if let Some(value) = lookup_class_metaclass_attr(class, &interned_name) {
        return Ok(cached_method_from_value(value));
    }

    crate::builtins::builtin_type_method_value(TypeId::TYPE, name)
        .map(CachedMethod::simple)
        .ok_or_else(|| RuntimeError::attribute_error("type", name))
}

/// Resolve a method on a primitive value.
fn resolve_primitive_method(
    obj: Value,
    type_id: TypeId,
    name: &str,
) -> Result<CachedMethod, RuntimeError> {
    let direct = match type_id {
        TypeId::STR => resolve_str_method(name),
        TypeId::INT => resolve_int_method(name),
        TypeId::FLOAT => resolve_float_method(name),
        TypeId::BOOL => resolve_bool_method(name),
        TypeId::NONE => Err(RuntimeError::attribute_error("NoneType", name)),
        _ => Err(RuntimeError::attribute_error(type_id.name(), name)),
    };

    if direct.is_ok() {
        return direct;
    }

    let interned_name = prism_core::intern::intern(name);
    for class_id in builtin_class_mro(type_id) {
        let builtin_owner = class_id_to_type_id(class_id);
        if let Some(value) = crate::builtins::builtin_bound_type_attribute_value_static(
            builtin_owner,
            obj,
            &interned_name,
        )? {
            return Ok(CachedMethod::descriptor(value));
        }
    }

    direct
}

// =============================================================================
// Type-Specific Method Resolution
// =============================================================================

/// Resolve builtin list methods.
fn resolve_list_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_list_method(name) {
        return Ok(cached);
    }

    match name {
        "pop" | "clear" | "index" | "count" => Err(RuntimeError::attribute_error(
            "list",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("list", name)),
    }
}

/// Resolve builtin deque methods.
fn resolve_deque_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_deque_method(name) {
        return Ok(cached);
    }

    match name {
        "extend" | "extendleft" | "rotate" | "clear" | "count" | "index" | "insert" | "remove"
        | "reverse" | "copy" => Err(RuntimeError::attribute_error(
            "deque",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("deque", name)),
    }
}

/// Resolve builtin regex pattern methods.
fn resolve_regex_pattern_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_regex_pattern_method(name) {
        return Ok(cached);
    }

    Err(RuntimeError::attribute_error("Pattern", name))
}

/// Resolve builtin regex match methods.
fn resolve_regex_match_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_regex_match_method(name) {
        return Ok(cached);
    }

    Err(RuntimeError::attribute_error("Match", name))
}

/// Resolve builtin dict methods.
fn resolve_dict_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_dict_method(name) {
        return Ok(cached);
    }

    match name {
        "get" | "pop" | "popitem" | "clear" | "update" | "setdefault" | "copy" => Err(
            RuntimeError::attribute_error("dict", format!("{} (not yet implemented)", name)),
        ),
        _ => Err(RuntimeError::attribute_error("dict", name)),
    }
}

/// Resolve builtin mappingproxy methods.
fn resolve_mapping_proxy_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_mapping_proxy_method(name) {
        return Ok(cached);
    }

    match name {
        "get" | "copy" | "keys" | "values" | "items" | "__len__" | "__contains__" => {
            Err(RuntimeError::attribute_error(
                "mappingproxy",
                format!("{} (not yet implemented)", name),
            ))
        }
        _ => Err(RuntimeError::attribute_error("mappingproxy", name)),
    }
}

/// Resolve builtin bytearray methods.
fn resolve_bytes_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_bytes_method(name) {
        return Ok(cached);
    }

    Err(RuntimeError::attribute_error("bytes", name))
}

/// Resolve builtin bytearray methods.
fn resolve_bytearray_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_bytearray_method(name) {
        return Ok(cached);
    }

    match name {
        "copy" => Err(RuntimeError::attribute_error(
            "bytearray",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("bytearray", name)),
    }
}

/// Resolve builtin tuple methods.
fn resolve_tuple_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    builtin_methods::resolve_tuple_method(name)
        .ok_or_else(|| RuntimeError::attribute_error("tuple", name))
}

/// Resolve builtin str methods.
fn resolve_str_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_str_method(name) {
        return Ok(cached);
    }

    match name {
        "upper" | "lower" | "strip" | "lstrip" | "rstrip" | "split" | "join" | "replace"
        | "find" | "rfind" | "index" | "rindex" | "count" | "startswith" | "endswith"
        | "isalpha" | "isdigit" | "isalnum" | "isspace" | "isupper" | "islower" | "title"
        | "capitalize" | "swapcase" | "center" | "ljust" | "rjust" | "zfill" | "format"
        | "encode" => Err(RuntimeError::attribute_error(
            "str",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("str", name)),
    }
}

/// Resolve builtin set methods.
fn resolve_set_method(type_id: TypeId, name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_set_method(type_id, name) {
        return Ok(cached);
    }

    let type_name = type_id.name();
    match name {
        "add"
        | "remove"
        | "discard"
        | "pop"
        | "clear"
        | "copy"
        | "update"
        | "union"
        | "intersection"
        | "difference"
        | "symmetric_difference"
        | "issubset"
        | "issuperset"
        | "isdisjoint" => Err(RuntimeError::attribute_error(
            type_name,
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error(type_name, name)),
    }
}

/// Resolve builtin int methods.
fn resolve_int_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_int_method(name) {
        return Ok(cached);
    }

    match name {
        "bit_length" | "bit_count" | "to_bytes" | "from_bytes" | "conjugate" | "numerator"
        | "denominator" | "real" | "imag" => Err(RuntimeError::attribute_error(
            "int",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("int", name)),
    }
}

/// Resolve builtin float methods.
fn resolve_float_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    match name {
        "is_integer" | "hex" | "fromhex" | "conjugate" | "real" | "imag" => Err(
            RuntimeError::attribute_error("float", format!("{} (not yet implemented)", name)),
        ),
        _ => Err(RuntimeError::attribute_error("float", name)),
    }
}

/// Resolve builtin generator methods.
fn resolve_generator_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_generator_method(name) {
        return Ok(cached);
    }

    match name {
        "send" => Err(RuntimeError::attribute_error(
            "generator",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("generator", name)),
    }
}

/// Resolve builtin property methods.
fn resolve_property_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_property_method(name) {
        return Ok(cached);
    }

    Err(RuntimeError::attribute_error("property", name))
}

fn resolve_exception_method(obj: Value, name: &str) -> Result<CachedMethod, RuntimeError> {
    if let Some(cached) = builtin_methods::resolve_exception_method(name) {
        return Ok(cached);
    }

    let type_name = unsafe {
        crate::builtins::ExceptionValue::from_value(obj)
            .map(|exception| exception.type_name())
            .unwrap_or("BaseException")
    };
    Err(RuntimeError::attribute_error(type_name, name))
}

/// Resolve builtin bool methods.
fn resolve_bool_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    // bool inherits from int
    resolve_int_method(name)
}

/// Resolve function/closure methods.
fn resolve_function_method(name: &str) -> Result<CachedMethod, RuntimeError> {
    builtin_methods::resolve_function_method(name)
        .ok_or_else(|| RuntimeError::attribute_error("function", name))
}

/// Resolve a method defined on a heap-allocated Python class.
fn resolve_user_defined_method(
    obj: Value,
    type_id: TypeId,
    name: &str,
) -> Result<CachedMethod, RuntimeError> {
    let class = global_class(ClassId(type_id.raw()))
        .ok_or_else(|| RuntimeError::attribute_error(type_id.name(), name))?;
    let interned_name = prism_core::intern::intern(name);

    if let Some(slot) = class.lookup_method_published(&interned_name) {
        return Ok(cached_method_from_user_class_slot(slot));
    }

    for &class_id in class.mro().iter().skip(1) {
        if class_id.0 >= TypeId::FIRST_USER_TYPE {
            continue;
        }

        let owner = class_id_to_type_id(class_id);
        if let Some(cached) = super::resolve_builtin_instance_method(owner, name) {
            return Ok(cached);
        }

        if let Some(value) =
            crate::builtins::builtin_bound_type_attribute_value_static(owner, obj, &interned_name)?
        {
            return Ok(CachedMethod::descriptor(value));
        }
    }

    Err(RuntimeError::attribute_error(type_id.name(), name))
}
