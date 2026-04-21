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
    get_attribute_value, lookup_class_metaclass_attr, super_attribute_value_static,
};
use prism_code::Instruction;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::class::PyClassObject;
use prism_runtime::object::descriptor::{ClassMethodDescriptor, StaticMethodDescriptor};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::shaped_object::ShapedObject;
use prism_runtime::object::type_builtins::{builtin_class_mro, class_id_to_type_id, global_class};
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
    let name_idx = inst.src2().0 as u16;

    let obj = vm.current_frame().get_reg(obj_reg);

    // Get the name string for error messages and slow path
    let name = vm.current_frame().get_name(name_idx).clone();

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

        if type_id == TypeId::TYPE {
            return match resolve_method(vm, obj, type_id, &name) {
                Ok(cached) => apply_cached_method(vm, dst, obj, cached),
                Err(e) => ControlFlow::Error(e),
            };
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
        if let Some(cached) = method_cache().get(type_id, name_ptr) {
            return apply_cached_method(vm, dst, obj, cached);
        }

        if type_id.raw() >= TypeId::FIRST_USER_TYPE {
            return match resolve_user_defined_method(obj, type_id, &name) {
                Ok(cached) => {
                    method_cache().insert(type_id, name_ptr, cached);
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
        match resolve_method(vm, obj, type_id, &name) {
            Ok(cached) => {
                // Populate cache for future calls
                method_cache().insert(type_id, name_ptr, cached);
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

        if let Some(cached) = method_cache().get(prim_type_id, name_ptr) {
            return apply_cached_method(vm, dst, obj, cached);
        }

        // Resolve method on primitive type
        match resolve_primitive_method(vm, obj, prim_type_id, &name) {
            Ok(cached) => {
                method_cache().insert(prim_type_id, name_ptr, cached);
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

pub(crate) fn resolve_special_method(
    obj: Value,
    name: &str,
) -> Result<BoundMethodTarget, RuntimeError> {
    let ptr = obj
        .as_object_ptr()
        .ok_or_else(|| RuntimeError::attribute_error(obj.type_name(), name))?;
    let type_id = extract_type_id(ptr);

    let cached = match type_id {
        TypeId::TYPE => resolve_type_object_special_method(obj, name)?,
        TypeId::OBJECT => resolve_object_instance_method(obj, name)?,
        _ if type_id.raw() >= TypeId::FIRST_USER_TYPE => {
            if let Some(cached) = resolve_user_defined_instance_method(obj, name) {
                cached
            } else {
                resolve_user_defined_method(obj, type_id, name)?
            }
        }
        _ => return Err(RuntimeError::attribute_error(type_id.name(), name)),
    };

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
        TypeId::FUNCTION | TypeId::CLOSURE | TypeId::BUILTIN_FUNCTION => {
            CachedMethod::simple(value)
        }
        _ => CachedMethod::descriptor(value),
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
fn resolve_method(
    vm: &VirtualMachine,
    obj: Value,
    type_id: TypeId,
    name: &str,
) -> Result<CachedMethod, RuntimeError> {
    // For dict objects, check __dict__ first
    // TODO: Implement full MRO traversal when type system is complete

    // Check for builtin methods based on type
    match type_id {
        TypeId::DEQUE => resolve_deque_method(name),
        TypeId::REGEX_PATTERN => resolve_regex_pattern_method(name),
        TypeId::REGEX_MATCH => resolve_regex_match_method(name),
        TypeId::LIST => resolve_list_method(name),
        TypeId::DICT => resolve_dict_method(name),
        TypeId::MAPPING_PROXY => resolve_mapping_proxy_method(name),
        TypeId::BYTES => resolve_bytes_method(name),
        TypeId::BYTEARRAY => resolve_bytearray_method(name),
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
    _vm: &VirtualMachine,
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
    match name {
        "count" | "index" => Err(RuntimeError::attribute_error(
            "tuple",
            format!("{} (not yet implemented)", name),
        )),
        _ => Err(RuntimeError::attribute_error("tuple", name)),
    }
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

    if let Some(slot) = class.lookup_method(&interned_name, global_class) {
        return Ok(cached_method_from_value(slot.value));
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VirtualMachine;
    use crate::import::ModuleObject;
    use prism_code::CodeObject;
    use prism_code::{Instruction, Opcode, Register};
    use prism_core::intern::intern;
    use prism_runtime::object::class::PyClassObject;
    use prism_runtime::object::mro::ClassId;
    use prism_runtime::object::shape::shape_registry;
    use prism_runtime::object::shaped_object::ShapedObject;
    use prism_runtime::object::type_builtins::{
        SubclassBitmap, builtin_class_mro, class_id_to_type_id, register_global_class,
    };
    use prism_runtime::object::views::CodeObjectView;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::function::FunctionObject;
    use std::sync::Arc;

    fn register_test_class(class: PyClassObject) -> Arc<PyClassObject> {
        let mut bitmap = SubclassBitmap::new();
        for &class_id in class.mro() {
            bitmap.set_bit(TypeId::from_raw(class_id.0));
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

    fn register_dict_subclass(name: &str) -> Arc<PyClassObject> {
        let class = PyClassObject::new(intern(name), &[ClassId(TypeId::DICT.raw())], |id| {
            (id.0 < TypeId::FIRST_USER_TYPE).then(|| {
                builtin_class_mro(class_id_to_type_id(id))
                    .into_iter()
                    .collect()
            })
        })
        .expect("dict subclass should build");
        register_test_class(class)
    }

    fn vm_with_names(names: &[&str]) -> VirtualMachine {
        let mut code = CodeObject::new("test_load_method", "<test>");
        code.names = names
            .iter()
            .map(|name| Arc::<str>::from(*name))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0).expect("frame push failed");
        vm
    }

    #[test]
    fn test_get_primitive_type_id() {
        assert_eq!(get_primitive_type_id(Value::none()), TypeId::NONE);
        assert_eq!(get_primitive_type_id(Value::bool(true)), TypeId::BOOL);
        assert_eq!(get_primitive_type_id(Value::int_unchecked(42)), TypeId::INT);
        assert_eq!(
            get_primitive_type_id(Value::string(prism_core::intern::intern("Path"))),
            TypeId::STR
        );
    }

    #[test]
    fn test_resolve_list_method_known() {
        let result = resolve_list_method("append");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_list_method_unimplemented_known_method() {
        let result = resolve_list_method("sort");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("sort"));
    }

    #[test]
    fn test_resolve_list_method_unknown() {
        // Unknown methods should return plain attribute error
        let result = resolve_list_method("foobar");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("foobar"));
    }

    #[test]
    fn test_resolve_deque_method_known() {
        let result = resolve_deque_method("append");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_dict_method_known() {
        let result = resolve_dict_method("keys");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_bytes_method_known() {
        let result = resolve_bytes_method("decode");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_str_method_known() {
        let result = resolve_str_method("upper");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_primitive_method_inherits_object_new_for_none() {
        let vm = vm_with_names(&[]);

        let cached = resolve_primitive_method(&vm, Value::none(), TypeId::NONE, "__new__")
            .expect("None should inherit object.__new__ for method calls");
        let method_ptr = cached
            .method
            .as_object_ptr()
            .expect("object.__new__ should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert!(cached.is_descriptor);
        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "object.__new__");
        assert!(builtin.bound_self().is_none());
    }

    #[test]
    fn test_resolve_int_method_known() {
        let result = resolve_int_method("bit_length");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_generator_close_method_known() {
        let result = resolve_generator_method("close");
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_generator_throw_method_known() {
        let result = resolve_generator_method("throw");
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_method_reads_module_attributes_without_binding_self() {
        let mut vm = vm_with_names(&["iskeyword"]);
        let module = Arc::new(ModuleObject::new("keyword"));
        let builtin_len = vm.builtins.get("len").expect("len builtin should exist");
        module.set_attr("iskeyword", builtin_len);
        vm.import_resolver.insert_module("keyword", module.clone());
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2), builtin_len);
        assert!(vm.current_frame().get_reg(3).is_none());
    }

    #[test]
    fn test_load_method_resolves_str_maketrans_without_implicit_self() {
        let mut vm = vm_with_names(&["maketrans"]);
        vm.current_frame_mut()
            .set_reg(1, Value::string(prism_core::intern::intern("seed")));

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("str.maketrans should resolve to a builtin function");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "str.maketrans");
        assert!(builtin.bound_self().is_none());
        assert!(vm.current_frame().get_reg(3).is_none());
    }

    #[test]
    fn test_load_method_resolves_bytes_decode_with_implicit_self() {
        let mut vm = vm_with_names(&["decode"]);
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
        let bytes_value = Value::object_ptr(bytes_ptr as *const ());
        vm.current_frame_mut().set_reg(1, bytes_value);

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("bytes.decode should resolve to a builtin function");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "bytes.decode");
        assert_eq!(vm.current_frame().get_reg(3), bytes_value);

        unsafe {
            drop(Box::from_raw(bytes_ptr));
        }
    }

    #[test]
    fn test_load_method_resolves_none_new_without_implicit_self() {
        let mut vm = vm_with_names(&["__new__"]);
        vm.current_frame_mut().set_reg(1, Value::none());

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("None.__new__ should resolve to a builtin function");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "object.__new__");
        assert!(builtin.bound_self().is_none());
        assert!(vm.current_frame().get_reg(3).is_none());
    }

    #[test]
    fn test_load_method_resolves_code_positions_without_implicit_self() {
        let mut vm = vm_with_names(&["co_positions"]);
        let code_view = CodeObjectView::new(Arc::new(CodeObject::new("trace_target", "<test>")));
        let code_value = Value::object_ptr(Box::into_raw(Box::new(code_view)) as *const ());
        vm.current_frame_mut().set_reg(1, code_value);

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("code.co_positions should resolve to a builtin function");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "code.co_positions");
        assert_eq!(builtin.bound_self(), Some(code_value));
        assert!(vm.current_frame().get_reg(3).is_none());
    }

    #[test]
    fn test_resolve_user_defined_method_inherits_from_parent() {
        let (func_ptr, func_value) = make_test_function_value("method");

        let mut parent = PyClassObject::new_simple(intern("Parent"));
        parent.set_attr(intern("method"), func_value);
        let parent = register_test_class(parent);

        let child = PyClassObject::new(intern("Child"), &[parent.class_id()], |id| {
            (id == parent.class_id()).then(|| parent.mro().iter().copied().collect())
        })
        .expect("child class should build");
        let child = register_test_class(child);
        let instance = ShapedObject::new(child.class_type_id(), child.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));
        let instance_value = Value::object_ptr(instance_ptr as *const ());

        let cached = resolve_user_defined_method(instance_value, child.class_type_id(), "method")
            .expect("expected inherited method lookup");
        assert_eq!(cached.method, func_value);
        assert!(!cached.is_descriptor);

        unsafe {
            drop(Box::from_raw(instance_ptr));
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_resolve_user_defined_non_callable_uses_unbound_marker() {
        let class = PyClassObject::new_simple(intern("AttrHolder"));
        class.set_attr(intern("value"), Value::int_unchecked(42));
        let class = register_test_class(class);
        let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));
        let instance_value = Value::object_ptr(instance_ptr as *const ());

        let cached = resolve_user_defined_method(instance_value, class.class_type_id(), "value")
            .expect("expected value");
        assert_eq!(cached.method, Value::int_unchecked(42));
        assert!(cached.is_descriptor);

        unsafe {
            drop(Box::from_raw(instance_ptr));
        }
    }

    #[test]
    fn test_resolve_user_defined_method_inherits_builtin_dict_setdefault_for_heap_class() {
        let class = register_dict_subclass("DictSubclassSetDefault");
        let instance =
            ShapedObject::new_dict_backed(class.class_type_id(), class.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));
        let instance_value = Value::object_ptr(instance_ptr as *const ());

        let cached =
            resolve_user_defined_method(instance_value, class.class_type_id(), "setdefault")
                .expect("heap dict subclass should inherit dict.setdefault");
        let method_ptr = cached
            .method
            .as_object_ptr()
            .expect("dict.setdefault should be heap allocated");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "dict.setdefault");
        assert!(!cached.is_descriptor);

        unsafe {
            drop(Box::from_raw(instance_ptr));
        }
    }

    #[test]
    fn test_resolve_user_defined_method_inherits_builtin_object_init_for_heap_class() {
        let class = register_test_class(PyClassObject::new_simple(intern("InitCarrier")));
        let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));
        let instance_value = Value::object_ptr(instance_ptr as *const ());

        let cached = resolve_user_defined_method(instance_value, class.class_type_id(), "__init__")
            .expect("heap instances should inherit object.__init__");
        let method_ptr = cached
            .method
            .as_object_ptr()
            .expect("object.__init__ should resolve to a bound builtin");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "object.__init__");
        assert!(cached.is_descriptor);

        unsafe {
            drop(Box::from_raw(instance_ptr));
        }
    }

    #[test]
    fn test_load_method_resolves_builtin_dict_setdefault_for_heap_class() {
        let class = register_dict_subclass("DictSubclassLoadMethod");
        let instance =
            ShapedObject::new_dict_backed(class.class_type_id(), class.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));

        let mut vm = vm_with_names(&["setdefault"]);
        let instance_value = Value::object_ptr(instance_ptr as *const ());
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("dict.setdefault should resolve to a builtin function");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "dict.setdefault");
        assert_eq!(vm.current_frame().get_reg(3), instance_value);

        unsafe {
            drop(Box::from_raw(instance_ptr));
        }
    }

    #[test]
    fn test_load_method_resolves_inherited_object_init_for_heap_class() {
        let class = register_test_class(PyClassObject::new_simple(intern("InitLoadCarrier")));
        let instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
        let instance_ptr = Box::into_raw(Box::new(instance));

        let mut vm = vm_with_names(&["__init__"]);
        let instance_value = Value::object_ptr(instance_ptr as *const ());
        vm.current_frame_mut().set_reg(1, instance_value);

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));

        let method_value = vm.current_frame().get_reg(2);
        let method_ptr = method_value
            .as_object_ptr()
            .expect("object.__init__ should resolve to a bound builtin");
        let builtin = unsafe { &*(method_ptr as *const crate::builtins::BuiltinFunctionObject) };

        assert_eq!(extract_type_id(method_ptr), TypeId::BUILTIN_FUNCTION);
        assert_eq!(builtin.name(), "object.__init__");
        assert!(vm.current_frame().get_reg(3).is_none());

        unsafe {
            drop(Box::from_raw(instance_ptr));
        }
    }

    #[test]
    fn test_resolve_type_object_method_inherits_builtin_object_ne_for_heap_class() {
        let class = register_test_class(PyClassObject::new_simple(intern("HeapType")));
        let class_value = Value::object_ptr(Arc::as_ptr(&class) as *const ());

        let cached = resolve_type_object_method(class_value, "__ne__")
            .expect("heap class should inherit object.__ne__");
        let method_ptr = cached
            .method
            .as_object_ptr()
            .expect("object.__ne__ should be heap allocated");
        let header = unsafe { &*(method_ptr as *const ObjectHeader) };
        assert_eq!(header.type_id, TypeId::BUILTIN_FUNCTION);
        assert!(!cached.is_descriptor);
    }

    #[test]
    fn test_resolve_object_instance_method_returns_callable_instance_property_unbound() {
        let (func_ptr, func_value) = make_test_function_value("instance_method");
        let registry = shape_registry();
        let mut object = ShapedObject::with_empty_shape(registry.empty_shape());
        object.set_property(intern("instance_method"), func_value, registry);
        let object_ptr = Box::into_raw(Box::new(object));
        let object_value = Value::object_ptr(object_ptr as *const ());

        let cached = resolve_object_instance_method(object_value, "instance_method")
            .expect("instance callable should resolve");
        assert_eq!(cached.method, func_value);
        assert!(cached.is_descriptor);

        unsafe {
            drop(Box::from_raw(object_ptr));
            drop(Box::from_raw(func_ptr));
        }
    }

    #[test]
    fn test_load_method_prefers_user_defined_instance_callable_without_binding_self() {
        let (func_ptr, func_value) = make_test_function_value("instance_callable");

        let class = register_test_class(PyClassObject::new_simple(intern("CallableHolder")));
        let mut instance = ShapedObject::new(class.class_type_id(), class.instance_shape().clone());
        instance.set_property(intern("instance_callable"), func_value, shape_registry());
        let instance_ptr = Box::into_raw(Box::new(instance));

        let mut vm = vm_with_names(&["instance_callable"]);
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(instance_ptr as *const ()));

        let inst = Instruction::op_dss(
            Opcode::LoadMethod,
            Register::new(2),
            Register::new(1),
            Register::new(0),
        );
        assert!(matches!(load_method(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(vm.current_frame().get_reg(2), func_value);
        assert!(vm.current_frame().get_reg(3).is_none());

        unsafe {
            drop(Box::from_raw(instance_ptr));
            drop(Box::from_raw(func_ptr));
        }
    }
}
