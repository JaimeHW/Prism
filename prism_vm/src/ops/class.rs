//! Class operation opcode handlers.
//!
//! Implements the BUILD_CLASS opcode for Python class creation.
//!
//! # Python Class Creation Protocol
//!
//! When Python executes a class statement like:
//! ```python
//! class MyClass(Base1, Base2):
//!     x = 1
//!     def method(self):
//!         pass
//! ```
//!
//! The compiler generates:
//! 1. Evaluate base classes into registers
//! 2. Create class body CodeObject
//! 3. Execute BUILD_CLASS opcode
//!
//! BUILD_CLASS then:
//! 1. Executes the class body CodeObject to populate namespace dict
//! 2. Calls the metaclass (default: `type`) with (name, bases, namespace)
//! 3. Returns the new class object
//!
//! # Performance Notes
//!
//! - Class creation is not on the hot path - happens once per class definition
//! - Focus is on correctness and CPython compatibility
//! - Future optimization: cache class objects for repeated module imports

use crate::VirtualMachine;
use crate::builtins::{builtin_type_object_for_type_id, builtin_type_object_type_id};
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::{invoke_callable_value, invoke_callable_value_with_keywords};
use crate::ops::objects::{get_attribute_value, resolve_class_attribute_in_vm};
use prism_code::{CodeObject, Instruction, Opcode};
use prism_core::Value;
#[cfg(test)]
use prism_core::intern::interned_by_ptr;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::class::{ClassDict, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{
    global_class, global_class_registry, register_global_class, type_new_with_metaclass,
    unregister_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
#[cfg(test)]
use prism_runtime::types::string::StringObject;
use prism_runtime::types::tuple::TupleObject;
use smallvec::SmallVec;
use std::sync::Arc;

// =============================================================================
// BUILD_CLASS Handler
// =============================================================================

/// BuildClass: Create a new class from bases and body code object.
///
/// # Opcode Format (DstSrcSrc)
/// - primary instruction dst: Destination register for the new class object
/// - primary instruction imm16: Class body `CodeObject` constant index
/// - trailing `ClassMeta` dst: Number of base classes
///
/// # Register Layout
/// ```text
/// r[dst]     <- result (new class object)
/// r[dst+1]   <- base class 0
/// r[dst+2]   <- base class 1
/// ...
/// r[dst+n]   <- base class n-1
/// ```
///
/// # Algorithm
/// 1. Load class body CodeObject from constants
/// 2. Extract class name from class body code object metadata
/// 3. Collect base classes from registers
/// 4. Create PyClassObject with name, bases, namespace
/// 5. Store result in destination register
///
/// # Error Conditions
/// - Invalid constant index
/// - Non-CodeObject in constant slot
/// - Invalid base class type
/// - MRO computation failure (diamond inheritance conflicts)
#[inline(always)]
pub fn build_class(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let metadata = match read_class_metadata(vm, inst) {
        Ok(metadata) => metadata,
        Err(err) => return ControlFlow::Error(err),
    };
    build_class_impl(vm, inst, metadata, None)
}

/// Build a new class using an explicitly supplied metaclass value.
#[inline(always)]
pub fn build_class_with_metaclass(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let metadata = match read_class_metadata(vm, inst) {
        Ok(metadata) => metadata,
        Err(err) => return ControlFlow::Error(err),
    };
    let base_count = metadata.base_count;
    let metaclass_reg = inst.dst().0 + 1 + base_count as u8;
    let explicit_metaclass = vm.current_frame().get_reg(metaclass_reg);
    build_class_impl(vm, inst, metadata, Some(explicit_metaclass))
}

/// Class metadata extensions should be consumed by the preceding build opcode.
#[inline(always)]
pub fn class_meta(_vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    ControlFlow::Error(RuntimeError::internal(
        "CLASS_META executed without a preceding BUILD_CLASS consumer",
    ))
}

#[inline(always)]
fn build_class_impl(
    vm: &mut VirtualMachine,
    inst: Instruction,
    metadata: ClassMetadata,
    explicit_metaclass: Option<Value>,
) -> ControlFlow {
    let dst_reg = inst.dst().0;
    let code_idx = metadata.code_idx;
    let base_count = metadata.base_count;
    let kwargc = metadata.kwargc;
    let kwnames_idx = metadata.kwnames_idx;

    // Resolve the class body code object and class name from the constant pool.
    let class_body = {
        let frame = vm.current_frame();
        let code_const = frame.get_const(code_idx);
        match extract_class_body_from_code_const(code_const, &frame.code.nested_code_objects) {
            Some(code) => code,
            None => {
                return ControlFlow::Error(RuntimeError::type_error(
                    "class body must be a valid code object constant",
                ));
            }
        }
    };
    let class_name = intern(class_body.name.as_ref());

    // Collect base classes from registers
    let frame = vm.current_frame();
    let mut base_class_ids = Vec::with_capacity(base_count);
    for i in 0..base_count {
        let base_val = frame.get_reg(dst_reg + 1 + i as u8);
        match extract_class_id(base_val) {
            Some(class_id) => base_class_ids.push(class_id),
            None => {
                // If no bases provided, inherit from object
                if base_count == 0 {
                    break;
                }
                return ControlFlow::Error(RuntimeError::type_error(format!(
                    "base class {} is not a valid class",
                    i
                )));
            }
        }
    }
    let class_keyword_args = match collect_class_keyword_args(
        vm.current_frame(),
        dst_reg,
        base_count,
        explicit_metaclass.is_some(),
        kwargc,
        kwnames_idx,
    ) {
        Ok(keyword_args) => keyword_args,
        Err(err) => return ControlFlow::Error(err),
    };
    let class_keyword_refs: SmallVec<[(&str, Value); 4]> = class_keyword_args
        .iter()
        .map(|(name, value)| (name.as_ref(), *value))
        .collect();

    let metaclass = match resolve_class_metaclass(&base_class_ids, explicit_metaclass) {
        Ok(metaclass) => metaclass,
        Err(err) => return ControlFlow::Error(err),
    };

    let prepared_namespace = match prepare_class_namespace(
        vm,
        class_name.clone(),
        &base_class_ids,
        metaclass,
        &class_keyword_refs,
    ) {
        Ok(namespace) => namespace,
        Err(err) => return ControlFlow::Error(err),
    };

    let class_body_result = match vm.execute_code_collect_locals_namespace_with_mapping(
        Arc::clone(&class_body),
        prepared_namespace,
    ) {
        Ok(result) => result,
        Err(err) => return ControlFlow::Error(err),
    };
    let namespace = class_body_result.namespace;

    let class_val = match construct_class_value(
        vm,
        class_name,
        &class_body,
        &base_class_ids,
        &namespace,
        metaclass,
        &class_keyword_refs,
        prepared_namespace,
    ) {
        Ok(value) => value,
        Err(err) => return ControlFlow::Error(err),
    };

    populate_class_cell(&class_body, class_body_result.closure.as_ref(), class_val);

    // Store result in destination register
    vm.current_frame_mut().set_reg(dst_reg, class_val);

    // For classes defined at module level, also store to globals
    // The compiler should emit a StoreGlobal after BuildClass if needed

    ControlFlow::Continue
}

#[derive(Clone, Copy)]
struct ClassMetadata {
    code_idx: u16,
    base_count: usize,
    kwargc: usize,
    kwnames_idx: u16,
}

#[inline]
fn read_class_metadata(
    vm: &mut VirtualMachine,
    inst: Instruction,
) -> Result<ClassMetadata, RuntimeError> {
    let frame = vm.current_frame_mut();
    let mut code_idx = inst.src1().0 as u16;
    let mut base_count = inst.src2().0 as usize;
    let mut ip = frame.ip as usize;

    if let Some(meta_inst) = frame.code.instructions.get(ip).copied()
        && meta_inst.opcode() == Opcode::ClassMeta as u8
    {
        code_idx = inst.imm16();
        base_count = meta_inst.dst().0 as usize;
        ip += 1;
        frame.ip = ip as u32;
    }

    let Some(ext_inst) = frame.code.instructions.get(ip).copied() else {
        return Ok(ClassMetadata {
            code_idx,
            base_count,
            kwargc: 0,
            kwnames_idx: 0,
        });
    };
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return Ok(ClassMetadata {
            code_idx,
            base_count,
            kwargc: 0,
            kwnames_idx: 0,
        });
    }

    frame.ip = (ip + 1) as u32;
    let kwargc = ext_inst.dst().0 as usize;
    let kwnames_idx = (ext_inst.src1().0 as u16) | ((ext_inst.src2().0 as u16) << 8);
    Ok(ClassMetadata {
        code_idx,
        base_count,
        kwargc,
        kwnames_idx,
    })
}

fn collect_class_keyword_args(
    frame: &crate::frame::Frame,
    dst_reg: u8,
    base_count: usize,
    has_explicit_metaclass: bool,
    kwargc: usize,
    kwnames_idx: u16,
) -> Result<SmallVec<[(Arc<str>, Value); 4]>, RuntimeError> {
    if kwargc == 0 {
        return Ok(SmallVec::new());
    }

    let kwnames_val = frame.get_const(kwnames_idx);
    let Some(kwnames_ptr) = kwnames_val.as_object_ptr() else {
        return Err(RuntimeError::internal(
            "Invalid class keyword names in constant pool",
        ));
    };
    let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };
    let keyword_base = dst_reg + 1 + base_count as u8 + u8::from(has_explicit_metaclass);

    let mut keyword_args: SmallVec<[(Arc<str>, Value); 4]> = SmallVec::with_capacity(kwargc);
    for index in 0..kwargc {
        let kw_name = kwnames
            .get(index)
            .ok_or_else(|| RuntimeError::internal("Invalid class keyword names tuple"))?;
        let kw_val = frame.get_reg(keyword_base + index as u8);
        keyword_args.push((Arc::clone(kw_name), kw_val));
    }

    Ok(keyword_args)
}

#[inline]
fn populate_class_cell(
    class_body: &Arc<CodeObject>,
    closure: Option<&Arc<crate::frame::ClosureEnv>>,
    class_value: Value,
) {
    let Some(closure) = closure else {
        return;
    };

    let Some(slot) = class_body
        .cellvars
        .iter()
        .position(|name| name.as_ref() == "__class__")
    else {
        return;
    };

    if slot < closure.len() {
        closure.set(slot, class_value);
    }
}

fn construct_class_value(
    vm: &mut VirtualMachine,
    class_name: InternedString,
    class_body: &Arc<CodeObject>,
    bases: &[ClassId],
    namespace: &ClassDict,
    metaclass: Value,
    class_keywords: &[(&str, Value)],
    prepared_namespace: Option<Value>,
) -> Result<Value, RuntimeError> {
    if class_keywords.is_empty() && should_use_native_type_new_fast_path(metaclass) {
        let result = type_new_with_metaclass(
            class_name,
            bases,
            namespace,
            metaclass,
            global_class_registry(),
        )
        .map_err(|err| RuntimeError::type_error(err.to_string()))?;
        let class_value = Value::object_ptr(Arc::as_ptr(&result.class) as *const ());
        let class_id = result.class.class_id();
        register_global_class(result.class.clone(), result.bitmap);
        if let Err(err) = invoke_descriptor_set_name_hooks(vm, class_value, namespace) {
            unregister_global_class(class_id);
            return Err(err);
        }
        if let Err(err) = invoke_init_subclass_hook(vm, class_value, &[]) {
            unregister_global_class(class_id);
            return Err(err);
        }
        return Ok(Value::object_ptr(Arc::into_raw(result.class) as *const ()));
    }

    let bases_value = alloc_heap_value(
        vm,
        TupleObject::from_vec(class_bases_to_values(bases)?),
        "class bases tuple",
    )?;
    let namespace_value = match prepared_namespace {
        // Class bodies with a prepared namespace write through the mapping
        // during execution via locals_mapping-aware store/delete opcodes.
        // Replaying the collected locals here would double-apply __setitem__
        // side effects and break metaclass protocols such as enum's namespace
        // guards for _generate_next_value_.
        Some(mapping) => mapping,
        None => alloc_heap_value(
            vm,
            class_namespace_to_dict(namespace),
            "class namespace dict",
        )?,
    };
    let class_args = [Value::string(class_name), bases_value, namespace_value];
    let class_value = if class_keywords.is_empty() {
        invoke_callable_value(vm, metaclass, &class_args)?
    } else {
        invoke_callable_value_with_keywords(vm, metaclass, &class_args, class_keywords)?
    };

    let Some(class_ptr) = class_value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "metaclass returned a non-type object",
        ));
    };
    if extract_type_id(class_ptr) != TypeId::TYPE {
        return Err(RuntimeError::type_error(format!(
            "metaclass returned non-type '{}'",
            extract_type_id(class_ptr).name()
        )));
    }

    Ok(class_value)
}

pub(crate) fn invoke_descriptor_set_name_hooks(
    vm: &mut VirtualMachine,
    class_value: Value,
    namespace: &ClassDict,
) -> Result<(), RuntimeError> {
    let set_name_attr = intern("__set_name__");
    let mut entries = SmallVec::<[(InternedString, Value); 16]>::new();
    namespace.for_each(|name, value| entries.push((name.clone(), value)));

    for (name, descriptor) in entries {
        let set_name = match get_attribute_value(vm, descriptor, &set_name_attr) {
            Ok(value) => value,
            Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => continue,
            Err(err) => return Err(err),
        };
        invoke_callable_value(vm, set_name, &[class_value, Value::string(name)])?;
    }

    Ok(())
}

pub(crate) fn invoke_init_subclass_hook(
    vm: &mut VirtualMachine,
    class_value: Value,
    class_keywords: &[(&str, Value)],
) -> Result<(), RuntimeError> {
    let Some(class_ptr) = class_value.as_object_ptr() else {
        return Ok(());
    };
    if extract_type_id(class_ptr) != TypeId::TYPE {
        return Ok(());
    }

    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    let hook_name = intern("__init_subclass__");

    for &class_id in class.mro().iter().skip(1) {
        if class_id.0 < TypeId::FIRST_USER_TYPE {
            continue;
        }

        let Some(parent) = global_class(class_id) else {
            continue;
        };
        let Some(raw_hook) = parent.get_attr(&hook_name) else {
            continue;
        };
        return invoke_init_subclass_callable(vm, class_value, raw_hook, class_keywords);
    }

    match crate::builtins::builtin_bound_type_attribute_value(
        vm,
        TypeId::OBJECT,
        class_value,
        &hook_name,
    )? {
        Some(callable) => {
            invoke_callable_value_with_keywords(vm, callable, &[], class_keywords).map(|_| ())
        }
        None => Ok(()),
    }
}

fn invoke_init_subclass_callable(
    vm: &mut VirtualMachine,
    class_value: Value,
    raw_hook: Value,
    class_keywords: &[(&str, Value)],
) -> Result<(), RuntimeError> {
    let callable = resolve_class_attribute_in_vm(vm, raw_hook, class_value)?;
    let needs_explicit_cls_arg = raw_hook
        .as_object_ptr()
        .is_some_and(|ptr| matches!(extract_type_id(ptr), TypeId::FUNCTION | TypeId::CLOSURE));

    if needs_explicit_cls_arg {
        invoke_callable_value_with_keywords(vm, callable, &[class_value], class_keywords)
            .map(|_| ())
    } else {
        invoke_callable_value_with_keywords(vm, callable, &[], class_keywords).map(|_| ())
    }
}

fn prepare_class_namespace(
    vm: &mut VirtualMachine,
    class_name: InternedString,
    bases: &[ClassId],
    metaclass: Value,
    class_keywords: &[(&str, Value)],
) -> Result<Option<Value>, RuntimeError> {
    if class_keywords.is_empty() && should_use_native_type_new_fast_path(metaclass) {
        return Ok(None);
    }

    let prepare_name = intern("__prepare__");
    let prepare = match get_attribute_value(vm, metaclass, &prepare_name) {
        Ok(value) => value,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };

    let bases_value = alloc_heap_value(
        vm,
        TupleObject::from_vec(class_bases_to_values(bases)?),
        "class bases tuple",
    )?;
    let prepare_args = [Value::string(class_name), bases_value];
    let namespace = if class_keywords.is_empty() {
        invoke_callable_value(vm, prepare, &prepare_args)?
    } else {
        invoke_callable_value_with_keywords(vm, prepare, &prepare_args, class_keywords)?
    };
    Ok(Some(namespace))
}

#[inline]
fn should_use_native_type_new_fast_path(metaclass: Value) -> bool {
    if metaclass.is_none() {
        return true;
    }

    let Some(ptr) = metaclass.as_object_ptr() else {
        return false;
    };

    builtin_type_object_type_id(ptr) == Some(TypeId::TYPE)
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

fn class_bases_to_values(bases: &[ClassId]) -> Result<Vec<Value>, RuntimeError> {
    bases.iter().copied().map(class_id_to_value).collect()
}

fn class_id_to_value(class_id: ClassId) -> Result<Value, RuntimeError> {
    if class_id == ClassId::OBJECT {
        return Ok(builtin_type_object_for_type_id(TypeId::OBJECT));
    }

    if let Some(value) = crate::builtins::exception_type_value_for_proxy_class_id(class_id) {
        return Ok(value);
    }

    if class_id.0 < TypeId::FIRST_USER_TYPE {
        return Ok(builtin_type_object_for_type_id(TypeId::from_raw(
            class_id.0,
        )));
    }

    global_class(class_id)
        .map(|class| Value::object_ptr(Arc::as_ptr(&class) as *const ()))
        .ok_or_else(|| RuntimeError::internal("class base missing from global registry"))
}

fn class_namespace_to_dict(namespace: &ClassDict) -> DictObject {
    let mut dict = DictObject::new();
    namespace.for_each(|name, value| {
        dict.set(Value::string(name.clone()), value);
    });
    dict
}

#[inline]
fn resolve_class_metaclass(
    base_class_ids: &[ClassId],
    explicit_metaclass: Option<Value>,
) -> Result<Value, RuntimeError> {
    let mut winner = validate_metaclass_value(explicit_metaclass.unwrap_or(Value::none()))?;

    for &base_id in base_class_ids {
        let base_metaclass = base_class_metaclass_value(base_id);
        winner = choose_more_derived_metaclass(winner, base_metaclass)?;
    }

    Ok(winner)
}

#[inline]
fn validate_metaclass_value(value: Value) -> Result<Value, RuntimeError> {
    if value.is_none() {
        return Ok(Value::none());
    }

    let Some(ptr) = value.as_object_ptr() else {
        return Err(RuntimeError::type_error(
            "metaclass must be a class object derived from type",
        ));
    };

    if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr) {
        return if represented == TypeId::TYPE {
            Ok(Value::none())
        } else {
            Err(RuntimeError::type_error(
                "metaclass must be a class object derived from type",
            ))
        };
    }

    if extract_type_id(ptr) != TypeId::TYPE {
        return Err(RuntimeError::type_error(
            "metaclass must be a class object derived from type",
        ));
    }

    let class = unsafe { &*(ptr as *const PyClassObject) };
    if !class
        .flags()
        .contains(prism_runtime::object::class::ClassFlags::METACLASS)
    {
        return Err(RuntimeError::type_error(
            "metaclass must be a class object derived from type",
        ));
    }

    Ok(value)
}

#[inline]
fn base_class_metaclass_value(base_id: ClassId) -> Value {
    if base_id.0 < TypeId::FIRST_USER_TYPE {
        return Value::none();
    }

    global_class(base_id)
        .map(|class| class.metaclass())
        .unwrap_or_else(Value::none)
}

#[inline]
fn choose_more_derived_metaclass(current: Value, candidate: Value) -> Result<Value, RuntimeError> {
    if metaclass_is_subclass(candidate, current) {
        return Ok(candidate);
    }
    if metaclass_is_subclass(current, candidate) {
        return Ok(current);
    }

    Err(RuntimeError::type_error(
        "metaclass conflict: metaclass hierarchy is incompatible",
    ))
}

#[inline]
fn metaclass_is_subclass(candidate: Value, target: Value) -> bool {
    if target.is_none() {
        return true;
    }
    if candidate.is_none() {
        return target.is_none();
    }
    if candidate == target {
        return true;
    }

    let Some(candidate_class) = class_object_from_value(candidate) else {
        return false;
    };
    let Some(target_class) = class_object_from_value(target) else {
        return false;
    };

    candidate_class
        .mro()
        .iter()
        .any(|&class_id| class_id == target_class.class_id())
}

#[inline]
fn class_object_from_value(value: Value) -> Option<&'static PyClassObject> {
    let ptr = value.as_object_ptr()?;
    if crate::builtins::builtin_type_object_type_id(ptr).is_some() {
        return None;
    }
    (extract_type_id(ptr) == TypeId::TYPE).then(|| unsafe { &*(ptr as *const PyClassObject) })
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract a string name from a Value.
///
/// Supports both interned strings (small) and heap-allocated strings.
#[inline]
#[cfg(test)]
fn extract_string_name(val: Value) -> Option<InternedString> {
    if val.is_string() {
        let ptr = val.as_string_object_ptr()?;
        return interned_by_ptr(ptr as *const u8);
    }

    let ptr = val.as_object_ptr()?;
    if extract_type_id(ptr) != TypeId::STR {
        return None;
    }

    let string = unsafe { &*(ptr as *const StringObject) };
    Some(intern(string.as_str()))
}

/// Resolve class name from a code-object constant by pointer identity against
/// the frame's nested code-object list.
#[inline]
fn extract_class_name_from_code_const(
    code_const: Value,
    nested_code_objects: &[Arc<CodeObject>],
) -> Option<InternedString> {
    extract_class_body_from_code_const(code_const, nested_code_objects)
        .map(|nested| intern(nested.name.as_ref()))
}

/// Resolve the class-body code object from a constant by pointer identity
/// against the enclosing frame's nested code-object list.
#[inline]
fn extract_class_body_from_code_const(
    code_const: Value,
    nested_code_objects: &[Arc<CodeObject>],
) -> Option<Arc<CodeObject>> {
    let code_ptr = code_const.as_object_ptr()? as *const CodeObject;
    nested_code_objects
        .iter()
        .find(|nested| Arc::as_ptr(nested) == code_ptr)
        .cloned()
}

/// Extract ClassId from a class object Value.
#[inline]
fn extract_class_id(val: Value) -> Option<ClassId> {
    if let Some(ptr) = val.as_object_ptr() {
        // Check if this is a PyClassObject
        let type_id = extract_type_id(ptr);
        if type_id == TypeId::TYPE {
            if let Some(builtin_type_id) = crate::builtins::builtin_type_object_type_id(ptr) {
                return Some(ClassId(builtin_type_id.raw()));
            }

            // This is a class object - extract its class_id
            // SAFETY: We verified type_id is TYPE, so ptr points to PyClassObject
            let class_obj = unsafe { &*(ptr as *const PyClassObject) };
            return Some(class_obj.class_id());
        }
        if type_id == TypeId::EXCEPTION_TYPE {
            return crate::builtins::exception_proxy_class_id_from_ptr(ptr);
        }
    }
    None
}

/// Extract TypeId from an object pointer.
///
/// # Safety
/// Pointer must point to a valid object with ObjectHeader at offset 0.
#[inline(always)]
fn extract_type_id(ptr: *const ()) -> TypeId {
    use prism_runtime::object::ObjectHeader;
    let header_ptr = ptr as *const ObjectHeader;
    // SAFETY: Caller guarantees ptr points to valid object with ObjectHeader at offset 0
    unsafe { (*header_ptr).type_id }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtins::builtin_type;
    use prism_core::intern::intern;
    use prism_runtime::object::class::PyClassObject;

    // =========================================================================
    // Unit Tests for Helper Functions
    // =========================================================================

    #[test]
    fn test_class_object_has_type_id() {
        use prism_runtime::object::PyObject;
        use prism_runtime::object::type_obj::TypeId;

        // Create a class object and verify its header has TYPE type_id via the trait
        let class = PyClassObject::new_simple(intern("TestClass"));

        // Use the PyObject trait to get the type_id (this is the safe way)
        assert_eq!(
            class.header().type_id,
            TypeId::TYPE,
            "PyClassObject header should have TypeId::TYPE"
        );
    }

    #[test]
    fn test_class_id_extraction_direct() {
        // Create a class and verify we can get its class_id through the object
        let class = PyClassObject::new_simple(intern("DirectTestClass"));
        let expected_id = class.class_id();
        let class = Arc::new(class);
        let class_ptr = Arc::as_ptr(&class) as *const PyClassObject;

        // Directly read from the object
        let class_ref = unsafe { &*class_ptr };
        assert_eq!(class_ref.class_id(), expected_id);
    }

    #[test]
    fn test_extract_class_id_returns_none_for_non_objects() {
        // Non-object values should return None
        assert_eq!(extract_class_id(Value::none()), None);
        assert_eq!(extract_class_id(Value::bool(true)), None);
        assert_eq!(extract_class_id(Value::int_unchecked(42)), None);
    }

    #[test]
    fn test_extract_class_id_supports_builtin_type_objects() {
        let int_type =
            builtin_type(&[Value::int(0).unwrap()]).expect("type(int_instance) should succeed");

        assert_eq!(extract_class_id(int_type), Some(ClassId(TypeId::INT.raw())));
    }

    #[test]
    fn test_extract_string_name_tagged() {
        let name = extract_string_name(Value::string(intern("TaggedClass")));
        assert_eq!(name.unwrap().as_ref(), "TaggedClass");
    }

    #[test]
    fn test_extract_string_name_heap_string() {
        let ptr = Box::into_raw(Box::new(StringObject::new("HeapClass")));
        let value = Value::object_ptr(ptr as *const ());
        let name = extract_string_name(value);
        assert_eq!(name.unwrap().as_ref(), "HeapClass");
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }

    #[test]
    fn test_extract_string_name_invalid_returns_none() {
        let name = extract_string_name(Value::none());
        assert!(name.is_none());
    }

    #[test]
    fn test_extract_class_name_from_code_const() {
        let code = Arc::new(CodeObject::new("ExtractedClass", "<test>"));
        let raw = Arc::into_raw(Arc::clone(&code)) as *const ();
        let code_const = Value::object_ptr(raw);

        let name = extract_class_name_from_code_const(code_const, &[Arc::clone(&code)]);
        assert_eq!(name.unwrap().as_ref(), "ExtractedClass");

        unsafe {
            let _ = Arc::from_raw(raw as *const CodeObject);
        }
    }

    #[test]
    fn test_extract_class_name_from_code_const_returns_none_when_pointer_not_nested() {
        let code = Arc::new(CodeObject::new("UnlistedClass", "<test>"));
        let raw = Arc::into_raw(Arc::clone(&code)) as *const ();
        let code_const = Value::object_ptr(raw);

        let name = extract_class_name_from_code_const(code_const, &[]);
        assert!(name.is_none());

        unsafe {
            let _ = Arc::from_raw(raw as *const CodeObject);
        }
    }

    // =========================================================================
    // Class Creation Tests
    // =========================================================================

    #[test]
    fn test_simple_class_creation() {
        let class = PyClassObject::new_simple(intern("SimpleClass"));
        assert_eq!(class.name().as_ref(), "SimpleClass");
        assert!(class.bases().is_empty());
        assert_eq!(class.mro().len(), 2); // [SimpleClass, object]
    }

    #[test]
    fn test_class_with_single_base() {
        // Create parent class
        let parent = Arc::new(PyClassObject::new_simple(intern("Parent")));
        let parent_id = parent.class_id();
        let parent_mro = parent.mro().to_vec();

        // Create child with parent as base
        let child = PyClassObject::new(intern("Child"), &[parent_id], |id| {
            if id == parent_id {
                Some(parent_mro.clone().into())
            } else {
                None
            }
        });

        assert!(child.is_ok());
        let child = child.unwrap();
        assert_eq!(child.bases().len(), 1);
        assert!(child.bases().contains(&parent_id));
    }

    #[test]
    fn test_class_type_id_uniqueness() {
        let class1 = PyClassObject::new_simple(intern("Class1"));
        let class2 = PyClassObject::new_simple(intern("Class2"));

        assert_ne!(class1.class_type_id(), class2.class_type_id());
        assert_ne!(class1.class_id(), class2.class_id());
    }

    #[test]
    fn test_class_dict_operations() {
        let namespace = ClassDict::new();

        // Initially empty
        assert!(namespace.is_empty());

        // Add attribute
        let attr_name = intern("my_method");
        namespace.set(attr_name.clone(), Value::int_unchecked(42));

        // Check attribute
        assert!(!namespace.is_empty());
        assert!(namespace.contains(&attr_name));
        assert_eq!(namespace.get(&attr_name), Some(Value::int_unchecked(42)));

        // Delete attribute
        let deleted = namespace.delete(&attr_name);
        assert_eq!(deleted, Some(Value::int_unchecked(42)));
        assert!(namespace.is_empty());
    }

    #[test]
    fn test_class_dict_multiple_attributes() {
        let namespace = ClassDict::new();

        // Add multiple attributes
        for i in 0..10 {
            let name = intern(&format!("attr_{}", i));
            namespace.set(name, Value::int_unchecked(i as i64));
        }

        assert_eq!(namespace.len(), 10);

        // Verify all attributes
        for i in 0..10 {
            let name = intern(&format!("attr_{}", i));
            assert!(namespace.contains(&name));
            assert_eq!(namespace.get(&name), Some(Value::int_unchecked(i as i64)));
        }
    }

    #[test]
    fn test_class_inherits_from_object() {
        let class = PyClassObject::new_simple(intern("Derived"));

        // MRO should end with object (ClassId::OBJECT)
        let mro = class.mro();
        assert_eq!(mro.len(), 2);
        assert_eq!(mro[0], class.class_id()); // Self first
        assert_eq!(mro[1], ClassId::OBJECT); // Object last
    }

    #[test]
    fn test_class_attribute_setting() {
        let class = PyClassObject::new_simple(intern("AttrTest"));

        // Set some attributes
        class.set_attr(intern("x"), Value::int_unchecked(10));
        class.set_attr(intern("y"), Value::int_unchecked(20));

        // Verify
        assert!(class.has_attr(&intern("x")));
        assert!(class.has_attr(&intern("y")));
        assert_eq!(class.get_attr(&intern("x")), Some(Value::int_unchecked(10)));
        assert_eq!(class.get_attr(&intern("y")), Some(Value::int_unchecked(20)));
    }

    #[test]
    fn test_class_flags_modification() {
        let mut class = PyClassObject::new_simple(intern("FlagsModTest"));

        // Modify flags
        class.mark_initialized();
        class.mark_final();
        class.mark_has_init();

        assert!(class.is_initialized());
        assert!(class.is_final());
        assert!(class.has_custom_init());
    }

    #[test]
    fn test_class_slots_definition() {
        let mut class = PyClassObject::new_simple(intern("SlottedClass"));

        // Define __slots__
        let slots = vec![intern("x"), intern("y"), intern("z")];
        class.set_slots(slots);

        assert!(class.has_slots());
        assert_eq!(class.slot_names().unwrap().len(), 3);
    }

    #[test]
    fn test_instantiation_hint_no_slots() {
        let class = PyClassObject::new_simple(intern("NoSlots"));

        // Without __init__, hint is DefaultInit
        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::DefaultInit);
    }

    #[test]
    fn test_instantiation_hint_with_init() {
        let mut class = PyClassObject::new_simple(intern("WithInit"));
        class.mark_has_init();

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::Generic);
    }

    #[test]
    fn test_instantiation_hint_inline_slots() {
        let mut class = PyClassObject::new_simple(intern("InlineSlots"));
        class.set_slots(vec![intern("x"), intern("y")]); // 2 slots, fits inline

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::InlineSlots);
    }

    #[test]
    fn test_instantiation_hint_fixed_slots() {
        let mut class = PyClassObject::new_simple(intern("FixedSlots"));
        // More than 4 slots - needs fixed allocation
        class.set_slots(vec![
            intern("a"),
            intern("b"),
            intern("c"),
            intern("d"),
            intern("e"),
            intern("f"),
        ]);

        use prism_runtime::object::class::InstantiationHint;
        assert_eq!(class.instantiation_hint(), InstantiationHint::FixedSlots);
    }

    // =========================================================================
    // Multiple Inheritance Tests
    // =========================================================================

    #[test]
    fn test_diamond_inheritance_mro() {
        use std::collections::HashMap;

        // Diamond: D(B, C) where B(A) and C(A)
        // Create A
        let a = Arc::new(PyClassObject::new_simple(intern("A")));
        let a_id = a.class_id();
        let a_mro = a.mro().to_vec();

        let mut registry: HashMap<ClassId, Vec<ClassId>> = HashMap::new();
        registry.insert(a_id, a_mro.clone());

        // Create B(A)
        let b = Arc::new(
            PyClassObject::new(intern("B"), &[a_id], |id| {
                registry.get(&id).cloned().map(|v| v.into())
            })
            .unwrap(),
        );
        let b_id = b.class_id();
        let b_mro = b.mro().to_vec();
        registry.insert(b_id, b_mro.clone());

        // Create C(A)
        let c = Arc::new(
            PyClassObject::new(intern("C"), &[a_id], |id| {
                registry.get(&id).cloned().map(|v| v.into())
            })
            .unwrap(),
        );
        let c_id = c.class_id();
        let c_mro = c.mro().to_vec();
        registry.insert(c_id, c_mro.clone());

        // Create D(B, C)
        let d = PyClassObject::new(intern("D"), &[b_id, c_id], |id| {
            registry.get(&id).cloned().map(|v| v.into())
        })
        .unwrap();

        // D's MRO should be [D, B, C, A, object]
        let d_mro = d.mro();
        assert_eq!(d_mro.len(), 5);
        assert_eq!(d_mro[0], d.class_id()); // D
        assert_eq!(d_mro[1], b_id); // B
        assert_eq!(d_mro[2], c_id); // C
        assert_eq!(d_mro[3], a_id); // A
        assert_eq!(d_mro[4], ClassId::OBJECT); // object
    }

    // =========================================================================
    // Class Value Conversion Tests
    // =========================================================================

    #[test]
    fn test_class_to_value_roundtrip() {
        let original = PyClassObject::new_simple(intern("Roundtrip"));
        let original_id = original.class_type_id();
        let original_name = original.name().clone();

        // Convert to Arc and then to Value
        let class = Arc::new(original);
        let class_ptr = Arc::into_raw(class) as *const ();
        let class_val = Value::object_ptr(class_ptr);

        // Should be able to check it's an object
        assert!(class_val.as_object_ptr().is_some());

        // Extract and verify
        let extracted_ptr = class_val.as_object_ptr().unwrap();
        let extracted_class = unsafe { &*(extracted_ptr as *const PyClassObject) };

        assert_eq!(extracted_class.class_type_id(), original_id);
        assert_eq!(extracted_class.name(), &original_name);

        // Clean up
        unsafe { Arc::from_raw(extracted_ptr as *const PyClassObject) };
    }

    // =========================================================================
    // Thread Safety Tests
    // =========================================================================

    #[test]
    fn test_class_dict_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let namespace = Arc::new(ClassDict::new());

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let ns = namespace.clone();
                thread::spawn(move || {
                    // Each thread sets its own attribute
                    let name = intern(&format!("thread_attr_{}", i));
                    ns.set(name.clone(), Value::int_unchecked(i as i64));

                    // Verify it was set
                    assert!(ns.contains(&name));
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        // All 4 attributes should exist
        assert_eq!(namespace.len(), 4);
    }

    #[test]
    fn test_class_object_concurrent_reads() {
        use std::sync::Arc;
        use std::thread;

        // Create class with some attributes
        let class = PyClassObject::new_simple(intern("ConcurrentRead"));
        class.set_attr(intern("x"), Value::int_unchecked(100));
        class.set_attr(intern("y"), Value::int_unchecked(200));
        let class = Arc::new(class);

        // Multiple threads reading simultaneously
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let c = class.clone();
                thread::spawn(move || {
                    for _ in 0..100 {
                        let x = c.get_attr(&intern("x"));
                        let y = c.get_attr(&intern("y"));
                        assert_eq!(x, Some(Value::int_unchecked(100)));
                        assert_eq!(y, Some(Value::int_unchecked(200)));
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }
    }
}
