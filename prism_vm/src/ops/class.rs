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
use crate::ops::iteration::collect_iterable_values;
use crate::ops::objects::{get_attribute_value, resolve_class_attribute_in_vm};
use crate::ops::unpack::{keyword_key_to_name, mapping_entries_for_unpack};
use prism_code::{
    CLASS_META_DYNAMIC_BASES_FLAG, CLASS_META_DYNAMIC_KEYWORDS_FLAG, CodeObject, Instruction,
    Opcode,
};
use prism_core::Value;
use prism_core::intern::{InternedString, intern};
use prism_runtime::object::class::{ClassDict, PyClassObject};
use prism_runtime::object::mro::ClassId;
use prism_runtime::object::type_builtins::{
    global_class, global_class_registry, register_global_class, type_new_with_metaclass,
    unregister_global_class,
};
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::tuple::{TupleObject, value_as_tuple_ref};
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

    let base_values = match collect_class_base_values(vm, dst_reg, metadata) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };
    let class_keyword_args =
        match collect_class_keyword_args(vm, dst_reg, metadata, explicit_metaclass.is_some()) {
            Ok(keyword_args) => keyword_args,
            Err(err) => return ControlFlow::Error(err),
        };
    let (explicit_metaclass, class_keyword_args) =
        match split_metaclass_keyword(explicit_metaclass, class_keyword_args) {
            Ok(parts) => parts,
            Err(err) => return ControlFlow::Error(err),
        };
    let (metaclass, base_class_ids) =
        match resolve_class_metaclass(&base_values, explicit_metaclass) {
            Ok(resolved) => resolved,
            Err(err) => return ControlFlow::Error(err),
        };
    let class_keyword_refs: SmallVec<[(&str, Value); 4]> = class_keyword_args
        .iter()
        .map(|(name, value)| (name.as_ref(), *value))
        .collect();

    let prepared_namespace = match prepare_class_namespace(
        vm,
        class_name.clone(),
        &base_values,
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
        &base_values,
        base_class_ids.as_deref(),
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
    flags: u8,
    dynamic_bases_reg: Option<u8>,
    kwargc: usize,
    kwnames_idx: u16,
    kwargs_dict_reg: Option<u8>,
}

#[inline]
fn read_class_metadata(
    vm: &mut VirtualMachine,
    inst: Instruction,
) -> Result<ClassMetadata, RuntimeError> {
    let frame = vm.current_frame_mut();
    let mut metadata = ClassMetadata {
        code_idx: inst.src1().0 as u16,
        base_count: inst.src2().0 as usize,
        flags: 0,
        dynamic_bases_reg: None,
        kwargc: 0,
        kwnames_idx: 0,
        kwargs_dict_reg: None,
    };
    let mut ip = frame.ip as usize;

    if let Some(meta_inst) = frame.code.instructions.get(ip).copied()
        && meta_inst.opcode() == Opcode::ClassMeta as u8
    {
        metadata.code_idx = inst.imm16();
        metadata.base_count = meta_inst.dst().0 as usize;
        metadata.flags = meta_inst.src2().0;
        if (metadata.flags & CLASS_META_DYNAMIC_BASES_FLAG) != 0 {
            metadata.dynamic_bases_reg = Some(meta_inst.src1().0);
        }
        ip += 1;
        frame.ip = ip as u32;
    }

    let Some(ext_inst) = frame.code.instructions.get(ip).copied() else {
        return Ok(metadata);
    };
    if ext_inst.opcode() != Opcode::CallKwEx as u8 {
        return Ok(metadata);
    }

    frame.ip = (ip + 1) as u32;
    if metadata_has_dynamic_keywords(metadata) {
        metadata.kwargs_dict_reg = Some(ext_inst.dst().0);
    } else {
        metadata.kwargc = ext_inst.dst().0 as usize;
        metadata.kwnames_idx = (ext_inst.src1().0 as u16) | ((ext_inst.src2().0 as u16) << 8);
    }
    Ok(metadata)
}

#[inline]
fn metadata_has_dynamic_keywords(metadata: ClassMetadata) -> bool {
    // Dynamic keyword metadata is encoded in ClassMeta.src2. If no ClassMeta was
    // present, this remains false for legacy bytecode.
    (metadata.flags & CLASS_META_DYNAMIC_KEYWORDS_FLAG) != 0
}

fn collect_class_base_values(
    vm: &mut VirtualMachine,
    dst_reg: u8,
    metadata: ClassMetadata,
) -> Result<SmallVec<[Value; 8]>, RuntimeError> {
    if let Some(bases_reg) = metadata.dynamic_bases_reg {
        let bases_value = vm.current_frame().get_reg(bases_reg);
        if let Some(tuple) = value_as_tuple_ref(bases_value) {
            return Ok(tuple.as_slice().iter().copied().collect());
        }
        return collect_iterable_values(vm, bases_value).map(|values| values.into_iter().collect());
    }

    let frame = vm.current_frame();
    let mut bases = SmallVec::with_capacity(metadata.base_count);
    for index in 0..metadata.base_count {
        bases.push(frame.get_reg(dst_reg + 1 + index as u8));
    }
    Ok(bases)
}

fn collect_class_keyword_args(
    vm: &mut VirtualMachine,
    dst_reg: u8,
    metadata: ClassMetadata,
    has_explicit_metaclass: bool,
) -> Result<SmallVec<[(Arc<str>, Value); 4]>, RuntimeError> {
    if let Some(kwargs_reg) = metadata.kwargs_dict_reg {
        let mapping = vm.current_frame().get_reg(kwargs_reg);
        if mapping.is_none() {
            return Ok(SmallVec::new());
        }
        let entries = mapping_entries_for_unpack(vm, mapping)?;
        let mut keyword_args: SmallVec<[(Arc<str>, Value); 4]> =
            SmallVec::with_capacity(entries.len());
        for (key, value) in entries {
            keyword_args.push((keyword_key_to_name(key)?, value));
        }
        return Ok(keyword_args);
    }

    if metadata.kwargc == 0 {
        return Ok(SmallVec::new());
    }

    let frame = vm.current_frame();
    let kwnames_idx = metadata.kwnames_idx;
    let kwnames_val = frame.get_const(kwnames_idx);
    let Some(kwnames_ptr) = kwnames_val.as_object_ptr() else {
        return Err(RuntimeError::internal(
            "Invalid class keyword names in constant pool",
        ));
    };
    let kwnames = unsafe { &*(kwnames_ptr as *const prism_code::KwNamesTuple) };
    let keyword_base = dst_reg + 1 + metadata.base_count as u8 + u8::from(has_explicit_metaclass);

    let mut keyword_args: SmallVec<[(Arc<str>, Value); 4]> =
        SmallVec::with_capacity(metadata.kwargc);
    for index in 0..metadata.kwargc {
        let kw_name = kwnames
            .get(index)
            .ok_or_else(|| RuntimeError::internal("Invalid class keyword names tuple"))?;
        let kw_val = frame.get_reg(keyword_base + index as u8);
        keyword_args.push((Arc::clone(kw_name), kw_val));
    }

    Ok(keyword_args)
}

fn split_metaclass_keyword(
    mut explicit_metaclass: Option<Value>,
    keyword_args: SmallVec<[(Arc<str>, Value); 4]>,
) -> Result<(Option<Value>, SmallVec<[(Arc<str>, Value); 4]>), RuntimeError> {
    let mut class_keywords = SmallVec::with_capacity(keyword_args.len());

    for (name, value) in keyword_args {
        if name.as_ref() == "metaclass" {
            if explicit_metaclass.is_some() {
                return Err(RuntimeError::type_error(
                    "class got multiple values for keyword argument 'metaclass'",
                ));
            }
            explicit_metaclass = Some(value);
        } else {
            class_keywords.push((name, value));
        }
    }

    Ok((explicit_metaclass, class_keywords))
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
    bases: &[Value],
    base_class_ids: Option<&[ClassId]>,
    namespace: &ClassDict,
    metaclass: Value,
    class_keywords: &[(&str, Value)],
    prepared_namespace: Option<Value>,
) -> Result<Value, RuntimeError> {
    if class_keywords.is_empty() && should_use_native_type_new_fast_path(metaclass) {
        let collected_base_class_ids;
        let base_class_ids = match base_class_ids {
            Some(ids) => ids,
            None => {
                collected_base_class_ids = collect_base_class_ids(bases)?;
                &collected_base_class_ids
            }
        };
        let result = type_new_with_metaclass(
            class_name,
            base_class_ids,
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
        vm.record_published_class(class_id);
        return Ok(Value::object_ptr(Arc::into_raw(result.class) as *const ()));
    }

    let bases_value = alloc_heap_value(vm, TupleObject::from_slice(bases), "class bases tuple")?;
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
    let metaclass_callable = materialize_metaclass_callable(metaclass);
    let class_value = if class_keywords.is_empty() {
        invoke_callable_value(vm, metaclass_callable, &class_args)?
    } else {
        invoke_callable_value_with_keywords(vm, metaclass_callable, &class_args, class_keywords)?
    };

    if metaclass_result_must_be_type(metaclass) {
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
    bases: &[Value],
    metaclass: Value,
    class_keywords: &[(&str, Value)],
) -> Result<Option<Value>, RuntimeError> {
    if class_keywords.is_empty() && should_use_native_type_new_fast_path(metaclass) {
        return Ok(None);
    }

    let prepare_name = intern("__prepare__");
    let metaclass_callable = materialize_metaclass_callable(metaclass);
    let prepare = match get_attribute_value(vm, metaclass_callable, &prepare_name) {
        Ok(value) => value,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Ok(None);
        }
        Err(err) => return Err(err),
    };

    let bases_value = alloc_heap_value(vm, TupleObject::from_slice(bases), "class bases tuple")?;
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

fn class_namespace_to_dict(namespace: &ClassDict) -> DictObject {
    let mut dict = DictObject::new();
    namespace.for_each(|name, value| {
        dict.set(Value::string(name.clone()), value);
    });
    dict
}

#[inline]
fn resolve_class_metaclass(
    base_values: &[Value],
    explicit_metaclass: Option<Value>,
) -> Result<(Value, Option<Vec<ClassId>>), RuntimeError> {
    let explicit_metaclass = explicit_metaclass.map(normalize_explicit_metaclass);
    if let Some(metaclass) = explicit_metaclass
        && !metaclass_participates_in_conflict_resolution(metaclass)
    {
        return Ok((metaclass, None));
    }

    let base_class_ids = collect_base_class_ids(base_values)?;
    let mut winner = explicit_metaclass.unwrap_or(Value::none());

    for &base_id in base_class_ids.iter() {
        let base_metaclass = base_class_metaclass_value(base_id);
        winner = choose_more_derived_metaclass(winner, base_metaclass)?;
    }

    Ok((winner, Some(base_class_ids)))
}

#[inline]
fn normalize_explicit_metaclass(value: Value) -> Value {
    if value.is_none() {
        return Value::none();
    }

    let Some(ptr) = value.as_object_ptr() else {
        return value;
    };

    if let Some(represented) = crate::builtins::builtin_type_object_type_id(ptr) {
        if represented == TypeId::TYPE {
            Value::none()
        } else {
            value
        }
    } else {
        value
    }
}

fn collect_base_class_ids(bases: &[Value]) -> Result<Vec<ClassId>, RuntimeError> {
    let mut base_class_ids = Vec::with_capacity(bases.len());
    for (index, &base) in bases.iter().enumerate() {
        match extract_class_id(base) {
            Some(class_id) => base_class_ids.push(class_id),
            None => {
                return Err(RuntimeError::type_error(format!(
                    "base class {} is not a valid class",
                    index
                )));
            }
        }
    }
    Ok(base_class_ids)
}

#[inline]
fn metaclass_participates_in_conflict_resolution(value: Value) -> bool {
    if value.is_none() {
        return true;
    }

    let Some(ptr) = value.as_object_ptr() else {
        return false;
    };
    if crate::builtins::builtin_type_object_type_id(ptr) == Some(TypeId::TYPE) {
        return true;
    }
    if extract_type_id(ptr) != TypeId::TYPE {
        return false;
    }
    let class = unsafe { &*(ptr as *const PyClassObject) };
    class
        .flags()
        .contains(prism_runtime::object::class::ClassFlags::METACLASS)
}

#[inline]
fn materialize_metaclass_callable(metaclass: Value) -> Value {
    if metaclass.is_none() {
        builtin_type_object_for_type_id(TypeId::TYPE)
    } else {
        metaclass
    }
}

#[inline]
fn metaclass_result_must_be_type(metaclass: Value) -> bool {
    metaclass_participates_in_conflict_resolution(metaclass)
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
