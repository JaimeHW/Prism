//! Load and store opcode handlers.
//!
//! Handles loading constants, locals, globals, closures, and register moves.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::objects::{alloc_heap_value, extract_type_id, get_attribute_value};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_code::{CodeFlags, Instruction};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::dict::DictObject;
use std::sync::Arc;

// =============================================================================
// Constants
// =============================================================================

/// LoadConst: dst = constants[imm16]
#[inline(always)]
pub fn load_const(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_const(inst.imm16());
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}

/// LoadNone: dst = None
#[inline(always)]
pub fn load_none(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut().set_reg(inst.dst().0, Value::none());
    ControlFlow::Continue
}

/// LoadTrue: dst = True
#[inline(always)]
pub fn load_true(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(true));
    ControlFlow::Continue
}

/// LoadFalse: dst = False
#[inline(always)]
pub fn load_false(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::bool(false));
    ControlFlow::Continue
}

/// SetupAnnotations: ensure the active module/class namespace has __annotations__.
#[inline]
pub fn setup_annotations(vm: &mut VirtualMachine, _inst: Instruction) -> ControlFlow {
    match ensure_annotations_namespace(vm) {
        Ok(()) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

// =============================================================================
// Locals
// =============================================================================

/// LoadLocal: dst = frame.registers[imm16]
/// Note: In our register-based VM, locals ARE registers. This opcode
/// may be used for explicit local variable access semantics.
#[inline(always)]
pub fn load_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    let dst = inst.dst().0;

    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let name = mapped_local_name(vm.current_frame(), slot);
        return match load_mapped_local(vm, mapping, &name) {
            Ok(value) => {
                vm.current_frame_mut().set_reg(dst, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    {
        let frame = vm.current_frame();
        if frame.code.flags.contains(CodeFlags::CLASS) && !frame.reg_is_written(slot) {
            let name = mapped_local_name(frame, slot);
            return match load_module_or_builtin(vm, &name) {
                Ok(value) => {
                    vm.current_frame_mut().set_reg(dst, value);
                    ControlFlow::Continue
                }
                Err(err) => ControlFlow::Error(err),
            };
        }
    }

    let frame = vm.current_frame();
    if !frame.reg_is_written(slot) {
        return ControlFlow::Error(RuntimeError::unbound_local(mapped_local_name(frame, slot)));
    }

    let value = frame.get_reg(slot);
    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// StoreLocal: frame.registers[imm16] = dst-register
#[inline(always)]
pub fn store_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let (value, name) = {
            let frame = vm.current_frame();
            (frame.get_reg(inst.dst().0), mapped_local_name(frame, slot))
        };
        return match store_mapped_local(vm, mapping, &name, value) {
            Ok(()) => {
                vm.current_frame_mut().set_reg(slot, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    let frame = vm.current_frame_mut();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the destination slot index.
    let value = frame.get_reg(inst.dst().0);
    frame.set_reg(slot, value);
    ControlFlow::Continue
}

/// DeleteLocal: frame.registers[imm16] = undefined
#[inline(always)]
pub fn delete_local(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let slot = inst.imm16() as u8;
    if let Some(mapping) = vm.current_frame().locals_mapping() {
        let name = mapped_local_name(vm.current_frame(), slot);
        return match delete_mapped_local(vm, mapping, &name) {
            Ok(true) => {
                vm.current_frame_mut().clear_reg(slot);
                crate::stdlib::_weakref::clear_unreachable_weakrefs_if_registered(vm);
                ControlFlow::Continue
            }
            Ok(false) => ControlFlow::Error(RuntimeError::name_error(name)),
            Err(err) => ControlFlow::Error(err),
        };
    }

    let frame = vm.current_frame_mut();
    frame.clear_reg(slot);
    crate::stdlib::_weakref::clear_unreachable_weakrefs_if_registered(vm);
    ControlFlow::Continue
}

#[inline]
fn mapped_local_name(frame: &crate::frame::Frame, slot: u8) -> Arc<str> {
    Arc::clone(frame.get_local_name(slot as u16))
}

#[inline]
fn mapped_local_key(name: &Arc<str>) -> Value {
    Value::string(intern(name.as_ref()))
}

fn ensure_annotations_namespace(vm: &mut VirtualMachine) -> Result<(), RuntimeError> {
    let name = Arc::<str>::from("__annotations__");

    if let Some(mapping) = vm.current_frame().locals_mapping() {
        if lookup_mapped_local(vm, mapping, &name)?.is_none() {
            let value = alloc_heap_value(vm, DictObject::new(), "__annotations__ dict")?;
            store_mapped_local(vm, mapping, &name, value)?;
        }
        return Ok(());
    }

    let class_local_slot = {
        let frame = vm.current_frame();
        if frame.code.flags.contains(CodeFlags::CLASS) {
            frame
                .code
                .locals
                .iter()
                .position(|local| local.as_ref() == "__annotations__")
                .map(|slot| slot as u8)
        } else {
            None
        }
    };

    if let Some(slot) = class_local_slot {
        if !vm.current_frame().reg_is_written(slot) {
            let value = alloc_heap_value(vm, DictObject::new(), "__annotations__ dict")?;
            vm.current_frame_mut().set_reg(slot, value);
        }
        return Ok(());
    }

    if vm.module_scope_value(&name).is_none() {
        let value = alloc_heap_value(vm, DictObject::new(), "__annotations__ dict")?;
        vm.set_module_scope_value(name, value);
    }

    Ok(())
}

fn load_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<Value, RuntimeError> {
    if let Some(value) = lookup_mapped_local(vm, mapping, name)? {
        return Ok(value);
    }

    load_module_or_builtin(vm, name)
}

fn load_module_or_builtin(vm: &mut VirtualMachine, name: &Arc<str>) -> Result<Value, RuntimeError> {
    if let Some(value) = vm.module_scope_value(name) {
        return Ok(value);
    }

    if let Some(value) = vm.builtins.get(name) {
        return Ok(value);
    }

    Err(RuntimeError::name_error(Arc::clone(name)))
}

fn lookup_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<Option<Value>, RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &*(ptr as *const DictObject) };
        return Ok(dict.get(key));
    }

    let getitem_name = intern("__getitem__");
    let getitem = get_attribute_value(vm, mapping, &getitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __getitem__")
        } else {
            err
        }
    })?;

    match invoke_callable_value(vm, getitem, &[key]) {
        Ok(value) => Ok(Some(value)),
        Err(err) if is_missing_mapping_key_error(&err) => Ok(None),
        Err(err) => Err(err),
    }
}

fn store_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
    value: Value,
) -> Result<(), RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        dict.set(key, value);
        return Ok(());
    }

    let setitem_name = intern("__setitem__");
    let setitem = get_attribute_value(vm, mapping, &setitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __setitem__")
        } else {
            err
        }
    })?;
    invoke_callable_value(vm, setitem, &[key, value])?;
    Ok(())
}

fn delete_mapped_local(
    vm: &mut VirtualMachine,
    mapping: Value,
    name: &Arc<str>,
) -> Result<bool, RuntimeError> {
    let key = mapped_local_key(name);

    if let Some(ptr) = mapping.as_object_ptr()
        && extract_type_id(ptr) == TypeId::DICT
    {
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        return Ok(dict.remove(key).is_some());
    }

    let delitem_name = intern("__delitem__");
    let delitem = get_attribute_value(vm, mapping, &delitem_name).map_err(|err| {
        if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) {
            RuntimeError::type_error("class namespace mapping must define __delitem__")
        } else {
            err
        }
    })?;

    match invoke_callable_value(vm, delitem, &[key]) {
        Ok(_) => Ok(true),
        Err(err) if is_missing_mapping_key_error(&err) => Ok(false),
        Err(err) => Err(err),
    }
}

#[inline]
fn is_missing_mapping_key_error(err: &RuntimeError) -> bool {
    matches!(err.kind, RuntimeErrorKind::KeyError { .. })
        || matches!(
            err.kind,
            RuntimeErrorKind::Exception { type_id, .. }
                if type_id == ExceptionTypeId::KeyError.as_u8() as u16
        )
        || matches!(
            &err.kind,
            RuntimeErrorKind::InternalError { message } if message.as_ref() == "key not found"
        )
}

// =============================================================================
// Globals
// =============================================================================

/// LoadGlobal: dst = globals[names[imm16]]
#[inline(always)]
pub fn load_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (name, locals_mapping) = {
        let frame = vm.current_frame();
        (frame.get_name(inst.imm16()).clone(), frame.locals_mapping())
    };

    if let Some(mapping) = locals_mapping {
        return match load_mapped_local(vm, mapping, &name) {
            Ok(value) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                ControlFlow::Continue
            }
            Err(err) => ControlFlow::Error(err),
        };
    }

    match vm.module_scope_value(&name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => {
            // Check builtins
            match vm.builtins.get(&name) {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
            }
        }
    }
}

/// LoadBuiltin: dst = builtins[names[imm16]]
#[inline(always)]
pub fn load_builtin(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.builtins.get(&name) {
        Some(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

/// StoreGlobal: globals[names[imm16]] = dst-register
#[inline(always)]
pub fn store_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the name index.
    let value = frame.get_reg(inst.dst().0);

    vm.set_module_scope_value(name, value);
    ControlFlow::Continue
}

/// DeleteGlobal: del globals[names[imm16]]
#[inline(always)]
pub fn delete_global(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name = frame.get_name(inst.imm16()).clone();

    match vm.delete_module_scope_value(&name) {
        Some(_) => {
            crate::stdlib::_weakref::clear_unreachable_weakrefs_if_registered(vm);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::name_error(name)),
    }
}

// =============================================================================
// Closures
// =============================================================================

/// LoadClosure: dst = closure[imm16].get()
///
/// Loads the value from a cell in the closure environment.
/// Returns an error if the cell is unbound (UnboundLocalError).
#[inline(always)]
pub fn load_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();

    match &frame.closure {
        Some(env) => {
            let idx = inst.imm16() as usize;
            let cell = env.get_cell(idx);
            match cell.get() {
                Some(value) => {
                    vm.current_frame_mut().set_reg(inst.dst().0, value);
                    ControlFlow::Continue
                }
                None => {
                    // Cell is unbound - variable was deleted or never assigned
                    ControlFlow::Error(crate::error::RuntimeError::unbound_local_cell(idx))
                }
            }
        }
        None => {
            let code = &frame.code;
            ControlFlow::Error(crate::error::RuntimeError::internal(format!(
                "LoadClosure without closure environment in {} (qualname={}, cellvars={}, freevars={})",
                code.name,
                code.qualname,
                code.cellvars.len(),
                code.freevars.len()
            )))
        }
    }
}

/// StoreClosure: closure[imm16].set(dst-register)
///
/// Stores a value into a cell in the closure environment.
/// The cell uses interior mutability (atomic operations), so this works
/// despite the Arc<ClosureEnv> being immutable.
#[inline(always)]
pub fn store_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    // Store opcodes use DstImm16 encoding where the source register is carried
    // in the dst field and imm16 is the closure slot.
    let value = frame.get_reg(inst.dst().0);
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.set(idx, value);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "StoreClosure without closure environment",
        )),
    }
}

/// DeleteClosure: del closure[imm16]
///
/// Clears a cell in the closure environment, making it unbound.
/// Subsequent reads will raise UnboundLocalError.
#[inline(always)]
pub fn delete_closure(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let idx = inst.imm16() as usize;

    match &frame.closure {
        Some(env) => {
            env.get_cell(idx).clear();
            crate::stdlib::_weakref::clear_unreachable_weakrefs_if_registered(vm);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(crate::error::RuntimeError::internal(
            "DeleteClosure without closure environment",
        )),
    }
}

// =============================================================================
// Move
// =============================================================================

/// Move: dst = src1
#[inline(always)]
pub fn move_reg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
}
