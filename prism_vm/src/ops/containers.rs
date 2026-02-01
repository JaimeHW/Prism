//! Container construction opcode handlers.
//!
//! Handles building lists, tuples, dicts, sets, and string interpolation.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Container Construction
// =============================================================================

/// BuildList: dst = [r(src1)..r(src1+src2)]
#[inline(always)]
pub fn build_list(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        values.push(frame.get_reg(start_reg + i as u8));
    }

    // Create list on heap and get stable pointer
    let list = Box::new(ListObject::from_slice(&values));
    let ptr = Box::into_raw(list) as *const ();

    // Store as object Value
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildTuple: dst = (r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_tuple(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers
    let values: Vec<Value> = (0..count)
        .map(|i| frame.get_reg(start_reg + i as u8))
        .collect();

    // Create tuple on heap and get stable pointer
    let tuple = Box::new(TupleObject::from_slice(&values));
    let ptr = Box::into_raw(tuple) as *const ();

    // Store as object Value
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildSet: dst = {r(src1)..r(src1+src2)}
#[inline(always)]
pub fn build_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Create set and add elements
    let mut set = SetObject::new();
    for i in 0..count {
        let value = frame.get_reg(start_reg + i as u8);
        set.add(value);
    }

    // Store on heap
    let set_box = Box::new(set);
    let ptr = Box::into_raw(set_box) as *const ();
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildDict: dst = {} with src2 key-value pairs starting at src1
#[inline(always)]
pub fn build_dict(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let pair_count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Create dict and insert key-value pairs
    let mut dict = DictObject::new();
    for i in 0..pair_count {
        let key = frame.get_reg(start_reg + (i * 2) as u8);
        let value = frame.get_reg(start_reg + (i * 2 + 1) as u8);
        dict.set(key, value);
    }

    // Store on heap
    let dict_box = Box::new(dict);
    let ptr = Box::into_raw(dict_box) as *const ();
    frame.set_reg(dst, Value::object_ptr(ptr));
    ControlFlow::Continue
}

/// BuildString: dst = "".join(r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_string(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect string parts
    let mut result = String::new();
    for i in 0..count {
        let val = frame.get_reg(start_reg + i as u8);
        // Convert value to string representation
        if let Some(s) = val.as_int() {
            result.push_str(&s.to_string());
        } else if let Some(f) = val.as_float() {
            result.push_str(&f.to_string());
        } else if val.is_none() {
            result.push_str("None");
        } else if let Some(b) = val.as_bool() {
            result.push_str(if b { "True" } else { "False" });
        }
        // TODO: Handle string objects and other types
    }

    // TODO: Create StringObject when fully wired
    // For now, store as a None placeholder
    frame.set_reg(dst, Value::none());
    ControlFlow::Continue
}

// =============================================================================
// Container Modification
// =============================================================================

/// ListAppend: src1.append(src2)
///
/// Appends a value to a list object in place.
#[inline(always)]
pub fn list_append(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let list_val = frame.get_reg(inst.src1().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the list pointer and cast back to mutable ListObject
    if let Some(ptr) = list_val.as_object_ptr() {
        // SAFETY: We know this is a ListObject because BuildList created it
        let list = unsafe { &mut *(ptr as *mut ListObject) };
        list.push(value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected list object"))
    }
}

/// SetAdd: src1.add(src2)
///
/// Adds a value to a set object in place.
#[inline(always)]
pub fn set_add(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let set_val = frame.get_reg(inst.src1().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the set pointer and cast back to mutable SetObject
    if let Some(ptr) = set_val.as_object_ptr() {
        // SAFETY: We know this is a SetObject because BuildSet created it
        let set = unsafe { &mut *(ptr as *mut SetObject) };
        set.add(value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected set object"))
    }
}

/// DictSet: src1[dst] = src2 (dst is key register)
///
/// Sets a key-value pair in a dict object.
#[inline(always)]
pub fn dict_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let dict_val = frame.get_reg(inst.src1().0);
    let key = frame.get_reg(inst.dst().0);
    let value = frame.get_reg(inst.src2().0);

    // Get the dict pointer and cast back to mutable DictObject
    if let Some(ptr) = dict_val.as_object_ptr() {
        // SAFETY: We know this is a DictObject because BuildDict created it
        let dict = unsafe { &mut *(ptr as *mut DictObject) };
        dict.set(key, value);
        ControlFlow::Continue
    } else {
        ControlFlow::Error(RuntimeError::type_error("expected dict object"))
    }
}

// =============================================================================
// Unpacking
// =============================================================================

/// UnpackSequence: r(dst)..r(dst+src2) = unpack(src1)
#[inline(always)]
pub fn unpack_sequence(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _sequence = frame.get_reg(inst.src1().0);
    let _count = inst.src2().0;
    let _dst_start = inst.dst().0;

    // TODO: Implement sequence unpacking
    ControlFlow::Error(RuntimeError::internal("UnpackSequence not yet implemented"))
}

/// UnpackEx: unpack with *rest
#[inline(always)]
pub fn unpack_ex(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _sequence = frame.get_reg(inst.src1().0);

    // TODO: Implement extended unpacking with *rest
    ControlFlow::Error(RuntimeError::internal("UnpackEx not yet implemented"))
}

/// BuildSlice: dst = slice(src1, src2)
#[inline(always)]
pub fn build_slice(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let _start = frame.get_reg(inst.src1().0);
    let _stop = frame.get_reg(inst.src2().0);

    // TODO: Create slice object
    ControlFlow::Error(RuntimeError::internal("BuildSlice not yet implemented"))
}

// =============================================================================
// Import (Stubs)
// =============================================================================

/// ImportName: dst = import(name_idx)
#[inline(always)]
pub fn import_name(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame();
    let name_idx = inst.imm16();
    let name = frame.get_name(name_idx).clone();

    ControlFlow::Error(RuntimeError::new(
        crate::error::RuntimeErrorKind::ImportError {
            module: name,
            message: "Import not yet implemented".into(),
        },
    ))
}

/// ImportFrom: dst = from module import name
#[inline(always)]
pub fn import_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement import from
    ControlFlow::Error(RuntimeError::internal("ImportFrom not yet implemented"))
}

/// ImportStar: from module import *
#[inline(always)]
pub fn import_star(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement import star
    ControlFlow::Error(RuntimeError::internal("ImportStar not yet implemented"))
}

#[cfg(test)]
mod tests {
    // Container tests require full VM setup with object system
}
