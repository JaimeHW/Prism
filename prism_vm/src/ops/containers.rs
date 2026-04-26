//! Container construction opcode handlers.
//!
//! Handles building lists, tuples, dicts, sets, and string interpolation.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use crate::ops::iteration::collect_iterable_values;
use crate::ops::objects::read_attr_name;
use prism_code::Instruction;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::types::dict::DictObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::set::SetObject;
use prism_runtime::types::string::{StringObject, value_as_string_ref};
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Container Construction
// =============================================================================

/// BuildList: dst = [r(src1)..r(src1+src2)]
#[inline(always)]
pub fn build_list(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers (borrow frame, then release)
    let values: Vec<Value> = {
        let frame = vm.current_frame();
        (0..count)
            .map(|i| frame.get_reg(start_reg + i as u8))
            .collect()
    };

    let list = ListObject::from_slice(&values);
    let value = alloc_value_in_current_heap_or_box(list);

    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// BuildTuple: dst = (r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_tuple(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values from registers (borrow frame, then release)
    let values: Vec<Value> = {
        let frame = vm.current_frame();
        (0..count)
            .map(|i| frame.get_reg(start_reg + i as u8))
            .collect()
    };

    let tuple = TupleObject::from_slice(&values);
    let value = alloc_value_in_current_heap_or_box(tuple);

    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// BuildSet: dst = {r(src1)..r(src1+src2)}
#[inline(always)]
pub fn build_set(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Collect values and build set (borrow frame, then release)
    let set = {
        let frame = vm.current_frame();
        let mut set = SetObject::new();
        for i in 0..count {
            let value = frame.get_reg(start_reg + i as u8);
            set.add(value);
        }
        set
    };

    let value = alloc_value_in_current_heap_or_box(set);

    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// BuildDict: dst = {} with src2 key-value pairs starting at src1
#[inline(always)]
pub fn build_dict(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let pair_count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    // Build dict (borrow frame, then release)
    let dict = {
        let frame = vm.current_frame();
        let mut dict = DictObject::new();
        for i in 0..pair_count {
            let key = frame.get_reg(start_reg + (i * 2) as u8);
            let value = frame.get_reg(start_reg + (i * 2 + 1) as u8);
            dict.set(key, value);
        }
        dict
    };

    let value = alloc_value_in_current_heap_or_box(dict);

    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// BuildString: dst = "".join(r(src1)..r(src1+src2))
#[inline(always)]
pub fn build_string(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let start_reg = inst.src1().0;
    let count = inst.src2().0 as usize;
    let dst = inst.dst().0;

    let values: Vec<Value> = {
        let frame = vm.current_frame();
        (0..count)
            .map(|i| frame.get_reg(start_reg + i as u8))
            .collect()
    };

    if values.is_empty() {
        vm.current_frame_mut()
            .set_reg(dst, Value::string(intern("")));
        return ControlFlow::Continue;
    }

    if values.len() == 1 {
        let value = values[0];
        if value_as_string_ref(value).is_none() {
            return ControlFlow::Error(RuntimeError::type_error(
                "BUILD_STRING expects string operands",
            ));
        }
        vm.current_frame_mut().set_reg(dst, value);
        return ControlFlow::Continue;
    }

    let mut parts = Vec::with_capacity(values.len());
    let mut total_len = 0usize;
    for value in values {
        let Some(part) = value_as_string_ref(value) else {
            return ControlFlow::Error(RuntimeError::type_error(
                "BUILD_STRING expects string operands",
            ));
        };
        total_len += part.len();
        parts.push(part);
    }

    let mut joined = String::with_capacity(total_len);
    for part in parts {
        joined.push_str(part.as_str());
    }

    if joined.is_empty() {
        vm.current_frame_mut()
            .set_reg(dst, Value::string(intern("")));
        return ControlFlow::Continue;
    }

    let string = StringObject::from_string(joined);
    let value = alloc_value_in_current_heap_or_box(string);

    vm.current_frame_mut().set_reg(dst, value);
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
///
/// Unpacks a sequence into consecutive registers starting at dst.
/// Follows Python's iterator protocol, including metaclass-defined `__iter__`.
///
/// # Performance
///
/// - List/Tuple: O(1) per element access via direct indexing
/// - String: O(n) due to UTF-8 character iteration (lazy single-pass)
/// - Range: O(1) per element via arithmetic computation
///
/// # Errors
///
/// Returns ValueError if the sequence length doesn't match the expected count.
#[inline(always)]
pub fn unpack_sequence(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_start = inst.dst().0;
    let count = inst.src2().0 as usize;

    // Get the sequence value
    let sequence = vm.current_frame().get_reg(inst.src1().0);
    let values: Vec<Value> = match collect_iterable_values(vm, sequence) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    if values.len() != count {
        return ControlFlow::Error(RuntimeError::value_error(format!(
            "not enough values to unpack (expected {}, got {})",
            count,
            values.len()
        )));
    }

    let frame = vm.current_frame_mut();
    for (index, value) in values.into_iter().enumerate() {
        frame.set_reg(dst_start + index as u8, value);
    }

    ControlFlow::Continue
}

/// UnpackEx: unpack with *rest
///
/// Extended unpacking for patterns like `a, *rest, b = sequence`.
/// The dst field encodes the before/after counts in a packed format.
///
/// Instruction format:
/// - dst: destination base register
/// - src1: sequence register
/// - src2: packed (before_count << 4) | after_count
///
/// # Performance
///
/// Uses a single pass over the sequence to collect all values,
/// then distributes them to registers.
#[inline(always)]
pub fn unpack_ex(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst_start = inst.dst().0;
    let sequence = vm.current_frame().get_reg(inst.src1().0);

    // Decode before/after counts from src2
    let packed = inst.src2().0;
    let before_count = (packed >> 4) as usize;
    let after_count = (packed & 0x0F) as usize;
    let min_required = before_count + after_count;

    let values: Vec<Value> = match collect_iterable_values(vm, sequence) {
        Ok(values) => values,
        Err(err) => return ControlFlow::Error(err),
    };

    let total = values.len();
    if total < min_required {
        return ControlFlow::Error(RuntimeError::value_error(format!(
            "not enough values to unpack (expected at least {}, got {})",
            min_required, total
        )));
    }

    let rest_count = total - min_required;

    // 1. Assign before values
    {
        let frame = vm.current_frame_mut();
        for i in 0..before_count {
            frame.set_reg(dst_start + i as u8, values[i]);
        }
    }

    // 2. Create rest list on the active heap and assign
    let rest_values: Vec<Value> = values[before_count..before_count + rest_count].to_vec();
    let rest_list = ListObject::from_slice(&rest_values);
    let rest_value = alloc_value_in_current_heap_or_box(rest_list);
    vm.current_frame_mut()
        .set_reg(dst_start + before_count as u8, rest_value);

    // 3. Assign after values
    {
        let frame = vm.current_frame_mut();
        for i in 0..after_count {
            let src_idx = before_count + rest_count + i;
            let dst_idx = before_count + 1 + i; // +1 for rest list register
            frame.set_reg(dst_start + dst_idx as u8, values[src_idx]);
        }
    }

    ControlFlow::Continue
}

/// BuildSlice: dst = slice(src1, src2[, step])
///
/// Creates a SliceObject from start and stop values.
/// For 3-arg slices, the compiler emits an extension instruction immediately
/// after BuildSlice:
/// - opcode: CallKwEx
/// - dst: step register index
/// - src1/src2: marker bytes ('S','L')
///
/// # Value Interpretation
///
/// - None values indicate "use default" (beginning/end)
/// - Integer values are used directly
/// - Other types raise TypeError
///
/// # Performance
///
/// O(1) allocation and construction. SliceObject is 40 bytes and fits
/// in a cache line for efficient access during slicing operations.
#[inline(always)]
pub fn build_slice(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use prism_runtime::types::SliceObject;

    const STEP_EXT_TAG_A: u8 = b'S';
    const STEP_EXT_TAG_B: u8 = b'L';

    // Read values from frame (borrow then release)
    let (start_val, stop_val) = {
        let frame = vm.current_frame();
        (frame.get_reg(inst.src1().0), frame.get_reg(inst.src2().0))
    };
    let dst = inst.dst().0;

    // Optional step from extension instruction.
    let mut step: Option<i64> = None;
    {
        let frame = vm.current_frame_mut();
        if (frame.ip as usize) < frame.code.instructions.len() {
            let next = frame.code.instructions[frame.ip as usize];
            if next.opcode() == prism_code::Opcode::CallKwEx as u8
                && next.src1().0 == STEP_EXT_TAG_A
                && next.src2().0 == STEP_EXT_TAG_B
            {
                let ext = frame.fetch();
                let step_val = frame.get_reg(ext.dst().0);
                step = match value_to_slice_index(step_val) {
                    Ok(v) => v,
                    Err(cf) => return cf,
                };
            }
        }
    }

    // Convert Values to Option<i64> with explicit error handling
    let start = match value_to_slice_index(start_val) {
        Ok(v) => v,
        Err(cf) => return cf,
    };
    let stop = match value_to_slice_index(stop_val) {
        Ok(v) => v,
        Err(cf) => return cf,
    };
    if step == Some(0) {
        return ControlFlow::Error(RuntimeError::value_error("slice step cannot be zero"));
    }

    // Create slice on the active heap
    let slice = SliceObject::new(start, stop, step);
    let value = alloc_value_in_current_heap_or_box(slice);

    vm.current_frame_mut().set_reg(dst, value);
    ControlFlow::Continue
}

/// Helper to convert a Value to an optional slice index.
///
/// Returns:
/// - Ok(None) for None value
/// - Ok(Some(i)) for integer value
/// - Err for other types
#[inline]
fn value_to_slice_index(val: Value) -> Result<Option<i64>, ControlFlow> {
    if val.is_none() {
        Ok(None)
    } else if let Some(i) = val.as_int() {
        Ok(Some(i))
    } else {
        Err(ControlFlow::Error(RuntimeError::type_error(
            "slice indices must be integers or None",
        )))
    }
}

// =============================================================================
// Import Operations
// =============================================================================

/// ImportName: dst = import(name_idx)
///
/// Imports a module by name index and stores the module object in dst register.
/// Uses the VM's ImportResolver for high-performance cached module lookup.
#[inline(always)]
pub fn import_name(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst();
    let name_idx = inst.imm16();

    let frame = vm.current_frame();
    let module_name = frame.get_name(name_idx).clone();

    match vm.import_module_named(&module_name) {
        Ok(module) => {
            // Store a stable pointer to the cached ModuleObject.
            // ImportResolver owns the Arc in sys.modules, so this pointer stays valid.
            let module_ptr = std::sync::Arc::as_ptr(&module) as *const ();
            vm.current_frame_mut()
                .set_reg(dst.0, Value::object_ptr(module_ptr));
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// ImportFrom: dst = from module import attr
///
/// Imports a specific attribute from a module object.
/// Encoding: dst=destination, src=module register, imm8=attr name index
#[inline(always)]
pub fn import_from(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let dst = inst.dst();
    let module_reg_idx = inst.src1().0;

    let attr_name = match read_attr_name(vm, inst.src2().0) {
        Ok(name) => name,
        Err(err) => return ControlFlow::Error(err),
    };
    let module_value = vm.current_frame().get_reg(module_reg_idx);

    let Some(module_ptr) = module_value.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::import_error(
            "<unknown>",
            "Cannot import from None",
        ));
    };

    let Some(module) = vm.import_resolver.module_from_ptr(module_ptr) else {
        return ControlFlow::Error(RuntimeError::import_error(
            "<unknown>",
            "cannot import from non-module object",
        ));
    };

    match vm.import_from_module(&module, &attr_name) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(dst.0, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err),
    }
}

/// ImportStar: from module import *
///
/// Imports all public names from a module into the current global scope.
/// Encoding: dst=unused, src=module register
#[inline(always)]
pub fn import_star(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let module_reg_idx = inst.src1().0;

    let frame = vm.current_frame();
    let module_value = frame.get_reg(module_reg_idx);

    let Some(module_ptr) = module_value.as_object_ptr() else {
        return ControlFlow::Error(RuntimeError::import_error(
            "<unknown>",
            "Cannot import * from None",
        ));
    };

    let Some(module) = vm.import_resolver.module_from_ptr(module_ptr) else {
        return ControlFlow::Error(RuntimeError::import_error(
            "<unknown>",
            "cannot import * from non-module object",
        ));
    };

    // Get all public names from the module. If __all__ is defined, use it
    // exactly; otherwise use all non-underscore names.
    match vm.import_star_into_current_scope(&module) {
        Ok(()) => ControlFlow::Continue,
        Err(err) => ControlFlow::Error(err),
    }
}

#[cfg(test)]
mod tests;
