//! Arithmetic opcode handlers.
//!
//! Provides type-specialized fast paths for int/float operations,
//! with fallback to generic polymorphic operations.
//!
//! # Type Feedback Integration
//!
//! Generic handlers collect type feedback via `BinaryOpFeedback` for JIT
//! specialization. After sufficient observations, the JIT can emit
//! specialized code paths.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::{RuntimeError, RuntimeErrorKind};
use crate::ops::calls::invoke_callable_value;
use crate::ops::method_dispatch::load_method::{BoundMethodTarget, resolve_special_method};
use crate::python_numeric::{
    complex_like_parts, float_like_value, int_like_value, is_complex_value,
};
use crate::type_feedback::BinaryOpFeedback;
use prism_code::Instruction;
use prism_core::Value;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::{
    StringObject, concat_string_objects, repeat_string_object, value_as_string_ref,
};
use prism_runtime::types::tuple::TupleObject;

// =============================================================================
// Integer Arithmetic (Fast Path)
// =============================================================================

/// AddInt: dst = src1 + src2 (integers only)
#[inline(always)]
pub fn add_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Fast path: both are integers
    match (int_like_value(a), int_like_value(b)) {
        (Some(x), Some(y)) => match x.checked_add(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("AddInt requires integers")),
    }
}

/// SubInt: dst = src1 - src2
#[inline(always)]
pub fn sub_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(x), Some(y)) => match x.checked_sub(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("SubInt requires integers")),
    }
}

/// MulInt: dst = src1 * src2
#[inline(always)]
pub fn mul_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(x), Some(y)) => match x.checked_mul(y) {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        _ => ControlFlow::Error(RuntimeError::type_error("MulInt requires integers")),
    }
}

/// FloorDivInt: dst = src1 // src2
#[inline(always)]
pub fn floor_div_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(_), Some(0)) => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style floor division
            let result = x.div_euclid(y);
            match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("FloorDivInt requires integers")),
    }
}

/// ModInt: dst = src1 % src2
#[inline(always)]
pub fn mod_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(_), Some(0)) => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style modulo
            let result = x.rem_euclid(y);
            match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("ModInt requires integers")),
    }
}

/// PowInt: dst = src1 ** src2
#[inline(always)]
pub fn pow_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(base), Some(exp)) => {
            if exp < 0 {
                // Negative exponent produces float
                let result = (base as f64).powi(exp as i32);
                frame.set_reg(inst.dst().0, Value::float(result));
                ControlFlow::Continue
            } else {
                match (base as i128).checked_pow(exp as u32) {
                    Some(result) if result >= i64::MIN as i128 && result <= i64::MAX as i128 => {
                        match Value::int(result as i64) {
                            Some(v) => {
                                frame.set_reg(inst.dst().0, v);
                                ControlFlow::Continue
                            }
                            None => ControlFlow::Error(RuntimeError::value_error(
                                "Integer too large for i48",
                            )),
                        }
                    }
                    _ => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
                }
            }
        }
        _ => ControlFlow::Error(RuntimeError::type_error("PowInt requires integers")),
    }
}

/// NegInt: dst = -src1
#[inline(always)]
pub fn neg_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match int_like_value(a) {
        Some(x) => match x.checked_neg() {
            Some(result) => match Value::int(result) {
                Some(v) => {
                    frame.set_reg(inst.dst().0, v);
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::value_error("Integer too large for i48")),
            },
            None => ControlFlow::Error(RuntimeError::value_error("Integer overflow")),
        },
        None => ControlFlow::Error(RuntimeError::type_error("NegInt requires integers")),
    }
}

/// PosInt: dst = +src1 (identity)
#[inline(always)]
pub fn pos_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    match int_like_value(value).and_then(Value::int) {
        Some(result) => {
            frame.set_reg(inst.dst().0, result);
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::type_error("PosInt requires integers")),
    }
}

// =============================================================================
// Float Arithmetic (Fast Path)
// =============================================================================

/// AddFloat: dst = src1 + src2 (floats)
#[inline(always)]
pub fn add_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x + y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("AddFloat requires floats")),
    }
}

/// SubFloat: dst = src1 - src2
#[inline(always)]
pub fn sub_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x - y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("SubFloat requires floats")),
    }
}

/// MulFloat: dst = src1 * src2
#[inline(always)]
pub fn mul_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x * y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("MulFloat requires floats")),
    }
}

/// DivFloat: dst = src1 / src2
#[inline(always)]
pub fn div_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x / y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("DivFloat requires floats")),
    }
}

/// FloorDivFloat: dst = src1 // src2
#[inline(always)]
pub fn floor_div_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float((x / y).floor()));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("FloorDivFloat requires floats")),
    }
}

/// ModFloat: dst = src1 % src2
#[inline(always)]
pub fn mod_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(_), Some(y)) if y == 0.0 => ControlFlow::Error(RuntimeError::zero_division()),
        (Some(x), Some(y)) => {
            // Python-style modulo
            let result = x - y * (x / y).floor();
            frame.set_reg(inst.dst().0, Value::float(result));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("ModFloat requires floats")),
    }
}

/// PowFloat: dst = src1 ** src2
#[inline(always)]
pub fn pow_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_float(), b.as_float()) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::float(x.powf(y)));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::type_error("PowFloat requires floats")),
    }
}

/// NegFloat: dst = -src1
#[inline(always)]
pub fn neg_float(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match a.as_float() {
        Some(x) => {
            frame.set_reg(inst.dst().0, Value::float(-x));
            ControlFlow::Continue
        }
        None => ControlFlow::Error(RuntimeError::type_error("NegFloat requires floats")),
    }
}

// =============================================================================
// Generic Arithmetic (Polymorphic - Slower)
// =============================================================================

/// Add: dst = src1 + src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
/// Records type feedback on slow path for future specialization.
#[inline(always)]
pub fn add(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{
        SpecResult, Speculation, spec_add_float, spec_add_int, spec_list_concat, spec_str_concat,
    };
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_add_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                // Deopt: invalidate cache and fall through to slow path
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_add_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrStr => {
                // String concatenation fast path
                let (result, value) = match spec_str_concat(vm, a, b) {
                    Ok(result) => result,
                    Err(err) => return ControlFlow::Error(err),
                };
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::ListList => {
                // List concatenation fast path
                let (result, value) = spec_list_concat(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrInt | Speculation::IntStr => {
                // StrInt/IntStr don't apply to addition (only mul for repetition)
            }
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================

    // Record type feedback for future speculation
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    // Update speculation cache for next time
    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    // Try int + int
    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if let Some(result) = x.checked_add(y) {
            if let Some(v) = Value::int(result) {
                vm.current_frame_mut().set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    if let Some(value) = try_add_complex_values(a, b) {
        vm.current_frame_mut().set_reg(inst.dst().0, value);
        return ControlFlow::Continue;
    }

    // Try str + str (supports both tagged interned strings and heap strings)
    match concat_string_value_in_vm(vm, a, b) {
        Ok(Some(value)) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            return ControlFlow::Continue;
        }
        Ok(None) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    // Try bytes/bytearray concatenation. Match CPython's sequence semantics:
    // accept byte sequences on both sides and preserve the left operand type.
    if let Some(value) = concat_byte_sequence_values(a, b) {
        vm.current_frame_mut().set_reg(inst.dst().0, value);
        return ControlFlow::Continue;
    }

    // Try list + list (slow path for list concatenation)
    if a.is_object() && b.is_object() {
        // Use spec_list_concat which handles type checking internally
        let (result, value) = spec_list_concat(a, b);
        if result == SpecResult::Success {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            return ControlFlow::Continue;
        }
        // If deopt, fall through to other type checks
    }

    // Try tuple + tuple
    if let (Some(a_ptr), Some(b_ptr)) = (a.as_object_ptr(), b.as_object_ptr()) {
        let a_header = unsafe { &*(a_ptr as *const ObjectHeader) };
        let b_header = unsafe { &*(b_ptr as *const ObjectHeader) };
        if a_header.type_id == TypeId::TUPLE && b_header.type_id == TypeId::TUPLE {
            let tuple = unsafe {
                (&*(a_ptr as *const TupleObject)).concat(&*(b_ptr as *const TupleObject))
            };
            let boxed = Box::new(tuple);
            let ptr = Box::into_raw(boxed) as *const ();
            vm.current_frame_mut()
                .set_reg(inst.dst().0, Value::object_ptr(ptr));
            return ControlFlow::Continue;
        }
    }

    // Try float + float or mixed int/float
    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "+",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "+",
            a.type_name(),
            b.type_name(),
        ));
    };

    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::float(x + y));
    ControlFlow::Continue
}

#[inline]
fn concat_byte_sequence_values(left: Value, right: Value) -> Option<Value> {
    let left_bytes = value_as_byte_sequence_ref(left)?;
    let right_bytes = value_as_byte_sequence_ref(right)?;
    let result = left_bytes.concat(right_bytes.as_bytes());
    Some(boxed_bytes_value(result))
}

#[inline]
pub(crate) fn alloc_string_value(
    vm: &VirtualMachine,
    string: StringObject,
) -> Result<Value, RuntimeError> {
    vm.allocator()
        .alloc_value(string)
        .ok_or_else(|| RuntimeError::internal("out of memory: failed to allocate string"))
}

#[inline]
pub(crate) fn concat_string_value_in_vm(
    vm: &VirtualMachine,
    left: Value,
    right: Value,
) -> Result<Option<Value>, RuntimeError> {
    let Some(left_ref) = value_as_string_ref(left) else {
        return Ok(None);
    };
    let Some(right_ref) = value_as_string_ref(right) else {
        return Ok(None);
    };

    if left_ref.is_empty() {
        return Ok(Some(right));
    }
    if right_ref.is_empty() {
        return Ok(Some(left));
    }

    let result = concat_string_objects(left, right).expect("validated string operands");
    alloc_string_value(vm, result).map(Some)
}

#[inline]
pub(crate) fn repeat_string_value_in_vm(
    vm: &VirtualMachine,
    string: Value,
    count: i64,
) -> Result<Option<Value>, RuntimeError> {
    let Some(string_ref) = value_as_string_ref(string) else {
        return Ok(None);
    };

    if count <= 0 || string_ref.is_empty() {
        return Ok(Some(Value::string(prism_core::intern::intern(""))));
    }
    if count == 1 {
        return Ok(Some(string));
    }

    let result = repeat_string_object(string, count).expect("validated string operand");
    alloc_string_value(vm, result).map(Some)
}

#[inline]
fn boxed_complex_value(real: f64, imag: f64) -> Value {
    let ptr = Box::into_raw(Box::new(ComplexObject::new(real, imag)));
    Value::object_ptr(ptr as *const ())
}

#[inline]
fn try_add_complex_values(left: Value, right: Value) -> Option<Value> {
    if !is_complex_value(left) && !is_complex_value(right) {
        return None;
    }

    let left = complex_like_parts(left)?;
    let right = complex_like_parts(right)?;
    Some(boxed_complex_value(
        left.real + right.real,
        left.imag + right.imag,
    ))
}

#[inline]
fn try_sub_complex_values(left: Value, right: Value) -> Option<Value> {
    if !is_complex_value(left) && !is_complex_value(right) {
        return None;
    }

    let left = complex_like_parts(left)?;
    let right = complex_like_parts(right)?;
    Some(boxed_complex_value(
        left.real - right.real,
        left.imag - right.imag,
    ))
}

#[inline]
fn value_as_byte_sequence_ref(value: Value) -> Option<&'static BytesObject> {
    let ptr = value.as_object_ptr()?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    match header.type_id {
        TypeId::BYTES | TypeId::BYTEARRAY => Some(unsafe { &*(ptr as *const BytesObject) }),
        _ => None,
    }
}

/// Sub: dst = src1 - src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
/// Records type feedback on slow path for future specialization.
#[inline(always)]
pub fn sub(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_sub_float, spec_sub_int};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_sub_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_sub_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if let Some(result) = x.checked_sub(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    if let Some(value) = try_sub_complex_values(a, b) {
        frame.set_reg(inst.dst().0, value);
        return ControlFlow::Continue;
    }

    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "-",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "-",
            a.type_name(),
            b.type_name(),
        ));
    };

    frame.set_reg(inst.dst().0, Value::float(x - y));
    ControlFlow::Continue
}

/// Mul: dst = src1 * src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn mul(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{
        SpecResult, Speculation, spec_mul_float, spec_mul_int, spec_str_repeat,
    };
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_mul_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_mul_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::StrInt | Speculation::IntStr => {
                // String repetition fast path (str * int or int * str)
                let (result, value) = match spec_str_repeat(vm, a, b) {
                    Ok(result) => result,
                    Err(err) => return ControlFlow::Error(err),
                };
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {}
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None | Speculation::StrStr | Speculation::ListList => {
                // StrStr and ListList don't apply to multiplication
            }
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if let Some(result) = x.checked_mul(y) {
            if let Some(v) = Value::int(result) {
                vm.current_frame_mut().set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    if let Some(n) = int_like_value(b) {
        match repeat_string_value_in_vm(vm, a, n) {
            Ok(Some(value)) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Ok(None) => {}
            Err(err) => return ControlFlow::Error(err),
        }
        match repeat_sequence_value(a, n) {
            Ok(Some(value)) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Ok(None) => {}
            Err(err) => return ControlFlow::Error(err),
        }
    }
    if let Some(n) = int_like_value(a) {
        match repeat_string_value_in_vm(vm, b, n) {
            Ok(Some(value)) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Ok(None) => {}
            Err(err) => return ControlFlow::Error(err),
        }
        match repeat_sequence_value(b, n) {
            Ok(Some(value)) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Ok(None) => {}
            Err(err) => return ControlFlow::Error(err),
        }
    }

    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "*",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "*",
            a.type_name(),
            b.type_name(),
        ));
    };

    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::float(x * y));
    ControlFlow::Continue
}

fn repeat_sequence_value(value: Value, count: i64) -> Result<Option<Value>, RuntimeError> {
    let Some(ptr) = value.as_object_ptr() else {
        return Ok(None);
    };

    let repeat =
        usize::try_from(count.max(0)).map_err(|_| RuntimeError::value_error("Integer overflow"))?;
    let header = unsafe { &*(ptr as *const ObjectHeader) };

    match header.type_id {
        TypeId::LIST => {
            let list = unsafe { &*(ptr as *const ListObject) };
            ensure_repeated_sequence_len(list.len(), repeat)?;
            Ok(Some(boxed_list_value(list.repeat(repeat))))
        }
        TypeId::TUPLE => {
            let tuple = unsafe { &*(ptr as *const TupleObject) };
            ensure_repeated_sequence_len(tuple.len(), repeat)?;
            Ok(Some(boxed_tuple_value(tuple.repeat(repeat))))
        }
        TypeId::BYTES | TypeId::BYTEARRAY => {
            let bytes = unsafe { &*(ptr as *const BytesObject) };
            ensure_repeated_sequence_len(bytes.len(), repeat)?;
            let repeated = bytes
                .repeat_sequence(repeat)
                .ok_or_else(|| RuntimeError::value_error("repeated sequence is too long"))?;
            Ok(Some(boxed_bytes_value(repeated)))
        }
        _ => Ok(None),
    }
}

fn ensure_repeated_sequence_len(len: usize, repeat: usize) -> Result<(), RuntimeError> {
    let _ = len
        .checked_mul(repeat)
        .ok_or_else(|| RuntimeError::value_error("repeated sequence is too long"))?;
    Ok(())
}

#[inline]
fn boxed_list_value(list: ListObject) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(list)) as *const ())
}

#[inline]
fn boxed_tuple_value(tuple: TupleObject) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(tuple)) as *const ())
}

#[inline]
fn boxed_bytes_value(bytes: BytesObject) -> Value {
    Value::object_ptr(Box::into_raw(Box::new(bytes)) as *const ())
}

/// TrueDiv: dst = src1 / src2 (always returns float, with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn true_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_div_float};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path (true_div always returns float)
    if let Some(spec) = vm.speculation_cache.get(site) {
        if spec.is_float() || spec == Speculation::IntInt {
            let (result, value) = spec_div_float(a, b);
            if result == SpecResult::Success {
                let frame = vm.current_frame_mut();
                frame.set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            // Division by zero or type mismatch
            if result == SpecResult::Overflow {
                return ControlFlow::Error(RuntimeError::zero_division());
            }
            vm.speculation_cache.invalidate(site);
        }
    }

    // Slow Path
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "/",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "/",
            a.type_name(),
            b.type_name(),
        ));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float(x / y));
    ControlFlow::Continue
}

/// FloorDiv: dst = src1 // src2 (generic with speculative fast-path)
///
/// Int // int returns int. Float // float returns float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn floor_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_floor_div_float, spec_floor_div_int};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_floor_div_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        // Division by zero
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_floor_div_float(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // Int // int returns int
    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if y == 0 {
            return ControlFlow::Error(RuntimeError::zero_division());
        }
        if let Some(v) = Value::int(x.div_euclid(y)) {
            frame.set_reg(inst.dst().0, v);
            return ControlFlow::Continue;
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    // Otherwise returns float
    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//",
            a.type_name(),
            b.type_name(),
        ));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float((x / y).floor()));
    ControlFlow::Continue
}

/// Mod: dst = src1 % src2 (generic with speculative fast-path)
///
/// Int % int returns int. Float % float returns float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn modulo(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_mod_float, spec_mod_int};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_mod_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_mod_float(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        return ControlFlow::Error(RuntimeError::zero_division());
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    if let Some(template) = value_as_string_ref(a) {
        match crate::builtins::percent_format_string(template.as_str(), b) {
            Ok(value) => {
                frame.set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Err(err) => return ControlFlow::Error(err.into()),
        }
    }

    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if y == 0 {
            return ControlFlow::Error(RuntimeError::zero_division());
        }
        if let Some(v) = Value::int(x.rem_euclid(y)) {
            frame.set_reg(inst.dst().0, v);
            return ControlFlow::Continue;
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "%",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "%",
            a.type_name(),
            b.type_name(),
        ));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    let result = x - y * (x / y).floor();
    frame.set_reg(inst.dst().0, Value::float(result));
    ControlFlow::Continue
}

/// Pow: dst = src1 ** src2 (generic with speculative fast-path)
///
/// Int ** positive int returns int (if no overflow). Otherwise float.
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn pow(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_pow_float, spec_pow_int};
    use crate::type_feedback::OperandPair;

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // =========================================================================
    // Speculative Fast Path (O(1) cache lookup)
    // =========================================================================
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_pow_int(a, b);
                match result {
                    SpecResult::Success => {
                        let frame = vm.current_frame_mut();
                        frame.set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                    SpecResult::Overflow => {
                        // Overflow: fall through to slow path which converts to float
                        // Don't invalidate - this is expected behavior
                    }
                    SpecResult::Deopt => {
                        vm.speculation_cache.invalidate(site);
                    }
                }
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_pow_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None
            | Speculation::StrStr
            | Speculation::StrInt
            | Speculation::IntStr
            | Speculation::ListList => {}
        }
    }

    // =========================================================================
    // Slow Path: Full type check + feedback recording
    // =========================================================================
    let pair = OperandPair::from_values(a, b);
    let feedback = BinaryOpFeedback::new(code_id, bc_offset, a, b);
    feedback.record(&mut vm.ic_manager);

    let spec = Speculation::from_operand_pair(pair);
    if spec != Speculation::None {
        vm.speculation_cache.insert(site, spec);
    }

    let frame = vm.current_frame_mut();

    // int ** positive int returns int
    if let (Some(base), Some(exp)) = (int_like_value(a), int_like_value(b)) {
        if exp >= 0 && exp <= 63 {
            if let Some(result) = (base as i128).checked_pow(exp as u32) {
                if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                    if let Some(v) = Value::int(result as i64) {
                        frame.set_reg(inst.dst().0, v);
                        return ControlFlow::Continue;
                    }
                }
            }
        }
        // Fall through to float for large or negative exponents
    }

    let Some(x) = float_like_value(a) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**",
            a.type_name(),
            b.type_name(),
        ));
    };
    let Some(y) = float_like_value(b) else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**",
            a.type_name(),
            b.type_name(),
        ));
    };

    frame.set_reg(inst.dst().0, Value::float(x.powf(y)));
    ControlFlow::Continue
}

/// Neg: dst = -src1 (generic)
#[inline(always)]
pub fn neg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    if let Some(x) = int_like_value(a) {
        if let Some(result) = x.checked_neg() {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    if let Some(x) = a.as_float() {
        frame.set_reg(inst.dst().0, Value::float(-x));
        return ControlFlow::Continue;
    }

    ControlFlow::Error(RuntimeError::type_error("bad operand type for unary -"))
}

/// Pos: dst = +src1 (generic)
#[inline(always)]
pub fn pos(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let src = vm.current_frame().get_reg(inst.src1().0);
    let result = match pos_value(vm, src) {
        Ok(result) => result,
        Err(err) => return ControlFlow::Error(err),
    };

    vm.current_frame_mut().set_reg(inst.dst().0, result);
    ControlFlow::Continue
}

#[inline]
fn pos_value(vm: &mut VirtualMachine, value: Value) -> Result<Value, RuntimeError> {
    if let Some(integer) = int_like_value(value) {
        return Value::int(integer)
            .ok_or_else(|| RuntimeError::value_error("Integer too large for i48"));
    }

    if let Some(float_value) = value.as_float() {
        return Ok(Value::float(float_value));
    }

    if is_complex_value(value) {
        let parts = complex_like_parts(value)
            .ok_or_else(|| RuntimeError::type_error("bad operand type for unary +"))?;
        return Ok(boxed_complex_value(parts.real, parts.imag));
    }

    let target = match resolve_special_method(value, "__pos__") {
        Ok(target) => target,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Err(RuntimeError::type_error(format!(
                "bad operand type for unary +: '{}'",
                value.type_name()
            )));
        }
        Err(err) => return Err(err),
    };

    invoke_zero_arg_bound_method(vm, target)
}

#[inline]
fn invoke_zero_arg_bound_method(
    vm: &mut VirtualMachine,
    target: BoundMethodTarget,
) -> Result<Value, RuntimeError> {
    match target.implicit_self {
        Some(implicit_self) => invoke_callable_value(vm, target.callable, &[implicit_self]),
        None => invoke_callable_value(vm, target.callable, &[]),
    }
}

#[cfg(test)]
mod tests {
    use super::{add, modulo, mul, neg, pos, pos_int, pow, sub, true_div};
    use crate::ControlFlow;
    use crate::VirtualMachine;
    use prism_code::{CodeObject, Instruction, Opcode, Register};
    use prism_core::Value;
    use prism_core::intern::{intern, interned_by_ptr};
    use prism_runtime::object::type_obj::TypeId;
    use prism_runtime::types::bytes::BytesObject;
    use prism_runtime::types::list::ListObject;
    use prism_runtime::types::string::value_as_string_ref;
    use prism_runtime::types::tuple::TupleObject;
    use std::sync::Arc;

    fn vm_with_frame() -> VirtualMachine {
        let mut code = CodeObject::new("test_add", "<test>");
        code.register_count = 16;
        let mut vm = VirtualMachine::new();
        vm.push_frame(Arc::new(code), 0).expect("frame push failed");
        vm
    }

    fn value_to_rust_string(value: Value) -> String {
        let string = value_as_string_ref(value).expect("value should be a Python string");
        string.as_str().to_string()
    }

    fn value_to_byte_vec(value: Value) -> Vec<u8> {
        let ptr = value
            .as_object_ptr()
            .expect("byte sequence should be object-backed");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        assert!(matches!(header.type_id, TypeId::BYTES | TypeId::BYTEARRAY));
        let bytes = unsafe { &*(ptr as *const BytesObject) };
        bytes.as_bytes().to_vec()
    }

    fn byte_sequence_type(value: Value) -> TypeId {
        let ptr = value
            .as_object_ptr()
            .expect("byte sequence should be object-backed");
        let header = unsafe { &*(ptr as *const prism_runtime::object::ObjectHeader) };
        header.type_id
    }

    fn binary_inst(opcode: Opcode) -> Instruction {
        Instruction::op_dss(opcode, Register::new(0), Register::new(1), Register::new(2))
    }

    fn unary_inst(opcode: Opcode) -> Instruction {
        Instruction::op_ds(opcode, Register::new(0), Register::new(1))
    }

    #[test]
    fn test_add_concatenates_tuples() {
        let mut vm = vm_with_frame();
        let left_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));
        let right_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::int(3).unwrap(),
            Value::int(4).unwrap(),
        ])));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(left_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(right_ptr as *const ()));

        let inst = Instruction::op_dss(
            Opcode::Add,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(add(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm
            .current_frame()
            .get_reg(0)
            .as_object_ptr()
            .expect("tuple concat should return tuple object");
        let result = unsafe { &*(result_ptr as *const TupleObject) };
        assert_eq!(result.len(), 4);
        assert_eq!(result.get(0).unwrap().as_int(), Some(1));
        assert_eq!(result.get(3).unwrap().as_int(), Some(4));

        unsafe {
            drop(Box::from_raw(left_ptr));
            drop(Box::from_raw(right_ptr));
            drop(Box::from_raw(result_ptr as *mut TupleObject));
        }
    }

    #[test]
    fn test_add_concatenates_tagged_strings() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("hello")));
        vm.current_frame_mut()
            .set_reg(2, Value::string(intern(" world")));

        let inst = Instruction::op_dss(
            Opcode::Add,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(add(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            value_to_rust_string(vm.current_frame().get_reg(0)),
            "hello world"
        );
    }

    #[test]
    fn test_add_concatenates_bytes() {
        let mut vm = vm_with_frame();
        let left_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
        let right_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"def")));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(left_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(right_ptr as *const ()));

        assert!(matches!(
            add(&mut vm, binary_inst(Opcode::Add)),
            ControlFlow::Continue
        ));

        let result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(result), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(result), b"abcdef");

        unsafe {
            drop(Box::from_raw(left_ptr));
            drop(Box::from_raw(right_ptr));
            drop(Box::from_raw(
                result
                    .as_object_ptr()
                    .expect("bytes concat result should be heap allocated")
                    as *mut BytesObject,
            ));
        }
    }

    #[test]
    fn test_add_concatenates_mixed_byte_sequences_using_left_operand_type() {
        let mut vm = vm_with_frame();
        let left_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"abc")));
        let right_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"def")));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(left_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(right_ptr as *const ()));

        assert!(matches!(
            add(&mut vm, binary_inst(Opcode::Add)),
            ControlFlow::Continue
        ));

        let first_result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(first_result), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(first_result), b"abcdef");

        let second_left_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"ghi")));
        let second_right_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"jkl")));
        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(second_left_ptr as *const ()));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(second_right_ptr as *const ()));

        assert!(matches!(
            add(&mut vm, binary_inst(Opcode::Add)),
            ControlFlow::Continue
        ));

        let second_result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(second_result), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(second_result), b"ghijkl");

        unsafe {
            drop(Box::from_raw(left_ptr));
            drop(Box::from_raw(right_ptr));
            drop(Box::from_raw(second_left_ptr));
            drop(Box::from_raw(second_right_ptr));
            drop(Box::from_raw(
                first_result
                    .as_object_ptr()
                    .expect("mixed bytes concat should allocate a result")
                    as *mut BytesObject,
            ));
            drop(Box::from_raw(
                second_result
                    .as_object_ptr()
                    .expect("mixed bytearray concat should allocate a result")
                    as *mut BytesObject,
            ));
        }
    }

    #[test]
    fn test_mul_repeats_tagged_strings() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("ab")));
        vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Mul,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            value_to_rust_string(vm.current_frame().get_reg(0)),
            "ababab"
        );
    }

    #[test]
    fn test_mul_negative_string_repeat_returns_empty() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("ab")));
        vm.current_frame_mut().set_reg(2, Value::int(-1).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Mul,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

        let result = vm.current_frame().get_reg(0);
        assert!(result.is_string());
        let ptr = result
            .as_string_object_ptr()
            .expect("empty repeat should return tagged empty string")
            as *const u8;
        assert_eq!(
            interned_by_ptr(ptr)
                .expect("empty repeat result should resolve through the interner")
                .as_str(),
            ""
        );
    }

    #[test]
    fn test_mul_repeats_tuples() {
        let mut vm = vm_with_frame();
        let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(tuple_ptr as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(2).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Mul,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm
            .current_frame()
            .get_reg(0)
            .as_object_ptr()
            .expect("tuple repeat should return tuple object");
        let result = unsafe { &*(result_ptr as *const TupleObject) };
        assert_eq!(
            result.as_slice(),
            &[
                Value::int_unchecked(1),
                Value::int_unchecked(2),
                Value::int_unchecked(1),
                Value::int_unchecked(2)
            ]
        );

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            drop(Box::from_raw(result_ptr as *mut TupleObject));
        }
    }

    #[test]
    fn test_mul_zero_repeats_tuples_to_empty() {
        let mut vm = vm_with_frame();
        let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(tuple_ptr as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(0).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Mul,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm
            .current_frame()
            .get_reg(0)
            .as_object_ptr()
            .expect("tuple repeat should return tuple object");
        let result = unsafe { &*(result_ptr as *const TupleObject) };
        assert!(result.is_empty());

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            drop(Box::from_raw(result_ptr as *mut TupleObject));
        }
    }

    #[test]
    fn test_mul_repeats_lists() {
        let mut vm = vm_with_frame();
        let list_ptr = Box::into_raw(Box::new(ListObject::from_slice(&[
            Value::int(1).unwrap(),
            Value::int(2).unwrap(),
        ])));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(list_ptr as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

        let inst = Instruction::op_dss(
            Opcode::Mul,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(mul(&mut vm, inst), ControlFlow::Continue));

        let result_ptr = vm
            .current_frame()
            .get_reg(0)
            .as_object_ptr()
            .expect("list repeat should return list object");
        let result = unsafe { &*(result_ptr as *const ListObject) };
        assert_eq!(
            result.as_slice(),
            &[
                Value::int_unchecked(1),
                Value::int_unchecked(2),
                Value::int_unchecked(1),
                Value::int_unchecked(2),
                Value::int_unchecked(1),
                Value::int_unchecked(2),
            ]
        );

        unsafe {
            drop(Box::from_raw(list_ptr));
            drop(Box::from_raw(result_ptr as *mut ListObject));
        }
    }

    #[test]
    fn test_mul_repeats_bytes() {
        let mut vm = vm_with_frame();
        let bytes_ptr = Box::into_raw(Box::new(BytesObject::from_slice(b"ab")));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(bytes_ptr as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(3).unwrap());

        assert!(matches!(
            mul(&mut vm, binary_inst(Opcode::Mul)),
            ControlFlow::Continue
        ));

        let result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(result), TypeId::BYTES);
        assert_eq!(value_to_byte_vec(result), b"ababab");

        unsafe {
            drop(Box::from_raw(bytes_ptr));
            drop(Box::from_raw(
                result
                    .as_object_ptr()
                    .expect("bytes repeat should allocate a result")
                    as *mut BytesObject,
            ));
        }
    }

    #[test]
    fn test_mul_repeats_bytearray_and_preserves_type() {
        let mut vm = vm_with_frame();
        let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xy")));

        vm.current_frame_mut().set_reg(1, Value::int(2).unwrap());
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(bytearray_ptr as *const ()));

        assert!(matches!(
            mul(&mut vm, binary_inst(Opcode::Mul)),
            ControlFlow::Continue
        ));

        let result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(result), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(result), b"xyxy");

        unsafe {
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(
                result
                    .as_object_ptr()
                    .expect("bytearray repeat should allocate a result")
                    as *mut BytesObject,
            ));
        }
    }

    #[test]
    fn test_mul_negative_byte_repeat_returns_empty_same_type() {
        let mut vm = vm_with_frame();
        let bytearray_ptr = Box::into_raw(Box::new(BytesObject::bytearray_from_slice(b"xy")));

        vm.current_frame_mut()
            .set_reg(1, Value::object_ptr(bytearray_ptr as *const ()));
        vm.current_frame_mut().set_reg(2, Value::int(-1).unwrap());

        assert!(matches!(
            mul(&mut vm, binary_inst(Opcode::Mul)),
            ControlFlow::Continue
        ));

        let result = vm.current_frame().get_reg(0);
        assert_eq!(byte_sequence_type(result), TypeId::BYTEARRAY);
        assert_eq!(value_to_byte_vec(result), b"");

        unsafe {
            drop(Box::from_raw(bytearray_ptr));
            drop(Box::from_raw(
                result
                    .as_object_ptr()
                    .expect("bytearray repeat should allocate a result")
                    as *mut BytesObject,
            ));
        }
    }

    #[test]
    fn test_modulo_formats_tagged_string_with_single_argument() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("hello %s")));
        vm.current_frame_mut()
            .set_reg(2, Value::string(intern("world")));

        let inst = Instruction::op_dss(
            Opcode::Mod,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(modulo(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            value_to_rust_string(vm.current_frame().get_reg(0)),
            "hello world"
        );
    }

    #[test]
    fn test_modulo_formats_tagged_string_with_tuple_arguments() {
        let mut vm = vm_with_frame();
        let tuple_ptr = Box::into_raw(Box::new(TupleObject::from_slice(&[
            Value::string(intern("value")),
            Value::int(7).unwrap(),
        ])));
        vm.current_frame_mut()
            .set_reg(1, Value::string(intern("%s = %d")));
        vm.current_frame_mut()
            .set_reg(2, Value::object_ptr(tuple_ptr as *const ()));

        let inst = Instruction::op_dss(
            Opcode::Mod,
            Register::new(0),
            Register::new(1),
            Register::new(2),
        );
        assert!(matches!(modulo(&mut vm, inst), ControlFlow::Continue));
        assert_eq!(
            value_to_rust_string(vm.current_frame().get_reg(0)),
            "value = 7"
        );

        unsafe {
            drop(Box::from_raw(tuple_ptr));
            if let Some(result_ptr) = vm.current_frame().get_reg(0).as_object_ptr() {
                drop(Box::from_raw(
                    result_ptr as *mut prism_runtime::types::string::StringObject,
                ));
            }
        }
    }

    #[test]
    fn test_bool_sub_uses_int_semantics() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::bool(false));

        assert!(matches!(
            sub(&mut vm, binary_inst(Opcode::Sub)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
    }

    #[test]
    fn test_bool_true_div_uses_real_number_semantics() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::int(1).unwrap());

        assert!(matches!(
            true_div(&mut vm, binary_inst(Opcode::TrueDiv)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(0).as_float(), Some(1.0));
    }

    #[test]
    fn test_bool_pow_returns_tagged_int_result() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));
        vm.current_frame_mut().set_reg(2, Value::int(2).unwrap());

        assert!(matches!(
            pow(&mut vm, binary_inst(Opcode::Pow)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(1));
    }

    #[test]
    fn test_bool_neg_returns_int() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));

        assert!(matches!(
            neg(&mut vm, unary_inst(Opcode::Neg)),
            ControlFlow::Continue
        ));
        assert_eq!(vm.current_frame().get_reg(0).as_int(), Some(-1));
    }

    #[test]
    fn test_bool_pos_int_returns_int_not_bool() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(false));

        assert!(matches!(
            pos_int(&mut vm, unary_inst(Opcode::PosInt)),
            ControlFlow::Continue
        ));
        let result = vm.current_frame().get_reg(0);
        assert_eq!(result.as_int(), Some(0));
        assert!(!result.is_bool());
    }

    #[test]
    fn test_bool_generic_pos_returns_int_not_bool() {
        let mut vm = vm_with_frame();
        vm.current_frame_mut().set_reg(1, Value::bool(true));

        assert!(matches!(
            pos(&mut vm, unary_inst(Opcode::Pos)),
            ControlFlow::Continue
        ));
        let result = vm.current_frame().get_reg(0);
        assert_eq!(result.as_int(), Some(1));
        assert!(!result.is_bool());
    }
}
