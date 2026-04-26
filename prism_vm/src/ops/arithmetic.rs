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
use crate::ops::protocols::binary_special_method;
use crate::python_numeric::{
    complex_like_parts, float_like_value, int_like_value, is_complex_value,
};
use crate::type_feedback::BinaryOpFeedback;
use num_bigint::BigInt;
use num_traits::Zero;
use prism_code::Instruction;
use prism_core::Value;
use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use prism_runtime::types::bytes::BytesObject;
use prism_runtime::types::complex::ComplexObject;
use prism_runtime::types::int::{bigint_to_value, value_to_bigint};
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
        (Some(x), Some(y)) => {
            if let Some(result) = x.checked_add(y).and_then(Value::int) {
                frame.set_reg(inst.dst().0, result);
                return ControlFlow::Continue;
            }

            frame.set_reg(
                inst.dst().0,
                bigint_to_value(BigInt::from(x) + BigInt::from(y)),
            );
            ControlFlow::Continue
        }
        _ => match integer_bigint_operands(a, b) {
            Some((x, y)) => {
                frame.set_reg(inst.dst().0, bigint_to_value(x + y));
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::type_error("AddInt requires integers")),
        },
    }
}

/// SubInt: dst = src1 - src2
#[inline(always)]
pub fn sub_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(x), Some(y)) => {
            if let Some(result) = x.checked_sub(y).and_then(Value::int) {
                frame.set_reg(inst.dst().0, result);
                return ControlFlow::Continue;
            }

            frame.set_reg(
                inst.dst().0,
                bigint_to_value(BigInt::from(x) - BigInt::from(y)),
            );
            ControlFlow::Continue
        }
        _ => match integer_bigint_operands(a, b) {
            Some((x, y)) => {
                frame.set_reg(inst.dst().0, bigint_to_value(x - y));
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::type_error("SubInt requires integers")),
        },
    }
}

/// MulInt: dst = src1 * src2
#[inline(always)]
pub fn mul_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (int_like_value(a), int_like_value(b)) {
        (Some(x), Some(y)) => {
            if let Some(result) = x.checked_mul(y).and_then(Value::int) {
                frame.set_reg(inst.dst().0, result);
                return ControlFlow::Continue;
            }

            frame.set_reg(
                inst.dst().0,
                bigint_to_value(BigInt::from(x) * BigInt::from(y)),
            );
            ControlFlow::Continue
        }
        _ => match integer_bigint_operands(a, b) {
            Some((x, y)) => {
                frame.set_reg(inst.dst().0, bigint_to_value(x * y));
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::type_error("MulInt requires integers")),
        },
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
            if x == i64::MIN && y == -1 {
                frame.set_reg(inst.dst().0, bigint_to_value(-BigInt::from(x)));
                return ControlFlow::Continue;
            }

            let (quotient, _) = i64_floor_divmod(x, y);
            frame.set_reg(
                inst.dst().0,
                Value::int(quotient).expect("floor division result should fit in i64"),
            );
            ControlFlow::Continue
        }
        _ => match integer_bigint_operands(a, b) {
            Some((x, y)) => match bigint_floor_divmod(&x, &y) {
                Some((quotient, _)) => {
                    frame.set_reg(inst.dst().0, bigint_to_value(quotient));
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::zero_division()),
            },
            None => ControlFlow::Error(RuntimeError::type_error("FloorDivInt requires integers")),
        },
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
            if x == i64::MIN && y == -1 {
                frame.set_reg(inst.dst().0, Value::int(0).expect("zero should fit in i64"));
                return ControlFlow::Continue;
            }

            let (_, remainder) = i64_floor_divmod(x, y);
            frame.set_reg(
                inst.dst().0,
                Value::int(remainder).expect("modulo result should fit in i64"),
            );
            ControlFlow::Continue
        }
        _ => match integer_bigint_operands(a, b) {
            Some((x, y)) => match bigint_floor_divmod(&x, &y) {
                Some((_, remainder)) => {
                    frame.set_reg(inst.dst().0, bigint_to_value(remainder));
                    ControlFlow::Continue
                }
                None => ControlFlow::Error(RuntimeError::zero_division()),
            },
            None => ControlFlow::Error(RuntimeError::type_error("ModInt requires integers")),
        },
    }
}

/// PowInt: dst = src1 ** src2
#[inline(always)]
pub fn pow_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let (a, b) = {
        let frame = vm.current_frame_mut();
        frame.get_regs2(inst.src1().0, inst.src2().0)
    };

    match (int_like_value(a), int_like_value(b)) {
        (Some(base), Some(exp)) => {
            if exp < 0 {
                // Negative exponent produces float
                let result = (base as f64).powi(exp as i32);
                vm.current_frame_mut()
                    .set_reg(inst.dst().0, Value::float(result));
                return ControlFlow::Continue;
            }

            if let Ok(exp_u32) = u32::try_from(exp) {
                if let Some(result) = (base as i128).checked_pow(exp_u32) {
                    if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                        if let Some(value) = Value::int(result as i64) {
                            vm.current_frame_mut().set_reg(inst.dst().0, value);
                            return ControlFlow::Continue;
                        }
                    }
                }
            }
        }
        _ => {
            if integer_bigint_operands(a, b).is_none() {
                return ControlFlow::Error(RuntimeError::type_error("PowInt requires integers"));
            }
        }
    }

    match crate::builtins::builtin_pow_vm(vm, &[a, b]) {
        Ok(value) => {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            ControlFlow::Continue
        }
        Err(err) => ControlFlow::Error(err.into()),
    }
}

/// NegInt: dst = -src1
#[inline(always)]
pub fn neg_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match int_like_value(a) {
        Some(x) => {
            if let Some(result) = x.checked_neg().and_then(Value::int) {
                frame.set_reg(inst.dst().0, result);
                return ControlFlow::Continue;
            }

            frame.set_reg(inst.dst().0, bigint_to_value(-BigInt::from(x)));
            ControlFlow::Continue
        }
        None => match value_to_bigint(a) {
            Some(value) => {
                frame.set_reg(inst.dst().0, bigint_to_value(-value));
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::type_error("NegInt requires integers")),
        },
    }
}

/// PosInt: dst = +src1 (identity)
#[inline(always)]
pub fn pos_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let value = frame.get_reg(inst.src1().0);
    match value.as_bool() {
        Some(boolean) => {
            let result = Value::int(i64::from(boolean)).expect("bool coerces to inline int");
            frame.set_reg(inst.dst().0, result);
            ControlFlow::Continue
        }
        None => match value_to_bigint(value) {
            Some(_) => {
                frame.set_reg(inst.dst().0, value);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::type_error("PosInt requires integers")),
        },
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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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
        if let Some(value) = x.checked_add(y).and_then(Value::int) {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            return ControlFlow::Continue;
        }
    }

    if let Some((x, y)) = integer_bigint_operands(a, b) {
        vm.current_frame_mut()
            .set_reg(inst.dst().0, bigint_to_value(x + y));
        return ControlFlow::Continue;
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
            vm.current_frame_mut()
                .set_reg(inst.dst().0, boxed_tuple_value(tuple));
            return ControlFlow::Continue;
        }
    }

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__add__", "__radd__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
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
    _vm: &VirtualMachine,
    string: StringObject,
) -> Result<Value, RuntimeError> {
    Ok(alloc_value_in_current_heap_or_box(string))
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
    crate::alloc_managed_value(ComplexObject::new(real, imag))
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
fn integer_like_bigint(value: Value) -> Option<BigInt> {
    value
        .as_bool()
        .map(|boolean| BigInt::from(u8::from(boolean)))
        .or_else(|| value_to_bigint(value))
}

#[inline]
fn integer_bigint_operands(left: Value, right: Value) -> Option<(BigInt, BigInt)> {
    Some((integer_like_bigint(left)?, integer_like_bigint(right)?))
}

#[inline]
fn bigint_floor_divmod(left: &BigInt, right: &BigInt) -> Option<(BigInt, BigInt)> {
    if right.is_zero() {
        return None;
    }

    let mut quotient = left / right;
    let mut remainder = left % right;
    if !remainder.is_zero() && remainder.sign() != right.sign() {
        quotient -= 1;
        remainder += right;
    }

    Some((quotient, remainder))
}

#[inline]
fn i64_floor_divmod(left: i64, right: i64) -> (i64, i64) {
    let mut quotient = left / right;
    let mut remainder = left % right;
    if remainder != 0 && remainder.signum() != right.signum() {
        quotient -= 1;
        remainder += right;
    }
    (quotient, remainder)
}

#[inline]
fn try_binary_special_method_result(
    vm: &mut VirtualMachine,
    dst: u8,
    left: Value,
    right: Value,
    left_method: &'static str,
    right_method: &'static str,
) -> Result<bool, RuntimeError> {
    if let Some(value) = binary_special_method(vm, left, right, left_method, right_method)? {
        vm.current_frame_mut().set_reg(dst, value);
        return Ok(true);
    }

    Ok(false)
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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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

    if let (Some(x), Some(y)) = (int_like_value(a), int_like_value(b)) {
        if let Some(value) = x.checked_sub(y).and_then(Value::int) {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            return ControlFlow::Continue;
        }
    }

    if let Some((x, y)) = integer_bigint_operands(a, b) {
        vm.current_frame_mut()
            .set_reg(inst.dst().0, bigint_to_value(x - y));
        return ControlFlow::Continue;
    }

    if let Some(value) = try_sub_complex_values(a, b) {
        vm.current_frame_mut().set_reg(inst.dst().0, value);
        return ControlFlow::Continue;
    }

    if let Some((left, right, result_type)) = crate::ops::comparison::set_binary_operands(a, b) {
        let value = crate::ops::comparison::boxed_set_result(left.difference(right), result_type);
        vm.current_frame_mut().set_reg(inst.dst().0, value);
        return ControlFlow::Continue;
    }

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__sub__", "__rsub__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    let frame = vm.current_frame_mut();

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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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
        if let Some(value) = x.checked_mul(y).and_then(Value::int) {
            vm.current_frame_mut().set_reg(inst.dst().0, value);
            return ControlFlow::Continue;
        }
    }

    if let Some((x, y)) = integer_bigint_operands(a, b) {
        vm.current_frame_mut()
            .set_reg(inst.dst().0, bigint_to_value(x * y));
        return ControlFlow::Continue;
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

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__mul__", "__rmul__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
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
    crate::alloc_managed_value(list)
}

#[inline]
fn boxed_tuple_value(tuple: TupleObject) -> Value {
    crate::alloc_managed_value(tuple)
}

#[inline]
fn boxed_bytes_value(bytes: BytesObject) -> Value {
    crate::alloc_managed_value(bytes)
}

/// TrueDiv: dst = src1 / src2 (always returns float, with speculative fast-path)
///
/// Uses O(1) speculation cache lookup to select optimized code path.
#[inline(always)]
pub fn true_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_div_float};
    use crate::type_feedback::OperandPair;

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__truediv__", "__rtruediv__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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
        if x == i64::MIN && y == -1 {
            frame.set_reg(inst.dst().0, bigint_to_value(-BigInt::from(x)));
            return ControlFlow::Continue;
        }
        let (quotient, _) = i64_floor_divmod(x, y);
        frame.set_reg(
            inst.dst().0,
            Value::int(quotient).expect("floor division result should fit in i64"),
        );
        return ControlFlow::Continue;
    }

    if let Some((x, y)) = integer_bigint_operands(a, b) {
        let Some((quotient, _)) = bigint_floor_divmod(&x, &y) else {
            return ControlFlow::Error(RuntimeError::zero_division());
        };
        frame.set_reg(inst.dst().0, bigint_to_value(quotient));
        return ControlFlow::Continue;
    }

    let _ = frame;

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__floordiv__", "__rfloordiv__")
    {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    let frame = vm.current_frame_mut();

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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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

    if let Some(template) = value_as_byte_sequence_ref(a) {
        match crate::builtins::percent_format_bytes(template, b) {
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
        if x == i64::MIN && y == -1 {
            frame.set_reg(inst.dst().0, Value::int(0).expect("zero should fit in i64"));
            return ControlFlow::Continue;
        }
        let (_, remainder) = i64_floor_divmod(x, y);
        frame.set_reg(
            inst.dst().0,
            Value::int(remainder).expect("modulo result should fit in i64"),
        );
        return ControlFlow::Continue;
    }

    if let Some((x, y)) = integer_bigint_operands(a, b) {
        let Some((_, remainder)) = bigint_floor_divmod(&x, &y) else {
            return ControlFlow::Error(RuntimeError::zero_division());
        };
        frame.set_reg(inst.dst().0, bigint_to_value(remainder));
        return ControlFlow::Continue;
    }

    let _ = frame;

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__mod__", "__rmod__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
    }

    let frame = vm.current_frame_mut();

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

    let (a, b, code_id, bc_offset) = {
        let frame = vm.current_frame();
        let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
        (a, b, frame.code_id(), frame.ip.saturating_sub(1) as u32)
    };
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

    // int ** positive int returns int
    if let (Some(base), Some(exp)) = (int_like_value(a), int_like_value(b)) {
        if exp >= 0 && exp <= 63 {
            if let Some(result) = (base as i128).checked_pow(exp as u32) {
                if result >= i64::MIN as i128 && result <= i64::MAX as i128 {
                    if let Some(value) = Value::int(result as i64) {
                        vm.current_frame_mut().set_reg(inst.dst().0, value);
                        return ControlFlow::Continue;
                    }
                }
            }
        }
    }

    if integer_bigint_operands(a, b).is_some() {
        match crate::builtins::builtin_pow_vm(vm, &[a, b]) {
            Ok(value) => {
                vm.current_frame_mut().set_reg(inst.dst().0, value);
                return ControlFlow::Continue;
            }
            Err(err) => return ControlFlow::Error(err.into()),
        }
    }

    match try_binary_special_method_result(vm, inst.dst().0, a, b, "__pow__", "__rpow__") {
        Ok(true) => return ControlFlow::Continue,
        Ok(false) => {}
        Err(err) => return ControlFlow::Error(err),
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

    vm.current_frame_mut()
        .set_reg(inst.dst().0, Value::float(x.powf(y)));
    ControlFlow::Continue
}

/// Neg: dst = -src1 (generic)
#[inline(always)]
pub fn neg(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let value = vm.current_frame().get_reg(inst.src1().0);
    let result = match neg_value(vm, value) {
        Ok(result) => result,
        Err(err) => return ControlFlow::Error(err),
    };

    vm.current_frame_mut().set_reg(inst.dst().0, result);
    ControlFlow::Continue
}

#[inline]
fn neg_value(vm: &mut VirtualMachine, value: Value) -> Result<Value, RuntimeError> {
    if let Some(x) = int_like_value(value) {
        if let Some(value) = x.checked_neg().and_then(Value::int) {
            return Ok(value);
        }

        return Ok(bigint_to_value(-BigInt::from(x)));
    }

    if let Some(value) = value_to_bigint(value) {
        return Ok(bigint_to_value(-value));
    }

    if let Some(x) = value.as_float() {
        return Ok(Value::float(-x));
    }

    if is_complex_value(value) {
        let parts = complex_like_parts(value)
            .ok_or_else(|| RuntimeError::type_error("bad operand type for unary -"))?;
        return Ok(boxed_complex_value(-parts.real, -parts.imag));
    }

    let target = match resolve_special_method(value, "__neg__") {
        Ok(target) => target,
        Err(err) if matches!(err.kind, RuntimeErrorKind::AttributeError { .. }) => {
            return Err(RuntimeError::type_error(format!(
                "bad operand type for unary -: '{}'",
                value.type_name()
            )));
        }
        Err(err) => return Err(err),
    };

    invoke_zero_arg_bound_method(vm, target)
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
    if let Some(boolean) = value.as_bool() {
        return Value::int(i64::from(boolean))
            .ok_or_else(|| RuntimeError::value_error("Integer too large for i48"));
    }

    if value_to_bigint(value).is_some() {
        return Ok(value);
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
