//! Arithmetic opcode handlers.
//!
//! Provides type-specialized fast paths for int/float operations,
//! with fallback to generic polymorphic operations.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Integer Arithmetic (Fast Path)
// =============================================================================

/// AddInt: dst = src1 + src2 (integers only)
#[inline(always)]
pub fn add_int(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Fast path: both are integers
    match (a.as_int(), b.as_int()) {
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

    match (a.as_int(), b.as_int()) {
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

    match (a.as_int(), b.as_int()) {
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

    match (a.as_int(), b.as_int()) {
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

    match (a.as_int(), b.as_int()) {
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

    match (a.as_int(), b.as_int()) {
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

    match a.as_int() {
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
    frame.set_reg(inst.dst().0, value);
    ControlFlow::Continue
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

/// Add: dst = src1 + src2 (generic)
#[inline(always)]
pub fn add(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Try int + int
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_add(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    // Try float + float or mixed int/float
    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("+", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("+", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x + y));
    ControlFlow::Continue
}

/// Sub: dst = src1 - src2 (generic)
#[inline(always)]
pub fn sub(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_sub(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("-", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("-", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x - y));
    ControlFlow::Continue
}

/// Mul: dst = src1 * src2 (generic)
#[inline(always)]
pub fn mul(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if let Some(result) = x.checked_mul(y) {
            if let Some(v) = Value::int(result) {
                frame.set_reg(inst.dst().0, v);
                return ControlFlow::Continue;
            }
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("*", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("*", "unknown", "unknown"));
    };

    frame.set_reg(inst.dst().0, Value::float(x * y));
    ControlFlow::Continue
}

/// TrueDiv: dst = src1 / src2 (always returns float)
#[inline(always)]
pub fn true_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("/", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("/", "unknown", "unknown"));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float(x / y));
    ControlFlow::Continue
}

/// FloorDiv: dst = src1 // src2 (generic)
#[inline(always)]
pub fn floor_div(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Int // int returns int
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
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
    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//", "unknown", "unknown",
        ));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "//", "unknown", "unknown",
        ));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    frame.set_reg(inst.dst().0, Value::float((x / y).floor()));
    ControlFlow::Continue
}

/// Mod: dst = src1 % src2 (generic)
#[inline(always)]
pub fn modulo(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        if y == 0 {
            return ControlFlow::Error(RuntimeError::zero_division());
        }
        if let Some(v) = Value::int(x.rem_euclid(y)) {
            frame.set_reg(inst.dst().0, v);
            return ControlFlow::Continue;
        }
        return ControlFlow::Error(RuntimeError::value_error("Integer overflow"));
    }

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("%", "unknown", "unknown"));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand("%", "unknown", "unknown"));
    };

    if y == 0.0 {
        return ControlFlow::Error(RuntimeError::zero_division());
    }

    let result = x - y * (x / y).floor();
    frame.set_reg(inst.dst().0, Value::float(result));
    ControlFlow::Continue
}

/// Pow: dst = src1 ** src2 (generic)
#[inline(always)]
pub fn pow(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // int ** positive int returns int
    if let (Some(base), Some(exp)) = (a.as_int(), b.as_int()) {
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

    let x = if let Some(f) = a.as_float() {
        f
    } else if let Some(i) = a.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**", "unknown", "unknown",
        ));
    };
    let y = if let Some(f) = b.as_float() {
        f
    } else if let Some(i) = b.as_int() {
        i as f64
    } else {
        return ControlFlow::Error(RuntimeError::unsupported_operand(
            "**", "unknown", "unknown",
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

    if let Some(x) = a.as_int() {
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

#[cfg(test)]
mod tests {
    // Arithmetic tests require full VM setup
}
