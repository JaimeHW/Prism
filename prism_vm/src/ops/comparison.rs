//! Comparison opcode handlers.
//!
//! Handles all comparison operations: <, <=, ==, !=, >, >=, is, in.

use crate::VirtualMachine;
use crate::dispatch::ControlFlow;
use crate::error::RuntimeError;
use prism_compiler::bytecode::Instruction;
use prism_core::Value;

// =============================================================================
// Numeric Comparisons
// =============================================================================

/// Lt: dst = src1 < src2
#[inline(always)]
pub fn lt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Int comparison
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x < y));
        return ControlFlow::Continue;
    }

    // Float comparison (including int/float mixed)
    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x < y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("<", "unknown", "unknown")),
    }
}

/// Le: dst = src1 <= src2
#[inline(always)]
pub fn le(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x <= y));
        return ControlFlow::Continue;
    }

    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x <= y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            "<=", "unknown", "unknown",
        )),
    }
}

/// Gt: dst = src1 > src2
#[inline(always)]
pub fn gt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x > y));
        return ControlFlow::Continue;
    }

    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x > y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(">", "unknown", "unknown")),
    }
}

/// Ge: dst = src1 >= src2
#[inline(always)]
pub fn ge(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x >= y));
        return ControlFlow::Continue;
    }

    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x >= y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            ">=", "unknown", "unknown",
        )),
    }
}

// =============================================================================
// Equality
// =============================================================================

/// Eq: dst = src1 == src2
#[inline(always)]
pub fn eq(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Check identical None/bool/int values
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(true));
        return ControlFlow::Continue;
    }

    if a.is_bool() && b.is_bool() {
        let result = a.as_bool() == b.as_bool();
        frame.set_reg(inst.dst().0, Value::bool(result));
        return ControlFlow::Continue;
    }

    // Int comparison
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x == y));
        return ControlFlow::Continue;
    }

    // Float comparison (including int/float mixed)
    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x == y));
            ControlFlow::Continue
        }
        _ => {
            // Different types that aren't comparable = not equal
            frame.set_reg(inst.dst().0, Value::bool(false));
            ControlFlow::Continue
        }
    }
}

/// Ne: dst = src1 != src2
#[inline(always)]
pub fn ne(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Check identical None values
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(false));
        return ControlFlow::Continue;
    }

    if a.is_bool() && b.is_bool() {
        let result = a.as_bool() != b.as_bool();
        frame.set_reg(inst.dst().0, Value::bool(result));
        return ControlFlow::Continue;
    }

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x != y));
        return ControlFlow::Continue;
    }

    let x = if let Some(f) = a.as_float() {
        Some(f)
    } else if let Some(i) = a.as_int() {
        Some(i as f64)
    } else {
        None
    };
    let y = if let Some(f) = b.as_float() {
        Some(f)
    } else if let Some(i) = b.as_int() {
        Some(i as f64)
    } else {
        None
    };

    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x != y));
            ControlFlow::Continue
        }
        _ => {
            // Different types = not equal
            frame.set_reg(inst.dst().0, Value::bool(true));
            ControlFlow::Continue
        }
    }
}

// =============================================================================
// Identity
// =============================================================================

/// Is: dst = src1 is src2
#[inline(always)]
pub fn is(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Identity check: comparing types and values
    let same = if a.is_none() && b.is_none() {
        true
    } else if a.is_bool() && b.is_bool() {
        a.as_bool() == b.as_bool()
    } else if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        x == y
    } else if a.is_object() && b.is_object() {
        a.as_object_ptr() == b.as_object_ptr()
    } else {
        false
    };
    frame.set_reg(inst.dst().0, Value::bool(same));
    ControlFlow::Continue
}

/// IsNot: dst = src1 is not src2
#[inline(always)]
pub fn is_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    // Identity check: comparing types and values
    let same = if a.is_none() && b.is_none() {
        true
    } else if a.is_bool() && b.is_bool() {
        a.as_bool() == b.as_bool()
    } else if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        x == y
    } else if a.is_object() && b.is_object() {
        a.as_object_ptr() == b.as_object_ptr()
    } else {
        false
    };
    frame.set_reg(inst.dst().0, Value::bool(!same));
    ControlFlow::Continue
}

// =============================================================================
// Membership (Stubs - require container support)
// =============================================================================

/// In: dst = src1 in src2
#[inline(always)]
pub fn in_op(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement membership testing for containers
    ControlFlow::Error(RuntimeError::internal("'in' operator not yet implemented"))
}

/// NotIn: dst = src1 not in src2
#[inline(always)]
pub fn not_in(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    // TODO: Implement membership testing for containers
    ControlFlow::Error(RuntimeError::internal(
        "'not in' operator not yet implemented",
    ))
}

// =============================================================================
// Logical/Bitwise
// =============================================================================

/// Not: dst = not src1
#[inline(always)]
pub fn not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    frame.set_reg(inst.dst().0, Value::bool(!a.is_truthy()));
    ControlFlow::Continue
}

/// BitwiseAnd: dst = src1 & src2
#[inline(always)]
pub fn bitwise_and(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x & y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        // Bool & Bool
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x && y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("&", "unknown", "unknown")),
    }
}

/// BitwiseOr: dst = src1 | src2
#[inline(always)]
pub fn bitwise_or(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x | y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x || y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("|", "unknown", "unknown")),
    }
}

/// BitwiseXor: dst = src1 ^ src2
#[inline(always)]
pub fn bitwise_xor(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) => match Value::int(x ^ y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (None, None) if a.is_bool() && b.is_bool() => {
            let x = a.is_truthy();
            let y = b.is_truthy();
            frame.set_reg(inst.dst().0, Value::bool(x != y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand("^", "unknown", "unknown")),
    }
}

/// BitwiseNot: dst = ~src1
#[inline(always)]
pub fn bitwise_not(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let a = frame.get_reg(inst.src1().0);

    match a.as_int() {
        Some(x) => match Value::int(!x) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        None => ControlFlow::Error(RuntimeError::type_error("bad operand type for unary ~")),
    }
}

/// Shl: dst = src1 << src2
#[inline(always)]
pub fn shl(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) if y >= 0 && y < 64 => match Value::int(x << y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (Some(_), Some(y)) if y < 0 => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            "<<", "unknown", "unknown",
        )),
    }
}

/// Shr: dst = src1 >> src2
#[inline(always)]
pub fn shr(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);

    match (a.as_int(), b.as_int()) {
        (Some(x), Some(y)) if y >= 0 && y < 64 => match Value::int(x >> y) {
            Some(v) => {
                frame.set_reg(inst.dst().0, v);
                ControlFlow::Continue
            }
            None => ControlFlow::Error(RuntimeError::value_error("Integer too large")),
        },
        (Some(_), Some(y)) if y < 0 => {
            ControlFlow::Error(RuntimeError::value_error("negative shift count"))
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(
            ">>", "unknown", "unknown",
        )),
    }
}

#[cfg(test)]
mod tests {
    // Comparison tests require full VM setup
}
