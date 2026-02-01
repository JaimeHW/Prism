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

/// Lt: dst = src1 < src2 (generic with speculative fast-path)
///
/// Uses O(1) speculation cache lookup for type-specialized comparison.
#[inline(always)]
pub fn lt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_lt_float, spec_lt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

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
                let (result, value) = spec_lt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_lt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
            }
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

/// Le: dst = src1 <= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn le(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_le_float, spec_le_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_le_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_le_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
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

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x <= y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
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

/// Gt: dst = src1 > src2 (generic with speculative fast-path)
#[inline(always)]
pub fn gt(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_gt_float, spec_gt_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_gt_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_gt_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
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

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x > y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
    match (x, y) {
        (Some(x), Some(y)) => {
            frame.set_reg(inst.dst().0, Value::bool(x > y));
            ControlFlow::Continue
        }
        _ => ControlFlow::Error(RuntimeError::unsupported_operand(">", "unknown", "unknown")),
    }
}

/// Ge: dst = src1 >= src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ge(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ge_float, spec_ge_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ge_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ge_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
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

    let frame = vm.current_frame_mut();
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x >= y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
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

/// Eq: dst = src1 == src2 (generic with speculative fast-path)
#[inline(always)]
pub fn eq(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_eq_float, spec_eq_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_eq_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_eq_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
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

    let frame = vm.current_frame_mut();

    // Check identical None/bool values (fast path for special types)
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(true));
        return ControlFlow::Continue;
    }
    if a.is_bool() && b.is_bool() {
        frame.set_reg(inst.dst().0, Value::bool(a.as_bool() == b.as_bool()));
        return ControlFlow::Continue;
    }

    // Int comparison
    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x == y));
        return ControlFlow::Continue;
    }

    // Float comparison
    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
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

/// Ne: dst = src1 != src2 (generic with speculative fast-path)
#[inline(always)]
pub fn ne(vm: &mut VirtualMachine, inst: Instruction) -> ControlFlow {
    use crate::ic_manager::ICSiteId;
    use crate::speculative::{SpecResult, Speculation, spec_ne_float, spec_ne_int};
    use crate::type_feedback::{BinaryOpFeedback, OperandPair};

    let frame = vm.current_frame_mut();
    let (a, b) = frame.get_regs2(inst.src1().0, inst.src2().0);
    let code_id = frame.code_id();
    let bc_offset = frame.ip.saturating_sub(1) as u32;
    let site = ICSiteId::new(code_id, bc_offset);

    // Speculative Fast Path
    if let Some(spec) = vm.speculation_cache.get(site) {
        match spec {
            Speculation::IntInt => {
                let (result, value) = spec_ne_int(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::FloatFloat | Speculation::IntFloat | Speculation::FloatInt => {
                let (result, value) = spec_ne_float(a, b);
                if result == SpecResult::Success {
                    let frame = vm.current_frame_mut();
                    frame.set_reg(inst.dst().0, value);
                    return ControlFlow::Continue;
                }
                vm.speculation_cache.invalidate(site);
            }
            Speculation::None | Speculation::StrStr | Speculation::StrInt | Speculation::IntStr => {
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

    let frame = vm.current_frame_mut();

    // Check identical None values
    if a.is_none() && b.is_none() {
        frame.set_reg(inst.dst().0, Value::bool(false));
        return ControlFlow::Continue;
    }
    if a.is_bool() && b.is_bool() {
        frame.set_reg(inst.dst().0, Value::bool(a.as_bool() != b.as_bool()));
        return ControlFlow::Continue;
    }

    if let (Some(x), Some(y)) = (a.as_int(), b.as_int()) {
        frame.set_reg(inst.dst().0, Value::bool(x != y));
        return ControlFlow::Continue;
    }

    let x = a.as_float().or_else(|| a.as_int().map(|i| i as f64));
    let y = b.as_float().or_else(|| b.as_int().map(|i| i as f64));
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
