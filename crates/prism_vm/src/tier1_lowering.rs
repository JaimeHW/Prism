//! Shared Tier 1 lowering utilities for VM-side JIT entry points.
//!
//! This module centralizes bytecode -> template lowering so sync and async
//! compilation paths use identical semantics.

use prism_code::{CodeFlags, CodeObject, Constant, Opcode};
use prism_jit::tier1::codegen::TemplateInstruction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnownType {
    Unknown,
    None,
    Bool,
    Int,
    Float,
    List,
    Tuple,
    Set,
    Dict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericKind {
    Int,
    Float,
    Fallback,
}

#[inline]
fn get_reg_type(reg_types: &[KnownType], reg: u8) -> KnownType {
    reg_types
        .get(reg as usize)
        .copied()
        .unwrap_or(KnownType::Unknown)
}

#[inline]
fn set_reg_type(reg_types: &mut [KnownType], reg: u8, ty: KnownType) {
    if let Some(slot) = reg_types.get_mut(reg as usize) {
        *slot = ty;
    }
}

fn initial_local_written(code: &CodeObject) -> Vec<bool> {
    let local_count = code
        .locals
        .len()
        .max(code.arg_count as usize + code.kwonlyarg_count as usize)
        .min(256);
    let mut written = vec![false; local_count];
    let initialized_params =
        (code.arg_count as usize + code.kwonlyarg_count as usize).min(local_count);
    for slot in &mut written[..initialized_params] {
        *slot = true;
    }
    written
}

#[inline]
fn ensure_local_capacity<T: Copy>(locals: &mut Vec<T>, slot: u16, default: T) {
    let needed = slot as usize + 1;
    if locals.len() < needed {
        locals.resize(needed, default);
    }
}

#[inline]
fn local_is_definitely_written(local_written: &[bool], slot: u16) -> bool {
    local_written.get(slot as usize).copied().unwrap_or(false)
}

#[inline]
fn set_local_written(local_written: &mut Vec<bool>, slot: u16, written: bool) {
    ensure_local_capacity(local_written, slot, false);
    local_written[slot as usize] = written;
}

#[inline]
fn get_local_type(local_types: &[KnownType], slot: u16) -> KnownType {
    local_types
        .get(slot as usize)
        .copied()
        .unwrap_or(KnownType::Unknown)
}

#[inline]
fn set_local_type(local_types: &mut Vec<KnownType>, slot: u16, ty: KnownType) {
    ensure_local_capacity(local_types, slot, KnownType::Unknown);
    local_types[slot as usize] = ty;
}

#[inline]
fn interpreter_fallback(bc_offset: u32) -> TemplateInstruction {
    TemplateInstruction::InterpreterFallback { bc_offset }
}

fn infer_numeric_kind(
    op: Opcode,
    bc_offset: u32,
    lhs: u8,
    rhs: u8,
    reg_types: &[KnownType],
) -> NumericKind {
    let lhs_ty = get_reg_type(reg_types, lhs);
    let rhs_ty = get_reg_type(reg_types, rhs);
    match (lhs_ty, rhs_ty) {
        (KnownType::Int, KnownType::Int) => NumericKind::Int,
        (KnownType::Float, KnownType::Float) => NumericKind::Float,
        _ => {
            let _ = (op, bc_offset, lhs, lhs_ty, rhs, rhs_ty);
            NumericKind::Fallback
        }
    }
}

/// Lower a code object to template IR for Tier 1 compilation.
///
/// This performs conservative lowering:
/// - safe bytecodes lower to concrete native templates
/// - unsupported or semantically complex bytecodes lower to an explicit
///   interpreter fallback at the same bytecode offset
/// - every jump target must resolve to a valid bytecode offset
pub(crate) fn lower_code_to_templates(
    code: &CodeObject,
) -> Result<Vec<TemplateInstruction>, String> {
    let mut templates = Vec::with_capacity(code.instructions.len());
    let mut reg_types = vec![KnownType::Unknown; code.register_count as usize];
    let mut local_written = initial_local_written(code);
    let mut local_types = vec![KnownType::Unknown; local_written.len()];
    let class_local_semantics = code.flags.contains(CodeFlags::CLASS);

    for (index, inst) in code.instructions.iter().enumerate() {
        let bc_offset = (index as u32) * 4;
        let op = Opcode::from_u8(inst.opcode())
            .ok_or_else(|| format!("invalid opcode byte {} at index {}", inst.opcode(), index))?;

        let template = match op {
            Opcode::Nop => TemplateInstruction::Nop { bc_offset },

            Opcode::LoadConst => {
                let dst = inst.dst().0;
                let (template, ty) =
                    load_const_template(code, bc_offset, dst, inst.imm16(), index)?;
                set_reg_type(&mut reg_types, dst, ty);
                template
            }
            Opcode::LoadBuiltin => interpreter_fallback(bc_offset),
            Opcode::LoadGlobal => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                if class_local_semantics {
                    interpreter_fallback(bc_offset)
                } else {
                    TemplateInstruction::LoadGlobal {
                        bc_offset,
                        dst,
                        name_idx: inst.imm16(),
                        helper_addr: crate::jit_runtime_helpers::tier1_load_global_addr(),
                    }
                }
            }
            Opcode::LoadNone => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::None);
                TemplateInstruction::LoadNone { bc_offset, dst }
            }
            Opcode::LoadTrue => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LoadBool {
                    bc_offset,
                    dst,
                    value: true,
                }
            }
            Opcode::LoadFalse => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LoadBool {
                    bc_offset,
                    dst,
                    value: false,
                }
            }
            Opcode::Move => {
                let dst = inst.dst().0;
                let src = inst.src1().0;
                let src_ty = get_reg_type(&reg_types, src);
                set_reg_type(&mut reg_types, dst, src_ty);
                TemplateInstruction::Move {
                    bc_offset,
                    dst,
                    src,
                }
            }
            Opcode::GetAttr => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                TemplateInstruction::GetAttr {
                    bc_offset,
                    dst,
                    obj: inst.src1().0,
                    name_idx: inst.src2().0,
                    ic_site_idx: None,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::SetAttr => TemplateInstruction::SetAttr {
                bc_offset,
                obj: inst.dst().0,
                name_idx: inst.src1().0,
                value: inst.src2().0,
                ic_site_idx: None,
                helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
            },
            Opcode::LoadMethod => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                set_reg_type(&mut reg_types, dst.saturating_add(1), KnownType::Unknown);
                TemplateInstruction::LoadMethod {
                    bc_offset,
                    dst,
                    obj: inst.src1().0,
                    name_idx: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::AttrName => TemplateInstruction::Nop { bc_offset },
            Opcode::LoadLocal => {
                let dst = inst.dst().0;
                let slot = inst.imm16();
                if class_local_semantics || !local_is_definitely_written(&local_written, slot) {
                    set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                    interpreter_fallback(bc_offset)
                } else {
                    let ty = get_local_type(&local_types, slot);
                    set_reg_type(&mut reg_types, dst, ty);
                    TemplateInstruction::LoadLocal {
                        bc_offset,
                        dst,
                        slot,
                    }
                }
            }
            Opcode::StoreLocal => {
                let src = inst.dst().0;
                let slot = inst.imm16();
                if class_local_semantics {
                    interpreter_fallback(bc_offset)
                } else {
                    set_local_written(&mut local_written, slot, true);
                    set_local_type(&mut local_types, slot, get_reg_type(&reg_types, src));
                    TemplateInstruction::StoreLocal {
                        bc_offset,
                        src,
                        slot,
                    }
                }
            }
            Opcode::DeleteLocal => {
                let slot = inst.imm16();
                if class_local_semantics {
                    interpreter_fallback(bc_offset)
                } else {
                    set_local_written(&mut local_written, slot, false);
                    set_local_type(&mut local_types, slot, KnownType::Unknown);
                    TemplateInstruction::DeleteLocal { bc_offset, slot }
                }
            }

            Opcode::AddInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Add => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntAdd {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatAdd {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::SubInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Sub => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntSub {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatSub {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::MulInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Mul => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntMul {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatMul {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::FloorDivInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::FloorDivFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatFloorDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::FloorDiv => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatFloorDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::ModInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntMod {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::ModFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatMod {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Mod => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => {
                        set_reg_type(&mut reg_types, dst, KnownType::Int);
                        TemplateInstruction::IntMod {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatMod {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::NegInt => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntNeg {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }
            Opcode::AddFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::SubFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatSub {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::MulFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatMul {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::DivFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::TrueDiv => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Float => {
                        set_reg_type(&mut reg_types, dst, KnownType::Float);
                        TemplateInstruction::FloatDiv {
                            bc_offset,
                            dst,
                            lhs,
                            rhs,
                        }
                    }
                    NumericKind::Int | NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::NegFloat => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Float);
                TemplateInstruction::FloatNeg {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }

            Opcode::Lt => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntLt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatLt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Le => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntLe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatLe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Eq => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntEq {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatEq {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Ne => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntNe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatNe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Gt => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntGt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatGt {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Ge => {
                let dst = inst.dst().0;
                let lhs = inst.src1().0;
                let rhs = inst.src2().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                match infer_numeric_kind(op, bc_offset, lhs, rhs, &reg_types) {
                    NumericKind::Int => TemplateInstruction::IntGe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Float => TemplateInstruction::FloatGe {
                        bc_offset,
                        dst,
                        lhs,
                        rhs,
                    },
                    NumericKind::Fallback => {
                        set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                        interpreter_fallback(bc_offset)
                    }
                }
            }
            Opcode::Is => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::Is {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::IsNot => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::IsNot {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseAnd => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntAnd {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseOr => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntOr {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseXor => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntXor {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::BitwiseNot => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntNot {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }
            Opcode::Shl => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntShl {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Shr => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                TemplateInstruction::IntShr {
                    bc_offset,
                    dst,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                }
            }
            Opcode::Not => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::LogicalNot {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                }
            }
            Opcode::GetItem => {
                let dst = inst.dst().0;
                let container = inst.src1().0;
                let key = inst.src2().0;
                let container_ty = get_reg_type(&reg_types, container);
                let key_ty = get_reg_type(&reg_types, key);
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);

                match (container_ty, key_ty) {
                    (KnownType::List, KnownType::Int) => TemplateInstruction::ListGetItem {
                        bc_offset,
                        dst,
                        list: container,
                        index: key,
                    },
                    (KnownType::Tuple, KnownType::Int) => TemplateInstruction::TupleGetItem {
                        bc_offset,
                        dst,
                        tuple: container,
                        index: key,
                    },
                    _ => TemplateInstruction::GetItem {
                        bc_offset,
                        dst,
                        container,
                        key,
                        helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                    },
                }
            }
            Opcode::SetItem => {
                let container = inst.src1().0;
                let key = inst.dst().0;
                let value = inst.src2().0;
                match (
                    get_reg_type(&reg_types, container),
                    get_reg_type(&reg_types, key),
                ) {
                    (KnownType::List, KnownType::Int) => TemplateInstruction::ListSetItem {
                        bc_offset,
                        list: container,
                        index: key,
                        value,
                    },
                    _ => TemplateInstruction::SetItem {
                        bc_offset,
                        container,
                        key,
                        value,
                        helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                    },
                }
            }
            Opcode::DelItem => TemplateInstruction::DelItem {
                bc_offset,
                container: inst.src1().0,
                key: inst.src2().0,
                helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
            },
            Opcode::Len => {
                let dst = inst.dst().0;
                let src = inst.src1().0;
                set_reg_type(&mut reg_types, dst, KnownType::Int);
                match get_reg_type(&reg_types, src) {
                    KnownType::List => TemplateInstruction::ListLen {
                        bc_offset,
                        dst,
                        list: src,
                    },
                    KnownType::Tuple => TemplateInstruction::TupleLen {
                        bc_offset,
                        dst,
                        tuple: src,
                    },
                    _ => TemplateInstruction::Len {
                        bc_offset,
                        dst,
                        src,
                        helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                    },
                }
            }
            Opcode::IsCallable => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Bool);
                TemplateInstruction::IsCallable {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::GetIter => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                TemplateInstruction::GetIter {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::ForIter => {
                let dst = inst.dst().0;
                let iter = dst.checked_sub(1).ok_or_else(|| {
                    "ForIter destination register must follow its iterator register".to_string()
                })?;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                TemplateInstruction::ForIter {
                    bc_offset,
                    dst,
                    iter,
                    target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::BuildList => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::List);
                TemplateInstruction::BuildList {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_build_list_addr(),
                }
            }
            Opcode::BuildTuple => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Tuple);
                TemplateInstruction::BuildTuple {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_build_tuple_addr(),
                }
            }
            Opcode::BuildSet => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Set);
                TemplateInstruction::BuildSet {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_build_set_addr(),
                }
            }
            Opcode::BuildDict => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Dict);
                TemplateInstruction::BuildDict {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_build_dict_addr(),
                }
            }
            Opcode::BuildString => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                TemplateInstruction::BuildString {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::BuildSlice => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                TemplateInstruction::BuildSlice {
                    bc_offset,
                    dst,
                    start: inst.src1().0,
                    stop: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::ListAppend => {
                let list = inst.src1().0;
                let value = inst.src2().0;
                if get_reg_type(&reg_types, list) == KnownType::List {
                    TemplateInstruction::ListAppendFast {
                        bc_offset,
                        list,
                        value,
                    }
                } else {
                    TemplateInstruction::ListAppend {
                        bc_offset,
                        list,
                        value,
                        helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                    }
                }
            }
            Opcode::SetAdd => TemplateInstruction::SetAdd {
                bc_offset,
                set: inst.src1().0,
                value: inst.src2().0,
                helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
            },
            Opcode::DictSet => TemplateInstruction::DictSet {
                bc_offset,
                dict: inst.src1().0,
                key: inst.dst().0,
                value: inst.src2().0,
                helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
            },
            Opcode::UnpackSequence => {
                let dst = inst.dst().0;
                for offset in 0..inst.src2().0 {
                    set_reg_type(
                        &mut reg_types,
                        dst.saturating_add(offset),
                        KnownType::Unknown,
                    );
                }
                TemplateInstruction::UnpackSequence {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                    count: inst.src2().0,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }
            Opcode::UnpackEx => {
                let dst = inst.dst().0;
                let counts = inst.src2().0;
                let before = counts >> 4;
                let after = counts & 0x0F;
                for offset in 0..before.saturating_add(1).saturating_add(after) {
                    set_reg_type(
                        &mut reg_types,
                        dst.saturating_add(offset),
                        KnownType::Unknown,
                    );
                }
                TemplateInstruction::UnpackEx {
                    bc_offset,
                    dst,
                    src: inst.src1().0,
                    before,
                    after,
                    helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                }
            }

            Opcode::Call => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                if class_local_semantics {
                    interpreter_fallback(bc_offset)
                } else {
                    TemplateInstruction::Call {
                        bc_offset,
                        dst,
                        func: inst.src1().0,
                        argc: inst.src2().0,
                        helper_addr: crate::jit_runtime_helpers::tier1_call_addr(),
                    }
                }
            }
            Opcode::CallMethod => {
                let dst = inst.dst().0;
                set_reg_type(&mut reg_types, dst, KnownType::Unknown);
                if class_local_semantics {
                    interpreter_fallback(bc_offset)
                } else {
                    TemplateInstruction::CallMethod {
                        bc_offset,
                        dst,
                        method: inst.src1().0,
                        argc: inst.src2().0,
                        helper_addr: crate::jit_runtime_helpers::tier1_bytecode_addr(),
                    }
                }
            }

            Opcode::Jump => TemplateInstruction::Jump {
                bc_offset,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfTrue => TemplateInstruction::BranchIfTrue {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfFalse => TemplateInstruction::BranchIfFalse {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfNone => TemplateInstruction::BranchIfNone {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::JumpIfNotNone => TemplateInstruction::BranchIfNotNone {
                bc_offset,
                cond: inst.dst().0,
                target: calculate_jump_target(bc_offset, inst.imm16() as i16)?,
            },
            Opcode::Return => TemplateInstruction::Return {
                bc_offset,
                value: inst.dst().0,
            },
            Opcode::ReturnNone => TemplateInstruction::ReturnNone { bc_offset },

            _ => interpreter_fallback(bc_offset),
        };

        templates.push(template);
    }

    validate_jump_targets(code, &templates)?;
    Ok(templates)
}

fn load_const_template(
    code: &CodeObject,
    bc_offset: u32,
    dst: u8,
    const_idx: u16,
    inst_index: usize,
) -> Result<(TemplateInstruction, KnownType), String> {
    let idx = const_idx as usize;
    let value = code.constants.get(idx).ok_or_else(|| {
        format!(
            "invalid constant index {} at instruction {} ({} constants)",
            const_idx,
            inst_index,
            code.constants.len()
        )
    })?;

    match value {
        Constant::Value(value) if value.is_none() => Ok((
            TemplateInstruction::LoadNone { bc_offset, dst },
            KnownType::None,
        )),
        Constant::Value(value) if value.as_bool().is_some() => {
            let b = value.as_bool().expect("guarded by is_some");
            Ok((
                TemplateInstruction::LoadBool {
                    bc_offset,
                    dst,
                    value: b,
                },
                KnownType::Bool,
            ))
        }
        Constant::Value(value) if value.as_int().is_some() => {
            let i = value.as_int().expect("guarded by is_some");
            Ok((
                TemplateInstruction::LoadInt {
                    bc_offset,
                    dst,
                    value: i,
                },
                KnownType::Int,
            ))
        }
        Constant::Value(value) if value.as_float().is_some() => {
            let f = value.as_float().expect("guarded by is_some");
            Ok((
                TemplateInstruction::LoadFloat {
                    bc_offset,
                    dst,
                    value: f,
                },
                KnownType::Float,
            ))
        }
        Constant::Value(_) | Constant::BigInt(_) => {
            Ok((interpreter_fallback(bc_offset), KnownType::Unknown))
        }
    }
}

fn calculate_jump_target(bc_offset: u32, relative: i16) -> Result<u32, String> {
    let target = bc_offset as i64 + 4 + (relative as i64 * 4);
    if !(0..=u32::MAX as i64).contains(&target) {
        return Err(format!(
            "jump target overflow from bytecode offset {} with delta {}",
            bc_offset, relative
        ));
    }
    Ok(target as u32)
}

fn validate_jump_targets(
    code: &CodeObject,
    templates: &[TemplateInstruction],
) -> Result<(), String> {
    let max_offset = (code.instructions.len() as u32) * 4;
    for template in templates {
        if let Some(target) = jump_target(template) {
            if target % 4 != 0 || target > max_offset {
                return Err(format!(
                    "invalid jump target {} from bytecode offset {} (max {})",
                    target,
                    template.bc_offset(),
                    max_offset
                ));
            }
        }
    }
    Ok(())
}

#[inline]
fn jump_target(template: &TemplateInstruction) -> Option<u32> {
    match template {
        TemplateInstruction::Jump { target, .. }
        | TemplateInstruction::BranchIfTrue { target, .. }
        | TemplateInstruction::BranchIfFalse { target, .. }
        | TemplateInstruction::BranchIfNone { target, .. }
        | TemplateInstruction::BranchIfNotNone { target, .. } => Some(*target),
        TemplateInstruction::EndAsyncFor { target, .. } => Some(*target as u32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_code::{FunctionBuilder, Instruction, Register};

    fn load_global_code(flags: CodeFlags) -> CodeObject {
        let mut builder = FunctionBuilder::new("load_global");
        builder.add_flags(flags);
        let name_idx = builder.add_name("answer");
        builder.emit_load_global(Register::new(0), name_idx);
        builder.emit_return(Register::new(0));
        builder.finish()
    }

    fn call_code(flags: CodeFlags) -> CodeObject {
        let mut builder = FunctionBuilder::new("call");
        builder.add_flags(flags);
        builder.emit_call(Register::new(0), Register::new(3), 2);
        builder.emit_return(Register::new(0));
        builder.finish()
    }

    fn get_attr_code(name_count_before_target: usize) -> CodeObject {
        let mut builder = FunctionBuilder::new("get_attr");
        for idx in 0..name_count_before_target {
            builder.add_name(format!("dummy_{idx}"));
        }
        let name_idx = builder.add_name("target");
        builder.emit_get_attr(Register::new(1), Register::new(0), name_idx);
        builder.emit_return(Register::new(1));
        builder.finish()
    }

    fn call_method_code(flags: CodeFlags) -> CodeObject {
        let mut builder = FunctionBuilder::new("call_method");
        builder.add_flags(flags);
        builder.emit_call_method(Register::new(0), Register::new(3), 2);
        builder.emit_return(Register::new(0));
        builder.finish()
    }

    fn len_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("len");
        builder.emit(Instruction::op_ds(
            Opcode::Len,
            Register::new(1),
            Register::new(0),
        ));
        builder.emit_return(Register::new(1));
        builder.finish()
    }

    fn get_iter_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("get_iter");
        builder.emit_get_iter(Register::new(1), Register::new(0));
        builder.emit_return(Register::new(1));
        builder.finish()
    }

    fn for_iter_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("for_iter");
        builder.emit(Instruction::op_di(Opcode::ForIter, Register::new(1), 1));
        builder.emit_return(Register::new(1));
        builder.finish()
    }

    fn build_list_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("build_list");
        builder.emit_build_list(Register::new(3), Register::new(0), 3);
        builder.emit_return(Register::new(3));
        builder.finish()
    }

    fn build_tuple_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("build_tuple");
        builder.emit_build_tuple(Register::new(3), Register::new(0), 3);
        builder.emit_return(Register::new(3));
        builder.finish()
    }

    fn build_set_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("build_set");
        builder.emit(Instruction::new(Opcode::BuildSet, 3, 0, 3));
        builder.emit_return(Register::new(3));
        builder.finish()
    }

    fn build_dict_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("build_dict");
        builder.emit(Instruction::new(Opcode::BuildDict, 4, 0, 2));
        builder.emit_return(Register::new(4));
        builder.finish()
    }

    fn list_get_item_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("list_get_item");
        builder.reserve_parameters(5);
        let first = builder.add_int(10);
        let second = builder.add_int(20);
        let index = builder.add_int(1);
        builder.emit_load_const(Register::new(0), first);
        builder.emit_load_const(Register::new(1), second);
        builder.emit_build_list(Register::new(2), Register::new(0), 2);
        builder.emit_load_const(Register::new(3), index);
        builder.emit_get_item(Register::new(4), Register::new(2), Register::new(3));
        builder.emit_return(Register::new(4));
        builder.finish()
    }

    fn tuple_get_item_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("tuple_get_item");
        builder.reserve_parameters(5);
        let first = builder.add_int(10);
        let second = builder.add_int(20);
        let index = builder.add_int(0);
        builder.emit_load_const(Register::new(0), first);
        builder.emit_load_const(Register::new(1), second);
        builder.emit_build_tuple(Register::new(2), Register::new(0), 2);
        builder.emit_load_const(Register::new(3), index);
        builder.emit_get_item(Register::new(4), Register::new(2), Register::new(3));
        builder.emit_return(Register::new(4));
        builder.finish()
    }

    fn list_set_item_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("list_set_item");
        builder.reserve_parameters(5);
        let first = builder.add_int(10);
        let second = builder.add_int(20);
        let index = builder.add_int(1);
        let value = builder.add_int(30);
        builder.emit_load_const(Register::new(0), first);
        builder.emit_load_const(Register::new(1), second);
        builder.emit_build_list(Register::new(2), Register::new(0), 2);
        builder.emit_load_const(Register::new(3), index);
        builder.emit_load_const(Register::new(4), value);
        builder.emit_set_item(Register::new(2), Register::new(3), Register::new(4));
        builder.emit_return(Register::new(2));
        builder.finish()
    }

    fn list_len_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("list_len");
        builder.reserve_parameters(4);
        builder.emit_build_list(Register::new(2), Register::new(0), 0);
        builder.emit(Instruction::op_ds(
            Opcode::Len,
            Register::new(3),
            Register::new(2),
        ));
        builder.emit_return(Register::new(3));
        builder.finish()
    }

    fn tuple_len_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("tuple_len");
        builder.reserve_parameters(4);
        builder.emit_build_tuple(Register::new(2), Register::new(0), 0);
        builder.emit(Instruction::op_ds(
            Opcode::Len,
            Register::new(3),
            Register::new(2),
        ));
        builder.emit_return(Register::new(3));
        builder.finish()
    }

    fn list_append_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("list_append");
        builder.reserve_parameters(4);
        let value = builder.add_int(30);
        builder.emit_build_list(Register::new(2), Register::new(0), 0);
        builder.emit_load_const(Register::new(3), value);
        builder.emit(Instruction::op_dss(
            Opcode::ListAppend,
            Register::new(0),
            Register::new(2),
            Register::new(3),
        ));
        builder.emit_return(Register::new(2));
        builder.finish()
    }

    fn dict_set_code() -> CodeObject {
        let mut builder = FunctionBuilder::new("dict_set");
        builder.emit(Instruction::op_dss(
            Opcode::DictSet,
            Register::new(2),
            Register::new(0),
            Register::new(1),
        ));
        builder.emit_return(Register::new(0));
        builder.finish()
    }

    #[test]
    fn normal_load_global_lowers_to_runtime_helper_template() {
        let code = load_global_code(CodeFlags::NONE);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::LoadGlobal {
                bc_offset,
                dst,
                name_idx,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 0);
                assert_eq!(*name_idx, 0);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected LoadGlobal template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
    }

    #[test]
    fn class_body_load_global_keeps_interpreter_semantics() {
        let code = load_global_code(CodeFlags::CLASS);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        assert!(matches!(
            templates[0],
            TemplateInstruction::InterpreterFallback { bc_offset: 0 }
        ));
        assert!(templates[0].requires_interpreter_in_tier1());
    }

    #[test]
    fn normal_call_lowers_to_runtime_helper_template() {
        let code = call_code(CodeFlags::NONE);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::Call {
                bc_offset,
                dst,
                func,
                argc,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 0);
                assert_eq!(*func, 3);
                assert_eq!(*argc, 2);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected Call template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_get_attr_lowers_to_runtime_helper_template() {
        let code = get_attr_code(0);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::GetAttr {
                bc_offset,
                dst,
                obj,
                name_idx,
                helper_addr,
                ..
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 1);
                assert_eq!(*obj, 0);
                assert_eq!(*name_idx, 0);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected GetAttr template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn extended_attr_name_metadata_lowers_to_nop() {
        let code = get_attr_code(255);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        assert!(matches!(
            templates[0],
            TemplateInstruction::GetAttr {
                name_idx: u8::MAX,
                ..
            }
        ));
        assert!(matches!(
            templates[1],
            TemplateInstruction::Nop { bc_offset: 4 }
        ));
    }

    #[test]
    fn normal_len_lowers_to_runtime_helper_template() {
        let code = len_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::Len {
                bc_offset,
                dst,
                src,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 1);
                assert_eq!(*src, 0);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected Len template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_get_iter_lowers_to_runtime_helper_template() {
        let code = get_iter_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::GetIter {
                bc_offset,
                dst,
                src,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 1);
                assert_eq!(*src, 0);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected GetIter template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_for_iter_lowers_to_branching_runtime_helper_template() {
        let code = for_iter_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::ForIter {
                bc_offset,
                dst,
                iter,
                target,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 1);
                assert_eq!(*iter, 0);
                assert_eq!(*target, 8);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected ForIter template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_build_list_lowers_to_direct_sequence_helper_template() {
        let code = build_list_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::BuildList {
                bc_offset,
                dst,
                start,
                count,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 3);
                assert_eq!(*start, 0);
                assert_eq!(*count, 3);
                assert_eq!(
                    *helper_addr,
                    crate::jit_runtime_helpers::tier1_build_list_addr()
                );
            }
            template => panic!("expected BuildList template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_build_tuple_lowers_to_direct_sequence_helper_template() {
        let code = build_tuple_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::BuildTuple {
                bc_offset,
                dst,
                start,
                count,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 3);
                assert_eq!(*start, 0);
                assert_eq!(*count, 3);
                assert_eq!(
                    *helper_addr,
                    crate::jit_runtime_helpers::tier1_build_tuple_addr()
                );
            }
            template => panic!("expected BuildTuple template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_build_set_lowers_to_direct_hash_container_helper_template() {
        let code = build_set_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::BuildSet {
                bc_offset,
                dst,
                start,
                count,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 3);
                assert_eq!(*start, 0);
                assert_eq!(*count, 3);
                assert_eq!(
                    *helper_addr,
                    crate::jit_runtime_helpers::tier1_build_set_addr()
                );
            }
            template => panic!("expected BuildSet template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_build_dict_lowers_to_direct_hash_container_helper_template() {
        let code = build_dict_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::BuildDict {
                bc_offset,
                dst,
                start,
                count,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 4);
                assert_eq!(*start, 0);
                assert_eq!(*count, 2);
                assert_eq!(
                    *helper_addr,
                    crate::jit_runtime_helpers::tier1_build_dict_addr()
                );
            }
            template => panic!("expected BuildDict template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn proven_list_index_lowers_to_native_fast_path() {
        let code = list_get_item_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        let get_item = templates
            .iter()
            .find(|template| matches!(template, TemplateInstruction::ListGetItem { .. }))
            .expect("expected a list getitem specialization");

        match get_item {
            TemplateInstruction::ListGetItem {
                bc_offset,
                dst,
                list,
                index,
            } => {
                assert_eq!(*bc_offset, 16);
                assert_eq!(*dst, 4);
                assert_eq!(*list, 2);
                assert_eq!(*index, 3);
            }
            template => panic!("expected ListGetItem template, got {template:?}"),
        }

        assert!(!get_item.requires_interpreter_in_tier1());
        assert!(get_item.can_deopt());
    }

    #[test]
    fn proven_tuple_index_lowers_to_native_fast_path() {
        let code = tuple_get_item_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        let get_item = templates
            .iter()
            .find(|template| matches!(template, TemplateInstruction::TupleGetItem { .. }))
            .expect("expected a tuple getitem specialization");

        match get_item {
            TemplateInstruction::TupleGetItem {
                bc_offset,
                dst,
                tuple,
                index,
            } => {
                assert_eq!(*bc_offset, 16);
                assert_eq!(*dst, 4);
                assert_eq!(*tuple, 2);
                assert_eq!(*index, 3);
            }
            template => panic!("expected TupleGetItem template, got {template:?}"),
        }

        assert!(!get_item.requires_interpreter_in_tier1());
        assert!(get_item.can_deopt());
    }

    #[test]
    fn proven_list_store_lowers_to_native_fast_path() {
        let code = list_set_item_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        let set_item = templates
            .iter()
            .find(|template| matches!(template, TemplateInstruction::ListSetItem { .. }))
            .expect("expected a list setitem specialization");

        match set_item {
            TemplateInstruction::ListSetItem {
                bc_offset,
                list,
                index,
                value,
            } => {
                assert_eq!(*bc_offset, 20);
                assert_eq!(*list, 2);
                assert_eq!(*index, 3);
                assert_eq!(*value, 4);
            }
            template => panic!("expected ListSetItem template, got {template:?}"),
        }

        assert!(!set_item.requires_interpreter_in_tier1());
        assert!(set_item.can_deopt());
    }

    #[test]
    fn proven_list_len_lowers_to_native_fast_path() {
        let code = list_len_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[1] {
            TemplateInstruction::ListLen {
                bc_offset,
                dst,
                list,
            } => {
                assert_eq!(*bc_offset, 4);
                assert_eq!(*dst, 3);
                assert_eq!(*list, 2);
            }
            template => panic!("expected ListLen template, got {template:?}"),
        }

        assert!(!templates[1].requires_interpreter_in_tier1());
        assert!(templates[1].can_deopt());
    }

    #[test]
    fn proven_tuple_len_lowers_to_native_fast_path() {
        let code = tuple_len_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[1] {
            TemplateInstruction::TupleLen {
                bc_offset,
                dst,
                tuple,
            } => {
                assert_eq!(*bc_offset, 4);
                assert_eq!(*dst, 3);
                assert_eq!(*tuple, 2);
            }
            template => panic!("expected TupleLen template, got {template:?}"),
        }

        assert!(!templates[1].requires_interpreter_in_tier1());
        assert!(templates[1].can_deopt());
    }

    #[test]
    fn proven_list_append_lowers_to_capacity_checked_fast_path() {
        let code = list_append_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[2] {
            TemplateInstruction::ListAppendFast {
                bc_offset,
                list,
                value,
            } => {
                assert_eq!(*bc_offset, 8);
                assert_eq!(*list, 2);
                assert_eq!(*value, 3);
            }
            template => panic!("expected ListAppendFast template, got {template:?}"),
        }

        assert!(!templates[2].requires_interpreter_in_tier1());
        assert!(templates[2].can_deopt());
    }

    #[test]
    fn normal_dict_set_lowers_to_runtime_helper_template() {
        let code = dict_set_code();
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::DictSet {
                bc_offset,
                dict,
                key,
                value,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dict, 0);
                assert_eq!(*key, 2);
                assert_eq!(*value, 1);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected DictSet template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn normal_call_method_lowers_to_runtime_helper_template() {
        let code = call_method_code(CodeFlags::NONE);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        match &templates[0] {
            TemplateInstruction::CallMethod {
                bc_offset,
                dst,
                method,
                argc,
                helper_addr,
            } => {
                assert_eq!(*bc_offset, 0);
                assert_eq!(*dst, 0);
                assert_eq!(*method, 3);
                assert_eq!(*argc, 2);
                assert_ne!(*helper_addr, 0);
            }
            template => panic!("expected CallMethod template, got {template:?}"),
        }

        assert!(!templates[0].requires_interpreter_in_tier1());
        assert!(templates[0].can_deopt());
    }

    #[test]
    fn class_body_call_keeps_interpreter_semantics() {
        let code = call_code(CodeFlags::CLASS);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        assert!(matches!(
            templates[0],
            TemplateInstruction::InterpreterFallback { bc_offset: 0 }
        ));
        assert!(templates[0].requires_interpreter_in_tier1());
    }

    #[test]
    fn class_body_call_method_keeps_interpreter_semantics() {
        let code = call_method_code(CodeFlags::CLASS);
        let templates = lower_code_to_templates(&code).expect("lowering should succeed");

        assert!(matches!(
            templates[0],
            TemplateInstruction::InterpreterFallback { bc_offset: 0 }
        ));
        assert!(templates[0].requires_interpreter_in_tier1());
    }
}
