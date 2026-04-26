use crate::ir::builder::{
    ArithmeticBuilder, BitwiseBuilder, ContainerBuilder, ControlBuilder, GraphBuilder,
    ObjectBuilder,
};
use crate::ir::graph::Graph;
use crate::opt::speculation::SpeculationProvider;
use prism_code::{CodeObject, Instruction, Opcode, Register};

/// Translator from Bytecode to Sea-of-Nodes IR.
pub struct BytecodeTranslator<'a> {
    builder: GraphBuilder,
    code: &'a CodeObject,
    /// Optional speculation provider for PGO-guided branch seeding.
    speculation: Option<Box<dyn SpeculationProvider>>,
}

impl<'a> BytecodeTranslator<'a> {
    /// Create a new translator.
    pub fn new(builder: GraphBuilder, code: &'a CodeObject) -> Self {
        BytecodeTranslator {
            builder,
            code,
            speculation: None,
        }
    }

    /// Attach a speculation provider for PGO-guided IR construction.
    pub fn with_speculation(mut self, provider: Box<dyn SpeculationProvider>) -> Self {
        self.speculation = Some(provider);
        self
    }

    /// Translate the bytecode to a Graph.
    pub fn translate(mut self) -> Result<Graph, String> {
        let instructions = &self.code.instructions;
        let len = instructions.len();
        let mut pc = 0;

        while pc < len {
            let offset = pc as u32;

            // Handle control flow merges
            self.builder.merge_state(offset);
            self.builder.set_bc_offset(offset);

            // Fetch instruction
            let inst = instructions[pc];

            // Dispatch instruction
            self.dispatch(inst, offset, len)?;

            pc += 1;
        }

        Ok(self.builder.finish())
    }

    /// Helper to get a value from a register.
    #[inline(always)]
    fn read_register(
        &mut self,
        reg: Register,
        offset: u32,
        op: Opcode,
    ) -> Result<crate::ir::node::NodeId, String> {
        let value = self.builder.get_register(reg.index() as u16);
        if value.is_valid() {
            Ok(value)
        } else {
            Err(format!(
                "read of uninitialized register r{} at instruction offset {} for opcode {:?}",
                reg.index(),
                offset,
                op
            ))
        }
    }

    /// Helper to set a value to a register.
    #[inline(always)]
    fn set_register(&mut self, reg: Register, node: crate::ir::node::NodeId) {
        self.builder.set_register(reg.index() as u16, node)
    }

    /// Resolve a signed relative jump target in instruction units.
    #[inline]
    fn resolve_jump_target(
        &self,
        offset: u32,
        relative: i16,
        instruction_count: usize,
    ) -> Result<u32, String> {
        let target = offset as i64 + 1 + relative as i64;
        let max = instruction_count as i64;
        if !(0..=max).contains(&target) {
            return Err(format!(
                "invalid jump target {target} at instruction offset {offset} (delta {relative}, max {max})"
            ));
        }
        Ok(target as u32)
    }

    /// Dispatch instruction to appropriate builder method.
    fn dispatch(
        &mut self,
        inst: Instruction,
        offset: u32,
        instruction_count: usize,
    ) -> Result<(), String> {
        let Some(op) = Opcode::from_u8(inst.opcode()) else {
            return Err(format!(
                "invalid opcode byte 0x{:02X} at instruction offset {}",
                inst.opcode(),
                offset
            ));
        };

        match op {
            Opcode::Nop => {}

            // Control Flow
            Opcode::Jump => {
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                self.builder.translate_jump(target);
            }
            Opcode::JumpIfFalse => {
                let cond = self.read_register(inst.dst(), offset, op)?;
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                // JumpIfFalse: if !cond goto target else goto fallthrough
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, fallthrough, target);
            }
            Opcode::JumpIfTrue => {
                let cond = self.read_register(inst.dst(), offset, op)?;
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }
            Opcode::JumpIfNone => {
                let value = self.read_register(inst.dst(), offset, op)?;
                let none = self.builder.const_none();
                let cond = self.builder.int_eq(value, none);
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }
            Opcode::JumpIfNotNone => {
                let value = self.read_register(inst.dst(), offset, op)?;
                let none = self.builder.const_none();
                let cond = self.builder.int_ne(value, none);
                let target =
                    self.resolve_jump_target(offset, inst.imm16() as i16, instruction_count)?;
                let fallthrough = offset + 1;
                self.builder.translate_branch(cond, target, fallthrough);
            }

            Opcode::Return => {
                let val = self.read_register(inst.dst(), offset, op)?;
                self.builder.return_value(val);
            }
            Opcode::ReturnNone => {
                let val = self.builder.const_none();
                self.builder.return_value(val);
            }

            // Constants / Loads
            Opcode::LoadConst => {
                let const_idx = inst.imm16() as usize;
                let val = self.code.constants.get(const_idx).ok_or_else(|| {
                    format!(
                        "invalid constant index {} at instruction offset {} ({} constants)",
                        const_idx,
                        offset,
                        self.code.constants.len()
                    )
                })?;

                let node = match val {
                    Constant::Value(val) => {
                        if let Some(i) = val.as_int() {
                            self.builder.const_int(i)
                        } else if let Some(f) = val.as_float() {
                            self.builder.const_float(f)
                        } else if let Some(b) = val.as_bool() {
                            self.builder.const_bool(b)
                        } else if val.is_none() {
                            self.builder.const_none()
                        } else {
                            return Err(format!(
                                "unsupported constant type at index {} for instruction offset {}",
                                const_idx, offset
                            ));
                        }
                    }
                    Constant::BigInt(_) => {
                        return Err(format!(
                            "unsupported bigint constant at index {} for instruction offset {}",
                            const_idx, offset
                        ));
                    }
                };
                self.set_register(inst.dst(), node);
            }
            Opcode::LoadNone => {
                let val = self.builder.const_none();
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadTrue => {
                let val = self.builder.const_bool(true);
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadFalse => {
                let val = self.builder.const_bool(false);
                self.set_register(inst.dst(), val);
            }
            Opcode::Move => {
                let val = self.read_register(inst.src1(), offset, op)?;
                self.set_register(inst.dst(), val);
            }
            Opcode::LoadLocal => {
                let src = Register((inst.imm16() & 0xFF) as u8);
                let val = self.read_register(src, offset, op)?;
                self.set_register(inst.dst(), val);
            }
            Opcode::StoreLocal => {
                let src = self.read_register(inst.dst(), offset, op)?;
                let dst = Register((inst.imm16() & 0xFF) as u8);
                self.set_register(dst, src);
            }

            // Arithmetic
            Opcode::AddInt | Opcode::AddFloat | Opcode::Add => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::AddInt => self.builder.int_add(lhs, rhs),
                    Opcode::AddFloat => self.builder.float_add(lhs, rhs),
                    _ => self.builder.generic_add(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::SubInt | Opcode::SubFloat | Opcode::Sub => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::SubInt => self.builder.int_sub(lhs, rhs),
                    Opcode::SubFloat => self.builder.float_sub(lhs, rhs),
                    _ => self.builder.generic_sub(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::MulInt | Opcode::MulFloat | Opcode::Mul => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::MulInt => self.builder.int_mul(lhs, rhs),
                    Opcode::MulFloat => self.builder.float_mul(lhs, rhs),
                    _ => self.builder.generic_mul(lhs, rhs),
                };
                self.set_register(inst.dst(), res);
            }
            Opcode::FloorDivInt => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.int_div(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::ModInt => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.int_mod(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::NegInt => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.int_neg(src);
                self.set_register(inst.dst(), res);
            }
            Opcode::PosInt => {
                let src = self.read_register(inst.src1(), offset, op)?;
                self.set_register(inst.dst(), src);
            }
            Opcode::DivFloat => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.float_div(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::FloorDivFloat | Opcode::ModFloat => {
                return Err(format!(
                    "unsupported opcode {:?} encountered at instruction offset {}",
                    op, offset
                ));
            }
            Opcode::NegFloat => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.float_neg(src);
                self.set_register(inst.dst(), res);
            }

            // Bitwise / logical
            Opcode::BitwiseAnd => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_and(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseOr => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_or(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseXor => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_xor(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::BitwiseNot => {
                let src = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.bitwise_not(src);
                self.set_register(inst.dst(), res);
            }
            Opcode::Shl => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_shl(lhs, rhs);
                self.set_register(inst.dst(), res);
            }
            Opcode::Shr => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.bitwise_shr(lhs, rhs);
                self.set_register(inst.dst(), res);
            }

            // Comparisons
            Opcode::Lt | Opcode::Le | Opcode::Eq | Opcode::Ne | Opcode::Gt | Opcode::Ge => {
                let lhs = self.read_register(inst.src1(), offset, op)?;
                let rhs = self.read_register(inst.src2(), offset, op)?;
                let res = match op {
                    Opcode::Lt => self.builder.generic_lt(lhs, rhs),
                    Opcode::Le => self.builder.generic_le(lhs, rhs),
                    Opcode::Eq => self.builder.generic_eq(lhs, rhs),
                    Opcode::Ne => self.builder.generic_ne(lhs, rhs),
                    Opcode::Gt => self.builder.generic_gt(lhs, rhs),
                    Opcode::Ge => self.builder.generic_ge(lhs, rhs),
                    _ => unreachable!(),
                };
                self.set_register(inst.dst(), res);
            }

            // Objects
            Opcode::GetAttr | Opcode::SetAttr | Opcode::GetItem | Opcode::SetItem => {
                self.dispatch_object_op(op, inst, offset)?;
            }

            // Calls
            Opcode::Call | Opcode::CallMethod => {
                self.dispatch_call_op(op, inst, offset)?;
            }

            // Containers
            Opcode::BuildList | Opcode::BuildTuple | Opcode::GetIter | Opcode::Len => {
                self.dispatch_container_op(op, inst, offset)?;
            }

            _ => {
                return Err(format!(
                    "unsupported opcode 0x{:02X} encountered at instruction offset {}",
                    inst.opcode(),
                    offset
                ));
            }
        }

        Ok(())
    }

    fn dispatch_object_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::GetAttr => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let name = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.get_attr(obj, name);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetAttr => {
                let obj = self.read_register(inst.dst(), offset, op)?;
                let name = self.read_register(inst.src1(), offset, op)?;
                let val = self.read_register(inst.src2(), offset, op)?;
                self.builder.set_attr(obj, name, val);
            }
            Opcode::GetItem => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let key = self.read_register(inst.src2(), offset, op)?;
                let res = self.builder.get_item(obj, key);
                self.set_register(inst.dst(), res);
            }
            Opcode::SetItem => {
                let obj = self.read_register(inst.dst(), offset, op)?;
                let key = self.read_register(inst.src1(), offset, op)?;
                let val = self.read_register(inst.src2(), offset, op)?;
                self.builder.set_item(obj, key, val);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch_container_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::BuildList => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.build_list(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::BuildTuple => {
                let start_reg = inst.src1().index();
                let count = inst.src2().index();
                let mut elements = Vec::with_capacity(count as usize);
                for i in 0..count {
                    elements.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.build_tuple(&elements);
                self.set_register(inst.dst(), res);
            }
            Opcode::GetIter => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.get_iter(obj);
                self.set_register(inst.dst(), res);
            }
            Opcode::Len => {
                let obj = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.len(obj);
                self.set_register(inst.dst(), res);
            }
            _ => {}
        }
        Ok(())
    }

    fn dispatch_call_op(
        &mut self,
        op: Opcode,
        inst: Instruction,
        offset: u32,
    ) -> Result<(), String> {
        match op {
            Opcode::Call => {
                let func = self.read_register(inst.src1(), offset, op)?;
                let argc = inst.src2().index();
                let start_reg = inst.dst().index() + 1;

                let mut args = Vec::with_capacity(argc as usize);
                for i in 0..argc {
                    args.push(self.read_register(Register(start_reg + i), offset, op)?);
                }
                let res = self.builder.call(func, &args);
                self.set_register(inst.dst(), res);
            }
            Opcode::CallMethod => {
                // CallMethod encoding:
                // - src1: method register (loaded by LoadMethod)
                // - src1+1: implicit self slot (or None marker for unbound)
                // - src2: explicit argument count
                // - src1+2..: explicit arguments
                let method_reg = inst.src1().index();
                let argc = inst.src2().index() as u16;

                let self_reg_idx = method_reg as u16 + 1;
                if self_reg_idx > u8::MAX as u16 {
                    return Err(format!(
                        "CallMethod self register overflow at instruction offset {} (base r{})",
                        offset, method_reg
                    ));
                }

                let mut args = Vec::with_capacity(argc as usize + 1);
                args.push(self.read_register(Register(self_reg_idx as u8), offset, op)?);

                for i in 0..argc {
                    let arg_reg_idx = method_reg as u16 + 2 + i;
                    if arg_reg_idx > u8::MAX as u16 {
                        return Err(format!(
                            "CallMethod argument register overflow at instruction offset {} (base r{}, arg index {})",
                            offset, method_reg, i
                        ));
                    }
                    args.push(self.read_register(Register(arg_reg_idx as u8), offset, op)?);
                }

                let func = self.read_register(inst.src1(), offset, op)?;
                let res = self.builder.call(func, &args);
                self.set_register(inst.dst(), res);
            }
            _ => {}
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
use prism_code::Constant;
