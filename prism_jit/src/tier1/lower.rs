//! Bytecode to Template IR lowering with speculation-guided specialization.
//!
//! This module converts VM bytecode into template IR for JIT compilation.
//! It uses speculation hints from the interpreter's type feedback system
//! to select specialized templates for polymorphic operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌───────────────────┐     ┌─────────────────┐
//! │   CodeObject    │────▶│   BytecodeLower   │────▶│ TemplateIR Vec  │
//! │   (bytecode)    │     │ + SpeculationHints│     │                 │
//! └─────────────────┘     └───────────────────┘     └─────────────────┘
//!                               │
//!                               ▼
//!                         ┌───────────────────┐
//!                         │ SpeculationProvider│
//!                         │  (type feedback)   │
//!                         └───────────────────┘
//! ```
//!
//! # Specialization Strategy
//!
//! For generic opcodes (Add, Sub, Mul, etc.), the lowerer:
//! 1. Queries the SpeculationProvider for type hints at the bytecode offset
//! 2. If hints are available, emits specialized templates with inline guards
//! 3. If no hints, emits generic templates with full type dispatch
//!
//! This allows hot paths to use fast specialized code while cold paths
//! remain correct via fallback to the slow path.

use prism_code::{CodeObject, Instruction, Opcode};
use prism_core::{SpeculationProvider, TypeHint};

use super::codegen::TemplateInstruction;
use crate::ic::{IcKind, IcManager};

// =============================================================================
// Lowering Configuration
// =============================================================================

/// Configuration for bytecode lowering.
#[derive(Debug, Clone)]
pub struct LoweringConfig {
    /// Enable speculation-based specialization.
    pub enable_speculation: bool,
    /// Emit type guards for speculative code.
    pub emit_guards: bool,
    /// Use aggressive inlining for arithmetic ops.
    pub aggressive_inline: bool,
    /// Enable inline caching for property access.
    pub enable_ic: bool,
}

impl Default for LoweringConfig {
    fn default() -> Self {
        Self {
            enable_speculation: true,
            emit_guards: true,
            aggressive_inline: true,
            enable_ic: true,
        }
    }
}

// =============================================================================
// Comparison Operation Type
// =============================================================================

/// Type of comparison operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

// =============================================================================
// Bytecode Lowerer
// =============================================================================

/// Lowers bytecode to template IR with speculation-guided specialization.
///
/// The lowerer is parameterized by a `SpeculationProvider` which supplies
/// type hints collected during interpretation. This enables adaptive
/// compilation where frequently executed paths get specialized code.
///
/// When an `IcManager` is provided via `with_ic_manager`, the lowerer will
/// allocate IC sites for property access operations (`GetAttr`, `SetAttr`),
/// enabling inline caching for O(1) property access on repeated operations.
pub struct BytecodeLowerer<'a, S: SpeculationProvider> {
    /// Speculation provider for type hints.
    speculation: &'a S,
    /// Code ID for speculation queries.
    code_id: u32,
    /// Lowering configuration.
    config: LoweringConfig,
    /// Output template instructions.
    output: Vec<TemplateInstruction>,
    /// Current bytecode offset (for error reporting).
    current_offset: u32,
    /// Optional IC manager for allocating IC sites during lowering.
    /// When Some, GetAttr/SetAttr operations will allocate IC sites.
    ic_manager: Option<&'a mut IcManager>,
}

impl<'a, S: SpeculationProvider> BytecodeLowerer<'a, S> {
    /// Create a new lowerer with the given speculation provider.
    ///
    /// This creates a lowerer without IC support. Use `with_ic_manager`
    /// to enable inline caching for property access operations.
    #[inline]
    pub fn new(speculation: &'a S, code_id: u32, config: LoweringConfig) -> Self {
        Self {
            speculation,
            code_id,
            config,
            output: Vec::with_capacity(128),
            current_offset: 0,
            ic_manager: None,
        }
    }

    /// Create a new lowerer with IC support.
    ///
    /// When IC is enabled in the config, this lowerer will allocate IC sites
    /// for `GetAttr` and `SetAttr` operations, enabling inline caching.
    #[inline]
    pub fn with_ic_manager(
        speculation: &'a S,
        code_id: u32,
        config: LoweringConfig,
        ic_manager: &'a mut IcManager,
    ) -> Self {
        Self {
            speculation,
            code_id,
            config,
            output: Vec::with_capacity(128),
            current_offset: 0,
            ic_manager: Some(ic_manager),
        }
    }

    /// Lower a code object to template IR.
    pub fn lower(&mut self, code: &CodeObject) -> Vec<TemplateInstruction> {
        self.output.clear();
        self.output.reserve(code.instructions.len());

        for (idx, inst) in code.instructions.iter().enumerate() {
            let bc_offset = (idx * 4) as u32; // Each instruction is 4 bytes
            self.current_offset = bc_offset;
            self.lower_instruction(*inst, bc_offset);
        }

        std::mem::take(&mut self.output)
    }

    /// Get the type hint for the current bytecode offset.
    #[inline]
    fn get_hint(&self) -> TypeHint {
        if self.config.enable_speculation {
            self.speculation
                .get_type_hint(self.code_id, self.current_offset)
        } else {
            TypeHint::None
        }
    }

    /// Lower a single instruction.
    fn lower_instruction(&mut self, inst: Instruction, bc_offset: u32) {
        let opcode = Opcode::from_u8(inst.opcode());

        match opcode {
            // =================================================================
            // Control Flow
            // =================================================================
            Some(Opcode::Nop) => {
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            Some(Opcode::Return) => {
                self.output.push(TemplateInstruction::Return {
                    bc_offset,
                    value: inst.dst().0,
                });
            }
            Some(Opcode::Jump) => {
                // Target is pc-relative, need to calculate absolute
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output
                    .push(TemplateInstruction::Jump { bc_offset, target });
            }
            Some(Opcode::JumpIfTrue) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfTrue {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::JumpIfFalse) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfFalse {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::JumpIfNone) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfNone {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::JumpIfNotNone) => {
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                self.output.push(TemplateInstruction::BranchIfNotNone {
                    bc_offset,
                    cond: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::ReturnNone) => {
                self.output
                    .push(TemplateInstruction::ReturnNone { bc_offset });
            }

            // =================================================================
            // Load/Store
            // =================================================================
            Some(Opcode::LoadNone) => {
                self.output.push(TemplateInstruction::LoadNone {
                    bc_offset,
                    dst: inst.dst().0,
                });
            }
            Some(Opcode::LoadTrue) => {
                self.output.push(TemplateInstruction::LoadBool {
                    bc_offset,
                    dst: inst.dst().0,
                    value: true,
                });
            }
            Some(Opcode::LoadFalse) => {
                self.output.push(TemplateInstruction::LoadBool {
                    bc_offset,
                    dst: inst.dst().0,
                    value: false,
                });
            }
            Some(Opcode::Move) => {
                self.output.push(TemplateInstruction::Move {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Local Variable Operations
            // =================================================================
            Some(Opcode::LoadLocal) => {
                self.output.push(TemplateInstruction::LoadLocal {
                    bc_offset,
                    dst: inst.dst().0,
                    slot: inst.imm16(),
                });
            }
            Some(Opcode::StoreLocal) => {
                self.output.push(TemplateInstruction::StoreLocal {
                    bc_offset,
                    src: inst.dst().0, // dst field stores source register
                    slot: inst.imm16(),
                });
            }
            Some(Opcode::DeleteLocal) => {
                self.output.push(TemplateInstruction::DeleteLocal {
                    bc_offset,
                    slot: inst.imm16(),
                });
            }

            // =================================================================
            // Global Variable Operations
            // =================================================================
            Some(Opcode::LoadGlobal) => {
                self.output.push(TemplateInstruction::LoadGlobal {
                    bc_offset,
                    dst: inst.dst().0,
                    name_idx: inst.imm16(),
                });
            }
            Some(Opcode::StoreGlobal) => {
                self.output.push(TemplateInstruction::StoreGlobal {
                    bc_offset,
                    src: inst.dst().0, // dst field stores source register
                    name_idx: inst.imm16(),
                });
            }
            Some(Opcode::DeleteGlobal) => {
                self.output.push(TemplateInstruction::DeleteGlobal {
                    bc_offset,
                    name_idx: inst.imm16(),
                });
            }

            // =================================================================
            // Specialized Integer Arithmetic (already typed)
            // =================================================================
            Some(Opcode::AddInt) => {
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::SubInt) => {
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::MulInt) => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::FloorDivInt) => {
                self.output.push(TemplateInstruction::IntDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::ModInt) => {
                self.output.push(TemplateInstruction::IntMod {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::NegInt) => {
                self.output.push(TemplateInstruction::IntNeg {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }
            Some(Opcode::PosInt) => {
                // Positive is a no-op for integers, just move the value
                self.output.push(TemplateInstruction::Move {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Specialized Float Arithmetic (already typed)
            // =================================================================
            Some(Opcode::AddFloat) => {
                self.output.push(TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::SubFloat) => {
                self.output.push(TemplateInstruction::FloatSub {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::MulFloat) => {
                self.output.push(TemplateInstruction::FloatMul {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::DivFloat) => {
                self.output.push(TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::FloorDivFloat) => {
                self.output.push(TemplateInstruction::FloatFloorDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::ModFloat) => {
                self.output.push(TemplateInstruction::FloatMod {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::NegFloat) => {
                self.output.push(TemplateInstruction::FloatNeg {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Bitwise Operations
            // =================================================================
            Some(Opcode::BitwiseAnd) => {
                self.output.push(TemplateInstruction::IntAnd {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::BitwiseOr) => {
                self.output.push(TemplateInstruction::IntOr {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::BitwiseXor) => {
                self.output.push(TemplateInstruction::IntXor {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::BitwiseNot) => {
                self.output.push(TemplateInstruction::IntNot {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }
            Some(Opcode::Shl) => {
                self.output.push(TemplateInstruction::IntShl {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::Shr) => {
                self.output.push(TemplateInstruction::IntShr {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::Not) => {
                self.output.push(TemplateInstruction::LogicalNot {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Identity Operations
            // =================================================================
            Some(Opcode::Is) => {
                self.output.push(TemplateInstruction::Is {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::IsNot) => {
                self.output.push(TemplateInstruction::IsNot {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Membership Operations
            // =================================================================
            Some(Opcode::In) => {
                self.output.push(TemplateInstruction::In {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }
            Some(Opcode::NotIn) => {
                self.output.push(TemplateInstruction::NotIn {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Closure Variable Operations
            // =================================================================
            Some(Opcode::LoadClosure) => {
                // LoadClosure uses op_di: dst = closure[imm16].get()
                self.output.push(TemplateInstruction::LoadClosure {
                    bc_offset,
                    dst: inst.dst().0,
                    cell_idx: inst.imm16(),
                });
            }
            Some(Opcode::StoreClosure) => {
                // StoreClosure uses op_di: closure[imm16].set(dst)
                // The register is passed via the dst field in op_di encoding
                self.output.push(TemplateInstruction::StoreClosure {
                    bc_offset,
                    src: inst.dst().0,
                    cell_idx: inst.imm16(),
                });
            }
            Some(Opcode::DeleteClosure) => {
                // DeleteClosure uses op_di: closure[imm16].clear()
                self.output.push(TemplateInstruction::DeleteClosure {
                    bc_offset,
                    cell_idx: inst.imm16(),
                });
            }

            // =================================================================
            // Object Attribute Operations
            // =================================================================
            Some(Opcode::GetAttr) => {
                // GetAttr: dst = src1.attr[src2]
                // Allocate IC site if IcManager is available and IC is enabled
                let ic_site_idx = if self.config.enable_ic {
                    self.ic_manager
                        .as_mut()
                        .and_then(|mgr| mgr.alloc_property_ic(bc_offset, IcKind::GetProperty))
                } else {
                    None
                };

                self.output.push(TemplateInstruction::GetAttr {
                    bc_offset,
                    dst: inst.dst().0,
                    obj: inst.src1().0,
                    name_idx: inst.src2().0,
                    ic_site_idx,
                });
            }
            Some(Opcode::SetAttr) => {
                // SetAttr: dst.attr[src1] = src2
                // Allocate IC site if IcManager is available and IC is enabled
                let ic_site_idx = if self.config.enable_ic {
                    self.ic_manager
                        .as_mut()
                        .and_then(|mgr| mgr.alloc_property_ic(bc_offset, IcKind::SetProperty))
                } else {
                    None
                };

                self.output.push(TemplateInstruction::SetAttr {
                    bc_offset,
                    obj: inst.dst().0,
                    name_idx: inst.src1().0,
                    value: inst.src2().0,
                    ic_site_idx,
                });
            }
            Some(Opcode::DelAttr) => {
                // DelAttr: del src1.attr[src2]
                self.output.push(TemplateInstruction::DelAttr {
                    bc_offset,
                    obj: inst.src1().0,
                    name_idx: inst.src2().0,
                });
            }
            Some(Opcode::LoadMethod) => {
                // LoadMethod: dst = obj.method (optimized method loading)
                self.output.push(TemplateInstruction::LoadMethod {
                    bc_offset,
                    dst: inst.dst().0,
                    obj: inst.src1().0,
                    name_idx: inst.src2().0,
                });
            }

            // =================================================================
            // Container Item Operations
            // =================================================================
            Some(Opcode::GetItem) => {
                // GetItem: dst = container[key]
                self.output.push(TemplateInstruction::GetItem {
                    bc_offset,
                    dst: inst.dst().0,
                    container: inst.src1().0,
                    key: inst.src2().0,
                });
            }
            Some(Opcode::SetItem) => {
                // SetItem: src1[dst] = src2 (key in dst field)
                self.output.push(TemplateInstruction::SetItem {
                    bc_offset,
                    container: inst.src1().0,
                    key: inst.dst().0,
                    value: inst.src2().0,
                });
            }
            Some(Opcode::DelItem) => {
                // DelItem: del src1[src2]
                self.output.push(TemplateInstruction::DelItem {
                    bc_offset,
                    container: inst.src1().0,
                    key: inst.src2().0,
                });
            }

            // =================================================================
            // Iteration Operations
            // =================================================================
            Some(Opcode::GetIter) => {
                // GetIter: dst = iter(src1)
                self.output.push(TemplateInstruction::GetIter {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }
            Some(Opcode::ForIter) => {
                let dst = inst.dst().0;
                let iter = dst
                    .checked_sub(1)
                    .expect("ForIter destination register must follow its iterator register");
                self.output.push(TemplateInstruction::ForIter {
                    bc_offset,
                    dst,
                    iter,
                    offset: inst.imm16() as i16,
                });
            }

            // =================================================================
            // Utility Operations
            // =================================================================
            Some(Opcode::Len) => {
                // Len: dst = len(src1)
                self.output.push(TemplateInstruction::Len {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }
            Some(Opcode::IsCallable) => {
                // IsCallable: dst = callable(src1)
                self.output.push(TemplateInstruction::IsCallable {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                });
            }

            // =================================================================
            // Container Building Operations
            // =================================================================
            Some(Opcode::BuildList) => {
                // BuildList: dst = [r(src1)..r(src1+src2)]
                self.output.push(TemplateInstruction::BuildList {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildTuple) => {
                // BuildTuple: dst = (r(src1)..r(src1+src2))
                self.output.push(TemplateInstruction::BuildTuple {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildSet) => {
                // BuildSet: dst = {r(src1)..r(src1+src2)}
                self.output.push(TemplateInstruction::BuildSet {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildDict) => {
                // BuildDict: dst = {} with src2 key-value pairs starting at src1
                self.output.push(TemplateInstruction::BuildDict {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildString) => {
                // BuildString: dst = "".join(r(src1)..r(src1+src2))
                self.output.push(TemplateInstruction::BuildString {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildSlice) => {
                // BuildSlice: dst = slice(src1, src2)
                self.output.push(TemplateInstruction::BuildSlice {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    stop: inst.src2().0,
                });
            }
            Some(Opcode::ListAppend) => {
                // ListAppend: src1.append(src2)
                self.output.push(TemplateInstruction::ListAppend {
                    bc_offset,
                    list: inst.src1().0,
                    value: inst.src2().0,
                });
            }
            Some(Opcode::SetAdd) => {
                // SetAdd: src1.add(src2)
                self.output.push(TemplateInstruction::SetAdd {
                    bc_offset,
                    set: inst.src1().0,
                    value: inst.src2().0,
                });
            }
            Some(Opcode::DictSet) => {
                // DictSet: src1[dst] = src2 (key in dst field)
                self.output.push(TemplateInstruction::DictSet {
                    bc_offset,
                    dict: inst.src1().0,
                    key: inst.dst().0,
                    value: inst.src2().0,
                });
            }
            Some(Opcode::UnpackSequence) => {
                // UnpackSequence: r(dst)..r(dst+src2) = unpack(src1)
                self.output.push(TemplateInstruction::UnpackSequence {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::UnpackEx) => {
                // UnpackEx: unpack with *rest - before/after counts encoded in src2
                // High nibble = before, low nibble = after
                let counts = inst.src2().0;
                self.output.push(TemplateInstruction::UnpackEx {
                    bc_offset,
                    dst: inst.dst().0,
                    src: inst.src1().0,
                    before: counts >> 4,
                    after: counts & 0x0F,
                });
            }

            // =================================================================
            // Function Call Operations
            // =================================================================
            Some(Opcode::Call) => {
                // Call: dst = func(args...)
                // src1 = function, src2 = argc
                self.output.push(TemplateInstruction::Call {
                    bc_offset,
                    dst: inst.dst().0,
                    func: inst.src1().0,
                    argc: inst.src2().0,
                });
            }
            Some(Opcode::CallKw) => {
                // CallKw: dst = func(args..., **kwargs)
                self.output.push(TemplateInstruction::CallKw {
                    bc_offset,
                    dst: inst.dst().0,
                    func: inst.src1().0,
                    argc: inst.src2().0,
                });
            }
            Some(Opcode::CallMethod) => {
                // CallMethod: dst = obj.method(args...)
                self.output.push(TemplateInstruction::CallMethod {
                    bc_offset,
                    dst: inst.dst().0,
                    method: inst.src1().0,
                    argc: inst.src2().0,
                });
            }
            Some(Opcode::TailCall) => {
                // TailCall: reuse current frame
                self.output.push(TemplateInstruction::TailCall {
                    bc_offset,
                    func: inst.src1().0,
                    argc: inst.src2().0,
                });
            }
            Some(Opcode::MakeFunction) => {
                // MakeFunction: dst = function(code_idx)
                self.output.push(TemplateInstruction::MakeFunction {
                    bc_offset,
                    dst: inst.dst().0,
                    code_idx: inst.imm16(),
                });
            }
            Some(Opcode::MakeClosure) => {
                // MakeClosure: dst = closure(code_idx, captures...)
                self.output.push(TemplateInstruction::MakeClosure {
                    bc_offset,
                    dst: inst.dst().0,
                    code_idx: inst.imm16(),
                });
            }
            Some(Opcode::CallKwEx) => {
                // CallKwEx: extension for CallKw
                // Extract kwargc and kwnames_idx from packed bytes
                let kwargc = inst.dst().0;
                let kwnames_idx = (inst.src1().0 as u16) | ((inst.src2().0 as u16) << 8);
                self.output.push(TemplateInstruction::CallKwEx {
                    bc_offset,
                    kwargc,
                    kwnames_idx,
                });
            }
            Some(Opcode::CallEx) => {
                // CallEx: dst = func(*args_tuple, **kwargs_dict)
                self.output.push(TemplateInstruction::CallEx {
                    bc_offset,
                    dst: inst.dst().0,
                    func: inst.src1().0,
                    args_tuple: inst.src2().0,
                    kwargs_dict: 0xFF, // Extension byte would contain this
                });
            }
            Some(Opcode::BuildTupleUnpack) => {
                // BuildTupleUnpack: dst = (*src1, *src2, ...)
                self.output.push(TemplateInstruction::BuildTupleUnpack {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }
            Some(Opcode::BuildDictUnpack) => {
                // BuildDictUnpack: dst = {**src1, **src2, ...}
                self.output.push(TemplateInstruction::BuildDictUnpack {
                    bc_offset,
                    dst: inst.dst().0,
                    start: inst.src1().0,
                    count: inst.src2().0,
                });
            }

            // =================================================================
            // Exception Handling Operations
            // =================================================================
            Some(Opcode::Raise) => {
                // Raise exception: raise exc_reg
                // Format: DstSrc (dst=exc_reg, no real destination - just operand encoding)
                self.output.push(TemplateInstruction::Raise {
                    bc_offset,
                    exc: inst.dst().0,
                });
            }
            Some(Opcode::Reraise) => {
                // Re-raise current exception (bare raise statement)
                // Format: NoOp
                self.output.push(TemplateInstruction::Reraise { bc_offset });
            }
            Some(Opcode::RaiseFrom) => {
                // Raise with chained cause: raise exc from cause
                // Format: DstSrc (dst=exc_reg, src=cause_reg)
                self.output.push(TemplateInstruction::RaiseFrom {
                    bc_offset,
                    exc: inst.dst().0,
                    cause: inst.src1().0,
                });
            }
            Some(Opcode::PopExceptHandler) => {
                // Pop exception handler from handler stack
                // Format: Imm16 (handler_idx)
                self.output.push(TemplateInstruction::PopExceptHandler {
                    bc_offset,
                    handler_idx: inst.imm16(),
                });
            }
            Some(Opcode::ExceptionMatch) => {
                // Check if exception matches type: dst = isinstance(exc, type)
                // Format: DstSrc (dst=result, src=exc_type)
                self.output.push(TemplateInstruction::ExceptionMatch {
                    bc_offset,
                    dst: inst.dst().0,
                    exc_type: inst.src1().0,
                });
            }
            Some(Opcode::LoadException) => {
                // Load current exception into register
                // Format: Dst
                self.output.push(TemplateInstruction::LoadException {
                    bc_offset,
                    dst: inst.dst().0,
                });
            }
            Some(Opcode::PushExcInfo) => {
                // Push exception info to stack for finally preservation
                // Format: NoOp
                self.output
                    .push(TemplateInstruction::PushExcInfo { bc_offset });
            }
            Some(Opcode::PopExcInfo) => {
                // Pop exception info from stack after finally
                // Format: NoOp
                self.output
                    .push(TemplateInstruction::PopExcInfo { bc_offset });
            }
            Some(Opcode::HasExcInfo) => {
                // Check if there's a pending exception
                // Format: Dst (dst = bool result)
                self.output.push(TemplateInstruction::HasExcInfo {
                    bc_offset,
                    dst: inst.dst().0,
                });
            }
            Some(Opcode::ClearException) => {
                // Clear exception state after handler processes exception
                // Format: NoOp
                self.output
                    .push(TemplateInstruction::ClearException { bc_offset });
            }
            Some(Opcode::EndFinally) => {
                // End finally block (re-raise or continue based on pending exception)
                // Format: NoOp
                self.output
                    .push(TemplateInstruction::EndFinally { bc_offset });
            }

            // =================================================================
            // Generator Operations (Phase 17)
            // =================================================================
            Some(Opcode::Yield) => {
                // Yield value from generator
                // Format: DstSrc (dst = value to yield)
                self.output.push(TemplateInstruction::Yield {
                    bc_offset,
                    value: inst.dst().0,
                });
            }
            Some(Opcode::YieldFrom) => {
                // Delegate to sub-generator: yield from iter
                // Format: DstSrc (dst = received value, src = sub-generator)
                self.output.push(TemplateInstruction::YieldFrom {
                    bc_offset,
                    dst: inst.dst().0,
                    iter: inst.src1().0,
                });
            }

            // =================================================================
            // Context Manager Operations (Phase 18)
            // =================================================================
            Some(Opcode::BeforeWith) => {
                // Prepare context manager: call __enter__
                // Format: DstSrc (dst = __enter__() result, src = context manager)
                self.output.push(TemplateInstruction::BeforeWith {
                    bc_offset,
                    dst: inst.dst().0,
                    mgr: inst.src1().0,
                });
            }
            Some(Opcode::ExitWith) => {
                // Normal exit from with block: call __exit__(None, None, None)
                // Format: DstSrc (dst = __exit__ result, src = context manager slot)
                self.output.push(TemplateInstruction::ExitWith {
                    bc_offset,
                    dst: inst.dst().0,
                    mgr: inst.src1().0,
                });
            }
            Some(Opcode::WithCleanup) => {
                // Exception exit from with block: call __exit__(exc_type, exc_val, exc_tb)
                // Format: DstSrc (dst = __exit__ result, src = context manager slot)
                self.output.push(TemplateInstruction::WithCleanup {
                    bc_offset,
                    dst: inst.dst().0,
                    mgr: inst.src1().0,
                });
            }

            // =================================================================
            // Import Operations (Phase 19)
            // =================================================================
            Some(Opcode::ImportName) => {
                // Import module: dst = import(name)
                // Format: DstImm16 (dst = module, imm16 = name_idx)
                self.output.push(TemplateInstruction::ImportName {
                    bc_offset,
                    dst: inst.dst().0,
                    name_idx: inst.imm16(),
                });
            }
            Some(Opcode::ImportFrom) => {
                // Import from module: dst = from module import name
                // Format: DstImm16 (dst = imported name, imm16 = name_idx)
                self.output.push(TemplateInstruction::ImportFrom {
                    bc_offset,
                    dst: inst.dst().0,
                    name_idx: inst.imm16(),
                });
            }
            Some(Opcode::ImportStar) => {
                // Import star: from module import *
                // Format: DstSrc (dst is unused, src = module)
                self.output.push(TemplateInstruction::ImportStar {
                    bc_offset,
                    module: inst.src1().0,
                });
            }

            // =================================================================
            // Pattern Matching Operations (Phase 20, PEP 634)
            // =================================================================
            Some(Opcode::MatchClass) => {
                // Match class pattern: dst = isinstance(subject, cls)
                // Format: DstSrcSrc (dst = bool, src1 = subject, src2 = class)
                self.output.push(TemplateInstruction::MatchClass {
                    bc_offset,
                    dst: inst.dst().0,
                    subject: inst.src1().0,
                    cls: inst.src2().0,
                });
            }
            Some(Opcode::MatchMapping) => {
                // Match mapping pattern: dst = is_mapping(subject)
                // Format: DstSrc (dst = bool, src = subject)
                self.output.push(TemplateInstruction::MatchMapping {
                    bc_offset,
                    dst: inst.dst().0,
                    subject: inst.src1().0,
                });
            }
            Some(Opcode::MatchSequence) => {
                // Match sequence pattern: dst = is_sequence(subject)
                // Format: DstSrc (dst = bool, src = subject)
                self.output.push(TemplateInstruction::MatchSequence {
                    bc_offset,
                    dst: inst.dst().0,
                    subject: inst.src1().0,
                });
            }
            Some(Opcode::MatchKeys) => {
                // Match keys: extract values from mapping by key tuple
                // Format: DstSrcSrc (dst = values tuple, src1 = mapping, src2 = keys)
                self.output.push(TemplateInstruction::MatchKeys {
                    bc_offset,
                    dst: inst.dst().0,
                    mapping: inst.src1().0,
                    keys: inst.src2().0,
                });
            }
            Some(Opcode::CopyDictWithoutKeys) => {
                // Copy dict without specified keys (for **rest capture)
                // Format: DstSrcSrc (dst = new dict, src1 = mapping, src2 = keys to exclude)
                self.output.push(TemplateInstruction::CopyDictWithoutKeys {
                    bc_offset,
                    dst: inst.dst().0,
                    mapping: inst.src1().0,
                    keys: inst.src2().0,
                });
            }
            Some(Opcode::GetMatchArgs) => {
                // Get __match_args__ from subject's type
                // Format: DstSrc (dst = __match_args__ tuple, src = subject)
                self.output.push(TemplateInstruction::GetMatchArgs {
                    bc_offset,
                    dst: inst.dst().0,
                    subject: inst.src1().0,
                });
            }

            // =================================================================
            // Async/Coroutine Operations (Phase 21, PEP 492/525/530)
            // =================================================================
            Some(Opcode::GetAwaitable) => {
                // Get awaitable from object for await expression
                // Format: DstSrc (dst = awaitable, src = object)
                self.output.push(TemplateInstruction::GetAwaitable {
                    bc_offset,
                    dst: inst.dst().0,
                    obj: inst.src1().0,
                });
            }
            Some(Opcode::GetAIter) => {
                // Get async iterator: dst = src.__aiter__()
                // Format: DstSrc (dst = async iterator, src = object)
                self.output.push(TemplateInstruction::GetAIter {
                    bc_offset,
                    dst: inst.dst().0,
                    obj: inst.src1().0,
                });
            }
            Some(Opcode::GetANext) => {
                // Get next from async iterator: dst = src.__anext__()
                // Format: DstSrc (dst = awaitable yielding next value, src = async iterator)
                self.output.push(TemplateInstruction::GetANext {
                    bc_offset,
                    dst: inst.dst().0,
                    iter: inst.src1().0,
                });
            }
            Some(Opcode::EndAsyncFor) => {
                // Handle StopAsyncIteration in async for loop
                // Format: DstImm16 (dst = value, imm16 = jump offset on StopAsyncIteration)
                let target = self.calculate_jump_target(bc_offset, inst.imm16() as i16);
                let target = u16::try_from(target).unwrap_or(u16::MAX);
                self.output.push(TemplateInstruction::EndAsyncFor {
                    bc_offset,
                    dst: inst.dst().0,
                    target,
                });
            }
            Some(Opcode::Send) => {
                // Send value to coroutine/generator: dst = gen.send(value)
                // Format: DstSrcSrc (dst = result, src1 = generator, src2 = value)
                self.output.push(TemplateInstruction::Send {
                    bc_offset,
                    dst: inst.dst().0,
                    generator: inst.src1().0,
                    value: inst.src2().0,
                });
            }

            // =================================================================
            // Generic Arithmetic (speculation-guided)
            // =================================================================
            Some(Opcode::Add) => {
                self.lower_generic_add(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::Sub) => {
                self.lower_generic_sub(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::Mul) => {
                self.lower_generic_mul(bc_offset, inst.dst().0, inst.src1().0, inst.src2().0);
            }
            Some(Opcode::TrueDiv) => {
                // True division always produces float
                self.output.push(TemplateInstruction::FloatDiv {
                    bc_offset,
                    dst: inst.dst().0,
                    lhs: inst.src1().0,
                    rhs: inst.src2().0,
                });
            }

            // =================================================================
            // Comparison Operations (speculation-guided)
            // =================================================================
            Some(Opcode::Lt) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Lt,
                );
            }
            Some(Opcode::Le) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Le,
                );
            }
            Some(Opcode::Gt) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Gt,
                );
            }
            Some(Opcode::Ge) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Ge,
                );
            }
            Some(Opcode::Eq) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Eq,
                );
            }
            Some(Opcode::Ne) => {
                self.lower_comparison(
                    bc_offset,
                    inst.dst().0,
                    inst.src1().0,
                    inst.src2().0,
                    ComparisonOp::Ne,
                );
            }

            // =================================================================
            // Unhandled opcodes - emit Nop for now
            // =================================================================
            _ => {
                // TODO: Implement remaining opcodes
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
        }
    }

    /// Calculate absolute jump target from pc-relative offset.
    #[inline]
    fn calculate_jump_target(&self, bc_offset: u32, relative: i16) -> u32 {
        // Jump target is relative to the NEXT instruction
        let next_pc = bc_offset.wrapping_add(4);
        if relative >= 0 {
            next_pc.wrapping_add(relative as u32 * 4)
        } else {
            next_pc.wrapping_sub((-relative) as u32 * 4)
        }
    }

    // =========================================================================
    // Speculation-Guided Lowering
    // =========================================================================

    /// Lower generic Add with speculation.
    fn lower_generic_add(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                // Speculate integer addition
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                // Speculate float addition
                self.output.push(TemplateInstruction::FloatAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::StrStr => {
                // String concatenation - not yet implemented
                // Fall back to generic (would emit runtime call)
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            TypeHint::None | TypeHint::StrInt | TypeHint::IntStr | TypeHint::ListList => {
                // No speculation or invalid type combo - emit generic
                // For now, fall back to IntAdd as placeholder
                // A real implementation would emit a runtime call
                self.output.push(TemplateInstruction::IntAdd {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower generic Sub with speculation.
    fn lower_generic_sub(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                self.output.push(TemplateInstruction::FloatSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            _ => {
                // Fall back to int sub
                self.output.push(TemplateInstruction::IntSub {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower generic Mul with speculation.
    fn lower_generic_mul(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                self.output.push(TemplateInstruction::FloatMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
            TypeHint::StrInt | TypeHint::IntStr => {
                // String repetition - not yet implemented
                self.output.push(TemplateInstruction::Nop { bc_offset });
            }
            _ => {
                self.output.push(TemplateInstruction::IntMul {
                    bc_offset,
                    dst,
                    lhs,
                    rhs,
                });
            }
        }
    }

    /// Lower comparison operation with speculation.
    fn lower_comparison(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8, op: ComparisonOp) {
        let hint = self.get_hint();

        match hint {
            TypeHint::IntInt => {
                // Emit integer comparison
                self.emit_int_comparison(bc_offset, dst, lhs, rhs, op);
            }
            TypeHint::FloatFloat | TypeHint::IntFloat | TypeHint::FloatInt => {
                // Emit float comparison
                self.emit_float_comparison(bc_offset, dst, lhs, rhs, op);
            }
            _ => {
                // No speculation or unsupported type - fall back to int comparison
                self.emit_int_comparison(bc_offset, dst, lhs, rhs, op);
            }
        }
    }

    /// Emit specialized integer comparison.
    fn emit_int_comparison(&mut self, bc_offset: u32, dst: u8, lhs: u8, rhs: u8, op: ComparisonOp) {
        let instr = match op {
            ComparisonOp::Lt => TemplateInstruction::IntLt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Le => TemplateInstruction::IntLe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Gt => TemplateInstruction::IntGt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ge => TemplateInstruction::IntGe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Eq => TemplateInstruction::IntEq {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ne => TemplateInstruction::IntNe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
        };
        self.output.push(instr);
    }

    /// Emit specialized float comparison.
    fn emit_float_comparison(
        &mut self,
        bc_offset: u32,
        dst: u8,
        lhs: u8,
        rhs: u8,
        op: ComparisonOp,
    ) {
        let instr = match op {
            ComparisonOp::Lt => TemplateInstruction::FloatLt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Le => TemplateInstruction::FloatLe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Gt => TemplateInstruction::FloatGt {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ge => TemplateInstruction::FloatGe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Eq => TemplateInstruction::FloatEq {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
            ComparisonOp::Ne => TemplateInstruction::FloatNe {
                bc_offset,
                dst,
                lhs,
                rhs,
            },
        };
        self.output.push(instr);
    }
}
