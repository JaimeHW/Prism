//! JIT stack frame layout.
//!
//! This module defines the layout of JIT-compiled function frames,
//! designed for:
//! - Compatibility with interpreter frames for OSR
//! - Efficient register-to-stack mapping
//! - GC root scanning
//!
//! # Frame Layout (growing downward)
//! ```text
//! ┌───────────────────────────────┐ High addresses
//! │     Return Address            │ ← Pushed by CALL
//! ├───────────────────────────────┤
//! │     Saved RBP                 │ ← Frame pointer chain
//! ├───────────────────────────────┤ ← RBP points here
//! │     Saved Callee-Save Regs    │
//! ├───────────────────────────────┤
//! │     Context Pointer           │ ← Runtime context (GC, etc.)
//! ├───────────────────────────────┤
//! │     Bytecode Registers [0..N] │ ← Value slots (8 bytes each)
//! ├───────────────────────────────┤
//! │     Spill Slots               │ ← Register allocator spills
//! ├───────────────────────────────┤
//! │     Scratch/Alignment         │
//! └───────────────────────────────┘ ← RSP (16-byte aligned)
//! ```

use crate::backend::x64::{Assembler, CallingConvention, Gpr, GprSet, MemOperand, Xmm};

/// Offset of `JitFrameState.frame_base`.
pub(crate) const JIT_FRAME_STATE_FRAME_BASE_OFFSET: i32 = 0;
/// Offset of `JitFrameState.const_pool`.
pub(crate) const JIT_FRAME_STATE_CONST_POOL_OFFSET: i32 = 16;
/// Offset of `JitFrameState.closure_env`.
pub(crate) const JIT_FRAME_STATE_CLOSURE_ENV_OFFSET: i32 = 24;
/// Offset of `JitFrameState.global_scope`.
pub(crate) const JIT_FRAME_STATE_GLOBAL_SCOPE_OFFSET: i32 = 32;

/// Offsets within the JIT stack frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameOffsets {
    /// Offset from RBP to start of bytecode register slots.
    pub registers_base: i32,
    /// Offset from RBP to context pointer.
    pub context: i32,
    /// Offset from RBP to constant pool pointer.
    pub const_pool: i32,
    /// Offset from RBP to global scope pointer.
    pub global_scope: i32,
    /// Offset from RBP to closure environment pointer.
    pub closure_env: i32,
    /// Offset from RBP to first spill slot.
    pub spills_base: i32,
    /// Total frame size (including alignment).
    pub total_size: i32,
}

/// Description of a JIT stack frame layout.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    /// Number of bytecode registers in the function.
    pub num_registers: u16,
    /// Number of spill slots needed.
    pub num_spills: u16,
    /// Callee-saved registers to preserve.
    pub saved_regs: GprSet,
    /// Computed frame offsets.
    pub offsets: FrameOffsets,
    /// Calling convention.
    pub calling_convention: CallingConvention,
}

impl FrameLayout {
    /// Size of a single value slot in bytes.
    pub const SLOT_SIZE: i32 = 8;

    /// Offset reserved for saved RBP.
    pub const SAVED_RBP_SIZE: i32 = 8;

    /// Offset reserved for return address (pushed by CALL).
    pub const RETURN_ADDRESS_SIZE: i32 = 8;

    /// Size of context pointer slot.
    pub const CONTEXT_SIZE: i32 = 8;

    /// Create a new frame layout for a function.
    ///
    /// # Arguments
    /// * `num_registers` - Number of bytecode registers in the function
    /// * `num_spills` - Number of spill slots needed by register allocator
    /// * `saved_regs` - Callee-saved registers to preserve
    pub fn new(num_registers: u16, num_spills: u16, saved_regs: GprSet) -> Self {
        let cc = CallingConvention::host();

        // Calculate sizes. Callee-saved registers are pushed explicitly after
        // RBP, so they affect offsets and alignment but are not part of the
        // local frame allocation.
        let saved_regs_size = (saved_regs.count() as i32) * 8;
        let metadata_size = 4 * Self::SLOT_SIZE;
        let registers_size = (num_registers as i32) * Self::SLOT_SIZE;
        let spills_size = (num_spills as i32) * Self::SLOT_SIZE;

        // Frame layout from RBP:
        // RBP+0: saved RBP
        // RBP-8..: callee-saved registers (if any)
        // next slots: context, const pool, global scope, closure env
        // next slots: bytecode registers
        // next slots: spill slots

        let metadata_base = saved_regs_size;
        let context_offset = -(metadata_base + Self::CONTEXT_SIZE);
        let const_pool_offset = context_offset - Self::SLOT_SIZE;
        let global_scope_offset = const_pool_offset - Self::SLOT_SIZE;
        let closure_env_offset = global_scope_offset - Self::SLOT_SIZE;
        let registers_base = -(metadata_base + metadata_size + registers_size);
        let spills_base = registers_base - spills_size;

        // After CALL and PUSH RBP the stack is 16-byte aligned. Pushed
        // callee-saved registers and the local allocation together must keep
        // it aligned before any future runtime call.
        let frame_content = metadata_size + registers_size + spills_size;
        let alignment_remainder = (saved_regs_size + frame_content) & 15;
        let total_size = if alignment_remainder == 0 {
            frame_content
        } else {
            frame_content + (16 - alignment_remainder)
        };

        FrameLayout {
            num_registers,
            num_spills,
            saved_regs,
            offsets: FrameOffsets {
                context: context_offset,
                const_pool: const_pool_offset,
                global_scope: global_scope_offset,
                closure_env: closure_env_offset,
                registers_base,
                spills_base,
                total_size,
            },
            calling_convention: cc,
        }
    }

    /// Create a minimal frame layout (for leaf functions).
    pub fn minimal(num_registers: u16) -> Self {
        Self::new(num_registers, 0, GprSet::EMPTY)
    }

    /// Create a frame layout for a function from a CodeObject.
    pub fn for_function(code: &prism_code::CodeObject) -> Self {
        Self::minimal(code.register_count)
    }

    /// Get memory operand for a bytecode register slot.
    #[inline]
    pub fn register_slot(&self, reg_idx: u16) -> MemOperand {
        let offset = self.offsets.registers_base + (reg_idx as i32) * Self::SLOT_SIZE;
        MemOperand::base_disp(Gpr::Rbp, offset)
    }

    /// Get memory operand for a spill slot.
    #[inline]
    pub fn spill_slot(&self, spill_idx: u16) -> MemOperand {
        let offset = self.offsets.spills_base + (spill_idx as i32) * Self::SLOT_SIZE;
        MemOperand::base_disp(Gpr::Rbp, offset)
    }

    /// Get memory operand for the context pointer.
    #[inline]
    pub fn context_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.context)
    }

    /// Get the frame pointer offset from RSP after prologue.
    #[inline]
    pub fn frame_size(&self) -> i32 {
        self.offsets.total_size
    }

    /// Check if a register is callee-saved and needs to be preserved.
    #[inline]
    pub fn is_callee_saved(&self, reg: Gpr) -> bool {
        self.saved_regs.contains(reg)
    }

    // =========================================================================
    // Additional Slot Accessors for Template JIT
    // =========================================================================

    /// Get memory operand for a local variable slot.
    ///
    /// In the current implementation, locals share the same slots as
    /// bytecode registers. This may be separated in the future.
    #[inline]
    pub fn local_slot(&self, local_idx: u16) -> MemOperand {
        self.register_slot(local_idx)
    }

    /// Get memory operand for the constant pool pointer.
    #[inline]
    pub fn const_pool_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.const_pool)
    }

    /// Get memory operand for the global scope pointer.
    #[inline]
    pub fn global_scope_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.global_scope)
    }

    /// Get memory operand for the closure environment pointer.
    #[inline]
    pub fn closure_env_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.closure_env)
    }
}

/// Initialize a Tier 1 stack mirror from the VM-owned `JitFrameState`.
pub(crate) fn emit_frame_state_initialization(
    asm: &mut Assembler,
    frame: &FrameLayout,
    state_arg: Gpr,
    frame_base_scratch: Gpr,
    value_scratch: Gpr,
) {
    asm.mov_mr(&frame.context_slot(), state_arg);

    asm.mov_rm(
        value_scratch,
        &MemOperand::base_disp(state_arg, JIT_FRAME_STATE_CONST_POOL_OFFSET),
    );
    asm.mov_mr(&frame.const_pool_slot(), value_scratch);

    asm.mov_rm(
        value_scratch,
        &MemOperand::base_disp(state_arg, JIT_FRAME_STATE_GLOBAL_SCOPE_OFFSET),
    );
    asm.mov_mr(&frame.global_scope_slot(), value_scratch);

    asm.mov_rm(
        value_scratch,
        &MemOperand::base_disp(state_arg, JIT_FRAME_STATE_CLOSURE_ENV_OFFSET),
    );
    asm.mov_mr(&frame.closure_env_slot(), value_scratch);

    asm.mov_rm(
        frame_base_scratch,
        &MemOperand::base_disp(state_arg, JIT_FRAME_STATE_FRAME_BASE_OFFSET),
    );
    for reg_idx in 0..frame.num_registers {
        let source = MemOperand::base_disp(
            frame_base_scratch,
            i32::from(reg_idx) * FrameLayout::SLOT_SIZE,
        );
        asm.mov_rm(value_scratch, &source);
        asm.mov_mr(&frame.register_slot(reg_idx), value_scratch);
    }
}

/// Write the Tier 1 stack mirror back into the interpreter register array.
pub(crate) fn emit_frame_state_writeback(
    asm: &mut Assembler,
    frame: &FrameLayout,
    frame_base_scratch: Gpr,
    value_scratch: Gpr,
) {
    asm.mov_rm(frame_base_scratch, &frame.context_slot());
    asm.mov_rm(
        frame_base_scratch,
        &MemOperand::base_disp(frame_base_scratch, JIT_FRAME_STATE_FRAME_BASE_OFFSET),
    );

    for reg_idx in 0..frame.num_registers {
        let destination = MemOperand::base_disp(
            frame_base_scratch,
            i32::from(reg_idx) * FrameLayout::SLOT_SIZE,
        );
        asm.mov_rm(value_scratch, &frame.register_slot(reg_idx));
        asm.mov_mr(&destination, value_scratch);
    }
}

/// Standard register assignments for the Template JIT.
#[derive(Debug, Clone, Copy)]
pub struct RegisterAssignment {
    /// Register holding the context pointer.
    pub context: Gpr,
    /// Register used for accumulator (primary value).
    pub accumulator: Gpr,
    /// Scratch register 1 (volatile, not preserved across calls).
    pub scratch1: Gpr,
    /// Scratch register 2.
    pub scratch2: Gpr,
    /// Register for bytecode PC (if keeping in register).
    pub pc: Option<Gpr>,
    /// XMM register 0 for floating-point operations.
    pub xmm0: Xmm,
    /// XMM register 1 for floating-point operations.
    pub xmm1: Xmm,
}

impl RegisterAssignment {
    /// Standard assignment for Windows x64.
    pub const WINDOWS: RegisterAssignment = RegisterAssignment {
        context: Gpr::R14,     // Callee-saved
        accumulator: Gpr::Rax, // Return value register
        scratch1: Gpr::R10,    // Volatile
        scratch2: Gpr::R11,    // Volatile
        pc: None,              // Keep in memory for simplicity
        xmm0: Xmm::Xmm0,       // Floating-point scratch
        xmm1: Xmm::Xmm1,       // Floating-point scratch
    };

    /// Standard assignment for System V.
    pub const SYSV: RegisterAssignment = RegisterAssignment {
        context: Gpr::R14,     // Callee-saved
        accumulator: Gpr::Rax, // Return value register
        scratch1: Gpr::R10,    // Volatile
        scratch2: Gpr::R11,    // Volatile
        pc: None,              // Keep in memory
        xmm0: Xmm::Xmm0,       // Floating-point scratch
        xmm1: Xmm::Xmm1,       // Floating-point scratch
    };

    /// Get the standard assignment for the current platform.
    pub const fn host() -> Self {
        #[cfg(target_os = "windows")]
        {
            Self::WINDOWS
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self::SYSV
        }
    }
}

/// Conventions for argument passing to JIT-compiled functions.
#[derive(Debug, Clone, Copy)]
pub struct JitCallingConvention {
    /// Register for first argument (context pointer).
    pub arg0: Gpr,
    /// Register for second argument (frame pointer / register array).
    pub arg1: Gpr,
    /// Return value register.
    pub return_reg: Gpr,
}

impl JitCallingConvention {
    /// Windows x64 convention.
    pub const WINDOWS: JitCallingConvention = JitCallingConvention {
        arg0: Gpr::Rcx,
        arg1: Gpr::Rdx,
        return_reg: Gpr::Rax,
    };

    /// System V convention.
    pub const SYSV: JitCallingConvention = JitCallingConvention {
        arg0: Gpr::Rdi,
        arg1: Gpr::Rsi,
        return_reg: Gpr::Rax,
    };

    /// Get the convention for the current platform.
    pub const fn host() -> Self {
        #[cfg(target_os = "windows")]
        {
            Self::WINDOWS
        }
        #[cfg(not(target_os = "windows"))]
        {
            Self::SYSV
        }
    }
}
