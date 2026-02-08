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

use crate::backend::x64::{CallingConvention, Gpr, GprSet, MemOperand, Xmm};

/// Offsets within the JIT stack frame.
#[derive(Debug, Clone, Copy)]
pub struct FrameOffsets {
    /// Offset from RBP to start of bytecode register slots.
    pub registers_base: i32,
    /// Offset from RBP to context pointer.
    pub context: i32,
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

        // Calculate sizes
        let saved_regs_size = (saved_regs.count() as i32) * 8;
        let registers_size = (num_registers as i32) * Self::SLOT_SIZE;
        let spills_size = (num_spills as i32) * Self::SLOT_SIZE;

        // Frame layout from RBP:
        // RBP+0: saved RBP
        // RBP-8 to RBP-(8+saved_regs_size): callee-saved registers
        // RBP-(8+saved_regs_size+8): context pointer
        // RBP-(8+saved_regs_size+8+registers_size): bytecode registers
        // RBP-(8+saved_regs_size+8+registers_size+spills_size): spill slots

        let base_offset = Self::SAVED_RBP_SIZE + saved_regs_size;
        let context_offset = -(base_offset + Self::CONTEXT_SIZE);
        let registers_base = context_offset - registers_size;
        let spills_base = registers_base - spills_size;

        // Calculate total frame size (align to 16 bytes)
        // Stack already has return address, so we need frame size to make it aligned
        let frame_content = saved_regs_size + Self::CONTEXT_SIZE + registers_size + spills_size;
        let total_size = (frame_content + 15) & !15;

        // Adjust if total would leave stack misaligned
        // After CALL: RSP points to return address (8 bytes)
        // After PUSH RBP: RSP is at 16-byte alignment - 16
        // So total_size should be (16k) to restore alignment
        let total_size = if (total_size + 16) % 16 != 0 {
            total_size + 8
        } else {
            total_size
        };

        FrameLayout {
            num_registers,
            num_spills,
            saved_regs,
            offsets: FrameOffsets {
                context: context_offset,
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
    pub fn for_function(code: &prism_compiler::bytecode::CodeObject) -> Self {
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
        MemOperand::base_disp(Gpr::Rbp, self.offsets.context - Self::SLOT_SIZE)
    }

    /// Get memory operand for the global scope pointer.
    #[inline]
    pub fn global_scope_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.context - 2 * Self::SLOT_SIZE)
    }

    /// Get memory operand for the closure environment pointer.
    #[inline]
    pub fn closure_env_slot(&self) -> MemOperand {
        MemOperand::base_disp(Gpr::Rbp, self.offsets.context - 3 * Self::SLOT_SIZE)
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_layout_basic() {
        let layout = FrameLayout::minimal(8);

        assert_eq!(layout.num_registers, 8);
        assert_eq!(layout.num_spills, 0);

        // Check that we can get register slots
        let slot0 = layout.register_slot(0);
        let slot7 = layout.register_slot(7);

        assert!(slot0.disp < 0); // Below RBP
        assert!(slot7.disp > slot0.disp); // Higher index = more positive offset from base
    }

    #[test]
    fn test_frame_layout_with_spills() {
        let saved = GprSet::EMPTY.insert(Gpr::Rbx).insert(Gpr::R12);
        let layout = FrameLayout::new(16, 4, saved);

        assert_eq!(layout.num_registers, 16);
        assert_eq!(layout.num_spills, 4);
        assert!(layout.saved_regs.contains(Gpr::Rbx));
        assert!(layout.saved_regs.contains(Gpr::R12));

        // Frame should be 16-byte aligned
        assert_eq!(layout.frame_size() % 16, 0);
    }

    #[test]
    fn test_register_slot_ordering() {
        let layout = FrameLayout::minimal(4);

        let slot0 = layout.register_slot(0);
        let slot1 = layout.register_slot(1);
        let slot2 = layout.register_slot(2);
        let slot3 = layout.register_slot(3);

        // Registers should be at consecutive 8-byte offsets (growing from base)
        assert_eq!(slot1.disp - slot0.disp, 8);
        assert_eq!(slot2.disp - slot1.disp, 8);
        assert_eq!(slot3.disp - slot2.disp, 8);
    }

    #[test]
    fn test_spill_slots() {
        let layout = FrameLayout::new(4, 2, GprSet::EMPTY);

        let spill0 = layout.spill_slot(0);
        let spill1 = layout.spill_slot(1);

        // Spills should be below registers (more negative offset)
        assert!(spill0.disp < layout.register_slot(3).disp);
        // Consecutive (growing from base)
        assert_eq!(spill1.disp - spill0.disp, 8);
    }

    #[test]
    fn test_register_assignment() {
        let assign = RegisterAssignment::host();

        // Basic sanity checks
        assert_eq!(assign.accumulator, Gpr::Rax);
        assert_ne!(assign.scratch1, assign.scratch2);
        assert_ne!(assign.context, assign.accumulator);
    }

    #[test]
    fn test_jit_calling_convention() {
        let cc = JitCallingConvention::host();

        assert_eq!(cc.return_reg, Gpr::Rax);
        assert_ne!(cc.arg0, cc.arg1);
    }
}
