//! Entry and exit stubs for transitioning between interpreter and JIT code.
//!
//! Entry stubs handle:
//! - Setting up the JIT frame from interpreter state
//! - Transitioning control to compiled code
//! - Handling returns and exceptions from JIT code
//!
//! Exit stubs handle:
//! - Deoptimization (returning to interpreter)
//! - Exception propagation
//! - OSR exit

use crate::backend::x64::assembler::Assembler;
use crate::backend::x64::registers::Gpr;

// =============================================================================
// Exit Reason
// =============================================================================

/// Reason for exiting JIT code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitReason {
    /// Normal return from function.
    Return = 0,
    /// Exception was thrown.
    Exception = 1,
    /// Deoptimization requested.
    Deoptimize = 2,
    /// OSR exit to interpreter.
    OsrExit = 3,
    /// Stack overflow detected.
    StackOverflow = 4,
    /// Tail call optimization.
    TailCall = 5,
}

impl ExitReason {
    /// Convert from raw value.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ExitReason::Return),
            1 => Some(ExitReason::Exception),
            2 => Some(ExitReason::Deoptimize),
            3 => Some(ExitReason::OsrExit),
            4 => Some(ExitReason::StackOverflow),
            5 => Some(ExitReason::TailCall),
            _ => None,
        }
    }
}

// =============================================================================
// Entry Stub
// =============================================================================

/// Entry stub for transitioning from interpreter to JIT code.
#[derive(Debug)]
pub struct EntryStub {
    /// Generated stub code.
    code: Vec<u8>,
    /// Offset to the actual entry point.
    entry_offset: usize,
}

impl EntryStub {
    /// Create a new entry stub.
    pub fn new() -> Self {
        let mut asm = Assembler::new();

        // Entry stub layout:
        // 1. Save callee-saved registers
        // 2. Setup JIT frame pointer
        // 3. Load arguments from interpreter state
        // 4. Jump to compiled code

        // Push callee-saved registers (System V AMD64: RBX, RBP, R12-R15)
        // On Windows: RBX, RBP, RDI, RSI, R12-R15
        asm.push(Gpr::Rbx);
        asm.push(Gpr::Rbp);
        asm.push(Gpr::R12);
        asm.push(Gpr::R13);
        asm.push(Gpr::R14);
        asm.push(Gpr::R15);

        // Setup frame pointer
        asm.mov_rr(Gpr::Rbp, Gpr::Rsp);

        // At this point:
        // - RDI/RCX = interpreter state pointer (first arg)
        // - RSI/RDX = JIT code entry pointer (second arg)
        // - RDX/R8 = argument count (third arg)

        // Store the actual entry address in R10 (scratch)
        #[cfg(target_os = "windows")]
        {
            asm.mov_rr(Gpr::R10, Gpr::Rdx); // Windows: entry is in RDX
        }
        #[cfg(not(target_os = "windows"))]
        {
            asm.mov_rr(Gpr::R10, Gpr::Rsi); // SysV: entry is in RSI
        }

        // Call into JIT code (indirect through R10)
        // call r10
        asm.emit_u8(0x41); // REX.B
        asm.emit_u8(0xFF); // FF /2
        asm.emit_u8(0xD2); // ModRM: mod=11, reg=2 (/2), rm=2 (R10)

        // Return value is in RAX

        // Restore callee-saved registers
        asm.pop(Gpr::R15);
        asm.pop(Gpr::R14);
        asm.pop(Gpr::R13);
        asm.pop(Gpr::R12);
        asm.pop(Gpr::Rbp);
        asm.pop(Gpr::Rbx);

        // Return
        asm.ret();

        Self {
            code: asm.code().to_vec(),
            entry_offset: 0,
        }
    }

    /// Get the entry stub code.
    #[inline]
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.code.len()
    }

    /// Get the entry offset.
    #[inline]
    pub fn entry_offset(&self) -> usize {
        self.entry_offset
    }
}

impl Default for EntryStub {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Deopt Stub
// =============================================================================

/// Deoptimization stub for exiting JIT code back to interpreter.
#[derive(Debug)]
pub struct DeoptStub {
    /// Generated stub code.
    code: Vec<u8>,
}

impl DeoptStub {
    /// Create a new deoptimization stub.
    pub fn new() -> Self {
        let mut asm = Assembler::new();

        // Deopt stub layout:
        // 1. Save all registers to a known location
        // 2. Set exit reason in a specific register
        // 3. Jump to common exit handler

        // Save return address (RIP at deopt point is on stack)
        // The deopt point pushes the deopt ID

        // Set exit reason
        asm.mov_ri32(Gpr::Rax, ExitReason::Deoptimize as u32);

        // Return to caller (which will handle the deopt)
        asm.ret();

        Self {
            code: asm.code().to_vec(),
        }
    }

    /// Get the deopt stub code.
    #[inline]
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.code.len()
    }
}

impl Default for DeoptStub {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Exception Stub
// =============================================================================

/// Exception handling stub for propagating exceptions from JIT code.
#[derive(Debug)]
pub struct ExceptionStub {
    /// Generated stub code.
    code: Vec<u8>,
}

impl ExceptionStub {
    /// Create a new exception stub.
    pub fn new() -> Self {
        let mut asm = Assembler::new();

        // Exception stub layout:
        // 1. Exception object is in RAX
        // 2. Set exit reason
        // 3. Return to caller

        // Move exception to a callee-saved register
        asm.mov_rr(Gpr::Rbx, Gpr::Rax);

        // Set exit reason
        asm.mov_ri32(Gpr::Rax, ExitReason::Exception as u32);

        // Return (caller will read exception from RBX)
        asm.ret();

        Self {
            code: asm.code().to_vec(),
        }
    }

    /// Get the exception stub code.
    #[inline]
    pub fn code(&self) -> &[u8] {
        &self.code
    }

    /// Get the code size.
    #[inline]
    pub fn code_size(&self) -> usize {
        self.code.len()
    }
}

impl Default for ExceptionStub {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Stub Generator
// =============================================================================

/// Generates various stubs used by the JIT runtime.
#[derive(Debug, Default)]
pub struct StubGenerator {
    /// Entry stub.
    pub entry: EntryStub,
    /// Deopt stub.
    pub deopt: DeoptStub,
    /// Exception stub.
    pub exception: ExceptionStub,
}

impl StubGenerator {
    /// Create a new stub generator with all stubs.
    pub fn new() -> Self {
        Self {
            entry: EntryStub::new(),
            deopt: DeoptStub::new(),
            exception: ExceptionStub::new(),
        }
    }

    /// Get total size of all stubs.
    pub fn total_size(&self) -> usize {
        self.entry.code_size() + self.deopt.code_size() + self.exception.code_size()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_reason_from_u8() {
        assert_eq!(ExitReason::from_u8(0), Some(ExitReason::Return));
        assert_eq!(ExitReason::from_u8(1), Some(ExitReason::Exception));
        assert_eq!(ExitReason::from_u8(2), Some(ExitReason::Deoptimize));
        assert_eq!(ExitReason::from_u8(99), None);
    }

    #[test]
    fn test_entry_stub_generation() {
        let stub = EntryStub::new();
        // Stub should have some code
        assert!(!stub.code().is_empty());
        assert!(stub.code_size() > 10); // Should have prologue + call + epilogue
    }

    #[test]
    fn test_deopt_stub_generation() {
        let stub = DeoptStub::new();
        assert!(!stub.code().is_empty());
    }

    #[test]
    fn test_exception_stub_generation() {
        let stub = ExceptionStub::new();
        assert!(!stub.code().is_empty());
    }

    #[test]
    fn test_stub_generator() {
        let generator = StubGenerator::new();
        assert!(generator.total_size() > 0);
        assert_eq!(
            generator.total_size(),
            generator.entry.code_size()
                + generator.deopt.code_size()
                + generator.exception.code_size()
        );
    }
}
