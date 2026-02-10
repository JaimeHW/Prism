//! Exception statement compilation.
//!
//! This module provides the `ExceptionCompiler` extension trait and helpers for compiling
//! try/except/finally statements to bytecode with zero-cost exception handling.
//!
//! # Architecture
//!
//! The Prism exception system uses compile-time exception tables rather than
//! runtime setup/teardown instructions, enabling true zero-cost exceptions:
//!
//! - **Happy path**: No overhead (no handler push/pop)
//! - **Exception path**: Binary search on exception table to find handler
//!
//! # Control Flow
//!
//! ```text
//! try:              ┌─────────────────┐
//!     body          │  [try body]     │  ← No setup opcode - zero cost!
//! except E as e:    │  Jump end       │  ← Normal exit skips handlers
//!     handler       ├─────────────────┤
//! finally:          │  [handler]      │  ← handler_pc in exception_table
//!     cleanup       │  Jump finally   │
//!                   ├─────────────────┤
//!                   │  [finally]      │  ← finally_pc in exception_table
//!                   │  EndFinally     │
//!                   └─────────────────┘
//! ```
//!
//! # Usage
//!
//! The `ExceptionCompiler` trait extends `Compiler` with exception-specific methods:
//!
//! ```ignore
//! impl ExceptionCompiler for Compiler {
//!     fn compile_try(&mut self, ...) -> CompileResult<()>;
//!     fn compile_raise(&mut self, ...) -> CompileResult<()>;
//! }
//! ```

#[allow(unused_imports)]
use crate::compiler::CompileResult;
use crate::{ExceptionEntry, FunctionBuilder, Instruction, Label, Opcode, Register};
use prism_parser::ast::{ExceptHandler, Expr, Stmt};

// =============================================================================
// Constants
// =============================================================================

/// Sentinel value indicating bare `except:` (catches all exceptions).
pub const CATCH_ALL: u16 = 0xFFFF;

/// Sentinel value indicating no finally block.
pub const NO_FINALLY: u32 = u32::MAX;

// =============================================================================
// Exception Entry Builder
// =============================================================================

/// A pending exception entry being built during try block compilation.
///
/// This tracks the state of an exception block as we compile it,
/// recording PC values that will be used to construct the final `ExceptionEntry`.
#[derive(Debug, Clone)]
pub struct PendingExceptionEntry {
    /// PC at start of try body.
    pub start_pc: u32,
    /// PC at end of try body (set when body compilation completes).
    pub end_pc: u32,
    /// PC of the first except handler.
    pub handler_pc: u32,
    /// PC of the finally block (NO_FINALLY if none).
    pub finally_pc: u32,
    /// Nesting depth for nested try blocks.
    pub depth: u16,
    /// Stack depth at try entry (for stack unwinding).
    pub stack_depth: u8,
}

impl PendingExceptionEntry {
    /// Creates a new pending entry at the current PC.
    pub fn new(start_pc: u32, depth: u16, stack_depth: u8) -> Self {
        Self {
            start_pc,
            end_pc: 0,
            handler_pc: 0,
            finally_pc: NO_FINALLY,
            depth,
            stack_depth,
        }
    }

    /// Converts this pending entry into a final `ExceptionEntry`.
    ///
    /// This is called when all PCs have been resolved.
    pub fn finalize(&self, exception_type_idx: u16) -> ExceptionEntry {
        ExceptionEntry {
            start_pc: self.start_pc,
            end_pc: self.end_pc,
            handler_pc: self.handler_pc,
            finally_pc: self.finally_pc,
            depth: self.depth,
            exception_type_idx,
        }
    }
}

// =============================================================================
// Exception Context
// =============================================================================

/// Context for tracking exception handling during compilation.
///
/// This maintains the state needed to compile nested try blocks
/// and build the exception table for the code object.
#[derive(Debug, Default)]
pub struct ExceptionContext {
    /// Stack of pending exception entries for nested try blocks.
    pending_entries: Vec<PendingExceptionEntry>,
    /// Finalized exception entries.
    entries: Vec<ExceptionEntry>,
    /// Current nesting depth.
    depth: u16,
}

impl ExceptionContext {
    /// Creates a new empty exception context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Begins a new try block at the given PC.
    ///
    /// Returns a handle (index) for the pending entry.
    pub fn begin_try(&mut self, pc: u32, stack_depth: u8) -> usize {
        let entry = PendingExceptionEntry::new(pc, self.depth, stack_depth);
        self.pending_entries.push(entry);
        self.depth += 1;
        self.pending_entries.len() - 1
    }

    /// Sets the end PC for the try body.
    pub fn end_try_body(&mut self, handle: usize, pc: u32) {
        if handle < self.pending_entries.len() {
            self.pending_entries[handle].end_pc = pc;
        }
    }

    /// Sets the handler PC (start of except block).
    pub fn set_handler_pc(&mut self, handle: usize, pc: u32) {
        if handle < self.pending_entries.len() {
            self.pending_entries[handle].handler_pc = pc;
        }
    }

    /// Sets the finally PC.
    pub fn set_finally_pc(&mut self, handle: usize, pc: u32) {
        if handle < self.pending_entries.len() {
            self.pending_entries[handle].finally_pc = pc;
        }
    }

    /// Finalizes a pending entry and adds it to the exception table.
    ///
    /// For multiple except handlers with different types, call this
    /// multiple times with different `exception_type_idx` values.
    pub fn finalize_handler(&mut self, handle: usize, exception_type_idx: u16) {
        if handle < self.pending_entries.len() {
            let entry = self.pending_entries[handle].finalize(exception_type_idx);
            self.entries.push(entry);
        }
    }

    /// Ends the current try block context.
    pub fn end_try(&mut self) {
        if self.depth > 0 {
            self.depth -= 1;
        }
        // Note: pending_entries are kept until finalized
    }

    /// Returns the current nesting depth.
    pub fn current_depth(&self) -> u16 {
        self.depth
    }

    /// Returns whether we're currently inside a try block.
    pub fn in_try_block(&self) -> bool {
        self.depth > 0
    }

    /// Consumes this context and returns the exception entries for the code object.
    ///
    /// The entries are sorted by start_pc for efficient binary search.
    pub fn into_entries(mut self) -> Vec<ExceptionEntry> {
        // Sort by start_pc, then by handler_pc for consistent ordering
        self.entries.sort_by_key(|e| (e.start_pc, e.handler_pc));
        self.entries
    }

    /// Returns reference to finalized entries.
    pub fn entries(&self) -> &[ExceptionEntry] {
        &self.entries
    }
}

// =============================================================================
// Handler Entry
// =============================================================================

/// A single except handler extracted from an `ExceptHandler` AST node.
///
/// This intermediate representation is used during compilation to track
/// handler metadata before emitting bytecode.
#[derive(Debug)]
pub struct CompiledHandler {
    /// Label at start of handler body.
    pub start_label: Label,
    /// Exception type constant index (CATCH_ALL for bare except).
    pub type_idx: u16,
    /// Optional binding name slot for `except E as e:`.
    pub binding_slot: Option<u16>,
    /// Whether this is a finally handler.
    pub is_finally: bool,
}

// =============================================================================
// ExceptionCompiler Trait
// =============================================================================

/// Extension trait for compiling exception-related statements.
///
/// This trait provides methods for compiling try/except/finally blocks
/// and raise statements. It is implemented for `Compiler`.
pub trait ExceptionCompiler {
    /// Compiles a try/except/finally statement.
    ///
    /// # Control Flow
    ///
    /// ```text
    /// [try body]
    ///     Jump to else_label (normal path) or end_label (if no else)
    /// [except handler 1]
    ///     Jump to finally_label (or end if no finally)
    /// [except handler N]
    ///     Jump to finally_label
    /// [else body] (if present)
    ///     Jump to finally_label
    /// [finally body] (if present)
    ///     EndFinally
    /// ```
    fn compile_try(
        &mut self,
        body: &[Stmt],
        handlers: &[ExceptHandler],
        orelse: &[Stmt],
        finalbody: &[Stmt],
    ) -> CompileResult<()>;

    /// Compiles a raise statement.
    ///
    /// Forms:
    /// - `raise` - reraise current exception
    /// - `raise E` - raise exception E
    /// - `raise E from C` - raise E with __cause__ set to C
    fn compile_raise(&mut self, exc: Option<&Expr>, cause: Option<&Expr>) -> CompileResult<()>;

    /// Compiles a single except handler.
    fn compile_except_handler(
        &mut self,
        handler: &ExceptHandler,
        finally_label: Option<Label>,
        end_label: Label,
    ) -> CompileResult<()>;
}

// =============================================================================
// Handler Table Helper Functions
// =============================================================================

/// Checks if an exception type matches another type or is a subclass.
///
/// Used at compile time to detect obviously redundant handlers.
#[inline]
pub fn is_subclass_match(_type_filter: u16, _actual_type: u16) -> bool {
    // At compile time, we can't generally know subclass relationships
    // since types may be dynamically loaded. Return false to be conservative.
    false
}

// =============================================================================
// Exception Compilation Helpers
// =============================================================================

/// Emits an exception type check instruction.
///
/// This is used in except handlers to check if the raised exception
/// matches the handler's filter type.
#[inline]
pub fn emit_exception_match(
    builder: &mut FunctionBuilder,
    dst: Register,
    exc_reg: Register,
    type_reg: Register,
) {
    // ExceptionMatch: dst = isinstance(exc_reg, type_reg)
    builder.emit(Instruction::op_dss(
        Opcode::ExceptionMatch,
        dst,
        exc_reg,
        type_reg,
    ));
}

/// Emits an EndFinally instruction.
///
/// This instruction checks the exception state and either:
/// - Continues normal execution if no exception is pending
/// - Re-raises the pending exception
#[inline]
pub fn emit_end_finally(builder: &mut FunctionBuilder) {
    builder.emit(Instruction::op(Opcode::EndFinally));
}

/// Emits a Raise instruction.
///
/// This raises the exception in the given register.
#[inline]
pub fn emit_raise(builder: &mut FunctionBuilder, exc_reg: Register) {
    builder.emit(Instruction::op_ds(Opcode::Raise, exc_reg, exc_reg));
}

/// Emits a Reraise instruction.
///
/// This re-raises the current exception (stored in exception state).
#[inline]
pub fn emit_reraise(builder: &mut FunctionBuilder) {
    builder.emit(Instruction::op(Opcode::Reraise));
}

/// Emits PopExceptHandler instruction.
///
/// This clears the current exception handler after it has been processed.
#[inline]
pub fn emit_pop_except_handler(builder: &mut FunctionBuilder) {
    builder.emit(Instruction::op(Opcode::PopExceptHandler));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ════════════════════════════════════════════════════════════════════════
    // PendingExceptionEntry Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_pending_entry_new() {
        let entry = PendingExceptionEntry::new(100, 0, 5);

        assert_eq!(entry.start_pc, 100);
        assert_eq!(entry.end_pc, 0);
        assert_eq!(entry.handler_pc, 0);
        assert_eq!(entry.finally_pc, NO_FINALLY);
        assert_eq!(entry.depth, 0);
        assert_eq!(entry.stack_depth, 5);
    }

    #[test]
    fn test_pending_entry_finalize() {
        let mut entry = PendingExceptionEntry::new(10, 1, 2);
        entry.end_pc = 50;
        entry.handler_pc = 100;
        entry.finally_pc = 150;

        let finalized = entry.finalize(42);

        assert_eq!(finalized.start_pc, 10);
        assert_eq!(finalized.end_pc, 50);
        assert_eq!(finalized.handler_pc, 100);
        assert_eq!(finalized.finally_pc, 150);
        assert_eq!(finalized.depth, 1);
        assert_eq!(finalized.exception_type_idx, 42);
    }

    #[test]
    fn test_pending_entry_finalize_catch_all() {
        let entry = PendingExceptionEntry::new(0, 0, 0);
        let finalized = entry.finalize(CATCH_ALL);

        assert_eq!(finalized.exception_type_idx, CATCH_ALL);
    }

    // ════════════════════════════════════════════════════════════════════════
    // ExceptionContext Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_context_new() {
        let ctx = ExceptionContext::new();

        assert_eq!(ctx.current_depth(), 0);
        assert!(!ctx.in_try_block());
        assert!(ctx.entries().is_empty());
    }

    #[test]
    fn test_context_begin_try() {
        let mut ctx = ExceptionContext::new();

        let handle = ctx.begin_try(10, 5);

        assert_eq!(handle, 0);
        assert_eq!(ctx.current_depth(), 1);
        assert!(ctx.in_try_block());
    }

    #[test]
    fn test_context_nested_try() {
        let mut ctx = ExceptionContext::new();

        let h1 = ctx.begin_try(10, 0);
        assert_eq!(ctx.current_depth(), 1);

        let h2 = ctx.begin_try(20, 1);
        assert_eq!(ctx.current_depth(), 2);
        assert_eq!(h1, 0);
        assert_eq!(h2, 1);

        ctx.end_try();
        assert_eq!(ctx.current_depth(), 1);

        ctx.end_try();
        assert_eq!(ctx.current_depth(), 0);
        assert!(!ctx.in_try_block());
    }

    #[test]
    fn test_context_set_pcs() {
        let mut ctx = ExceptionContext::new();
        let handle = ctx.begin_try(10, 0);

        ctx.end_try_body(handle, 50);
        ctx.set_handler_pc(handle, 100);
        ctx.set_finally_pc(handle, 150);
        ctx.finalize_handler(handle, 42);

        let entries = ctx.into_entries();
        assert_eq!(entries.len(), 1);

        let entry = &entries[0];
        assert_eq!(entry.start_pc, 10);
        assert_eq!(entry.end_pc, 50);
        assert_eq!(entry.handler_pc, 100);
        assert_eq!(entry.finally_pc, 150);
        assert_eq!(entry.exception_type_idx, 42);
    }

    #[test]
    fn test_context_multiple_handlers() {
        let mut ctx = ExceptionContext::new();
        let handle = ctx.begin_try(10, 0);

        ctx.end_try_body(handle, 50);
        ctx.set_handler_pc(handle, 100);

        // Two handlers with different types
        ctx.finalize_handler(handle, 1); // Handler for type 1
        ctx.finalize_handler(handle, 2); // Handler for type 2

        let entries = ctx.into_entries();
        assert_eq!(entries.len(), 2);

        assert_eq!(entries[0].exception_type_idx, 1);
        assert_eq!(entries[1].exception_type_idx, 2);
    }

    #[test]
    fn test_context_into_entries_sorted() {
        let mut ctx = ExceptionContext::new();

        // Add entries in reverse order
        let h1 = ctx.begin_try(100, 0);
        ctx.end_try_body(h1, 150);
        ctx.set_handler_pc(h1, 200);
        ctx.finalize_handler(h1, 1);
        ctx.end_try();

        let h2 = ctx.begin_try(10, 0);
        ctx.end_try_body(h2, 50);
        ctx.set_handler_pc(h2, 60);
        ctx.finalize_handler(h2, 2);
        ctx.end_try();

        let entries = ctx.into_entries();

        // Should be sorted by start_pc
        assert_eq!(entries[0].start_pc, 10);
        assert_eq!(entries[1].start_pc, 100);
    }

    #[test]
    fn test_context_end_try_at_zero_depth() {
        let mut ctx = ExceptionContext::new();

        // Should not panic or underflow
        ctx.end_try();
        assert_eq!(ctx.current_depth(), 0);
    }

    // ════════════════════════════════════════════════════════════════════════
    // CompiledHandler Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_compiled_handler_size() {
        // CompiledHandler should be reasonably sized
        let size = std::mem::size_of::<CompiledHandler>();
        assert!(size <= 24, "CompiledHandler is {} bytes", size);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Constant Tests
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_catch_all_constant() {
        assert_eq!(CATCH_ALL, 0xFFFF);
        assert_eq!(CATCH_ALL, u16::MAX);
    }

    #[test]
    fn test_no_finally_constant() {
        assert_eq!(NO_FINALLY, u32::MAX);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Integration Tests (require full compiler setup)
    // ════════════════════════════════════════════════════════════════════════

    #[test]
    fn test_exception_entry_size() {
        // ExceptionEntry should be compact for cache efficiency
        let size = std::mem::size_of::<ExceptionEntry>();
        assert_eq!(size, 20, "ExceptionEntry should be 20 bytes, was {}", size);
    }

    #[test]
    fn test_exception_context_default() {
        let ctx1 = ExceptionContext::new();
        let ctx2 = ExceptionContext::default();

        assert_eq!(ctx1.current_depth(), ctx2.current_depth());
    }
}
