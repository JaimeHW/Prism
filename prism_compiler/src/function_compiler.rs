//! Function-level compiler with scope-aware variable resolution.
//!
//! This module provides `FunctionCompiler`, a dedicated compiler for individual
//! Python functions and class bodies. It maintains scope context from the
//! symbol table to correctly emit closure-aware bytecode.
//!
//! # Architecture
//!
//! The function compiler separates concerns:
//! - **Scope tracking**: Maintains reference to current scope from analysis
//! - **Variable resolution**: Determines if name is local/cell/free/global
//! - **Bytecode emission**: Uses appropriate opcodes based on variable type
//!
//! # Closure Variable Types
//!
//! | Type   | Flag | Description | Load Opcode | Store Opcode |
//! |--------|------|-------------|-------------|--------------|
//! | Local  | -    | Regular stack slot | LoadLocal | StoreLocal |
//! | Cell   | CELL | Captured by inner scope | LoadClosure | StoreClosure |
//! | Free   | FREE | Captured from outer scope | LoadClosure | StoreClosure |
//! | Global | GLOBAL | Module-level | LoadGlobal | StoreGlobal |

use crate::bytecode::{FunctionBuilder, LocalSlot, Register};
use crate::scope::{Scope, Symbol};
use std::sync::Arc;

// =============================================================================
// Variable Resolution
// =============================================================================

/// Location of a resolved variable.
///
/// This determines which bytecode instructions to use for accessing the variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarLocation {
    /// Local variable at given slot on the stack frame.
    Local(u16),

    /// Global variable - will be looked up by name.
    /// The u16 is unused here; global names use the name table.
    Global,

    /// Closure variable (cell or free) at given slot in the ClosureEnv.
    /// Cell variables are captured by inner scopes.
    /// Free variables are captured from outer scopes.
    Closure(u16),
}

/// Determines how a variable should be accessed.
///
/// This trait is implemented by scope-aware structures that can resolve
/// variable names to their appropriate access mechanism.
pub trait VariableResolver {
    /// Resolve a variable name to its location.
    ///
    /// # Arguments
    /// * `name` - The variable name to resolve
    ///
    /// # Returns
    /// The location where this variable should be accessed from.
    fn resolve(&self, name: &str) -> VarLocation;
}

// =============================================================================
// Scope Reference
// =============================================================================

/// Reference to a scope with cached cellvar count for efficient slot calculation.
///
/// This is a lightweight wrapper around a scope reference that caches
/// frequently accessed information for performance.
#[derive(Debug, Clone)]
pub struct ScopeRef<'a> {
    /// The underlying scope.
    scope: &'a Scope,

    /// Cached count of cell variables (captured by inner scopes).
    /// Free variables are indexed after cells in the closure environment.
    cellvar_count: u16,
}

impl<'a> ScopeRef<'a> {
    /// Create a new scope reference with cached cellvar count.
    #[inline]
    pub fn new(scope: &'a Scope) -> Self {
        let cellvar_count = scope.cellvars().count() as u16;
        Self {
            scope,
            cellvar_count,
        }
    }

    /// Get the underlying scope.
    #[inline]
    pub fn scope(&self) -> &'a Scope {
        self.scope
    }

    /// Get the number of cell variables.
    #[inline]
    pub fn cellvar_count(&self) -> u16 {
        self.cellvar_count
    }

    /// Look up a symbol in this scope.
    #[inline]
    pub fn lookup(&self, name: &str) -> Option<&'a Symbol> {
        self.scope.lookup(name)
    }
}

impl<'a> VariableResolver for ScopeRef<'a> {
    fn resolve(&self, name: &str) -> VarLocation {
        if let Some(symbol) = self.lookup(name) {
            // Check closure variables first (cells and frees use same opcodes)
            if symbol.is_cell() || symbol.is_free() {
                if let Some(slot) = symbol.closure_slot {
                    return VarLocation::Closure(slot);
                }
            }

            // Check local variables
            if symbol.is_local() && !symbol.is_cell() {
                if let Some(slot) = symbol.local_slot {
                    return VarLocation::Local(slot);
                }
            }
        }

        // Default to global for undefined or explicitly global symbols
        VarLocation::Global
    }
}

// =============================================================================
// Function Compiler Context
// =============================================================================

/// Context for compiling a single function with scope-aware variable resolution.
///
/// This structure maintains the connection between the bytecode builder
/// and the scope analysis results to emit correct variable access instructions.
#[derive(Debug)]
pub struct FunctionContext<'a> {
    /// The scope for this function from analysis.
    scope_ref: ScopeRef<'a>,

    /// Whether this function captures any variables (has a closure environment).
    has_closure: bool,
}

impl<'a> FunctionContext<'a> {
    /// Create a new function context from a scope.
    pub fn new(scope: &'a Scope) -> Self {
        let scope_ref = ScopeRef::new(scope);
        let has_closure = scope.cellvars().next().is_some() || scope.freevars().next().is_some();

        Self {
            scope_ref,
            has_closure,
        }
    }

    /// Resolve a variable name to its location.
    #[inline]
    pub fn resolve_variable(&self, name: &str) -> VarLocation {
        self.scope_ref.resolve(name)
    }

    /// Check if this function has any closure variables.
    #[inline]
    pub fn has_closure(&self) -> bool {
        self.has_closure
    }

    /// Get the underlying scope.
    #[inline]
    pub fn scope(&self) -> &'a Scope {
        self.scope_ref.scope()
    }

    /// Get iterator over cell variable symbols.
    pub fn cellvars(&self) -> impl Iterator<Item = &'a Symbol> {
        self.scope_ref.scope.cellvars()
    }

    /// Get iterator over free variable symbols.
    pub fn freevars(&self) -> impl Iterator<Item = &'a Symbol> {
        self.scope_ref.scope.freevars()
    }
}

// =============================================================================
// Bytecode Emission Helpers
// =============================================================================

/// Helper trait for emitting variable access instructions.
///
/// This provides a clean interface for emitting the correct load/store
/// instructions based on variable location.
pub trait VariableEmitter {
    /// Emit a load instruction for the given variable location.
    fn emit_load_var(&mut self, dst: Register, location: VarLocation, name: Option<&str>);

    /// Emit a store instruction for the given variable location.
    fn emit_store_var(&mut self, location: VarLocation, src: Register, name: Option<&str>);

    /// Emit a delete instruction for the given variable location.
    fn emit_delete_var(&mut self, location: VarLocation, name: Option<&str>);
}

impl VariableEmitter for FunctionBuilder {
    fn emit_load_var(&mut self, dst: Register, location: VarLocation, name: Option<&str>) {
        match location {
            VarLocation::Local(slot) => {
                self.emit_load_local(dst, LocalSlot::new(slot));
            }
            VarLocation::Closure(slot) => {
                self.emit_load_closure(dst, slot);
            }
            VarLocation::Global => {
                let name_idx = self.add_name(Arc::from(name.unwrap_or("")));
                self.emit_load_global(dst, name_idx);
            }
        }
    }

    fn emit_store_var(&mut self, location: VarLocation, src: Register, name: Option<&str>) {
        match location {
            VarLocation::Local(slot) => {
                self.emit_store_local(LocalSlot::new(slot), src);
            }
            VarLocation::Closure(slot) => {
                self.emit_store_closure(slot, src);
            }
            VarLocation::Global => {
                let name_idx = self.add_name(Arc::from(name.unwrap_or("")));
                self.emit_store_global(name_idx, src);
            }
        }
    }

    fn emit_delete_var(&mut self, location: VarLocation, name: Option<&str>) {
        use crate::bytecode::{Instruction, Opcode};

        match location {
            VarLocation::Local(slot) => {
                self.emit(Instruction::op_di(
                    Opcode::DeleteLocal,
                    Register::new(0),
                    slot,
                ));
            }
            VarLocation::Closure(slot) => {
                self.emit_delete_closure(slot);
            }
            VarLocation::Global => {
                let name_idx = self.add_name(Arc::from(name.unwrap_or("")));
                self.emit(Instruction::op_di(
                    Opcode::DeleteGlobal,
                    Register::new(0),
                    name_idx,
                ));
            }
        }
    }
}

// =============================================================================
// Closure Info Builder
// =============================================================================

/// Information about a function's closure requirements.
///
/// This is used when compiling `MakeClosure` to properly set up
/// the closure environment with the captured variables.
#[derive(Debug, Clone)]
pub struct ClosureInfo {
    /// Names of cell variables (captured by inner scopes).
    pub cellvar_names: Vec<Arc<str>>,

    /// Names of free variables (captured from outer scopes).
    pub freevar_names: Vec<Arc<str>>,
}

impl ClosureInfo {
    /// Create closure info from a function context.
    pub fn from_context(ctx: &FunctionContext<'_>) -> Self {
        let cellvar_names: Vec<Arc<str>> = ctx.cellvars().map(|s| s.name.clone()).collect();

        let freevar_names: Vec<Arc<str>> = ctx.freevars().map(|s| s.name.clone()).collect();

        Self {
            cellvar_names,
            freevar_names,
        }
    }

    /// Check if there are any closure variables.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cellvar_names.is_empty() && self.freevar_names.is_empty()
    }

    /// Total number of closure slots needed.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.cellvar_names.len() + self.freevar_names.len()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scope::{ScopeKind, SymbolFlags};

    // =========================================================================
    // VarLocation Tests
    // =========================================================================

    #[test]
    fn test_var_location_equality() {
        assert_eq!(VarLocation::Local(0), VarLocation::Local(0));
        assert_ne!(VarLocation::Local(0), VarLocation::Local(1));
        assert_ne!(VarLocation::Local(0), VarLocation::Closure(0));
        assert_ne!(VarLocation::Local(0), VarLocation::Global);
        assert_eq!(VarLocation::Global, VarLocation::Global);
    }

    #[test]
    fn test_var_location_debug_format() {
        assert!(format!("{:?}", VarLocation::Local(42)).contains("42"));
        assert!(format!("{:?}", VarLocation::Closure(7)).contains("7"));
        assert!(format!("{:?}", VarLocation::Global).contains("Global"));
    }

    // =========================================================================
    // ScopeRef Tests
    // =========================================================================

    #[test]
    fn test_scope_ref_creation() {
        let mut scope = Scope::new(ScopeKind::Function, "test_func");
        scope.define("x", SymbolFlags::DEF);

        let scope_ref = ScopeRef::new(&scope);
        assert_eq!(scope_ref.cellvar_count(), 0);
        assert!(scope_ref.lookup("x").is_some());
        assert!(scope_ref.lookup("y").is_none());
    }

    #[test]
    fn test_scope_ref_with_cellvars() {
        let mut scope = Scope::new(ScopeKind::Function, "outer");

        // Define a cell variable
        let sym = scope
            .symbols
            .entry(Arc::from("captured"))
            .or_insert_with(|| Symbol::new("captured"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(0);

        let scope_ref = ScopeRef::new(&scope);
        assert_eq!(scope_ref.cellvar_count(), 1);
    }

    // =========================================================================
    // Variable Resolution Tests
    // =========================================================================

    #[test]
    fn test_resolve_local_variable() {
        let mut scope = Scope::new(ScopeKind::Function, "func");

        // Define local variable
        let sym = scope
            .symbols
            .entry(Arc::from("x"))
            .or_insert_with(|| Symbol::new("x"));
        sym.flags |= SymbolFlags::DEF;
        sym.local_slot = Some(0);

        let scope_ref = ScopeRef::new(&scope);
        assert_eq!(scope_ref.resolve("x"), VarLocation::Local(0));
    }

    #[test]
    fn test_resolve_cell_variable() {
        let mut scope = Scope::new(ScopeKind::Function, "outer");

        // Define cell variable (captured by inner scope)
        let sym = scope
            .symbols
            .entry(Arc::from("captured"))
            .or_insert_with(|| Symbol::new("captured"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(0);
        sym.local_slot = Some(0); // Also has local slot

        let scope_ref = ScopeRef::new(&scope);
        // Should prefer closure slot since it's a cell
        assert_eq!(scope_ref.resolve("captured"), VarLocation::Closure(0));
    }

    #[test]
    fn test_resolve_free_variable() {
        let mut scope = Scope::new(ScopeKind::Function, "inner");

        // Define free variable (captured from outer scope)
        let sym = scope
            .symbols
            .entry(Arc::from("outer_var"))
            .or_insert_with(|| Symbol::new("outer_var"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
        sym.closure_slot = Some(0);

        let scope_ref = ScopeRef::new(&scope);
        assert_eq!(scope_ref.resolve("outer_var"), VarLocation::Closure(0));
    }

    #[test]
    fn test_resolve_global_variable() {
        let mut scope = Scope::new(ScopeKind::Function, "func");

        // Define global variable
        let sym = scope
            .symbols
            .entry(Arc::from("global_var"))
            .or_insert_with(|| Symbol::new("global_var"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::GLOBAL_EXPLICIT;

        let scope_ref = ScopeRef::new(&scope);
        assert_eq!(scope_ref.resolve("global_var"), VarLocation::Global);
    }

    #[test]
    fn test_resolve_undefined_variable() {
        let scope = Scope::new(ScopeKind::Function, "func");
        let scope_ref = ScopeRef::new(&scope);

        // Undefined variable defaults to global
        assert_eq!(scope_ref.resolve("undefined"), VarLocation::Global);
    }

    // =========================================================================
    // FunctionContext Tests
    // =========================================================================

    #[test]
    fn test_function_context_no_closure() {
        let mut scope = Scope::new(ScopeKind::Function, "simple");
        scope.define("x", SymbolFlags::DEF);

        let ctx = FunctionContext::new(&scope);
        assert!(!ctx.has_closure());
        assert_eq!(ctx.cellvars().count(), 0);
        assert_eq!(ctx.freevars().count(), 0);
    }

    #[test]
    fn test_function_context_with_closure() {
        let mut scope = Scope::new(ScopeKind::Function, "outer");

        // Add cell variable
        let sym = scope
            .symbols
            .entry(Arc::from("captured"))
            .or_insert_with(|| Symbol::new("captured"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;

        let ctx = FunctionContext::new(&scope);
        assert!(ctx.has_closure());
        assert_eq!(ctx.cellvars().count(), 1);
    }

    #[test]
    fn test_function_context_resolve() {
        let mut scope = Scope::new(ScopeKind::Function, "func");

        // Local
        let sym = scope
            .symbols
            .entry(Arc::from("local"))
            .or_insert_with(|| Symbol::new("local"));
        sym.flags |= SymbolFlags::DEF;
        sym.local_slot = Some(0);

        // Cell
        let sym = scope
            .symbols
            .entry(Arc::from("cell"))
            .or_insert_with(|| Symbol::new("cell"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(0);

        let ctx = FunctionContext::new(&scope);
        assert_eq!(ctx.resolve_variable("local"), VarLocation::Local(0));
        assert_eq!(ctx.resolve_variable("cell"), VarLocation::Closure(0));
        assert_eq!(ctx.resolve_variable("unknown"), VarLocation::Global);
    }

    // =========================================================================
    // ClosureInfo Tests
    // =========================================================================

    #[test]
    fn test_closure_info_empty() {
        let scope = Scope::new(ScopeKind::Function, "simple");
        let ctx = FunctionContext::new(&scope);
        let info = ClosureInfo::from_context(&ctx);

        assert!(info.is_empty());
        assert_eq!(info.slot_count(), 0);
    }

    #[test]
    fn test_closure_info_with_cells() {
        let mut scope = Scope::new(ScopeKind::Function, "outer");

        // Add two cell variables
        for (i, name) in ["a", "b"].iter().enumerate() {
            let sym = scope
                .symbols
                .entry(Arc::from(*name))
                .or_insert_with(|| Symbol::new(*name));
            sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
            sym.closure_slot = Some(i as u16);
        }

        let ctx = FunctionContext::new(&scope);
        let info = ClosureInfo::from_context(&ctx);

        assert!(!info.is_empty());
        assert_eq!(info.cellvar_names.len(), 2);
        assert_eq!(info.freevar_names.len(), 0);
        assert_eq!(info.slot_count(), 2);
    }

    #[test]
    fn test_closure_info_with_freevars() {
        let mut scope = Scope::new(ScopeKind::Function, "inner");

        // Add free variable
        let sym = scope
            .symbols
            .entry(Arc::from("outer_x"))
            .or_insert_with(|| Symbol::new("outer_x"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
        sym.closure_slot = Some(0);

        let ctx = FunctionContext::new(&scope);
        let info = ClosureInfo::from_context(&ctx);

        assert!(!info.is_empty());
        assert_eq!(info.cellvar_names.len(), 0);
        assert_eq!(info.freevar_names.len(), 1);
        assert_eq!(info.slot_count(), 1);
    }

    #[test]
    fn test_closure_info_mixed() {
        let mut scope = Scope::new(ScopeKind::Function, "middle");

        // Cell (captured by grandchild)
        let sym = scope
            .symbols
            .entry(Arc::from("mine"))
            .or_insert_with(|| Symbol::new("mine"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(0);

        // Free (from parent)
        let sym = scope
            .symbols
            .entry(Arc::from("parents"))
            .or_insert_with(|| Symbol::new("parents"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
        sym.closure_slot = Some(1);

        let ctx = FunctionContext::new(&scope);
        let info = ClosureInfo::from_context(&ctx);

        assert!(!info.is_empty());
        assert_eq!(info.cellvar_names.len(), 1);
        assert_eq!(info.freevar_names.len(), 1);
        assert_eq!(info.slot_count(), 2);
    }

    // =========================================================================
    // Variable Emitter Tests
    // =========================================================================

    #[test]
    fn test_emit_load_local() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_load_var(r0, VarLocation::Local(5), None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadLocal as u8);
        assert_eq!(code.instructions[0].imm16(), 5);
    }

    #[test]
    fn test_emit_load_closure() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_load_var(r0, VarLocation::Closure(3), None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadClosure as u8);
        assert_eq!(code.instructions[0].imm16(), 3);
    }

    #[test]
    fn test_emit_load_global() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_load_var(r0, VarLocation::Global, Some("my_global"));

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::LoadGlobal as u8);
    }

    #[test]
    fn test_emit_store_local() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_store_var(VarLocation::Local(2), r0, None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::StoreLocal as u8);
        assert_eq!(code.instructions[0].imm16(), 2);
    }

    #[test]
    fn test_emit_store_closure() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_store_var(VarLocation::Closure(1), r0, None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::StoreClosure as u8);
        assert_eq!(code.instructions[0].imm16(), 1);
    }

    #[test]
    fn test_emit_store_global() {
        let mut builder = FunctionBuilder::new("test");
        let r0 = builder.alloc_register();

        builder.emit_store_var(VarLocation::Global, r0, Some("glob"));

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::StoreGlobal as u8);
    }

    #[test]
    fn test_emit_delete_local() {
        let mut builder = FunctionBuilder::new("test");

        builder.emit_delete_var(VarLocation::Local(4), None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::DeleteLocal as u8);
        assert_eq!(code.instructions[0].imm16(), 4);
    }

    #[test]
    fn test_emit_delete_closure() {
        let mut builder = FunctionBuilder::new("test");

        builder.emit_delete_var(VarLocation::Closure(2), None);

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::DeleteClosure as u8);
        assert_eq!(code.instructions[0].imm16(), 2);
    }

    #[test]
    fn test_emit_delete_global() {
        let mut builder = FunctionBuilder::new("test");

        builder.emit_delete_var(VarLocation::Global, Some("to_delete"));

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 1);

        use crate::bytecode::Opcode;
        assert_eq!(code.instructions[0].opcode(), Opcode::DeleteGlobal as u8);
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_complete_closure_pattern() {
        // Simulate outer function with captured variable
        let mut outer_scope = Scope::new(ScopeKind::Function, "make_counter");

        // Parameter 'start' - local and cell (captured by inner)
        let sym = outer_scope
            .symbols
            .entry(Arc::from("count"))
            .or_insert_with(|| Symbol::new("count"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.local_slot = Some(0);
        sym.closure_slot = Some(0);

        let outer_ctx = FunctionContext::new(&outer_scope);
        assert!(outer_ctx.has_closure());

        // count should be accessed via closure since it's a cell
        assert_eq!(outer_ctx.resolve_variable("count"), VarLocation::Closure(0));

        // Simulate inner function with free variable
        let mut inner_scope = Scope::new(ScopeKind::Function, "increment");

        let sym = inner_scope
            .symbols
            .entry(Arc::from("count"))
            .or_insert_with(|| Symbol::new("count"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
        sym.closure_slot = Some(0);

        let inner_ctx = FunctionContext::new(&inner_scope);
        assert!(inner_ctx.has_closure());
        assert_eq!(inner_ctx.resolve_variable("count"), VarLocation::Closure(0));
    }

    #[test]
    fn test_multiple_scope_levels() {
        // Three-level closure:
        // grandparent (defines x) -> parent (passes through) -> child (uses x)

        // Grandparent scope
        let mut gp_scope = Scope::new(ScopeKind::Function, "grandparent");
        let sym = gp_scope
            .symbols
            .entry(Arc::from("x"))
            .or_insert_with(|| Symbol::new("x"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.local_slot = Some(0);
        sym.closure_slot = Some(0);

        let gp_ctx = FunctionContext::new(&gp_scope);
        assert!(gp_ctx.has_closure());
        assert_eq!(gp_ctx.resolve_variable("x"), VarLocation::Closure(0));

        // Parent scope - x is both FREE (from grandparent) and CELL (for child)
        let mut p_scope = Scope::new(ScopeKind::Function, "parent");
        let sym = p_scope
            .symbols
            .entry(Arc::from("x"))
            .or_insert_with(|| Symbol::new("x"));
        sym.flags |= SymbolFlags::FREE | SymbolFlags::CELL;
        sym.closure_slot = Some(0);

        let p_ctx = FunctionContext::new(&p_scope);
        assert!(p_ctx.has_closure());
        assert_eq!(p_ctx.resolve_variable("x"), VarLocation::Closure(0));

        // Child scope - x is just FREE
        let mut c_scope = Scope::new(ScopeKind::Function, "child");
        let sym = c_scope
            .symbols
            .entry(Arc::from("x"))
            .or_insert_with(|| Symbol::new("x"));
        sym.flags |= SymbolFlags::USE | SymbolFlags::FREE;
        sym.closure_slot = Some(0);

        let c_ctx = FunctionContext::new(&c_scope);
        assert!(c_ctx.has_closure());
        assert_eq!(c_ctx.resolve_variable("x"), VarLocation::Closure(0));
    }

    #[test]
    fn test_mixed_variable_types() {
        // Function with all variable types
        let mut scope = Scope::new(ScopeKind::Function, "mixed");

        // Pure local
        let sym = scope
            .symbols
            .entry(Arc::from("local"))
            .or_insert_with(|| Symbol::new("local"));
        sym.flags |= SymbolFlags::DEF;
        sym.local_slot = Some(0);

        // Cell variable
        let sym = scope
            .symbols
            .entry(Arc::from("cell"))
            .or_insert_with(|| Symbol::new("cell"));
        sym.flags |= SymbolFlags::DEF | SymbolFlags::CELL;
        sym.closure_slot = Some(0);

        // Free variable
        let sym = scope
            .symbols
            .entry(Arc::from("free"))
            .or_insert_with(|| Symbol::new("free"));
        sym.flags |= SymbolFlags::FREE;
        sym.closure_slot = Some(1);

        // Explicit global
        let sym = scope
            .symbols
            .entry(Arc::from("glob"))
            .or_insert_with(|| Symbol::new("glob"));
        sym.flags |= SymbolFlags::GLOBAL_EXPLICIT;

        let ctx = FunctionContext::new(&scope);

        assert_eq!(ctx.resolve_variable("local"), VarLocation::Local(0));
        assert_eq!(ctx.resolve_variable("cell"), VarLocation::Closure(0));
        assert_eq!(ctx.resolve_variable("free"), VarLocation::Closure(1));
        assert_eq!(ctx.resolve_variable("glob"), VarLocation::Global);
        assert_eq!(ctx.resolve_variable("unknown"), VarLocation::Global);
    }
}
