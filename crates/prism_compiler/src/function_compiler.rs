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

    /// Get cell variables ordered by closure slot.
    pub fn ordered_cellvars(&self) -> Vec<&'a Symbol> {
        self.scope_ref.scope.ordered_cellvars()
    }

    /// Get free variables ordered by closure slot.
    pub fn ordered_freevars(&self) -> Vec<&'a Symbol> {
        self.scope_ref.scope.ordered_freevars()
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
        let cellvar_names: Vec<Arc<str>> = ctx
            .ordered_cellvars()
            .into_iter()
            .map(|symbol| symbol.name.clone())
            .collect();

        let freevar_names: Vec<Arc<str>> = ctx
            .ordered_freevars()
            .into_iter()
            .map(|symbol| symbol.name.clone())
            .collect();

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
