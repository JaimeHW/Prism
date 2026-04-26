//! Closure analysis for cell and free variable detection.
//!
//! This module provides the `ClosureAnalyzer` that runs after scope analysis
//! to determine which variables need cell wrappers for closure support.
//!
//! # Python Closure Semantics
//!
//! Python closures capture variables by reference, not value. When a nested
//! function references a variable from an enclosing scope, both scopes share
//! a "cell" object that holds the value. This allows mutations in either:
//!
//! ```python
//! def outer():
//!     x = 1
//!     def inner():
//!         nonlocal x
//!         x = 2  # Modifies outer's x
//!     inner()
//!     print(x)  # Prints 2
//! ```
//!
//! # Cell vs Free Variables
//!
//! - **Cell variable**: A local variable in an enclosing scope that is captured
//!   by one or more nested functions. The outer scope creates the cell.
//! - **Free variable**: A reference to a cell variable from a nested scope.
//!   The nested scope receives the cell via the closure environment.
//!
//! # Algorithm
//!
//! 1. Walk the scope tree bottom-up (inner scopes first)
//! 2. For each scope with free variables, find the enclosing scope that defines them
//! 3. Mark those definitions as CELL and assign closure slots
//! 4. Assign closure slots to free variables (matching the enclosing cell slots)
//! 5. Handle nonlocal declarations specially (explicit capture request)

use super::symbol::{Scope, ScopeKind, Symbol, SymbolFlags};
use std::sync::Arc;

/// Closure analyzer that computes cell/freevar relationships.
///
/// This is a post-processor that runs after `ScopeAnalyzer` to complete
/// the closure analysis by propagating CELL flags upward and assigning
/// closure slot indices.
#[derive(Debug)]
pub struct ClosureAnalyzer {
    /// Statistics for debugging/optimization
    stats: ClosureStats,
}

/// Statistics about closure analysis.
#[derive(Debug, Default, Clone)]
pub struct ClosureStats {
    /// Total number of scopes analyzed
    pub scopes_analyzed: usize,
    /// Number of scopes with freevars
    pub scopes_with_freevars: usize,
    /// Total number of cell variables created
    pub cell_count: usize,
    /// Total number of free variable references
    pub freevar_count: usize,
    /// Maximum closure depth encountered
    pub max_closure_depth: usize,
}

impl ClosureAnalyzer {
    /// Create a new closure analyzer.
    #[inline]
    pub fn new() -> Self {
        Self {
            stats: ClosureStats::default(),
        }
    }

    /// Analyze a scope tree and complete cell/freevar resolution.
    ///
    /// This modifies the scope tree in place, setting CELL flags on captured
    /// variables and assigning closure slot indices.
    ///
    /// # Arguments
    /// * `root` - The root (module) scope to analyze
    ///
    /// # Returns
    /// Statistics about the analysis.
    pub fn analyze(&mut self, root: &mut Scope) -> ClosureStats {
        // Process all scopes recursively
        self.process_scope(root, &[]);

        self.stats.clone()
    }

    /// Process a scope and its children, propagating closure information.
    ///
    /// # Arguments
    /// * `scope` - The scope to process
    /// * `enclosing_scopes` - Stack of enclosing scopes (innermost last)
    fn process_scope(&mut self, scope: &mut Scope, enclosing_scopes: &[ScopeInfo]) {
        self.stats.scopes_analyzed += 1;

        // Build scope info for this scope
        let current_info = ScopeInfo {
            kind: scope.kind,
            name: scope.name.clone(),
            // Will be filled in during child processing
            local_names: scope
                .symbols
                .iter()
                .filter(|(_, sym)| is_closure_local_definition(sym))
                .map(|(name, _)| name.clone())
                .collect(),
        };

        // Build extended enclosing scopes stack for children
        let mut child_enclosing: Vec<ScopeInfo> = enclosing_scopes.to_vec();
        child_enclosing.push(current_info);

        // First, process all children (bottom-up traversal)
        // We need to collect child scopes to avoid borrow issues
        let num_children = scope.children.len();
        for i in 0..num_children {
            // Temporarily take the child
            let mut child =
                std::mem::replace(&mut scope.children[i], Scope::new(ScopeKind::Module, ""));

            // Process the child
            self.process_scope(&mut child, &child_enclosing);

            // Put it back
            scope.children[i] = child;
        }

        // Now process freevars in this scope and propagate CELL to enclosing scopes
        let freevars: Vec<Arc<str>> = scope.freevars().map(|s| s.name.clone()).collect();

        if !freevars.is_empty() {
            self.stats.scopes_with_freevars += 1;
            self.stats.freevar_count += freevars.len();

            // Track closure depth (how many scopes up we need to look)
            let mut max_depth = 0;

            for freevar_name in &freevars {
                // Find which enclosing scope defines this variable
                let found =
                    self.find_and_mark_cell(scope, freevar_name, enclosing_scopes, &mut max_depth);

                if !found {
                    // Could be a truly global name or an error
                    // Mark as implicit global if not found in any enclosing scope
                    if let Some(sym) = scope.symbols.get_mut(freevar_name.as_ref()) {
                        // Remove FREE flag and add GLOBAL_IMPLICIT
                        sym.flags = sym
                            .flags
                            .remove(SymbolFlags::FREE)
                            .union(SymbolFlags::GLOBAL_IMPLICIT);
                    }
                }
            }

            self.stats.max_closure_depth = self.stats.max_closure_depth.max(max_depth);
        }

        // Also handle nonlocal declarations
        self.process_nonlocals(scope, enclosing_scopes);

        // Assign closure slot indices
        self.assign_closure_slots(scope);
    }

    /// Find the enclosing scope that defines a variable and mark it as CELL.
    ///
    /// # Returns
    /// `true` if the variable was found in an enclosing scope.
    fn find_and_mark_cell(
        &mut self,
        _current_scope: &mut Scope,
        var_name: &Arc<str>,
        enclosing_scopes: &[ScopeInfo],
        depth: &mut usize,
    ) -> bool {
        // Search enclosing scopes from innermost to outermost
        for (i, enc_info) in enclosing_scopes.iter().rev().enumerate() {
            // Class and module scopes don't provide closure capture:
            // - Class vars are accessed via attributes, not lexical cells.
            // - Module vars are globals, not closure cells.
            if enc_info.kind == ScopeKind::Module {
                continue;
            }
            if enc_info.kind == ScopeKind::Class && var_name.as_ref() != "__class__" {
                continue;
            }

            // Check if this scope defines the variable
            if enc_info.local_names.contains(var_name) {
                *depth = (*depth).max(i + 1);

                // We found the defining scope, but we can't modify it directly
                // because we only have ScopeInfo. Instead, we record that this
                // variable needs to be marked as CELL.
                //
                // The actual marking will be done in a second pass or we need
                // to restructure. For now, we record success.
                //
                // Note: In a production implementation, we'd either:
                // 1. Use RefCell/Rc for the scope tree
                // 2. Do two passes (collect then mark)
                // 3. Pass mutable references more cleverly
                //
                // For this implementation, we'll mark during slot assignment.

                return true;
            }
        }

        false
    }

    /// Process nonlocal declarations, marking variables as CELL in enclosing scopes.
    fn process_nonlocals(&mut self, scope: &mut Scope, _enclosing_scopes: &[ScopeInfo]) {
        // Nonlocal declarations explicitly request capture
        // These are already marked with NONLOCAL flag by ScopeAnalyzer
        // We need to ensure the corresponding enclosing variable becomes a cell

        let nonlocals: Vec<Arc<str>> = scope
            .symbols
            .iter()
            .filter(|(_, sym)| sym.flags.contains(SymbolFlags::NONLOCAL))
            .map(|(name, _)| name.clone())
            .collect();

        for name in nonlocals {
            // Mark as free (it's captured from enclosing scope)
            if let Some(sym) = scope.symbols.get_mut(name.as_ref()) {
                sym.flags |= SymbolFlags::FREE;
                self.stats.freevar_count += 1;
            }
        }
    }

    /// Assign closure slot indices to cell and free variables.
    fn assign_closure_slots(&mut self, scope: &mut Scope) {
        // CPython layout: cellvars first, then freevars
        // This allows indexing cells[i] where i < len(cellvars) for local cells
        // and i >= len(cellvars) for captured freevars

        let mut slot_index: u16 = 0;

        // Collect all freevars from children that need to be handled
        // Skip variables that are already marked as global (they don't need closure capture)
        // Use HashSet to deduplicate (same var may be referenced by multiple children)
        let mut child_freevars: std::collections::HashSet<Arc<str>> =
            std::collections::HashSet::new();
        for child in &scope.children {
            for freevar in child.freevars() {
                // Skip if this is actually a global reference (marked by previous analysis)
                if is_closure_capture_reference(freevar) {
                    child_freevars.insert(freevar.name.clone());
                }
            }
            // Also check for nonlocal references
            for (name, child_sym) in &child.symbols {
                if child_sym.flags.contains(SymbolFlags::NONLOCAL) {
                    child_freevars.insert(name.clone());
                }
            }
        }

        // For each child freevar:
        // - If we define it: mark as CELL (we create the cell)
        // - If we don't define it: mark as FREE (we pass it through from our parent)
        for freevar_name in &child_freevars {
            if let Some(sym) = scope.symbols.get_mut(freevar_name.as_ref()) {
                // Only mark as CELL if:
                // 1. We define this symbol (DEF flag)
                // 2. Not already a CELL
                // 3. NOT in module scope (module-level vars are globals, not cells)
                if sym.flags.contains(SymbolFlags::DEF)
                    && !sym.flags.contains(SymbolFlags::CELL)
                    && scope.kind != ScopeKind::Module
                    && !sym.flags.contains(SymbolFlags::GLOBAL_EXPLICIT)
                    && !sym.flags.contains(SymbolFlags::GLOBAL_IMPLICIT)
                    && !sym.flags.contains(SymbolFlags::NONLOCAL)
                {
                    // We define it - mark as cell
                    sym.flags |= SymbolFlags::CELL;
                    self.stats.cell_count += 1;
                }
            } else {
                // We don't define it - we need to capture it as a freevar to pass through
                // (unless it would be global)
                if scope.kind != ScopeKind::Module {
                    // Create a new symbol as FREE
                    let mut new_sym = super::symbol::Symbol::new(freevar_name.clone());
                    new_sym.flags |= SymbolFlags::FREE | SymbolFlags::USE;
                    scope.symbols.insert(freevar_name.clone(), new_sym);
                    self.stats.freevar_count += 1;
                }
            }
        }

        // Assign slots to cell variables first (order matters for indexing)
        let mut cellvar_names: Vec<Arc<str>> = scope.cellvars().map(|s| s.name.clone()).collect();
        cellvar_names.sort(); // Deterministic ordering

        for name in &cellvar_names {
            if let Some(sym) = scope.symbols.get_mut(name.as_ref()) {
                sym.closure_slot = Some(slot_index);
                slot_index += 1;
            }
        }

        // Assign slots to free variables
        let mut freevar_names: Vec<Arc<str>> = scope.freevars().map(|s| s.name.clone()).collect();
        freevar_names.sort(); // Deterministic ordering

        for name in &freevar_names {
            if let Some(sym) = scope.symbols.get_mut(name.as_ref()) {
                sym.closure_slot = Some(slot_index);
                slot_index += 1;
            }
        }
    }

    /// Get analysis statistics.
    pub fn stats(&self) -> &ClosureStats {
        &self.stats
    }
}

impl Default for ClosureAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn is_closure_local_definition(symbol: &Symbol) -> bool {
    symbol.flags.contains(SymbolFlags::DEF)
        && !symbol.flags.contains(SymbolFlags::GLOBAL_EXPLICIT)
        && !symbol.flags.contains(SymbolFlags::GLOBAL_IMPLICIT)
        && !symbol.flags.contains(SymbolFlags::NONLOCAL)
}

#[inline]
fn is_closure_capture_reference(symbol: &Symbol) -> bool {
    !symbol.flags.contains(SymbolFlags::GLOBAL_EXPLICIT)
        && !symbol.flags.contains(SymbolFlags::GLOBAL_IMPLICIT)
}

/// Information about a scope for closure resolution.
#[derive(Debug, Clone)]
struct ScopeInfo {
    /// Kind of scope
    kind: ScopeKind,
    /// Scope name (kept for debugging purposes)
    #[allow(dead_code)]
    name: Arc<str>,
    /// Names defined locally in this scope
    local_names: Vec<Arc<str>>,
}

// =============================================================================
// Closure Slot Index
// =============================================================================

/// Index into the closure environment.
///
/// This is used by the compiler to reference cell/free variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClosureSlot(pub u16);

impl ClosureSlot {
    /// Create a new closure slot.
    #[inline]
    pub const fn new(index: u16) -> Self {
        Self(index)
    }

    /// Get the slot index.
    #[inline]
    pub const fn index(self) -> u16 {
        self.0
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a scope can provide closure variables.
///
/// Class scopes don't provide closure variables in Python - variables
/// defined in a class body are accessed via `self`, not through closures.
#[inline]
pub fn scope_provides_closures(kind: ScopeKind) -> bool {
    matches!(
        kind,
        ScopeKind::Function | ScopeKind::Lambda | ScopeKind::Comprehension
    )
}

/// Check if a scope can have free variables.
#[inline]
pub fn scope_can_have_freevars(kind: ScopeKind) -> bool {
    // All non-module scopes can have free variables
    kind != ScopeKind::Module
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
