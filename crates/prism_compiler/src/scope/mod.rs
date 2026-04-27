//! Scope analysis for variable binding and closure detection.
//!
//! This module performs semantic analysis on the AST to determine:
//! - Variable scopes (local, global, free, cell)
//! - Closure captures
//! - Name resolution
//!
//! # Algorithm
//!
//! 1. **Collection pass**: Walk AST, collect all name definitions and uses
//! 2. **Classification pass**: Classify each name as local/global/free/cell
//! 3. **Closure analysis**: Determine which variables are captured by nested functions

mod closure;
mod symbol;
mod visitor;

pub use closure::{
    ClosureAnalyzer, ClosureSlot, ClosureStats, scope_can_have_freevars, scope_provides_closures,
};
pub use symbol::{Scope, ScopeKind, Symbol, SymbolFlags, SymbolTable};
pub use visitor::ScopeAnalyzer;
