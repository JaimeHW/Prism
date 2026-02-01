//! Symbol table and scope definitions.

use std::collections::HashMap;
use std::sync::Arc;

/// The kind of scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeKind {
    /// Module-level scope.
    Module,
    /// Class body scope.
    Class,
    /// Function scope.
    Function,
    /// Comprehension scope (list comp, dict comp, etc.).
    Comprehension,
    /// Lambda expression scope.
    Lambda,
}

/// Flags for symbol classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SymbolFlags(u16);

impl SymbolFlags {
    /// No flags.
    pub const NONE: SymbolFlags = SymbolFlags(0);
    /// Symbol is defined in this scope.
    pub const DEF: SymbolFlags = SymbolFlags(1 << 0);
    /// Symbol is used in this scope.
    pub const USE: SymbolFlags = SymbolFlags(1 << 1);
    /// Symbol is a parameter.
    pub const PARAM: SymbolFlags = SymbolFlags(1 << 2);
    /// Symbol is declared global.
    pub const GLOBAL_EXPLICIT: SymbolFlags = SymbolFlags(1 << 3);
    /// Symbol is implicitly global (module scope).
    pub const GLOBAL_IMPLICIT: SymbolFlags = SymbolFlags(1 << 4);
    /// Symbol is declared nonlocal.
    pub const NONLOCAL: SymbolFlags = SymbolFlags(1 << 5);
    /// Symbol is a free variable (captured from outer scope).
    pub const FREE: SymbolFlags = SymbolFlags(1 << 6);
    /// Symbol is a cell variable (captured by inner scope).
    pub const CELL: SymbolFlags = SymbolFlags(1 << 7);
    /// Symbol is annotated.
    pub const ANNOTATED: SymbolFlags = SymbolFlags(1 << 8);
    /// Symbol is imported.
    pub const IMPORTED: SymbolFlags = SymbolFlags(1 << 9);

    /// Check if a flag is set.
    #[inline]
    pub const fn contains(self, other: SymbolFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Combine flags.
    #[inline]
    pub const fn union(self, other: SymbolFlags) -> SymbolFlags {
        SymbolFlags(self.0 | other.0)
    }

    /// Check if this is a local variable.
    #[inline]
    pub const fn is_local(self) -> bool {
        self.contains(Self::DEF)
            && !self.contains(Self::GLOBAL_EXPLICIT)
            && !self.contains(Self::NONLOCAL)
    }

    /// Check if this is a global variable.
    #[inline]
    pub const fn is_global(self) -> bool {
        self.contains(Self::GLOBAL_EXPLICIT) || self.contains(Self::GLOBAL_IMPLICIT)
    }

    /// Check if this is a free variable.
    #[inline]
    pub const fn is_free(self) -> bool {
        self.contains(Self::FREE)
    }

    /// Check if this is a cell variable.
    #[inline]
    pub const fn is_cell(self) -> bool {
        self.contains(Self::CELL)
    }
}

impl std::ops::BitOr for SymbolFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for SymbolFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Information about a symbol in a scope.
#[derive(Debug, Clone)]
pub struct Symbol {
    /// The symbol name.
    pub name: Arc<str>,
    /// Classification flags.
    pub flags: SymbolFlags,
    /// Local slot index (if local).
    pub local_slot: Option<u16>,
    /// Closure slot index (if free or cell).
    pub closure_slot: Option<u16>,
}

impl Symbol {
    /// Create a new symbol.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        Self {
            name: name.into(),
            flags: SymbolFlags::NONE,
            local_slot: None,
            closure_slot: None,
        }
    }

    /// Check if this is a local variable.
    #[inline]
    pub fn is_local(&self) -> bool {
        self.flags.is_local()
    }

    /// Check if this is a global variable.
    #[inline]
    pub fn is_global(&self) -> bool {
        self.flags.is_global()
    }

    /// Check if this is a free variable.
    #[inline]
    pub fn is_free(&self) -> bool {
        self.flags.is_free()
    }

    /// Check if this is a cell variable.
    #[inline]
    pub fn is_cell(&self) -> bool {
        self.flags.is_cell()
    }
}

/// A scope in the symbol table.
#[derive(Debug)]
pub struct Scope {
    /// Scope kind.
    pub kind: ScopeKind,
    /// Scope name (function/class name or "<module>").
    pub name: Arc<str>,
    /// Symbols defined/used in this scope.
    pub symbols: HashMap<Arc<str>, Symbol>,
    /// Child scopes.
    pub children: Vec<Scope>,
    /// Whether this scope has yield (is generator).
    pub has_yield: bool,
    /// Whether this scope has await (is coroutine).
    pub has_await: bool,
    /// Whether this scope has *args.
    pub has_varargs: bool,
    /// Whether this scope has **kwargs.
    pub has_varkw: bool,
    /// Number of parameters.
    pub arg_count: u16,
    /// Number of positional-only parameters.
    pub posonlyarg_count: u16,
    /// Number of keyword-only parameters.
    pub kwonlyarg_count: u16,
}

impl Scope {
    /// Create a new scope.
    pub fn new(kind: ScopeKind, name: impl Into<Arc<str>>) -> Self {
        Self {
            kind,
            name: name.into(),
            symbols: HashMap::new(),
            children: Vec::new(),
            has_yield: false,
            has_await: false,
            has_varargs: false,
            has_varkw: false,
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
        }
    }

    /// Define a symbol in this scope.
    pub fn define(&mut self, name: impl Into<Arc<str>>, flags: SymbolFlags) {
        let name = name.into();
        self.symbols
            .entry(name.clone())
            .or_insert_with(|| Symbol::new(name))
            .flags |= flags | SymbolFlags::DEF;
    }

    /// Mark a symbol as used in this scope.
    pub fn use_symbol(&mut self, name: impl Into<Arc<str>>) {
        let name = name.into();
        self.symbols
            .entry(name.clone())
            .or_insert_with(|| Symbol::new(name))
            .flags |= SymbolFlags::USE;
    }

    /// Look up a symbol.
    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }

    /// Get all local variables (excluding cells).
    pub fn locals(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols
            .values()
            .filter(|s| s.is_local() && !s.is_cell())
    }

    /// Get all free variables.
    pub fn freevars(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.values().filter(|s| s.is_free())
    }

    /// Get all cell variables.
    pub fn cellvars(&self) -> impl Iterator<Item = &Symbol> {
        self.symbols.values().filter(|s| s.is_cell())
    }
}

/// Complete symbol table for a module.
#[derive(Debug)]
pub struct SymbolTable {
    /// Root (module) scope.
    pub root: Scope,
}

impl SymbolTable {
    /// Create a new symbol table with a module scope.
    pub fn new(module_name: impl Into<Arc<str>>) -> Self {
        Self {
            root: Scope::new(ScopeKind::Module, module_name),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_flags() {
        let flags = SymbolFlags::DEF | SymbolFlags::USE;
        assert!(flags.contains(SymbolFlags::DEF));
        assert!(flags.contains(SymbolFlags::USE));
        assert!(!flags.contains(SymbolFlags::PARAM));
    }

    #[test]
    fn test_scope_define() {
        let mut scope = Scope::new(ScopeKind::Function, "test");
        scope.define("x", SymbolFlags::PARAM);
        scope.use_symbol("y");

        let x = scope.lookup("x").unwrap();
        assert!(x.flags.contains(SymbolFlags::DEF));
        assert!(x.flags.contains(SymbolFlags::PARAM));

        let y = scope.lookup("y").unwrap();
        assert!(y.flags.contains(SymbolFlags::USE));
        assert!(!y.flags.contains(SymbolFlags::DEF));
    }

    #[test]
    fn test_symbol_classification() {
        let mut sym = Symbol::new("x");
        sym.flags |= SymbolFlags::DEF;
        assert!(sym.is_local());
        assert!(!sym.is_global());

        sym.flags |= SymbolFlags::GLOBAL_EXPLICIT;
        assert!(!sym.is_local());
        assert!(sym.is_global());
    }
}
