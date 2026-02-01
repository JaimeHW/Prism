//! AST visitor for scope analysis.

use super::symbol::{Scope, ScopeKind, SymbolFlags, SymbolTable};
use prism_parser::ast::{Expr, ExprKind, Module, Pattern, PatternKind, Stmt, StmtKind};
use std::sync::Arc;

/// Scope analyzer that walks the AST and builds symbol tables.
pub struct ScopeAnalyzer {
    /// Stack of scopes being analyzed.
    scope_stack: Vec<Scope>,
}

impl ScopeAnalyzer {
    /// Create a new scope analyzer.
    pub fn new() -> Self {
        Self {
            scope_stack: Vec::new(),
        }
    }

    /// Analyze a module and return its symbol table.
    pub fn analyze(mut self, module: &Module, name: &str) -> SymbolTable {
        // Create module scope
        let mut module_scope = Scope::new(ScopeKind::Module, name);

        // Analyze all statements in module
        for stmt in &module.body {
            self.visit_stmt(&mut module_scope, stmt);
        }

        // Post-process: classify remaining unbound names as global
        self.classify_unbound(&mut module_scope);

        SymbolTable { root: module_scope }
    }

    /// Enter a new scope.
    fn push_scope(&mut self, scope: Scope) {
        self.scope_stack.push(scope);
    }

    /// Leave the current scope and return it.
    fn pop_scope(&mut self) -> Scope {
        self.scope_stack.pop().expect("scope stack underflow")
    }

    /// Visit a statement.
    fn visit_stmt(&mut self, scope: &mut Scope, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::FunctionDef {
                name,
                args,
                body,
                decorator_list,
                returns,
                ..
            }
            | StmtKind::AsyncFunctionDef {
                name,
                args,
                body,
                decorator_list,
                returns,
                ..
            } => {
                // Function name is defined in enclosing scope
                let name_arc: Arc<str> = name.clone().into();
                scope.define(name_arc.clone(), SymbolFlags::DEF);

                // Visit decorators in enclosing scope
                for dec in decorator_list {
                    self.visit_expr(scope, dec);
                }

                // Visit return annotation in enclosing scope
                if let Some(ret) = returns {
                    self.visit_expr(scope, ret);
                }

                // Create function scope
                let is_async = matches!(stmt.kind, StmtKind::AsyncFunctionDef { .. });
                let mut func_scope = Scope::new(ScopeKind::Function, name_arc);

                if is_async {
                    func_scope.has_await = true;
                }

                // Parameters
                func_scope.arg_count = args.args.len() as u16 + args.posonlyargs.len() as u16;
                func_scope.posonlyarg_count = args.posonlyargs.len() as u16;
                func_scope.kwonlyarg_count = args.kwonlyargs.len() as u16;

                for arg in &args.posonlyargs {
                    func_scope.define(arg.arg.clone(), SymbolFlags::PARAM);
                }
                for arg in &args.args {
                    func_scope.define(arg.arg.clone(), SymbolFlags::PARAM);
                }
                if let Some(vararg) = &args.vararg {
                    func_scope.define(vararg.arg.clone(), SymbolFlags::PARAM);
                    func_scope.has_varargs = true;
                }
                for arg in &args.kwonlyargs {
                    func_scope.define(arg.arg.clone(), SymbolFlags::PARAM);
                }
                if let Some(kwarg) = &args.kwarg {
                    func_scope.define(kwarg.arg.clone(), SymbolFlags::PARAM);
                    func_scope.has_varkw = true;
                }

                // Default values are evaluated in enclosing scope
                for default in &args.defaults {
                    self.visit_expr(scope, default);
                }
                for default in &args.kw_defaults {
                    if let Some(d) = default {
                        self.visit_expr(scope, d);
                    }
                }

                // Analyze function body
                self.push_scope(func_scope);
                for s in body {
                    self.visit_stmt_in_current(s);
                }
                let mut func_scope = self.pop_scope();

                // Classify symbols
                self.classify_unbound(&mut func_scope);

                // Add as child scope
                scope.children.push(func_scope);
            }

            StmtKind::ClassDef {
                name,
                bases,
                keywords,
                body,
                decorator_list,
                ..
            } => {
                // Class name is defined in enclosing scope
                let name_arc: Arc<str> = name.clone().into();
                scope.define(name_arc.clone(), SymbolFlags::DEF);

                // Visit decorators in enclosing scope
                for dec in decorator_list {
                    self.visit_expr(scope, dec);
                }

                // Visit bases in enclosing scope
                for base in bases {
                    self.visit_expr(scope, base);
                }
                for kw in keywords {
                    self.visit_expr(scope, &kw.value);
                }

                // Create class scope
                let mut class_scope = Scope::new(ScopeKind::Class, name_arc);

                // Analyze class body
                self.push_scope(class_scope);
                for s in body {
                    self.visit_stmt_in_current(s);
                }
                let mut class_scope = self.pop_scope();

                self.classify_unbound(&mut class_scope);
                scope.children.push(class_scope);
            }

            StmtKind::Assign { targets, value } => {
                for target in targets {
                    self.define_target(scope, target);
                }
                self.visit_expr(scope, value);
            }

            StmtKind::AnnAssign {
                target,
                annotation,
                value,
                ..
            } => {
                self.define_target(scope, target);
                self.visit_expr(scope, annotation);
                if let Some(v) = value {
                    self.visit_expr(scope, v);
                }
            }

            StmtKind::AugAssign { target, value, .. } => {
                // Aug assign uses and defines
                self.visit_expr(scope, target);
                self.define_target(scope, target);
                self.visit_expr(scope, value);
            }

            StmtKind::For {
                target,
                iter,
                body,
                orelse,
            }
            | StmtKind::AsyncFor {
                target,
                iter,
                body,
                orelse,
            } => {
                self.define_target(scope, target);
                self.visit_expr(scope, iter);
                for s in body {
                    self.visit_stmt(scope, s);
                }
                for s in orelse {
                    self.visit_stmt(scope, s);
                }
            }

            StmtKind::While { test, body, orelse } => {
                self.visit_expr(scope, test);
                for s in body {
                    self.visit_stmt(scope, s);
                }
                for s in orelse {
                    self.visit_stmt(scope, s);
                }
            }

            StmtKind::If { test, body, orelse } => {
                self.visit_expr(scope, test);
                for s in body {
                    self.visit_stmt(scope, s);
                }
                for s in orelse {
                    self.visit_stmt(scope, s);
                }
            }

            StmtKind::With { items, body } | StmtKind::AsyncWith { items, body } => {
                for item in items {
                    self.visit_expr(scope, &item.context_expr);
                    if let Some(opt) = &item.optional_vars {
                        self.define_target(scope, opt);
                    }
                }
                for s in body {
                    self.visit_stmt(scope, s);
                }
            }

            StmtKind::Try {
                body,
                handlers,
                orelse,
                finalbody,
            }
            | StmtKind::TryStar {
                body,
                handlers,
                orelse,
                finalbody,
            } => {
                for s in body {
                    self.visit_stmt(scope, s);
                }
                for handler in handlers {
                    if let Some(name) = &handler.name {
                        scope.define(name.clone(), SymbolFlags::DEF);
                    }
                    if let Some(typ) = &handler.typ {
                        self.visit_expr(scope, typ);
                    }
                    for s in &handler.body {
                        self.visit_stmt(scope, s);
                    }
                }
                for s in orelse {
                    self.visit_stmt(scope, s);
                }
                for s in finalbody {
                    self.visit_stmt(scope, s);
                }
            }

            StmtKind::Global(names) => {
                for name in names {
                    scope.define(name.clone(), SymbolFlags::GLOBAL_EXPLICIT);
                }
            }

            StmtKind::Nonlocal(names) => {
                for name in names {
                    scope.define(name.clone(), SymbolFlags::NONLOCAL);
                }
            }

            StmtKind::Import(aliases) => {
                for alias in aliases {
                    let name = alias.asname.as_ref().unwrap_or(&alias.name);
                    // Get the first component of the module name
                    let local_name: Arc<str> = if let Some(pos) = name.find('.') {
                        name[..pos].into()
                    } else {
                        name.clone().into()
                    };
                    scope.define(local_name, SymbolFlags::DEF | SymbolFlags::IMPORTED);
                }
            }

            StmtKind::ImportFrom { names, .. } => {
                for alias in names {
                    let name = alias.asname.as_ref().unwrap_or(&alias.name);
                    scope.define(name.clone(), SymbolFlags::DEF | SymbolFlags::IMPORTED);
                }
            }

            StmtKind::Return(value) => {
                if let Some(v) = value {
                    self.visit_expr(scope, v);
                }
            }

            StmtKind::Raise { exc, cause } => {
                if let Some(e) = exc {
                    self.visit_expr(scope, e);
                }
                if let Some(c) = cause {
                    self.visit_expr(scope, c);
                }
            }

            StmtKind::Assert { test, msg } => {
                self.visit_expr(scope, test);
                if let Some(m) = msg {
                    self.visit_expr(scope, m);
                }
            }

            StmtKind::Delete(targets) => {
                for t in targets {
                    self.visit_expr(scope, t);
                }
            }

            StmtKind::Expr(value) => {
                self.visit_expr(scope, value);
            }

            StmtKind::Pass | StmtKind::Break | StmtKind::Continue => {}

            StmtKind::Match { subject, cases } => {
                self.visit_expr(scope, subject);
                for case in cases {
                    // Patterns can bind names
                    self.visit_pattern_bindings(scope, &case.pattern);
                    if let Some(guard) = &case.guard {
                        self.visit_expr(scope, guard);
                    }
                    for s in &case.body {
                        self.visit_stmt(scope, s);
                    }
                }
            }

            StmtKind::TypeAlias { name, value, .. } => {
                self.visit_expr(scope, name);
                self.visit_expr(scope, value);
            }
        }
    }

    /// Visit a statement using the current scope on the stack.
    fn visit_stmt_in_current(&mut self, stmt: &Stmt) {
        // We need to temporarily take the scope to satisfy borrow checker
        let scope = self.scope_stack.pop().expect("no current scope");
        let mut scope = scope;
        self.visit_stmt(&mut scope, stmt);
        self.scope_stack.push(scope);
    }

    /// Visit an expression.
    fn visit_expr(&mut self, scope: &mut Scope, expr: &Expr) {
        match &expr.kind {
            ExprKind::Name(name) => {
                scope.use_symbol(name.clone());
            }

            ExprKind::Lambda { args, body } => {
                // Create lambda scope
                let mut lambda_scope = Scope::new(ScopeKind::Lambda, "<lambda>");

                // Parameters
                for arg in &args.args {
                    lambda_scope.define(arg.arg.clone(), SymbolFlags::PARAM);
                }

                // Defaults in enclosing scope
                for default in &args.defaults {
                    self.visit_expr(scope, default);
                }

                // Analyze body
                self.push_scope(lambda_scope);
                self.visit_expr_in_current(body);
                let mut lambda_scope = self.pop_scope();

                self.classify_unbound(&mut lambda_scope);
                scope.children.push(lambda_scope);
            }

            ExprKind::ListComp { elt, generators }
            | ExprKind::SetComp { elt, generators }
            | ExprKind::GeneratorExp { elt, generators } => {
                // First generator's iter is in enclosing scope
                if let Some(first) = generators.first() {
                    self.visit_expr(scope, &first.iter);
                }

                // Create comprehension scope
                let mut comp_scope = Scope::new(ScopeKind::Comprehension, "<comprehension>");

                for (i, generator) in generators.iter().enumerate() {
                    self.define_target(&mut comp_scope, &generator.target);
                    if i > 0 {
                        self.push_scope(comp_scope);
                        self.visit_expr_in_current(&generator.iter);
                        comp_scope = self.pop_scope();
                    }
                    for cond in &generator.ifs {
                        self.push_scope(comp_scope);
                        self.visit_expr_in_current(cond);
                        comp_scope = self.pop_scope();
                    }
                }

                self.push_scope(comp_scope);
                self.visit_expr_in_current(elt);
                let mut comp_scope = self.pop_scope();

                self.classify_unbound(&mut comp_scope);
                scope.children.push(comp_scope);
            }

            ExprKind::DictComp {
                key,
                value,
                generators,
            } => {
                if let Some(first) = generators.first() {
                    self.visit_expr(scope, &first.iter);
                }

                let mut comp_scope = Scope::new(ScopeKind::Comprehension, "<dictcomp>");

                for (i, generator) in generators.iter().enumerate() {
                    self.define_target(&mut comp_scope, &generator.target);
                    if i > 0 {
                        self.push_scope(comp_scope);
                        self.visit_expr_in_current(&generator.iter);
                        comp_scope = self.pop_scope();
                    }
                    for cond in &generator.ifs {
                        self.push_scope(comp_scope);
                        self.visit_expr_in_current(cond);
                        comp_scope = self.pop_scope();
                    }
                }

                self.push_scope(comp_scope);
                self.visit_expr_in_current(key);
                self.visit_expr_in_current(value);
                let mut comp_scope = self.pop_scope();

                self.classify_unbound(&mut comp_scope);
                scope.children.push(comp_scope);
            }

            ExprKind::BinOp { left, right, .. } => {
                self.visit_expr(scope, left);
                self.visit_expr(scope, right);
            }

            ExprKind::UnaryOp { operand, .. } => {
                self.visit_expr(scope, operand);
            }

            ExprKind::Compare {
                left, comparators, ..
            } => {
                self.visit_expr(scope, left);
                for c in comparators {
                    self.visit_expr(scope, c);
                }
            }

            ExprKind::BoolOp { values, .. } => {
                for v in values {
                    self.visit_expr(scope, v);
                }
            }

            ExprKind::Call {
                func,
                args,
                keywords,
            } => {
                self.visit_expr(scope, func);
                for arg in args {
                    self.visit_expr(scope, arg);
                }
                for kw in keywords {
                    self.visit_expr(scope, &kw.value);
                }
            }

            ExprKind::Attribute { value, .. } => {
                self.visit_expr(scope, value);
            }

            ExprKind::Subscript { value, slice, .. } => {
                self.visit_expr(scope, value);
                self.visit_expr(scope, slice);
            }

            ExprKind::Tuple(elts) | ExprKind::List(elts) | ExprKind::Set(elts) => {
                for e in elts {
                    self.visit_expr(scope, e);
                }
            }

            ExprKind::Dict { keys, values } => {
                for k in keys.iter().flatten() {
                    self.visit_expr(scope, k);
                }
                for v in values {
                    self.visit_expr(scope, v);
                }
            }

            ExprKind::IfExp { test, body, orelse } => {
                self.visit_expr(scope, test);
                self.visit_expr(scope, body);
                self.visit_expr(scope, orelse);
            }

            ExprKind::Starred(value) => {
                self.visit_expr(scope, value);
            }

            ExprKind::Slice { lower, upper, step } => {
                if let Some(l) = lower {
                    self.visit_expr(scope, l);
                }
                if let Some(u) = upper {
                    self.visit_expr(scope, u);
                }
                if let Some(s) = step {
                    self.visit_expr(scope, s);
                }
            }

            ExprKind::Await(value) => {
                scope.has_await = true;
                self.visit_expr(scope, value);
            }

            ExprKind::Yield(value) => {
                scope.has_yield = true;
                if let Some(v) = value {
                    self.visit_expr(scope, v);
                }
            }

            ExprKind::YieldFrom(value) => {
                scope.has_yield = true;
                self.visit_expr(scope, value);
            }

            ExprKind::NamedExpr { target, value } => {
                // Named expression target is defined in enclosing non-comprehension scope
                // For now, just define in current scope
                if let ExprKind::Name(name) = &target.kind {
                    scope.define(name.clone(), SymbolFlags::DEF);
                }
                self.visit_expr(scope, value);
            }

            ExprKind::FormattedValue {
                value, format_spec, ..
            } => {
                self.visit_expr(scope, value);
                if let Some(spec) = format_spec {
                    self.visit_expr(scope, spec);
                }
            }

            ExprKind::JoinedStr(parts) => {
                for part in parts {
                    self.visit_expr(scope, part);
                }
            }

            // Literals don't reference any names
            ExprKind::Int(_)
            | ExprKind::BigInt(_)
            | ExprKind::Float(_)
            | ExprKind::Complex { .. }
            | ExprKind::String(_)
            | ExprKind::Bytes(_)
            | ExprKind::Bool(_)
            | ExprKind::None
            | ExprKind::Ellipsis => {}
        }
    }

    /// Visit an expression using the current scope on the stack.
    fn visit_expr_in_current(&mut self, expr: &Expr) {
        let scope = self.scope_stack.pop().expect("no current scope");
        let mut scope = scope;
        self.visit_expr(&mut scope, expr);
        self.scope_stack.push(scope);
    }

    /// Define names from an assignment target.
    fn define_target(&mut self, scope: &mut Scope, target: &Expr) {
        match &target.kind {
            ExprKind::Name(name) => {
                scope.define(name.clone(), SymbolFlags::DEF);
            }
            ExprKind::Tuple(elts) | ExprKind::List(elts) => {
                for e in elts {
                    self.define_target(scope, e);
                }
            }
            ExprKind::Starred(value) => {
                self.define_target(scope, value);
            }
            ExprKind::Attribute { value, .. } | ExprKind::Subscript { value, .. } => {
                // These don't define new names, but we need to visit the value
                self.visit_expr(scope, value);
            }
            _ => {}
        }
    }

    /// Visit pattern bindings in match statements.
    fn visit_pattern_bindings(&mut self, scope: &mut Scope, pattern: &Pattern) {
        match &pattern.kind {
            PatternKind::MatchAs { name, pattern } => {
                if let Some(n) = name {
                    scope.define(n.clone(), SymbolFlags::DEF);
                }
                if let Some(p) = pattern {
                    self.visit_pattern_bindings(scope, p);
                }
            }
            PatternKind::MatchStar(name) => {
                if let Some(n) = name {
                    scope.define(n.clone(), SymbolFlags::DEF);
                }
            }
            PatternKind::MatchSequence(patterns) => {
                for p in patterns {
                    self.visit_pattern_bindings(scope, p);
                }
            }
            PatternKind::MatchOr(patterns) => {
                // All alternatives must bind the same names
                for p in patterns {
                    self.visit_pattern_bindings(scope, p);
                }
            }
            PatternKind::MatchMapping {
                keys,
                patterns,
                rest,
            } => {
                for p in patterns {
                    self.visit_pattern_bindings(scope, p);
                }
                if let Some(r) = rest {
                    scope.define(r.clone(), SymbolFlags::DEF);
                }
                for k in keys {
                    self.visit_expr(scope, k);
                }
            }
            PatternKind::MatchClass {
                cls,
                patterns,
                kwd_patterns,
                ..
            } => {
                self.visit_expr(scope, cls);
                for p in patterns {
                    self.visit_pattern_bindings(scope, p);
                }
                for p in kwd_patterns {
                    self.visit_pattern_bindings(scope, p);
                }
            }
            PatternKind::MatchValue(expr) => {
                self.visit_expr(scope, expr);
            }
            PatternKind::MatchSingleton(_) => {}
        }
    }

    /// Classify unbound names as global (implicit) or free.
    fn classify_unbound(&self, scope: &mut Scope) {
        // In module scope, unbound names are global
        if scope.kind == ScopeKind::Module {
            for sym in scope.symbols.values_mut() {
                if !sym.flags.contains(SymbolFlags::DEF) && sym.flags.contains(SymbolFlags::USE) {
                    sym.flags |= SymbolFlags::GLOBAL_IMPLICIT;
                }
            }
        } else {
            // In other scopes, unbound names become free variables
            for sym in scope.symbols.values_mut() {
                if !sym.flags.contains(SymbolFlags::DEF)
                    && sym.flags.contains(SymbolFlags::USE)
                    && !sym.flags.contains(SymbolFlags::GLOBAL_EXPLICIT)
                {
                    sym.flags |= SymbolFlags::FREE;
                }
            }
        }

        // TODO: Mark corresponding symbols as CELL in enclosing scopes
    }
}

impl Default for ScopeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyze(source: &str) -> SymbolTable {
        let module = prism_parser::parse(source).expect("parse error");
        ScopeAnalyzer::new().analyze(&module, "<test>")
    }

    #[test]
    fn test_simple_assignment() {
        let table = analyze("x = 1\nprint(x)");
        let x = table.root.lookup("x").unwrap();
        assert!(x.flags.contains(SymbolFlags::DEF));
        assert!(x.flags.contains(SymbolFlags::USE));
    }

    #[test]
    fn test_function_scope() {
        let table = analyze("def foo(a, b):\n    x = a + b\n    return x");
        let foo = table.root.lookup("foo").unwrap();
        assert!(foo.flags.contains(SymbolFlags::DEF));

        assert_eq!(table.root.children.len(), 1);
        let func_scope = &table.root.children[0];
        assert_eq!(&*func_scope.name, "foo");

        let a = func_scope.lookup("a").unwrap();
        assert!(a.flags.contains(SymbolFlags::PARAM));

        let x = func_scope.lookup("x").unwrap();
        assert!(x.is_local());
    }

    #[test]
    fn test_global_declaration() {
        let table = analyze("x = 1\ndef foo():\n    global x\n    x = 2");

        let func_scope = &table.root.children[0];
        let x = func_scope.lookup("x").unwrap();
        assert!(x.flags.contains(SymbolFlags::GLOBAL_EXPLICIT));
    }
}
