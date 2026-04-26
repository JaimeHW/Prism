//! AST to register-based bytecode compiler.
//!
//! The compiler transforms a parsed Python AST into executable bytecode
//! using a two-phase approach:
//!
//! 1. **Scope analysis**: Builds symbol tables and determines variable scopes
//! 2. **Code generation**: Emits register-based bytecode instructions

use crate::bytecode::{
    CodeFlags, CodeObject, FunctionBuilder, Instruction, Label, LocalSlot, Opcode, Register,
};
use crate::function_compiler::{VarLocation, VariableEmitter};
use crate::scope::{ClosureAnalyzer, ScopeAnalyzer, ScopeKind, SymbolFlags, SymbolTable};

use num_bigint::BigInt;
use prism_core::Span;
use prism_parser::ast::{
    AugOp, BinOp, BoolOp, CmpOp, ExceptHandler, Expr, ExprKind, Module, Stmt, StmtKind, UnaryOp,
};
use smallvec::SmallVec;
use std::sync::Arc;

/// Stack-allocated loop context stack for typical loop nesting depths.
/// Most code has ≤4 nested loops, so we avoid heap allocation in the common case.
type LoopStack = SmallVec<[LoopContext; 4]>;
/// Stack-allocated finally context stack for typical nesting depths.
type FinallyStack = SmallVec<[FinallyContext; 4]>;
/// BuildSlice step-extension marker byte 1 (`CallKwEx.dst = step register`).
const SLICE_STEP_EXT_TAG_A: u8 = b'S';
/// BuildSlice step-extension marker byte 2.
const SLICE_STEP_EXT_TAG_B: u8 = b'L';

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SequenceLiteralKind {
    List,
    Tuple,
    Set,
}

fn stmt_kind_name(kind: &StmtKind) -> &'static str {
    match kind {
        StmtKind::Expr(_) => "Expr",
        StmtKind::Assign { .. } => "Assign",
        StmtKind::AugAssign { .. } => "AugAssign",
        StmtKind::AnnAssign { .. } => "AnnAssign",
        StmtKind::Return(_) => "Return",
        StmtKind::Delete(_) => "Delete",
        StmtKind::Pass => "Pass",
        StmtKind::Break => "Break",
        StmtKind::Continue => "Continue",
        StmtKind::Raise { .. } => "Raise",
        StmtKind::Assert { .. } => "Assert",
        StmtKind::Global(_) => "Global",
        StmtKind::Nonlocal(_) => "Nonlocal",
        StmtKind::Import(_) => "Import",
        StmtKind::ImportFrom { .. } => "ImportFrom",
        StmtKind::If { .. } => "If",
        StmtKind::For { .. } => "For",
        StmtKind::AsyncFor { .. } => "AsyncFor",
        StmtKind::While { .. } => "While",
        StmtKind::With { .. } => "With",
        StmtKind::AsyncWith { .. } => "AsyncWith",
        StmtKind::Try { .. } => "Try",
        StmtKind::TryStar { .. } => "TryStar",
        StmtKind::Match { .. } => "Match",
        StmtKind::FunctionDef { .. } => "FunctionDef",
        StmtKind::AsyncFunctionDef { .. } => "AsyncFunctionDef",
        StmtKind::ClassDef { .. } => "ClassDef",
        StmtKind::TypeAlias { .. } => "TypeAlias",
    }
}

fn expr_kind_name(kind: &ExprKind) -> &'static str {
    match kind {
        ExprKind::Int(_) => "Int",
        ExprKind::BigInt(_) => "BigInt",
        ExprKind::Float(_) => "Float",
        ExprKind::Complex { .. } => "Complex",
        ExprKind::String(_) => "String",
        ExprKind::Bytes(_) => "Bytes",
        ExprKind::Bool(_) => "Bool",
        ExprKind::None => "None",
        ExprKind::Ellipsis => "Ellipsis",
        ExprKind::Name(_) => "Name",
        ExprKind::NamedExpr { .. } => "NamedExpr",
        ExprKind::List(_) => "List",
        ExprKind::Tuple(_) => "Tuple",
        ExprKind::Set(_) => "Set",
        ExprKind::Dict { .. } => "Dict",
        ExprKind::ListComp { .. } => "ListComp",
        ExprKind::SetComp { .. } => "SetComp",
        ExprKind::DictComp { .. } => "DictComp",
        ExprKind::GeneratorExp { .. } => "GeneratorExp",
        ExprKind::BinOp { .. } => "BinOp",
        ExprKind::UnaryOp { .. } => "UnaryOp",
        ExprKind::BoolOp { .. } => "BoolOp",
        ExprKind::Compare { .. } => "Compare",
        ExprKind::Attribute { .. } => "Attribute",
        ExprKind::Subscript { .. } => "Subscript",
        ExprKind::Slice { .. } => "Slice",
        ExprKind::Starred(_) => "Starred",
        ExprKind::Call { .. } => "Call",
        ExprKind::Lambda { .. } => "Lambda",
        ExprKind::IfExp { .. } => "IfExp",
        ExprKind::Await(_) => "Await",
        ExprKind::Yield(_) => "Yield",
        ExprKind::YieldFrom(_) => "YieldFrom",
        ExprKind::JoinedStr(_) => "JoinedStr",
        ExprKind::FormattedValue { .. } => "FormattedValue",
    }
}

fn pattern_kind_name(kind: &prism_parser::ast::PatternKind) -> &'static str {
    use prism_parser::ast::PatternKind;

    match kind {
        PatternKind::MatchValue(_) => "MatchValue",
        PatternKind::MatchSingleton(_) => "MatchSingleton",
        PatternKind::MatchSequence(_) => "MatchSequence",
        PatternKind::MatchMapping { .. } => "MatchMapping",
        PatternKind::MatchClass { .. } => "MatchClass",
        PatternKind::MatchStar(_) => "MatchStar",
        PatternKind::MatchAs { .. } => "MatchAs",
        PatternKind::MatchOr(_) => "MatchOr",
    }
}

#[derive(Debug, Clone)]
struct SourceLineMap {
    line_starts: Arc<[u32]>,
}

impl SourceLineMap {
    fn new(source: &str) -> Self {
        let mut line_starts =
            Vec::with_capacity(source.bytes().filter(|byte| *byte == b'\n').count() + 1);
        line_starts.push(0);
        for (index, byte) in source.bytes().enumerate() {
            if byte == b'\n' {
                let next = index.saturating_add(1).min(u32::MAX as usize) as u32;
                line_starts.push(next);
            }
        }
        Self {
            line_starts: line_starts.into_boxed_slice().into(),
        }
    }

    fn line_for_offset(&self, offset: u32) -> u32 {
        self.line_starts.partition_point(|start| *start <= offset) as u32
    }
}

/// Compilation error.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// Error message.
    pub message: String,
    /// Line number (1-indexed).
    pub line: u32,
    /// Column number (0-indexed).
    pub column: u32,
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for CompileError {}

/// Result type for compilation.
pub type CompileResult<T> = Result<T, CompileError>;

/// Compiler optimization level.
///
/// Mirrors Python's `-O` / `-OO` behavior:
/// - `None`: no optimization
/// - `Basic` (`-O`): strip assert statements
/// - `Full` (`-OO`): strip asserts and docstrings
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum OptimizationLevel {
    /// No optimization.
    #[default]
    None = 0,
    /// `-O`: strip assert statements.
    Basic = 1,
    /// `-OO`: strip asserts and docstrings.
    Full = 2,
}

/// Namespace semantics for module code generation.
///
/// Regular module compilation binds top-level assignments directly into module
/// globals. Dynamic `exec`/`eval` with an explicit locals mapping instead needs
/// name access that targets the provided mapping while still preserving module
/// closure rules for nested functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModuleNamespaceMode {
    /// Emit ordinary module-global name access.
    #[default]
    Standard,
    /// Emit top-level local slots so execution can route name access through a
    /// live locals mapping.
    DynamicLocals,
}

// =============================================================================
// Loop Context for break/continue
// =============================================================================

/// Context for tracking loop nesting and jump targets.
///
/// Each loop (while, for) pushes a context onto the stack to enable
/// break and continue statements to emit correct jump instructions.
#[derive(Debug, Clone, Copy)]
struct LoopContext {
    /// Label to jump to for `break` (after loop body, before else clause).
    /// Note: break skips the else clause in Python.
    break_label: Label,
    /// Label to jump to for `continue` (back to loop condition/iterator).
    continue_label: Label,
    /// Number of active cleanup scopes at loop entry.
    ///
    /// `break` and `continue` targets stay inside any cleanup scopes that
    /// already enclosed the loop, but must unwind cleanup scopes entered within
    /// the loop body.
    finally_depth: usize,
}

/// Deferred jump that must run a finally body before continuing.
#[derive(Debug, Clone, Copy)]
struct FinallyJumpContinuation {
    /// Cleanup block label.
    cleanup_label: Label,
    /// Original jump target after cleanup.
    target_label: Label,
    /// Cleanup stack depth that should remain active for the jump target.
    preserve_finally_depth: usize,
}

/// Context for routing control flow through an active finally block.
#[derive(Debug)]
struct FinallyContext {
    /// Shared cleanup block for returns from the protected body.
    return_label: Label,
    /// Register that carries the pending return value into `return_label`.
    return_value_reg: Register,
    /// True once a return statement has targeted `return_label`.
    return_used: bool,
    /// Break/continue continuations that need this finally body before jumping.
    jump_continuations: SmallVec<[FinallyJumpContinuation; 4]>,
}

// =============================================================================
// Compiler
// =============================================================================

/// Bytecode compiler.
pub struct Compiler {
    /// Current function builder.
    builder: FunctionBuilder,
    /// Symbol table from scope analysis.
    symbol_table: SymbolTable,
    /// Source filename.
    #[allow(dead_code)]
    filename: Arc<str>,
    /// Optional source map for converting parser byte offsets to source lines.
    line_map: Option<SourceLineMap>,
    /// Stack of active loop contexts for break/continue.
    /// Innermost loop is at the end (top) of the stack.
    /// Uses SmallVec to avoid heap allocation for typical nesting depths (≤4).
    loop_stack: LoopStack,
    /// Stack of active finally contexts for non-exceptional control flow.
    finally_stack: FinallyStack,
    /// Whether we are inside an async function.
    /// This is used to validate `await` expressions and `async for`/`async with` statements.
    in_async_context: bool,
    /// Whether we are inside a function/generator context.
    /// This is used to validate `yield` expressions.
    in_function_context: bool,
    /// Compiler optimization level.
    optimize: OptimizationLevel,
    /// Namespace lowering policy for module-scope names.
    module_namespace_mode: ModuleNamespaceMode,
    /// Path to the current scope in the symbol table.
    /// Empty path means module root scope.
    scope_path: Vec<usize>,
    /// Per-scope child cursor for deterministic nested-scope lookup.
    /// Indexed by scope depth (`scope_path.len()`).
    scope_child_offsets: Vec<usize>,
}

impl Compiler {
    /// Create a new compiler for a module.
    pub fn new(filename: impl Into<Arc<str>>) -> Self {
        Self::new_with_optimization(filename, OptimizationLevel::None)
    }

    /// Create a new compiler for a module with an explicit optimization level.
    pub fn new_with_optimization(
        filename: impl Into<Arc<str>>,
        optimize: OptimizationLevel,
    ) -> Self {
        let filename = filename.into();
        Self {
            builder: FunctionBuilder::new("<module>"),
            symbol_table: SymbolTable::new("<module>"),
            filename,
            line_map: None,
            loop_stack: LoopStack::new(),
            finally_stack: FinallyStack::new(),
            in_async_context: false,
            in_function_context: false,
            optimize,
            module_namespace_mode: ModuleNamespaceMode::Standard,
            scope_path: Vec::new(),
            scope_child_offsets: vec![0],
        }
    }

    /// Create a new compiler with source-backed line mapping.
    pub fn new_with_source(
        filename: impl Into<Arc<str>>,
        source: &str,
        optimize: OptimizationLevel,
    ) -> Self {
        let mut compiler = Self::new_with_optimization(filename, optimize);
        compiler.line_map = Some(SourceLineMap::new(source));
        compiler
    }

    /// Compile a module to bytecode.
    pub fn compile_module(module: &Module, filename: &str) -> CompileResult<CodeObject> {
        Self::compile_module_with_optimization(module, filename, OptimizationLevel::None)
    }

    /// Compile a module to bytecode with explicit namespace lowering semantics.
    pub fn compile_module_with_namespace_mode(
        module: &Module,
        filename: &str,
        optimize: OptimizationLevel,
        module_namespace_mode: ModuleNamespaceMode,
    ) -> CompileResult<CodeObject> {
        Self::compile_module_with_line_map(module, filename, optimize, module_namespace_mode, None)
    }

    /// Compile a source-backed module to bytecode with explicit namespace
    /// lowering semantics and precise parser-offset to source-line mapping.
    pub fn compile_module_with_source_and_namespace_mode(
        module: &Module,
        source: &str,
        filename: &str,
        optimize: OptimizationLevel,
        module_namespace_mode: ModuleNamespaceMode,
    ) -> CompileResult<CodeObject> {
        Self::compile_module_with_line_map(
            module,
            filename,
            optimize,
            module_namespace_mode,
            Some(SourceLineMap::new(source)),
        )
    }

    fn compile_module_with_line_map(
        module: &Module,
        filename: &str,
        optimize: OptimizationLevel,
        module_namespace_mode: ModuleNamespaceMode,
        line_map: Option<SourceLineMap>,
    ) -> CompileResult<CodeObject> {
        // Phase 1: Scope analysis
        let mut symbol_table = ScopeAnalyzer::new().analyze(module, "<module>");
        ClosureAnalyzer::new().analyze(&mut symbol_table.root);

        // Phase 2: Code generation
        let mut compiler = Compiler {
            builder: FunctionBuilder::new("<module>"),
            symbol_table,
            filename: filename.into(),
            line_map,
            loop_stack: LoopStack::new(),
            finally_stack: FinallyStack::new(),
            in_async_context: false,
            in_function_context: false,
            optimize,
            module_namespace_mode,
            scope_path: Vec::new(),
            scope_child_offsets: vec![0],
        };

        compiler.builder.set_filename(filename);
        compiler.builder.add_flags(CodeFlags::MODULE);
        if Self::body_needs_runtime_annotations(&module.body) {
            compiler.builder.emit_setup_annotations();
        }

        for (index, stmt) in module.body.iter().enumerate() {
            if compiler.should_strip_docstring_stmt(index, stmt) {
                continue;
            }
            compiler.compile_stmt(stmt)?;
        }

        // Implicit return None at end of module
        compiler.builder.emit_return_none();

        Ok(compiler.builder.finish())
    }

    /// Compile a module to bytecode with an explicit optimization level.
    pub fn compile_module_with_optimization(
        module: &Module,
        filename: &str,
        optimize: OptimizationLevel,
    ) -> CompileResult<CodeObject> {
        Self::compile_module_with_namespace_mode(
            module,
            filename,
            optimize,
            ModuleNamespaceMode::Standard,
        )
    }

    /// Whether this statement is a docstring candidate.
    #[inline]
    fn is_docstring_stmt(stmt: &Stmt) -> bool {
        matches!(&stmt.kind, StmtKind::Expr(expr) if matches!(&expr.kind, ExprKind::String(_)))
    }

    /// Whether to strip this statement as a docstring under `-OO`.
    #[inline]
    fn should_strip_docstring_stmt(&self, index: usize, stmt: &Stmt) -> bool {
        self.optimize >= OptimizationLevel::Full && index == 0 && Self::is_docstring_stmt(stmt)
    }

    fn body_needs_runtime_annotations(body: &[Stmt]) -> bool {
        body.iter().any(Self::stmt_needs_runtime_annotations)
    }

    fn stmt_needs_runtime_annotations(stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::AnnAssign { target, simple, .. } => {
                *simple && matches!(&target.kind, ExprKind::Name(_))
            }
            StmtKind::If { body, orelse, .. } | StmtKind::While { body, orelse, .. } => {
                Self::body_needs_runtime_annotations(body)
                    || Self::body_needs_runtime_annotations(orelse)
            }
            StmtKind::For { body, orelse, .. } | StmtKind::AsyncFor { body, orelse, .. } => {
                Self::body_needs_runtime_annotations(body)
                    || Self::body_needs_runtime_annotations(orelse)
            }
            StmtKind::With { body, .. } | StmtKind::AsyncWith { body, .. } => {
                Self::body_needs_runtime_annotations(body)
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
                Self::body_needs_runtime_annotations(body)
                    || handlers
                        .iter()
                        .any(|handler| Self::body_needs_runtime_annotations(&handler.body))
                    || Self::body_needs_runtime_annotations(orelse)
                    || Self::body_needs_runtime_annotations(finalbody)
            }
            StmtKind::Match { cases, .. } => cases
                .iter()
                .any(|case| Self::body_needs_runtime_annotations(&case.body)),
            StmtKind::FunctionDef { .. }
            | StmtKind::AsyncFunctionDef { .. }
            | StmtKind::ClassDef { .. } => false,
            _ => false,
        }
    }

    #[inline]
    fn line_for_span(&self, span: Span) -> u32 {
        self.line_map
            .as_ref()
            .map(|line_map| line_map.line_for_offset(span.start))
            .unwrap_or_else(|| span.start.max(1))
    }

    /// Emit a return, routing it through active finally blocks when needed.
    fn emit_return_value(&mut self, value_reg: Register) {
        if let Some(finally_ctx) = self.finally_stack.last_mut() {
            finally_ctx.return_used = true;
            let return_value_reg = finally_ctx.return_value_reg;
            let return_label = finally_ctx.return_label;
            self.builder.emit_move(return_value_reg, value_reg);
            self.builder.emit_jump(return_label);
        } else {
            self.builder.emit_return(value_reg);
        }
    }

    /// Emit `return None`, routing it through active finally blocks when needed.
    fn emit_return_none_value(&mut self) {
        if let Some(finally_ctx) = self.finally_stack.last_mut() {
            finally_ctx.return_used = true;
            let return_value_reg = finally_ctx.return_value_reg;
            let return_label = finally_ctx.return_label;
            self.builder.emit_load_none(return_value_reg);
            self.builder.emit_jump(return_label);
        } else {
            self.builder.emit_return_none();
        }
    }

    /// Emit a jump, routing it through active cleanup blocks when needed.
    fn emit_jump_through_finally_until(
        &mut self,
        target_label: Label,
        preserve_finally_depth: usize,
    ) {
        if self.finally_stack.len() <= preserve_finally_depth {
            self.builder.emit_jump(target_label);
            return;
        }

        let cleanup_label = self.builder.create_label();
        self.finally_stack
            .last_mut()
            .expect("finally stack checked above")
            .jump_continuations
            .push(FinallyJumpContinuation {
                cleanup_label,
                target_label,
                preserve_finally_depth,
            });
        self.builder.emit_jump(cleanup_label);
    }

    /// Compile finally cleanup for a deferred control-flow path.
    fn compile_finally_cleanup_body(&mut self, finalbody: &[Stmt]) -> CompileResult<()> {
        for stmt in finalbody {
            self.compile_stmt(stmt)?;
        }
        Ok(())
    }

    /// Emit a non-exceptional context-manager exit call.
    fn emit_context_exit_none(&mut self, exit_method_reg: Register) {
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(exit_method_reg.0 + 2),
        ));
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(exit_method_reg.0 + 3),
        ));
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(exit_method_reg.0 + 4),
        ));

        let exit_result_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(exit_result_reg, exit_method_reg, 3);
        self.builder.free_register(exit_result_reg);
    }

    /// Emit an awaited non-exceptional async context-manager exit call.
    fn emit_async_context_exit_none(&mut self, aexit_method_reg: Register) {
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(aexit_method_reg.0 + 2),
        ));
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(aexit_method_reg.0 + 3),
        ));
        self.builder.emit(Instruction::op_d(
            Opcode::LoadNone,
            Register::new(aexit_method_reg.0 + 4),
        ));

        let aexit_awaitable_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(aexit_awaitable_reg, aexit_method_reg, 3);

        let aexit_result_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            aexit_result_reg,
            aexit_awaitable_reg,
        ));
        self.builder.free_register(aexit_awaitable_reg);
        self.emit_yield_from(aexit_result_reg, aexit_result_reg);
        self.builder.free_register(aexit_result_reg);
    }

    /// Emit the concrete class load used for context-manager exception exits.
    #[inline]
    fn emit_exception_type_attr(&mut self, dst: Register, exception: Register) {
        let class_name_idx = self.builder.add_name("__class__");
        self.builder.emit_get_attr(dst, exception, class_name_idx);
    }

    /// Resolve a variable name to its location.
    ///
    /// Uses the symbol table to determine whether the variable is local,
    /// closure (cell/free), or global.
    ///
    /// For nested functions, also checks the builder's local map for
    /// parameters and locals defined via define_local().
    fn resolve_variable(&mut self, name: &str) -> VarLocation {
        let in_module_scope = self.current_scope().kind == ScopeKind::Module;
        let use_dynamic_module_locals =
            in_module_scope && self.module_namespace_mode == ModuleNamespaceMode::DynamicLocals;
        let symbol_info = self.current_scope().lookup(name).map(|symbol| {
            (
                symbol.flags.contains(SymbolFlags::GLOBAL_EXPLICIT),
                symbol.is_cell(),
                symbol.is_free(),
                symbol.is_local(),
                symbol.closure_slot,
            )
        });

        if let Some((is_explicit_global, is_cell, is_free, is_local, closure_slot)) = symbol_info {
            if (is_cell || is_free) && closure_slot.is_some() {
                return VarLocation::Closure(closure_slot.unwrap());
            }

            if is_explicit_global {
                return VarLocation::Global;
            }

            if use_dynamic_module_locals && !is_explicit_global {
                if let Some(slot) = self.builder.lookup_local(name) {
                    return VarLocation::Local(slot.0);
                }
                return VarLocation::Local(self.builder.define_local(name).0);
            }

            if is_local && !is_cell {
                // Module-scope names are always global namespace entries.
                if in_module_scope && !use_dynamic_module_locals {
                    return VarLocation::Global;
                }
                if let Some(slot) = self.builder.lookup_local(name) {
                    return VarLocation::Local(slot.0);
                }
                // Lazily materialize locals discovered by scope analysis.
                return VarLocation::Local(self.builder.define_local(name).0);
            }
        }

        if use_dynamic_module_locals {
            if let Some(slot) = self.builder.lookup_local(name) {
                return VarLocation::Local(slot.0);
            }
            return VarLocation::Local(self.builder.define_local(name).0);
        }

        if !in_module_scope || use_dynamic_module_locals {
            if let Some(slot) = self.builder.lookup_local(name) {
                return VarLocation::Local(slot.0);
            }
        }

        // Fall back to global for undefined or explicitly global symbols.
        VarLocation::Global
    }

    /// Get the current scope from the symbol table.
    fn current_scope(&self) -> &crate::scope::Scope {
        let mut scope = &self.symbol_table.root;
        for &child_idx in &self.scope_path {
            scope = &scope.children[child_idx];
        }
        scope
    }

    fn qualified_name_for_nested_definition(&self, name: &str) -> Arc<str> {
        if self.scope_path.is_empty() {
            return Arc::from(name);
        }

        let mut scope = &self.symbol_table.root;
        let mut qualname = String::new();

        for &child_idx in &self.scope_path {
            scope = &scope.children[child_idx];
            if scope.kind == ScopeKind::Module {
                continue;
            }

            if !qualname.is_empty() {
                qualname.push('.');
            }
            qualname.push_str(scope.name.as_ref());

            if matches!(
                scope.kind,
                ScopeKind::Function | ScopeKind::Lambda | ScopeKind::Comprehension
            ) {
                qualname.push_str(".<locals>");
            }
        }

        if !qualname.is_empty() {
            qualname.push('.');
        }
        qualname.push_str(name);
        Arc::from(qualname)
    }

    fn emit_implicit_class_metadata_bindings(&mut self, qualname: &str) {
        let module_slot = self.builder.define_local("__module__");
        let qualname_slot = self.builder.define_local("__qualname__");

        let module_reg = self.builder.alloc_register();
        let module_name_idx = self.builder.add_name("__name__");
        self.builder.emit_load_global(module_reg, module_name_idx);
        self.builder.emit_store_local(module_slot, module_reg);
        self.builder.free_register(module_reg);

        let qualname_reg = self.builder.alloc_register();
        let qualname_idx = self.builder.add_string(qualname);
        self.builder.emit_load_const(qualname_reg, qualname_idx);
        self.builder.emit_store_local(qualname_slot, qualname_reg);
        self.builder.free_register(qualname_reg);
    }

    fn ordered_cellvar_names(scope: &crate::scope::Scope) -> Vec<Arc<str>> {
        scope
            .ordered_cellvars()
            .into_iter()
            .map(|symbol| Arc::clone(&symbol.name))
            .collect()
    }

    fn ordered_freevar_names(scope: &crate::scope::Scope) -> Vec<Arc<str>> {
        scope
            .ordered_freevars()
            .into_iter()
            .map(|symbol| Arc::clone(&symbol.name))
            .collect()
    }

    /// Enter a child scope by index.
    fn enter_child_scope(&mut self, child_idx: usize) {
        self.scope_path.push(child_idx);
        self.scope_child_offsets.push(0);
    }

    /// Exit the current scope.
    fn exit_child_scope(&mut self) {
        let _ = self.scope_path.pop();
        if self.scope_child_offsets.len() > 1 {
            self.scope_child_offsets.pop();
        }
    }

    /// Compile a statement.
    fn compile_stmt(&mut self, stmt: &Stmt) -> CompileResult<()> {
        let stmt_line = self.line_for_span(stmt.span);
        self.builder.set_line(stmt_line);

        match &stmt.kind {
            StmtKind::Expr(value) => {
                // Expression statement - evaluate and discard
                let reg = self.compile_expr(value)?;
                self.builder.free_register(reg);
            }

            StmtKind::Assign { targets, value } => {
                let value_reg = self.compile_expr(value)?;

                for target in targets {
                    self.compile_store(target, value_reg)?;
                }

                self.builder.free_register(value_reg);
            }

            StmtKind::AnnAssign {
                target,
                annotation,
                value,
                simple,
            } => {
                self.compile_annotated_assignment(
                    stmt,
                    target,
                    annotation,
                    value.as_deref(),
                    *simple,
                )?;
            }

            StmtKind::AugAssign { target, op, value } => {
                self.validate_augassign_target(target)?;
                let left_reg = self.compile_expr(target)?;
                let right_reg = self.compile_expr(value)?;
                let result_reg = self.builder.alloc_register();

                self.emit_augop(*op, result_reg, left_reg, right_reg);

                self.compile_store(target, result_reg)?;

                self.builder.free_register(left_reg);
                self.builder.free_register(right_reg);
                self.builder.free_register(result_reg);
            }

            StmtKind::Return(value) => {
                if let Some(val) = value {
                    let reg = self.compile_expr(val)?;
                    self.emit_return_value(reg);
                    self.builder.free_register(reg);
                } else {
                    self.emit_return_none_value();
                }
            }

            StmtKind::Pass => {
                // No-op
            }

            StmtKind::Break => {
                // Jump to the break label of the innermost loop
                if let Some(ctx) = self.loop_stack.last().copied() {
                    self.emit_jump_through_finally_until(ctx.break_label, ctx.finally_depth);
                } else {
                    return Err(CompileError {
                        message: "'break' outside loop".to_string(),
                        line: stmt_line,
                        column: 0,
                    });
                }
            }

            StmtKind::Continue => {
                // Jump to the continue label of the innermost loop
                if let Some(ctx) = self.loop_stack.last().copied() {
                    self.emit_jump_through_finally_until(ctx.continue_label, ctx.finally_depth);
                } else {
                    return Err(CompileError {
                        message: "'continue' outside loop".to_string(),
                        line: stmt_line,
                        column: 0,
                    });
                }
            }

            StmtKind::If { test, body, orelse } => {
                let cond_reg = self.compile_expr(test)?;
                let else_label = self.builder.create_label();
                let end_label = self.builder.create_label();

                self.builder.emit_jump_if_false(cond_reg, else_label);
                self.builder.free_register(cond_reg);

                // Then branch
                for s in body {
                    self.compile_stmt(s)?;
                }

                if !orelse.is_empty() {
                    self.builder.emit_jump(end_label);
                }

                self.builder.bind_label(else_label);

                // Else branch
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                if !orelse.is_empty() {
                    self.builder.bind_label(end_label);
                }
            }

            StmtKind::While { test, body, orelse } => {
                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                // Push loop context for break/continue
                // break jumps to loop_end, continue jumps to loop_start
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                    finally_depth: self.finally_stack.len(),
                });

                self.builder.bind_label(loop_start);

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_false(cond_reg, loop_else);
                self.builder.free_register(cond_reg);

                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                // Else clause only executes if loop completed normally (no break)
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Pop loop context
                self.loop_stack.pop();
            }

            StmtKind::For {
                target,
                iter,
                body,
                orelse,
            } => {
                // Compile iterator
                let iter_reg = self.compile_expr(iter)?;
                let loop_regs = self.builder.alloc_register_block(2);
                let iterator_reg = loop_regs;
                let item_reg = Register::new(loop_regs.0 + 1);
                self.builder.emit_get_iter(iterator_reg, iter_reg);
                self.builder.free_register(iter_reg);

                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                // Push loop context for break/continue
                // break jumps to loop_end, continue jumps to loop_start
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                    finally_depth: self.finally_stack.len(),
                });

                self.builder.bind_label(loop_start);

                // Get next item
                self.builder.emit_for_iter(item_reg, loop_else);

                // Store to target
                self.compile_store(target, item_reg)?;
                self.builder.free_register(item_reg);

                // Loop body
                for s in body {
                    self.compile_stmt(s)?;
                }

                self.builder.emit_jump(loop_start);
                self.builder.bind_label(loop_else);

                // Else clause only executes if loop completed normally (no break)
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Pop loop context
                self.loop_stack.pop();

                self.builder.free_register(iterator_reg);
            }

            // =========================================================================
            // Async For Loop (PEP 492)
            // =========================================================================
            // `async for target in iter:` compiles to:
            //   aiter = GetAIter(iter)            # Call __aiter__
            //   loop:
            //     try:
            //       anext_awaitable = GetANext(aiter)  # Call __anext__
            //       awaitable = GetAwaitable(anext_awaitable)
            //       result = YieldFrom(awaitable)  # Await the result
            //     except StopAsyncIteration:       # EndAsyncFor handles this
            //       goto else_block
            //     target = result
            //     <body>
            //     goto loop
            //   else_block:
            //     <orelse>
            //   end_block:
            // =========================================================================
            StmtKind::AsyncFor {
                target,
                iter,
                body,
                orelse,
            } => {
                // Validate async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'async for' outside async function".to_string(),
                        line: stmt_line,
                        column: 0,
                    });
                }

                // Step 1: Get async iterator
                let iter_expr = self.compile_expr(iter)?;
                let aiter_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetAIter, aiter_reg, iter_expr));
                self.builder.free_register(iter_expr);

                // Create labels
                let loop_start = self.builder.create_label();
                let loop_else = self.builder.create_label();
                let loop_end = self.builder.create_label();

                // Push loop context
                self.loop_stack.push(LoopContext {
                    break_label: loop_end,
                    continue_label: loop_start,
                    finally_depth: self.finally_stack.len(),
                });

                self.builder.bind_label(loop_start);

                // Step 2: Get next awaitable from async iterator
                let anext_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetANext, anext_reg, aiter_reg));

                // Step 3: Convert to awaitable and await
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAwaitable,
                    anext_reg,
                    anext_reg,
                ));
                self.emit_yield_from(anext_reg, anext_reg);

                // Step 4: Check for StopAsyncIteration
                // EndAsyncFor checks if the awaited result indicates StopAsyncIteration
                // If so, clears the exception and jumps to the else label
                // Otherwise continues with the value
                self.builder.emit_end_async_for(anext_reg, loop_else);

                // Step 5: Store result to target
                self.compile_store(target, anext_reg)?;
                self.builder.free_register(anext_reg);

                // Step 6: Compile loop body
                for s in body {
                    self.compile_stmt(s)?;
                }

                // Loop back
                self.builder.emit_jump(loop_start);

                // Else clause
                self.builder.bind_label(loop_else);
                for s in orelse {
                    self.compile_stmt(s)?;
                }

                self.builder.bind_label(loop_end);

                // Cleanup
                self.loop_stack.pop();
                self.builder.free_register(aiter_reg);
            }

            StmtKind::ClassDef {
                name,
                bases,
                keywords,
                body,
                decorator_list,
                type_params: _type_params,
            } => {
                // Class definition compilation follows CPython's BUILD_CLASS protocol:
                // 1. Compile decorators (evaluated first, applied last)
                // 2. Evaluate base classes
                // 3. Create child CodeObject for class body
                // 4. Emit BUILD_CLASS instruction
                // 5. Apply decorators in reverse order
                // 6. Store result in enclosing scope

                // Step 1: Compile decorators and save to registers
                let decorator_regs: Vec<Register> = decorator_list
                    .iter()
                    .map(|d| self.compile_expr(d))
                    .collect::<Result<_, _>>()?;

                // Step 2: Evaluate base classes into registers
                let base_count = bases.len();
                let explicit_metaclass = keywords
                    .iter()
                    .find(|keyword| keyword.arg.as_deref() == Some("metaclass"))
                    .map(|keyword| &keyword.value);
                let class_keywords: Vec<_> = keywords
                    .iter()
                    .filter(|keyword| keyword.arg.as_deref() != Some("metaclass"))
                    .collect();
                let base_count_u8 = u8::try_from(base_count).map_err(|_| CompileError {
                    message: "class definition supports at most 255 base classes".to_string(),
                    line: stmt_line,
                    column: 0,
                })?;
                let class_block_size = u8::try_from(
                    base_count
                        + 1
                        + usize::from(explicit_metaclass.is_some())
                        + class_keywords.len(),
                )
                .map_err(|_| CompileError {
                    message: "class definition requires more registers than the VM can encode"
                        .to_string(),
                    line: stmt_line,
                    column: 0,
                })?;
                // Reserve the BUILD_CLASS register block up front and compile bases
                // directly into their final positions. This avoids keeping a second
                // layer of temporary base registers live under high register pressure.
                let result_reg = self.builder.alloc_register_block(class_block_size);
                for (index, base) in bases.iter().enumerate() {
                    let base_dst = Register::new(result_reg.0 + 1 + index as u8);
                    self.compile_expr_into(base, base_dst)?;
                }
                if let Some(metaclass_expr) = explicit_metaclass {
                    let meta_dst = Register::new(result_reg.0 + 1 + base_count as u8);
                    self.compile_expr_into(metaclass_expr, meta_dst)?;
                }
                let keyword_base =
                    result_reg.0 + 1 + base_count as u8 + u8::from(explicit_metaclass.is_some());
                for (index, keyword) in class_keywords.iter().enumerate() {
                    let kw_dst = Register::new(keyword_base + index as u8);
                    self.compile_expr_into(&keyword.value, kw_dst)?;
                }

                // Step 3: Create the class body code object using builder-swap pattern
                // Find the scope for this class from the symbol table
                let class_scope_idx = self.find_child_scope(ScopeKind::Class, name.as_ref());
                let (class_cellvars, class_freevars, class_locals) =
                    if let Some(scope_idx) = class_scope_idx {
                        let scope = &self.current_scope().children[scope_idx];
                        let cellvars = Self::ordered_cellvar_names(scope)
                            .into_iter()
                            .filter(|name| name.as_ref() != "__class__")
                            .collect::<Vec<_>>();
                        let freevars = Self::ordered_freevar_names(scope);
                        let mut locals = scope
                            .locals()
                            .map(|sym| Arc::from(sym.name.as_ref()))
                            .collect::<Vec<Arc<str>>>();
                        locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
                        (cellvars, freevars, locals)
                    } else {
                        (Vec::new(), Vec::new(), Vec::new())
                    };

                // Create a new FunctionBuilder for the class body
                let class_qualname = self.qualified_name_for_nested_definition(name.as_ref());
                let mut class_builder = FunctionBuilder::new(name.clone());
                class_builder.set_filename(self.builder.get_filename());
                class_builder.set_first_lineno(stmt_line);
                class_builder.set_qualname(Arc::clone(&class_qualname));
                class_builder.add_flag(CodeFlags::CLASS);

                // Check if any method uses zero-arg super() and inject __class__ cell
                let uses_zero_arg_super =
                    crate::class_compiler::ClassCompiler::uses_zero_arg_super(body);
                if uses_zero_arg_super {
                    // __class__ is implicitly a cell variable in class bodies that use super()
                    class_builder.add_cellvar("__class__");
                }

                // Register cell and free variables from scope analysis
                for name in class_locals {
                    class_builder.define_local(name);
                }
                for name in class_cellvars {
                    class_builder.add_cellvar(name);
                }
                for name in class_freevars {
                    class_builder.add_freevar(name);
                }

                // Swap builders to compile class body
                let parent_builder = std::mem::replace(&mut self.builder, class_builder);
                let parent_finally_stack = std::mem::take(&mut self.finally_stack);

                if let Some(scope_idx) = class_scope_idx {
                    self.enter_child_scope(scope_idx);
                }

                // CPython seeds every class body namespace with __module__ and
                // __qualname__ before user statements execute.
                self.emit_implicit_class_metadata_bindings(class_qualname.as_ref());
                if Self::body_needs_runtime_annotations(body) {
                    self.builder.emit_setup_annotations();
                }

                // Compile class body statements (method definitions, class variables, etc.)
                for (index, stmt) in body.iter().enumerate() {
                    if self.should_strip_docstring_stmt(index, stmt) {
                        continue;
                    }
                    self.compile_stmt(stmt)?;
                }

                if class_scope_idx.is_some() {
                    self.exit_child_scope();
                }

                // Class body returns the namespace dict (implicit)
                self.builder.emit_return_none();
                self.finally_stack = parent_finally_stack;

                // Swap back and get finished class body code
                let class_builder = std::mem::replace(&mut self.builder, parent_builder);
                let class_code = class_builder.finish();

                // Store the nested CodeObject as a constant
                let code_idx = self.builder.add_code_object(Arc::new(class_code));

                // Step 4: Emit BUILD_CLASS instruction
                // BUILD_CLASS consumes bases from the contiguous block
                // [result, base0, base1, ...].
                if explicit_metaclass.is_some() {
                    self.builder
                        .emit_build_class_with_meta(result_reg, code_idx, base_count_u8);
                } else {
                    self.builder
                        .emit_build_class(result_reg, code_idx, base_count_u8);
                }
                if !class_keywords.is_empty() {
                    let names = class_keywords
                        .iter()
                        .map(|keyword| {
                            keyword
                                .arg
                                .as_deref()
                                .ok_or_else(|| CompileError {
                                    message:
                                        "class definition does not support unpacked keyword arguments"
                                            .to_string(),
                                    line: stmt_line,
                                    column: 0,
                                })
                                .map(Arc::from)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let kwnames_idx = self.builder.add_kwnames_tuple(names);
                    self.builder.emit(Instruction::new(
                        Opcode::CallKwEx,
                        class_keywords.len() as u8,
                        (kwnames_idx & 0xFF) as u8,
                        (kwnames_idx >> 8) as u8,
                    ));
                }

                // Step 5: Apply decorators in reverse order
                // @decorator1
                // @decorator2
                // class Foo: ...
                // is equivalent to: Foo = decorator1(decorator2(Foo))
                for decorator_reg in decorator_regs.into_iter().rev() {
                    // Decorator calls use [result, arg0] in a dedicated block to
                    // avoid aliasing the live class register or base block.
                    let call_block = self.builder.alloc_register_block(2);
                    self.builder
                        .emit_move(Register::new(call_block.0 + 1), result_reg);
                    self.builder.emit(Instruction::op_dss(
                        Opcode::Call,
                        call_block,
                        decorator_reg,
                        Register::new(1), // 1 argument
                    ));
                    self.builder.emit_move(result_reg, call_block);
                    self.builder.free_register_block(call_block, 2);
                    self.builder.free_register(decorator_reg);
                }

                // Step 6: Store the class in the enclosing scope
                let location = self.resolve_variable(name);
                self.builder
                    .emit_store_var(location, result_reg, Some(name));
                self.builder
                    .free_register_block(result_reg, class_block_size);
            }

            StmtKind::Import(aliases) => {
                // import module1, module2 as alias, ...
                // For each alias:
                //   1. Emit ImportName to load the module
                //   2. Store in the target name (alias if present, else module name)
                for alias in aliases {
                    // Use asname if present, otherwise use the module name.
                    // For `import foo.bar`, CPython stores the top-level module `foo`
                    // while still importing the full dotted chain for side effects.
                    let local_name = alias.asname.as_ref().unwrap_or(&alias.name);

                    let store_name = if alias.asname.is_some() {
                        local_name.as_str()
                    } else {
                        alias.name.split('.').next().unwrap_or(&alias.name)
                    };

                    let module_name_idx = self.builder.add_name(alias.name.clone());
                    let store_name_idx = self.builder.add_name(store_name.to_string());

                    // Always import the full dotted name so parent packages and leaf
                    // modules are initialized and cached.
                    let imported_reg = self.builder.alloc_register();
                    self.builder.emit_import_name(imported_reg, module_name_idx);

                    // `import foo.bar` without `as` binds `foo`, not `foo.bar`.
                    // Re-importing the top-level name is effectively free after the
                    // cache-warming import above and keeps the lowering simple/correct.
                    let value_reg = if alias.asname.is_none() && alias.name.contains('.') {
                        let top_level_idx = self.builder.add_name(store_name.to_string());
                        let top_level_reg = self.builder.alloc_register();
                        self.builder.emit_import_name(top_level_reg, top_level_idx);
                        top_level_reg
                    } else {
                        imported_reg
                    };

                    // Store the module in the appropriate scope
                    match self.resolve_variable(store_name) {
                        VarLocation::Local(slot) => {
                            self.builder
                                .emit_store_local(LocalSlot::new(slot as u16), value_reg);
                        }
                        VarLocation::Closure(slot) => {
                            self.builder.emit_store_closure(slot, value_reg);
                        }
                        VarLocation::Global => {
                            self.builder.emit_store_global(store_name_idx, value_reg);
                        }
                    }

                    if value_reg != imported_reg {
                        self.builder.free_register(value_reg);
                    }
                    self.builder.free_register(imported_reg);
                }
            }

            StmtKind::ImportFrom {
                module,
                names,
                level,
            } => {
                // from module import name1, name2 as alias, ...
                // 1. Import the source module first
                // 2. For each name, import the attribute
                let module_spec = format!(
                    "{}{}",
                    ".".repeat(*level as usize),
                    module.as_deref().unwrap_or("")
                );

                // Handle `from module import *` case
                let is_star = names.len() == 1 && names[0].name == "*";

                if is_star {
                    // from module import *
                    let mod_name_idx = self.builder.add_name(module_spec.clone());
                    let mod_reg = self.builder.alloc_register();

                    self.builder.emit_import_name(mod_reg, mod_name_idx);
                    self.builder.emit_import_star(Register::new(0), mod_reg);

                    self.builder.free_register(mod_reg);
                } else {
                    // from module import name1, name2, ...
                    let mod_name_idx = self.builder.add_name(module_spec);
                    let mod_reg = self.builder.alloc_register();
                    self.builder.emit_import_name(mod_reg, mod_name_idx);

                    for alias in names {
                        let local_name = alias.asname.as_ref().unwrap_or(&alias.name);
                        let attr_name_idx = self.builder.add_name(alias.name.clone());
                        let store_name_idx = self.builder.add_name(local_name.clone());

                        // Emit ImportFrom: reg = module.attr
                        let attr_reg = self.builder.alloc_register();
                        self.builder
                            .emit_import_from(attr_reg, mod_reg, attr_name_idx);

                        // Store in the appropriate scope
                        match self.resolve_variable(local_name) {
                            VarLocation::Local(slot) => {
                                self.builder
                                    .emit_store_local(LocalSlot::new(slot as u16), attr_reg);
                            }
                            VarLocation::Closure(slot) => {
                                self.builder.emit_store_closure(slot, attr_reg);
                            }
                            VarLocation::Global => {
                                self.builder.emit_store_global(store_name_idx, attr_reg);
                            }
                        }

                        self.builder.free_register(attr_reg);
                    }

                    self.builder.free_register(mod_reg);
                }
            }

            StmtKind::Global(_) | StmtKind::Nonlocal(_) => {
                // These are handled during scope analysis
            }

            StmtKind::Delete(targets) => {
                for target in targets {
                    self.compile_delete_target(target)?;
                }
            }

            StmtKind::Assert { test, msg } => {
                // `-O`/`-OO`: strip assert statements entirely.
                if self.optimize >= OptimizationLevel::Basic {
                    return Ok(());
                }

                let pass_label = self.builder.create_label();

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_true(cond_reg, pass_label);
                self.builder.free_register(cond_reg);

                // Synthesize: raise AssertionError([message])
                // We lower through the regular call path so the raised value is a real
                // exception object (compatible with except AssertionError as e).
                let assertion_name_idx = self.builder.add_name(Arc::from("AssertionError"));
                let mut ctor_reg = self.builder.alloc_register();
                self.builder.emit_load_global(ctor_reg, assertion_name_idx);

                let arg_count = if msg.is_some() { 1u8 } else { 0u8 };

                let (call_dst, call_block, block_size) = if arg_count == 0 {
                    (ctor_reg, None, 0)
                } else {
                    // Reserve a fresh contiguous block [dst, arg0] for Call layout.
                    let block = self.builder.alloc_register_block(1 + arg_count);
                    (block, Some(block), 1 + arg_count)
                };

                // If the constructor register falls within the call block, move it
                // to a safe register to avoid clobbering.
                if let Some(block) = call_block {
                    let block_end = block.0 + block_size;
                    if ctor_reg.0 >= block.0 && ctor_reg.0 < block_end {
                        let safe_reg = self.builder.alloc_register();
                        self.builder.emit_move(safe_reg, ctor_reg);
                        self.builder.free_register(ctor_reg);
                        ctor_reg = safe_reg;
                    }
                }

                if let Some(msg_expr) = msg {
                    let arg_dst = Register::new(call_dst.0 + 1);
                    let temp = self.compile_expr(msg_expr)?;
                    if temp != arg_dst {
                        self.builder.emit_move(arg_dst, temp);
                    }
                    self.builder.free_register(temp);
                }

                self.builder.emit_call(call_dst, ctor_reg, arg_count);

                if let Some(block) = call_block {
                    if call_dst != ctor_reg {
                        self.builder.emit_move(ctor_reg, call_dst);
                    }
                    self.builder.free_register_block(block, block_size);
                }

                self.builder
                    .emit(Instruction::op_d(Opcode::Raise, ctor_reg));
                self.builder.free_register(ctor_reg);

                self.builder.bind_label(pass_label);
            }

            StmtKind::Raise { exc, cause } => {
                match (exc, cause) {
                    // raise X from Y - exception chaining with explicit cause
                    (Some(e), Some(c)) => {
                        let exc_reg = self.compile_expr(e)?;
                        let cause_reg = self.compile_expr(c)?;
                        self.builder.emit(Instruction::op_ds(
                            Opcode::RaiseFrom,
                            exc_reg,
                            cause_reg,
                        ));
                        self.builder.free_register(cause_reg);
                        self.builder.free_register(exc_reg);
                    }
                    // raise X - simple exception raise
                    (Some(e), None) => {
                        let reg = self.compile_expr(e)?;
                        self.builder.emit(Instruction::op_d(Opcode::Raise, reg));
                        self.builder.free_register(reg);
                    }
                    // bare raise - reraise current exception
                    (None, _) => {
                        self.builder.emit(Instruction::op(Opcode::Reraise));
                    }
                }
            }

            StmtKind::FunctionDef {
                name,
                args,
                body,
                decorator_list,
                ..
            } => {
                self.compile_function_def(name, args, body, decorator_list, false, stmt_line)?;
            }

            StmtKind::AsyncFunctionDef {
                name,
                args,
                body,
                decorator_list,
                ..
            } => {
                self.compile_function_def(name, args, body, decorator_list, true, stmt_line)?;
            }

            StmtKind::Try {
                body,
                handlers,
                orelse,
                finalbody,
            } => {
                self.compile_try(body, handlers, orelse, finalbody)?;
            }

            StmtKind::TryStar { .. } => {
                return Err(self.unsupported_stmt_error(
                    stmt,
                    "try/except* requires ExceptionGroup splitting semantics",
                ));
            }

            StmtKind::With { items, body } => {
                self.compile_with(items, body)?;
            }

            StmtKind::AsyncWith { items, body } => {
                // Validate async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'async with' outside async function".to_string(),
                        line: stmt_line,
                        column: 0,
                    });
                }

                // Async with uses the same structure as sync with but awaits the
                // __aenter__ and __aexit__ calls. For now, we compile it using the
                // sync version's structure foundation but add await points.
                self.compile_async_with(items, body)?;
            }

            StmtKind::Match { subject, cases } => {
                self.compile_match(subject, cases)?;
            }

            StmtKind::TypeAlias { .. } => {
                return Err(self.unsupported_stmt_error(
                    stmt,
                    "type alias statements require Python 3.12 TypeAliasType semantics",
                ));
            }
        }

        Ok(())
    }

    fn compile_annotated_assignment(
        &mut self,
        stmt: &Stmt,
        target: &Expr,
        annotation: &Expr,
        value: Option<&Expr>,
        simple: bool,
    ) -> CompileResult<()> {
        if let Some(value) = value {
            let value_reg = self.compile_expr(value)?;
            self.compile_store(target, value_reg)?;
            self.builder.free_register(value_reg);
        }

        if simple && self.current_scope_records_runtime_annotations() {
            let ExprKind::Name(name) = &target.kind else {
                return Err(self.unsupported_stmt_error(
                    stmt,
                    "simple annotated assignments must target a name",
                ));
            };

            let annotations_location = self.resolve_variable("__annotations__");
            let annotations_reg = self.builder.alloc_register();
            let key_reg = self.builder.alloc_register();
            let value_reg = self.compile_expr(annotation)?;

            self.builder.emit_load_var(
                annotations_reg,
                annotations_location,
                Some("__annotations__"),
            );
            let key_idx = self.builder.add_string(name.as_str());
            self.builder.emit_load_const(key_reg, key_idx);
            self.builder
                .emit_set_item(annotations_reg, key_reg, value_reg);

            self.builder.free_register(annotations_reg);
            self.builder.free_register(key_reg);
            self.builder.free_register(value_reg);
        } else if value.is_none() && !simple {
            return Err(self.unsupported_stmt_error(
                stmt,
                "annotated attribute and subscript declarations without a value need target evaluation semantics",
            ));
        }

        Ok(())
    }

    #[inline]
    fn current_scope_records_runtime_annotations(&self) -> bool {
        matches!(
            self.current_scope().kind,
            ScopeKind::Module | ScopeKind::Class
        )
    }

    #[inline]
    fn unsupported_stmt_error(&self, stmt: &Stmt, reason: &'static str) -> CompileError {
        CompileError {
            message: format!(
                "unsupported statement {}: {reason}",
                stmt_kind_name(&stmt.kind)
            ),
            line: self.line_for_span(stmt.span),
            column: 0,
        }
    }

    #[inline]
    fn unsupported_expr_error(&self, expr: &Expr, reason: &'static str) -> CompileError {
        CompileError {
            message: format!(
                "unsupported expression {}: {reason}",
                expr_kind_name(&expr.kind)
            ),
            line: self.line_for_span(expr.span),
            column: 0,
        }
    }

    #[inline]
    fn unsupported_pattern_error(
        &self,
        pattern: &prism_parser::ast::Pattern,
        reason: &'static str,
    ) -> CompileError {
        CompileError {
            message: format!(
                "unsupported pattern {}: {reason}",
                pattern_kind_name(&pattern.kind)
            ),
            line: self.line_for_span(pattern.span),
            column: 0,
        }
    }

    /// Compile an expression and return the register containing the result.
    fn compile_expr(&mut self, expr: &Expr) -> CompileResult<Register> {
        let reg = self.builder.alloc_register();
        self.compile_expr_into(expr, reg)?;
        Ok(reg)
    }

    fn compile_unpacking_sequence_literal(
        &mut self,
        elts: &[Expr],
        reg: Register,
        line: u32,
        kind: SequenceLiteralKind,
    ) -> CompileResult<()> {
        if elts.len() > 24 {
            return Err(CompileError {
                message: "starred sequence literal supports at most 24 source entries".to_string(),
                line,
                column: 0,
            });
        }

        let count = elts.len() as u8;
        let base_reg = self.builder.alloc_register_block(count);
        let mut unpack_flags = 0_u32;

        for (i, elt) in elts.iter().enumerate() {
            let item_reg = Register::new(base_reg.0 + i as u8);
            let temp = match &elt.kind {
                ExprKind::Starred(inner) => {
                    unpack_flags |= 1 << i;
                    self.compile_expr(inner)?
                }
                _ => self.compile_expr(elt)?,
            };

            if temp != item_reg {
                self.builder.emit_move(item_reg, temp);
                self.builder.free_register(temp);
            }
        }

        match kind {
            SequenceLiteralKind::List => {
                self.builder
                    .emit_build_list_unpack(reg, base_reg, count, unpack_flags);
            }
            SequenceLiteralKind::Tuple => {
                self.builder
                    .emit_build_tuple_unpack(reg, base_reg, count, unpack_flags);
            }
            SequenceLiteralKind::Set => {
                self.builder
                    .emit_build_set_unpack(reg, base_reg, count, unpack_flags);
            }
        }

        self.builder.free_register_block(base_reg, count);
        Ok(())
    }

    fn parse_bigint_literal(literal: &str) -> Option<BigInt> {
        let (radix, digits) = if let Some(rest) = literal
            .strip_prefix("0x")
            .or_else(|| literal.strip_prefix("0X"))
        {
            (16, rest)
        } else if let Some(rest) = literal
            .strip_prefix("0o")
            .or_else(|| literal.strip_prefix("0O"))
        {
            (8, rest)
        } else if let Some(rest) = literal
            .strip_prefix("0b")
            .or_else(|| literal.strip_prefix("0B"))
        {
            (2, rest)
        } else {
            (10, literal)
        };

        let normalized: String = digits.chars().filter(|c| *c != '_').collect();
        if normalized.is_empty() {
            return None;
        }

        BigInt::parse_bytes(normalized.as_bytes(), radix)
    }

    fn compile_large_dict_literal_into(
        &mut self,
        dst: Register,
        keys: &[Option<Expr>],
        values: &[Expr],
    ) -> CompileResult<()> {
        self.builder
            .emit(Instruction::new(Opcode::BuildDict, dst.0, dst.0, 0));

        for (key_expr, value_expr) in keys.iter().zip(values.iter()) {
            let key_expr = key_expr
                .as_ref()
                .expect("large dict literal fallback only supports concrete keys");

            let key_reg = self.compile_expr(key_expr)?;
            let value_reg = self.compile_expr(value_expr)?;

            self.builder.emit(Instruction::new(
                Opcode::DictSet,
                key_reg.0,
                dst.0,
                value_reg.0,
            ));

            self.builder.free_register(key_reg);
            self.builder.free_register(value_reg);
        }

        Ok(())
    }

    /// Compile an expression into a specific destination register.
    fn compile_expr_into(&mut self, expr: &Expr, reg: Register) -> CompileResult<()> {
        let expr_line = self.line_for_span(expr.span);
        match &expr.kind {
            ExprKind::Int(n) => {
                let idx = self.builder.add_int(*n);
                self.builder.emit_load_const(reg, idx);
            }

            ExprKind::BigInt(literal) => {
                let value = Self::parse_bigint_literal(literal).ok_or_else(|| CompileError {
                    message: format!("invalid integer literal: {literal}"),
                    line: expr_line,
                    column: 0,
                })?;
                let idx = self.builder.add_bigint(value);
                self.builder.emit_load_const(reg, idx);
            }

            ExprKind::Float(n) => {
                let idx = self.builder.add_float(*n);
                self.builder.emit_load_const(reg, idx);
            }

            ExprKind::Complex { real, imag } => {
                self.compile_complex_literal(*real, *imag, reg)?;
                return Ok(());
            }

            ExprKind::Bool(b) => {
                if *b {
                    self.builder.emit_load_true(reg);
                } else {
                    self.builder.emit_load_false(reg);
                }
            }

            ExprKind::None => {
                self.builder.emit_load_none(reg);
            }

            ExprKind::Ellipsis => {
                self.emit_load_builtin_name(reg, "Ellipsis");
            }

            ExprKind::String(s) => {
                // Add string to constant pool with automatic interning and deduplication
                let str_idx = self.builder.add_string(s.value.as_str());
                self.builder.emit_load_const(reg, str_idx);
            }

            ExprKind::Bytes(bytes) => {
                self.compile_bytes_literal(bytes, reg)?;
            }

            ExprKind::JoinedStr(parts) => {
                self.compile_joined_str(parts, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::FormattedValue {
                value,
                conversion,
                format_spec,
            } => {
                self.compile_formatted_value(value, *conversion, format_spec.as_deref(), reg)?;
                return Ok(());
            }

            ExprKind::Name(name) => {
                // Use scope-aware variable resolution
                let location = self.resolve_variable(name);
                self.builder
                    .emit_load_var(reg, location, Some(name.as_ref()));
            }

            ExprKind::BinOp { op, left, right } => {
                let left_reg = self.compile_expr(left)?;
                let right_reg = self.compile_expr(right)?;

                self.emit_binop(*op, reg, left_reg, right_reg);

                self.builder.free_register(left_reg);
                self.builder.free_register(right_reg);
            }

            ExprKind::UnaryOp { op, operand } => {
                let operand_reg = self.compile_expr(operand)?;

                match op {
                    UnaryOp::USub => self.builder.emit_neg(reg, operand_reg),
                    UnaryOp::UAdd => self.builder.emit_pos(reg, operand_reg),
                    UnaryOp::Not => self.builder.emit_not(reg, operand_reg),
                    UnaryOp::Invert => self.builder.emit_bitwise_not(reg, operand_reg),
                }

                self.builder.free_register(operand_reg);
            }

            ExprKind::Compare {
                left,
                ops,
                comparators,
            } => {
                // Handle chained comparisons: a < b < c becomes (a < b) and (b < c)
                let left_reg = self.compile_expr(left)?;
                let result_reg = reg;

                if ops.len() == 1 {
                    // Simple case: single comparison
                    let right_reg = self.compile_expr(&comparators[0])?;
                    self.emit_cmpop(ops[0], result_reg, left_reg, right_reg);
                    self.builder.free_register(right_reg);
                } else {
                    // Chained comparisons
                    self.builder.emit_load_true(result_reg);
                    let mut prev_reg = left_reg;

                    for (op, comp) in ops.iter().zip(comparators.iter()) {
                        let next_reg = self.compile_expr(comp)?;
                        let cmp_reg = self.builder.alloc_register();

                        self.emit_cmpop(*op, cmp_reg, prev_reg, next_reg);

                        // result = result and cmp
                        self.builder
                            .emit_bitwise_and(result_reg, result_reg, cmp_reg);

                        self.builder.free_register(cmp_reg);
                        if prev_reg != left_reg {
                            self.builder.free_register(prev_reg);
                        }
                        prev_reg = next_reg;
                    }

                    self.builder.free_register(prev_reg);
                }

                self.builder.free_register(left_reg);
            }

            ExprKind::BoolOp { op, values } => {
                // Short-circuit evaluation
                let end_label = self.builder.create_label();

                let first_reg = self.compile_expr(&values[0])?;
                self.builder.emit_move(reg, first_reg);
                self.builder.free_register(first_reg);

                for value in &values[1..] {
                    match op {
                        BoolOp::And => {
                            self.builder.emit_jump_if_false(reg, end_label);
                        }
                        BoolOp::Or => {
                            self.builder.emit_jump_if_true(reg, end_label);
                        }
                    }

                    let next_reg = self.compile_expr(value)?;
                    self.builder.emit_move(reg, next_reg);
                    self.builder.free_register(next_reg);
                }

                self.builder.bind_label(end_label);
            }

            ExprKind::Call {
                func,
                args,
                keywords,
            } => {
                // Check for call-site unpacking:
                // - *args: positional arg is ExprKind::Starred
                // - **kwargs: keyword with arg = None
                let has_star_unpack = args
                    .iter()
                    .any(|arg| matches!(&arg.kind, ExprKind::Starred(_)));
                let has_dstar_unpack = keywords.iter().any(|kw| kw.arg.is_none());

                if has_star_unpack || has_dstar_unpack {
                    // Dynamic call path: build tuple/dict and call with unpacking
                    self.compile_dynamic_call(func, args, keywords, reg, expr_line)?;
                    return Ok(());
                }

                // =====================================================================
                // OPTIMIZATION: Method call pattern detection
                // =====================================================================
                // If func is an Attribute expression (obj.method), use optimized
                // LoadMethod/CallMethod sequence instead of GetAttr + Call.
                // This avoids creating a bound method object on every call.
                if let ExprKind::Attribute { value, attr, .. } = &func.kind {
                    // Only use method call optimization for simple cases:
                    // - No keyword arguments (CallMethodKw not yet implemented)
                    // - No *args/**kwargs unpacking (already handled above)
                    if keywords.is_empty() {
                        self.compile_method_call(value, attr, args, reg)?;
                        return Ok(());
                    }
                    // TODO: Implement compile_method_call_kw for keyword arguments
                }

                // Strategy to avoid register collisions:
                // CRITICAL: The Call instruction uses consecutive registers [dst, dst+1, dst+2, ...].
                // The VM reads args from registers [dst+1, dst+2, ...]. If `reg` (allocated by compile_expr)
                // is from the free list at a low position, `reg+1` could clobber a live register
                // like list_reg in a list comprehension.
                //
                // SOLUTION: When there are arguments (argc > 0), allocate a fresh contiguous block
                // from next_register to avoid any collision. For zero-arg calls, use reg directly
                // since there are no consecutive arg writes to cause clobbering.

                let posargc = args.len();
                let kwargc = keywords.len();
                let total_argc = posargc + kwargc;

                // Compile function expression first (before allocating call block)
                let mut func_reg = self.compile_expr(func)?;

                // For calls WITH arguments, use fresh contiguous block to prevent clobbering.
                // For zero-arg calls, use `reg` directly (no arg writes to worry about).
                let (call_dst, call_block, block_size) = if total_argc > 0 {
                    // Allocate fresh contiguous block for [call_dst, arg0, arg1, ...]
                    let size = (1 + total_argc) as u8;
                    let block = self.builder.alloc_register_block(size);
                    (block, Some(block), size)
                } else {
                    // No arguments - use reg directly, no block needed
                    (reg, None, 0)
                };

                // Check if func_reg is inside our call block range (would be clobbered)
                if let Some(block) = call_block {
                    let block_end = block.0 + block_size;
                    if func_reg.0 >= block.0 && func_reg.0 < block_end {
                        // Move func to a safe register outside the block
                        let safe_reg = self.builder.alloc_register();
                        self.builder.emit_move(safe_reg, func_reg);
                        self.builder.free_register(func_reg);
                        func_reg = safe_reg;
                    }
                }

                // Compile positional arguments to call_dst+1..call_dst+posargc
                for (i, arg) in args.iter().enumerate() {
                    let arg_dst = Register::new(call_dst.0 + 1 + i as u8);
                    self.compile_expr_into(arg, arg_dst)?;
                }

                // Handle keyword arguments
                if keywords.is_empty() {
                    // No keywords - use simple Call instruction
                    self.builder.emit_call(call_dst, func_reg, posargc as u8);
                } else {
                    // Compile keyword argument values to consecutive registers
                    // after positional arguments
                    for (i, kw) in keywords.iter().enumerate() {
                        let kw_dst = Register::new(call_dst.0 + 1 + posargc as u8 + i as u8);
                        self.compile_expr_into(&kw.value, kw_dst)?;
                    }

                    // Build keyword names tuple for the constant pool
                    let kw_names: Vec<std::sync::Arc<str>> = keywords
                        .iter()
                        .map(|kw| {
                            // We already checked that arg is Some (no **kwargs unpacking)
                            std::sync::Arc::from(kw.arg.as_ref().unwrap().as_str())
                        })
                        .collect();
                    let kwnames_idx = self.builder.add_kwnames_tuple(kw_names);

                    // Emit CallKw instruction pair
                    self.builder.emit_call_kw(
                        call_dst,
                        func_reg,
                        posargc as u8,
                        kwargc as u8,
                        kwnames_idx,
                    );
                }

                self.builder.free_register(func_reg);

                // If we used a block, move result to expected destination and free block
                if let Some(block) = call_block {
                    if call_dst != reg {
                        self.builder.emit_move(reg, call_dst);
                    }
                    self.builder.free_register_block(block, block_size);
                }
            }

            ExprKind::Attribute { value, attr, .. } => {
                let obj_reg = self.compile_expr(value)?;
                let name_idx = self.builder.add_name(attr.clone());
                self.builder.emit_get_attr(reg, obj_reg, name_idx);
                self.builder.free_register(obj_reg);
            }

            ExprKind::Subscript { value, slice, .. } => {
                let obj_reg = self.compile_expr(value)?;
                let key_reg = self.compile_expr(slice)?;
                self.builder.emit_get_item(reg, obj_reg, key_reg);
                self.builder.free_register(obj_reg);
                self.builder.free_register(key_reg);
            }

            ExprKind::Slice { lower, upper, step } => {
                // Build a slice object for subscription keys.
                let start_reg = if let Some(lower_expr) = lower {
                    self.compile_expr(lower_expr)?
                } else {
                    let none_reg = self.builder.alloc_register();
                    self.builder.emit_load_none(none_reg);
                    none_reg
                };

                let stop_reg = if let Some(upper_expr) = upper {
                    self.compile_expr(upper_expr)?
                } else {
                    let none_reg = self.builder.alloc_register();
                    self.builder.emit_load_none(none_reg);
                    none_reg
                };

                let step_reg = if let Some(step_expr) = step {
                    Some(self.compile_expr(step_expr)?)
                } else {
                    None
                };

                self.builder.emit(Instruction::op_dss(
                    Opcode::BuildSlice,
                    reg,
                    start_reg,
                    stop_reg,
                ));

                // Encode optional step in an extension instruction consumed by BuildSlice.
                if let Some(step_reg) = step_reg {
                    self.builder.emit(Instruction::new(
                        Opcode::CallKwEx,
                        step_reg.0,
                        SLICE_STEP_EXT_TAG_A,
                        SLICE_STEP_EXT_TAG_B,
                    ));
                    self.builder.free_register(step_reg);
                }

                self.builder.free_register(start_reg);
                self.builder.free_register(stop_reg);
            }

            ExprKind::List(elts) => {
                if elts
                    .iter()
                    .any(|elt| matches!(&elt.kind, ExprKind::Starred(_)))
                {
                    self.compile_unpacking_sequence_literal(
                        elts,
                        reg,
                        expr_line,
                        SequenceLiteralKind::List,
                    )?;
                    return Ok(());
                }

                // BuildList expects a consecutive register block.
                if elts.is_empty() {
                    self.builder.emit_build_list(reg, reg, 0);
                } else {
                    if elts.len() > u8::MAX as usize {
                        return Err(CompileError {
                            message: "list literal has too many elements".to_string(),
                            line: expr_line,
                            column: 0,
                        });
                    }

                    let count = elts.len() as u8;
                    let first_elem = self.builder.alloc_register_block(count);

                    for (i, elt) in elts.iter().enumerate() {
                        let elem_reg = Register::new(first_elem.0 + i as u8);
                        let temp = self.compile_expr(elt)?;
                        if temp != elem_reg {
                            self.builder.emit_move(elem_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }

                    self.builder.emit_build_list(reg, first_elem, count);
                    self.builder.free_register_block(first_elem, count);
                }
            }

            ExprKind::Set(elts) => {
                if elts
                    .iter()
                    .any(|elt| matches!(&elt.kind, ExprKind::Starred(_)))
                {
                    self.compile_unpacking_sequence_literal(
                        elts,
                        reg,
                        expr_line,
                        SequenceLiteralKind::Set,
                    )?;
                    return Ok(());
                }

                // BuildSet also reads a consecutive register range [start, start+count).
                if elts.is_empty() {
                    self.builder
                        .emit(Instruction::new(Opcode::BuildSet, reg.0, reg.0, 0));
                } else {
                    if elts.len() > u8::MAX as usize {
                        return Err(CompileError {
                            message: "set literal has too many elements".to_string(),
                            line: expr_line,
                            column: 0,
                        });
                    }
                    let count = elts.len() as u8;
                    let first_elem = self.builder.alloc_register_block(count);

                    for (i, elt) in elts.iter().enumerate() {
                        let elem_reg = Register::new(first_elem.0 + i as u8);
                        let temp = self.compile_expr(elt)?;
                        if temp != elem_reg {
                            self.builder.emit_move(elem_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }

                    self.builder.emit(Instruction::new(
                        Opcode::BuildSet,
                        reg.0,
                        first_elem.0,
                        count,
                    ));
                    self.builder.free_register_block(first_elem, count);
                }
            }

            ExprKind::Dict { keys, values } => {
                if keys.len() != values.len() {
                    return Err(CompileError {
                        message: "dict literal has mismatched keys/values".to_string(),
                        line: expr_line,
                        column: 0,
                    });
                }

                let entry_count = values.len();
                if entry_count == 0 {
                    self.builder
                        .emit(Instruction::new(Opcode::BuildDict, reg.0, reg.0, 0));
                } else {
                    let has_unpack = keys.iter().any(|k| k.is_none());

                    if !has_unpack {
                        // Fast path: direct BuildDict from contiguous [k0, v0, k1, v1, ...].
                        if entry_count > (u8::MAX as usize / 2) {
                            self.compile_large_dict_literal_into(reg, keys, values)?;
                            return Ok(());
                        }

                        let pair_regs = (entry_count * 2) as u8;
                        let first_pair = self.builder.alloc_register_block(pair_regs);

                        for i in 0..entry_count {
                            let key_reg = Register::new(first_pair.0 + (i * 2) as u8);
                            let val_reg = Register::new(first_pair.0 + (i * 2 + 1) as u8);

                            let key_expr = keys[i].as_ref().expect("checked no unpack above");
                            let key_tmp = self.compile_expr(key_expr)?;
                            if key_tmp != key_reg {
                                self.builder.emit_move(key_reg, key_tmp);
                            }
                            self.builder.free_register(key_tmp);

                            let val_tmp = self.compile_expr(&values[i])?;
                            if val_tmp != val_reg {
                                self.builder.emit_move(val_reg, val_tmp);
                            }
                            self.builder.free_register(val_tmp);
                        }

                        self.builder.emit(Instruction::new(
                            Opcode::BuildDict,
                            reg.0,
                            first_pair.0,
                            entry_count as u8,
                        ));
                        self.builder.free_register_block(first_pair, pair_regs);
                    } else {
                        // General path: materialize each entry as a mapping and merge.
                        // Static k:v entries become singleton dicts; **m entries merge directly.
                        if entry_count > 24 {
                            return Err(CompileError {
                                message: "dict unpack supports at most 24 entries".to_string(),
                                line: expr_line,
                                column: 0,
                            });
                        }

                        let base = self.builder.alloc_register_block(entry_count as u8);
                        let mut unpack_flags: u32 = 0;

                        for i in 0..entry_count {
                            let entry_reg = Register::new(base.0 + i as u8);

                            if let Some(key_expr) = &keys[i] {
                                let pair_base = self.builder.alloc_register_block(2);
                                let key_reg = pair_base;
                                let val_reg = Register::new(pair_base.0 + 1);

                                let key_tmp = self.compile_expr(key_expr)?;
                                if key_tmp != key_reg {
                                    self.builder.emit_move(key_reg, key_tmp);
                                }
                                self.builder.free_register(key_tmp);

                                let val_tmp = self.compile_expr(&values[i])?;
                                if val_tmp != val_reg {
                                    self.builder.emit_move(val_reg, val_tmp);
                                }
                                self.builder.free_register(val_tmp);

                                self.builder.emit(Instruction::new(
                                    Opcode::BuildDict,
                                    entry_reg.0,
                                    pair_base.0,
                                    1,
                                ));
                                self.builder.free_register_block(pair_base, 2);
                            } else {
                                let mapping_tmp = self.compile_expr(&values[i])?;
                                if mapping_tmp != entry_reg {
                                    self.builder.emit_move(entry_reg, mapping_tmp);
                                }
                                self.builder.free_register(mapping_tmp);
                            }

                            unpack_flags |= 1 << i;
                        }

                        self.builder.emit_build_dict_unpack(
                            reg,
                            base,
                            entry_count as u8,
                            unpack_flags,
                        );
                        self.builder.free_register_block(base, entry_count as u8);
                    }
                }
            }

            ExprKind::Tuple(elts) => {
                if elts
                    .iter()
                    .any(|elt| matches!(&elt.kind, ExprKind::Starred(_)))
                {
                    self.compile_unpacking_sequence_literal(
                        elts,
                        reg,
                        expr_line,
                        SequenceLiteralKind::Tuple,
                    )?;
                    return Ok(());
                }

                // CRITICAL: BuildTuple expects consecutive registers [first, first+1, ...].
                // Using alloc_register() individually can allocate non-contiguous registers
                // from the free list, breaking this invariant.
                // Use alloc_register_block to guarantee contiguity.
                if elts.is_empty() {
                    // Empty tuple - just build with no elements
                    self.builder.emit_build_tuple(reg, reg, 0);
                } else {
                    let count = elts.len() as u8;
                    let first_elem = self.builder.alloc_register_block(count);

                    for (i, elt) in elts.iter().enumerate() {
                        let elem_reg = Register::new(first_elem.0 + i as u8);
                        let temp = self.compile_expr(elt)?;
                        if temp != elem_reg {
                            self.builder.emit_move(elem_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }

                    self.builder.emit_build_tuple(reg, first_elem, count);

                    // Free the element register block
                    self.builder.free_register_block(first_elem, count);
                }
            }

            ExprKind::IfExp { test, body, orelse } => {
                let else_label = self.builder.create_label();
                let end_label = self.builder.create_label();

                let cond_reg = self.compile_expr(test)?;
                self.builder.emit_jump_if_false(cond_reg, else_label);
                self.builder.free_register(cond_reg);

                let body_reg = self.compile_expr(body)?;
                self.builder.emit_move(reg, body_reg);
                self.builder.free_register(body_reg);
                self.builder.emit_jump(end_label);

                self.builder.bind_label(else_label);
                let else_reg = self.compile_expr(orelse)?;
                self.builder.emit_move(reg, else_reg);
                self.builder.free_register(else_reg);

                self.builder.bind_label(end_label);
            }

            // =========================================================================
            // Async/Await and Generator Expressions (PEP 492/255)
            // =========================================================================
            ExprKind::Await(value) => {
                // Validate we're in async context
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "'await' outside async function".to_string(),
                        line: expr_line,
                        column: 0,
                    });
                }

                // Step 1: Compile the awaitable expression
                let awaitable_reg = self.compile_expr(value)?;

                // Step 2: GetAwaitable - Convert to awaitable (validates __await__)
                // This handles:
                //   - Coroutines → pass through
                //   - Async generators → pass through
                //   - Generators with CO_ITERABLE_COROUTINE → pass through
                //   - Objects with __await__ → call and verify
                self.builder
                    .emit(Instruction::op_ds(Opcode::GetAwaitable, reg, awaitable_reg));
                self.builder.free_register(awaitable_reg);

                // Step 3: YieldFrom - Delegate to the awaitable until completion
                // This suspends the coroutine and returns control to the event loop
                // The result ends up in reg when the awaitable completes
                self.emit_yield_from(reg, reg);
            }

            ExprKind::Yield(value) => {
                // Yield must be inside a function (generator context)
                if !self.in_function_context {
                    return Err(CompileError {
                        message: "'yield' outside function".into(),
                        line: expr_line,
                        column: 0,
                    });
                }

                if let Some(val) = value {
                    // yield <expr>
                    let val_reg = self.compile_expr(val)?;
                    self.builder
                        .emit(Instruction::op_ds(Opcode::Yield, reg, val_reg));
                    self.builder.free_register(val_reg);
                } else {
                    // yield (no value - yields None)
                    let none_reg = self.builder.alloc_register();
                    self.builder.emit_load_none(none_reg);
                    self.builder
                        .emit(Instruction::op_ds(Opcode::Yield, reg, none_reg));
                    self.builder.free_register(none_reg);
                }
            }

            ExprKind::YieldFrom(value) => {
                // YieldFrom must be inside a function (generator context)
                if !self.in_function_context {
                    return Err(CompileError {
                        message: "'yield from' outside function".into(),
                        line: expr_line,
                        column: 0,
                    });
                }

                // Compile the iterable
                let iter_reg = self.compile_expr(value)?;

                // YieldFrom: dst = result, src = iterator
                self.emit_yield_from(reg, iter_reg);
                self.builder.free_register(iter_reg);
            }

            ExprKind::Lambda { args, body } => {
                // Lambda expressions create nested code objects just like functions.
                // The key difference is:
                // 1. Body is a single expression (not statements)
                // 2. Result is implicitly returned
                // 3. Lambda inherits async context from enclosing scope
                self.compile_lambda(args, body, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::ListComp { elt, generators } => {
                // List comprehensions create nested code objects for proper scoping.
                // This prevents loop variables from leaking into enclosing scope.
                self.compile_listcomp(elt, generators, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::SetComp { elt, generators } => {
                // Set comprehensions follow same pattern as list comprehensions
                self.compile_setcomp(elt, generators, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::DictComp {
                key,
                value,
                generators,
            } => {
                // Dict comprehensions create nested code for proper scoping
                self.compile_dictcomp(key, value, generators, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::GeneratorExp { elt, generators } => {
                // Generator expressions are lazy - create generator function
                self.compile_genexp(elt, generators, reg, expr_line)?;
                return Ok(());
            }

            ExprKind::NamedExpr { target, value } => {
                self.compile_expr_into(value, reg)?;
                match &target.kind {
                    ExprKind::Name(_) => self.compile_store(target, reg)?,
                    _ => {
                        return Err(CompileError {
                            message: "assignment expression target must be an identifier"
                                .to_string(),
                            line: self.line_for_span(target.span),
                            column: 0,
                        });
                    }
                }
            }

            ExprKind::Starred(_) => {
                return Err(self.unsupported_expr_error(
                    expr,
                    "starred expressions are only valid in calls, literals, and assignment targets",
                ));
            }
        }

        Ok(())
    }

    fn emit_load_builtin_name(&mut self, dst: Register, name: &str) {
        let name_idx = self.builder.add_name(name);
        self.builder.emit_load_builtin(dst, name_idx);
    }

    fn compile_complex_literal(
        &mut self,
        real: f64,
        imag: f64,
        dst: Register,
    ) -> CompileResult<()> {
        let mut ctor_reg = self.builder.alloc_register();
        self.emit_load_builtin_name(ctor_reg, "complex");

        let call_block = self.builder.alloc_register_block(3);
        let call_block_end = call_block.0 + 3;
        if ctor_reg.0 >= call_block.0 && ctor_reg.0 < call_block_end {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, ctor_reg);
            self.builder.free_register(ctor_reg);
            ctor_reg = safe_reg;
        }

        let real_reg = Register::new(call_block.0 + 1);
        let imag_reg = Register::new(call_block.0 + 2);

        let real_idx = self.builder.add_float(real);
        self.builder.emit_load_const(real_reg, real_idx);

        let imag_idx = self.builder.add_float(imag);
        self.builder.emit_load_const(imag_reg, imag_idx);

        self.builder.emit_call(call_block, ctor_reg, 2);
        self.builder.emit_move(dst, call_block);

        self.builder.free_register(ctor_reg);
        self.builder.free_register_block(call_block, 3);
        Ok(())
    }

    fn compile_bytes_literal(&mut self, bytes: &[u8], dst: Register) -> CompileResult<()> {
        let mut ctor_reg = self.builder.alloc_register();
        self.emit_load_builtin_name(ctor_reg, "bytes");

        if bytes.is_empty() {
            self.builder.emit_call(dst, ctor_reg, 0);
            self.builder.free_register(ctor_reg);
            return Ok(());
        }

        let call_block = self.builder.alloc_register_block(3);
        let call_block_end = call_block.0 + 3;
        if ctor_reg.0 >= call_block.0 && ctor_reg.0 < call_block_end {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, ctor_reg);
            self.builder.free_register(ctor_reg);
            ctor_reg = safe_reg;
        }

        let literal_reg = Register::new(call_block.0 + 1);
        let encoding_reg = Register::new(call_block.0 + 2);

        let mut latin1_text = String::with_capacity(bytes.len());
        for byte in bytes {
            latin1_text.push(char::from(*byte));
        }

        let literal_idx = self.builder.add_string(&latin1_text);
        self.builder.emit_load_const(literal_reg, literal_idx);

        let encoding_idx = self.builder.add_string("latin-1");
        self.builder.emit_load_const(encoding_reg, encoding_idx);

        self.builder.emit_call(call_block, ctor_reg, 2);
        self.builder.emit_move(dst, call_block);

        self.builder.free_register(ctor_reg);
        self.builder.free_register_block(call_block, 3);

        Ok(())
    }

    fn compile_joined_str(
        &mut self,
        parts: &[Expr],
        dst: Register,
        line: u32,
    ) -> CompileResult<()> {
        if parts.len() > u8::MAX as usize {
            return Err(CompileError {
                message: "f-string has too many parts".to_string(),
                line,
                column: 0,
            });
        }

        if parts.is_empty() {
            let empty_idx = self.builder.add_string("");
            self.builder.emit_load_const(dst, empty_idx);
            return Ok(());
        }

        let count = parts.len() as u8;
        let first_part = self.builder.alloc_register_block(count);
        for (index, part) in parts.iter().enumerate() {
            let part_reg = Register::new(first_part.0 + index as u8);
            self.compile_expr_into(part, part_reg)?;
        }

        self.builder.emit(Instruction::new(
            Opcode::BuildString,
            dst.0,
            first_part.0,
            count,
        ));
        self.builder.free_register_block(first_part, count);
        Ok(())
    }

    fn compile_formatted_value(
        &mut self,
        value: &Expr,
        conversion: i8,
        format_spec: Option<&Expr>,
        dst: Register,
    ) -> CompileResult<()> {
        let mut current_reg = self.compile_expr(value)?;

        if conversion >= 0 {
            let builtin_name = match conversion as u8 as char {
                's' => "str",
                'r' => "repr",
                'a' => "ascii",
                other => {
                    return Err(CompileError {
                        message: format!("unsupported f-string conversion '!{other}'"),
                        line: self.line_for_span(value.span),
                        column: 0,
                    });
                }
            };

            let converted_reg = self.builder.alloc_register();
            self.emit_named_call_from_regs(builtin_name, &[current_reg], converted_reg)?;
            self.builder.free_register(current_reg);
            current_reg = converted_reg;
        }

        let formatted_reg = if let Some(spec_expr) = format_spec {
            let spec_reg = self.compile_expr(spec_expr)?;
            let result_reg = self.builder.alloc_register();
            self.emit_named_call_from_regs("format", &[current_reg, spec_reg], result_reg)?;
            self.builder.free_register(spec_reg);
            result_reg
        } else {
            let result_reg = self.builder.alloc_register();
            self.emit_named_call_from_regs("format", &[current_reg], result_reg)?;
            result_reg
        };

        self.builder.free_register(current_reg);
        if formatted_reg != dst {
            self.builder.emit_move(dst, formatted_reg);
            self.builder.free_register(formatted_reg);
        }

        Ok(())
    }

    fn emit_named_call_from_regs(
        &mut self,
        name: &str,
        args: &[Register],
        dst: Register,
    ) -> CompileResult<()> {
        if args.len() > u8::MAX as usize {
            return Err(CompileError {
                message: format!("too many arguments for builtin call '{name}'"),
                line: 0,
                column: 0,
            });
        }

        let mut func_reg = self.builder.alloc_register();
        let location = self.resolve_variable(name);
        self.builder.emit_load_var(func_reg, location, Some(name));

        if args.is_empty() {
            self.builder.emit_call(dst, func_reg, 0);
            self.builder.free_register(func_reg);
            return Ok(());
        }

        let block_size = 1 + args.len() as u8;
        let call_block = self.builder.alloc_register_block(block_size);
        let call_block_end = call_block.0 + block_size;

        if func_reg.0 >= call_block.0 && func_reg.0 < call_block_end {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, func_reg);
            self.builder.free_register(func_reg);
            func_reg = safe_reg;
        }

        for (index, arg) in args.iter().enumerate() {
            let dst_reg = Register::new(call_block.0 + 1 + index as u8);
            if *arg != dst_reg {
                self.builder.emit_move(dst_reg, *arg);
            }
        }

        self.builder
            .emit_call(call_block, func_reg, args.len() as u8);
        if call_block != dst {
            self.builder.emit_move(dst, call_block);
        }

        self.builder.free_register(func_reg);
        self.builder.free_register_block(call_block, block_size);
        Ok(())
    }

    /// Compile an optimized method call using LoadMethod/CallMethod.
    ///
    /// This is used when the call expression is of the form `obj.method(args...)`.
    /// Instead of:
    ///   1. GetAttr to load bound method
    ///   2. Call the bound method
    ///
    /// We emit:
    ///   1. LoadMethod: loads method and self into consecutive registers
    ///   2. CallMethod: calls with self already in place
    ///
    /// This optimization avoids allocating a BoundMethod object on every call,
    /// providing 15-30% speedup on method-heavy code.
    ///
    /// # Register Layout
    /// ```text
    /// [method_reg]:     method/function object
    /// [method_reg+1]:   self instance
    /// [method_reg+2..]: explicit arguments
    /// ```
    fn compile_method_call(
        &mut self,
        obj_expr: &Expr,
        method_name: &str,
        args: &[Expr],
        dst: Register,
    ) -> CompileResult<Register> {
        // Step 1: Compile the object expression
        let obj_reg = self.compile_expr(obj_expr)?;

        // Step 2: Reserve a contiguous block for [method, self, arg0, arg1, ...]
        let block_size = 2u8
            .checked_add(args.len() as u8)
            .expect("method call register block overflow");
        let method_reg = self.builder.alloc_register_block(block_size);
        let self_reg = Register::new(method_reg.0 + 1);

        // Step 3: Emit LoadMethod - this populates method_reg and self_reg
        let name_idx = self.builder.add_name(method_name);
        self.builder.emit_load_method(method_reg, obj_reg, name_idx);

        // Free the object register since LoadMethod copies self to self_reg
        self.builder.free_register(obj_reg);

        // Step 4: Compile arguments into consecutive registers after self
        let arg_base = self_reg.0 + 1;
        let mut arg_regs = Vec::with_capacity(args.len());

        for (i, arg) in args.iter().enumerate() {
            let arg_dst = Register::new(arg_base + i as u8);
            // Reserve the register if it might be allocated elsewhere
            let temp = self.compile_expr(arg)?;
            if temp != arg_dst {
                self.builder.emit_move(arg_dst, temp);
                self.builder.free_register(temp);
            }
            arg_regs.push(arg_dst);
        }

        // Step 5: Emit CallMethod
        self.builder
            .emit_call_method(dst, method_reg, args.len() as u8);

        // Step 6: Cleanup - free the reserved method-call block
        let _ = arg_regs;
        self.builder.free_register_block(method_reg, block_size);

        Ok(dst)
    }

    /// Compile a store to a target.

    fn compile_store(&mut self, target: &Expr, value: Register) -> CompileResult<()> {
        match &target.kind {
            ExprKind::Name(name) => {
                // Use scope-aware variable resolution
                let location = self.resolve_variable(name);
                self.builder
                    .emit_store_var(location, value, Some(name.as_ref()));
            }

            ExprKind::Attribute {
                value: obj, attr, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let name_idx = self.builder.add_name(attr.clone());
                self.builder.emit_set_attr(obj_reg, name_idx, value);
                self.builder.free_register(obj_reg);
            }

            ExprKind::Subscript {
                value: obj, slice, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let key_reg = self.compile_expr(slice)?;
                self.builder.emit_set_item(obj_reg, key_reg, value);
                self.builder.free_register(obj_reg);
                self.builder.free_register(key_reg);
            }

            ExprKind::Tuple(elts) | ExprKind::List(elts) => {
                let starred_indices: Vec<usize> = elts
                    .iter()
                    .enumerate()
                    .filter_map(|(i, elt)| match &elt.kind {
                        ExprKind::Starred(_) => Some(i),
                        _ => None,
                    })
                    .collect();

                if starred_indices.len() > 1 {
                    return Err(CompileError {
                        message: "multiple starred expressions in assignment".to_string(),
                        line: self.line_for_span(target.span),
                        column: 0,
                    });
                }

                if let Some(star_idx) = starred_indices.first().copied() {
                    if elts.len() > u8::MAX as usize {
                        return Err(CompileError {
                            message: "too many assignment targets to unpack".to_string(),
                            line: self.line_for_span(target.span),
                            column: 0,
                        });
                    }

                    let before_count = star_idx;
                    let after_count = elts.len().saturating_sub(star_idx + 1);

                    if before_count > 0x0F || after_count > 0x0F {
                        return Err(CompileError {
                            message: "starred unpacking supports at most 15 items before/after '*'"
                                .to_string(),
                            line: self.line_for_span(target.span),
                            column: 0,
                        });
                    }

                    let packed = ((before_count as u8) << 4) | (after_count as u8);
                    let base = self.builder.alloc_register_block(elts.len() as u8);
                    self.builder.emit(Instruction::op_dss(
                        Opcode::UnpackEx,
                        base,
                        value,
                        Register::new(packed),
                    ));

                    for (i, elt) in elts.iter().enumerate() {
                        let item_reg = Register::new(base.0 + i as u8);
                        match &elt.kind {
                            ExprKind::Starred(inner) => self.compile_store(inner, item_reg)?,
                            _ => self.compile_store(elt, item_reg)?,
                        }
                    }

                    self.builder.free_register_block(base, elts.len() as u8);
                } else {
                    if elts.len() > u8::MAX as usize {
                        return Err(CompileError {
                            message: "too many assignment targets to unpack".to_string(),
                            line: self.line_for_span(target.span),
                            column: 0,
                        });
                    }

                    let base = self.builder.alloc_register_block(elts.len() as u8);
                    self.builder.emit(Instruction::op_dss(
                        Opcode::UnpackSequence,
                        base,
                        value,
                        Register::new(elts.len() as u8),
                    ));

                    for (i, elt) in elts.iter().enumerate() {
                        let item_reg = Register::new(base.0 + i as u8);
                        self.compile_store(elt, item_reg)?;
                    }

                    self.builder.free_register_block(base, elts.len() as u8);
                }
            }

            _ => {
                return Err(CompileError {
                    message: format!("cannot assign to {:?}", target.kind),
                    line: self.line_for_span(target.span),
                    column: 0,
                });
            }
        }

        Ok(())
    }

    fn validate_augassign_target(&self, target: &Expr) -> CompileResult<()> {
        match &target.kind {
            ExprKind::Tuple(_) | ExprKind::List(_) => Err(CompileError {
                message: "illegal expression for augmented assignment".to_string(),
                line: self.line_for_span(target.span),
                column: 0,
            }),
            _ => Ok(()),
        }
    }

    fn compile_delete_target(&mut self, target: &Expr) -> CompileResult<()> {
        match &target.kind {
            ExprKind::Name(name) => {
                let location = self.resolve_variable(name);
                self.builder.emit_delete_var(location, Some(name));
            }
            ExprKind::Attribute {
                value: obj, attr, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let name_idx = self.builder.add_name(attr.clone());
                self.builder.emit_del_attr(obj_reg, name_idx);
                self.builder.free_register(obj_reg);
            }
            ExprKind::Subscript {
                value: obj, slice, ..
            } => {
                let obj_reg = self.compile_expr(obj)?;
                let key_reg = self.compile_expr(slice)?;
                self.builder.emit_del_item(obj_reg, key_reg);
                self.builder.free_register(obj_reg);
                self.builder.free_register(key_reg);
            }
            ExprKind::Tuple(elts) | ExprKind::List(elts) => {
                for elt in elts {
                    self.compile_delete_target(elt)?;
                }
            }
            _ => {
                return Err(CompileError {
                    message: format!("cannot delete {:?}", target.kind),
                    line: self.line_for_span(target.span),
                    column: 0,
                });
            }
        }

        Ok(())
    }

    /// Emit a binary operation.
    fn emit_binop(&mut self, op: BinOp, dst: Register, left: Register, right: Register) {
        match op {
            BinOp::Add => self.builder.emit_add(dst, left, right),
            BinOp::Sub => self.builder.emit_sub(dst, left, right),
            BinOp::Mult => self.builder.emit_mul(dst, left, right),
            BinOp::Div => self.builder.emit_div(dst, left, right),
            BinOp::FloorDiv => self.builder.emit_floor_div(dst, left, right),
            BinOp::Mod => self.builder.emit_mod(dst, left, right),
            BinOp::Pow => self.builder.emit_pow(dst, left, right),
            BinOp::LShift => self.builder.emit_shl(dst, left, right),
            BinOp::RShift => self.builder.emit_shr(dst, left, right),
            BinOp::BitAnd => self.builder.emit_bitwise_and(dst, left, right),
            BinOp::BitOr => self.builder.emit_bitwise_or(dst, left, right),
            BinOp::BitXor => self.builder.emit_bitwise_xor(dst, left, right),
            BinOp::MatMult => {
                // Matrix multiplication - use generic multiply for now
                self.builder.emit_mul(dst, left, right)
            }
        }
    }

    /// Emit an augmented assignment operation.
    fn emit_augop(&mut self, op: AugOp, dst: Register, left: Register, right: Register) {
        match op {
            AugOp::Add => self.builder.emit_add(dst, left, right),
            AugOp::Sub => self.builder.emit_sub(dst, left, right),
            AugOp::Mult => self.builder.emit_mul(dst, left, right),
            AugOp::Div => self.builder.emit_div(dst, left, right),
            AugOp::FloorDiv => self.builder.emit_floor_div(dst, left, right),
            AugOp::Mod => self.builder.emit_mod(dst, left, right),
            AugOp::Pow => self.builder.emit_pow(dst, left, right),
            AugOp::LShift => self.builder.emit_shl(dst, left, right),
            AugOp::RShift => self.builder.emit_shr(dst, left, right),
            AugOp::BitAnd => self.builder.emit_bitwise_and(dst, left, right),
            AugOp::BitOr => self.builder.emit_bitwise_or(dst, left, right),
            AugOp::BitXor => self.builder.emit_bitwise_xor(dst, left, right),
            AugOp::MatMult => self.builder.emit_mul(dst, left, right),
        }
    }

    /// Emit a comparison operation.
    fn emit_cmpop(&mut self, op: CmpOp, dst: Register, left: Register, right: Register) {
        match op {
            CmpOp::Lt => self.builder.emit_lt(dst, left, right),
            CmpOp::LtE => self.builder.emit_le(dst, left, right),
            CmpOp::Eq => self.builder.emit_eq(dst, left, right),
            CmpOp::NotEq => self.builder.emit_ne(dst, left, right),
            CmpOp::Gt => self.builder.emit_gt(dst, left, right),
            CmpOp::GtE => self.builder.emit_ge(dst, left, right),
            CmpOp::Is => self
                .builder
                .emit(Instruction::op_dss(Opcode::Is, dst, left, right)),
            CmpOp::IsNot => self
                .builder
                .emit(Instruction::op_dss(Opcode::IsNot, dst, left, right)),
            CmpOp::In => self
                .builder
                .emit(Instruction::op_dss(Opcode::In, dst, left, right)),
            CmpOp::NotIn => self
                .builder
                .emit(Instruction::op_dss(Opcode::NotIn, dst, left, right)),
        }
    }

    fn emit_yield_from(&mut self, dst: Register, iterable_reg: Register) {
        if dst == iterable_reg {
            let source_reg = self.builder.alloc_register();
            self.builder.emit_move(source_reg, iterable_reg);
            self.builder.emit_load_none(dst);
            self.builder
                .emit(Instruction::op_ds(Opcode::YieldFrom, dst, source_reg));
            self.builder.free_register(source_reg);
            return;
        }

        self.builder.emit_load_none(dst);
        self.builder
            .emit(Instruction::op_ds(Opcode::YieldFrom, dst, iterable_reg));
    }

    // =========================================================================
    // Dynamic Call Compilation (with *args/**kwargs unpacking)
    // =========================================================================

    /// Compile a function call that contains *args or **kwargs unpacking.
    ///
    /// This builds a tuple for positional arguments and a dict for keyword arguments,
    /// then uses CallEx to invoke the function with the unpacked args.
    ///
    /// # Algorithm
    /// 1. Compile function expression
    /// 2. Build positional args into a tuple (merging any *iterables)
    /// 3. Build keyword args into a dict (merging any **mappings)
    /// 4. Emit CallEx(dst, func, args_tuple, kwargs_dict)
    fn compile_dynamic_call(
        &mut self,
        func: &Expr,
        args: &[Expr],
        keywords: &[prism_parser::ast::Keyword],
        dst: Register,
        line: u32,
    ) -> CompileResult<Register> {
        // Step 1: Compile function
        let func_reg = self.compile_expr(func)?;

        // Step 2: Build positional args tuple
        // Separate regular args from starred args (for unpack flags)
        let args_tuple_reg = if args.is_empty() {
            // Empty tuple - use BuildTuple with 0 count
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple(tuple_reg, tuple_reg, 0);
            tuple_reg
        } else {
            if args.len() > 24 {
                return Err(CompileError {
                    message: "call-site *args unpack supports at most 24 positional entries"
                        .to_string(),
                    line,
                    column: 0,
                });
            }
            // Compile each arg and track which are starred
            let base_reg = self.builder.alloc_register_block(args.len() as u8);
            let mut unpack_flags: u32 = 0;

            for (i, arg) in args.iter().enumerate() {
                let arg_reg = Register::new(base_reg.0 + i as u8);

                match &arg.kind {
                    ExprKind::Starred(inner) => {
                        // This is a *iterable - compile the inner expression
                        let temp = self.compile_expr(inner)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                        // Mark this position for unpacking
                        unpack_flags |= 1 << i;
                    }
                    _ => {
                        // Regular arg - compile directly
                        let temp = self.compile_expr(arg)?;
                        if temp != arg_reg {
                            self.builder.emit_move(arg_reg, temp);
                        }
                        self.builder.free_register(temp);
                    }
                }
            }

            // Build tuple with unpacking (merges starred iterables)
            let tuple_reg = self.builder.alloc_register();
            self.builder.emit_build_tuple_unpack(
                tuple_reg,
                base_reg,
                args.len() as u8,
                unpack_flags,
            );

            // Free arg register block
            self.builder.free_register_block(base_reg, args.len() as u8);

            tuple_reg
        };

        // Step 3: Build keyword args dict (if any)
        let kwargs_dict_reg = if keywords.is_empty() {
            None
        } else {
            if keywords.len() > 24 {
                return Err(CompileError {
                    message: "call-site **kwargs unpack supports at most 24 keyword entries"
                        .to_string(),
                    line,
                    column: 0,
                });
            }

            // Represent every entry as a mapping in `base_reg+i`, then merge.
            let base_reg = self.builder.alloc_register_block(keywords.len() as u8);
            let mut unpack_flags: u32 = 0;

            for (i, kw) in keywords.iter().enumerate() {
                let entry_reg = Register::new(base_reg.0 + i as u8);

                if kw.arg.is_none() {
                    // **mapping entry
                    let temp = self.compile_expr(&kw.value)?;
                    if temp != entry_reg {
                        self.builder.emit_move(entry_reg, temp);
                    }
                    self.builder.free_register(temp);
                    unpack_flags |= 1 << i; // merge this mapping
                } else {
                    // Static keyword: build singleton dict {"name": value}
                    let key_name = kw.arg.as_ref().unwrap();
                    let key_idx = self.builder.add_string(key_name);
                    let pair_base = self.builder.alloc_register_block(2);
                    let key_reg = pair_base;
                    let val_reg = Register::new(pair_base.0 + 1);
                    self.builder.emit_load_const(key_reg, key_idx);

                    let temp = self.compile_expr(&kw.value)?;
                    if temp != val_reg {
                        self.builder.emit_move(val_reg, temp);
                    }
                    self.builder.free_register(temp);

                    self.builder.emit(Instruction::new(
                        Opcode::BuildDict,
                        entry_reg.0,
                        pair_base.0,
                        1,
                    ));
                    self.builder.free_register_block(pair_base, 2);
                    unpack_flags |= 1 << i; // merge singleton mapping
                }
            }

            // Build dict with potential unpacking
            let dict_reg = self.builder.alloc_register();
            self.builder.emit_build_dict_unpack(
                dict_reg,
                base_reg,
                keywords.len() as u8,
                unpack_flags,
            );

            // Free mapping entry block
            self.builder
                .free_register_block(base_reg, keywords.len() as u8);

            Some(dict_reg)
        };

        // Step 4: Emit CallEx
        self.builder
            .emit_call_ex(dst, func_reg, args_tuple_reg, kwargs_dict_reg);

        // Cleanup
        self.builder.free_register(func_reg);
        self.builder.free_register(args_tuple_reg);
        if let Some(kr) = kwargs_dict_reg {
            self.builder.free_register(kr);
        }

        Ok(dst)
    }

    // =========================================================================
    // Exception Handling Compilation
    // =========================================================================

    /// Compile a try/except/finally statement with zero-cost exception handling.
    ///
    /// This generates exception table entries for the VM's table-driven unwinder.
    /// No runtime opcodes are executed on try block entry/exit - the exception
    /// table is consulted only when an exception is raised.
    ///
    /// # Layout
    /// ```text
    /// try_start:
    ///     <try body>              # Protected by exception entry
    ///     JUMP end_label          # Skip handlers on normal exit
    /// handler_0:                  # except Type1 as e:
    ///     <check exception type>
    ///     <handler body>
    ///     JUMP end_label
    /// handler_1:                  # except Type2:
    ///     <handler body>
    ///     JUMP end_label
    /// finally:                    # finally:
    ///     <finally body>
    /// end_label:
    /// ```
    fn compile_try(
        &mut self,
        body: &[Stmt],
        handlers: &[ExceptHandler],
        orelse: &[Stmt],
        finalbody: &[Stmt],
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        // =================================================================
        // Analysis Phase - Determine handler structure
        // =================================================================

        // Check if there's a bare except clause (catches all exceptions)
        let has_bare_except = handlers.iter().any(|h| h.typ.is_none());

        // Check if there are any typed handlers that need matching
        let has_typed_handlers = handlers.iter().any(|h| h.typ.is_some());

        // =================================================================
        // Label Creation Phase
        // =================================================================

        let end_label = self.builder.create_label();

        let orelse_label = if !orelse.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };

        let finally_label = if !finalbody.is_empty() {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Only create reraise label if we have typed handlers AND no bare except
        // (if there's a bare except, it will catch everything, so no reraise needed)
        let reraise_label = if has_typed_handlers && !has_bare_except {
            Some(self.builder.create_label())
        } else {
            None
        };

        // Create handler labels (one per except clause)
        let handler_labels: Vec<_> = handlers
            .iter()
            .map(|_| self.builder.create_label())
            .collect();

        let has_finally = !finalbody.is_empty();
        if has_finally {
            let return_label = self.builder.create_label();
            let return_value_reg = self.builder.alloc_register();
            self.finally_stack.push(FinallyContext {
                return_label,
                return_value_reg,
                return_used: false,
                jump_continuations: SmallVec::new(),
            });
        }

        // =================================================================
        // Try Body Compilation
        // =================================================================

        let try_start_pc = self.builder.current_pc();
        let stack_depth = self.builder.current_stack_depth();

        for stmt in body {
            self.compile_stmt(stmt)?;
        }

        let try_end_pc = self.builder.current_pc();

        // Jump to else/finally/end on normal completion (no exception)
        if let Some(else_label) = orelse_label {
            self.builder.emit_jump(else_label);
        } else if let Some(fin_label) = finally_label {
            self.builder.emit_jump(fin_label);
        } else if !handlers.is_empty() {
            self.builder.emit_jump(end_label);
        }

        // =================================================================
        // Exception Handler Compilation
        // =================================================================

        for (i, handler) in handlers.iter().enumerate() {
            self.builder.bind_label(handler_labels[i]);

            let handler_start_pc = self.builder.current_pc();
            let handler_abort_label = self.builder.create_label();

            // Compile handler match logic
            if let Some(type_expr) = &handler.typ {
                // -----------------------------------------------------------
                // Typed handler: except SomeException as e:
                // -----------------------------------------------------------

                // Compile the exception type expression to get the type class
                let type_reg = self.compile_expr(type_expr)?;

                // Load the current exception into a register for later binding
                let exc_reg = self.builder.alloc_register();
                self.builder
                    .emit(Instruction::op_d(Opcode::LoadException, exc_reg));

                // Check if exception matches type using dynamic matching
                // Note: ExceptionMatch reads src1 as the type, gets exception from VM state
                let match_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_ds(
                    Opcode::ExceptionMatch,
                    match_reg,
                    type_reg,
                ));

                // Determine where to jump if no match
                let no_match_target = if i + 1 < handlers.len() {
                    // Try next handler
                    handler_labels[i + 1]
                } else if let Some(reraise_lbl) = reraise_label {
                    // No more handlers, reraise the exception
                    reraise_lbl
                } else if let Some(fin_label) = finally_label {
                    // No reraise needed, go to finally (bare except will catch)
                    fin_label
                } else {
                    // Should not happen if has_bare_except is true
                    end_label
                };

                self.builder.emit_jump_if_false(match_reg, no_match_target);

                self.builder.free_register(match_reg);
                self.builder.free_register(type_reg);

                // If handler has a name binding (except E as e:), store the exception
                if let Some(name) = &handler.name {
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                }

                self.builder.free_register(exc_reg);
            } else {
                // -----------------------------------------------------------
                // Bare except: catches all exceptions
                // -----------------------------------------------------------

                if let Some(name) = &handler.name {
                    let exc_reg = self.builder.alloc_register();
                    self.builder
                        .emit(Instruction::op_d(Opcode::LoadException, exc_reg));
                    let location = self.resolve_variable(name);
                    self.builder
                        .emit_store_var(location, exc_reg, Some(name.as_ref()));
                    self.builder.free_register(exc_reg);
                }
            };

            // =============================================================
            // Handler Body Execution
            // =============================================================

            self.builder.emit(Instruction::op(Opcode::EnterExcept));
            let handler_body_start_pc = self.builder.current_pc();

            // Compile handler body
            for stmt in &handler.body {
                self.compile_stmt(stmt)?;
            }
            let handler_body_end_pc = self.builder.current_pc();

            // Successful handler completion restores any outer handler context
            // and fully clears the exception when this was the outermost handler.
            self.builder.emit(Instruction::op(Opcode::ExitExcept));

            // Jump to finally or end after successful handler execution
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }

            self.builder.bind_label(handler_abort_label);
            let handler_abort_pc = self.builder.current_pc();
            self.builder.emit(Instruction::op(Opcode::AbortExcept));
            self.builder.emit(Instruction::op(Opcode::Reraise));

            // Add exception entry for this handler
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: handler_start_pc,
                finally_pc: u32::MAX,
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX,
            });

            if handler_body_start_pc < handler_body_end_pc {
                self.builder.add_exception_entry(ExceptionEntry {
                    start_pc: handler_body_start_pc,
                    end_pc: handler_body_end_pc,
                    handler_pc: handler_abort_pc,
                    finally_pc: u32::MAX,
                    depth: stack_depth as u16,
                    exception_type_idx: u16::MAX,
                });
            }
        }

        // =================================================================
        // Else Block Compilation (runs only if no exception occurred)
        // =================================================================

        if let Some(else_label) = orelse_label {
            self.builder.bind_label(else_label);
            for stmt in orelse {
                self.compile_stmt(stmt)?;
            }
            if let Some(fin_label) = finally_label {
                self.builder.emit_jump(fin_label);
            } else {
                self.builder.emit_jump(end_label);
            }
        }

        // =================================================================
        // Reraise Path (only if typed handlers exist without bare except)
        // =================================================================

        if let Some(reraise_lbl) = reraise_label {
            self.builder.bind_label(reraise_lbl);

            if let Some(fin_label) = finally_label {
                // Execute finally before reraising
                self.builder.emit_jump(fin_label);
            } else {
                // No finally, reraise immediately
                self.builder.emit(Instruction::op(Opcode::Reraise));
            }
        }

        // =================================================================
        // Finally Block Compilation
        // =================================================================

        let cleanup_context = if has_finally {
            Some(
                self.finally_stack
                    .pop()
                    .expect("try/finally should have an active finally context"),
            )
        } else {
            None
        };

        if let Some(fin_label) = finally_label {
            self.builder.bind_label(fin_label);
            let finally_start_pc = self.builder.current_pc();

            // Push exception info to preserve state during finally execution
            self.builder.emit(Instruction::op(Opcode::PushExcInfo));

            // Compile finally body
            for stmt in finalbody {
                self.compile_stmt(stmt)?;
            }

            // Pop exception info
            self.builder.emit(Instruction::op(Opcode::PopExcInfo));

            // EndFinally will reraise if there's a pending exception
            self.builder.emit(Instruction::op(Opcode::EndFinally));
            self.builder.emit_jump(end_label);

            // Add finally exception entry
            self.builder.add_exception_entry(ExceptionEntry {
                start_pc: try_start_pc,
                end_pc: try_end_pc,
                handler_pc: finally_start_pc,
                finally_pc: finally_start_pc,
                depth: stack_depth as u16,
                exception_type_idx: u16::MAX,
            });
        }

        if let Some(cleanup_context) = cleanup_context {
            if cleanup_context.return_used {
                self.builder.bind_label(cleanup_context.return_label);
                self.compile_finally_cleanup_body(finalbody)?;
                self.emit_return_value(cleanup_context.return_value_reg);
            } else {
                self.builder.free_register(cleanup_context.return_value_reg);
            }

            for continuation in cleanup_context.jump_continuations {
                self.builder.bind_label(continuation.cleanup_label);
                self.compile_finally_cleanup_body(finalbody)?;
                self.emit_jump_through_finally_until(
                    continuation.target_label,
                    continuation.preserve_finally_depth,
                );
            }
        }

        // =================================================================
        // End Label - Normal exit point
        // =================================================================

        self.builder.bind_label(end_label);

        Ok(())
    }

    // =========================================================================
    // With Statement (Context Manager) Compilation
    // =========================================================================

    /// Compile a with statement.
    ///
    /// The with statement implements the context manager protocol:
    ///
    /// ```python
    /// with expr as var:
    ///     body
    /// ```
    ///
    /// Is equivalent to:
    /// ```python
    /// mgr = expr
    /// value = mgr.__enter__()
    /// try:
    ///     var = value  # if as clause present
    ///     body
    /// except:
    ///     if not mgr.__exit__(*sys.exc_info()):
    ///         raise
    /// else:
    ///     mgr.__exit__(None, None, None)
    /// ```
    ///
    /// For multiple context managers, they are nested from left to right:
    /// ```python
    /// with a as x, b as y:
    ///     body
    /// # is equivalent to:
    /// with a as x:
    ///     with b as y:
    ///         body
    /// ```
    fn compile_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        // Compile nested context managers recursively
        self.compile_with_items(items, body, 0)
    }

    /// Compile with statement items recursively for nested context managers.
    fn compile_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __enter__ and __exit__ methods
        let enter_name_idx = self.builder.add_name("__enter__");
        let exit_name_idx = self.builder.add_name("__exit__");

        // Step 3: Load __exit__ method (need to store for cleanup)
        // We store both the manager and __exit__ bound method for cleanup
        let exit_method_reg = self.builder.alloc_register_block(5);
        self.builder
            .emit_load_method(exit_method_reg, mgr_reg, exit_name_idx);

        // Step 4: Load __enter__ method
        let enter_method_reg = self.builder.alloc_register_block(2);
        self.builder
            .emit_load_method(enter_method_reg, mgr_reg, enter_name_idx);

        // Step 5: Call __enter__() -> value
        let value_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(value_reg, enter_method_reg, 0);
        self.builder.free_register_block(enter_method_reg, 2);

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        // Record try block start position
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();
        let return_label = self.builder.create_label();
        let return_value_reg = self.builder.alloc_register();
        self.finally_stack.push(FinallyContext {
            return_label,
            return_value_reg,
            return_used: false,
            jump_continuations: SmallVec::new(),
        });

        // Step 8: Compile nested items and body
        self.compile_with_items(items, body, depth + 1)?;

        // Step 9: Record try block end position (normal exit path)
        let try_end_pc = self.builder.current_pc();
        let cleanup_context = self
            .finally_stack
            .pop()
            .expect("with statement should have an active cleanup context");

        // Step 10: Normal exit - call __exit__(None, None, None)
        self.emit_context_exit_none(exit_method_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 11: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        // Load the active exception instance and compute its concrete class for
        // __exit__(exc_type, exc, tb) compatibility with unittest/assertRaises.
        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));
        self.emit_exception_type_attr(exc_type_reg, exc_val_reg);
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __exit__(type, value, tb)
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 2),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 3),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(exit_method_reg.0 + 4),
            exc_tb_reg,
        ));
        self.builder
            .emit_call_method(suppress_reg, exit_method_reg, 3);

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // A truthy __exit__ suppresses the active exception entirely.
        let suppress_label = self.builder.create_label();
        self.builder.emit_jump_if_true(suppress_reg, suppress_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        self.builder.bind_label(suppress_label);
        self.builder.emit(Instruction::op(Opcode::ClearException));
        self.builder.emit_jump(end_label);

        if cleanup_context.return_used {
            self.builder.bind_label(cleanup_context.return_label);
            self.emit_context_exit_none(exit_method_reg);
            self.emit_return_value(cleanup_context.return_value_reg);
        } else {
            self.builder.free_register(cleanup_context.return_value_reg);
        }

        for continuation in cleanup_context.jump_continuations {
            self.builder.bind_label(continuation.cleanup_label);
            self.emit_context_exit_none(exit_method_reg);
            self.emit_jump_through_finally_until(
                continuation.target_label,
                continuation.preserve_finally_depth,
            );
        }

        // Step 12: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register_block(exit_method_reg, 5);
        self.builder.free_register(mgr_reg);

        // Step 13: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX, // No separate finally, cleanup handles both
            depth: depth as u16,
            exception_type_idx: u16::MAX, // Catches all exceptions
        });

        Ok(())
    }

    // =========================================================================
    // Async With Statement Compilation
    // =========================================================================

    /// Compile async with statement with awaited __aenter__/__aexit__.
    ///
    /// `async with ctx as var:` compiles to roughly:
    ///   mgr = ctx
    ///   aexit = mgr.__aexit__
    ///   aenter = mgr.__aenter__
    ///   val = await aenter()
    ///   var = val
    ///   try:
    ///       <body>
    ///   except:
    ///       if not await aexit(type, val, tb): raise
    ///   else:
    ///       await aexit(None, None, None)
    fn compile_async_with(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
    ) -> CompileResult<()> {
        self.compile_async_with_items(items, body, 0)
    }

    /// Compile async with statement items recursively for nested async context managers.
    fn compile_async_with_items(
        &mut self,
        items: &[prism_parser::ast::WithItem],
        body: &[Stmt],
        depth: usize,
    ) -> CompileResult<()> {
        use crate::ExceptionEntry;

        if depth >= items.len() {
            // All async context managers set up, compile the body
            for stmt in body {
                self.compile_stmt(stmt)?;
            }
            return Ok(());
        }

        let item = &items[depth];

        // Step 1: Evaluate context expression -> mgr
        let mgr_reg = self.compile_expr(&item.context_expr)?;

        // Step 2: Look up __aenter__ and __aexit__ methods
        let aenter_name_idx = self.builder.add_name("__aenter__");
        let aexit_name_idx = self.builder.add_name("__aexit__");

        // Step 3: Load __aexit__ method (need to store for cleanup)
        let aexit_method_reg = self.builder.alloc_register_block(5);
        self.builder
            .emit_load_method(aexit_method_reg, mgr_reg, aexit_name_idx);

        // Step 4: Load __aenter__ method
        let aenter_method_reg = self.builder.alloc_register_block(2);
        self.builder
            .emit_load_method(aenter_method_reg, mgr_reg, aenter_name_idx);

        // Step 5: Call __aenter__() and AWAIT the result
        let aenter_awaitable_reg = self.builder.alloc_register();
        self.builder
            .emit_call_method(aenter_awaitable_reg, aenter_method_reg, 0);
        self.builder.free_register_block(aenter_method_reg, 2);

        // Await the __aenter__ result: GetAwaitable + YieldFrom
        let value_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            value_reg,
            aenter_awaitable_reg,
        ));
        self.builder.free_register(aenter_awaitable_reg);
        self.emit_yield_from(value_reg, value_reg);

        // Step 6: If there's an as-clause, bind the value
        if let Some(optional_vars) = &item.optional_vars {
            self.compile_store(optional_vars, value_reg)?;
        }
        self.builder.free_register(value_reg);

        // Step 7: Set up exception handling for the body
        let try_start_pc = self.builder.current_pc();
        let cleanup_label = self.builder.create_label();
        let end_label = self.builder.create_label();
        let return_label = self.builder.create_label();
        let return_value_reg = self.builder.alloc_register();
        self.finally_stack.push(FinallyContext {
            return_label,
            return_value_reg,
            return_used: false,
            jump_continuations: SmallVec::new(),
        });

        // Step 8: Compile nested items and body
        self.compile_async_with_items(items, body, depth + 1)?;

        // Step 9: Normal exit path
        let try_end_pc = self.builder.current_pc();
        let cleanup_context = self
            .finally_stack
            .pop()
            .expect("async with statement should have an active cleanup context");

        self.emit_async_context_exit_none(aexit_method_reg);

        // Jump to end (skip exception path)
        self.builder.emit_jump(end_label);

        // Step 10: Exception cleanup path
        self.builder.bind_label(cleanup_label);
        let cleanup_start_pc = self.builder.current_pc();

        // Push exception info for cleanup
        self.builder.emit(Instruction::op(Opcode::PushExcInfo));

        // Load exception info registers
        let exc_type_reg = self.builder.alloc_register();
        let exc_val_reg = self.builder.alloc_register();
        let exc_tb_reg = self.builder.alloc_register();

        self.builder
            .emit(Instruction::op_d(Opcode::LoadException, exc_val_reg));
        self.emit_exception_type_attr(exc_type_reg, exc_val_reg);
        self.builder
            .emit(Instruction::op_d(Opcode::LoadNone, exc_tb_reg));

        // Call __aexit__(type, value, tb)
        let suppress_awaitable_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 2),
            exc_type_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 3),
            exc_val_reg,
        ));
        self.builder.emit(Instruction::op_ds(
            Opcode::Move,
            Register::new(aexit_method_reg.0 + 4),
            exc_tb_reg,
        ));
        self.builder
            .emit_call_method(suppress_awaitable_reg, aexit_method_reg, 3);

        // Await the __aexit__ result for exception case
        let suppress_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_ds(
            Opcode::GetAwaitable,
            suppress_reg,
            suppress_awaitable_reg,
        ));
        self.builder.free_register(suppress_awaitable_reg);
        self.emit_yield_from(suppress_reg, suppress_reg);

        // Pop exception info
        self.builder.emit(Instruction::op(Opcode::PopExcInfo));

        // A truthy __aexit__ suppresses the active exception entirely.
        let suppress_label = self.builder.create_label();
        self.builder.emit_jump_if_true(suppress_reg, suppress_label);

        self.builder.free_register(exc_type_reg);
        self.builder.free_register(exc_val_reg);
        self.builder.free_register(exc_tb_reg);
        self.builder.free_register(suppress_reg);

        // Reraise the exception
        self.builder.emit(Instruction::op(Opcode::Reraise));

        self.builder.bind_label(suppress_label);
        self.builder.emit(Instruction::op(Opcode::ClearException));
        self.builder.emit_jump(end_label);

        if cleanup_context.return_used {
            self.builder.bind_label(cleanup_context.return_label);
            self.emit_async_context_exit_none(aexit_method_reg);
            self.emit_return_value(cleanup_context.return_value_reg);
        } else {
            self.builder.free_register(cleanup_context.return_value_reg);
        }

        for continuation in cleanup_context.jump_continuations {
            self.builder.bind_label(continuation.cleanup_label);
            self.emit_async_context_exit_none(aexit_method_reg);
            self.emit_jump_through_finally_until(
                continuation.target_label,
                continuation.preserve_finally_depth,
            );
        }

        // Step 11: End label
        self.builder.bind_label(end_label);

        // Free the stored method and manager registers
        self.builder.free_register_block(aexit_method_reg, 5);
        self.builder.free_register(mgr_reg);

        // Step 12: Add exception table entry for cleanup
        self.builder.add_exception_entry(ExceptionEntry {
            start_pc: try_start_pc,
            end_pc: try_end_pc,
            handler_pc: cleanup_start_pc,
            finally_pc: u32::MAX,
            depth: depth as u16,
            exception_type_idx: u16::MAX,
        });

        Ok(())
    }

    // =========================================================================
    // Match Statement (Pattern Matching) Compilation
    // =========================================================================

    /// Compile a match statement using Maranget's decision tree algorithm.
    ///
    /// This implements Python 3.10+ structural pattern matching (PEP 634).
    /// The algorithm:
    /// 1. Evaluate the subject expression once
    /// 2. Build a pattern matrix from all cases
    /// 3. Generate a decision tree for optimal pattern testing
    /// 4. Emit bytecode that traverses the decision tree
    fn compile_match(
        &mut self,
        subject: &Expr,
        cases: &[prism_parser::ast::MatchCase],
    ) -> CompileResult<()> {
        // Step 1: Compile subject expression and store in register
        let subject_reg = self.compile_expr(subject)?;

        // Step 2: Create labels for each case and the end
        let end_label = self.builder.create_label();
        let case_labels: Vec<Label> = cases.iter().map(|_| self.builder.create_label()).collect();

        // Step 3: Compile pattern tests and bindings for each case
        // We compile cases in order, with fallthrough to next case on failure
        for (i, case) in cases.iter().enumerate() {
            let next_label = if i + 1 < cases.len() {
                case_labels[i + 1]
            } else {
                end_label
            };

            // Compile pattern match
            self.compile_pattern_match(&case.pattern, subject_reg, next_label)?;

            // Compile guard if present
            if let Some(guard) = &case.guard {
                let guard_reg = self.compile_expr(guard)?;
                self.builder.emit_jump_if_false(guard_reg, next_label);
                self.builder.free_register(guard_reg);
            }

            // Compile case body if pattern (and guard) matched
            for stmt in &case.body {
                self.compile_stmt(stmt)?;
            }

            // Jump to end after executing matched case
            self.builder.emit_jump(end_label);

            // Bind next case label for fallthrough
            if i + 1 < cases.len() {
                self.builder.bind_label(case_labels[i + 1]);
            }
        }

        // End label
        self.builder.bind_label(end_label);
        self.builder.free_register(subject_reg);

        Ok(())
    }

    /// Compile a pattern match test.
    ///
    /// On success, any bindings are stored to locals and execution continues.
    /// On failure, jumps to fail_label.
    fn compile_pattern_match(
        &mut self,
        pattern: &prism_parser::ast::Pattern,
        subject_reg: Register,
        fail_label: Label,
    ) -> CompileResult<()> {
        use prism_parser::ast::PatternKind;

        match &pattern.kind {
            PatternKind::MatchValue(expr) => {
                // Value pattern: subject == expr
                let value_reg = self.compile_expr(expr)?;
                let result_reg = self.builder.alloc_register();
                self.builder.emit_eq(result_reg, subject_reg, value_reg);
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(value_reg);
            }

            PatternKind::MatchSingleton(singleton) => {
                // Singleton pattern: subject is True/False/None
                use prism_parser::ast::Singleton;
                let cmp_reg = self.builder.alloc_register();
                match singleton {
                    Singleton::True => self.builder.emit_load_true(cmp_reg),
                    Singleton::False => self.builder.emit_load_false(cmp_reg),
                    Singleton::None => self.builder.emit_load_none(cmp_reg),
                }
                let result_reg = self.builder.alloc_register();
                self.builder.emit(Instruction::op_dss(
                    Opcode::Is,
                    result_reg,
                    subject_reg,
                    cmp_reg,
                ));
                self.builder.emit_jump_if_false(result_reg, fail_label);
                self.builder.free_register(result_reg);
                self.builder.free_register(cmp_reg);
            }

            PatternKind::MatchSequence(patterns) => {
                if patterns
                    .iter()
                    .any(|sub_pattern| matches!(sub_pattern.kind, PatternKind::MatchStar(_)))
                {
                    return Err(self.unsupported_pattern_error(
                        pattern,
                        "sequence star patterns require variable-length slicing and rest binding semantics",
                    ));
                }

                // Sequence pattern: [a, b, c]
                // First check if subject is a sequence type using MatchSequence opcode
                let is_seq_reg = self.builder.alloc_register();
                crate::match_compiler::emit_match_sequence(
                    &mut self.builder,
                    is_seq_reg,
                    subject_reg,
                );
                self.builder.emit_jump_if_false(is_seq_reg, fail_label);
                self.builder.free_register(is_seq_reg);

                // Check length
                let len_reg = self.builder.alloc_register();
                let len_name = self.builder.add_name(Arc::from("__len__"));
                let method_reg = self.builder.alloc_register();
                self.builder
                    .emit_get_attr(method_reg, subject_reg, len_name);
                self.builder.emit_call(len_reg, method_reg, 0);
                self.builder.free_register(method_reg);

                // Check length
                let expected_len = self.builder.add_int(patterns.len() as i64);
                let expected_reg = self.builder.alloc_register();
                self.builder.emit_load_const(expected_reg, expected_len);
                let cmp_reg = self.builder.alloc_register();
                self.builder.emit_eq(cmp_reg, len_reg, expected_reg);
                self.builder.emit_jump_if_false(cmp_reg, fail_label);
                self.builder.free_register(cmp_reg);
                self.builder.free_register(expected_reg);
                self.builder.free_register(len_reg);

                // Match each element
                for (idx, sub_pattern) in patterns.iter().enumerate() {
                    let idx_const = self.builder.add_int(idx as i64);
                    let idx_reg = self.builder.alloc_register();
                    self.builder.emit_load_const(idx_reg, idx_const);
                    let elem_reg = self.builder.alloc_register();
                    self.builder.emit_get_item(elem_reg, subject_reg, idx_reg);
                    self.compile_pattern_match(sub_pattern, elem_reg, fail_label)?;
                    self.builder.free_register(elem_reg);
                    self.builder.free_register(idx_reg);
                }
            }

            PatternKind::MatchMapping { .. } => {
                return Err(self.unsupported_pattern_error(
                    pattern,
                    "mapping patterns require missing-key failure, duplicate-key validation, rest-copy allocation, and full mapping protocol semantics",
                ));
            }

            PatternKind::MatchClass {
                cls,
                patterns,
                kwd_attrs,
                kwd_patterns,
            } => {
                self.compile_class_pattern_match(
                    pattern,
                    cls,
                    patterns,
                    kwd_attrs,
                    kwd_patterns,
                    subject_reg,
                    fail_label,
                )?;
            }

            PatternKind::MatchStar(_name) => {
                return Err(self.unsupported_pattern_error(
                    pattern,
                    "star patterns are only valid inside fully implemented sequence patterns",
                ));
            }

            PatternKind::MatchAs { pattern, name } => {
                // As pattern: pattern as name, or just name (wildcard)
                // First match the inner pattern if any
                if let Some(inner) = pattern {
                    self.compile_pattern_match(inner, subject_reg, fail_label)?;
                }

                // Bind the name if present (None means wildcard _)
                if let Some(bound_name) = name {
                    // Use resolve_variable to properly handle scope
                    match self.resolve_variable(bound_name) {
                        VarLocation::Local(slot) => {
                            self.builder
                                .emit_store_local(LocalSlot::new(slot), subject_reg);
                        }
                        VarLocation::Global => {
                            let name_idx = self.builder.add_name(Arc::from(bound_name.as_ref()));
                            self.builder.emit_store_global(name_idx, subject_reg);
                        }
                        VarLocation::Closure(slot) => {
                            self.builder.emit_store_closure(slot, subject_reg);
                        }
                    }
                }
                // If name is None, it's just a wildcard _ which always matches
            }

            PatternKind::MatchOr(alternatives) => {
                // Or pattern: pattern1 | pattern2 | ...
                // Match succeeds if any alternative matches
                let success_label = self.builder.create_label();

                for (i, alt) in alternatives.iter().enumerate() {
                    let is_last = i + 1 == alternatives.len();

                    if is_last {
                        // Last alternative - fail to outer fail_label
                        self.compile_pattern_match(alt, subject_reg, fail_label)?;
                    } else {
                        // Not last - create temp fail label
                        let temp_fail = self.builder.create_label();
                        self.compile_pattern_match(alt, subject_reg, temp_fail)?;
                        // Match succeeded - jump to success
                        self.builder.emit_jump(success_label);
                        self.builder.bind_label(temp_fail);
                    }
                }

                self.builder.bind_label(success_label);
            }
        }

        Ok(())
    }

    fn compile_class_pattern_match(
        &mut self,
        pattern: &prism_parser::ast::Pattern,
        cls: &Expr,
        patterns: &[prism_parser::ast::Pattern],
        kwd_attrs: &[String],
        kwd_patterns: &[prism_parser::ast::Pattern],
        subject_reg: Register,
        fail_label: Label,
    ) -> CompileResult<()> {
        if kwd_attrs.len() != kwd_patterns.len() {
            return Err(self.unsupported_pattern_error(
                pattern,
                "class pattern keyword attribute and pattern counts differ",
            ));
        }

        let mut seen_attrs = std::collections::HashSet::new();
        for attr in kwd_attrs {
            if !seen_attrs.insert(attr.as_str()) {
                return Err(CompileError {
                    message: format!("duplicate attribute name in class pattern: {attr}"),
                    line: self.line_for_span(pattern.span),
                    column: 0,
                });
            }
        }

        if patterns.len() > u8::MAX as usize {
            return Err(CompileError {
                message: "class pattern has too many positional subpatterns".to_string(),
                line: self.line_for_span(pattern.span),
                column: 0,
            });
        }

        let class_reg = self.compile_expr(cls)?;
        let matches_class_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_dss(
            Opcode::MatchClass,
            matches_class_reg,
            subject_reg,
            class_reg,
        ));
        self.builder
            .emit_jump_if_false(matches_class_reg, fail_label);
        self.builder.free_register(matches_class_reg);

        if !patterns.is_empty() {
            let match_args_reg = self.builder.alloc_register();
            self.builder.emit(Instruction::op_ds(
                Opcode::GetMatchArgs,
                match_args_reg,
                class_reg,
            ));

            for (index, sub_pattern) in patterns.iter().enumerate() {
                let index_const = self.builder.add_int(index as i64);
                let index_reg = self.builder.alloc_register();
                self.builder.emit_load_const(index_reg, index_const);

                let attr_name_reg = self.builder.alloc_register();
                self.builder
                    .emit_get_item(attr_name_reg, match_args_reg, index_reg);
                self.builder.free_register(index_reg);

                let attr_value_reg = self.builder.alloc_register();
                self.emit_named_call_from_regs(
                    "getattr",
                    &[subject_reg, attr_name_reg],
                    attr_value_reg,
                )?;
                self.builder.free_register(attr_name_reg);

                self.compile_pattern_match(sub_pattern, attr_value_reg, fail_label)?;
                self.builder.free_register(attr_value_reg);
            }

            self.builder.free_register(match_args_reg);
        }

        for (attr, sub_pattern) in kwd_attrs.iter().zip(kwd_patterns.iter()) {
            let attr_reg = self.builder.alloc_register();
            let attr_name_idx = self.builder.add_name(Arc::<str>::from(attr.as_str()));
            self.builder
                .emit_get_attr(attr_reg, subject_reg, attr_name_idx);
            self.compile_pattern_match(sub_pattern, attr_reg, fail_label)?;
            self.builder.free_register(attr_reg);
        }

        self.builder.free_register(class_reg);
        Ok(())
    }

    // =========================================================================
    // Function Definition Compilation
    // =========================================================================

    /// Compile positional default expressions into a tuple register.
    fn compile_positional_defaults_tuple(
        &mut self,
        defaults: &[Expr],
    ) -> CompileResult<Option<Register>> {
        if defaults.is_empty() {
            return Ok(None);
        }
        if defaults.len() > u8::MAX as usize {
            return Err(CompileError {
                message: "too many positional defaults".to_string(),
                line: self.line_for_span(defaults[0].span),
                column: 0,
            });
        }

        let count = defaults.len() as u8;
        let first = self.builder.alloc_register_block(count);
        for (i, expr) in defaults.iter().enumerate() {
            let dst = Register::new(first.0 + i as u8);
            let tmp = self.compile_expr(expr)?;
            if tmp != dst {
                self.builder.emit_move(dst, tmp);
            }
            self.builder.free_register(tmp);
        }

        let tuple_reg = self.builder.alloc_register();
        self.builder.emit_build_tuple(tuple_reg, first, count);
        self.builder.free_register_block(first, count);
        Ok(Some(tuple_reg))
    }

    /// Compile keyword-only defaults into a dict register (name -> default value).
    fn compile_kw_defaults_dict(
        &mut self,
        kwonlyargs: &[prism_parser::ast::Arg],
        kw_defaults: &[Option<Expr>],
    ) -> CompileResult<Option<Register>> {
        if kwonlyargs.len() != kw_defaults.len() {
            return Err(CompileError {
                message: "internal error: kwonly args/defaults length mismatch".to_string(),
                line: 0,
                column: 0,
            });
        }

        let mut entries: Vec<(&str, &Expr)> = Vec::new();
        for (arg, default_expr) in kwonlyargs.iter().zip(kw_defaults.iter()) {
            if let Some(expr) = default_expr {
                entries.push((arg.arg.as_str(), expr));
            }
        }

        if entries.is_empty() {
            return Ok(None);
        }
        if entries.len() > (u8::MAX as usize / 2) {
            return Err(CompileError {
                message: "too many keyword-only defaults".to_string(),
                line: self.line_for_span(entries[0].1.span),
                column: 0,
            });
        }

        let pair_count = entries.len() as u8;
        let pair_regs = pair_count
            .checked_mul(2)
            .expect("pair_count bounded by u8::MAX/2");
        let first_pair = self.builder.alloc_register_block(pair_regs);

        for (i, (name, value_expr)) in entries.iter().enumerate() {
            let key_reg = Register::new(first_pair.0 + (i as u8 * 2));
            let value_reg = Register::new(key_reg.0 + 1);

            let key_idx = self.builder.add_string(*name);
            self.builder.emit_load_const(key_reg, key_idx);

            let value_tmp = self.compile_expr(value_expr)?;
            if value_tmp != value_reg {
                self.builder.emit_move(value_reg, value_tmp);
            }
            self.builder.free_register(value_tmp);
        }

        let dict_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::new(
            Opcode::BuildDict,
            dict_reg.0,
            first_pair.0,
            pair_count,
        ));
        self.builder.free_register_block(first_pair, pair_regs);
        Ok(Some(dict_reg))
    }

    /// Compile a function definition (FunctionDef or AsyncFunctionDef).
    ///
    /// This creates a nested CodeObject for the function body and emits
    /// MakeFunction or MakeClosure opcode to create the function object.
    ///
    /// # Arguments
    ///
    /// * `name` - Function name
    /// * `args` - Function arguments specification
    /// * `body` - Function body statements
    /// * `decorator_list` - Decorators to apply
    /// * `is_async` - Whether this is an async function
    fn compile_function_def(
        &mut self,
        name: &str,
        args: &prism_parser::ast::Arguments,
        body: &[Stmt],
        decorator_list: &[Expr],
        is_async: bool,
        definition_line: u32,
    ) -> CompileResult<()> {
        // Find the scope for this function from the symbol table
        // We need to look it up by name in the current scope's children
        let func_scope_idx = self.find_child_scope(ScopeKind::Function, name);
        let (func_cellvars, func_freevars, func_locals, scope_has_yield) =
            if let Some(scope_idx) = func_scope_idx {
                let scope = &self.current_scope().children[scope_idx];
                let cellvars = Self::ordered_cellvar_names(scope);
                let freevars = Self::ordered_freevar_names(scope);
                let mut locals = scope
                    .locals()
                    .map(|sym| Arc::from(sym.name.as_ref()))
                    .collect::<Vec<Arc<str>>>();
                locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
                (cellvars, freevars, locals, scope.has_yield)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), false)
            };

        // Create a new FunctionBuilder for the function body
        let mut func_builder = FunctionBuilder::new(name);
        func_builder.set_filename(&*self.filename);
        func_builder.set_first_lineno(definition_line);

        // Set function flags
        if is_async {
            func_builder.add_flags(CodeFlags::COROUTINE);
        }

        // Count parameters
        let posonly_count = args.posonlyargs.len() as u16;
        let kwonly_count = args.kwonlyargs.len() as u16;
        let total_positional = args.posonlyargs.len() + args.args.len();

        // Set parameter counts on the builder
        func_builder.set_arg_count(total_positional as u16);
        func_builder.set_kwonlyarg_count(kwonly_count);
        func_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            func_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            func_builder.add_flags(CodeFlags::VARKEYWORDS);
        }

        // Register parameters as locals (they occupy the first slots)
        // Python parameter order: posonly, regular args, vararg, kwonly, kwarg

        // Position-only parameters
        for arg in &args.posonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // Regular positional parameters
        for arg in &args.args {
            func_builder.define_local(arg.arg.as_str());
        }

        // *args
        if let Some(ref vararg) = args.vararg {
            func_builder.define_local(vararg.arg.as_str());
        }

        // Keyword-only parameters
        for arg in &args.kwonlyargs {
            func_builder.define_local(arg.arg.as_str());
        }

        // **kwargs
        if let Some(ref kwarg) = args.kwarg {
            func_builder.define_local(kwarg.arg.as_str());
        }

        // Register non-cell locals from scope analysis so nested captures can
        // reliably resolve by name from `code.locals`.
        for name in func_locals {
            func_builder.define_local(name);
        }

        // Register cell and free variables from scope analysis.
        // Cellvars are materialized per invocation, while freevars require
        // definition-time capture from the enclosing scope.
        let has_cellvars = !func_cellvars.is_empty();
        let captures_freevars = !func_freevars.is_empty();
        // Cell variables: locals captured by inner functions
        for name in func_cellvars {
            func_builder.add_cellvar(name);
        }

        // Free variables: captured from outer scopes
        for name in func_freevars {
            func_builder.add_freevar(name);
        }

        if has_cellvars {
            func_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            func_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Set generator flag from scope analysis
        if scope_has_yield {
            if is_async {
                // async def with yield = async generator
                func_builder.add_flags(CodeFlags::ASYNC_GENERATOR);
            } else {
                // regular generator
                func_builder.add_flags(CodeFlags::GENERATOR);
            }
        }

        if has_cellvars || captures_freevars {
            func_builder.add_flags(CodeFlags::NESTED);
        }

        // Swap builders to compile function body
        let parent_builder = std::mem::replace(&mut self.builder, func_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);

        // Save and update context for function body compilation
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        self.in_async_context = is_async;
        self.in_function_context = true;
        if let Some(scope_idx) = func_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Compile function body
        for (index, stmt) in body.iter().enumerate() {
            if self.should_strip_docstring_stmt(index, stmt) {
                continue;
            }
            self.compile_stmt(stmt)?;
        }

        if func_scope_idx.is_some() {
            self.exit_child_scope();
        }

        // Ensure function returns None if no explicit return
        self.builder.emit_return_none();

        // Restore contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;
        self.finally_stack = parent_finally_stack;

        // Swap back and get finished function code
        let func_builder = std::mem::replace(&mut self.builder, parent_builder);
        let func_code = func_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(func_code));

        // Compile decorators in reverse order (they'll wrap the function)
        // Decorators are compiled first, then applied after function creation
        let decorator_regs: Vec<Register> = decorator_list
            .iter()
            .map(|d| self.compile_expr(d))
            .collect::<Result<_, _>>()?;

        let positional_defaults_reg = self.compile_positional_defaults_tuple(&args.defaults)?;
        let kw_defaults_reg = self.compile_kw_defaults_dict(&args.kwonlyargs, &args.kw_defaults)?;

        // Emit function/closure creation
        let func_reg = self.builder.alloc_register();

        if captures_freevars {
            // MakeClosure is only needed when freevars must be captured from
            // the enclosing scope. Cellvars are created fresh for each call.
            self.builder.emit(Instruction::op_di(
                Opcode::MakeClosure,
                func_reg,
                code_const_idx,
            ));
        } else {
            // MakeFunction: simple function without captures
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                func_reg,
                code_const_idx,
            ));
        }

        if positional_defaults_reg.is_some() || kw_defaults_reg.is_some() {
            let none_reg = if positional_defaults_reg.is_none() || kw_defaults_reg.is_none() {
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                Some(reg)
            } else {
                None
            };
            let positional_reg = positional_defaults_reg
                .or(none_reg)
                .expect("positional defaults register must exist");
            let kw_reg = kw_defaults_reg
                .or(none_reg)
                .expect("keyword defaults register must exist");
            self.builder
                .emit_set_function_defaults(func_reg, positional_reg, kw_reg);

            if let Some(reg) = positional_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = kw_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = none_reg {
                self.builder.free_register(reg);
            }
        }

        // Apply decorators in reverse order
        // @decorator1
        // @decorator2
        // def func(): ...
        // is equivalent to: func = decorator1(decorator2(func))
        for decorator_reg in decorator_regs.into_iter().rev() {
            // Decorator calls need a dedicated contiguous block [result, arg0]
            // so the argument register cannot alias any live temporary reused
            // by the allocator between decorated definitions.
            let call_block = self.builder.alloc_register_block(2);
            self.builder
                .emit_move(Register::new(call_block.0 + 1), func_reg);
            self.builder.emit(Instruction::op_dss(
                Opcode::Call,
                call_block,
                decorator_reg,
                Register::new(1), // 1 argument
            ));
            self.builder.emit_move(func_reg, call_block);
            self.builder.free_register_block(call_block, 2);
            self.builder.free_register(decorator_reg);
        }

        // Store function using lexical scope resolution.
        let location = self.resolve_variable(name);
        self.builder.emit_store_var(location, func_reg, Some(name));
        self.builder.free_register(func_reg);

        Ok(())
    }

    /// Find a child scope by kind and name in the current scope.
    ///
    /// Uses a per-scope cursor so repeated nested definitions with the same
    /// name (e.g. redefinitions) resolve deterministically in source order.
    fn find_child_scope(&mut self, kind: ScopeKind, name: &str) -> Option<usize> {
        let depth = self.scope_path.len();
        let start = *self.scope_child_offsets.get(depth).unwrap_or(&0);
        let child_count = self.current_scope().children.len();

        let mut found = None;
        for idx in start..child_count {
            let child = &self.current_scope().children[idx];
            if child.kind == kind && child.name.as_ref() == name {
                found = Some(idx);
                break;
            }
        }

        if found.is_none() {
            for idx in 0..start.min(child_count) {
                let child = &self.current_scope().children[idx];
                if child.kind == kind && child.name.as_ref() == name {
                    found = Some(idx);
                    break;
                }
            }
        }

        if let Some(idx) = found {
            if let Some(offset) = self.scope_child_offsets.get_mut(depth) {
                *offset = idx + 1;
            }
            Some(idx)
        } else {
            None
        }
    }

    /// Advance to the next comprehension child scope in source order.
    ///
    /// Comprehensions participate in the same child-scope cursor as functions,
    /// lambdas, and classes. Even comprehensions that are currently compiled
    /// inline must still consume their analyzed scope entry so later sibling
    /// scopes remain aligned with the AST traversal order.
    fn next_comprehension_scope(&mut self, name: &str) -> Option<usize> {
        self.find_child_scope(ScopeKind::Comprehension, name)
    }

    // =========================================================================
    // Lambda Expression Compilation
    // =========================================================================

    /// Compile a lambda expression.
    ///
    /// Lambda expressions create nested code objects like functions, but with:
    /// - Single expression body (not statements)
    /// - Implicit return of expression result
    /// - Anonymous name (`<lambda>`)
    /// - Inherits async context from enclosing scope
    ///
    /// # Performance Optimizations
    /// - Uses register-based evaluation for body expression
    /// - Direct return without intermediate storage
    /// - Closure handling only when capturing variables
    fn compile_lambda(
        &mut self,
        args: &prism_parser::ast::Arguments,
        body: &Expr,
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        // Find lambda scope from symbol table (lambdas are named "<lambda>" in scope analysis)
        let lambda_scope_idx = self.find_child_scope(ScopeKind::Lambda, "<lambda>");
        let (lambda_cellvars, lambda_freevars, lambda_locals, lambda_has_yield) =
            if let Some(scope_idx) = lambda_scope_idx {
                let scope = &self.current_scope().children[scope_idx];
                let cellvars = Self::ordered_cellvar_names(scope);
                let freevars = Self::ordered_freevar_names(scope);
                let mut locals = scope
                    .locals()
                    .map(|sym| Arc::from(sym.name.as_ref()))
                    .collect::<Vec<Arc<str>>>();
                locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
                (cellvars, freevars, locals, scope.has_yield)
            } else {
                (Vec::new(), Vec::new(), Vec::new(), false)
            };

        // Create a new FunctionBuilder for the lambda body
        let mut lambda_builder = FunctionBuilder::new("<lambda>");
        lambda_builder.set_filename(&*self.filename);
        lambda_builder.set_first_lineno(definition_line);

        // Calculate argument counts
        let posonly_count = args.posonlyargs.len() as u16;
        let regular_args = args.args.len() as u16;
        let total_positional = posonly_count + regular_args;
        let kwonly_count = args.kwonlyargs.len() as u16;

        // Set parameter counts
        lambda_builder.set_arg_count(total_positional);
        lambda_builder.set_kwonlyarg_count(kwonly_count);
        lambda_builder.set_posonlyarg_count(posonly_count);

        // Handle varargs and kwargs
        if args.vararg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARARGS);
        }
        if args.kwarg.is_some() {
            lambda_builder.add_flags(CodeFlags::VARKEYWORDS);
        }
        if lambda_has_yield {
            lambda_builder.add_flags(CodeFlags::GENERATOR);
        }

        // Register parameters as locals
        for arg in &args.posonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        for arg in &args.args {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref vararg) = args.vararg {
            lambda_builder.define_local(vararg.arg.as_str());
        }
        for arg in &args.kwonlyargs {
            lambda_builder.define_local(arg.arg.as_str());
        }
        if let Some(ref kwarg) = args.kwarg {
            lambda_builder.define_local(kwarg.arg.as_str());
        }

        for name in lambda_locals {
            lambda_builder.define_local(name);
        }

        // Register cell and free variables from scope analysis.
        // Cellvars are invocation-local; only freevars require MakeClosure.
        let has_cellvars = !lambda_cellvars.is_empty();
        let captures_freevars = !lambda_freevars.is_empty();
        for name in lambda_cellvars {
            lambda_builder.add_cellvar(name);
        }
        for name in lambda_freevars {
            lambda_builder.add_freevar(name);
        }

        if has_cellvars {
            lambda_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            lambda_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Swap builders to compile lambda body
        let parent_builder = std::mem::replace(&mut self.builder, lambda_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        let parent_async_context = self.in_async_context;
        let parent_function_context = self.in_function_context;
        // Lambda inherits async context from enclosing scope but sets function context
        self.in_function_context = true;
        if let Some(scope_idx) = lambda_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Compile the expression body
        let result_reg = self.compile_expr(body)?;

        if lambda_scope_idx.is_some() {
            self.exit_child_scope();
        }

        // Emit implicit return of the expression result
        self.builder.emit_return(result_reg);

        // Restore parent contexts
        self.in_async_context = parent_async_context;
        self.in_function_context = parent_function_context;
        self.finally_stack = parent_finally_stack;

        // Swap back and get finished lambda code
        let lambda_builder = std::mem::replace(&mut self.builder, parent_builder);
        let lambda_code = lambda_builder.finish();

        // Store the nested CodeObject as a constant
        let code_const_idx = self.builder.add_code_object(Arc::new(lambda_code));

        let positional_defaults_reg = self.compile_positional_defaults_tuple(&args.defaults)?;
        let kw_defaults_reg = self.compile_kw_defaults_dict(&args.kwonlyargs, &args.kw_defaults)?;

        // Emit function/closure creation
        if captures_freevars {
            self.builder
                .emit(Instruction::op_di(Opcode::MakeClosure, dst, code_const_idx));
        } else {
            self.builder.emit(Instruction::op_di(
                Opcode::MakeFunction,
                dst,
                code_const_idx,
            ));
        }

        if positional_defaults_reg.is_some() || kw_defaults_reg.is_some() {
            let none_reg = if positional_defaults_reg.is_none() || kw_defaults_reg.is_none() {
                let reg = self.builder.alloc_register();
                self.builder.emit_load_none(reg);
                Some(reg)
            } else {
                None
            };
            let positional_reg = positional_defaults_reg
                .or(none_reg)
                .expect("positional defaults register must exist");
            let kw_reg = kw_defaults_reg
                .or(none_reg)
                .expect("keyword defaults register must exist");
            self.builder
                .emit_set_function_defaults(dst, positional_reg, kw_reg);

            if let Some(reg) = positional_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = kw_defaults_reg {
                self.builder.free_register(reg);
            }
            if let Some(reg) = none_reg {
                self.builder.free_register(reg);
            }
        }

        Ok(dst)
    }

    // =========================================================================
    // Comprehension Expression Compilation
    // =========================================================================

    /// Compile a list comprehension.
    ///
    /// List comprehensions create a nested scope (as a hidden function) to prevent
    /// loop variables from leaking into the enclosing scope. This matches Python 3
    /// semantics.
    ///
    /// # Bytecode Strategy
    /// 1. Create a hidden function containing the comprehension logic
    /// 2. Inside: create empty list, iterate with FOR_ITER, append elements
    /// 3. Call the hidden function with the first iterator
    /// 4. Result is the completed list
    ///
    /// # Performance Optimizations
    /// - Uses LIST_APPEND opcode for O(1) amortized append
    /// - Inlines filter conditions to avoid function call overhead
    /// - Pre-allocates result register for minimal register pressure
    fn compile_listcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        self.compile_sequence_comprehension(
            "<listcomp>",
            "<comprehension>",
            elt,
            generators,
            dst,
            ComprehensionKind::List,
            definition_line,
        )
    }

    /// Compile a set comprehension.
    fn compile_setcomp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        self.compile_sequence_comprehension(
            "<setcomp>",
            "<comprehension>",
            elt,
            generators,
            dst,
            ComprehensionKind::Set,
            definition_line,
        )
    }

    /// Compile a dict comprehension.
    fn compile_dictcomp(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let comp_scope_idx = self.next_comprehension_scope("<dictcomp>");
        let (cellvars, freevars, locals) = self.comprehension_scope_layout(comp_scope_idx);
        let captures_freevars = !freevars.is_empty();

        let mut comp_builder = FunctionBuilder::new("<dictcomp>");
        comp_builder.set_filename(&*self.filename);
        comp_builder.set_first_lineno(definition_line);
        comp_builder.set_arg_count(1);
        comp_builder.define_local(".0");
        for name in locals {
            comp_builder.define_local(name);
        }
        Self::configure_closure_layout(&mut comp_builder, cellvars, freevars);

        let parent_builder = std::mem::replace(&mut self.builder, comp_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = comp_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        let dict_reg = self.builder.alloc_register();
        self.builder
            .emit(Instruction::op_d(Opcode::BuildDict, dict_reg));
        self.compile_dict_comprehension_generators(key, value, generators, dict_reg, 0, iter_reg)?;
        self.builder.emit_return(dict_reg);
        self.builder.free_register(dict_reg);
        self.builder.free_register(iter_reg);

        if comp_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        let comp_builder = std::mem::replace(&mut self.builder, parent_builder);
        let comp_code = comp_builder.finish();
        self.emit_comprehension_call(
            comp_code,
            captures_freevars,
            &generators[0].iter,
            generators[0].is_async,
            dst,
        )
    }

    fn compile_sequence_comprehension(
        &mut self,
        code_name: &'static str,
        scope_name: &'static str,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        kind: ComprehensionKind,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let comp_scope_idx = self.next_comprehension_scope(scope_name);
        let (cellvars, freevars, locals) = self.comprehension_scope_layout(comp_scope_idx);
        let captures_freevars = !freevars.is_empty();

        let mut comp_builder = FunctionBuilder::new(code_name);
        comp_builder.set_filename(&*self.filename);
        comp_builder.set_first_lineno(definition_line);
        comp_builder.set_arg_count(1);
        comp_builder.define_local(".0");
        for name in locals {
            comp_builder.define_local(name);
        }
        Self::configure_closure_layout(&mut comp_builder, cellvars, freevars);

        let parent_builder = std::mem::replace(&mut self.builder, comp_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = comp_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        let result_reg = self.builder.alloc_register();
        match kind {
            ComprehensionKind::List => self.builder.emit_build_list(result_reg, result_reg, 0),
            ComprehensionKind::Set => self
                .builder
                .emit(Instruction::op_d(Opcode::BuildSet, result_reg)),
        }
        self.compile_comprehension_generators(elt, generators, result_reg, kind, 0, iter_reg)?;
        self.builder.emit_return(result_reg);
        self.builder.free_register(result_reg);
        self.builder.free_register(iter_reg);

        if comp_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        let comp_builder = std::mem::replace(&mut self.builder, parent_builder);
        let comp_code = comp_builder.finish();
        self.emit_comprehension_call(
            comp_code,
            captures_freevars,
            &generators[0].iter,
            generators[0].is_async,
            dst,
        )
    }

    fn comprehension_scope_layout(
        &self,
        scope_idx: Option<usize>,
    ) -> (Vec<Arc<str>>, Vec<Arc<str>>, Vec<Arc<str>>) {
        if let Some(scope_idx) = scope_idx {
            let scope = &self.current_scope().children[scope_idx];
            let cellvars = Self::ordered_cellvar_names(scope);
            let freevars = Self::ordered_freevar_names(scope);
            let mut locals = scope
                .locals()
                .map(|sym| Arc::from(sym.name.as_ref()))
                .collect::<Vec<Arc<str>>>();
            locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
            (cellvars, freevars, locals)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        }
    }

    fn configure_closure_layout(
        builder: &mut FunctionBuilder,
        cellvars: Vec<Arc<str>>,
        freevars: Vec<Arc<str>>,
    ) {
        let has_cellvars = !cellvars.is_empty();
        let captures_freevars = !freevars.is_empty();

        for name in cellvars {
            builder.add_cellvar(name);
        }
        for name in freevars {
            builder.add_freevar(name);
        }

        if has_cellvars {
            builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            builder.add_flags(CodeFlags::HAS_FREEVARS);
        }
    }

    fn emit_comprehension_call(
        &mut self,
        comp_code: CodeObject,
        captures_freevars: bool,
        first_iter_expr: &Expr,
        first_iter_is_async: bool,
        dst: Register,
    ) -> CompileResult<Register> {
        let code_idx = self.builder.add_code_object(Arc::new(comp_code));
        let mut func_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_di(
            if captures_freevars {
                Opcode::MakeClosure
            } else {
                Opcode::MakeFunction
            },
            func_reg,
            code_idx,
        ));

        let call_block = self.builder.alloc_register_block(2);
        if func_reg.0 >= call_block.0 && func_reg.0 < call_block.0 + 2 {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, func_reg);
            self.builder.free_register(func_reg);
            func_reg = safe_reg;
        }

        let first_iter = self.compile_expr(first_iter_expr)?;
        let arg_reg = Register::new(call_block.0 + 1);
        if first_iter_is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAIter, arg_reg, first_iter));
        } else {
            self.builder.emit_get_iter(arg_reg, first_iter);
        }
        self.builder.free_register(first_iter);

        self.builder.emit_call(call_block, func_reg, 1);
        if call_block != dst {
            self.builder.emit_move(dst, call_block);
        }
        self.builder.free_register(func_reg);
        self.builder.free_register_block(call_block, 2);

        Ok(dst)
    }

    /// Compile a generator expression.
    ///
    /// Generator expressions are lazy - they create a generator function that
    /// yields values on demand. This is more memory efficient for large sequences.
    fn compile_genexp(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        dst: Register,
        definition_line: u32,
    ) -> CompileResult<Register> {
        let gen_scope_idx = self.next_comprehension_scope("<comprehension>");
        let (gen_cellvars, gen_freevars, gen_locals) = if let Some(scope_idx) = gen_scope_idx {
            let scope = &self.current_scope().children[scope_idx];
            let cellvars = Self::ordered_cellvar_names(scope);
            let freevars = Self::ordered_freevar_names(scope);
            let mut locals = scope
                .locals()
                .map(|sym| Arc::from(sym.name.as_ref()))
                .collect::<Vec<Arc<str>>>();
            locals.sort_unstable_by(|a, b| a.as_ref().cmp(b.as_ref()));
            (cellvars, freevars, locals)
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        // Create a generator function that yields each element
        let mut gen_builder = FunctionBuilder::new("<genexpr>");
        gen_builder.set_filename(&*self.filename);
        gen_builder.set_first_lineno(definition_line);
        gen_builder.add_flags(CodeFlags::GENERATOR);

        // First iterator is passed as argument
        gen_builder.set_arg_count(1);
        gen_builder.define_local(".0"); // Hidden argument for first iterator
        for name in gen_locals {
            gen_builder.define_local(name);
        }

        let has_cellvars = !gen_cellvars.is_empty();
        let captures_freevars = !gen_freevars.is_empty();
        for name in gen_cellvars {
            gen_builder.add_cellvar(name);
        }
        for name in gen_freevars {
            gen_builder.add_freevar(name);
        }

        if has_cellvars {
            gen_builder.add_flags(CodeFlags::HAS_CELLVARS);
        }
        if captures_freevars {
            gen_builder.add_flags(CodeFlags::HAS_FREEVARS);
        }

        // Swap builders
        let parent_builder = std::mem::replace(&mut self.builder, gen_builder);
        let parent_finally_stack = std::mem::take(&mut self.finally_stack);
        if let Some(scope_idx) = gen_scope_idx {
            self.enter_child_scope(scope_idx);
        }

        // Get the first iterator from argument
        let iter_reg = self.builder.alloc_register();
        self.builder
            .emit_load_local(iter_reg, crate::bytecode::LocalSlot::new(0));

        // Compile generator loops (yields instead of appending)
        self.compile_genexp_generators(elt, generators, 0, iter_reg)?;
        self.builder.free_register(iter_reg);

        // Return None at end
        self.builder.emit_return_none();

        if gen_scope_idx.is_some() {
            self.exit_child_scope();
        }

        self.finally_stack = parent_finally_stack;
        // Swap back
        let gen_builder = std::mem::replace(&mut self.builder, parent_builder);
        let gen_code = gen_builder.finish();

        // Store code object and create function
        let code_idx = self.builder.add_code_object(Arc::new(gen_code));
        let mut func_reg = self.builder.alloc_register();
        self.builder.emit(Instruction::op_di(
            if captures_freevars {
                Opcode::MakeClosure
            } else {
                Opcode::MakeFunction
            },
            func_reg,
            code_idx,
        ));

        // Reserve a fresh contiguous block [result, arg0] so the iterator
        // argument cannot overwrite the callable register before the call.
        let call_block = self.builder.alloc_register_block(2);
        if func_reg.0 >= call_block.0 && func_reg.0 < call_block.0 + 2 {
            let safe_reg = self.builder.alloc_register();
            self.builder.emit_move(safe_reg, func_reg);
            self.builder.free_register(func_reg);
            func_reg = safe_reg;
        }

        // Compile first iterator and pass it in the dedicated arg slot.
        let first_iter = self.compile_expr(&generators[0].iter)?;
        let arg_reg = Register::new(call_block.0 + 1);
        if generators[0].is_async {
            if !self.in_async_context {
                return Err(CompileError {
                    message: "asynchronous comprehension outside of an async function".to_string(),
                    line: 0,
                    column: 0,
                });
            }
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAIter, arg_reg, first_iter));
        } else {
            self.builder.emit_get_iter(arg_reg, first_iter);
        }
        self.builder.free_register(first_iter);

        self.builder.emit_call(call_block, func_reg, 1);
        if call_block != dst {
            self.builder.emit_move(dst, call_block);
        }
        self.builder.free_register(func_reg);
        self.builder.free_register_block(call_block, 2);

        Ok(dst)
    }

    /// Helper to compile comprehension generators (list/set).
    fn compile_comprehension_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
        kind: ComprehensionKind,
        depth: usize,
        first_iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute element and add to collection
            let elem_reg = self.compile_expr(elt)?;
            match kind {
                ComprehensionKind::List => {
                    // ListAppend: src1.append(src2) - list in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::ListAppend,
                        Register(0), // dst unused for ListAppend
                        result_reg,  // src1 = list
                        elem_reg,    // src2 = element
                    ));
                }
                ComprehensionKind::Set => {
                    // SetAdd: src1.add(src2) - set in src1, element in src2
                    self.builder.emit(Instruction::op_dss(
                        Opcode::SetAdd,
                        Register(0), // dst unused for SetAdd
                        result_reg,  // src1 = set
                        elem_reg,    // src2 = element
                    ));
                }
            }
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        let loop_regs = self.builder.alloc_register_block(2);
        let iter_reg = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);

        if depth == 0 {
            self.builder.emit_move(iter_reg, first_iter_reg);
        } else {
            // Compile iterator
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;

            // Get iterator (sync or async)
            if comp_gen.is_async {
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "asynchronous comprehension outside of an async function"
                            .to_string(),
                        line: 0,
                        column: 0,
                    });
                }
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAIter,
                    iter_reg,
                    iter_expr_reg,
                ));
            } else {
                self.builder.emit_get_iter(iter_reg, iter_expr_reg);
            }
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            // await the result
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.emit_yield_from(item_reg, item_reg);
        } else {
            self.builder.emit_for_iter(item_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators or emit element
        self.compile_comprehension_generators(
            elt,
            rest,
            result_reg,
            kind,
            depth + 1,
            Register::new(0),
        )?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile dict comprehension generators.
    fn compile_dict_comprehension_generators(
        &mut self,
        key: &Expr,
        value: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        result_reg: Register,
        depth: usize,
        first_iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: compute key-value and add to dict
            let key_reg = self.compile_expr(key)?;
            let val_reg = self.compile_expr(value)?;
            self.builder.emit_set_item(result_reg, key_reg, val_reg);
            self.builder.free_register(key_reg);
            self.builder.free_register(val_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        let loop_regs = self.builder.alloc_register_block(2);
        let iter_reg = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);

        if depth == 0 {
            self.builder.emit_move(iter_reg, first_iter_reg);
        } else {
            // Compile iterator
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;

            if comp_gen.is_async {
                if !self.in_async_context {
                    return Err(CompileError {
                        message: "asynchronous comprehension outside of an async function"
                            .to_string(),
                        line: 0,
                        column: 0,
                    });
                }
                self.builder.emit(Instruction::op_ds(
                    Opcode::GetAIter,
                    iter_reg,
                    iter_expr_reg,
                ));
            } else {
                self.builder.emit_get_iter(iter_reg, iter_expr_reg);
            }
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        if comp_gen.is_async {
            self.builder
                .emit(Instruction::op_ds(Opcode::GetANext, item_reg, iter_reg));
            self.builder
                .emit(Instruction::op_ds(Opcode::GetAwaitable, item_reg, item_reg));
            self.emit_yield_from(item_reg, item_reg);
        } else {
            self.builder.emit_for_iter(item_reg, loop_end);
        }

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators
        self.compile_dict_comprehension_generators(
            key,
            value,
            rest,
            result_reg,
            depth + 1,
            Register::new(0),
        )?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(iter_reg);
        self.builder.free_register(item_reg);

        Ok(())
    }

    /// Helper to compile generator expression generators.
    fn compile_genexp_generators(
        &mut self,
        elt: &Expr,
        generators: &[prism_parser::ast::Comprehension],
        depth: usize,
        iter_reg: Register,
    ) -> CompileResult<()> {
        if generators.is_empty() {
            // Base case: yield element
            let elem_reg = self.compile_expr(elt)?;
            let yield_result = self.builder.alloc_register();
            self.builder
                .emit(Instruction::op_ds(Opcode::Yield, yield_result, elem_reg));
            self.builder.free_register(yield_result);
            self.builder.free_register(elem_reg);
            return Ok(());
        }

        let comp_gen = &generators[0];
        let rest = &generators[1..];

        // For depth > 0, compile iterator; depth 0 uses passed-in iter_reg
        let loop_regs = self.builder.alloc_register_block(2);
        let actual_iter = loop_regs;
        let item_reg = Register::new(loop_regs.0 + 1);
        if depth == 0 {
            self.builder.emit_move(actual_iter, iter_reg);
        } else {
            let iter_expr_reg = self.compile_expr(&comp_gen.iter)?;
            self.builder.emit_get_iter(actual_iter, iter_expr_reg);
            self.builder.free_register(iter_expr_reg);
        }

        // Create loop labels
        let loop_start = self.builder.create_label();
        let loop_end = self.builder.create_label();

        self.builder.bind_label(loop_start);

        // Get next item
        self.builder.emit_for_iter(item_reg, loop_end);

        // Unpack target
        self.compile_store(&comp_gen.target, item_reg)?;

        // Compile filter conditions
        for if_expr in &comp_gen.ifs {
            let cond_reg = self.compile_expr(if_expr)?;
            self.builder.emit_jump_if_false(cond_reg, loop_start);
            self.builder.free_register(cond_reg);
        }

        // Recurse for nested generators.
        // Deeper recursion compiles each nested iterator expression in the
        // current loop scope, so only the root generator consumes `iter_reg`.
        self.compile_genexp_generators(elt, rest, depth + 1, Register::new(0))?;

        // Jump back to loop start
        self.builder.emit_jump(loop_start);

        self.builder.bind_label(loop_end);
        self.builder.free_register(item_reg);
        self.builder.free_register(actual_iter);

        Ok(())
    }
}

/// Kind of comprehension being compiled.
#[derive(Debug, Clone, Copy)]
enum ComprehensionKind {
    List,
    Set,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use prism_code::Constant;

    fn compile(source: &str) -> CodeObject {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module(&module, "<test>").expect("compile error")
    }

    fn compile_with_dynamic_locals(source: &str) -> CodeObject {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module_with_namespace_mode(
            &module,
            "<test>",
            OptimizationLevel::None,
            ModuleNamespaceMode::DynamicLocals,
        )
        .expect("compile error")
    }

    fn large_call_then_functools_style_listcomp_source() -> String {
        let large_arg_list = (0..250)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "def helper(*args, **kwargs):\n    return args\n\n\
             def stressed(seq, abcs):\n    helper({large_arg_list})\n    return [helper(base, abcs=abcs) for base in seq]\n"
        )
    }

    fn large_call_then_class_definition_source() -> String {
        let large_arg_list = (0..250)
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        format!(
            "def helper(*args, **kwargs):\n    return object\n\n\
             def stressed():\n    helper({large_arg_list})\n    class Derived(helper()):\n        pass\n    return Derived\n"
        )
    }

    fn compile_with_optimization(source: &str, optimize: OptimizationLevel) -> CodeObject {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module_with_optimization(&module, "<test>", optimize)
            .expect("compile error")
    }

    fn try_compile(source: &str) -> Result<CodeObject, CompileError> {
        let module = prism_parser::parse(source).expect("parse error");
        Compiler::compile_module(&module, "<test>")
    }

    #[test]
    fn test_compile_wide_i64_literal_emits_integer_constant() {
        let code = compile("value = 2305843009213693952");

        assert!(code.constants.iter().any(|value| {
            matches!(
                value,
                Constant::BigInt(constant)
                    if constant == &BigInt::from(2_305_843_009_213_693_952_i64)
            )
        }));
    }

    #[test]
    fn test_compile_bigint_literal_emits_arbitrary_precision_constant() {
        let expected = BigInt::from(1_u8) << 100_u32;
        let code = compile("value = 1267650600228229401496703205376");

        assert!(
            code.constants
                .iter()
                .any(|value| matches!(value, Constant::BigInt(constant) if constant == &expected))
        );
    }

    #[test]
    fn test_compile_with_uses_call_method_for_context_manager_protocol() {
        let code = compile(
            r#"
with manager:
    pass
"#,
        );

        let load_method_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadMethod as u8)
            .count();
        let call_method_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::CallMethod as u8)
            .count();

        assert_eq!(
            load_method_count, 2,
            "with should load __enter__ and __exit__"
        );
        assert_eq!(
            call_method_count, 3,
            "with should call __enter__ plus both normal/exception __exit__ paths via CallMethod"
        );
        assert!(
            !code
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::Call as u8),
            "with should not use generic Call for context-manager methods"
        );
    }

    #[test]
    fn test_compile_async_with_uses_call_method_for_context_manager_protocol() {
        let code = compile(
            r#"
async def run():
    async with manager:
        pass
"#,
        );

        let async_fn = code
            .nested_code_objects
            .first()
            .expect("expected nested async function");
        let load_method_count = async_fn
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadMethod as u8)
            .count();
        let call_method_count = async_fn
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::CallMethod as u8)
            .count();

        assert_eq!(
            load_method_count, 2,
            "async with should load __aenter__ and __aexit__"
        );
        assert_eq!(
            call_method_count, 3,
            "async with should call __aenter__ plus both normal/exception __aexit__ paths via CallMethod"
        );
        assert!(
            !async_fn
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::Call as u8),
            "async with should not use generic Call for context-manager methods"
        );
    }

    #[test]
    fn test_compile_simple_expr() {
        let code = compile("1 + 2");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_assignment() {
        let code = compile("x = 42");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_attribute_assignment_emits_set_attr() {
        let code = compile(
            r#"
class Holder:
    pass

def configure(obj, value):
    obj.answer = value
"#,
        );
        let configure = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "configure")
            .expect("expected nested configure function");

        assert!(
            configure
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::SetAttr as u8),
            "attribute assignment should lower to SetAttr"
        );
    }

    #[test]
    fn test_compile_if() {
        let code = compile("if True:\n    pass");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_while() {
        let code = compile("x = 0\nwhile x < 10:\n    x = x + 1");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_function_call() {
        let code = compile("print(42)");
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_redefined_functions_preserve_distinct_nested_code_objects() {
        let code = compile(
            r#"
def value(self):
    return self

def value(self, new_value):
    return new_value
"#,
        );

        let value_defs = code
            .nested_code_objects
            .iter()
            .filter(|nested| nested.name.as_ref() == "value")
            .collect::<Vec<_>>();

        assert_eq!(value_defs.len(), 2);
        assert_eq!(value_defs[0].arg_count, 1);
        assert_eq!(value_defs[1].arg_count, 2);
    }

    #[test]
    fn test_compile_bytes_literal_uses_builtin_constructor_lowering() {
        let code = compile("value = b'AB'");
        let call = code
            .instructions
            .iter()
            .find(|inst| inst.opcode() == Opcode::Call as u8)
            .expect("expected bytes literal lowering to emit a call");

        assert_eq!(
            call.src2().0,
            2,
            "bytes literal should call bytes(..., encoding)"
        );
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::LoadBuiltin as u8),
            "bytes literal lowering should bypass shadowable globals"
        );
        assert!(
            code.names.iter().any(|name| &**name == "bytes"),
            "bytes constructor should be resolved by name"
        );
    }

    #[test]
    fn test_compile_complex_literal_uses_builtin_constructor_lowering() {
        let code = compile("value = 0j");
        let call = code
            .instructions
            .iter()
            .find(|inst| inst.opcode() == Opcode::Call as u8)
            .expect("expected complex literal lowering to emit a call");

        assert_eq!(
            call.src2().0,
            2,
            "complex literal lowering should call complex(real, imag)"
        );
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::LoadBuiltin as u8),
            "complex literal lowering should bypass shadowable globals"
        );
        assert!(
            code.names.iter().any(|name| &**name == "complex"),
            "complex constructor should be resolved by name"
        );
    }

    #[test]
    fn test_compile_empty_bytes_literal_avoids_placeholder_none_lowering() {
        let bytes_code = compile("value = b''");
        let string_code = compile("value = ''");
        let bytes_load_none_count = bytes_code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadNone as u8)
            .count();
        let string_load_none_count = string_code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadNone as u8)
            .count();

        assert_eq!(
            bytes_load_none_count, string_load_none_count,
            "empty bytes literals should compile like ordinary literals instead of the unimplemented-expression fallback"
        );
    }

    #[test]
    fn test_compile_try_star_rejects_instead_of_lowering_as_regular_try() {
        let module = Module::new(
            vec![Stmt::new(
                StmtKind::TryStar {
                    body: vec![Stmt::new(StmtKind::Pass, Span::new(1, 1))],
                    handlers: Vec::new(),
                    orelse: Vec::new(),
                    finalbody: Vec::new(),
                },
                Span::new(1, 1),
            )],
            Span::new(1, 1),
        );
        let err = Compiler::compile_module(&module, "<test>")
            .expect_err("try/except* must not compile through the regular try path");

        assert!(err.message.contains("TryStar"));
        assert!(err.message.contains("ExceptionGroup"));
    }

    #[test]
    fn test_compile_type_alias_rejects_unimplemented_semantics() {
        let err = try_compile("type Alias = int")
            .expect_err("type alias must not silently compile as a no-op");

        assert!(err.message.contains("TypeAlias"));
        assert!(err.message.contains("TypeAliasType"));
    }

    #[test]
    fn test_compile_match_singleton_uses_identity_opcode() {
        let code = compile(
            r#"
match value:
    case True:
        result = 1
    case _:
        result = 0
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::Is as u8),
            "singleton patterns must compile to identity checks"
        );
    }

    #[test]
    fn test_compile_module_annotations_emit_runtime_namespace_setup() {
        let code = compile(
            r#"
x: int = 1
y: str
"#,
        );

        let setup_annotations_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::SetupAnnotations as u8)
            .count();
        let set_item_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::SetItem as u8)
            .count();

        assert_eq!(
            setup_annotations_count, 1,
            "module annotations should initialize __annotations__ once"
        );
        assert_eq!(
            set_item_count, 2,
            "both simple module annotations should be recorded at runtime"
        );
    }

    #[test]
    fn test_compile_listcomp_reuses_register_pool_after_large_call_blocks() {
        let source = large_call_then_functools_style_listcomp_source();
        let code = compile(&source);
        let stressed = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "stressed")
            .expect("expected nested stressed function");

        assert!(
            !stressed.instructions.is_empty(),
            "list comprehension should compile into a real function body"
        );
    }

    #[test]
    fn test_compile_class_definition_reuses_register_pool_after_large_call_blocks() {
        let source = large_call_then_class_definition_source();
        let code = compile(&source);
        let stressed = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "stressed")
            .expect("expected nested stressed function");

        assert!(
            !stressed.instructions.is_empty(),
            "class definition should compile into a real function body"
        );
    }

    #[test]
    fn test_compile_dotted_import_binds_top_level_name() {
        let code = compile("import pkg.helper");
        let import_name_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::ImportName as u8)
            .count();

        assert_eq!(import_name_count, 2);
        assert!(code.names.iter().any(|name| name.as_ref() == "pkg.helper"));
        assert!(code.names.iter().any(|name| name.as_ref() == "pkg"));
    }

    #[test]
    fn test_compile_relative_import_from_submodule_preserves_level() {
        let code = compile("from .helper import VALUE");
        let import_name_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::ImportName as u8)
            .count();
        let import_from_count = code
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::ImportFrom as u8)
            .count();

        assert_eq!(import_name_count, 1);
        assert_eq!(import_from_count, 1);
        assert!(code.names.iter().any(|name| name.as_ref() == ".helper"));
        assert!(code.names.iter().any(|name| name.as_ref() == "VALUE"));
    }

    #[test]
    fn test_compile_relative_import_without_module_preserves_level() {
        let code = compile("from . import helper");

        assert!(
            code.names.iter().any(|name| name.as_ref() == "."),
            "expected bare relative import to encode its level"
        );
        assert!(code.names.iter().any(|name| name.as_ref() == "helper"));
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::ImportFrom as u8)
        );
    }

    #[test]
    fn test_compile_relative_star_import_preserves_parent_level() {
        let code = compile("from ..pkg import *");

        assert!(code.names.iter().any(|name| name.as_ref() == "..pkg"));
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::ImportStar as u8)
        );
    }

    #[test]
    fn test_compile_assert_emits_raise_path() {
        let code = compile("assert False");
        let opcodes: Vec<u8> = code.instructions.iter().map(|inst| inst.opcode()).collect();

        assert!(
            opcodes.iter().any(|op| *op == Opcode::LoadGlobal as u8),
            "assert should load AssertionError constructor"
        );
        assert!(
            opcodes.iter().any(|op| *op == Opcode::Call as u8),
            "assert should call AssertionError constructor"
        );
        assert!(
            opcodes.iter().any(|op| *op == Opcode::Raise as u8),
            "assert should raise the constructed exception"
        );
    }

    #[test]
    fn test_compile_assert_with_message_emits_call_with_one_arg() {
        let code = compile("assert False, 42");

        let call = code
            .instructions
            .iter()
            .find(|inst| inst.opcode() == Opcode::Call as u8)
            .expect("assert with message should emit Call");
        assert_eq!(
            call.src2().0,
            1,
            "assert message should be passed as 1 call arg"
        );
    }

    #[test]
    fn test_compile_assert_stripped_with_optimize_basic() {
        let code = compile_with_optimization("assert False", OptimizationLevel::Basic);
        let opcodes: Vec<u8> = code.instructions.iter().map(|inst| inst.opcode()).collect();

        assert!(
            !opcodes.iter().any(|op| *op == Opcode::Raise as u8),
            "assert should be stripped under -O"
        );
    }

    #[test]
    fn test_compile_module_docstring_stripped_with_optimize_full() {
        let source = r#"
"""module doc"""
x = 1
"#;
        let unoptimized = compile_with_optimization(source, OptimizationLevel::None);
        let optimized = compile_with_optimization(source, OptimizationLevel::Full);

        let unoptimized_load_consts = unoptimized
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
            .count();
        let optimized_load_consts = optimized
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
            .count();

        assert!(
            optimized_load_consts < unoptimized_load_consts,
            "module docstring should be removed under -OO"
        );
    }

    #[test]
    fn test_compile_function_docstring_stripped_with_optimize_full() {
        let source = r#"
def f():
    """function doc"""
    return 1
"#;
        let unoptimized = compile_with_optimization(source, OptimizationLevel::None);
        let optimized = compile_with_optimization(source, OptimizationLevel::Full);

        let fn_unoptimized = unoptimized
            .nested_code_objects
            .first()
            .map(Arc::as_ref)
            .expect("expected nested function code object");
        let fn_optimized = optimized
            .nested_code_objects
            .first()
            .map(Arc::as_ref)
            .expect("expected nested function code object");

        let unoptimized_load_consts = fn_unoptimized
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
            .count();
        let optimized_load_consts = fn_optimized
            .instructions
            .iter()
            .filter(|inst| inst.opcode() == Opcode::LoadConst as u8)
            .count();

        assert!(
            optimized_load_consts < unoptimized_load_consts,
            "function docstring should be removed under -OO"
        );
    }

    #[test]
    fn test_compile_closure_metadata_follows_closure_slot_order() {
        let code = compile(
            r#"
def outer():
    b = 1
    a = 2

    def inner():
        return a, b

    return inner
"#,
        );

        let outer = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "outer")
            .map(Arc::as_ref)
            .expect("expected outer function code object");
        let inner = outer
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "inner")
            .map(Arc::as_ref)
            .expect("expected inner function code object");

        assert_eq!(
            outer
                .cellvars
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        assert_eq!(
            inner
                .freevars
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn test_compile_explicit_global_in_comprehension_stays_global() {
        let code = compile(
            r#"
seed = [10]

def outer():
    global seed
    return [x + seed[0] for x in range(2)]
"#,
        );

        let outer = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "outer")
            .map(Arc::as_ref)
            .expect("expected outer function code object");
        let listcomp = outer
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "<listcomp>")
            .map(Arc::as_ref)
            .expect("expected nested listcomp code object");

        assert!(
            outer.cellvars.is_empty(),
            "explicit globals must not be materialized as function cellvars"
        );
        assert!(
            listcomp.freevars.is_empty(),
            "comprehension should load explicit outer globals from module globals"
        );
    }

    #[test]
    fn test_register_count() {
        let code = compile("a = 1\nb = 2\nc = a + b");
        // Should use some registers
        assert!(code.register_count > 0);
    }

    // =========================================================================
    // Loop Control Flow Tests (break/continue)
    // =========================================================================

    #[test]
    fn test_break_in_while_loop() {
        // Basic break in while loop
        let code = compile(
            r#"
i = 0
while True:
    i = i + 1
    if i >= 5:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have Jump instructions for break
        let has_jump = code.instructions.iter().any(|i| {
            let opcode = i.opcode();
            opcode == Opcode::Jump as u8
        });
        assert!(has_jump, "expected Jump instruction for break");
    }

    #[test]
    fn test_continue_in_while_loop() {
        // Continue in while loop
        let code = compile(
            r#"
total = 0
i = 0
while i < 10:
    i = i + 1
    if i % 2 == 0:
        continue
    total = total + i
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have Jump instructions for continue
        let has_jump = code.instructions.iter().any(|i| {
            let opcode = i.opcode();
            opcode == Opcode::Jump as u8
        });
        assert!(has_jump, "expected Jump instruction for continue");
    }

    #[test]
    fn test_break_in_for_loop() {
        // Break in for loop
        let code = compile(
            r#"
result = 0
for x in range(100):
    if x == 5:
        break
    result = result + x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_continue_in_for_loop() {
        // Continue in for loop
        let code = compile(
            r#"
total = 0
for x in range(10):
    if x % 2 == 0:
        continue
    total = total + x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_nested_loops_with_break() {
        // Break in nested loops - should only break inner loop
        let code = compile(
            r#"
found = False
for i in range(5):
    for j in range(5):
        if i == 2 and j == 3:
            found = True
            break
    if found:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_nested_loops_with_continue() {
        // Continue in nested loops
        let code = compile(
            r#"
total = 0
for i in range(5):
    for j in range(5):
        if j % 2 == 0:
            continue
        total = total + 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_while_with_else_and_break() {
        // While-else with break (else should be skipped on break)
        let code = compile(
            r#"
i = 0
while i < 10:
    if i == 5:
        break
    i = i + 1
else:
    x = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_for_with_else_and_break() {
        // For-else with break
        let code = compile(
            r#"
for x in range(10):
    if x == 5:
        break
else:
    y = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_break_outside_loop_error() {
        // Break outside loop should be an error
        let result = try_compile("break");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("'break' outside loop"),
            "expected 'break' outside loop error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_continue_outside_loop_error() {
        // Continue outside loop should be an error
        let result = try_compile("continue");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.message.contains("'continue' outside loop"),
            "expected 'continue' outside loop error, got: {}",
            err.message
        );
    }

    #[test]
    fn test_break_in_if_inside_loop() {
        // Break in if statement inside loop is valid
        let code = compile(
            r#"
for x in range(10):
    if x > 5:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_deeply_nested_break() {
        // Break in deeply nested structure
        let code = compile(
            r#"
for a in range(5):
    for b in range(5):
        for c in range(5):
            if a + b + c > 10:
                break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_multiple_breaks_in_loop() {
        // Multiple break statements in same loop
        let code = compile(
            r#"
for x in range(100):
    if x == 5:
        break
    if x == 10:
        break
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_break_and_continue_in_same_loop() {
        // Both break and continue in same loop
        let code = compile(
            r#"
for x in range(100):
    if x == 50:
        break
    if x % 2 == 0:
        continue
    y = x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    // =========================================================================
    // Class Compilation Tests
    // =========================================================================

    #[test]
    fn test_compile_empty_class() {
        // Simplest possible class definition
        let code = compile(
            r#"
class Empty:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Class body code should be in constants
        assert!(
            !code.constants.is_empty(),
            "Class should have nested code object"
        );
    }

    #[test]
    fn test_compile_build_class_encodes_code_const_index_and_base_count() {
        let code = compile(
            r#"
class Child(Base1, Base2):
    pass
"#,
        );

        let build_class = code
            .instructions
            .iter()
            .find(|inst| inst.opcode() == Opcode::BuildClass as u8)
            .expect("expected BUILD_CLASS instruction");

        let build_index = code
            .instructions
            .iter()
            .position(|inst| inst.opcode() == Opcode::BuildClass as u8)
            .expect("expected BUILD_CLASS instruction");
        let meta = code
            .instructions
            .get(build_index + 1)
            .copied()
            .expect("BUILD_CLASS should be followed by ClassMeta");

        assert_eq!(meta.opcode(), Opcode::ClassMeta as u8);
        assert_eq!(meta.dst().0, 2, "base count must match source");

        let code_idx = build_class.imm16() as usize;
        let code_const = code
            .constants
            .get(code_idx)
            .expect("BUILD_CLASS code index must be in constant pool");
        let code_ptr = match code_const {
            Constant::Value(value) => value
                .as_object_ptr()
                .expect("BUILD_CLASS constant must be a code object pointer"),
            Constant::BigInt(_) => panic!("BUILD_CLASS constant must not be a bigint"),
        };

        let nested = code
            .nested_code_objects
            .iter()
            .find(|nested| Arc::as_ptr(nested) as *const () == code_ptr)
            .expect("BUILD_CLASS code object must exist in nested_code_objects");

        assert_eq!(nested.name.as_ref(), "Child");
    }

    #[test]
    fn test_compile_build_class_compiles_bases_into_contiguous_result_block() {
        let code = compile(
            r#"
class Child(Base1, Base2):
    pass
"#,
        );

        let (build_index, build_class) = code
            .instructions
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.opcode() == Opcode::BuildClass as u8)
            .expect("expected BUILD_CLASS instruction");
        let result_reg = build_class.dst().0;

        assert!(
            code.instructions[..build_index].iter().any(|inst| {
                inst.opcode() == Opcode::LoadGlobal as u8 && inst.dst().0 == result_reg + 1
            }),
            "expected first base to be compiled into BUILD_CLASS base slot"
        );
        assert!(
            code.instructions[..build_index].iter().any(|inst| {
                inst.opcode() == Opcode::LoadGlobal as u8 && inst.dst().0 == result_reg + 2
            }),
            "expected second base to be compiled into BUILD_CLASS base slot"
        );
    }

    #[test]
    fn test_compile_build_class_emits_keyword_metadata_for_class_keywords() {
        let code = compile(
            r#"
class Child(Base, answer=42):
    pass
"#,
        );

        let (build_index, _) = code
            .instructions
            .iter()
            .enumerate()
            .find(|(_, inst)| inst.opcode() == Opcode::BuildClass as u8)
            .expect("expected BUILD_CLASS instruction");
        let ext = code
            .instructions
            .get(build_index + 2)
            .copied()
            .expect("BUILD_CLASS with keywords must be followed by metadata");

        assert_eq!(ext.opcode(), Opcode::CallKwEx as u8);
        assert_eq!(ext.dst().0, 1, "expected one class keyword");

        let kwnames_idx = (ext.src1().0 as u16) | ((ext.src2().0 as u16) << 8);
        let names_ptr = code
            .constants
            .get(kwnames_idx as usize)
            .and_then(|value| match value {
                Constant::Value(value) => value.as_object_ptr(),
                Constant::BigInt(_) => None,
            })
            .expect("class keyword metadata should point at keyword names");
        let names = unsafe { &*(names_ptr as *const crate::bytecode::KwNamesTuple) };
        assert_eq!(names.get(0).map(|name| name.as_ref()), Some("answer"));
    }

    #[test]
    fn test_compile_class_with_method() {
        // Class with a simple method
        let code = compile(
            r#"
class Counter:
    def increment(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have nested code object for class body
        assert!(
            !code.constants.is_empty(),
            "Class should have nested code objects"
        );
    }

    #[test]
    fn test_compile_class_with_init() {
        // Class with __init__ method
        let code = compile(
            r#"
class MyClass:
    def __init__(self, x):
        self.x = x
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_variable() {
        // Class with class-level variable
        let code = compile(
            r#"
class Config:
    DEBUG = True
    VERSION = 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_body_predefines_scope_locals() {
        let code = compile(
            r#"
class Config:
    DEBUG = True

    def build(self):
        return DEBUG
"#,
        );

        let class_body = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "Config")
            .map(Arc::as_ref)
            .expect("expected class body code object");

        let locals = class_body
            .locals
            .iter()
            .map(|name| name.as_ref())
            .collect::<Vec<_>>();
        assert!(locals.contains(&"DEBUG"));
        assert!(locals.contains(&"build"));
    }

    #[test]
    fn test_compile_nested_class_binds_into_class_namespace() {
        let code = compile(
            r#"
class Outer:
    class Inner:
        pass
"#,
        );

        let outer = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "Outer")
            .map(Arc::as_ref)
            .expect("expected outer class body");

        assert!(
            outer
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
            "nested class bindings inside a class body should target class locals"
        );
    }

    #[test]
    fn test_compile_class_delete_name_targets_class_locals() {
        let code = compile(
            r#"
class Example:
    value = 1
    del value
"#,
        );

        let class_body = code
            .nested_code_objects
            .iter()
            .find(|nested| nested.name.as_ref() == "Example")
            .map(Arc::as_ref)
            .expect("expected class body code object");

        assert!(
            class_body
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::DeleteLocal as u8),
            "class body deletes should target class locals"
        );
        assert!(
            !class_body
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::DeleteGlobal as u8),
            "class body deletes should not be lowered as global deletes"
        );
    }

    #[test]
    fn test_compile_dynamic_locals_binds_module_assignments_as_locals() {
        let code = compile_with_dynamic_locals(
            r#"
x = 1
y = x
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
            "dynamic-locals compilation should route module assignments through local slots",
        );
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::LoadLocal as u8),
            "dynamic-locals compilation should route module lookups through local slots",
        );
        assert!(
            !code
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::StoreGlobal as u8),
            "ordinary dynamic-locals bindings should not be lowered as global stores",
        );
    }

    #[test]
    fn test_compile_dynamic_locals_preserves_explicit_global_bindings() {
        let code = compile_with_dynamic_locals(
            r#"
global shared
x = 1
shared = x
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::StoreLocal as u8),
            "local dynamic bindings should still use local slots when globals are declared elsewhere",
        );
        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::StoreGlobal as u8),
            "explicit global statements must continue to target module globals",
        );
    }

    #[test]
    fn test_compile_dynamic_locals_uses_local_lookups_for_unbound_names() {
        let code = compile_with_dynamic_locals(
            r#"
result = missing_name
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::LoadLocal as u8),
            "dynamic-locals compilation should use locals-first lookups even for unbound names",
        );
        assert!(
            !code
                .instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::LoadGlobal as u8),
            "unbound dynamic-locals lookups should not bypass the locals mapping",
        );
    }

    #[test]
    fn test_compile_delete_subscript_lowers_to_del_item() {
        let code = compile(
            r#"
mapping = {"token": 1}
del mapping["token"]
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::DelItem as u8),
            "subscript deletes should lower to DelItem"
        );
    }

    #[test]
    fn test_compile_delete_attribute_lowers_to_del_attr() {
        let code = compile(
            r#"
class Box:
    pass

box = Box()
del box.value
"#,
        );

        assert!(
            code.instructions
                .iter()
                .any(|inst| inst.opcode() == Opcode::DelAttr as u8),
            "attribute deletes should lower to DelAttr"
        );
    }

    #[test]
    fn test_compile_class_with_single_base() {
        // Simple inheritance
        let code = compile(
            r#"
class Child(Parent):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_multiple_bases() {
        // Multiple inheritance
        let code = compile(
            r#"
class Multi(Base1, Base2, Base3):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_decorator() {
        // Decorated class
        let code = compile(
            r#"
@decorator
class MyClass:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
        // Should have CALL for decorator application
    }

    #[test]
    fn test_compile_class_with_multiple_decorators() {
        // Multiple decorators
        let code = compile(
            r#"
@decorator1
@decorator2
@decorator3
class MyClass:
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_decorator_call() {
        // Decorator with arguments
        let code = compile(
            r#"
@dataclass(frozen=True)
class Point:
    x: int
    y: int
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_multiple_methods() {
        // Class with multiple methods
        let code = compile(
            r#"
class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_static_method() {
        // Class with static method
        let code = compile(
            r#"
class Utils:
    @staticmethod
    def helper(x):
        return x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_method() {
        // Class with class method
        let code = compile(
            r#"
class Factory:
    @classmethod
    def create(cls):
        return cls()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_property() {
        // Class with property decorator
        let code = compile(
            r#"
class Circle:
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_nested_class() {
        // Nested class definition
        let code = compile(
            r#"
class Outer:
    class Inner:
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_deeply_nested_class() {
        // Deeply nested class definitions
        let code = compile(
            r#"
class Level1:
    class Level2:
        class Level3:
            value = 42
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_docstring() {
        // Class with docstring
        let code = compile(
            r#"
class Documented:
    """This is a docstring."""
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_super_init() {
        // Class calling super().__init__
        let code = compile(
            r#"
class Child(Parent):
    def __init__(self):
        super().__init__()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_explicit_super() {
        // Class using explicit super(ClassName, self)
        let code = compile(
            r#"
class Child(Parent):
    def __init__(self):
        super(Child, self).__init__()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_super_method_call() {
        // Class calling super() method
        let code = compile(
            r#"
class Child(Parent):
    def process(self):
        return super().process() + 1
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_zero_arg_super_captures_class_cell() {
        let code = compile(
            r#"
class Child(Parent):
    def method(self):
        return super().process()
"#,
        );

        let class_body = code
            .nested_code_objects
            .first()
            .expect("class body code should be nested in module");
        assert!(
            class_body
                .cellvars
                .iter()
                .any(|name| name.as_ref() == "__class__"),
            "class body should expose __class__ as a cellvar"
        );

        let method_code = class_body
            .nested_code_objects
            .first()
            .expect("method code should be nested in class body");
        assert!(
            method_code
                .freevars
                .iter()
                .any(|name| name.as_ref() == "__class__"),
            "method using zero-arg super should capture __class__ as a freevar"
        );
    }

    #[test]
    fn test_compile_class_with_dunder_methods() {
        // Class with magic methods
        let code = compile(
            r#"
class Custom:
    def __str__(self):
        return "Custom"
    
    def __repr__(self):
        return "Custom()"
    
    def __len__(self):
        return 0
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_operator_overloading() {
        // Class with operator overloading
        let code = compile(
            r#"
class Vector:
    def __add__(self, other):
        pass
    
    def __sub__(self, other):
        pass
    
    def __mul__(self, scalar):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_slots() {
        // Class with __slots__ definition
        let code = compile(
            r#"
class Point:
    __slots__ = ['x', 'y']
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_class_body_expression() {
        // Class with expression in body
        let code = compile(
            r#"
class Computed:
    VALUE = 1 + 2 + 3
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_conditional() {
        // Class with conditional in body
        let code = compile(
            r#"
class Conditional:
    if True:
        x = 1
    else:
        x = 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_for_loop() {
        // Class with for loop in body
        let code = compile(
            r#"
class Generated:
    items = []
    for i in range(5):
        items.append(i)
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_comprehension() {
        // Class with comprehension in body
        let code = compile(
            r#"
class WithComprehension:
    squares = [x**2 for x in range(10)]
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_inheriting_from_expression() {
        // Class inheriting from expression
        let code = compile(
            r#"
class Sub(get_base()):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_method_decorator() {
        // Method with multiple decorators
        let code = compile(
            r#"
class Service:
    @decorator1
    @decorator2
    def method(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_private_method() {
        // Class with private method (name mangling)
        let code = compile(
            r#"
class Private:
    def __secret(self):
        pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_closure_in_method() {
        // Method containing a closure
        let code = compile(
            r#"
class WithClosure:
    def outer(self):
        x = 1
        def inner():
            return x
        return inner
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_multiple_classes() {
        // Multiple class definitions in same module
        let code = compile(
            r#"
class First:
    pass

class Second:
    pass

class Third(First, Second):
    pass
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_and_function() {
        // Class and function in same module
        let code = compile(
            r#"
def helper():
    pass

class MyClass:
    def method(self):
        helper()
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_dataclass_like() {
        // Dataclass-like pattern
        let code = compile(
            r#"
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_singleton_pattern() {
        // Singleton pattern
        let code = compile(
            r#"
class Singleton:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_class_with_lambda_in_body() {
        // Class with lambda in body
        let code = compile(
            r#"
class WithLambda:
    transform = lambda x: x * 2
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_class_code_object_has_class_flag() {
        // Verify class compilation produces code object
        let code = compile(
            r#"
class Flagged:
    pass
"#,
        );
        // Verify we have constants (class body code object)
        assert!(
            !code.constants.is_empty(),
            "Class body code object should exist in constants"
        );
        // Verify instructions are generated
        assert!(!code.instructions.is_empty());
    }

    // =========================================================================
    // Exception Compilation Tests
    // =========================================================================

    #[test]
    fn test_compile_simple_try_except() {
        let code = compile(
            r#"
try:
    x = 1
except:
    y = 2
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_except_with_type() {
        let code = compile(
            r#"
try:
    x = dangerous()
except ValueError:
    y = fallback()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_typed_except_emits_verifiable_dynamic_handler_metadata() {
        let code = compile(
            r#"
try:
    from _abc import get_cache_token
except ImportError:
    get_cache_token = None
except (AttributeError, TypeError):
    get_cache_token = lambda: None
"#,
        );

        code.validate()
            .expect("typed except handler metadata should validate");
        assert!(
            code.exception_table
                .iter()
                .all(|entry| entry.exception_type_idx == u16::MAX),
            "typed except matching is dynamic and should not encode handler PCs as type metadata"
        );
    }

    #[test]
    fn test_compile_try_except_else() {
        let code = compile(
            r#"
try:
    x = 1
except:
    y = 2
else:
    z = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_finally() {
        let code = compile(
            r#"
try:
    x = 1
finally:
    cleanup()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(!code.exception_table.is_empty());
    }

    #[test]
    fn test_compile_try_except_finally() {
        let code = compile(
            r#"
try:
    x = 1
except ValueError:
    y = 2
finally:
    cleanup()
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 2);
    }

    #[test]
    fn test_compile_multiple_except_handlers() {
        let code = compile(
            r#"
try:
    x = risky()
except ValueError:
    a = 1
except TypeError:
    b = 2
except:
    c = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 3);
    }

    #[test]
    fn test_compile_nested_try_except() {
        let code = compile(
            r#"
try:
    try:
        x = 1
    except:
        y = 2
except:
    z = 3
"#,
        );
        assert!(!code.instructions.is_empty());
        assert!(code.exception_table.len() >= 2);
    }

    #[test]
    fn test_compile_try_in_function() {
        let code = compile(
            r#"
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return 0
"#,
        );
        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_augassign_rejects_unpacking_targets() {
        let err = try_compile("x, b += 3").expect_err("tuple augassign target should fail");
        assert!(
            err.message
                .contains("illegal expression for augmented assignment"),
            "unexpected compile error: {err:?}"
        );
    }
}
