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

mod comprehensions;
mod control_flow;
mod dynamic_calls;
mod function_defs;

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
        if let Some(doc) = Self::body_docstring(&module.body)
            && compiler.optimize < OptimizationLevel::Full
        {
            compiler.emit_namespace_docstring_binding(doc);
        }
        if Self::body_needs_runtime_annotations(&module.body) {
            compiler.builder.emit_setup_annotations();
        }

        for (index, stmt) in module.body.iter().enumerate() {
            if compiler.should_strip_docstring_stmt(index, stmt)
                || compiler.should_skip_emitted_docstring_stmt(index, stmt)
            {
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

    /// Return the literal docstring for a Python body, if one is present.
    #[inline]
    fn body_docstring(body: &[Stmt]) -> Option<&str> {
        match body.first().map(|stmt| &stmt.kind) {
            Some(StmtKind::Expr(expr)) => match &expr.kind {
                ExprKind::String(literal) => Some(literal.value.as_str()),
                _ => None,
            },
            _ => None,
        }
    }

    /// Whether to strip this statement as a docstring under `-OO`.
    #[inline]
    fn should_strip_docstring_stmt(&self, index: usize, stmt: &Stmt) -> bool {
        self.optimize >= OptimizationLevel::Full && index == 0 && Self::is_docstring_stmt(stmt)
    }

    /// Whether this docstring statement has already been emitted as `__doc__`.
    #[inline]
    fn should_skip_emitted_docstring_stmt(&self, index: usize, stmt: &Stmt) -> bool {
        index == 0 && Self::is_docstring_stmt(stmt) && self.optimize < OptimizationLevel::Full
    }

    fn emit_namespace_docstring_binding(&mut self, doc: &str) {
        let doc_reg = self.builder.alloc_register();
        let doc_idx = self.builder.add_string(doc);
        self.builder.emit_load_const(doc_reg, doc_idx);

        if self.current_scope().kind == ScopeKind::Module
            && self.module_namespace_mode != ModuleNamespaceMode::DynamicLocals
        {
            let name_idx = self.builder.add_name("__doc__");
            self.builder.emit_store_global(name_idx, doc_reg);
        } else {
            let slot = self.builder.define_local("__doc__");
            self.builder.emit_store_local(slot, doc_reg);
        }

        self.builder.free_register(doc_reg);
    }

    fn emit_function_docstring_attribute(&mut self, func_reg: Register, doc: &str) {
        let doc_reg = self.builder.alloc_register();
        let doc_idx = self.builder.add_string(doc);
        self.builder.emit_load_const(doc_reg, doc_idx);
        let name_idx = self.builder.add_name("__doc__");
        self.builder.emit_set_attr(func_reg, name_idx, doc_reg);
        self.builder.free_register(doc_reg);
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
                if let Some(doc) = Self::body_docstring(body)
                    && self.optimize < OptimizationLevel::Full
                {
                    self.emit_namespace_docstring_binding(doc);
                }
                if Self::body_needs_runtime_annotations(body) {
                    self.builder.emit_setup_annotations();
                }

                // Compile class body statements (method definitions, class variables, etc.)
                for (index, stmt) in body.iter().enumerate() {
                    if self.should_strip_docstring_stmt(index, stmt)
                        || self.should_skip_emitted_docstring_stmt(index, stmt)
                    {
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
            self.compile_annotation_target_evaluation(target)?;
        }

        Ok(())
    }

    fn compile_annotation_target_evaluation(&mut self, target: &Expr) -> CompileResult<()> {
        match &target.kind {
            ExprKind::Name(_) => Ok(()),
            ExprKind::Attribute { value, .. } => {
                let reg = self.compile_expr(value)?;
                self.builder.free_register(reg);
                Ok(())
            }
            ExprKind::Subscript { value, slice } => {
                let value_reg = self.compile_expr(value)?;
                self.builder.free_register(value_reg);
                self.compile_annotation_slice_evaluation(slice)
            }
            ExprKind::Tuple(elts) | ExprKind::List(elts) => {
                for elt in elts {
                    self.compile_annotation_target_evaluation(elt)?;
                }
                Ok(())
            }
            ExprKind::Starred(inner) => self.compile_annotation_target_evaluation(inner),
            _ => {
                let reg = self.compile_expr(target)?;
                self.builder.free_register(reg);
                Ok(())
            }
        }
    }

    fn compile_annotation_slice_evaluation(&mut self, slice: &Expr) -> CompileResult<()> {
        if let ExprKind::Slice { lower, upper, step } = &slice.kind {
            for part in [lower.as_deref(), upper.as_deref(), step.as_deref()]
                .into_iter()
                .flatten()
            {
                let reg = self.compile_expr(part)?;
                self.builder.free_register(reg);
            }
            Ok(())
        } else {
            let reg = self.compile_expr(slice)?;
            self.builder.free_register(reg);
            Ok(())
        }
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
            ExprKind::Name(_) | ExprKind::Attribute { .. } | ExprKind::Subscript { .. } => Ok(()),
            _ => Err(CompileError {
                message: "illegal expression for augmented assignment".to_string(),
                line: self.line_for_span(target.span),
                column: 0,
            }),
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
            BinOp::MatMult => self.builder.emit_matmul(dst, left, right),
        }
    }

    /// Emit an augmented assignment operation.
    fn emit_augop(&mut self, op: AugOp, dst: Register, left: Register, right: Register) {
        match op {
            AugOp::Add => self.builder.emit_inplace_add(dst, left, right),
            AugOp::Sub => self.builder.emit_inplace_sub(dst, left, right),
            AugOp::Mult => self.builder.emit_inplace_mul(dst, left, right),
            AugOp::Div => self.builder.emit_inplace_div(dst, left, right),
            AugOp::FloorDiv => self.builder.emit_inplace_floor_div(dst, left, right),
            AugOp::Mod => self.builder.emit_inplace_mod(dst, left, right),
            AugOp::Pow => self.builder.emit_inplace_pow(dst, left, right),
            AugOp::LShift => self.builder.emit_inplace_shl(dst, left, right),
            AugOp::RShift => self.builder.emit_inplace_shr(dst, left, right),
            AugOp::BitAnd => self.builder.emit_inplace_bitwise_and(dst, left, right),
            AugOp::BitOr => self.builder.emit_inplace_bitwise_or(dst, left, right),
            AugOp::BitXor => self.builder.emit_inplace_bitwise_xor(dst, left, right),
            AugOp::MatMult => self.builder.emit_inplace_matmul(dst, left, right),
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
}
