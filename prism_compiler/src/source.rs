//! Source-to-bytecode frontend helpers.
//!
//! This module owns the high-level "source text -> parsed AST -> compiled
//! code object" flow that multiple layers use.

use crate::CodeObject;
use crate::compiler::{CompileError, Compiler, ModuleNamespaceMode, OptimizationLevel};
use prism_core::PrismError;
use prism_parser::ast::Module;
use std::sync::Arc;

/// Full compilation result for a source-backed module.
#[derive(Debug)]
pub struct SourceCompilation {
    /// Parsed module AST.
    pub module: Module,
    /// Compiled code object for the module body.
    pub code: Arc<CodeObject>,
}

/// Errors that can occur while compiling source text into bytecode.
#[derive(Debug, Clone)]
pub enum SourceCompileError {
    /// Parsing the source text failed.
    Parse(PrismError),
    /// Lowering a parsed AST into bytecode failed.
    Compile(CompileError),
}

impl SourceCompileError {
    /// Returns the underlying parser error when compilation failed before
    /// bytecode generation started.
    #[inline]
    pub fn as_parse_error(&self) -> Option<&PrismError> {
        match self {
            Self::Parse(err) => Some(err),
            Self::Compile(_) => None,
        }
    }

    /// Returns the underlying compiler error when AST lowering failed.
    #[inline]
    pub fn as_compile_error(&self) -> Option<&CompileError> {
        match self {
            Self::Parse(_) => None,
            Self::Compile(err) => Some(err),
        }
    }
}

impl std::fmt::Display for SourceCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(err) => err.fmt(f),
            Self::Compile(err) => err.fmt(f),
        }
    }
}

impl std::error::Error for SourceCompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Parse(err) => Some(err),
            Self::Compile(err) => Some(err),
        }
    }
}

impl From<PrismError> for SourceCompileError {
    fn from(value: PrismError) -> Self {
        Self::Parse(value)
    }
}

impl From<CompileError> for SourceCompileError {
    fn from(value: CompileError) -> Self {
        Self::Compile(value)
    }
}

/// Parse and compile a source-backed module with standard module namespace
/// semantics.
#[inline]
pub fn compile_source_module(
    source: &str,
    filename: &str,
    optimize: OptimizationLevel,
) -> Result<SourceCompilation, SourceCompileError> {
    compile_source_module_with_namespace_mode(
        source,
        filename,
        optimize,
        ModuleNamespaceMode::Standard,
    )
}

/// Parse and compile a source-backed module with explicit namespace lowering
/// semantics.
pub fn compile_source_module_with_namespace_mode(
    source: &str,
    filename: &str,
    optimize: OptimizationLevel,
    module_namespace_mode: ModuleNamespaceMode,
) -> Result<SourceCompilation, SourceCompileError> {
    let module = prism_parser::parse(source).map_err(SourceCompileError::Parse)?;
    let code = Compiler::compile_module_with_source_and_namespace_mode(
        &module,
        source,
        filename,
        optimize,
        module_namespace_mode,
    )
    .map(Arc::new)
    .map_err(SourceCompileError::Compile)?;

    Ok(SourceCompilation { module, code })
}

/// Compile a source-backed module and return only the emitted code object.
#[inline]
pub fn compile_source_code(
    source: &str,
    filename: &str,
    optimize: OptimizationLevel,
) -> Result<Arc<CodeObject>, SourceCompileError> {
    compile_source_code_with_namespace_mode(
        source,
        filename,
        optimize,
        ModuleNamespaceMode::Standard,
    )
}

/// Compile a source-backed module into bytecode with explicit namespace
/// lowering semantics and return only the emitted code object.
#[inline]
pub fn compile_source_code_with_namespace_mode(
    source: &str,
    filename: &str,
    optimize: OptimizationLevel,
    module_namespace_mode: ModuleNamespaceMode,
) -> Result<Arc<CodeObject>, SourceCompileError> {
    compile_source_module_with_namespace_mode(source, filename, optimize, module_namespace_mode)
        .map(|compilation| compilation.code)
}
