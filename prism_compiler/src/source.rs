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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_source_module_returns_ast_and_code() {
        let compilation = compile_source_module("value = 42\n", "<test>", OptimizationLevel::Basic)
            .expect("source compilation should succeed");

        assert_eq!(compilation.module.body.len(), 1);
        assert!(!compilation.code.instructions.is_empty());
    }

    #[test]
    fn test_compile_source_code_returns_parse_error_variant() {
        let err = compile_source_code("def\n", "<test>", OptimizationLevel::None)
            .expect_err("invalid syntax should fail parsing");

        assert!(err.as_parse_error().is_some());
        assert!(err.as_compile_error().is_none());
    }

    #[test]
    fn test_compile_source_code_returns_compile_error_variant() {
        let err = compile_source_code("continue\n", "<test>", OptimizationLevel::None)
            .expect_err("invalid control flow should fail compilation");

        assert!(err.as_parse_error().is_none());
        let compile_error = err
            .as_compile_error()
            .expect("expected compilation error variant");
        assert!(compile_error.message.contains("continue"));
    }

    #[test]
    fn test_compile_source_code_with_namespace_mode_emits_bytecode() {
        let code = compile_source_code_with_namespace_mode(
            "x = 1\n",
            "<test>",
            OptimizationLevel::Basic,
            ModuleNamespaceMode::DynamicLocals,
        )
        .expect("dynamic locals compilation should succeed");

        assert!(!code.instructions.is_empty());
    }

    #[test]
    fn test_compile_source_code_maps_byte_offsets_to_source_lines() {
        let source = "\n\nclass C:\n    def run(self):\n        raise ValueError('x')\n";
        let code = compile_source_code(source, "lineprobe.py", OptimizationLevel::None)
            .expect("source compilation should succeed");

        let class_code = code
            .nested_code_objects
            .first()
            .expect("class body code object should be emitted");
        let function_code = class_code
            .nested_code_objects
            .first()
            .expect("method code object should be emitted");

        assert_eq!(class_code.first_lineno, 3);
        assert_eq!(function_code.first_lineno, 4);
        assert!(
            function_code.line_table.iter().any(|entry| entry.line == 5),
            "method line table should point at the raise statement: {:?}",
            function_code.line_table
        );
        assert!(
            function_code
                .line_table
                .iter()
                .all(|entry| entry.line < 100),
            "byte offsets must not leak into line-table entries: {:?}",
            function_code.line_table
        );
    }
}
