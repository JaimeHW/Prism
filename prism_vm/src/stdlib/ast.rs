//! Native `ast` module facade.
//!
//! Prism keeps the public AST entry points native so parser and compiler
//! tooling can bootstrap without importing CPython's large pure-Python
//! `ast.py`. The implementation validates source through Prism's parser,
//! returns real `_ast` root instances, and preserves source text for lossless
//! `compile(ast_obj, ...)` round trips while the full Python AST graph is
//! filled out incrementally.

use super::{_ast, Module, ModuleError, ModuleResult};
use crate::VirtualMachine;
use crate::builtins::{BuiltinError, BuiltinFunctionObject};
use prism_core::PrismError;
use prism_core::Value;
use prism_core::intern::intern;
use prism_parser::{parse as parse_module_source, parse_expression};
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use std::sync::{Arc, LazyLock};

static PARSE_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new_vm_kw(Arc::from("ast.parse"), ast_parse));

const AST_EXPORTS: &[&str] = &[
    "parse",
    "AST",
    "mod",
    "expr",
    "expr_context",
    "operator",
    "Constant",
    "Tuple",
    "Name",
    "Attribute",
    "Load",
    "Add",
    "Sub",
    "BitOr",
    "Module",
    "Expression",
    "Interactive",
    "arg",
    "PyCF_ONLY_AST",
    "PyCF_TYPE_COMMENTS",
    "PyCF_ALLOW_TOP_LEVEL_AWAIT",
];

/// Native `ast` module descriptor.
#[derive(Debug, Clone)]
pub struct AstModule {
    attrs: Vec<Arc<str>>,
    all: Value,
}

impl AstModule {
    /// Create a native `ast` module.
    pub fn new() -> Self {
        Self {
            attrs: AST_EXPORTS
                .iter()
                .copied()
                .chain(["__all__"])
                .map(Arc::from)
                .collect(),
            all: string_list_value(AST_EXPORTS),
        }
    }
}

impl Default for AstModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for AstModule {
    fn name(&self) -> &str {
        "ast"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "parse" => Ok(builtin_value(&PARSE_FUNCTION)),
            attr => _ast::exported_attr(attr).ok_or_else(|| {
                ModuleError::AttributeError(format!("module 'ast' has no attribute '{}'", attr))
            }),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

/// Parse source and return a Python-visible `_ast` root object.
pub fn parse_source_to_ast_value(
    source: &str,
    filename: &str,
    mode: _ast::AstParseMode,
) -> Result<Value, BuiltinError> {
    validate_source_for_mode(source, filename, mode)?;
    Ok(_ast::parsed_ast_value(source, filename, mode))
}

/// Extract the original source from a Prism parsed-AST object.
#[inline]
pub fn parsed_ast_source(value: Value) -> Option<String> {
    _ast::parsed_ast_source(value)
}

#[inline]
pub const fn compiler_only_ast_flag() -> i64 {
    _ast::PYCF_ONLY_AST
}

#[inline]
pub const fn compiler_allowed_flags() -> i64 {
    _ast::PYCF_ONLY_AST | _ast::PYCF_TYPE_COMMENTS | _ast::PYCF_ALLOW_TOP_LEVEL_AWAIT
}

fn ast_parse(
    _vm: &mut VirtualMachine,
    args: &[Value],
    keywords: &[(&str, Value)],
) -> Result<Value, BuiltinError> {
    let parsed = bind_parse_args(args, keywords)?;
    parse_source_to_ast_value(&parsed.source, &parsed.filename, parsed.mode)
}

struct ParseArgs {
    source: String,
    filename: String,
    mode: _ast::AstParseMode,
}

fn bind_parse_args(args: &[Value], keywords: &[(&str, Value)]) -> Result<ParseArgs, BuiltinError> {
    if args.len() > 3 {
        return Err(BuiltinError::TypeError(format!(
            "parse() takes from 1 to 3 positional arguments but {} were given",
            args.len()
        )));
    }

    let mut source = args.first().copied();
    let mut filename = args.get(1).copied();
    let mut mode = args.get(2).copied();
    let mut type_comments = None;
    let mut feature_version = None;

    for &(name, value) in keywords {
        match name {
            "source" => assign_parse_keyword(&mut source, value, "source")?,
            "filename" => assign_parse_keyword(&mut filename, value, "filename")?,
            "mode" => assign_parse_keyword(&mut mode, value, "mode")?,
            "type_comments" => assign_parse_keyword(&mut type_comments, value, "type_comments")?,
            "feature_version" => {
                assign_parse_keyword(&mut feature_version, value, "feature_version")?
            }
            unknown => {
                return Err(BuiltinError::TypeError(format!(
                    "parse() got an unexpected keyword argument '{}'",
                    unknown
                )));
            }
        }
    }

    let source = source
        .ok_or_else(|| BuiltinError::TypeError("parse() missing required argument 'source'".into()))
        .and_then(source_text)?;
    let filename = filename
        .map(filename_text)
        .transpose()?
        .unwrap_or_else(|| "<unknown>".to_string());
    let mode = mode
        .map(parse_mode_value)
        .transpose()?
        .unwrap_or(_ast::AstParseMode::Exec);

    Ok(ParseArgs {
        source,
        filename,
        mode,
    })
}

#[inline]
fn assign_parse_keyword(
    slot: &mut Option<Value>,
    value: Value,
    name: &'static str,
) -> Result<(), BuiltinError> {
    if slot.is_some() {
        return Err(BuiltinError::TypeError(format!(
            "parse() got multiple values for argument '{}'",
            name
        )));
    }
    *slot = Some(value);
    Ok(())
}

fn source_text(value: Value) -> Result<String, BuiltinError> {
    if let Some(string) = value_as_string_ref(value) {
        return Ok(string.as_str().to_string());
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        return std::str::from_utf8(bytes.as_bytes())
            .map(str::to_string)
            .map_err(|_| {
                BuiltinError::SyntaxError(
                    "source code string cannot contain undecodable bytes".to_string(),
                )
            });
    }

    Err(BuiltinError::TypeError(
        "ast.parse() source must be a string or bytes".to_string(),
    ))
}

fn filename_text(value: Value) -> Result<String, BuiltinError> {
    value_as_string_ref(value)
        .map(|filename| filename.as_str().to_string())
        .ok_or_else(|| BuiltinError::TypeError("ast.parse() filename must be a string".to_string()))
}

fn parse_mode_value(value: Value) -> Result<_ast::AstParseMode, BuiltinError> {
    let mode = value_as_string_ref(value)
        .ok_or_else(|| BuiltinError::TypeError("ast.parse() mode must be a string".to_string()))?;
    _ast::AstParseMode::from_str(mode.as_str()).ok_or_else(|| {
        BuiltinError::ValueError("compile() mode must be 'exec', 'eval' or 'single'".to_string())
    })
}

fn validate_source_for_mode(
    source: &str,
    _filename: &str,
    mode: _ast::AstParseMode,
) -> Result<(), BuiltinError> {
    match mode {
        _ast::AstParseMode::Exec | _ast::AstParseMode::Single => parse_module_source(source)
            .map(|_| ())
            .map_err(syntax_error_from_parse_error),
        _ast::AstParseMode::Eval => parse_expression(source)
            .map(|_| ())
            .map_err(syntax_error_from_parse_error),
    }
}

fn syntax_error_from_parse_error(err: PrismError) -> BuiltinError {
    BuiltinError::SyntaxError(err.to_string())
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}
