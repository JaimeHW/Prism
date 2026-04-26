//! Error types and result definitions for Prism.
//!
//! This module provides a comprehensive error hierarchy covering all phases of execution:
//! - Lexical errors (tokenization failures)
//! - Syntax errors (parsing failures)
//! - Compilation errors (bytecode generation failures)
//! - Runtime errors (execution failures)
//! - Type errors (dynamic type mismatches)

use crate::span::Span;
use std::fmt;
use thiserror::Error;

/// The unified result type used throughout Prism.
pub type PrismResult<T> = Result<T, PrismError>;

/// Comprehensive error type covering all Prism error conditions.
#[derive(Error, Debug, Clone)]
pub enum PrismError {
    /// Lexical analysis error.
    #[error("SyntaxError: {message}")]
    LexError {
        /// Error description.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Syntax/parsing error.
    #[error("SyntaxError: {message}")]
    SyntaxError {
        /// Error description.
        message: String,
        /// Source location.
        span: Span,
    },

    /// Compilation error.
    #[error("CompileError: {message}")]
    CompileError {
        /// Error description.
        message: String,
        /// Source location.
        span: Option<Span>,
    },

    /// Runtime error during execution.
    #[error("{kind}: {message}")]
    RuntimeError {
        /// The Python exception type name.
        kind: RuntimeErrorKind,
        /// Error description.
        message: String,
    },

    /// Name not found in scope.
    #[error("NameError: name '{name}' is not defined")]
    NameError {
        /// The undefined name.
        name: String,
    },

    /// Type mismatch error.
    #[error("TypeError: {message}")]
    TypeError {
        /// Error description.
        message: String,
    },

    /// Value error.
    #[error("ValueError: {message}")]
    ValueError {
        /// Error description.
        message: String,
    },

    /// Attribute access error.
    #[error("AttributeError: {message}")]
    AttributeError {
        /// Error description.
        message: String,
    },

    /// Index out of bounds.
    #[error("IndexError: {message}")]
    IndexError {
        /// Error description.
        message: String,
    },

    /// Key not found in mapping.
    #[error("KeyError: {key}")]
    KeyError {
        /// The missing key representation.
        key: String,
    },

    /// Zero division.
    #[error("ZeroDivisionError: {message}")]
    ZeroDivisionError {
        /// Error description.
        message: String,
    },

    /// Import failure.
    #[error("ImportError: {message}")]
    ImportError {
        /// Error description.
        message: String,
    },

    /// Assertion failure.
    #[error("AssertionError: {message}")]
    AssertionError {
        /// Error description.
        message: String,
    },

    /// Stop iteration signal (not actually an error in normal flow).
    #[error("StopIteration")]
    StopIteration,

    /// Overflow error.
    #[error("OverflowError: {message}")]
    OverflowError {
        /// Error description.
        message: String,
    },

    /// Recursion limit exceeded.
    #[error("RecursionError: maximum recursion depth exceeded")]
    RecursionError,

    /// Memory allocation failure.
    #[error("MemoryError: {message}")]
    MemoryError {
        /// Error description.
        message: String,
    },

    /// Internal VM error (should never occur in correct implementation).
    #[error("InternalError: {message}")]
    InternalError {
        /// Error description.
        message: String,
    },
}

impl PrismError {
    /// Create a lex error with location.
    #[must_use]
    pub fn lex(message: impl Into<String>, span: Span) -> Self {
        Self::LexError {
            message: message.into(),
            span,
        }
    }

    /// Create a syntax error with location.
    #[must_use]
    pub fn syntax(message: impl Into<String>, span: Span) -> Self {
        Self::SyntaxError {
            message: message.into(),
            span,
        }
    }

    /// Create a compile error.
    #[must_use]
    pub fn compile(message: impl Into<String>, span: Option<Span>) -> Self {
        Self::CompileError {
            message: message.into(),
            span,
        }
    }

    /// Create a runtime error with kind.
    #[must_use]
    pub fn runtime(kind: RuntimeErrorKind, message: impl Into<String>) -> Self {
        Self::RuntimeError {
            kind,
            message: message.into(),
        }
    }

    /// Create a name error.
    #[must_use]
    pub fn name(name: impl Into<String>) -> Self {
        Self::NameError { name: name.into() }
    }

    /// Create a type error.
    #[must_use]
    pub fn type_error(message: impl Into<String>) -> Self {
        Self::TypeError {
            message: message.into(),
        }
    }

    /// Create a value error.
    #[must_use]
    pub fn value_error(message: impl Into<String>) -> Self {
        Self::ValueError {
            message: message.into(),
        }
    }

    /// Create an attribute error.
    #[must_use]
    pub fn attribute(message: impl Into<String>) -> Self {
        Self::AttributeError {
            message: message.into(),
        }
    }

    /// Create an index error.
    #[must_use]
    pub fn index(message: impl Into<String>) -> Self {
        Self::IndexError {
            message: message.into(),
        }
    }

    /// Create a key error.
    #[must_use]
    pub fn key(key: impl Into<String>) -> Self {
        Self::KeyError { key: key.into() }
    }

    /// Create a zero division error.
    #[must_use]
    pub fn zero_division(message: impl Into<String>) -> Self {
        Self::ZeroDivisionError {
            message: message.into(),
        }
    }

    /// Create an import error.
    #[must_use]
    pub fn import(message: impl Into<String>) -> Self {
        Self::ImportError {
            message: message.into(),
        }
    }

    /// Create an assertion error.
    #[must_use]
    pub fn assertion(message: impl Into<String>) -> Self {
        Self::AssertionError {
            message: message.into(),
        }
    }

    /// Create an internal error.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
        }
    }

    /// Get the Python exception type name.
    #[must_use]
    pub fn exception_type(&self) -> &'static str {
        match self {
            Self::LexError { .. } | Self::SyntaxError { .. } => "SyntaxError",
            Self::CompileError { .. } => "SyntaxError",
            Self::RuntimeError { kind, .. } => kind.as_str(),
            Self::NameError { .. } => "NameError",
            Self::TypeError { .. } => "TypeError",
            Self::ValueError { .. } => "ValueError",
            Self::AttributeError { .. } => "AttributeError",
            Self::IndexError { .. } => "IndexError",
            Self::KeyError { .. } => "KeyError",
            Self::ZeroDivisionError { .. } => "ZeroDivisionError",
            Self::ImportError { .. } => "ImportError",
            Self::AssertionError { .. } => "AssertionError",
            Self::StopIteration => "StopIteration",
            Self::OverflowError { .. } => "OverflowError",
            Self::RecursionError => "RecursionError",
            Self::MemoryError { .. } => "MemoryError",
            Self::InternalError { .. } => "SystemError",
        }
    }
}

/// Runtime error classification matching Python's exception hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuntimeErrorKind {
    /// Generic runtime error.
    Runtime,
    /// Exception raised by user code.
    Exception,
    /// System exit requested.
    SystemExit,
    /// Keyboard interrupt.
    KeyboardInterrupt,
    /// Generator exit.
    GeneratorExit,
}

impl RuntimeErrorKind {
    /// Get the Python exception type name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Runtime => "RuntimeError",
            Self::Exception => "Exception",
            Self::SystemExit => "SystemExit",
            Self::KeyboardInterrupt => "KeyboardInterrupt",
            Self::GeneratorExit => "GeneratorExit",
        }
    }
}

impl fmt::Display for RuntimeErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests;
