//! # Prism Parser
//!
//! Complete Python 3.12 lexer and parser.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod ast;
pub mod lexer;
pub mod parser;
pub mod token;

pub use ast::*;
pub use lexer::{Lexer, tokenize};
pub use parser::{Parser, parse, parse_expression};
pub use token::{Token, TokenKind};
