//! High-performance I/O types for Prism Python runtime.
//!
//! This module implements the Python `io` module's type hierarchy with a focus
//! on zero-allocation fast paths and minimal syscall overhead.
//!
//! # Architecture
//!
//! The I/O stack follows Python's layered design:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    TextIOWrapper                        │
//! │  (encoding/decoding, line buffering, universal newlines)│
//! ├─────────────────────────────────────────────────────────┤
//! │                  BufferedReader/Writer                  │
//! │           (read-ahead, write-behind buffering)          │
//! ├─────────────────────────────────────────────────────────┤
//! │                       FileIO                            │
//! │              (raw unbuffered file access)               │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Performance Optimizations
//!
//! - **Buffer Pooling**: Thread-local buffer pools eliminate allocation overhead
//! - **SIMD Line Scanning**: Fast ASCII newline detection for text mode
//! - **Read-Ahead Heuristics**: Adaptive buffer sizing based on access patterns
//! - **Zero-Copy Paths**: Direct memcpy for binary mode bulk transfers
//!
//! # Module Organization
//!
//! - [`file_io`]: Raw unbuffered file operations (`FileIO`)
//! - [`buffered`]: Buffered I/O wrappers (`BufferedReader`, `BufferedWriter`)
//! - [`text`]: Text mode with encoding (`TextIOWrapper`)
//! - [`mode`]: File mode parsing and validation
//! - [`buffer_pool`]: Thread-local buffer recycling

pub mod buffer_pool;
pub mod buffered;
pub mod file_io;
pub mod mode;
pub mod text;

// Re-export primary types
pub use buffer_pool::BufferPool;
pub use buffered::{BufferedRandom, BufferedReader, BufferedWriter};
pub use file_io::FileIO;
pub use mode::{FileMode, ParseModeError};
pub use text::TextIOWrapper;
