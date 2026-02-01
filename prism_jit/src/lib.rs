//! JVM/V8-tier JIT compiler for Prism.
//!
//! Custom high-performance JIT backend with:
//! - Sea-of-Nodes IR
//! - Advanced optimization passes
//! - Graph-coloring register allocation
//! - Native x64 code generation
#![deny(unsafe_op_in_unsafe_fn)]
pub mod backend;
pub mod ir;
pub mod opt;
pub mod regalloc;
