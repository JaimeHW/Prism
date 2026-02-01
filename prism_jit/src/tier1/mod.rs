//! Template JIT (Tier 1) code generation.
//!
//! The Template JIT provides fast 1:1 bytecode-to-native translation with:
//! - Per-opcode code templates for direct compilation
//! - Simple stack-based frame layout for easy OSR
//! - Inline caches for type feedback
//! - Deoptimization stubs for fallback to interpreter

pub mod codegen;
pub mod deopt;
pub mod frame;
pub mod template;

// Re-export main types
pub use codegen::TemplateCompiler;
pub use deopt::DeoptInfo;
pub use frame::FrameLayout;
pub use template::OpcodeTemplate;
