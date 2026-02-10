//! Method dispatch module for optimized method calls.
//!
//! This module provides high-performance method lookup and invocation via
//! the LoadMethod/CallMethod opcode pair. Key optimizations:
//!
//! - **Three-Tier Caching**:
//!   1. Inline cache (per call-site, ~3 cycles)
//!   2. Method cache (per type+name, ~10 cycles)
//!   3. Full MRO traversal (~100+ cycles)
//!
//! - **Register Layout**:
//!   ```text
//!   LoadMethod stores: [method_reg] = method, [method_reg+1] = self
//!   CallMethod reads:  [method_reg] = method, [method_reg+1] = self, [+2..] = args
//!   ```
//!
//! - **Type-Specialized Dispatch**:
//!   - FunctionObject: Direct frame push
//!   - BuiltinFunction: Inline call
//!   - BoundMethod: Extract and recurse

pub mod call_method;
pub mod load_method;
pub mod method_cache;

pub use call_method::call_method;
pub use load_method::load_method;
pub use method_cache::{CachedMethod, MethodCache, method_cache};
