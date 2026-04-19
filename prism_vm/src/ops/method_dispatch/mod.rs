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

mod builtin_methods;
pub mod call_method;
pub mod load_method;
pub mod method_cache;

use prism_runtime::object::type_obj::TypeId;

pub use call_method::call_method;
pub use load_method::load_method;
pub use method_cache::{CachedMethod, MethodCache, method_cache};

/// Resolve builtin instance methods that can also be materialized through normal
/// attribute access.
pub(crate) fn resolve_builtin_instance_method(type_id: TypeId, name: &str) -> Option<CachedMethod> {
    let resolved = match type_id {
        TypeId::OBJECT => builtin_methods::resolve_object_method(name),
        TypeId::TYPE => builtin_methods::resolve_type_method(name),
        TypeId::INT => builtin_methods::resolve_int_method(name),
        TypeId::FLOAT => builtin_methods::resolve_float_method(name),
        TypeId::BOOL => builtin_methods::resolve_bool_method(name),
        TypeId::DEQUE => builtin_methods::resolve_deque_method(name),
        TypeId::REGEX_PATTERN => builtin_methods::resolve_regex_pattern_method(name),
        TypeId::REGEX_MATCH => builtin_methods::resolve_regex_match_method(name),
        TypeId::LIST => builtin_methods::resolve_list_method(name),
        TypeId::DICT => builtin_methods::resolve_dict_method(name),
        TypeId::MAPPING_PROXY => builtin_methods::resolve_mapping_proxy_method(name),
        TypeId::BYTEARRAY => builtin_methods::resolve_bytearray_method(name),
        TypeId::PROPERTY => builtin_methods::resolve_property_method(name),
        TypeId::STR => builtin_methods::resolve_str_method(name),
        TypeId::SET | TypeId::FROZENSET => builtin_methods::resolve_set_method(type_id, name),
        TypeId::GENERATOR => builtin_methods::resolve_generator_method(name),
        TypeId::FUNCTION | TypeId::CLOSURE => builtin_methods::resolve_function_method(name),
        _ => None,
    };
    resolved.or_else(|| builtin_methods::resolve_generic_dunder_method(type_id, name))
}
