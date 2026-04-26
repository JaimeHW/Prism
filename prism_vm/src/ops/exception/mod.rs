//! Exception opcode support modules.
//!
//! This module provides high-performance infrastructure for exception handling
//! opcodes, including type extraction, dynamic matching, traceback construction,
//! and sys.exc_info() support.
//!
//! # Architecture
//!
//! ```text
//! ops/exception/
//! ├── mod.rs         # This file - module exports and integration
//! ├── helpers.rs     # Type extraction, dynamic matching, tuple matching
//! ├── traceback.rs   # Traceback construction and extraction
//! └── exc_info.rs    # sys.exc_info() triple support
//! ```
//!
//! # Performance Design
//!
//! All operations are designed for minimal overhead:
//!
//! | Module | Hot Path | Cold Path |
//! |--------|----------|-----------|
//! | helpers | O(1) type checks | O(N) tuple iteration |
//! | traceback | O(1) extraction | O(N) capture |
//! | exc_info | O(1) build | O(1) write |
//!
//! # Usage
//!
//! These modules are used by the main exception opcode handlers:
//!
//! ```ignore
//! // In ops/exception.rs (main file)
//! use crate::ops::exception::helpers::{extract_type_id, check_tuple_match};
//! use crate::ops::exception::traceback::{capture_traceback, extract_traceback};
//! use crate::ops::exception::exc_info::{build_exc_info, write_exc_info_to_registers};
//! ```

pub mod exc_info;
pub mod helpers;
pub mod opcodes;
pub mod traceback;

// Re-export opcode handlers for direct use
pub use opcodes::{
    abort_except, bind_exception, check_exc_match, check_exc_match_tuple, clear_exception,
    end_finally, enter_except, exception_match, exit_except, get_exception,
    get_exception_traceback, has_exc_info, load_exception, pop_exc_info, pop_except_handler,
    push_exc_info, raise, raise_from, raise_with_cause, reraise, setup_except,
};

// Re-export commonly used items
pub use exc_info::{ExcInfo, build_exc_info, write_empty_exc_info, write_exc_info_to_registers};
pub use helpers::{
    DEFAULT_EXCEPTION_TYPE, NO_TYPE_ID, check_dynamic_match, check_tuple_match, extract_type_id,
    extract_type_id_from_value, is_subclass,
};
pub use traceback::{
    build_frame_info, build_frame_info_with_line, capture_traceback, extend_traceback,
    extract_traceback, has_traceback, traceback_to_value, value_to_traceback,
};
