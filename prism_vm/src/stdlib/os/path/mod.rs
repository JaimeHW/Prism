//! os.path module - Path manipulation operations.
//!
//! High-performance implementation of Python's `os.path` module providing:
//! - Path joining and normalization (`join`, `abspath`, `normpath`, `realpath`)
//! - Path splitting (`basename`, `dirname`, `splitext`, `split`, `splitdrive`)
//! - Path queries (`exists`, `isfile`, `isdir`, `islink`, `isabs`, `lexists`, `ismount`)
//! - Path comparison (`commonpath`, `commonprefix`, `relpath`, `samefile`)
//! - Timestamp access (`getmtime`, `getatime`, `getctime`, `lgetmtime`)

mod compare;
mod join;
mod normalize;
mod query;
mod split;
mod time_access;

pub use compare::*;
pub use join::*;
pub use normalize::*;
pub use query::*;
pub use split::*;
pub use time_access::*;

use super::constants::SEP;
use crate::stdlib::{Module, ModuleError};
use prism_core::Value;
use std::sync::Arc;

/// Minimal module wrapper for `os.path`.
///
/// Prism already implements the path algorithms in this module tree; this
/// wrapper exposes the submodule itself to the import system so `import os.path`
/// and `from os import path` behave like CPython.
pub struct OsPathModule;

impl OsPathModule {
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

impl Default for OsPathModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsPathModule {
    fn name(&self) -> &str {
        "os.path"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            "sep" => Ok(Value::none()),
            "exists" | "lexists" | "isfile" | "isdir" | "isabs" | "ismount" | "join"
            | "abspath" | "normpath" | "realpath" | "basename" | "dirname" | "split"
            | "splitdrive" | "splitext" | "commonpath" | "commonprefix" | "relpath"
            | "samefile" | "getmtime" | "getatime" | "getctime" => Ok(Value::none()),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'os.path' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            Arc::from("sep"),
            Arc::from("exists"),
            Arc::from("lexists"),
            Arc::from("isfile"),
            Arc::from("isdir"),
            Arc::from("isabs"),
            Arc::from("ismount"),
            Arc::from("join"),
            Arc::from("abspath"),
            Arc::from("normpath"),
            Arc::from("realpath"),
            Arc::from("basename"),
            Arc::from("dirname"),
            Arc::from("split"),
            Arc::from("splitdrive"),
            Arc::from("splitext"),
            Arc::from("commonpath"),
            Arc::from("commonprefix"),
            Arc::from("relpath"),
            Arc::from("samefile"),
            Arc::from("getmtime"),
            Arc::from("getatime"),
            Arc::from("getctime"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Ensure all submodules export correctly
        let _ = exists(".");
        let _ = join("a", "b");
        let _ = basename("/foo/bar");
        let _ = abspath(".");
    }

    #[test]
    fn test_module_wrapper_exposes_os_path_name() {
        let module = OsPathModule::new();
        assert_eq!(module.name(), "os.path");
        assert!(module.dir().contains(&Arc::from("join")));
    }
}
