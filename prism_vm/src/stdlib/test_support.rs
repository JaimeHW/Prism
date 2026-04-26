//! Minimal native `test.support` surface for CPython regression tests.
//!
//! CPython's `test.support` package imports a large portion of the stdlib at
//! module import time. Prism exposes the tiny support subset needed by early
//! compatibility targets natively, so individual regression modules can run
//! before the whole support stack is implemented.

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use std::sync::Arc;

const TESTFN: &str = "@prism_test_tmp";

/// Native `test.support` module descriptor.
#[derive(Debug, Clone)]
pub struct SupportModule {
    attrs: Vec<Arc<str>>,
}

impl SupportModule {
    /// Create a new `test.support` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("os_helper")],
        }
    }
}

impl Default for SupportModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SupportModule {
    fn name(&self) -> &str {
        "test.support"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        Err(ModuleError::AttributeError(format!(
            "module 'test.support' has no attribute '{}'",
            name
        )))
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

/// Native `test.support.os_helper` module descriptor.
#[derive(Debug, Clone)]
pub struct OsHelperModule {
    attrs: Vec<Arc<str>>,
}

impl OsHelperModule {
    /// Create a new `test.support.os_helper` module descriptor.
    pub fn new() -> Self {
        Self {
            attrs: vec![Arc::from("TESTFN")],
        }
    }
}

impl Default for OsHelperModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsHelperModule {
    fn name(&self) -> &str {
        "test.support.os_helper"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "TESTFN" => Ok(Value::string(intern(TESTFN))),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'test.support.os_helper' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
