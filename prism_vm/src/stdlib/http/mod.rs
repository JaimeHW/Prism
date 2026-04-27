//! Native `http` package bootstrap surface.

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_runtime::types::list::ListObject;
use std::sync::Arc;

pub mod cookies;

/// Native `http` package descriptor.
#[derive(Debug, Clone, Default)]
pub struct HttpModule;

impl HttpModule {
    /// Create a native `http` package module.
    pub fn new() -> Self {
        Self
    }
}

impl Module for HttpModule {
    fn name(&self) -> &str {
        "http"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__path__" => Ok(crate::alloc_managed_value(ListObject::new())),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'http' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![Arc::from("__path__")]
    }
}
