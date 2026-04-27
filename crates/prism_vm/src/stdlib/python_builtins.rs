//! Importable Python `builtins` module backed by the VM builtin registry.
//!
//! CPython exposes builtins both through direct global lookup and as the
//! importable `builtins` module. Prism mirrors that model by projecting the
//! shared builtin registry through the stdlib module abstraction instead of
//! special-casing imports in the VM.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::BuiltinRegistry;
use std::sync::Arc;

/// Importable `builtins` module backed by a builtin registry snapshot.
#[derive(Debug, Clone)]
pub struct BuiltinsModule {
    builtins: BuiltinRegistry,
    attrs: Vec<Arc<str>>,
}

impl BuiltinsModule {
    /// Create a builtins module backed by the provided registry.
    pub fn new(builtins: BuiltinRegistry) -> Self {
        let mut attrs: Vec<_> = builtins.iter().map(|(name, _)| Arc::clone(name)).collect();
        attrs.sort_unstable_by(|left, right| left.as_ref().cmp(right.as_ref()));

        Self { builtins, attrs }
    }
}

impl Module for BuiltinsModule {
    fn name(&self) -> &str {
        "builtins"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        self.builtins.get(name).ok_or_else(|| {
            ModuleError::AttributeError(format!("module 'builtins' has no attribute '{}'", name))
        })
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}
