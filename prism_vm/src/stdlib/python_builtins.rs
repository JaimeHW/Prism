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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtins_module_exposes_core_functions() {
        let module = BuiltinsModule::new(BuiltinRegistry::with_standard_builtins());

        assert!(module.get_attr("open").is_ok());
        assert!(module.get_attr("len").is_ok());
        assert!(module.get_attr("type").is_ok());
    }

    #[test]
    fn test_builtins_module_reports_missing_attr() {
        let module = BuiltinsModule::new(BuiltinRegistry::with_standard_builtins());
        let err = module
            .get_attr("definitely_missing")
            .expect_err("missing attrs should error");

        match err {
            ModuleError::AttributeError(message) => {
                assert!(message.contains("definitely_missing"));
            }
            other => panic!("expected AttributeError, got {other:?}"),
        }
    }

    #[test]
    fn test_builtins_module_dir_is_sorted_and_complete() {
        let builtins = BuiltinRegistry::with_standard_builtins();
        let module = BuiltinsModule::new(builtins.clone());
        let dir = module.dir();

        assert!(dir.windows(2).all(|window| window[0] <= window[1]));
        assert_eq!(dir.len(), builtins.len());
        assert!(dir.iter().any(|name| name.as_ref() == "open"));
        assert!(dir.iter().any(|name| name.as_ref() == "__import__"));
    }

    #[test]
    fn test_builtins_module_preserves_builtin_identity() {
        let builtins = BuiltinRegistry::with_standard_builtins();
        let module = BuiltinsModule::new(builtins.clone());

        let registry_open = builtins
            .get("open")
            .and_then(|value| value.as_object_ptr())
            .expect("open should be a builtin function object");
        let module_open = module
            .get_attr("open")
            .expect("builtins.open should exist")
            .as_object_ptr()
            .expect("builtins.open should be a builtin function object");

        assert_eq!(module_open, registry_open);
    }
}
