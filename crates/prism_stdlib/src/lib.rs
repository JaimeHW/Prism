//! Static metadata for Prism's native stdlib surface.
//!
//! This crate owns the stable inventory of native stdlib modules and builtin
//! importer metadata that higher-level layers use for planning and import
//! policy decisions.

/// Import resolution policy for a native stdlib module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StdlibResolutionPolicy {
    /// Always load Prism's native module implementation.
    PreferNative,
    /// Prefer a filesystem-backed Python source module when available.
    PreferSourceWhenAvailable,
}

/// Generated metadata for a Prism-native stdlib module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeModuleMetadata {
    name: &'static str,
    policy: StdlibResolutionPolicy,
    platform: StdlibPlatform,
}

impl NativeModuleMetadata {
    /// Returns the fully qualified Python module name.
    #[inline]
    pub const fn name(self) -> &'static str {
        self.name
    }

    /// Returns the import resolution policy for this module.
    #[inline]
    pub const fn policy(self) -> StdlibResolutionPolicy {
        self.policy
    }

    #[inline]
    fn is_available_on_current_platform(self) -> bool {
        self.platform.is_current()
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StdlibPlatform {
    All,
    Unix,
    Windows,
    Other,
}

impl StdlibPlatform {
    #[inline]
    fn is_current(self) -> bool {
        match self {
            StdlibPlatform::All => true,
            StdlibPlatform::Unix => cfg!(unix),
            StdlibPlatform::Windows => cfg!(windows),
            StdlibPlatform::Other => !cfg!(unix) && !cfg!(windows),
        }
    }
}

/// Absolute path to Prism's source-backed Python stdlib modules in this checkout.
///
/// Native modules remain the preferred implementation for hot runtime paths.
/// This directory is for compatibility modules whose performance is irrelevant
/// to user code startup or steady-state execution, such as test infrastructure.
#[inline]
pub fn source_stdlib_path() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/python")
}

include!(concat!(env!("OUT_DIR"), "/stdlib_metadata.rs"));

/// Returns the builtin modules exposed through importlib's builtin importer.
pub fn builtin_module_names() -> &'static [&'static str] {
    if cfg!(windows) {
        WINDOWS_BUILTIN_MODULE_NAMES
    } else if cfg!(unix) {
        UNIX_BUILTIN_MODULE_NAMES
    } else {
        OTHER_BUILTIN_MODULE_NAMES
    }
}

/// Returns whether a module is exposed through importlib's builtin importer.
#[inline]
pub fn is_builtin_module_name(name: &str) -> bool {
    builtin_module_names().binary_search(&name).is_ok()
}

/// Returns generated metadata for Prism-native stdlib modules on this platform.
pub fn native_modules() -> impl Iterator<Item = NativeModuleMetadata> + 'static {
    NATIVE_MODULES
        .iter()
        .copied()
        .filter(|module| module.is_available_on_current_platform())
}

/// Returns the native stdlib resolution policy for a module, if Prism ships one.
pub fn native_module_policy(name: &str) -> Option<StdlibResolutionPolicy> {
    native_modules()
        .find(|module| module.name() == name)
        .map(NativeModuleMetadata::policy)
}

/// Returns whether Prism ships a native stdlib module implementation for `name`.
#[inline]
pub fn is_native_stdlib_module(name: &str) -> bool {
    native_module_policy(name).is_some()
}

/// Returns whether Prism should prefer a source-backed stdlib module for `name`.
#[inline]
pub fn prefers_source_when_available(name: &str) -> bool {
    native_module_policy(name) == Some(StdlibResolutionPolicy::PreferSourceWhenAvailable)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn builtin_module_names_are_sorted_and_unique() {
        assert_sorted_unique(WINDOWS_BUILTIN_MODULE_NAMES);
        assert_sorted_unique(UNIX_BUILTIN_MODULE_NAMES);
        assert_sorted_unique(OTHER_BUILTIN_MODULE_NAMES);
    }

    #[test]
    fn native_module_names_are_unique_on_current_platform() {
        let names: Vec<_> = native_modules().map(NativeModuleMetadata::name).collect();
        assert_sorted_unique(&names);
    }

    fn assert_sorted_unique(names: &[&str]) {
        let sorted = names.windows(2).all(|pair| pair[0] < pair[1]);
        assert!(sorted, "module names must be strictly sorted: {names:?}");

        let unique_count = names.iter().copied().collect::<BTreeSet<_>>().len();
        assert_eq!(unique_count, names.len(), "module names must be unique");
    }
}
