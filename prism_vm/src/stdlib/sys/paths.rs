//! Module search path management.
//!
//! Manages sys.path for module importing with efficient
//! path manipulation and caching.

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Module search paths container.
///
/// Implements Python's sys.path semantics with efficient
/// path storage and lookup.
#[derive(Debug, Clone)]
pub struct SysPaths {
    /// The path list.
    paths: Vec<Arc<str>>,
}

/// Installation prefix configuration surfaced through `sys.prefix` and
/// related attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SysPrefixes {
    prefix: Arc<str>,
    exec_prefix: Arc<str>,
    base_prefix: Arc<str>,
    base_exec_prefix: Arc<str>,
}

impl SysPrefixes {
    /// Detect prefix values using `PYTHONHOME` when available, otherwise fall
    /// back to the executable directory.
    pub fn from_env(executable: Option<&Path>) -> Self {
        if let Some((prefix, exec_prefix)) = python_home_prefixes() {
            return Self {
                prefix: Arc::from(prefix.as_str()),
                exec_prefix: Arc::from(exec_prefix.as_str()),
                base_prefix: Arc::from(prefix.as_str()),
                base_exec_prefix: Arc::from(exec_prefix.as_str()),
            };
        }

        let install_root = executable
            .and_then(Path::parent)
            .map(|path| path.to_string_lossy().into_owned())
            .unwrap_or_default();

        Self {
            prefix: Arc::from(install_root.as_str()),
            exec_prefix: Arc::from(install_root.as_str()),
            base_prefix: Arc::from(install_root.as_str()),
            base_exec_prefix: Arc::from(install_root.as_str()),
        }
    }

    #[inline]
    pub fn prefix(&self) -> &str {
        self.prefix.as_ref()
    }

    #[inline]
    pub fn exec_prefix(&self) -> &str {
        self.exec_prefix.as_ref()
    }

    #[inline]
    pub fn base_prefix(&self) -> &str {
        self.base_prefix.as_ref()
    }

    #[inline]
    pub fn base_exec_prefix(&self) -> &str {
        self.base_exec_prefix.as_ref()
    }
}

fn python_home_prefixes() -> Option<(String, String)> {
    let home = std::env::var("PYTHONHOME").ok()?;
    if home.is_empty() {
        return None;
    }

    let mut parts = home.split(path_separator());
    let prefix = parts.next()?.trim();
    if prefix.is_empty() {
        return None;
    }
    let exec_prefix = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(prefix);

    Some((prefix.to_string(), exec_prefix.to_string()))
}

impl SysPaths {
    /// Create empty paths.
    #[inline]
    pub fn new() -> Self {
        Self { paths: Vec::new() }
    }

    /// Create with initial paths.
    #[inline]
    pub fn with_paths(paths: Vec<String>) -> Self {
        Self {
            paths: paths.into_iter().map(|s| s.into()).collect(),
        }
    }

    /// Create default paths from environment.
    pub fn from_env() -> Self {
        let mut paths = Vec::new();

        // Add current directory (empty string in Python)
        paths.push(Arc::from(""));

        // Add PYTHONPATH entries if set
        if let Ok(pythonpath) = std::env::var("PYTHONPATH") {
            for path in pythonpath.split(path_separator()) {
                if !path.is_empty() {
                    paths.push(Arc::from(path));
                }
            }
        }

        Self { paths }
    }

    /// Get number of paths.
    #[inline]
    pub fn len(&self) -> usize {
        self.paths.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Get path by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.paths.get(index)
    }

    /// Append a path.
    #[inline]
    pub fn append(&mut self, path: impl Into<Arc<str>>) {
        self.paths.push(path.into());
    }

    /// Insert a path at index.
    #[inline]
    pub fn insert(&mut self, index: usize, path: impl Into<Arc<str>>) {
        if index <= self.paths.len() {
            self.paths.insert(index, path.into());
        }
    }

    /// Remove path at index.
    #[inline]
    pub fn remove(&mut self, index: usize) -> Option<Arc<str>> {
        if index < self.paths.len() {
            Some(self.paths.remove(index))
        } else {
            None
        }
    }

    /// Clear all paths.
    #[inline]
    pub fn clear(&mut self) {
        self.paths.clear();
    }

    /// Check if path exists in list.
    #[inline]
    pub fn contains(&self, path: &str) -> bool {
        self.paths.iter().any(|p| p.as_ref() == path)
    }

    /// Iterate over paths.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.paths.iter()
    }

    /// Get as slice.
    #[inline]
    pub fn as_slice(&self) -> &[Arc<str>] {
        &self.paths
    }

    /// Convert sys.path to a Python list value (`list[str]`).
    pub fn to_value(&self) -> Value {
        let values: Vec<Value> = self
            .paths
            .iter()
            .map(|path| Value::string(intern(path.as_ref())))
            .collect();
        let list = ListObject::from_slice(&values);
        crate::alloc_managed_value(list)
    }

    /// Resolve a module name to a path.
    ///
    /// Searches paths in order for a matching module file.
    pub fn resolve_module(&self, module_name: &str) -> Option<PathBuf> {
        let module_file = format!("{}.py", module_name.replace('.', "/"));
        let package_init = format!("{}/__init__.py", module_name.replace('.', "/"));

        for path in &self.paths {
            let base = if path.is_empty() {
                PathBuf::from(".")
            } else {
                PathBuf::from(path.as_ref())
            };

            // Check for module file
            let module_path = base.join(&module_file);
            if module_path.exists() {
                return Some(module_path);
            }

            // Check for package
            let package_path = base.join(&package_init);
            if package_path.exists() {
                return Some(package_path);
            }
        }

        None
    }
}

impl Default for SysPaths {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for &'a SysPaths {
    type Item = &'a Arc<str>;
    type IntoIter = std::slice::Iter<'a, Arc<str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.paths.iter()
    }
}

// =============================================================================
// Platform Path Separator
// =============================================================================

/// Get the platform-specific path separator for PATH variables.
#[inline]
const fn path_separator() -> char {
    #[cfg(target_os = "windows")]
    {
        ';'
    }
    #[cfg(not(target_os = "windows"))]
    {
        ':'
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
