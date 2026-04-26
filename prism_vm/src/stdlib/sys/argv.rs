//! Command-line argument handling.
//!
//! Provides zero-copy access to command-line arguments with
//! efficient sharing via Arc<str>.

use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::list::ListObject;
use std::sync::Arc;

/// Command-line arguments container.
///
/// Uses `Arc<str>` for individual arguments to enable zero-copy sharing
/// when arguments are accessed from Python.
#[derive(Debug, Clone)]
pub struct SysArgv {
    /// The argument list.
    args: Arc<[Arc<str>]>,
}

impl SysArgv {
    /// Create from a vector of strings.
    #[inline]
    pub fn new(args: Vec<String>) -> Self {
        let args: Vec<Arc<str>> = args.into_iter().map(|s| s.into()).collect();
        Self { args: args.into() }
    }

    /// Create from environment (std::env::args).
    #[inline]
    pub fn from_env() -> Self {
        let args: Vec<Arc<str>> = std::env::args().map(|s| s.into()).collect();
        Self { args: args.into() }
    }

    /// Create empty argv.
    #[inline]
    pub fn empty() -> Self {
        Self {
            args: Arc::from([]),
        }
    }

    /// Get the number of arguments.
    #[inline]
    pub fn len(&self) -> usize {
        self.args.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.args.is_empty()
    }

    /// Get argument by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.args.get(index)
    }

    /// Get the script name (first argument).
    #[inline]
    pub fn script(&self) -> Option<&Arc<str>> {
        self.args.first()
    }

    /// Iterate over arguments.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.args.iter()
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[Arc<str>] {
        &self.args
    }

    /// Convert argv to a Python list value (`list[str]`).
    ///
    /// This builds a list of interned string Values and returns it as an object Value.
    pub fn to_value(&self) -> Value {
        let values: Vec<Value> = self
            .args
            .iter()
            .map(|arg| Value::string(intern(arg.as_ref())))
            .collect();
        let list = ListObject::from_slice(&values);
        crate::alloc_managed_value(list)
    }
}

impl Default for SysArgv {
    fn default() -> Self {
        Self::empty()
    }
}

impl<'a> IntoIterator for &'a SysArgv {
    type Item = &'a Arc<str>;
    type IntoIter = std::slice::Iter<'a, Arc<str>>;

    fn into_iter(self) -> Self::IntoIter {
        self.args.iter()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
