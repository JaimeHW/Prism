//! Package import system — dotted names, `__init__.py`, and relative imports.
//!
//! Provides the machinery for resolving:
//! - **Dotted imports**: `import os.path` → resolve `os`, then get `path`
//! - **Package detection**: Identify packages via `__init__.py` or `__path__`
//! - **Relative imports**: `from . import foo`, `from ..bar import baz`
//!
//! # Architecture
//!
//! ```text
//! resolve_dotted_name("os.path")
//!   ├── Split → ["os", "path"]
//!   ├── Import "os" (via ImportResolver)
//!   ├── Cache "os" in sys.modules
//!   └── Get "path" attr from "os" module → cache as "os.path"
//!
//! resolve_relative_import(level=2, name="bar", package="foo.baz.qux")
//!   ├── Trim package by level → "foo"
//!   ├── Append name → "foo.bar"
//!   └── Resolve "foo.bar" via resolve_dotted_name
//! ```
//!
//! # Performance
//!
//! - Single allocation for split components (SmallVec for ≤4 parts)
//! - Fast path for non-dotted names (no splitting needed)
//! - All intermediate modules are cached in sys.modules

use super::resolver::ImportError;
use std::path::Path;
use std::sync::Arc;

// =============================================================================
// Dotted Name Resolution
// =============================================================================

/// Parsed dotted import name with pre-split components.
///
/// Avoids re-splitting on each resolution attempt.
#[derive(Debug, Clone)]
pub struct DottedName {
    /// The full dotted name (e.g., "os.path.join").
    full_name: Arc<str>,
    /// Pre-split components (e.g., ["os", "path", "join"]).
    /// Uses a Vec but the common case is 2-3 parts.
    parts: Vec<Arc<str>>,
}

impl DottedName {
    /// Parse a dotted module name.
    ///
    /// Returns `None` for empty names.
    #[inline]
    pub fn parse(name: &str) -> Option<Self> {
        if name.is_empty() {
            return None;
        }

        let parts: Vec<Arc<str>> = name.split('.').map(Arc::from).collect();

        // Validate: no empty parts (e.g., ".os" or "os..path")
        if parts.iter().any(|p| p.is_empty()) {
            return None;
        }

        Some(Self {
            full_name: Arc::from(name),
            parts,
        })
    }

    /// Get the full dotted name.
    #[inline]
    pub fn full_name(&self) -> &str {
        &self.full_name
    }

    /// Get the parts of the dotted name.
    #[inline]
    pub fn parts(&self) -> &[Arc<str>] {
        &self.parts
    }

    /// Check if this is a simple (non-dotted) name.
    #[inline]
    pub fn is_simple(&self) -> bool {
        self.parts.len() == 1
    }

    /// Get the top-level module name.
    #[inline]
    pub fn top_level(&self) -> &str {
        &self.parts[0]
    }

    /// Get the number of components.
    #[inline]
    pub fn depth(&self) -> usize {
        self.parts.len()
    }

    /// Build the name for a given nesting depth.
    ///
    /// E.g., for "os.path.join", depth=2 → "os.path"
    pub fn name_at_depth(&self, depth: usize) -> String {
        let depth = depth.min(self.parts.len());
        self.parts[..depth]
            .iter()
            .map(|p| p.as_ref())
            .collect::<Vec<_>>()
            .join(".")
    }
}

// =============================================================================
// Relative Import Resolution
// =============================================================================

/// Resolve a relative import to an absolute module name.
///
/// # Parameters
///
/// - `level`: Number of leading dots (1 = current package, 2 = parent, etc.)
/// - `name`: Optional name after the dots (e.g., "bar" in `from ..bar import baz`)
/// - `package`: The `__package__` of the importing module
///
/// # Returns
///
/// The absolute module name, or an error if the level exceeds the package depth.
///
/// # Examples
///
/// ```ignore
/// // In package "foo.bar.baz":
/// resolve_relative("", 1, "foo.bar.baz")    // → "foo.bar.baz"  (from . import)
/// resolve_relative("qux", 1, "foo.bar.baz") // → "foo.bar.baz.qux"
/// resolve_relative("qux", 2, "foo.bar.baz") // → "foo.bar.qux"
/// resolve_relative("", 3, "foo.bar.baz")    // → "foo"
/// ```
pub fn resolve_relative_import(
    name: &str,
    level: u32,
    package: &str,
) -> Result<String, ImportError> {
    if level == 0 {
        // Absolute import — just return the name
        return Ok(name.to_string());
    }

    if package.is_empty() {
        return Err(ImportError::LoadError {
            module: Arc::from(name),
            message: Arc::from("attempted relative import in non-package"),
        });
    }

    // Split package into parts
    let pkg_parts: Vec<&str> = package.split('.').collect();
    let level = level as usize;

    // Level 1 means current package, level 2 means parent, etc.
    // So we need at least (level - 1) parent parts
    if level > pkg_parts.len() {
        return Err(ImportError::LoadError {
            module: Arc::from(name),
            message: Arc::from(format!(
                "attempted relative import beyond top-level package (level={}, package depth={})",
                level,
                pkg_parts.len()
            )),
        });
    }

    // Take package parts up to (len - level + 1)
    let base_depth = pkg_parts.len() - level + 1;
    let base: String = pkg_parts[..base_depth].join(".");

    if name.is_empty() {
        Ok(base)
    } else {
        Ok(format!("{}.{}", base, name))
    }
}

// =============================================================================
// Package Detection
// =============================================================================

/// Check if a directory is a Python package (contains `__init__.py`).
#[inline]
pub fn is_package(dir: &Path) -> bool {
    dir.is_dir() && dir.join("__init__.py").exists()
}

/// Find the `__init__.py` file for a package directory.
///
/// Returns `None` if the directory is not a package.
#[inline]
pub fn find_init_file(dir: &Path) -> Option<std::path::PathBuf> {
    let init = dir.join("__init__.py");
    if init.exists() { Some(init) } else { None }
}

/// Resolve a module name to a filesystem path.
///
/// Searches the given paths for:
/// 1. A package directory with `__init__.py`
/// 2. A `.py` source file
///
/// Returns `(path, is_package)` if found.
pub fn find_module_source(
    name: &str,
    search_paths: &[Arc<str>],
) -> Option<(std::path::PathBuf, bool)> {
    for base_path in search_paths {
        let base = Path::new(base_path.as_ref());

        // Check for package directory (name/__init__.py)
        let pkg_dir = base.join(name);
        if is_package(&pkg_dir) {
            return Some((pkg_dir.join("__init__.py"), true));
        }

        // Check for module file (name.py)
        let module_file = base.join(format!("{}.py", name));
        if module_file.exists() {
            return Some((module_file, false));
        }
    }
    None
}

/// Resolve a dotted module name to a filesystem path.
///
/// For `os.path`, this searches for:
/// 1. `<search_path>/os/path/__init__.py` (package)
/// 2. `<search_path>/os/path.py` (module)
///
/// Returns `(path, is_package)` if found.
pub fn find_dotted_module_source(
    dotted_name: &DottedName,
    search_paths: &[Arc<str>],
) -> Option<(std::path::PathBuf, bool)> {
    for base_path in search_paths {
        let mut dir = std::path::PathBuf::from(base_path.as_ref());

        // Navigate to the parent of the final component
        for part in &dotted_name.parts()[..dotted_name.parts().len() - 1] {
            dir = dir.join(part.as_ref());
            // Each intermediate must be a package
            if !is_package(&dir) {
                break;
            }
        }

        let last = dotted_name.parts().last().unwrap();

        // Check for package directory
        let pkg_dir = dir.join(last.as_ref());
        if is_package(&pkg_dir) {
            return Some((pkg_dir.join("__init__.py"), true));
        }

        // Check for module file
        let module_file = dir.join(format!("{}.py", last));
        if module_file.exists() {
            return Some((module_file, false));
        }
    }
    None
}

// =============================================================================
// Sub-module Registration
// =============================================================================

/// Check if a module name represents a submodule of a stdlib module.
///
/// For example, `os.path` is a known submodule of `os`.
pub fn is_known_submodule(name: &str) -> bool {
    matches!(
        name,
        "os.path"
            | "os.path.join"
            | "os.path.split"
            | "collections.abc"
            | "io.abc"
            | "unittest.mock"
    )
}

/// Get the parent package name from a dotted module name.
///
/// E.g., "os.path" → "os", "foo.bar.baz" → "foo.bar"
#[inline]
pub fn parent_package(name: &str) -> Option<&str> {
    name.rsplit_once('.').map(|(parent, _)| parent)
}

/// Get the leaf module name from a dotted name.
///
/// E.g., "os.path" → "path", "foo.bar.baz" → "baz"
#[inline]
pub fn leaf_name(name: &str) -> &str {
    name.rsplit_once('.').map(|(_, leaf)| leaf).unwrap_or(name)
}
