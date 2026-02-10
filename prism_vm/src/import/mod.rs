//! High-performance Python import system.
//!
//! This module provides the core import machinery for loading and caching
//! Python modules with optimal performance characteristics.
//!
//! # Architecture
//!
//! ```text
//! ImportResolver
//!   ├── sys.modules cache (RwLock<FxHashMap>)
//!   ├── StdlibRegistry (math, os, sys, etc.)
//!   ├── PackageSystem (dotted names, __init__.py, relative imports)
//!   └── SourceLoader (future: .py/.pyc loading)
//! ```
//!
//! # Usage
//!
//! The import resolver is accessed through the VirtualMachine:
//! ```ignore
//! let module = vm.import_resolver().import_module("math")?;
//! let value = vm.import_resolver().import_from(&module, "sqrt")?;
//! ```

pub mod module_object;
pub mod package;
pub mod resolver;

#[cfg(test)]
mod tests;

pub use module_object::ModuleObject;
pub use package::{DottedName, resolve_relative_import};
pub use resolver::{ImportError, ImportResolver};
