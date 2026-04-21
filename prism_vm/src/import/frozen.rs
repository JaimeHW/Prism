use prism_code::CodeObject;
use std::sync::Arc;

/// Runtime-ready source module loaded from a frozen bundle.
#[derive(Debug, Clone)]
pub struct FrozenModuleSource {
    /// Compiled code object for the module body.
    pub code: Arc<CodeObject>,
    /// Source filename for diagnostics and `__file__`.
    pub filename: Arc<str>,
    /// Package context for relative imports.
    pub package_name: Arc<str>,
    /// Whether the module was produced from a package `__init__.py`.
    pub is_package: bool,
}

impl FrozenModuleSource {
    /// Construct a new frozen module descriptor.
    pub fn new(
        code: Arc<CodeObject>,
        filename: impl Into<Arc<str>>,
        package_name: impl Into<Arc<str>>,
        is_package: bool,
    ) -> Self {
        Self {
            code,
            filename: filename.into(),
            package_name: package_name.into(),
            is_package,
        }
    }
}
