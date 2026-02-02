//! Python `sys` module implementation.
//!
//! High-performance implementation of Python's sys module providing:
//! - Runtime version and platform information
//! - Command-line argument access
//! - Module search path management
//! - Standard stream access
//! - System limits and configuration

mod argv;
mod hooks;
mod internals;
mod limits;
mod paths;
mod runtime;
mod streams;

pub use argv::*;
pub use hooks::*;
pub use internals::*;
pub use limits::*;
pub use paths::*;
pub use runtime::*;
pub use streams::*;

use super::{Module, ModuleError};
use prism_core::Value;
use std::sync::Arc;

/// The sys module providing runtime system configuration.
pub struct SysModule {
    /// Cached platform
    platform: Platform,
    /// Cached executable path
    executable: Arc<str>,
    /// Command-line arguments
    argv: SysArgv,
    /// Module search paths
    path: SysPaths,
    /// Recursion limit
    recursion_limit: RecursionLimit,
}

impl SysModule {
    /// Create a new sys module with default configuration.
    #[inline]
    pub fn new() -> Self {
        Self {
            platform: Platform::detect(),
            executable: std::env::current_exe()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| String::new())
                .into(),
            argv: SysArgv::from_env(),
            path: SysPaths::default(),
            recursion_limit: RecursionLimit::new(),
        }
    }

    /// Create sys module with custom arguments (for testing).
    pub fn with_args(args: Vec<String>) -> Self {
        Self {
            platform: Platform::detect(),
            executable: std::env::current_exe()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| String::new())
                .into(),
            argv: SysArgv::new(args),
            path: SysPaths::default(),
            recursion_limit: RecursionLimit::new(),
        }
    }

    /// Get command-line arguments.
    #[inline]
    pub fn argv(&self) -> &SysArgv {
        &self.argv
    }

    /// Get module search paths.
    #[inline]
    pub fn path(&self) -> &SysPaths {
        &self.path
    }

    /// Get mutable module search paths.
    #[inline]
    pub fn path_mut(&mut self) -> &mut SysPaths {
        &mut self.path
    }

    /// Get the recursion limit.
    #[inline]
    pub fn getrecursionlimit(&self) -> u32 {
        self.recursion_limit.get()
    }

    /// Set the recursion limit.
    #[inline]
    pub fn setrecursionlimit(&mut self, limit: u32) -> Result<(), ModuleError> {
        self.recursion_limit.set(limit)
    }

    /// Get the platform.
    #[inline]
    pub fn platform(&self) -> Platform {
        self.platform
    }

    /// Get the version string.
    #[inline]
    pub fn version(&self) -> &'static str {
        VERSION_STRING
    }
}

impl Default for SysModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SysModule {
    fn name(&self) -> &str {
        "sys"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            // Version information (return ints for now, strings need InternedString)
            "hexversion" => Ok(Value::int(HEXVERSION as i64).unwrap()),
            "api_version" => Ok(Value::int(API_VERSION as i64).unwrap()),

            // Limits
            "maxsize" => Ok(Value::int(MAX_SIZE).unwrap()),
            "maxunicode" => Ok(Value::int(MAX_UNICODE as i64).unwrap()),

            // Recursion limit as int
            "recursion_limit" => Ok(Value::int(self.recursion_limit.get() as i64).unwrap()),

            // Functions - return None placeholder (would be callable)
            "exit" | "getrecursionlimit" | "setrecursionlimit" | "getsizeof" | "getrefcount"
            | "intern" => {
                Ok(Value::none()) // Placeholder for callable
            }

            // String attributes - return None for now (need InternedString infrastructure)
            "version" | "platform" | "executable" | "byteorder" | "copyright" => {
                Ok(Value::none()) // TODO: Return interned strings
            }

            // Tuples/objects - return None placeholder
            "version_info" | "implementation" | "float_info" | "int_info" | "hash_info" => {
                Ok(Value::none()) // TODO: Return tuple/namespace objects
            }

            // Streams - return None placeholder (would be file objects)
            "stdin" | "stdout" | "stderr" | "__stdin__" | "__stdout__" | "__stderr__" => {
                Ok(Value::none())
            }

            // Hooks - return None placeholder
            "displayhook" | "excepthook" => Ok(Value::none()),

            // Path list - return None placeholder (would be list)
            "path" | "argv" => {
                Ok(Value::none()) // TODO: Return list objects
            }

            _ => Err(ModuleError::AttributeError(format!(
                "module 'sys' has no attribute '{}'",
                name
            ))),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // SysModule Creation Tests
    // =========================================================================

    #[test]
    fn test_sys_module_new() {
        let sys = SysModule::new();
        assert_eq!(sys.name(), "sys");
    }

    #[test]
    fn test_sys_module_with_args() {
        let sys = SysModule::with_args(vec!["test.py".to_string(), "--flag".to_string()]);
        assert_eq!(sys.argv().len(), 2);
    }

    #[test]
    fn test_sys_module_default() {
        let sys = SysModule::default();
        assert_eq!(sys.name(), "sys");
    }

    // =========================================================================
    // Integer Attribute Tests
    // =========================================================================

    #[test]
    fn test_hexversion_attribute() {
        let sys = SysModule::new();
        let hexversion = sys.get_attr("hexversion").unwrap();
        let v = hexversion.as_int().unwrap();
        // Python 3.12.0 = 0x030C00F0
        assert!(v > 0);
        assert_eq!(v, HEXVERSION as i64);
    }

    #[test]
    fn test_maxsize_attribute() {
        let sys = SysModule::new();
        let maxsize = sys.get_attr("maxsize").unwrap();
        let m = maxsize.as_int().unwrap();
        // maxsize should be positive and equal to SMALL_INT_MAX
        assert!(m > 0);
        assert_eq!(m, MAX_SIZE);
    }

    #[test]
    fn test_maxunicode_attribute() {
        let sys = SysModule::new();
        let maxunicode = sys.get_attr("maxunicode").unwrap();
        let m = maxunicode.as_int().unwrap();
        assert_eq!(m, 0x10FFFF);
    }

    #[test]
    fn test_api_version_attribute() {
        let sys = SysModule::new();
        let api = sys.get_attr("api_version").unwrap();
        let a = api.as_int().unwrap();
        assert!(a > 0);
    }

    // =========================================================================
    // Recursion Limit Tests
    // =========================================================================

    #[test]
    fn test_getrecursionlimit() {
        let sys = SysModule::new();
        let limit = sys.getrecursionlimit();
        assert_eq!(limit, DEFAULT_RECURSION_LIMIT);
    }

    #[test]
    fn test_setrecursionlimit() {
        let mut sys = SysModule::new();
        sys.setrecursionlimit(2000).unwrap();
        assert_eq!(sys.getrecursionlimit(), 2000);
    }

    #[test]
    fn test_setrecursionlimit_too_low() {
        let mut sys = SysModule::new();
        let result = sys.setrecursionlimit(1);
        assert!(result.is_err());
    }

    #[test]
    fn test_recursion_limit_attribute() {
        let sys = SysModule::new();
        let limit = sys.get_attr("recursion_limit").unwrap();
        assert_eq!(limit.as_int().unwrap(), DEFAULT_RECURSION_LIMIT as i64);
    }

    // =========================================================================
    // Placeholder Attribute Tests
    // =========================================================================

    #[test]
    fn test_stdin_attribute() {
        let sys = SysModule::new();
        let stdin = sys.get_attr("stdin").unwrap();
        assert!(stdin.is_none());
    }

    #[test]
    fn test_stdout_attribute() {
        let sys = SysModule::new();
        let stdout = sys.get_attr("stdout").unwrap();
        assert!(stdout.is_none());
    }

    #[test]
    fn test_stderr_attribute() {
        let sys = SysModule::new();
        let stderr = sys.get_attr("stderr").unwrap();
        assert!(stderr.is_none());
    }

    #[test]
    fn test_version_placeholder() {
        let sys = SysModule::new();
        let version = sys.get_attr("version").unwrap();
        assert!(version.is_none()); // Placeholder until InternedString
    }

    #[test]
    fn test_platform_placeholder() {
        let sys = SysModule::new();
        let platform = sys.get_attr("platform").unwrap();
        assert!(platform.is_none()); // Placeholder
    }

    #[test]
    fn test_exit_placeholder() {
        let sys = SysModule::new();
        let exit = sys.get_attr("exit").unwrap();
        assert!(exit.is_none()); // Placeholder for callable
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_unknown_attribute_error() {
        let sys = SysModule::new();
        let result = sys.get_attr("unknown_attr");
        assert!(result.is_err());
        match result {
            Err(ModuleError::AttributeError(msg)) => {
                assert!(msg.contains("no attribute"));
            }
            _ => panic!("Expected AttributeError"),
        }
    }

    // =========================================================================
    // Platform and Version Tests (Direct Access)
    // =========================================================================

    #[test]
    fn test_platform_direct() {
        let sys = SysModule::new();
        let platform = sys.platform();
        assert!(matches!(
            platform,
            Platform::Windows | Platform::Linux | Platform::MacOS | Platform::FreeBSD
        ));
    }

    #[test]
    fn test_version_direct() {
        let sys = SysModule::new();
        let version = sys.version();
        assert!(version.contains("3.12"));
    }
}
