//! Python `os` module implementation.
//!
//! High-performance implementation of Python's os module providing:
//! - Platform-specific constants (name, sep, pathsep)
//! - Environment variable access with lazy loading
//! - Current directory operations with thread-local caching
//! - Path query operations with branch-free stat
//! - File system operations (mkdir, rmdir, remove, rename)
//! - Process information (getpid, getppid)
//!
//! # Performance Characteristics
//!
//! - Zero heap allocation for path operations (stack buffers)
//! - Thread-local cached current directory
//! - Lazy environment variable loading
//! - Direct OS syscalls without abstraction overhead

mod constants;
mod cwd;
mod environ;
mod error;
mod file;
pub mod path;
mod process;

pub use constants::*;
pub use cwd::*;
pub use environ::*;
pub use error::*;
pub use file::*;
pub use process::*;

use super::{Module, ModuleError};
use prism_core::Value;
use std::sync::Arc;

/// The os module providing operating system interface.
pub struct OsModule {
    /// Cached environment (lazy-loaded)
    environ: Environ,
}

impl OsModule {
    /// Create a new os module.
    #[inline]
    pub fn new() -> Self {
        Self {
            environ: Environ::new(),
        }
    }

    /// Get the environment dictionary.
    #[inline]
    pub fn environ(&self) -> &Environ {
        &self.environ
    }

    /// Get mutable environment dictionary.
    #[inline]
    pub fn environ_mut(&mut self) -> &mut Environ {
        &mut self.environ
    }
}

impl Default for OsModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for OsModule {
    fn name(&self) -> &str {
        "os"
    }

    fn get_attr(&self, name: &str) -> Result<Value, ModuleError> {
        match name {
            // Platform constants
            "name" => Ok(Value::none()), // TODO: Return interned "nt" or "posix"
            "sep" => Ok(Value::none()),  // TODO: Return interned "/" or "\\"
            "pathsep" => Ok(Value::none()), // TODO: Return interned ":" or ";"
            "linesep" => Ok(Value::none()), // TODO: Return interned "\n" or "\r\n"
            "curdir" => Ok(Value::none()), // "."
            "pardir" => Ok(Value::none()), // ".."
            "extsep" => Ok(Value::none()), // "."
            "altsep" => Ok(Value::none()), // "/" on Windows, None on Unix
            "devnull" => Ok(Value::none()), // "/dev/null" or "nul"

            // Functions (return None placeholders for now)
            "getcwd" | "chdir" | "mkdir" | "makedirs" | "rmdir" | "removedirs" | "remove"
            | "unlink" | "rename" | "replace" | "stat" | "lstat" | "listdir" | "scandir"
            | "walk" | "fwalk" | "getenv" | "putenv" | "unsetenv" | "getpid" | "getppid"
            | "kill" | "system" | "popen" | "access" | "chmod" | "chown" | "link" | "symlink"
            | "readlink" | "urandom" => {
                Ok(Value::none()) // Placeholder for callable
            }

            // Submodule
            "path" => Ok(Value::none()), // TODO: Return os.path module

            // Environ dict
            "environ" => Ok(Value::none()), // TODO: Return environ dict

            // O_* flags as integers
            "O_RDONLY" => Ok(Value::int(O_RDONLY as i64).unwrap()),
            "O_WRONLY" => Ok(Value::int(O_WRONLY as i64).unwrap()),
            "O_RDWR" => Ok(Value::int(O_RDWR as i64).unwrap()),
            "O_CREAT" => Ok(Value::int(O_CREAT as i64).unwrap()),
            "O_TRUNC" => Ok(Value::int(O_TRUNC as i64).unwrap()),
            "O_APPEND" => Ok(Value::int(O_APPEND as i64).unwrap()),
            "O_EXCL" => Ok(Value::int(O_EXCL as i64).unwrap()),

            _ => Err(ModuleError::AttributeError(format!(
                "module 'os' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        vec![
            // Constants
            Arc::from("name"),
            Arc::from("sep"),
            Arc::from("pathsep"),
            Arc::from("linesep"),
            Arc::from("curdir"),
            Arc::from("pardir"),
            Arc::from("extsep"),
            Arc::from("altsep"),
            Arc::from("devnull"),
            // Submodule
            Arc::from("path"),
            // Environ
            Arc::from("environ"),
            // Directory operations
            Arc::from("getcwd"),
            Arc::from("chdir"),
            Arc::from("mkdir"),
            Arc::from("makedirs"),
            Arc::from("rmdir"),
            Arc::from("removedirs"),
            Arc::from("listdir"),
            Arc::from("scandir"),
            Arc::from("walk"),
            // File operations
            Arc::from("remove"),
            Arc::from("unlink"),
            Arc::from("rename"),
            Arc::from("replace"),
            Arc::from("stat"),
            Arc::from("lstat"),
            Arc::from("access"),
            Arc::from("chmod"),
            Arc::from("link"),
            Arc::from("symlink"),
            Arc::from("readlink"),
            // Process
            Arc::from("getpid"),
            Arc::from("getppid"),
            Arc::from("kill"),
            Arc::from("system"),
            Arc::from("popen"),
            // Environment
            Arc::from("getenv"),
            Arc::from("putenv"),
            Arc::from("unsetenv"),
            // Misc
            Arc::from("urandom"),
            // Flags
            Arc::from("O_RDONLY"),
            Arc::from("O_WRONLY"),
            Arc::from("O_RDWR"),
            Arc::from("O_CREAT"),
            Arc::from("O_TRUNC"),
            Arc::from("O_APPEND"),
            Arc::from("O_EXCL"),
        ]
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Module Creation Tests
    // =========================================================================

    #[test]
    fn test_os_module_new() {
        let os = OsModule::new();
        assert_eq!(os.name(), "os");
    }

    #[test]
    fn test_os_module_default() {
        let os = OsModule::default();
        assert_eq!(os.name(), "os");
    }

    // =========================================================================
    // O_* Flag Tests
    // =========================================================================

    #[test]
    fn test_o_rdonly() {
        let os = OsModule::new();
        let val = os.get_attr("O_RDONLY").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_wronly() {
        let os = OsModule::new();
        let val = os.get_attr("O_WRONLY").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_rdwr() {
        let os = OsModule::new();
        let val = os.get_attr("O_RDWR").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_creat() {
        let os = OsModule::new();
        let val = os.get_attr("O_CREAT").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_trunc() {
        let os = OsModule::new();
        let val = os.get_attr("O_TRUNC").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_append() {
        let os = OsModule::new();
        let val = os.get_attr("O_APPEND").unwrap();
        assert!(val.as_int().is_some());
    }

    #[test]
    fn test_o_excl() {
        let os = OsModule::new();
        let val = os.get_attr("O_EXCL").unwrap();
        assert!(val.as_int().is_some());
    }

    // =========================================================================
    // Placeholder Attribute Tests
    // =========================================================================

    #[test]
    fn test_name_placeholder() {
        let os = OsModule::new();
        let name = os.get_attr("name").unwrap();
        assert!(name.is_none()); // Placeholder
    }

    #[test]
    fn test_sep_placeholder() {
        let os = OsModule::new();
        let sep = os.get_attr("sep").unwrap();
        assert!(sep.is_none());
    }

    #[test]
    fn test_getcwd_placeholder() {
        let os = OsModule::new();
        let getcwd = os.get_attr("getcwd").unwrap();
        assert!(getcwd.is_none());
    }

    #[test]
    fn test_path_placeholder() {
        let os = OsModule::new();
        let path = os.get_attr("path").unwrap();
        assert!(path.is_none());
    }

    #[test]
    fn test_environ_placeholder() {
        let os = OsModule::new();
        let environ = os.get_attr("environ").unwrap();
        assert!(environ.is_none());
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_unknown_attribute_error() {
        let os = OsModule::new();
        let result = os.get_attr("nonexistent");
        assert!(result.is_err());
        match result {
            Err(ModuleError::AttributeError(msg)) => {
                assert!(msg.contains("no attribute"));
            }
            _ => panic!("Expected AttributeError"),
        }
    }

    // =========================================================================
    // Dir Tests
    // =========================================================================

    #[test]
    fn test_dir_contains_name() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.contains(&Arc::from("name")));
    }

    #[test]
    fn test_dir_contains_getcwd() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.contains(&Arc::from("getcwd")));
    }

    #[test]
    fn test_dir_contains_path() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.contains(&Arc::from("path")));
    }

    #[test]
    fn test_dir_contains_environ() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.contains(&Arc::from("environ")));
    }

    #[test]
    fn test_dir_contains_flags() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.contains(&Arc::from("O_RDONLY")));
        assert!(attrs.contains(&Arc::from("O_WRONLY")));
        assert!(attrs.contains(&Arc::from("O_RDWR")));
    }

    #[test]
    fn test_dir_length() {
        let os = OsModule::new();
        let attrs = os.dir();
        assert!(attrs.len() >= 40); // Many attributes
    }

    // =========================================================================
    // Environ Access Tests
    // =========================================================================

    #[test]
    fn test_environ_access() {
        let os = OsModule::new();
        let _environ = os.environ();
    }

    #[test]
    fn test_environ_mut_access() {
        let mut os = OsModule::new();
        let _environ = os.environ_mut();
    }
}
