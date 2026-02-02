//! Path normalization operations (abspath, normpath, realpath).

use std::env;
use std::path::{Path, PathBuf};

/// Get absolute path.
pub fn abspath<P: AsRef<Path>>(path: P) -> String {
    let path = path.as_ref();
    if path.is_absolute() {
        path.to_string_lossy().into_owned()
    } else {
        env::current_dir()
            .map(|cwd| cwd.join(path))
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .into_owned()
    }
}

/// Normalize a path (remove . and .. components).
pub fn normpath<P: AsRef<Path>>(path: P) -> String {
    let path = path.as_ref();
    let mut components: Vec<&std::ffi::OsStr> = Vec::new();

    #[cfg(windows)]
    const ROOT_SEP: &str = "\\";
    #[cfg(not(windows))]
    const ROOT_SEP: &str = "/";

    for comp in path.components() {
        use std::path::Component;
        match comp {
            Component::CurDir => {} // Skip .
            Component::ParentDir => {
                if !components.is_empty() {
                    components.pop();
                }
            }
            Component::Normal(s) => components.push(s),
            Component::RootDir => {
                components.clear();
                components.push(std::ffi::OsStr::new(ROOT_SEP));
            }
            Component::Prefix(p) => {
                components.clear();
                components.push(p.as_os_str());
            }
        }
    }

    if components.is_empty() {
        ".".to_string()
    } else {
        let mut result = PathBuf::new();
        for c in components {
            result.push(c);
        }
        result.to_string_lossy().into_owned()
    }
}

/// Get canonical absolute path (resolving symlinks).
pub fn realpath<P: AsRef<Path>>(path: P) -> std::io::Result<String> {
    std::fs::canonicalize(path.as_ref()).map(|p| p.to_string_lossy().into_owned())
}

/// Expand ~ to home directory.
pub fn expanduser<P: AsRef<Path>>(path: P) -> String {
    let path_str = path.as_ref().to_string_lossy();
    if path_str.starts_with('~') {
        if let Some(home) = home_dir() {
            return path_str.replacen('~', &home, 1);
        }
    }
    path_str.into_owned()
}

/// Get home directory.
fn home_dir() -> Option<String> {
    #[cfg(windows)]
    {
        env::var("USERPROFILE").ok()
    }
    #[cfg(not(windows))]
    {
        env::var("HOME").ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abspath_absolute() {
        #[cfg(windows)]
        let p = abspath("C:\\Windows");
        #[cfg(not(windows))]
        let p = abspath("/usr");
        assert!(Path::new(&p).is_absolute());
    }

    #[test]
    fn test_abspath_relative() {
        let p = abspath("foo");
        assert!(Path::new(&p).is_absolute());
    }

    #[test]
    fn test_normpath_dot() {
        let p = normpath("./foo/./bar");
        assert!(!p.contains("./"));
    }

    #[test]
    fn test_normpath_dotdot() {
        let p = normpath("foo/bar/../baz");
        assert!(!p.contains(".."));
        assert!(p.contains("baz"));
    }

    #[test]
    fn test_normpath_empty() {
        let p = normpath("");
        assert_eq!(p, ".");
    }

    #[test]
    fn test_realpath_curdir() {
        let p = realpath(".").unwrap();
        assert!(Path::new(&p).is_absolute());
    }

    #[test]
    fn test_realpath_nonexistent() {
        assert!(realpath("/nonexistent_12345").is_err());
    }

    #[test]
    fn test_expanduser_no_tilde() {
        assert_eq!(expanduser("/foo/bar"), "/foo/bar");
    }

    #[test]
    fn test_expanduser_tilde() {
        let p = expanduser("~/foo");
        assert!(!p.starts_with('~') || home_dir().is_none());
    }
}
