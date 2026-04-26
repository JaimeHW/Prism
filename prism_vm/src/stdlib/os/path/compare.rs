//! Path comparison and relative path operations.
//!
//! High-performance implementations of Python's `os.path` comparison functions:
//! - `commonpath` — longest common sub-path of a sequence of paths
//! - `commonprefix` — longest common string prefix (character-level)
//! - `relpath` — compute relative path between two locations
//! - `samefile` — check if two paths refer to the same file
//!
//! # Performance
//!
//! - `commonpath` uses component-wise comparison, no string allocation until result
//! - `commonprefix` operates on raw bytes for maximum throughput
//! - `relpath` pre-computes component arrays to minimize allocations
//! - `samefile` uses a single `metadata()` call per path

use std::fs;
use std::path::{Component, Path, PathBuf};

/// Return the longest common sub-path of each pathname in `paths`.
///
/// Unlike `commonprefix`, this returns a valid path. All paths must be
/// either all absolute or all relative; a `ValueError`-equivalent error
/// is returned if they're mixed.
///
/// Equivalent to Python's `os.path.commonpath()`.
///
/// # Examples
///
/// ```ignore
/// assert_eq!(commonpath(&["/usr/lib", "/usr/local"]).unwrap(), "/usr");
/// ```
pub fn commonpath<P: AsRef<Path>>(paths: &[P]) -> Result<String, CommonPathError> {
    if paths.is_empty() {
        return Err(CommonPathError::Empty);
    }

    if paths.len() == 1 {
        return Ok(paths[0].as_ref().to_string_lossy().into_owned());
    }

    // Collect components for each path
    let component_lists: Vec<Vec<Component<'_>>> = paths
        .iter()
        .map(|p| p.as_ref().components().collect())
        .collect();

    // Check consistency: all absolute or all relative
    let first_is_absolute = paths[0].as_ref().is_absolute();
    for p in &paths[1..] {
        if p.as_ref().is_absolute() != first_is_absolute {
            return Err(CommonPathError::MixedAbsoluteRelative);
        }
    }

    // Find the minimum component count
    let min_len = component_lists.iter().map(|c| c.len()).min().unwrap_or(0);

    // Find the longest common prefix by component
    let mut common_len = 0;
    for i in 0..min_len {
        let first = &component_lists[0][i];
        if component_lists[1..].iter().all(|cl| &cl[i] == first) {
            common_len = i + 1;
        } else {
            break;
        }
    }

    if common_len == 0 {
        if first_is_absolute {
            // Absolute paths always share at least the root
            #[cfg(windows)]
            {
                // On Windows, check if they share a drive prefix
                let first_comps = &component_lists[0];
                if !first_comps.is_empty() {
                    if let Component::Prefix(_) = first_comps[0] {
                        // Check if all share the same prefix
                        if component_lists[1..]
                            .iter()
                            .all(|cl| !cl.is_empty() && cl[0] == first_comps[0])
                        {
                            let mut result = PathBuf::new();
                            result.push(first_comps[0].as_os_str());
                            if first_comps.len() > 1 {
                                if let Component::RootDir = first_comps[1] {
                                    if component_lists[1..]
                                        .iter()
                                        .all(|cl| cl.len() > 1 && cl[1] == first_comps[1])
                                    {
                                        result.push(std::path::MAIN_SEPARATOR.to_string());
                                    }
                                }
                            }
                            return Ok(result.to_string_lossy().into_owned());
                        }
                    }
                }
            }
            #[cfg(not(windows))]
            {
                return Ok("/".to_string());
            }
        }
        return Err(CommonPathError::NoCommonPath);
    }

    let mut result = PathBuf::new();
    for comp in &component_lists[0][..common_len] {
        result.push(comp.as_os_str());
    }
    Ok(result.to_string_lossy().into_owned())
}

/// Error type for `commonpath`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommonPathError {
    /// No paths provided.
    Empty,
    /// Mixed absolute and relative paths.
    MixedAbsoluteRelative,
    /// No common sub-path exists.
    NoCommonPath,
}

impl std::fmt::Display for CommonPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Empty => write!(f, "commonpath() arg is an empty sequence"),
            Self::MixedAbsoluteRelative => {
                write!(f, "Can't mix absolute and relative paths")
            }
            Self::NoCommonPath => write!(f, "Paths have no common sub-path"),
        }
    }
}

impl std::error::Error for CommonPathError {}

/// Return the longest path prefix (character by character) that is a
/// prefix of all paths in the list.
///
/// Unlike `commonpath`, this operates on raw strings and may return
/// an invalid path (e.g., splitting mid-component).
///
/// Equivalent to Python's `os.path.commonprefix()`.
///
/// # Performance
///
/// Operates on raw bytes — no path parsing, no allocations until result.
pub fn commonprefix(paths: &[&str]) -> String {
    if paths.is_empty() {
        return String::new();
    }
    if paths.len() == 1 {
        return paths[0].to_string();
    }

    let bytes: Vec<&[u8]> = paths.iter().map(|s| s.as_bytes()).collect();
    let min_len = bytes.iter().map(|b| b.len()).min().unwrap_or(0);

    let mut prefix_len = 0;
    for i in 0..min_len {
        let c = bytes[0][i];
        if bytes[1..].iter().all(|b| b[i] == c) {
            prefix_len = i + 1;
        } else {
            break;
        }
    }

    // SAFETY: We're slicing the original UTF-8 string at a position where
    // all strings agree, so the result is valid UTF-8 if the inputs are.
    paths[0][..prefix_len].to_string()
}

/// Return a relative filepath to `path` either from the current directory
/// or from an optional `start` directory.
///
/// Equivalent to Python's `os.path.relpath()`.
///
/// # Performance
///
/// Pre-computes component arrays. Uses a single allocation for the result.
pub fn relpath<P: AsRef<Path>, S: AsRef<Path>>(path: P, start: S) -> String {
    let path = path.as_ref();
    let start = start.as_ref();

    // Canonicalize both paths to absolute for comparison
    let abs_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let abs_start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(start)
    };

    let path_comps: Vec<Component<'_>> = abs_path.components().collect();
    let start_comps: Vec<Component<'_>> = abs_start.components().collect();

    // Find common prefix length
    let mut common = 0;
    let min_len = path_comps.len().min(start_comps.len());
    for i in 0..min_len {
        if path_comps[i] == start_comps[i] {
            common = i + 1;
        } else {
            break;
        }
    }

    // Build relative path: go up from start to common ancestor, then down to path
    let ups = start_comps.len() - common;
    let mut result = PathBuf::new();
    for _ in 0..ups {
        result.push("..");
    }
    for comp in &path_comps[common..] {
        result.push(comp.as_os_str());
    }

    if result.as_os_str().is_empty() {
        ".".to_string()
    } else {
        result.to_string_lossy().into_owned()
    }
}

/// Check if two paths refer to the same file or directory.
///
/// Follows symlinks. Returns `false` if either path doesn't exist.
///
/// Equivalent to Python's `os.path.samefile()`.
///
/// # Platform behavior
///
/// - **Unix**: Compares (st_dev, st_ino) for identity
/// - **Windows**: Compares canonical paths (symlink-resolved)
pub fn samefile<P: AsRef<Path>, Q: AsRef<Path>>(path1: P, path2: Q) -> bool {
    samefile_impl(path1.as_ref(), path2.as_ref())
}

#[cfg(unix)]
fn samefile_impl(path1: &Path, path2: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let Ok(m1) = fs::metadata(path1) else {
        return false;
    };
    let Ok(m2) = fs::metadata(path2) else {
        return false;
    };
    m1.dev() == m2.dev() && m1.ino() == m2.ino()
}

#[cfg(windows)]
fn samefile_impl(path1: &Path, path2: &Path) -> bool {
    let Ok(c1) = fs::canonicalize(path1) else {
        return false;
    };
    let Ok(c2) = fs::canonicalize(path2) else {
        return false;
    };
    c1 == c2
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;
