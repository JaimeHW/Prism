//! Path split operations (basename, dirname, splitext).

use std::path::Path;

/// Get the base name (final component) of a path.
#[inline]
pub fn basename<P: AsRef<Path>>(path: P) -> String {
    path.as_ref()
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Get the directory name (all but final component) of a path.
#[inline]
pub fn dirname<P: AsRef<Path>>(path: P) -> String {
    path.as_ref()
        .parent()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Split path into (root, ext) where ext includes the dot.
pub fn splitext<P: AsRef<Path>>(path: P) -> (String, String) {
    let path = path.as_ref();
    let name = path
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();

    if let Some(dot_pos) = name.rfind('.') {
        if dot_pos > 0 {
            let parent = dirname(&path);
            let stem = &name[..dot_pos];
            let ext = &name[dot_pos..];
            let root = if parent.is_empty() {
                stem.to_string()
            } else {
                format!("{}{}{}", parent, std::path::MAIN_SEPARATOR, stem)
            };
            return (root, ext.to_string());
        }
    }
    (path.to_string_lossy().into_owned(), String::new())
}

/// Split path into (head, tail) where tail is final component.
pub fn split<P: AsRef<Path>>(path: P) -> (String, String) {
    (dirname(&path), basename(&path))
}

/// Split a path into (drive, tail) where drive is a drive letter spec
/// or UNC path prefix, and tail is everything else.
///
/// On Unix, drive is always empty.
///
/// Equivalent to Python's `os.path.splitdrive()`.
///
/// # Examples
///
/// ```ignore
/// // Windows
/// assert_eq!(splitdrive("C:\\foo\\bar"), ("C:".into(), "\\foo\\bar".into()));
/// // Unix
/// assert_eq!(splitdrive("/foo/bar"), ("".into(), "/foo/bar".into()));
/// ```
pub fn splitdrive<P: AsRef<Path>>(path: P) -> (String, String) {
    let s = path.as_ref().to_string_lossy();
    splitdrive_str(&s)
}

/// Internal string-based splitdrive implementation.
fn splitdrive_str(path: &str) -> (String, String) {
    #[cfg(windows)]
    {
        let bytes = path.as_bytes();

        // UNC path: \\server\share
        if bytes.len() >= 2
            && (bytes[0] == b'\\' || bytes[0] == b'/')
            && (bytes[1] == b'\\' || bytes[1] == b'/')
        {
            // Find the end of server\share
            let rest = &path[2..];
            if let Some(sep_pos) = rest.find(|c: char| c == '\\' || c == '/') {
                let after_server = &rest[sep_pos + 1..];
                if let Some(share_end) = after_server.find(|c: char| c == '\\' || c == '/') {
                    let drive_end = 2 + sep_pos + 1 + share_end;
                    return (path[..drive_end].to_string(), path[drive_end..].to_string());
                } else {
                    // \\server\share with no trailing path
                    return (path.to_string(), String::new());
                }
            }
            return (path.to_string(), String::new());
        }

        // Drive letter: X:
        if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
            return (path[..2].to_string(), path[2..].to_string());
        }

        // No drive
        (String::new(), path.to_string())
    }

    #[cfg(not(windows))]
    {
        // On Unix, there's no drive concept
        (String::new(), path.to_string())
    }
}
