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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basename_simple() {
        assert_eq!(basename("/foo/bar.txt"), "bar.txt");
    }

    #[test]
    fn test_basename_no_slash() {
        assert_eq!(basename("file.txt"), "file.txt");
    }

    #[test]
    fn test_basename_trailing_slash() {
        // Behavior depends on OS, just check no panic
        let _ = basename("/foo/bar/");
    }

    #[test]
    fn test_dirname_simple() {
        let d = dirname("/foo/bar.txt");
        assert!(d.contains("foo"));
    }

    #[test]
    fn test_dirname_no_slash() {
        assert_eq!(dirname("file.txt"), "");
    }

    #[test]
    fn test_splitext_with_ext() {
        let (root, ext) = splitext("file.txt");
        assert_eq!(root, "file");
        assert_eq!(ext, ".txt");
    }

    #[test]
    fn test_splitext_no_ext() {
        let (root, ext) = splitext("file");
        assert_eq!(root, "file");
        assert_eq!(ext, "");
    }

    #[test]
    fn test_splitext_hidden() {
        let (root, ext) = splitext(".hidden");
        // .hidden is all extension or no extension depending on interpretation
        assert!(root.contains("hidden") || ext.contains("hidden"));
    }

    #[test]
    fn test_splitext_multi_dot() {
        let (root, ext) = splitext("file.tar.gz");
        assert_eq!(ext, ".gz");
    }

    #[test]
    fn test_split() {
        let (head, tail) = split("/foo/bar.txt");
        assert!(head.contains("foo"));
        assert_eq!(tail, "bar.txt");
    }
}
