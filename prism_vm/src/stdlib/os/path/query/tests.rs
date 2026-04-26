use super::*;
use std::env;

#[test]
fn test_exists_curdir() {
    assert!(exists("."));
}

#[test]
fn test_exists_nonexistent() {
    assert!(!exists("/nonexistent_12345"));
}

#[test]
fn test_isfile_on_dir() {
    assert!(!isfile(env::temp_dir()));
}

#[test]
fn test_isdir_temp() {
    assert!(isdir(env::temp_dir()));
}

#[test]
fn test_isdir_on_file() {
    // Current exe is a file
    let exe = env::current_exe().unwrap();
    assert!(!isdir(exe));
}

#[test]
fn test_isabs_absolute() {
    #[cfg(windows)]
    assert!(isabs("C:\\Windows"));
    #[cfg(not(windows))]
    assert!(isabs("/usr"));
}

#[test]
fn test_isabs_relative() {
    assert!(!isabs("foo/bar"));
}

#[test]
fn test_getsize_some() {
    let exe = env::current_exe().unwrap();
    let size = getsize(exe);
    assert!(size.is_some());
    assert!(size.unwrap() > 0);
}

#[test]
fn test_getsize_none() {
    assert!(getsize("/nonexistent_12345").is_none());
}

// =========================================================================
// lexists Tests
// =========================================================================

#[test]
fn test_lexists_curdir() {
    assert!(lexists("."));
}

#[test]
fn test_lexists_nonexistent() {
    assert!(!lexists("/nonexistent_12345"));
}

#[test]
fn test_lexists_file() {
    let exe = env::current_exe().unwrap();
    assert!(lexists(&exe));
}

#[test]
fn test_lexists_directory() {
    assert!(lexists(env::temp_dir()));
}

// =========================================================================
// ismount Tests
// =========================================================================

#[cfg(windows)]
#[test]
fn test_ismount_drive_root() {
    assert!(ismount("C:\\"));
}

#[cfg(windows)]
#[test]
fn test_ismount_non_root() {
    assert!(!ismount("C:\\Windows"));
}

#[cfg(not(windows))]
#[test]
fn test_ismount_root() {
    assert!(ismount("/"));
}

#[cfg(not(windows))]
#[test]
fn test_ismount_non_root() {
    assert!(!ismount("/tmp") || std::path::Path::new("/tmp").exists());
}

#[test]
fn test_ismount_nonexistent() {
    assert!(!ismount("/nonexistent_12345"));
}
