use super::*;
use std::env;

#[test]
fn test_stat_temp_dir() {
    let s = stat(env::temp_dir()).unwrap();
    assert!(s.is_dir());
}

#[test]
fn test_stat_nonexistent() {
    assert!(stat("/nonexistent_12345").is_err());
}

#[test]
fn test_mkdir_rmdir() {
    let dir = env::temp_dir().join("_test_mkdir_12345");
    let _ = rmdir(&dir);
    mkdir(&dir).unwrap();
    assert!(stat(&dir).unwrap().is_dir());
    rmdir(&dir).unwrap();
}
