use super::*;
use std::env;
use std::sync::Mutex;

/// Global lock to serialize tests that modify the process-wide current directory.
/// `std::env::set_current_dir` affects all threads, so tests must run sequentially.
static CWD_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_getcwd_returns_path() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    let cwd = getcwd().unwrap();
    assert!(!cwd.is_empty());
}

#[test]
fn test_getcwd_is_absolute() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    let cwd = getcwd().unwrap();
    assert!(Path::new(&*cwd).is_absolute());
}

#[test]
fn test_getcwd_caching() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    invalidate_cwd_cache();
    let cwd1 = getcwd().unwrap();
    let cwd2 = getcwd().unwrap();
    assert!(Arc::ptr_eq(&cwd1, &cwd2));
}

#[test]
fn test_chdir_and_back() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    let original = getcwd().unwrap();
    let temp = env::temp_dir();
    chdir(&temp).unwrap();
    chdir(&*original).unwrap();
    assert_eq!(&*getcwd().unwrap(), &*original);
}

#[test]
fn test_chdir_nonexistent() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    assert!(chdir("/nonexistent_12345").is_err());
}

#[test]
fn test_chdir_guard() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    let original = getcwd().unwrap();
    {
        let _g = ChdirGuard::new(env::temp_dir()).unwrap();
    }
    assert_eq!(&*getcwd().unwrap(), &*original);
}

#[test]
fn test_refresh_cache() {
    let _guard = CWD_TEST_LOCK.lock().unwrap();
    let cwd1 = getcwd().unwrap();
    let cwd2 = refresh_cwd_cache().unwrap();
    assert_eq!(&*cwd1, &*cwd2);
    assert!(!Arc::ptr_eq(&cwd1, &cwd2));
}
