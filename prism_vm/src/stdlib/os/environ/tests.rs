use super::*;
use std::sync::{Mutex, MutexGuard};

/// Global lock to serialize ALL tests that access the process environment.
/// This is needed because:
/// 1. `std::env::set_var` and `std::env::remove_var` are unsafe for concurrent access
/// 2. `std::env::vars()` iterator can be invalidated by concurrent modifications
/// 3. `Environ::do_load` iterates `std::env::vars()` without internal synchronization
///
/// ALL tests that modify env vars OR create Environ instances MUST hold this lock.
static ENV_TEST_LOCK: Mutex<()> = Mutex::new(());

/// Acquire the environment test lock. Hold this for the entire test duration.
/// Recovers from poisoned lock to prevent cascade failures.
fn lock_env() -> MutexGuard<'static, ()> {
    ENV_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Set an env var. Caller MUST hold ENV_TEST_LOCK via lock_env().
fn set_var_locked(_guard: &MutexGuard<'static, ()>, key: &str, value: &str) {
    unsafe { std::env::set_var(key, value) };
}

/// Remove an env var. Caller MUST hold ENV_TEST_LOCK via lock_env().
fn remove_var_locked(_guard: &MutexGuard<'static, ()>, key: &str) {
    unsafe { std::env::remove_var(key) };
}

// Backward compatible wrappers - these are ONLY safe if test_set_var and
// test_remove_var calls happen atomically within a test that holds no Environ
// instances during modifications. For proper safety, use lock_env() +
// set_var_locked/remove_var_locked.
//
// NOTE: These acquire the lock per-call, so parallel tests that interleave
// env modifications with Environ::new() access can still race on the
// std::env::vars() iterator. Tests should prefer the locked variants.
#[allow(dead_code)]
fn test_set_var(key: &str, value: &str) {
    let _guard = ENV_TEST_LOCK.lock().unwrap();
    unsafe { std::env::set_var(key, value) };
}

#[allow(dead_code)]
fn test_remove_var(key: &str) {
    let _guard = ENV_TEST_LOCK.lock().unwrap();
    unsafe { std::env::remove_var(key) };
}

// =========================================================================
// Environ Creation Tests
// =========================================================================

#[test]
fn test_environ_new() {
    let _guard = lock_env();
    let env = Environ::new();
    // Should not be loaded yet
    assert!(!env.loaded.load(Ordering::Relaxed));
}

// =========================================================================
// Lazy Loading Tests
// =========================================================================

#[test]
fn test_environ_lazy_load_on_get() {
    let _guard = lock_env();
    let env = Environ::new();
    assert!(!env.loaded.load(Ordering::Relaxed));
    let _ = env.get("PATH");
    assert!(env.loaded.load(Ordering::Relaxed));
}

#[test]
fn test_environ_lazy_load_on_contains() {
    let _guard = lock_env();
    let env = Environ::new();
    let _ = env.contains("PATH");
    assert!(env.loaded.load(Ordering::Relaxed));
}

#[test]
fn test_environ_lazy_load_on_len() {
    let _guard = lock_env();
    let env = Environ::new();
    let _ = env.len();
    assert!(env.loaded.load(Ordering::Relaxed));
}

// =========================================================================
// Get Tests
// =========================================================================

#[test]
fn test_environ_get_existing() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_GET_EXISTING__", "test_value");
    let env = Environ::new();
    let result = env.get("__TEST_GET_EXISTING__");
    assert!(result.is_some());
    assert_eq!(&*result.unwrap(), "test_value");
    remove_var_locked(&guard, "__TEST_GET_EXISTING__");
}

#[test]
fn test_environ_get_nonexistent() {
    let _guard = lock_env();
    let env = Environ::new();
    let result = env.get("__NONEXISTENT_VAR_12345__");
    assert!(result.is_none());
}

#[test]
fn test_environ_get_or_existing() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_GET_OR__", "test_value");
    let env = Environ::new();
    let result = env.get_or("__TEST_GET_OR__", "default");
    assert_eq!(&*result, "test_value");
    remove_var_locked(&guard, "__TEST_GET_OR__");
}

#[test]
fn test_environ_get_or_nonexistent() {
    let _guard = lock_env();
    let env = Environ::new();
    let result = env.get_or("__NONEXISTENT_VAR_12345__", "default");
    assert_eq!(&*result, "default");
}

// =========================================================================
// Contains Tests
// =========================================================================

#[test]
fn test_environ_contains_existing() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_CONTAINS__", "value");
    let env = Environ::new();
    assert!(env.contains("__TEST_CONTAINS__"));
    remove_var_locked(&guard, "__TEST_CONTAINS__");
}

#[test]
fn test_environ_contains_nonexistent() {
    let _guard = lock_env();
    let env = Environ::new();
    assert!(!env.contains("__NONEXISTENT_VAR_12345__"));
}

// =========================================================================
// Set/Remove Tests
// =========================================================================

#[test]
fn test_environ_set() {
    let guard = lock_env();
    let mut env = Environ::new();
    env.set("__TEST_SET_VAR__", "test_value");

    // Check cache
    assert_eq!(env.get("__TEST_SET_VAR__").as_deref(), Some("test_value"));

    // Check actual environment
    assert_eq!(
        std::env::var("__TEST_SET_VAR__").ok().as_deref(),
        Some("test_value")
    );

    // Cleanup
    remove_var_locked(&guard, "__TEST_SET_VAR__");
}

#[test]
fn test_environ_set_overwrite() {
    let guard = lock_env();
    let mut env = Environ::new();
    env.set("__TEST_SET_OVERWRITE__", "value1");
    env.set("__TEST_SET_OVERWRITE__", "value2");

    assert_eq!(env.get("__TEST_SET_OVERWRITE__").as_deref(), Some("value2"));

    remove_var_locked(&guard, "__TEST_SET_OVERWRITE__");
}

#[test]
fn test_environ_remove() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_REMOVE__", "value");
    let mut env = Environ::new();

    let removed = env.remove("__TEST_REMOVE__");
    assert_eq!(removed.as_deref(), Some("value"));

    // Check it's gone from cache
    assert!(env.get("__TEST_REMOVE__").is_none());

    // Check it's gone from actual environment
    assert!(std::env::var("__TEST_REMOVE__").is_err());
}

#[test]
fn test_environ_remove_nonexistent() {
    let _guard = lock_env();
    let mut env = Environ::new();
    let removed = env.remove("__NONEXISTENT_VAR_12345__");
    assert!(removed.is_none());
}

// =========================================================================
// Size Tests
// =========================================================================

#[test]
fn test_environ_len_nonzero() {
    let _guard = lock_env();
    let env = Environ::new();
    // Most systems have at least some environment variables
    assert!(env.len() > 0);
}

#[test]
fn test_environ_is_empty() {
    let _guard = lock_env();
    let env = Environ::new();
    // Most systems have environment variables
    assert!(!env.is_empty());
}

// =========================================================================
// Keys/Values Tests
// =========================================================================

#[test]
fn test_environ_keys() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_KEYS__", "value");
    let env = Environ::new();
    let keys = env.keys();
    assert!(keys.iter().any(|k| &**k == "__TEST_KEYS__"));
    remove_var_locked(&guard, "__TEST_KEYS__");
}

#[test]
fn test_environ_values() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_VALUES__", "unique_test_value_12345");
    let env = Environ::new();
    let values = env.values();
    assert!(values.iter().any(|v| &**v == "unique_test_value_12345"));
    remove_var_locked(&guard, "__TEST_VALUES__");
}

// =========================================================================
// Iterator Tests
// =========================================================================

#[test]
fn test_environ_iter() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_ITER__", "iter_value");
    let env = Environ::new();
    let found = env
        .iter()
        .any(|(k, v)| &**k == "__TEST_ITER__" && &**v == "iter_value");
    assert!(found);
    remove_var_locked(&guard, "__TEST_ITER__");
}

// =========================================================================
// Clear/Reload Tests
// =========================================================================

#[test]
fn test_environ_clear_cache() {
    let _guard = lock_env();
    let mut env = Environ::new();
    let _ = env.len(); // Force load
    assert!(env.loaded.load(Ordering::Relaxed));

    env.clear_cache();
    assert!(!env.loaded.load(Ordering::Relaxed));
}

#[test]
fn test_environ_reload() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_RELOAD__", "value1");
    let mut env = Environ::new();
    let _ = env.get("__TEST_RELOAD__");

    // Change actual environment
    set_var_locked(&guard, "__TEST_RELOAD__", "value2");

    // Cache should still have old value
    // (Note: our cache updates on set, so we need to change it externally)

    // Reload
    env.reload();

    // Should now have new value
    assert_eq!(env.get("__TEST_RELOAD__").as_deref(), Some("value2"));

    remove_var_locked(&guard, "__TEST_RELOAD__");
}

// =========================================================================
// Clone Tests
// =========================================================================

#[test]
fn test_environ_clone() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_CLONE__", "clone_value");
    let env = Environ::new();
    let _ = env.get("__TEST_CLONE__");

    let cloned = env.clone();
    assert_eq!(cloned.get("__TEST_CLONE__").as_deref(), Some("clone_value"));

    remove_var_locked(&guard, "__TEST_CLONE__");
}

// =========================================================================
// Standalone Function Tests
// =========================================================================

#[test]
fn test_getenv_existing() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_GETENV__", "value");
    assert_eq!(getenv("__TEST_GETENV__"), Some("value".to_string()));
    remove_var_locked(&guard, "__TEST_GETENV__");
}

#[test]
fn test_getenv_nonexistent() {
    assert!(getenv("__NONEXISTENT_VAR_12345__").is_none());
}

#[test]
fn test_getenv_or() {
    assert_eq!(getenv_or("__NONEXISTENT_VAR_12345__", "default"), "default");
}

#[test]
fn test_putenv() {
    let guard = lock_env();
    putenv("__TEST_PUTENV__", "value");
    assert_eq!(
        std::env::var("__TEST_PUTENV__").ok(),
        Some("value".to_string())
    );
    remove_var_locked(&guard, "__TEST_PUTENV__");
}

#[test]
fn test_unsetenv() {
    let guard = lock_env();
    set_var_locked(&guard, "__TEST_UNSETENV__", "value");
    unsetenv("__TEST_UNSETENV__");
    assert!(std::env::var("__TEST_UNSETENV__").is_err());
}

// =========================================================================
// Thread Safety Tests
// =========================================================================

#[test]
fn test_environ_concurrent_read() {
    use std::sync::Arc;
    use std::thread;

    let guard = lock_env();
    set_var_locked(&guard, "__TEST_CONCURRENT__", "value");
    let env = Arc::new(Environ::new());
    // Drop guard before spawning threads - we've set the env and loaded Environ
    // The threads only do reads, so no further modifications are needed
    drop(guard);

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let env = Arc::clone(&env);
            thread::spawn(move || {
                for _ in 0..100 {
                    let _ = env.get("__TEST_CONCURRENT__");
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    let guard = lock_env();
    remove_var_locked(&guard, "__TEST_CONCURRENT__");
}
