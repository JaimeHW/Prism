//! Environment variable handling with lazy loading.
//!
//! High-performance environment access with:
//! - Lazy loading (don't read until first access)
//! - Mutation tracking for putenv/unsetenv
//! - Zero-copy key lookup using borrowed strings
//! - Thread-safe access via atomic operations

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

// =============================================================================
// Environ
// =============================================================================

/// Lazy-loading environment variable dictionary.
///
/// Environment variables are not loaded until first access, avoiding
/// startup overhead for programs that don't use environ.
pub struct Environ {
    /// Whether the environment has been loaded.
    loaded: AtomicBool,
    /// Mutex for synchronizing load operation (allows reload unlike Once).
    load_mutex: Mutex<()>,
    /// Cached environment variables.
    /// Uses Arc<str> for zero-copy sharing.
    vars: std::cell::UnsafeCell<HashMap<Arc<str>, Arc<str>>>,
}

// Custom Debug impl to avoid issues with Mutex
impl std::fmt::Debug for Environ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Environ")
            .field("loaded", &self.loaded)
            .finish_non_exhaustive()
    }
}

// SAFETY: Environ is thread-safe because:
// 1. loaded is AtomicBool for fast-path check
// 2. load_mutex protects the actual load operation
// 3. vars is only mutated while holding load_mutex with loaded=false
// 4. After initialization, vars is only read
unsafe impl Sync for Environ {}
unsafe impl Send for Environ {}

impl Environ {
    /// Create a new lazy environ.
    #[inline]
    pub fn new() -> Self {
        Self {
            loaded: AtomicBool::new(false),
            load_mutex: Mutex::new(()),
            vars: std::cell::UnsafeCell::new(HashMap::new()),
        }
    }

    /// Ensure environment is loaded.
    #[inline]
    fn ensure_loaded(&self) {
        if !self.loaded.load(Ordering::Acquire) {
            self.do_load();
        }
    }

    /// Actually load the environment (cold path).
    #[cold]
    fn do_load(&self) {
        // Take the mutex to synchronize loading
        let _guard = self.load_mutex.lock().unwrap_or_else(|p| p.into_inner());

        // Double-check after acquiring lock
        if self.loaded.load(Ordering::Acquire) {
            return;
        }

        // SAFETY: We're holding the lock and loaded is false, guaranteeing exclusive access
        let vars = unsafe { &mut *self.vars.get() };

        for (key, value) in std::env::vars() {
            vars.insert(Arc::from(key.as_str()), Arc::from(value.as_str()));
        }

        self.loaded.store(true, Ordering::Release);
    }

    /// Get an environment variable.
    ///
    /// Returns `None` if the variable is not set.
    #[inline]
    pub fn get(&self, key: &str) -> Option<Arc<str>> {
        self.ensure_loaded();
        // SAFETY: After ensure_loaded, vars is immutable
        let vars = unsafe { &*self.vars.get() };
        vars.get(key).cloned()
    }

    /// Get an environment variable with a default.
    #[inline]
    pub fn get_or(&self, key: &str, default: &str) -> Arc<str> {
        self.get(key).unwrap_or_else(|| Arc::from(default))
    }

    /// Check if an environment variable is set.
    #[inline]
    pub fn contains(&self, key: &str) -> bool {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.contains_key(key)
    }

    /// Set an environment variable.
    ///
    /// This also updates the actual process environment.
    pub fn set(&mut self, key: &str, value: &str) {
        self.ensure_loaded();

        // Update process environment
        // SAFETY: We're the only writer to this key in this code path
        unsafe { std::env::set_var(key, value) };

        // Update cache
        let vars = self.vars.get_mut();
        vars.insert(Arc::from(key), Arc::from(value));
    }

    /// Remove an environment variable.
    ///
    /// This also removes it from the actual process environment.
    pub fn remove(&mut self, key: &str) -> Option<Arc<str>> {
        self.ensure_loaded();

        // Remove from process environment
        // SAFETY: We're the only writer to this key in this code path
        unsafe { std::env::remove_var(key) };

        // Remove from cache
        let vars = self.vars.get_mut();
        vars.remove(key)
    }

    /// Get the number of environment variables.
    #[inline]
    pub fn len(&self) -> usize {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.len()
    }

    /// Check if the environment is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get all environment variable keys.
    pub fn keys(&self) -> Vec<Arc<str>> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.keys().cloned().collect()
    }

    /// Get all environment variable values.
    pub fn values(&self) -> Vec<Arc<str>> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.values().cloned().collect()
    }

    /// Iterate over all environment variables.
    pub fn iter(&self) -> impl Iterator<Item = (&Arc<str>, &Arc<str>)> {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        vars.iter()
    }

    /// Clear all environment variables from the cache.
    ///
    /// Note: This does NOT clear the actual process environment.
    pub fn clear_cache(&mut self) {
        let vars = self.vars.get_mut();
        vars.clear();
        self.loaded.store(false, Ordering::Release);
    }

    /// Force reload from process environment.
    pub fn reload(&mut self) {
        self.clear_cache();
        self.ensure_loaded();
    }
}

impl Default for Environ {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Environ {
    fn clone(&self) -> Self {
        self.ensure_loaded();
        let vars = unsafe { &*self.vars.get() };
        let mut new_environ = Self::new();
        *new_environ.vars.get_mut() = vars.clone();
        new_environ.loaded.store(true, Ordering::Release);
        new_environ
    }
}

// =============================================================================
// Standalone Functions
// =============================================================================

/// Get an environment variable from the process (not cached).
///
/// This always reads from the actual environment.
#[inline]
pub fn getenv(key: &str) -> Option<String> {
    std::env::var(key).ok()
}

/// Get an environment variable with a default.
#[inline]
pub fn getenv_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Set an environment variable in the process.
///
/// # Safety
/// This function is safe to call as long as the environment variable
/// is not being concurrently accessed by other threads.
#[inline]
pub fn putenv(key: &str, value: &str) {
    // SAFETY: Caller ensures thread-safe access
    unsafe { std::env::set_var(key, value) };
}

/// Remove an environment variable from the process.
///
/// # Safety
/// This function is safe to call as long as the environment variable
/// is not being concurrently accessed by other threads.
#[inline]
pub fn unsetenv(key: &str) {
    // SAFETY: Caller ensures thread-safe access
    unsafe { std::env::remove_var(key) };
}
