//! Current working directory operations with thread-local caching.

use super::error::OsError;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;

thread_local! {
    static CACHED_CWD: RefCell<Option<Arc<str>>> = const { RefCell::new(None) };
}

/// Get the current working directory (cached).
#[inline]
pub fn getcwd() -> Result<Arc<str>, OsError> {
    CACHED_CWD.with(|cached| {
        let mut cached = cached.borrow_mut();
        if let Some(ref cwd) = *cached {
            return Ok(Arc::clone(cwd));
        }
        let cwd = getcwd_uncached()?;
        *cached = Some(Arc::clone(&cwd));
        Ok(cwd)
    })
}

/// Get the current working directory without cache.
pub fn getcwd_uncached() -> Result<Arc<str>, OsError> {
    std::env::current_dir()
        .map(|p| Arc::from(p.to_string_lossy().as_ref()))
        .map_err(|e| OsError::from_io_error(&e, "getcwd"))
}

/// Get the current working directory as PathBuf.
#[inline]
pub fn getcwd_pathbuf() -> Result<PathBuf, OsError> {
    getcwd().map(|s| PathBuf::from(&*s))
}

/// Change the current working directory.
pub fn chdir<P: AsRef<Path>>(path: P) -> Result<(), OsError> {
    let path = path.as_ref();
    std::env::set_current_dir(path)
        .map_err(|e| OsError::from_io_error(&e, path.to_string_lossy().as_ref()))?;
    invalidate_cwd_cache();
    Ok(())
}

/// Invalidate the CWD cache.
#[inline]
pub fn invalidate_cwd_cache() {
    CACHED_CWD.with(|cached| *cached.borrow_mut() = None);
}

/// Refresh the CWD cache.
#[inline]
pub fn refresh_cwd_cache() -> Result<Arc<str>, OsError> {
    invalidate_cwd_cache();
    getcwd()
}

/// RAII guard that restores original directory on drop.
#[derive(Debug)]
pub struct ChdirGuard {
    original: Arc<str>,
}

impl ChdirGuard {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, OsError> {
        let original = getcwd()?;
        chdir(path)?;
        Ok(Self { original })
    }

    #[inline]
    pub fn original(&self) -> &str {
        &self.original
    }
}

impl Drop for ChdirGuard {
    fn drop(&mut self) {
        let _ = chdir(&*self.original);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_getcwd_returns_path() {
        let cwd = getcwd().unwrap();
        assert!(!cwd.is_empty());
    }

    #[test]
    fn test_getcwd_is_absolute() {
        let cwd = getcwd().unwrap();
        assert!(Path::new(&*cwd).is_absolute());
    }

    #[test]
    fn test_getcwd_caching() {
        invalidate_cwd_cache();
        let cwd1 = getcwd().unwrap();
        let cwd2 = getcwd().unwrap();
        assert!(Arc::ptr_eq(&cwd1, &cwd2));
    }

    #[test]
    fn test_chdir_and_back() {
        let original = getcwd().unwrap();
        let temp = env::temp_dir();
        chdir(&temp).unwrap();
        chdir(&*original).unwrap();
        assert_eq!(&*getcwd().unwrap(), &*original);
    }

    #[test]
    fn test_chdir_nonexistent() {
        assert!(chdir("/nonexistent_12345").is_err());
    }

    #[test]
    fn test_chdir_guard() {
        let original = getcwd().unwrap();
        {
            let _g = ChdirGuard::new(env::temp_dir()).unwrap();
        }
        assert_eq!(&*getcwd().unwrap(), &*original);
    }

    #[test]
    fn test_refresh_cache() {
        let cwd1 = getcwd().unwrap();
        let cwd2 = refresh_cwd_cache().unwrap();
        assert_eq!(&*cwd1, &*cwd2);
        assert!(!Arc::ptr_eq(&cwd1, &cwd2));
    }
}
