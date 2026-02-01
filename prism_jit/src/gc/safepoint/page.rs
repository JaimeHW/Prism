//! Safepoint page management with mprotect-based arming.
//!
//! The safepoint page is a single 4KB memory page that JIT code polls.
//! When GC is needed, the page is protected (PROT_NONE), causing polls
//! to trap with SIGSEGV/SIGBUS.
//!
//! # Design
//!
//! - Cache-line aligned (64 bytes) for optimal memory access
//! - Atomic state machine for lock-free coordination
//! - Single TLB entry via 64KB region allocation
//!
//! # Usage
//!
//! ```ignore
//! let page = SafepointPage::new()?;
//!
//! // Load into R15 for JIT code
//! let addr = page.poll_address();
//!
//! // Arm for GC (makes polls trap)
//! page.arm()?;
//!
//! // ... GC runs ...
//!
//! // Disarm (polls succeed again)
//! page.disarm()?;
//! ```

use std::io;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(unix)]
use libc::{MAP_ANONYMOUS, MAP_PRIVATE, PROT_NONE, PROT_READ, c_void, mmap, mprotect, munmap};

#[cfg(windows)]
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, VirtualAlloc, VirtualFree,
    VirtualProtect,
};

use super::{SAFEPOINT_PAGE_SIZE, SAFEPOINT_REGION_SIZE};

// =============================================================================
// SafepointState
// =============================================================================

/// State of the safepoint page.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafepointState {
    /// Normal operation - page is readable, polls succeed.
    Enabled = 0,
    /// GC requested - page is protected, polls will trap.
    Armed = 1,
    /// GC in progress - threads are stopped at safepoints.
    Triggered = 2,
}

impl SafepointState {
    /// Convert from raw u32 value.
    #[inline]
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(SafepointState::Enabled),
            1 => Some(SafepointState::Armed),
            2 => Some(SafepointState::Triggered),
            _ => None,
        }
    }
}

// =============================================================================
// SafepointPage
// =============================================================================

/// A memory page used for safepoint polling.
///
/// The page is allocated with mmap/VirtualAlloc and its protection is
/// toggled to trigger safepoint traps.
#[repr(C, align(64))] // Cache-line aligned
pub struct SafepointPage {
    /// Base address of the 64KB region (page is at offset 0).
    region_base: NonNull<u8>,

    /// Current state of the safepoint.
    state: AtomicU32,

    /// Number of threads currently in a safepoint.
    threads_stopped: AtomicU32,

    /// OS page size (cached for performance).
    page_size: usize,

    /// Padding to fill cache line.
    _padding: [u8; 40],
}

// Safety: SafepointPage is designed for concurrent access.
unsafe impl Send for SafepointPage {}
unsafe impl Sync for SafepointPage {}

/// Error types for safepoint operations.
#[derive(Debug, Clone)]
pub enum SafepointError {
    /// Failed to allocate safepoint region.
    AllocationFailed(String),
    /// Failed to change page protection.
    ProtectionFailed(String),
    /// Invalid state transition.
    InvalidState(SafepointState, SafepointState),
}

impl std::fmt::Display for SafepointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SafepointError::AllocationFailed(msg) => {
                write!(f, "safepoint allocation failed: {}", msg)
            }
            SafepointError::ProtectionFailed(msg) => {
                write!(f, "safepoint protection change failed: {}", msg)
            }
            SafepointError::InvalidState(from, to) => {
                write!(
                    f,
                    "invalid safepoint state transition: {:?} -> {:?}",
                    from, to
                )
            }
        }
    }
}

impl std::error::Error for SafepointError {}

impl SafepointPage {
    /// Create a new safepoint page.
    ///
    /// Allocates a 64KB region with the safepoint page at offset 0.
    /// The remaining 60KB acts as a guard zone.
    pub fn new() -> Result<Self, SafepointError> {
        let page_size = Self::get_page_size();
        let region_base = Self::allocate_region()?;

        Ok(SafepointPage {
            region_base,
            state: AtomicU32::new(SafepointState::Enabled as u32),
            threads_stopped: AtomicU32::new(0),
            page_size,
            _padding: [0; 40],
        })
    }

    /// Get the address to poll in JIT code.
    ///
    /// This address should be loaded into R15 in the function prologue.
    #[inline]
    pub fn poll_address(&self) -> usize {
        self.region_base.as_ptr() as usize
    }

    /// Get the current state.
    #[inline]
    pub fn state(&self) -> SafepointState {
        let raw = self.state.load(Ordering::Acquire);
        SafepointState::from_u32(raw).unwrap_or(SafepointState::Enabled)
    }

    /// Check if the page is currently armed.
    #[inline]
    pub fn is_armed(&self) -> bool {
        self.state.load(Ordering::Acquire) != SafepointState::Enabled as u32
    }

    /// Arm the safepoint page for GC.
    ///
    /// Changes page protection to PROT_NONE, causing subsequent polls to trap.
    /// This is an atomic operation with Acquire-Release semantics.
    pub fn arm(&self) -> Result<(), SafepointError> {
        // Transition: Enabled -> Armed
        let result = self.state.compare_exchange(
            SafepointState::Enabled as u32,
            SafepointState::Armed as u32,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        if result.is_err() {
            let current =
                SafepointState::from_u32(result.unwrap_err()).unwrap_or(SafepointState::Enabled);
            return Err(SafepointError::InvalidState(current, SafepointState::Armed));
        }

        // Make page inaccessible
        self.protect_page(false)?;
        Ok(())
    }

    /// Disarm the safepoint page.
    ///
    /// Changes page protection back to PROT_READ, allowing polls to succeed.
    pub fn disarm(&self) -> Result<(), SafepointError> {
        // Make page readable first (before state transition)
        self.protect_page(true)?;

        // Transition: Armed/Triggered -> Enabled
        self.state
            .store(SafepointState::Enabled as u32, Ordering::Release);

        // Reset stopped count
        self.threads_stopped.store(0, Ordering::Release);

        Ok(())
    }

    /// Mark the safepoint as triggered (GC in progress).
    #[inline]
    pub fn mark_triggered(&self) {
        self.state
            .store(SafepointState::Triggered as u32, Ordering::Release);
    }

    /// Increment the count of stopped threads.
    #[inline]
    pub fn thread_stopped(&self) -> u32 {
        self.threads_stopped.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Get the number of stopped threads.
    #[inline]
    pub fn stopped_count(&self) -> u32 {
        self.threads_stopped.load(Ordering::Acquire)
    }

    /// Check if an address is within our safepoint page.
    #[inline]
    pub fn contains_address(&self, addr: usize) -> bool {
        let base = self.poll_address();
        addr >= base && addr < base + SAFEPOINT_PAGE_SIZE
    }

    // =========================================================================
    // Platform-specific implementation
    // =========================================================================

    #[cfg(unix)]
    fn allocate_region() -> Result<NonNull<u8>, SafepointError> {
        unsafe {
            let ptr = mmap(
                std::ptr::null_mut(),
                SAFEPOINT_REGION_SIZE,
                PROT_READ,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );

            if ptr == libc::MAP_FAILED {
                return Err(SafepointError::AllocationFailed(
                    io::Error::last_os_error().to_string(),
                ));
            }

            NonNull::new(ptr as *mut u8)
                .ok_or_else(|| SafepointError::AllocationFailed("mmap returned null".to_string()))
        }
    }

    #[cfg(windows)]
    fn allocate_region() -> Result<NonNull<u8>, SafepointError> {
        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                SAFEPOINT_REGION_SIZE,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READONLY,
            );

            if ptr.is_null() {
                return Err(SafepointError::AllocationFailed(
                    io::Error::last_os_error().to_string(),
                ));
            }

            NonNull::new(ptr as *mut u8).ok_or_else(|| {
                SafepointError::AllocationFailed("VirtualAlloc returned null".to_string())
            })
        }
    }

    #[cfg(unix)]
    fn protect_page(&self, readable: bool) -> Result<(), SafepointError> {
        unsafe {
            let prot = if readable { PROT_READ } else { PROT_NONE };
            let result = mprotect(
                self.region_base.as_ptr() as *mut c_void,
                self.page_size,
                prot,
            );

            if result != 0 {
                return Err(SafepointError::ProtectionFailed(
                    io::Error::last_os_error().to_string(),
                ));
            }
            Ok(())
        }
    }

    #[cfg(windows)]
    fn protect_page(&self, readable: bool) -> Result<(), SafepointError> {
        unsafe {
            let protect = if readable {
                PAGE_READONLY
            } else {
                PAGE_NOACCESS
            };
            let mut old_protect = 0;
            let result = VirtualProtect(
                self.region_base.as_ptr() as *mut _,
                self.page_size,
                protect,
                &mut old_protect,
            );

            if result == 0 {
                return Err(SafepointError::ProtectionFailed(
                    io::Error::last_os_error().to_string(),
                ));
            }
            Ok(())
        }
    }

    #[cfg(unix)]
    fn get_page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }

    #[cfg(windows)]
    fn get_page_size() -> usize {
        use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
        unsafe {
            let mut info: SYSTEM_INFO = std::mem::zeroed();
            GetSystemInfo(&mut info);
            info.dwPageSize as usize
        }
    }
}

impl Drop for SafepointPage {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            munmap(
                self.region_base.as_ptr() as *mut c_void,
                SAFEPOINT_REGION_SIZE,
            );
        }

        #[cfg(windows)]
        unsafe {
            VirtualFree(self.region_base.as_ptr() as *mut _, 0, MEM_RELEASE);
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safepoint_page_creation() {
        let page = SafepointPage::new().expect("Failed to create safepoint page");
        assert_eq!(page.state(), SafepointState::Enabled);
        assert!(!page.is_armed());
        assert!(page.poll_address() != 0);
    }

    #[test]
    fn test_safepoint_arm_disarm() {
        let page = SafepointPage::new().expect("Failed to create safepoint page");

        // Arm the page
        page.arm().expect("Failed to arm");
        assert!(page.is_armed());
        assert_eq!(page.state(), SafepointState::Armed);

        // Disarm the page
        page.disarm().expect("Failed to disarm");
        assert!(!page.is_armed());
        assert_eq!(page.state(), SafepointState::Enabled);
    }

    #[test]
    fn test_safepoint_double_arm_fails() {
        let page = SafepointPage::new().expect("Failed to create safepoint page");

        page.arm().expect("First arm should succeed");

        // Second arm should fail
        let result = page.arm();
        assert!(result.is_err());

        page.disarm().expect("Disarm should succeed");
    }

    #[test]
    fn test_safepoint_contains_address() {
        let page = SafepointPage::new().expect("Failed to create safepoint page");
        let addr = page.poll_address();

        assert!(page.contains_address(addr));
        assert!(page.contains_address(addr + 100));
        assert!(!page.contains_address(addr + SAFEPOINT_PAGE_SIZE));
        assert!(!page.contains_address(0));
    }

    #[test]
    fn test_safepoint_thread_counting() {
        let page = SafepointPage::new().expect("Failed to create safepoint page");

        assert_eq!(page.stopped_count(), 0);

        let count1 = page.thread_stopped();
        assert_eq!(count1, 1);
        assert_eq!(page.stopped_count(), 1);

        let count2 = page.thread_stopped();
        assert_eq!(count2, 2);
        assert_eq!(page.stopped_count(), 2);
    }

    #[test]
    fn test_safepoint_state_from_u32() {
        assert_eq!(SafepointState::from_u32(0), Some(SafepointState::Enabled));
        assert_eq!(SafepointState::from_u32(1), Some(SafepointState::Armed));
        assert_eq!(SafepointState::from_u32(2), Some(SafepointState::Triggered));
        assert_eq!(SafepointState::from_u32(99), None);
    }

    #[test]
    fn test_safepoint_cache_line_aligned() {
        // Verify SafepointPage is cache-line aligned (64 bytes)
        assert_eq!(std::mem::align_of::<SafepointPage>(), 64);
    }
}
