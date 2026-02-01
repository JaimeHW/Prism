//! Signal/exception handler for safepoint traps.
//!
//! When the safepoint page is armed (PROT_NONE), JIT code polling it
//! will trigger a memory access violation. This handler catches those
//! faults and coordinates with the GC.
//!
//! # Platform Support
//!
//! | Platform | Mechanism |
//! |----------|-----------|
//! | Linux    | SIGSEGV via sigaction |
//! | macOS    | SIGBUS via sigaction |
//! | Windows  | VEH (Vectored Exception Handler) |

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::page::SafepointPage;

// =============================================================================
// TrapContext
// =============================================================================

/// Context passed to the safepoint handler.
///
/// Contains the thread's register state at the point of trap.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TrapContext {
    /// Faulting instruction address.
    pub fault_addr: usize,
    /// Return address (for stack walking).
    pub return_addr: usize,
    /// Stack pointer.
    pub rsp: usize,
    /// Base pointer.
    pub rbp: usize,
    /// Thread ID.
    pub thread_id: u64,
}

impl TrapContext {
    /// Create a new trap context.
    pub fn new(fault_addr: usize, return_addr: usize, rsp: usize, rbp: usize) -> Self {
        TrapContext {
            fault_addr,
            return_addr,
            rsp,
            rbp,
            thread_id: current_thread_id(),
        }
    }
}

// =============================================================================
// SafepointHandler Trait
// =============================================================================

/// Platform-agnostic safepoint trap handling.
pub trait SafepointHandler: Send + Sync {
    /// Install the safepoint trap handler.
    fn install(&self) -> Result<(), HandlerError>;

    /// Uninstall the handler.
    fn uninstall(&self) -> Result<(), HandlerError>;

    /// Check if an address is the safepoint page.
    fn is_safepoint_fault(&self, fault_addr: usize) -> bool;

    /// Handle a safepoint trap (called from signal handler).
    ///
    /// # Safety
    ///
    /// This is called from a signal handler context. Limited operations
    /// are safe (async-signal-safe functions only).
    unsafe fn handle_trap(&self, context: &TrapContext);
}

/// Error types for handler operations.
#[derive(Debug, Clone)]
pub enum HandlerError {
    /// Handler already installed.
    AlreadyInstalled,
    /// Failed to install handler.
    InstallFailed(String),
    /// Handler not installed.
    NotInstalled,
    /// Platform not supported.
    UnsupportedPlatform,
}

impl std::fmt::Display for HandlerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandlerError::AlreadyInstalled => write!(f, "handler already installed"),
            HandlerError::InstallFailed(msg) => write!(f, "handler install failed: {}", msg),
            HandlerError::NotInstalled => write!(f, "handler not installed"),
            HandlerError::UnsupportedPlatform => write!(f, "platform not supported"),
        }
    }
}

impl std::error::Error for HandlerError {}

// =============================================================================
// Global Handler State
// =============================================================================

/// Global safepoint page reference for signal handlers.
static SAFEPOINT_PAGE_ADDR: AtomicUsize = AtomicUsize::new(0);

/// Whether the handler is currently installed.
static HANDLER_INSTALLED: AtomicBool = AtomicBool::new(false);

/// Callback function for safepoint traps.
type TrapCallback = fn(&TrapContext);

/// Thread-local storage for trap callback.
thread_local! {
    static TRAP_CALLBACK: RefCell<Option<TrapCallback>> = const { RefCell::new(None) };
}

/// Set the global safepoint page address for signal handlers.
pub fn set_safepoint_page_address(addr: usize) {
    SAFEPOINT_PAGE_ADDR.store(addr, Ordering::Release);
}

/// Get the global safepoint page address.
pub fn get_safepoint_page_address() -> usize {
    SAFEPOINT_PAGE_ADDR.load(Ordering::Acquire)
}

/// Set the thread-local trap callback.
pub fn set_trap_callback(callback: TrapCallback) {
    TRAP_CALLBACK.with(|cb| {
        *cb.borrow_mut() = Some(callback);
    });
}

/// Check if an address is within the safepoint page.
#[inline]
pub fn is_safepoint_address(addr: usize) -> bool {
    let page_addr = SAFEPOINT_PAGE_ADDR.load(Ordering::Acquire);
    if page_addr == 0 {
        return false;
    }
    addr >= page_addr && addr < page_addr + super::SAFEPOINT_PAGE_SIZE
}

// =============================================================================
// Platform-Specific Installation
// =============================================================================

/// Install the platform-specific safepoint handler.
///
/// This must be called before arming the safepoint page.
///
/// # Arguments
///
/// * `page` - The safepoint page to monitor
#[cfg(unix)]
pub fn install_handler(page: &SafepointPage) -> Result<(), HandlerError> {
    if HANDLER_INSTALLED.swap(true, Ordering::AcqRel) {
        return Err(HandlerError::AlreadyInstalled);
    }

    // Store page address for signal handler
    set_safepoint_page_address(page.poll_address());

    unsafe {
        // Install SIGSEGV handler
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = sigsegv_handler as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;

        if libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut()) != 0 {
            HANDLER_INSTALLED.store(false, Ordering::Release);
            return Err(HandlerError::InstallFailed(
                std::io::Error::last_os_error().to_string(),
            ));
        }

        // Also install SIGBUS for macOS
        #[cfg(target_os = "macos")]
        {
            if libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut()) != 0 {
                HANDLER_INSTALLED.store(false, Ordering::Release);
                return Err(HandlerError::InstallFailed(
                    std::io::Error::last_os_error().to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Install the platform-specific safepoint handler (Windows).
#[cfg(windows)]
pub fn install_handler(page: &SafepointPage) -> Result<(), HandlerError> {
    if HANDLER_INSTALLED.swap(true, Ordering::AcqRel) {
        return Err(HandlerError::AlreadyInstalled);
    }

    set_safepoint_page_address(page.poll_address());

    unsafe {
        let handle = AddVectoredExceptionHandler(1, Some(vectored_exception_handler));
        if handle.is_null() {
            HANDLER_INSTALLED.store(false, Ordering::Release);
            return Err(HandlerError::InstallFailed(
                "AddVectoredExceptionHandler failed".to_string(),
            ));
        }
    }

    Ok(())
}

// Windows VEH types and FFI
#[cfg(windows)]
#[allow(non_camel_case_types)]
type PVECTORED_EXCEPTION_HANDLER =
    Option<unsafe extern "system" fn(*mut EXCEPTION_POINTERS) -> i32>;

#[cfg(windows)]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn AddVectoredExceptionHandler(
        first: u32,
        handler: PVECTORED_EXCEPTION_HANDLER,
    ) -> *mut std::ffi::c_void;
    fn GetCurrentThreadId() -> u32;
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct EXCEPTION_POINTERS {
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ContextRecord: *mut CONTEXT,
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct EXCEPTION_RECORD {
    ExceptionCode: i32, // NTSTATUS is signed
    ExceptionFlags: u32,
    ExceptionRecord: *mut EXCEPTION_RECORD,
    ExceptionAddress: *mut std::ffi::c_void,
    NumberParameters: u32,
    ExceptionInformation: [usize; 15],
}

#[cfg(windows)]
#[repr(C)]
#[allow(non_snake_case)]
struct CONTEXT {
    // Minimal x64 CONTEXT structure - only fields we need
    _padding1: [u8; 0x78], // Offset to Rax
    Rax: u64,
    Rcx: u64,
    Rdx: u64,
    Rbx: u64,
    Rsp: u64,
    Rbp: u64,
    Rsi: u64,
    Rdi: u64,
    R8: u64,
    R9: u64,
    R10: u64,
    R11: u64,
    R12: u64,
    R13: u64,
    R14: u64,
    R15: u64,
    Rip: u64,
    _remainder: [u8; 0x200], // Rest of the structure
}

/// Uninstall the safepoint handler.
pub fn uninstall_handler() -> Result<(), HandlerError> {
    if !HANDLER_INSTALLED.swap(false, Ordering::AcqRel) {
        return Err(HandlerError::NotInstalled);
    }

    // Clear the page address
    SAFEPOINT_PAGE_ADDR.store(0, Ordering::Release);

    // Platform-specific cleanup would go here
    // (signal handlers remain installed but will pass through non-safepoint faults)

    Ok(())
}

// =============================================================================
// Unix Signal Handler
// =============================================================================

#[cfg(unix)]
extern "C" fn sigsegv_handler(
    _sig: libc::c_int,
    info: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    unsafe {
        let fault_addr = (*info).si_addr() as usize;

        // Check if this is a safepoint fault
        if !is_safepoint_address(fault_addr) {
            // Not our fault - restore default handler and re-raise
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = libc::SIG_DFL;
            libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut());
            return;
        }

        // Extract register context
        #[cfg(target_arch = "x86_64")]
        let (rsp, rbp, rip) = {
            let uc = context as *mut libc::ucontext_t;
            #[cfg(target_os = "linux")]
            {
                let mcontext = &(*uc).uc_mcontext;
                (
                    mcontext.gregs[libc::REG_RSP as usize] as usize,
                    mcontext.gregs[libc::REG_RBP as usize] as usize,
                    mcontext.gregs[libc::REG_RIP as usize] as usize,
                )
            }
            #[cfg(target_os = "macos")]
            {
                let mcontext = (*uc).uc_mcontext;
                (
                    (*mcontext).__ss.__rsp as usize,
                    (*mcontext).__ss.__rbp as usize,
                    (*mcontext).__ss.__rip as usize,
                )
            }
        };

        let trap_context = TrapContext::new(fault_addr, rip, rsp, rbp);

        // Call the registered callback
        TRAP_CALLBACK.with(|cb| {
            if let Some(callback) = *cb.borrow() {
                callback(&trap_context);
            }
        });

        // Skip past the faulting instruction (3 bytes for test [r15], al)
        #[cfg(target_arch = "x86_64")]
        {
            let uc = context as *mut libc::ucontext_t;
            #[cfg(target_os = "linux")]
            {
                (*uc).uc_mcontext.gregs[libc::REG_RIP as usize] +=
                    super::poll::SAFEPOINT_POLL_SIZE as i64;
            }
            #[cfg(target_os = "macos")]
            {
                let mcontext = (*uc).uc_mcontext;
                (*mcontext).__ss.__rip += super::poll::SAFEPOINT_POLL_SIZE as u64;
            }
        }
    }
}

// =============================================================================
// Windows Vectored Exception Handler
// =============================================================================

#[cfg(windows)]
unsafe extern "system" fn vectored_exception_handler(
    exception_info: *mut EXCEPTION_POINTERS,
) -> i32 {
    // NTSTATUS is i32
    const EXCEPTION_ACCESS_VIOLATION: i32 = 0xC0000005u32 as i32;
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    // SAFETY: exception_info is provided by Windows and is valid
    let record = unsafe { (*exception_info).ExceptionRecord };
    let context = unsafe { (*exception_info).ContextRecord };

    // Only handle access violations
    if unsafe { (*record).ExceptionCode } != EXCEPTION_ACCESS_VIOLATION {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Get faulting address (second parameter of access violation)
    let fault_addr = unsafe { (*record).ExceptionInformation[1] };

    // Check if this is a safepoint fault
    if !is_safepoint_address(fault_addr) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    // Extract register context
    let trap_context = unsafe {
        TrapContext::new(
            fault_addr,
            (*context).Rip as usize,
            (*context).Rsp as usize,
            (*context).Rbp as usize,
        )
    };

    // Call the registered callback
    TRAP_CALLBACK.with(|cb| {
        if let Some(callback) = *cb.borrow() {
            callback(&trap_context);
        }
    });

    // Skip past the faulting instruction (3 bytes for test [r15], al)
    unsafe { (*context).Rip += super::poll::SAFEPOINT_POLL_SIZE as u64 };

    EXCEPTION_CONTINUE_EXECUTION
}

// =============================================================================
// Thread ID Helper
// =============================================================================

#[cfg(unix)]
fn current_thread_id() -> u64 {
    unsafe { libc::pthread_self() as u64 }
}

#[cfg(windows)]
fn current_thread_id() -> u64 {
    unsafe { GetCurrentThreadId() as u64 }
}

// =============================================================================
// Default Handler Implementation
// =============================================================================

/// Default safepoint handler using the platform-specific mechanisms.
pub struct DefaultSafepointHandler {
    page: Arc<SafepointPage>,
}

impl DefaultSafepointHandler {
    /// Create a new handler for the given safepoint page.
    pub fn new(page: Arc<SafepointPage>) -> Self {
        DefaultSafepointHandler { page }
    }
}

impl SafepointHandler for DefaultSafepointHandler {
    fn install(&self) -> Result<(), HandlerError> {
        install_handler(&self.page)
    }

    fn uninstall(&self) -> Result<(), HandlerError> {
        uninstall_handler()
    }

    fn is_safepoint_fault(&self, fault_addr: usize) -> bool {
        self.page.contains_address(fault_addr)
    }

    unsafe fn handle_trap(&self, context: &TrapContext) {
        // Mark thread as stopped
        self.page.thread_stopped();

        // Wait for GC to complete (coordinator will disarm page)
        while self.page.is_armed() {
            std::hint::spin_loop();
        }

        // Log for debugging
        #[cfg(debug_assertions)]
        eprintln!(
            "[safepoint] Thread {} stopped at 0x{:x}",
            context.thread_id, context.fault_addr
        );
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trap_context_creation() {
        let ctx = TrapContext::new(0x1000, 0x2000, 0x3000, 0x4000);
        assert_eq!(ctx.fault_addr, 0x1000);
        assert_eq!(ctx.return_addr, 0x2000);
        assert_eq!(ctx.rsp, 0x3000);
        assert_eq!(ctx.rbp, 0x4000);
        assert!(ctx.thread_id != 0);
    }

    #[test]
    fn test_is_safepoint_address_no_page() {
        // With no page set, should return false
        let old = SAFEPOINT_PAGE_ADDR.swap(0, Ordering::SeqCst);
        assert!(!is_safepoint_address(0x12345));
        SAFEPOINT_PAGE_ADDR.store(old, Ordering::SeqCst);
    }

    #[test]
    fn test_is_safepoint_address_with_page() {
        let old = SAFEPOINT_PAGE_ADDR.swap(0x10000, Ordering::SeqCst);

        assert!(is_safepoint_address(0x10000));
        assert!(is_safepoint_address(0x10FFF));
        assert!(!is_safepoint_address(0x11000)); // Past page
        assert!(!is_safepoint_address(0x00000)); // Before page

        SAFEPOINT_PAGE_ADDR.store(old, Ordering::SeqCst);
    }

    #[test]
    fn test_handler_error_display() {
        assert_eq!(
            HandlerError::AlreadyInstalled.to_string(),
            "handler already installed"
        );
        assert_eq!(
            HandlerError::NotInstalled.to_string(),
            "handler not installed"
        );
    }
}
