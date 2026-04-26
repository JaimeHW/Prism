//! Generator object implementation.
//!
//! This module provides the `GeneratorObject` which represents a suspended
//! Python generator. The object captures:
//!
//! - Execution state (via `GeneratorHeader`)
//! - Live registers (via `FrameStorage`)
//! - Code reference and instruction pointer
//! - Sent/thrown values for coroutine protocol
//!
//! # Memory Layout (96 bytes target)
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────┐
//! │ GeneratorHeader (4 bytes) │ Flags (2 bytes) │ Pad (2 bytes)   │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ Code: Arc<CodeObject>                                         │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ IP: u32 │ Receive: Option<u8>                                 │ 8 bytes
//! ├───────────────────────────────────────────────────────────────┤
//! │ FrameStorage (inline [Value; 8] + metadata)                   │ ~72 bytes
//! └───────────────────────────────────────────────────────────────┘
//! ```

use crate::frame::ClosureEnv;
use prism_code::{CodeFlags, CodeObject};
use prism_core::Value;
use prism_gc::Trace;
use prism_gc::trace::Tracer;
use prism_runtime::object::ObjectHeader;
use prism_runtime::object::type_obj::TypeId;
use std::fmt;
use std::sync::Arc;

use super::state::{GeneratorHeader, GeneratorState};
use super::storage::{FrameStorage, LivenessMap};

// ============================================================================
// Generator Flags
// ============================================================================

/// Generator object flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GeneratorFlags(u16);

impl GeneratorFlags {
    /// Generator has JIT-compiled code available.
    pub const HAS_JIT: Self = Self(0b0000_0001);
    /// Generator is using inline frame storage.
    pub const INLINE_STORAGE: Self = Self(0b0000_0010);
    /// Generator has a closure environment.
    pub const HAS_CLOSURE: Self = Self(0b0000_0100);
    /// Generator is a coroutine (uses send/throw).
    pub const IS_COROUTINE: Self = Self(0b0000_1000);
    /// Generator is an async generator.
    pub const IS_ASYNC: Self = Self(0b0001_0000);
    /// Generator has been started at least once.
    pub const STARTED: Self = Self(0b0010_0000);
    /// Empty flag set.
    pub const EMPTY: Self = Self(0);

    /// Returns true if this contains all bits of `other`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the union of self and other.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the raw bits.
    #[inline]
    pub const fn bits(self) -> u16 {
        self.0
    }
}

impl std::ops::BitOr for GeneratorFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for GeneratorFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

// ============================================================================
// Generator Object
// ============================================================================

/// A Python generator object.
///
/// This represents a suspended generator function, storing all state
/// necessary to resume execution at the next yield point.
///
/// # Performance
///
/// - Creation: ~15 cycles (inline storage, no allocation)
/// - Yield: ~3-5 cycles (state update + value copy)
/// - Resume: ~5 cycles (state check + dispatch)
/// - Memory: 96 bytes typical (2 cache lines)
#[repr(C)]
pub struct GeneratorObject {
    // === PYOBJECT HEADER (16 bytes) ===
    /// Object header for O(1) VM type dispatch.
    pub header: ObjectHeader,

    // === HEADER (8 bytes) ===
    /// Tagged state + resume index.
    state_header: GeneratorHeader,
    /// Configuration flags.
    flags: GeneratorFlags,
    /// Padding for alignment.
    _pad: u16,

    // === CODE REFERENCE (8 bytes) ===
    /// Reference to the generator's code object.
    code: Arc<CodeObject>,

    // === EXECUTION STATE (8 bytes) ===
    /// Instruction pointer to resume at.
    ip: u32,
    /// Liveness map for current yield point (compact form).
    liveness_bits: u32,

    // === FRAME STORAGE ===
    /// Storage for live register values.
    storage: FrameStorage,

    // === SEND/THROW VALUE ===
    /// Value received via send() or throw().
    receive_value: Option<Value>,

    /// Captured closure environment, if this generator was created from a closure.
    closure: Option<Arc<ClosureEnv>>,

    /// Module globals backing this generator's code.
    module_ptr: *const (),
}

impl GeneratorObject {
    /// Creates a new generator object for the given code.
    ///
    /// The generator starts in `Created` state and must be primed
    /// with the first call to `next()`.
    #[inline]
    pub fn new(code: Arc<CodeObject>) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::GENERATOR),
            state_header: GeneratorHeader::new(),
            flags: GeneratorFlags::INLINE_STORAGE,
            _pad: 0,
            code,
            ip: 0,
            liveness_bits: 0,
            storage: FrameStorage::new(),
            receive_value: None,
            closure: None,
            module_ptr: std::ptr::null(),
        }
    }

    /// Creates a generator with specific flags (e.g., for async generators).
    #[inline]
    pub fn with_flags(code: Arc<CodeObject>, flags: GeneratorFlags) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::GENERATOR),
            state_header: GeneratorHeader::new(),
            flags,
            _pad: 0,
            code,
            ip: 0,
            liveness_bits: 0,
            storage: FrameStorage::new(),
            receive_value: None,
            closure: None,
            module_ptr: std::ptr::null(),
        }
    }

    /// Creates a generator object from code flags.
    ///
    /// This maps compiler-level function flags onto runtime generator flags.
    #[inline]
    pub fn from_code(code: Arc<CodeObject>) -> Self {
        let mut flags = GeneratorFlags::INLINE_STORAGE;
        if code.flags.contains(CodeFlags::COROUTINE) {
            flags |= GeneratorFlags::IS_COROUTINE;
        }
        if code.flags.contains(CodeFlags::ASYNC_GENERATOR) {
            flags |= GeneratorFlags::IS_ASYNC;
        }
        Self::with_flags(code, flags)
    }

    /// Returns a shared generator reference if the value is a generator object.
    #[inline]
    pub fn from_value(value: Value) -> Option<&'static Self> {
        let ptr = value.as_object_ptr()?;
        Self::from_object_ptr(ptr)
    }

    /// Returns a mutable generator reference if the value is a generator object.
    #[inline]
    pub fn from_value_mut(value: Value) -> Option<&'static mut Self> {
        let ptr = value.as_object_ptr()?;
        if object_type_id(ptr) != TypeId::GENERATOR {
            return None;
        }
        Some(unsafe { &mut *(ptr as *mut Self) })
    }

    /// Returns a shared generator reference from a raw object pointer.
    #[inline]
    pub fn from_object_ptr(ptr: *const ()) -> Option<&'static Self> {
        if object_type_id(ptr) != TypeId::GENERATOR {
            return None;
        }
        Some(unsafe { &*(ptr as *const Self) })
    }

    // ═══════════════════════════════════════════════════════════════════════
    // State Accessors
    // ═══════════════════════════════════════════════════════════════════════

    /// Returns the current generator state.
    #[inline(always)]
    pub fn state(&self) -> GeneratorState {
        self.state_header.state()
    }

    /// Returns the resume index (yield point ID).
    #[inline(always)]
    pub fn resume_index(&self) -> u32 {
        self.state_header.resume_index()
    }

    /// Returns true if the generator can be resumed.
    #[inline(always)]
    pub fn is_resumable(&self) -> bool {
        self.state_header.is_resumable()
    }

    /// Returns true if the generator is exhausted.
    #[inline(always)]
    pub fn is_exhausted(&self) -> bool {
        self.state_header.is_exhausted()
    }

    /// Returns true if the generator is currently running.
    #[inline(always)]
    pub fn is_running(&self) -> bool {
        self.state_header.is_running()
    }

    /// Returns the generator's flags.
    #[inline(always)]
    pub fn flags(&self) -> GeneratorFlags {
        self.flags
    }

    /// Returns a reference to the code object.
    #[inline(always)]
    pub fn code(&self) -> &Arc<CodeObject> {
        &self.code
    }

    /// Returns the instruction pointer.
    #[inline(always)]
    pub fn ip(&self) -> u32 {
        self.ip
    }

    /// Returns the liveness map for the current yield point.
    #[inline]
    pub fn liveness(&self) -> LivenessMap {
        LivenessMap::from_bits(self.liveness_bits as u64)
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Lifecycle Methods
    // ═══════════════════════════════════════════════════════════════════════

    /// Attempts to start or resume the generator.
    ///
    /// Returns the previous state if successful, or None if the generator
    /// cannot be resumed (already running or exhausted).
    #[inline]
    pub fn try_start(&self) -> Option<GeneratorState> {
        self.state_header.try_start()
    }

    /// Suspends the generator at a yield point.
    ///
    /// # Parameters
    /// - `ip`: Instruction pointer for the next resume
    /// - `resume_index`: Yield point index for dispatch
    /// - `registers`: Current register file to capture
    /// - `liveness`: Bitmap of live registers at this yield
    #[inline]
    pub fn suspend(
        &mut self,
        ip: u32,
        resume_index: u32,
        registers: &[Value; 256],
        liveness: LivenessMap,
    ) {
        self.ip = ip;
        self.liveness_bits = liveness.bits() as u32;
        self.storage.capture(registers, liveness);
        self.state_header.suspend(resume_index);
    }

    /// Marks the generator as exhausted (returned or closed).
    #[inline]
    pub fn exhaust(&self) {
        self.state_header.exhaust();
    }

    /// Restores the generator's frame to a register file.
    #[inline]
    pub fn restore(&self, registers: &mut [Value; 256]) {
        let liveness = self.liveness();
        self.storage.restore(registers, liveness);
    }

    /// Seeds the initial local register snapshot used when this generator starts.
    ///
    /// This is called from the function call path when a generator/coroutine object
    /// is created from bound call arguments.
    #[inline]
    pub fn seed_locals(&mut self, registers: &[Value; 256], liveness: LivenessMap) {
        self.liveness_bits = liveness.bits() as u32;
        self.storage.capture(registers, liveness);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Send/Throw Protocol
    // ═══════════════════════════════════════════════════════════════════════

    /// Sets a value to be received by the generator on resume.
    ///
    /// This is used by `send()` to inject values into the generator.
    #[inline]
    pub fn set_send_value(&mut self, value: Value) {
        self.receive_value = Some(value);
    }

    /// Takes the pending receive value, if any.
    #[inline]
    pub fn take_receive_value(&mut self) -> Option<Value> {
        self.receive_value.take()
    }

    /// Returns the pending receive value without consuming it.
    #[inline]
    pub fn peek_receive_value(&self) -> Option<Value> {
        self.receive_value
    }

    /// Record the module globals context that should be restored on resume.
    #[inline]
    pub fn set_module_ptr(&mut self, module_ptr: *const ()) {
        self.module_ptr = module_ptr;
    }

    /// Get the module globals pointer captured for this generator.
    #[inline]
    pub fn module_ptr(&self) -> *const () {
        self.module_ptr
    }

    /// Record the closure environment that should be restored on resume.
    #[inline]
    pub fn set_closure(&mut self, closure: Arc<ClosureEnv>) {
        self.flags |= GeneratorFlags::HAS_CLOSURE;
        self.closure = Some(closure);
    }

    /// Get the closure environment captured for this generator, if any.
    #[inline]
    pub fn closure(&self) -> Option<&Arc<ClosureEnv>> {
        self.closure.as_ref()
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Flag Manipulation
    // ═══════════════════════════════════════════════════════════════════════

    /// Sets the STARTED flag.
    #[inline]
    pub fn mark_started(&mut self) {
        self.flags |= GeneratorFlags::STARTED;
    }

    /// Returns true if the generator has been started.
    #[inline]
    pub fn is_started(&self) -> bool {
        self.flags.contains(GeneratorFlags::STARTED)
    }

    /// Returns true if this is a coroutine.
    #[inline]
    pub fn is_coroutine(&self) -> bool {
        self.flags.contains(GeneratorFlags::IS_COROUTINE)
    }

    /// Returns true if this is an async generator.
    #[inline]
    pub fn is_async(&self) -> bool {
        self.flags.contains(GeneratorFlags::IS_ASYNC)
    }
}

impl Clone for GeneratorObject {
    fn clone(&self) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::GENERATOR),
            state_header: self.state_header.clone(),
            flags: self.flags,
            _pad: 0,
            code: Arc::clone(&self.code),
            ip: self.ip,
            liveness_bits: self.liveness_bits,
            storage: self.storage.clone(),
            receive_value: self.receive_value,
            closure: self.closure.clone(),
            module_ptr: self.module_ptr,
        }
    }
}

impl fmt::Debug for GeneratorObject {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GeneratorObject")
            .field("state", &self.state())
            .field("resume_index", &self.resume_index())
            .field("ip", &self.ip)
            .field("flags", &self.flags)
            .field("has_closure", &self.closure.is_some())
            .field("storage_len", &self.storage.len())
            .finish()
    }
}

// SAFETY: GeneratorObject is safe to send between threads
// (though concurrent access to a single generator is not safe)
unsafe impl Send for GeneratorObject {}

unsafe impl Trace for GeneratorObject {
    fn trace(&self, tracer: &mut dyn Tracer) {
        for idx in 0..self.storage.len() {
            self.storage.get(idx).trace(tracer);
        }
        self.receive_value.trace(tracer);
        if let Some(closure) = &self.closure {
            for idx in 0..closure.len() {
                if let Some(value) = closure.get_cell(idx).get() {
                    value.trace(tracer);
                }
            }
        }
    }
}

#[inline(always)]
fn object_type_id(ptr: *const ()) -> TypeId {
    let header = ptr as *const ObjectHeader;
    unsafe { (*header).type_id }
}
