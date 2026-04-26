//! Call frame management with stack-based register file.
//!
//! The Frame struct is the core execution context for a function call.
//! It uses a stack-allocated register file for maximum performance.

use crate::exception::InlineHandlerCache;
use crate::import::ModuleObject;
use prism_code::{CodeFlags, CodeObject, Constant};
use prism_core::Value;
use prism_runtime::types::int::bigint_to_value;
use std::sync::Arc;
use std::sync::OnceLock;

/// Maximum recursion depth before RecursionError.
pub const MAX_RECURSION_DEPTH: usize = 1000;

/// Number of registers per frame.
pub const REGISTER_COUNT: usize = 256;
const REGISTER_WORD_COUNT: usize = REGISTER_COUNT / 64;

#[derive(Clone, Copy)]
pub struct RegisterSnapshot {
    reg: u8,
    value: Value,
    written: bool,
}

/// A call frame representing a function invocation.
///
/// The frame contains:
/// - Reference to the code object being executed
/// - Instruction pointer (current position)
/// - Stack-based register file (256 values, ~2KB)
/// - Optional closure environment
/// - Return information for popping the frame
///
/// # Performance
///
/// The register file is stored inline (not boxed) to avoid heap allocation
/// and ensure L1 cache locality. At 256 * 8 = 2048 bytes, it fits comfortably
/// in modern L1 caches (typically 32-64KB).
#[repr(C)]
pub struct Frame {
    /// Code object being executed.
    pub code: Arc<CodeObject>,

    /// Instruction pointer (index into code.instructions).
    /// Using u32 for compact representation (4GB of instructions is plenty).
    pub ip: u32,

    /// Return address: index of caller frame in the frame stack.
    /// None for the top-level module frame.
    pub return_frame: Option<u32>,

    /// Register in the caller frame where the return value should be stored.
    pub return_reg: u8,

    /// Module globals backing this frame.
    pub module: Option<Arc<ModuleObject>>,

    /// Closure environment for captured variables.
    /// Only set for closures, None for regular functions.
    pub closure: Option<Arc<ClosureEnv>>,

    /// Stack-based register file.
    /// Registers r0-r255 used for local computation.
    /// Parameters are passed in r0, r1, r2, ...
    pub registers: [Value; REGISTER_COUNT],

    /// Bitset tracking which registers have been explicitly written.
    ///
    /// This lets the VM distinguish "never assigned" from an explicit
    /// `None` value for Python local-slot semantics.
    written_registers: [u64; REGISTER_WORD_COUNT],

    /// Optional frame-local storage for code objects whose local slot count is
    /// too large to reserve one register per local.
    separate_locals: Vec<Value>,

    /// Bitset tracking writes into `separate_locals`.
    written_separate_locals: Vec<u64>,

    /// Number of registers that must be cleared before reusing this frame.
    active_register_count: u16,

    /// Optional live locals mapping for scopes that execute against a custom
    /// namespace object, such as metaclass `__prepare__` class bodies.
    locals_mapping: Option<Value>,

    // =========================================================================
    // Generator Support
    // =========================================================================
    /// Yield point index for generators (0 = not a generator or not yet yielded).
    /// When a generator yields, this stores the resume table index for O(1) dispatch.
    pub yield_point: u32,

    /// Inline cache for exception-handler lookup at this frame's current PC.
    pub handler_cache: InlineHandlerCache,
}

fn pooled_frame_code() -> Arc<CodeObject> {
    static POOLED_FRAME_CODE: OnceLock<Arc<CodeObject>> = OnceLock::new();
    Arc::clone(
        POOLED_FRAME_CODE.get_or_init(|| Arc::new(CodeObject::new("<frame-pool>", "<internal>"))),
    )
}

#[derive(Default)]
pub struct FramePool {
    free_frames: Vec<Frame>,
}

impl FramePool {
    pub fn new() -> Self {
        Self {
            free_frames: Vec::with_capacity(64),
        }
    }

    pub fn acquire(
        &mut self,
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Option<Arc<ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
    ) -> Frame {
        if let Some(mut frame) = self.free_frames.pop() {
            frame.reinitialize(code, return_frame, return_reg, closure, module);
            frame
        } else {
            Frame::from_parts(code, return_frame, return_reg, closure, module)
        }
    }

    pub fn release(&mut self, mut frame: Frame) {
        if self.free_frames.len() >= MAX_RECURSION_DEPTH {
            return;
        }
        frame.prepare_for_pool();
        self.free_frames.push(frame);
    }
}

pub use prism_runtime::types::function::ClosureEnv;

impl Frame {
    fn blank() -> Self {
        Self {
            code: pooled_frame_code(),
            ip: 0,
            return_frame: None,
            return_reg: 0,
            module: None,
            closure: None,
            registers: [Value::none(); REGISTER_COUNT],
            written_registers: [0; REGISTER_WORD_COUNT],
            separate_locals: Vec::new(),
            written_separate_locals: Vec::new(),
            active_register_count: 0,
            locals_mapping: None,
            yield_point: 0,
            handler_cache: InlineHandlerCache::new(),
        }
    }

    #[inline]
    fn active_register_count_for(code: &CodeObject) -> u16 {
        code.register_count.max(1)
    }

    fn from_parts(
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Option<Arc<ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
    ) -> Self {
        let mut frame = Self::blank();
        frame.reinitialize(code, return_frame, return_reg, closure, module);
        frame
    }

    fn reinitialize(
        &mut self,
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Option<Arc<ClosureEnv>>,
        module: Option<Arc<ModuleObject>>,
    ) {
        self.code = code;
        self.ip = 0;
        self.return_frame = return_frame;
        self.return_reg = return_reg;
        self.module = module;
        self.closure = closure;
        self.active_register_count = Self::active_register_count_for(&self.code);
        self.prepare_separate_locals();
        self.locals_mapping = None;
        self.yield_point = 0;
        self.handler_cache = InlineHandlerCache::new();
    }

    fn prepare_for_pool(&mut self) {
        self.clear_register_window(self.active_register_count.into());
        self.written_registers.fill(0);
        self.clear_separate_locals();
        self.code = pooled_frame_code();
        self.ip = 0;
        self.return_frame = None;
        self.return_reg = 0;
        self.module = None;
        self.closure = None;
        self.active_register_count = 0;
        self.locals_mapping = None;
        self.yield_point = 0;
        self.handler_cache = InlineHandlerCache::new();
    }

    #[inline]
    fn clear_register_window(&mut self, count: usize) {
        if count > 0 {
            self.registers[..count].fill(Value::none());
        }
    }

    #[inline]
    fn uses_separate_locals_code(code: &CodeObject) -> bool {
        code.flags.contains(CodeFlags::SEPARATE_LOCALS)
    }

    fn prepare_separate_locals(&mut self) {
        if !Self::uses_separate_locals_code(&self.code) {
            self.clear_separate_locals();
            return;
        }

        let local_count = self.code.locals.len();
        self.separate_locals.resize(local_count, Value::none());
        self.separate_locals.fill(Value::none());

        let word_count = local_count.div_ceil(64);
        self.written_separate_locals.resize(word_count, 0);
        self.written_separate_locals.fill(0);
    }

    fn clear_separate_locals(&mut self) {
        self.separate_locals.fill(Value::none());
        self.separate_locals.clear();
        self.written_separate_locals.fill(0);
        self.written_separate_locals.clear();
    }

    /// Create a new frame for executing a code object.
    ///
    /// # Arguments
    /// * `code` - The code object to execute
    /// * `return_frame` - Index of the caller frame (None for module level)
    /// * `return_reg` - Register in caller to store return value
    #[inline]
    pub fn new(code: Arc<CodeObject>, return_frame: Option<u32>, return_reg: u8) -> Self {
        Self::new_with_module(code, return_frame, return_reg, None)
    }

    /// Create a new frame with an explicit module context.
    #[inline]
    pub fn new_with_module(
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        module: Option<Arc<ModuleObject>>,
    ) -> Self {
        Self::from_parts(code, return_frame, return_reg, None, module)
    }

    /// Create a frame with a closure environment.
    #[inline]
    pub fn with_closure(
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Arc<ClosureEnv>,
    ) -> Self {
        Self::with_closure_and_module(code, return_frame, return_reg, closure, None)
    }

    /// Create a frame with a closure environment and explicit module context.
    #[inline]
    pub fn with_closure_and_module(
        code: Arc<CodeObject>,
        return_frame: Option<u32>,
        return_reg: u8,
        closure: Arc<ClosureEnv>,
        module: Option<Arc<ModuleObject>>,
    ) -> Self {
        Self::from_parts(code, return_frame, return_reg, Some(closure), module)
    }

    // =========================================================================
    // Register Access (Inlined for Performance)
    // =========================================================================

    /// Get a register value.
    #[inline(always)]
    pub fn get_reg(&self, reg: u8) -> Value {
        // Safety: reg is u8, so always in bounds for 256-element array
        unsafe { *self.registers.get_unchecked(reg as usize) }
    }

    /// Set a register value.
    #[inline(always)]
    pub fn set_reg(&mut self, reg: u8, value: Value) {
        // Safety: reg is u8, so always in bounds for 256-element array
        let reg_idx = reg as usize;
        unsafe { *self.registers.get_unchecked_mut(reg_idx) = value };
        let word = reg_idx / 64;
        let bit = reg_idx % 64;
        self.written_registers[word] |= 1u64 << bit;
    }

    /// Clear a register and mark it as logically unassigned.
    #[inline(always)]
    pub fn clear_reg(&mut self, reg: u8) {
        let reg_idx = reg as usize;
        unsafe { *self.registers.get_unchecked_mut(reg_idx) = Value::none() };
        let word = reg_idx / 64;
        let bit = reg_idx % 64;
        self.written_registers[word] &= !(1u64 << bit);
    }

    /// Mark a register as logically assigned without changing its value.
    #[inline(always)]
    pub fn mark_reg_written(&mut self, reg: u8) {
        let reg_idx = reg as usize;
        let word = reg_idx / 64;
        let bit = reg_idx % 64;
        self.written_registers[word] |= 1u64 << bit;
    }

    #[inline(always)]
    pub fn snapshot_register(&self, reg: u8) -> RegisterSnapshot {
        RegisterSnapshot {
            reg,
            value: self.get_reg(reg),
            written: self.reg_is_written(reg),
        }
    }

    #[inline(always)]
    pub fn restore_register(&mut self, snapshot: RegisterSnapshot) {
        let reg_idx = snapshot.reg as usize;
        unsafe { *self.registers.get_unchecked_mut(reg_idx) = snapshot.value };

        let word = reg_idx / 64;
        let bit = reg_idx % 64;
        if snapshot.written {
            self.written_registers[word] |= 1u64 << bit;
        } else {
            self.written_registers[word] &= !(1u64 << bit);
        }
    }

    /// Check whether a register has been explicitly written.
    #[inline(always)]
    pub fn reg_is_written(&self, reg: u8) -> bool {
        let reg_idx = reg as usize;
        let word = reg_idx / 64;
        let bit = reg_idx % 64;
        (self.written_registers[word] & (1u64 << bit)) != 0
    }

    #[inline(always)]
    pub fn uses_separate_locals(&self) -> bool {
        Self::uses_separate_locals_code(&self.code)
    }

    #[inline(always)]
    pub fn get_local(&self, slot: u16) -> Value {
        if self.uses_separate_locals() {
            unsafe { *self.separate_locals.get_unchecked(slot as usize) }
        } else {
            self.get_reg(slot as u8)
        }
    }

    #[inline(always)]
    pub fn set_local(&mut self, slot: u16, value: Value) {
        if self.uses_separate_locals() {
            let slot_idx = slot as usize;
            unsafe { *self.separate_locals.get_unchecked_mut(slot_idx) = value };
            let word = slot_idx / 64;
            let bit = slot_idx % 64;
            unsafe { *self.written_separate_locals.get_unchecked_mut(word) |= 1u64 << bit };
        } else {
            self.set_reg(slot as u8, value);
        }
    }

    #[inline(always)]
    pub fn clear_local(&mut self, slot: u16) {
        if self.uses_separate_locals() {
            let slot_idx = slot as usize;
            unsafe { *self.separate_locals.get_unchecked_mut(slot_idx) = Value::none() };
            let word = slot_idx / 64;
            let bit = slot_idx % 64;
            unsafe { *self.written_separate_locals.get_unchecked_mut(word) &= !(1u64 << bit) };
        } else {
            self.clear_reg(slot as u8);
        }
    }

    #[inline(always)]
    pub fn local_is_written(&self, slot: u16) -> bool {
        if self.uses_separate_locals() {
            let slot_idx = slot as usize;
            let word = slot_idx / 64;
            let bit = slot_idx % 64;
            unsafe { (*self.written_separate_locals.get_unchecked(word) & (1u64 << bit)) != 0 }
        } else {
            self.reg_is_written(slot as u8)
        }
    }

    #[inline(always)]
    pub fn active_register_count(&self) -> u16 {
        self.active_register_count
    }

    /// Return the live locals mapping for this frame, when one is active.
    #[inline(always)]
    pub fn locals_mapping(&self) -> Option<Value> {
        self.locals_mapping
    }

    /// Install or clear the live locals mapping for this frame.
    #[inline(always)]
    pub fn set_locals_mapping(&mut self, mapping: Option<Value>) {
        self.locals_mapping = mapping;
    }

    /// Get two register values (common for binary ops).
    #[inline(always)]
    pub fn get_regs2(&self, r1: u8, r2: u8) -> (Value, Value) {
        unsafe {
            (
                *self.registers.get_unchecked(r1 as usize),
                *self.registers.get_unchecked(r2 as usize),
            )
        }
    }

    /// Get three register values (common for ternary ops).
    #[inline(always)]
    pub fn get_regs3(&self, r1: u8, r2: u8, r3: u8) -> (Value, Value, Value) {
        unsafe {
            (
                *self.registers.get_unchecked(r1 as usize),
                *self.registers.get_unchecked(r2 as usize),
                *self.registers.get_unchecked(r3 as usize),
            )
        }
    }

    // =========================================================================
    // Instruction Fetching
    // =========================================================================

    /// Fetch the current instruction and advance IP.
    #[inline(always)]
    pub fn fetch(&mut self) -> prism_code::Instruction {
        let inst = unsafe { *self.code.instructions.get_unchecked(self.ip as usize) };
        self.ip += 1;
        inst
    }

    /// Peek at the current instruction without advancing.
    #[inline(always)]
    pub fn peek(&self) -> prism_code::Instruction {
        unsafe { *self.code.instructions.get_unchecked(self.ip as usize) }
    }

    /// Check if execution is complete (IP past end of instructions).
    #[inline(always)]
    pub fn is_done(&self) -> bool {
        self.ip as usize >= self.code.instructions.len()
    }

    // =========================================================================
    // Constant/Name Access
    // =========================================================================

    /// Get a constant from the constant pool.
    #[inline(always)]
    pub fn get_const(&self, idx: u16) -> Value {
        match unsafe { self.code.constants.get_unchecked(idx as usize) } {
            Constant::Value(value) => *value,
            Constant::BigInt(value) => bigint_to_value(value.clone()),
        }
    }

    /// Get a name from the name table.
    #[inline(always)]
    pub fn get_name(&self, idx: u16) -> &Arc<str> {
        unsafe { self.code.names.get_unchecked(idx as usize) }
    }

    /// Get a local variable name.
    #[inline(always)]
    pub fn get_local_name(&self, idx: u16) -> &Arc<str> {
        unsafe { self.code.locals.get_unchecked(idx as usize) }
    }

    /// Get the unique CodeId for this frame's code object.
    ///
    /// This is used for IC site identification and type feedback collection.
    #[inline(always)]
    pub fn code_id(&self) -> crate::profiler::CodeId {
        crate::profiler::CodeId::from_ptr(std::sync::Arc::as_ptr(&self.code) as *const ())
    }

    // =========================================================================
    // Generator Support
    // =========================================================================

    /// Set the yield point for generator suspension.
    ///
    /// The yield point is an index into the resume table, enabling O(1)
    /// dispatch when the generator is resumed.
    #[inline(always)]
    pub fn set_yield_point(&mut self, point: u32) {
        self.yield_point = point;
    }

    /// Get the current yield point.
    ///
    /// Returns 0 if the frame is not a generator or hasn't yielded yet.
    #[inline(always)]
    pub fn get_yield_point(&self) -> u32 {
        self.yield_point
    }

    /// Check if this frame is a suspended generator.
    #[inline(always)]
    pub fn is_generator_suspended(&self) -> bool {
        self.yield_point != 0
    }

    /// Clear the yield point (used when resuming).
    #[inline(always)]
    pub fn clear_yield_point(&mut self) {
        self.yield_point = 0;
    }
}
