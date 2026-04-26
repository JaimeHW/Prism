//! Call frame management with stack-based register file.
//!
//! The Frame struct is the core execution context for a function call.
//! It uses a stack-allocated register file for maximum performance.

use crate::exception::InlineHandlerCache;
use crate::import::ModuleObject;
use prism_code::{CodeObject, Constant};
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

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.free_frames.len()
    }
}

/// Closure environment holding captured variables.
///
/// Closures share their environment with the enclosing scope,
/// allowing mutations to be visible across closure invocations.
///
/// # Optimization: Inline Cell Storage
///
/// Most closures capture ≤4 variables (Python statistics). This struct
/// uses inline storage for the common case, avoiding heap allocation.
/// Larger closures fall back to overflow storage.
///
/// # Memory Layout
///
/// - inline_cells: 4 × Arc<Cell> inline (32 bytes on 64-bit)
/// - cell_count: u8 (1 byte)
/// - overflow: Option<Arc<[Arc<Cell>]>> for >4 cells
///
/// # Thread Safety
///
/// ClosureEnv is immutable (cells are shared via Arc). The Cell objects
/// themselves are thread-safe, allowing concurrent closure invocations.
#[derive(Clone)]
pub struct ClosureEnv {
    /// Inline storage for ≤4 cells (common case).
    /// Using MaybeUninit for efficiency when not all slots are used.
    inline_cells: [Option<Arc<prism_runtime::types::Cell>>; 4],
    /// Number of cells in this environment (0-255).
    cell_count: u8,
    /// Overflow storage for >4 cells.
    overflow: Option<Arc<[Arc<prism_runtime::types::Cell>]>>,
}

impl ClosureEnv {
    /// Maximum number of inline cells.
    pub const INLINE_LIMIT: usize = 4;

    /// Create a new closure environment with the given cells.
    ///
    /// # Arguments
    ///
    /// * `cells` - Vector of cells to capture
    ///
    /// # Returns
    ///
    /// A new closure environment containing the cells.
    pub fn new(cells: Vec<Arc<prism_runtime::types::Cell>>) -> Self {
        let cell_count = cells.len() as u8;

        if cells.len() <= Self::INLINE_LIMIT {
            // Inline case: store directly in struct
            let mut inline_cells: [Option<Arc<prism_runtime::types::Cell>>; 4] = Default::default();
            for (i, cell) in cells.into_iter().enumerate() {
                inline_cells[i] = Some(cell);
            }
            Self {
                inline_cells,
                cell_count,
                overflow: None,
            }
        } else {
            // Overflow case: store in heap-allocated slice
            Self {
                inline_cells: Default::default(),
                cell_count,
                overflow: Some(cells.into()),
            }
        }
    }

    /// Create an empty closure environment.
    #[inline]
    pub fn empty() -> Self {
        Self {
            inline_cells: Default::default(),
            cell_count: 0,
            overflow: None,
        }
    }

    /// Create a closure environment with pre-initialized unbound cells.
    ///
    /// Used when creating cells for local variables that will be captured.
    pub fn with_unbound_cells(count: usize) -> Self {
        let cells: Vec<Arc<prism_runtime::types::Cell>> = (0..count)
            .map(|_| Arc::new(prism_runtime::types::Cell::unbound()))
            .collect();
        Self::new(cells)
    }

    /// Get a cell from the environment.
    ///
    /// # Arguments
    ///
    /// * `idx` - Index of the cell to retrieve
    ///
    /// # Panics
    ///
    /// Panics if `idx >= cell_count` in debug builds.
    #[inline]
    pub fn get_cell(&self, idx: usize) -> &Arc<prism_runtime::types::Cell> {
        debug_assert!(idx < self.cell_count as usize);

        if self.overflow.is_none() {
            // Inline access
            unsafe {
                self.inline_cells
                    .get_unchecked(idx)
                    .as_ref()
                    .unwrap_unchecked()
            }
        } else {
            // Overflow access
            unsafe { self.overflow.as_ref().unwrap_unchecked().get_unchecked(idx) }
        }
    }

    /// Get the value from a cell.
    ///
    /// Convenience method that dereferences the cell.
    #[inline]
    pub fn get(&self, idx: usize) -> Value {
        self.get_cell(idx).get_or_none()
    }

    /// Set the value in a cell.
    ///
    /// Note: This mutates the shared cell, affecting all closures
    /// that captured this variable.
    #[inline]
    pub fn set(&self, idx: usize, value: Value) {
        self.get_cell(idx).set(value);
    }

    /// Get the number of cells in this environment.
    #[inline]
    pub fn len(&self) -> usize {
        self.cell_count as usize
    }

    /// Check if the environment is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.cell_count == 0
    }

    /// Check if using inline storage.
    #[inline]
    pub fn is_inline(&self) -> bool {
        self.overflow.is_none()
    }
}

impl std::fmt::Debug for ClosureEnv {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureEnv")
            .field("cell_count", &self.cell_count)
            .field("is_inline", &self.is_inline())
            .finish()
    }
}

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
        self.locals_mapping = None;
        self.yield_point = 0;
        self.handler_cache = InlineHandlerCache::new();
    }

    fn prepare_for_pool(&mut self) {
        self.clear_register_window(self.active_register_count.into());
        self.written_registers.fill(0);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_size() {
        // Frame should be approximately 2KB + overhead
        let size = std::mem::size_of::<Frame>();
        // 256 * 8 (registers) + code arc + ip + return info + closure
        assert!(size >= REGISTER_COUNT * 8);
        assert!(size < 4096, "Frame too large: {} bytes", size);
    }

    #[test]
    fn test_register_access() {
        let code = Arc::new(CodeObject::new("test", "test.py"));
        let mut frame = Frame::new(code, None, 0);

        // Test set/get
        frame.set_reg(0, Value::int(42).unwrap());
        assert_eq!(frame.get_reg(0).as_int(), Some(42));

        // Test boundary registers
        frame.set_reg(255, Value::float(3.14));
        assert!((frame.get_reg(255).as_float().unwrap() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_register_multi_access() {
        let code = Arc::new(CodeObject::new("test", "test.py"));
        let mut frame = Frame::new(code, None, 0);

        frame.set_reg(1, Value::int(10).unwrap());
        frame.set_reg(2, Value::int(20).unwrap());
        frame.set_reg(3, Value::int(30).unwrap());

        let (a, b) = frame.get_regs2(1, 2);
        assert_eq!(a.as_int(), Some(10));
        assert_eq!(b.as_int(), Some(20));

        let (x, y, z) = frame.get_regs3(1, 2, 3);
        assert_eq!(x.as_int(), Some(10));
        assert_eq!(y.as_int(), Some(20));
        assert_eq!(z.as_int(), Some(30));
    }

    #[test]
    fn test_frame_tracks_active_register_count() {
        let mut code = CodeObject::new("test", "test.py");
        code.register_count = 7;
        let frame = Frame::new(Arc::new(code), None, 0);
        assert_eq!(frame.active_register_count(), 7);
    }

    #[test]
    fn test_frame_pool_reuses_and_clears_live_window() {
        let mut pool = FramePool::new();
        let mut code = CodeObject::new("pool", "test.py");
        code.register_count = 2;

        let mut frame = pool.acquire(Arc::new(code), None, 0, None, None);
        frame.set_reg(0, Value::int(41).unwrap());
        frame.set_reg(1, Value::int(99).unwrap());
        pool.release(frame);

        assert_eq!(pool.len(), 1);

        let mut code = CodeObject::new("pool-reuse", "test.py");
        code.register_count = 2;
        let frame = pool.acquire(Arc::new(code), None, 0, None, None);
        assert!(frame.get_reg(0).is_none());
        assert!(frame.get_reg(1).is_none());
        assert!(!frame.reg_is_written(0));
        assert!(!frame.reg_is_written(1));
    }

    #[test]
    fn test_mark_reg_written_preserves_value() {
        let code = Arc::new(CodeObject::new("test", "test.py"));
        let mut frame = Frame::new(code, None, 0);

        assert!(frame.get_reg(0).is_none());
        assert!(!frame.reg_is_written(0));

        frame.mark_reg_written(0);

        assert!(frame.get_reg(0).is_none());
        assert!(frame.reg_is_written(0));
    }

    #[test]
    fn test_frame_pool_clears_registers_before_storing_frame() {
        let mut pool = FramePool::new();
        let mut code = CodeObject::new("pool-store", "test.py");
        code.register_count = 3;

        let mut frame = pool.acquire(Arc::new(code), None, 0, None, None);
        frame.set_reg(0, Value::int(1).unwrap());
        frame.set_reg(2, Value::int(3).unwrap());
        pool.release(frame);

        let pooled = pool
            .free_frames
            .last()
            .expect("frame should be stored in pool");
        assert!(pooled.get_reg(0).is_none());
        assert!(pooled.get_reg(2).is_none());
        assert!(!pooled.reg_is_written(0));
        assert!(!pooled.reg_is_written(2));
        assert_eq!(pooled.active_register_count(), 0);
    }

    #[test]
    fn test_frame_pool_clears_larger_future_window() {
        let mut pool = FramePool::new();

        let mut large_code = CodeObject::new("large", "test.py");
        large_code.register_count = 4;
        let mut frame = pool.acquire(Arc::new(large_code), None, 0, None, None);
        frame.set_reg(3, Value::int(7).unwrap());
        pool.release(frame);

        let mut small_code = CodeObject::new("small", "test.py");
        small_code.register_count = 1;
        let frame = pool.acquire(Arc::new(small_code), None, 0, None, None);
        assert!(frame.get_reg(3).is_none());
    }

    #[test]
    fn test_closure_env_inline() {
        // Test inline storage (≤4 cells)
        let env = ClosureEnv::with_unbound_cells(3);
        assert!(env.is_inline());
        assert_eq!(env.len(), 3);

        // Set values through cells
        env.set(0, Value::int(100).unwrap());
        env.set(1, Value::float(2.5));
        env.set(2, Value::none());

        assert_eq!(env.get(0).as_int(), Some(100));
        assert!((env.get(1).as_float().unwrap() - 2.5).abs() < 0.001);
        assert!(env.get(2).is_none());
    }

    #[test]
    fn test_closure_env_overflow() {
        // Test overflow storage (>4 cells)
        let env = ClosureEnv::with_unbound_cells(6);
        assert!(!env.is_inline());
        assert_eq!(env.len(), 6);

        for i in 0..6 {
            env.set(i, Value::int(i as i64).unwrap());
        }

        for i in 0..6 {
            assert_eq!(env.get(i).as_int(), Some(i as i64));
        }
    }

    #[test]
    fn test_closure_env_empty() {
        let env = ClosureEnv::empty();
        assert!(env.is_empty());
        assert_eq!(env.len(), 0);
        assert!(env.is_inline());
    }

    #[test]
    fn test_closure_env_shared_mutation() {
        // Test that cells are shared (mutations visible across closures)
        let cell = Arc::new(prism_runtime::types::Cell::new(Value::int(42).unwrap()));
        let cells = vec![Arc::clone(&cell)];
        let env = ClosureEnv::new(cells);

        // Modify through env
        env.set(0, Value::int(100).unwrap());

        // Should be visible through original cell
        assert_eq!(cell.get().unwrap().as_int(), Some(100));
    }

    #[test]
    fn test_closure_env_clone() {
        let env1 = ClosureEnv::with_unbound_cells(2);
        env1.set(0, Value::int(42).unwrap());

        let env2 = env1.clone();

        // Cloned env shares the same cells
        assert_eq!(env2.get(0).as_int(), Some(42));

        // Mutation in env2 visible in env1 (shared cells)
        env2.set(0, Value::int(99).unwrap());
        assert_eq!(env1.get(0).as_int(), Some(99));
    }
}
