//! Function builder for bytecode emission.
//!
//! The `FunctionBuilder` provides a high-level API for constructing bytecode
//! with automatic register allocation and label resolution.

use super::code_object::{CodeFlags, CodeObject, Constant, ExceptionEntry, LineTableEntry};
use super::instruction::{ConstIndex, Instruction, LocalSlot, Opcode, Register};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use prism_core::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// A tuple of keyword argument names for CallKw instructions.
///
/// This is stored in the constant pool and referenced by CallKwEx.
/// Using a dedicated struct allows efficient lookup during argument binding.
#[derive(Debug, Clone)]
pub struct KwNamesTuple {
    /// Keyword argument names in call order.
    pub names: Box<[Arc<str>]>,
}

impl KwNamesTuple {
    /// Create a new keyword names tuple.
    pub fn new(names: Vec<Arc<str>>) -> Self {
        Self {
            names: names.into_boxed_slice(),
        }
    }

    /// Get the number of keyword arguments.
    #[inline]
    pub fn len(&self) -> usize {
        self.names.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Get a keyword name by index.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Arc<str>> {
        self.names.get(index)
    }

    /// Iterate over keyword names.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Arc<str>> {
        self.names.iter()
    }
}

/// A label for jump targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Label(u32);

/// A forward reference to a label that needs patching.
#[derive(Debug)]
struct ForwardRef {
    /// Instruction index containing the jump.
    instruction_index: usize,
    /// The label being jumped to.
    label: Label,
}

/// Builder for constructing code objects.
///
/// This provides a high-level interface for:
/// - Emitting bytecode instructions
/// - Managing virtual registers
/// - Defining and resolving labels
/// - Managing constant and name pools
///
/// # Example
/// ```ignore
/// let mut builder = FunctionBuilder::new("add");
/// let r0 = builder.alloc_register(); // x
/// let r1 = builder.alloc_register(); // y
/// let r2 = builder.alloc_register(); // result
///
/// builder.emit_load_local(r0, 0); // load x
/// builder.emit_load_local(r1, 1); // load y
/// builder.emit_add(r2, r0, r1);   // r2 = x + y
/// builder.emit_return(r2);
///
/// let code = builder.finish();
/// ```
pub struct FunctionBuilder {
    /// Function name.
    name: Arc<str>,
    /// Qualified name.
    qualname: Arc<str>,
    /// Filename.
    filename: Arc<str>,
    /// First line number.
    first_lineno: u32,
    /// Current line number (for line table).
    current_line: u32,

    /// Emitted instructions.
    instructions: Vec<Instruction>,

    /// Constant pool.
    constants: Vec<Constant>,
    /// Constant deduplication map.
    constant_map: HashMap<ConstantKey, ConstIndex>,

    /// Local variable names.
    locals: Vec<Arc<str>>,
    /// Local name to slot map.
    local_map: HashMap<Arc<str>, LocalSlot>,

    /// Global/attribute names.
    names: Vec<Arc<str>>,
    /// Name to index map.
    name_map: HashMap<Arc<str>, u16>,

    /// Free variable names.
    freevars: Vec<Arc<str>>,
    /// Cell variable names.
    cellvars: Vec<Arc<str>>,

    /// Number of parameters.
    arg_count: u16,
    posonlyarg_count: u16,
    kwonlyarg_count: u16,

    /// Code flags.
    flags: CodeFlags,
    /// Whether local slots are stored separately from the register file.
    separate_locals: bool,

    /// Next register to allocate.
    next_register: u8,
    /// Maximum registers used (high water mark).
    max_registers: u8,
    /// Register free list for reuse.
    free_registers: Vec<Register>,

    /// Label counter.
    next_label: u32,
    /// Label to instruction index map.
    labels: HashMap<Label, usize>,
    /// Forward references that need patching.
    forward_refs: Vec<ForwardRef>,

    /// Line number table entries.
    line_table: Vec<LineTableEntry>,
    /// Start PC for current line.
    line_start_pc: u32,
    /// Exception table entries.
    exception_entries: Vec<ExceptionEntry>,
    /// Nested code objects (functions, classes defined within this code).
    nested_code_objects: Vec<Arc<CodeObject>>,
}

/// Key type for constant deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstantKey {
    None,
    Bool(bool),
    Int(i64),
    BigInt(BigInt),
    /// Float bits for exact comparison.
    Float(u64),
    String(Arc<str>),
    /// Tuple of strings (for keyword argument names).
    KwNamesTuple(Box<[Arc<str>]>),
}

impl ConstantKey {
    fn from_value(value: &Value) -> Option<Self> {
        // We can only deduplicate simple types
        if value.is_none() {
            Some(ConstantKey::None)
        } else if let Some(b) = value.as_bool() {
            Some(ConstantKey::Bool(b))
        } else if let Some(i) = value.as_int() {
            Some(ConstantKey::Int(i))
        } else if let Some(f) = value.as_float() {
            Some(ConstantKey::Float(f.to_bits()))
        } else {
            // Can't deduplicate complex types (objects, lists, etc.)
            None
        }
    }
}

impl FunctionBuilder {
    /// Inline attribute-name operand value that signals a trailing AttrName extension.
    const EXTENDED_ATTR_NAME_SENTINEL: u8 = u8::MAX;

    /// Create a new function builder.
    pub fn new(name: impl Into<Arc<str>>) -> Self {
        let name = name.into();
        Self {
            qualname: name.clone(),
            name,
            filename: "<unknown>".into(),
            first_lineno: 1,
            current_line: 1,
            instructions: Vec::new(),
            constants: Vec::new(),
            constant_map: HashMap::new(),
            locals: Vec::new(),
            local_map: HashMap::new(),
            names: Vec::new(),
            name_map: HashMap::new(),
            freevars: Vec::new(),
            cellvars: Vec::new(),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            flags: CodeFlags::NONE,
            separate_locals: false,
            next_register: 0,
            max_registers: 0,
            free_registers: Vec::new(),
            next_label: 0,
            labels: HashMap::new(),
            forward_refs: Vec::new(),
            line_table: Vec::new(),
            line_start_pc: 0,
            exception_entries: Vec::new(),
            nested_code_objects: Vec::new(),
        }
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    /// Set the qualified name.
    pub fn set_qualname(&mut self, qualname: impl Into<Arc<str>>) {
        self.qualname = qualname.into();
    }

    /// Set the filename.
    pub fn set_filename(&mut self, filename: impl Into<Arc<str>>) {
        self.filename = filename.into();
    }

    /// Get the filename.
    #[inline]
    pub fn get_filename(&self) -> Arc<str> {
        self.filename.clone()
    }

    /// Set the first line number.
    pub fn set_first_lineno(&mut self, line: u32) {
        self.first_lineno = line;
        self.current_line = line;
    }

    /// Add a flag to the code object.
    pub fn add_flag(&mut self, flag: CodeFlags) {
        self.flags = self.flags | flag;
    }

    /// Set the current line number for subsequent instructions.
    pub fn set_line(&mut self, line: u32) {
        if line != self.current_line {
            // Record the previous line range
            let current_pc = self.instructions.len() as u32;
            if current_pc > self.line_start_pc {
                self.line_table.push(LineTableEntry {
                    start_pc: self.line_start_pc,
                    end_pc: current_pc,
                    line: self.current_line,
                });
            }
            self.current_line = line;
            self.line_start_pc = current_pc;
        }
    }

    /// Set the number of parameters.
    pub fn set_arg_count(&mut self, count: u16) {
        self.arg_count = count;
    }

    /// Set the number of positional-only parameters.
    pub fn set_posonlyarg_count(&mut self, count: u16) {
        self.posonlyarg_count = count;
    }

    /// Set the number of keyword-only parameters.
    pub fn set_kwonlyarg_count(&mut self, count: u16) {
        self.kwonlyarg_count = count;
    }

    /// Add code flags.
    pub fn add_flags(&mut self, flags: CodeFlags) {
        self.flags |= flags;
    }

    /// Store locals in a dedicated frame-local array instead of reserving a
    /// register for every local slot.
    ///
    /// This mode is used for large Python functions where CPython permits
    /// hundreds or thousands of parameters. It keeps the fixed 8-bit register
    /// bytecode format hot for ordinary functions while preserving correct
    /// semantics for oversized local layouts.
    pub fn use_separate_locals(&mut self) {
        self.separate_locals = true;
        self.flags |= CodeFlags::SEPARATE_LOCALS;
    }

    // =========================================================================
    // Register Management
    // =========================================================================

    /// Allocate a new virtual register.
    #[inline]
    pub fn alloc_register(&mut self) -> Register {
        // Try to reuse a freed register first
        if let Some(reg) = self.free_registers.pop() {
            return reg;
        }

        let reg = Register(self.next_register);
        self.next_register = self
            .next_register
            .checked_add(1)
            .expect("register overflow");
        self.max_registers = self.max_registers.max(self.next_register);
        reg
    }

    /// Free a register for reuse.
    #[inline]
    pub fn free_register(&mut self, reg: Register) {
        if !self.separate_locals && (reg.0 as usize) < self.locals.len() {
            return;
        }
        self.free_registers.push(reg);
    }

    /// Reserve registers for function parameters.
    pub fn reserve_parameters(&mut self, count: u16) {
        for _ in 0..count {
            self.alloc_register();
        }
    }

    /// Allocate a contiguous block of registers for function calls.
    ///
    /// This reserves `count` consecutive registers. The allocator first tries to
    /// reclaim a fully free contiguous span from the free list; if none exists,
    /// it extends the high-water mark.
    ///
    /// Call instructions use consecutive registers: [result, arg0, arg1, ...]
    /// This method ensures all those registers are properly reserved.
    ///
    /// # Returns
    /// The base register of the block. Registers [base..base+count) are reserved.
    #[inline]
    pub fn alloc_register_block(&mut self, count: u8) -> Register {
        if let Some(base) = self.take_contiguous_free_block(count) {
            return base;
        }

        let base = Register(self.next_register);
        self.next_register = self
            .next_register
            .checked_add(count)
            .expect("register overflow");
        self.max_registers = self.max_registers.max(self.next_register);
        base
    }

    fn take_contiguous_free_block(&mut self, count: u8) -> Option<Register> {
        if count == 0 || self.free_registers.len() < count as usize {
            return None;
        }

        let mut entries: Vec<(usize, u8)> = self
            .free_registers
            .iter()
            .enumerate()
            .map(|(index, register)| (index, register.0))
            .collect();
        entries.sort_unstable_by_key(|&(_, register)| register);

        let span_len = count as usize;
        for start in 0..=entries.len() - span_len {
            let base = entries[start].1;
            let is_contiguous = entries[start..start + span_len].iter().enumerate().all(
                |(offset, &(_, register))| {
                    base.checked_add(offset as u8)
                        .map(|expected| register == expected)
                        .unwrap_or(false)
                },
            );
            if !is_contiguous {
                continue;
            }

            let mut removal_indices: Vec<usize> = entries[start..start + span_len]
                .iter()
                .map(|&(index, _)| index)
                .collect();
            removal_indices.sort_unstable_by(|left, right| right.cmp(left));
            for index in removal_indices {
                self.free_registers.swap_remove(index);
            }
            return Some(Register(base));
        }

        None
    }

    /// Free a contiguous block of registers (for cleanup after call).
    #[inline]
    pub fn free_register_block(&mut self, base: Register, count: u8) {
        for i in 0..count {
            self.free_register(Register(base.0 + i));
        }
    }

    // =========================================================================
    // Constant Pool
    // =========================================================================

    /// Add a constant and return its index.
    pub fn add_constant(&mut self, value: Value) -> ConstIndex {
        // Try deduplication for simple types
        if let Some(key) = ConstantKey::from_value(&value) {
            if let Some(&idx) = self.constant_map.get(&key) {
                return idx;
            }
            let idx = ConstIndex::new(self.constants.len() as u16);
            self.constants.push(Constant::Value(value));
            self.constant_map.insert(key, idx);
            idx
        } else {
            // No deduplication for complex types
            let idx = ConstIndex::new(self.constants.len() as u16);
            self.constants.push(Constant::Value(value));
            idx
        }
    }

    /// Add an integer constant.
    pub fn add_int(&mut self, value: i64) -> ConstIndex {
        match Value::int(value) {
            Some(value) => self.add_constant(value),
            None => self.add_bigint(BigInt::from(value)),
        }
    }

    /// Add an arbitrary-precision integer constant.
    pub fn add_bigint(&mut self, value: BigInt) -> ConstIndex {
        if let Some(i) = value
            .to_i64()
            .filter(|candidate| Value::int(*candidate).is_some())
        {
            return self.add_int(i);
        }

        let key = ConstantKey::BigInt(value.clone());
        if let Some(&idx) = self.constant_map.get(&key) {
            return idx;
        }

        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants.push(Constant::BigInt(value));
        self.constant_map.insert(key, idx);
        idx
    }

    /// Add a float constant.
    pub fn add_float(&mut self, value: f64) -> ConstIndex {
        self.add_constant(Value::float(value))
    }

    /// Add a string constant with automatic interning and deduplication.
    ///
    /// This method:
    /// 1. Interns the string using the global string interner for O(1) equality
    /// 2. Deduplicates identical strings in the constant pool
    /// 3. Returns an index suitable for LoadConst instruction
    ///
    /// # Performance
    ///
    /// - O(1) lookup for already-added strings via ConstantKey::String deduplication
    /// - Interned strings enable pointer equality at runtime
    ///
    /// # Example
    ///
    /// ```ignore
    /// let idx = builder.add_string("hello");
    /// builder.emit_load_const(dst, idx);
    /// ```
    pub fn add_string(&mut self, s: impl AsRef<str>) -> ConstIndex {
        let s_str = s.as_ref();
        let arc_str: Arc<str> = Arc::from(s_str);
        let key = ConstantKey::String(arc_str.clone());

        // Deduplication: return existing index if this string was already added
        if let Some(&idx) = self.constant_map.get(&key) {
            return idx;
        }

        // Intern the string for runtime O(1) equality checks
        let interned = prism_core::intern::intern(s_str);
        let value = Value::string(interned);

        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants.push(Constant::Value(value));
        self.constant_map.insert(key, idx);
        idx
    }

    /// Add a nested code object constant.
    ///
    /// Code objects are stored in the constant pool for MakeFunction/MakeClosure
    /// to create function objects at runtime.
    ///
    /// Returns the constant index that can be used with MakeFunction/MakeClosure opcodes.
    pub fn add_code_object(&mut self, code: Arc<CodeObject>) -> u16 {
        // Store the Arc<CodeObject> as an object pointer constant
        // At runtime, the VM will interpret this as a code object reference
        let code_ptr = Arc::into_raw(Arc::clone(&code)) as *const ();
        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants
            .push(Constant::Value(Value::object_ptr(code_ptr)));

        // Store Arc in nested_code_objects for test accessibility
        self.nested_code_objects.push(code);

        idx.0
    }

    // =========================================================================
    // Local Variables
    // =========================================================================

    /// Define a local variable and return its slot.
    pub fn define_local(&mut self, name: impl Into<Arc<str>>) -> LocalSlot {
        let name = name.into();
        if let Some(&slot) = self.local_map.get(&name) {
            return slot;
        }
        let slot = LocalSlot::new(self.locals.len() as u16);
        self.local_map.insert(name.clone(), slot);
        self.locals.push(name);
        if !self.separate_locals {
            // In the compact hot-frame layout locals are addressed by register
            // index at runtime. Reserve the backing register so temporary
            // allocation never clobbers locals.
            let required = (slot.0 as usize) + 1;
            if required > self.next_register as usize {
                self.next_register = u8::try_from(required).expect("register overflow");
                self.max_registers = self.max_registers.max(self.next_register);
            }
            self.free_registers
                .retain(|reg| (reg.0 as usize) >= self.locals.len());
        }
        slot
    }

    /// Look up a local variable by name.
    pub fn lookup_local(&self, name: &str) -> Option<LocalSlot> {
        self.local_map.get(name).copied()
    }

    // =========================================================================
    // Names (globals, attributes)
    // =========================================================================

    /// Add a name and return its index.
    pub fn add_name(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        if let Some(&idx) = self.name_map.get(&name) {
            return idx;
        }
        let idx = self.names.len() as u16;
        self.name_map.insert(name.clone(), idx);
        self.names.push(name);
        idx
    }

    // =========================================================================
    // Labels
    // =========================================================================

    /// Create a new label for a jump target.
    pub fn create_label(&mut self) -> Label {
        let label = Label(self.next_label);
        self.next_label += 1;
        label
    }

    /// Mark the current position as the target for a label.
    pub fn bind_label(&mut self, label: Label) {
        let pc = self.instructions.len();
        self.labels.insert(label, pc);
    }

    /// Get the current instruction offset (for relative jumps).
    pub fn current_offset(&self) -> usize {
        self.instructions.len()
    }

    // =========================================================================
    // Instruction Emission
    // =========================================================================

    /// Emit a raw instruction.
    #[inline]
    pub fn emit(&mut self, inst: Instruction) {
        self.instructions.push(inst);
    }

    /// Emit a NOP instruction.
    pub fn emit_nop(&mut self) {
        self.emit(Instruction::op(Opcode::Nop));
    }

    // --- Load/Store ---

    /// Load a constant into a register.
    pub fn emit_load_const(&mut self, dst: Register, idx: ConstIndex) {
        self.emit(Instruction::op_di(Opcode::LoadConst, dst, idx.0));
    }

    /// Load None into a register.
    pub fn emit_load_none(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadNone, dst));
    }

    /// Load True into a register.
    pub fn emit_load_true(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadTrue, dst));
    }

    /// Load False into a register.
    pub fn emit_load_false(&mut self, dst: Register) {
        self.emit(Instruction::op_d(Opcode::LoadFalse, dst));
    }

    /// Load a local variable into a register.
    pub fn emit_load_local(&mut self, dst: Register, slot: LocalSlot) {
        self.emit(Instruction::op_di(Opcode::LoadLocal, dst, slot.0));
    }

    /// Store a register into a local variable.
    pub fn emit_store_local(&mut self, slot: LocalSlot, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreLocal, src, slot.0));
    }

    /// Load a global variable into a register.
    pub fn emit_load_global(&mut self, dst: Register, name_idx: u16) {
        self.emit(Instruction::op_di(Opcode::LoadGlobal, dst, name_idx));
    }

    /// Load a builtin directly from the builtin registry into a register.
    pub fn emit_load_builtin(&mut self, dst: Register, name_idx: u16) {
        self.emit(Instruction::op_di(Opcode::LoadBuiltin, dst, name_idx));
    }

    /// Ensure the current module/class namespace has a `__annotations__` dict.
    pub fn emit_setup_annotations(&mut self) {
        self.emit(Instruction::op(Opcode::SetupAnnotations));
    }

    /// Store a register into a global variable.
    pub fn emit_store_global(&mut self, name_idx: u16, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreGlobal, src, name_idx));
    }

    // --- Closure Variables ---

    /// Load a closure variable (cell or free) into a register.
    ///
    /// Closure slot indices are determined by scope analysis:
    /// - Cell variables (captured by inner scopes) come first
    /// - Free variables (captured from outer scopes) follow
    #[inline]
    pub fn emit_load_closure(&mut self, dst: Register, slot: u16) {
        self.emit(Instruction::op_di(Opcode::LoadClosure, dst, slot));
    }

    /// Store a register into a closure variable.
    ///
    /// Uses Cell interior mutability for proper closure semantics.
    #[inline]
    pub fn emit_store_closure(&mut self, slot: u16, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreClosure, src, slot));
    }

    /// Delete (mark as unbound) a closure variable.
    ///
    /// Subsequent reads will raise UnboundLocalError.
    #[inline]
    pub fn emit_delete_closure(&mut self, slot: u16) {
        self.emit(Instruction::op_di(
            Opcode::DeleteClosure,
            Register::new(0),
            slot,
        ));
    }

    /// Add a cell variable (captured by inner scopes).
    ///
    /// Returns the closure slot index for this cell.
    pub fn add_cellvar(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        let slot = self.cellvars.len() as u16;
        self.cellvars.push(name);
        slot
    }

    /// Add a free variable (captured from outer scope).
    ///
    /// Returns the closure slot index for this freevar.
    /// Note: Free variables are indexed after all cell variables.
    pub fn add_freevar(&mut self, name: impl Into<Arc<str>>) -> u16 {
        let name = name.into();
        // Free vars come after cell vars in the closure environment
        let slot = (self.cellvars.len() + self.freevars.len()) as u16;
        self.freevars.push(name);
        slot
    }

    /// Get the number of cell variables.
    #[inline]
    pub fn cellvar_count(&self) -> usize {
        self.cellvars.len()
    }

    /// Get the number of free variables.
    #[inline]
    pub fn freevar_count(&self) -> usize {
        self.freevars.len()
    }

    /// Check if this function has any closure variables.
    #[inline]
    pub fn has_closure(&self) -> bool {
        !self.cellvars.is_empty() || !self.freevars.is_empty()
    }

    // --- Class Construction ---

    /// Emit BUILD_CLASS and its metadata extension.
    ///
    /// The primary instruction stores the class-body code constant index in a
    /// full 16-bit immediate. A trailing `ClassMeta` extension stores the base
    /// count, keeping class construction resilient to large constant pools in
    /// real stdlib modules.
    pub fn emit_build_class(&mut self, dst: Register, code_idx: u16, base_count: u8) {
        self.emit(Instruction::op_di(Opcode::BuildClass, dst, code_idx));
        self.emit(Instruction::op_d(
            Opcode::ClassMeta,
            Register::new(base_count),
        ));
    }

    /// Emit BUILD_CLASS_WITH_META and its metadata extension.
    ///
    /// Register layout:
    /// - `dst` = result class object
    /// - `dst+1..dst+base_count` = base classes
    /// - `dst+1+base_count` = explicit metaclass value
    pub fn emit_build_class_with_meta(&mut self, dst: Register, code_idx: u16, base_count: u8) {
        self.emit(Instruction::op_di(
            Opcode::BuildClassWithMeta,
            dst,
            code_idx,
        ));
        self.emit(Instruction::op_d(
            Opcode::ClassMeta,
            Register::new(base_count),
        ));
    }

    /// Move value between registers.
    pub fn emit_move(&mut self, dst: Register, src: Register) {
        if dst != src {
            self.emit(Instruction::op_ds(Opcode::Move, dst, src));
        }
    }

    // --- Arithmetic ---

    /// Generic add: dst = src1 + src2.
    pub fn emit_add(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Add, dst, src1, src2));
    }

    /// Generic subtract: dst = src1 - src2.
    pub fn emit_sub(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Sub, dst, src1, src2));
    }

    /// Generic multiply: dst = src1 * src2.
    pub fn emit_mul(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Mul, dst, src1, src2));
    }

    /// Generic true divide: dst = src1 / src2.
    pub fn emit_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::TrueDiv, dst, src1, src2));
    }

    /// Generic floor divide: dst = src1 // src2.
    pub fn emit_floor_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::FloorDiv, dst, src1, src2));
    }

    /// Generic modulo: dst = src1 % src2.
    pub fn emit_mod(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Mod, dst, src1, src2));
    }

    /// Generic power: dst = src1 ** src2.
    pub fn emit_pow(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Pow, dst, src1, src2));
    }

    /// Generic matrix multiply: dst = src1 @ src2.
    pub fn emit_matmul(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::MatMul, dst, src1, src2));
    }

    /// In-place add: dst = src1; dst += src2.
    pub fn emit_inplace_add(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceAdd, dst, src1, src2));
    }

    /// In-place subtract: dst = src1; dst -= src2.
    pub fn emit_inplace_sub(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceSub, dst, src1, src2));
    }

    /// In-place multiply: dst = src1; dst *= src2.
    pub fn emit_inplace_mul(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceMul, dst, src1, src2));
    }

    /// In-place true divide: dst = src1; dst /= src2.
    pub fn emit_inplace_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceTrueDiv, dst, src1, src2));
    }

    /// In-place floor divide: dst = src1; dst //= src2.
    pub fn emit_inplace_floor_div(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(
            Opcode::InPlaceFloorDiv,
            dst,
            src1,
            src2,
        ));
    }

    /// In-place modulo: dst = src1; dst %= src2.
    pub fn emit_inplace_mod(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceMod, dst, src1, src2));
    }

    /// In-place power: dst = src1; dst **= src2.
    pub fn emit_inplace_pow(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlacePow, dst, src1, src2));
    }

    /// Generic negate: dst = -src.
    pub fn emit_neg(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Neg, dst, src));
    }

    /// Generic unary plus: dst = +src.
    pub fn emit_pos(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Pos, dst, src));
    }

    // --- Comparison ---

    /// Less than: dst = src1 < src2.
    pub fn emit_lt(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Lt, dst, src1, src2));
    }

    /// Less than or equal: dst = src1 <= src2.
    pub fn emit_le(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Le, dst, src1, src2));
    }

    /// Equal: dst = src1 == src2.
    pub fn emit_eq(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Eq, dst, src1, src2));
    }

    /// Not equal: dst = src1 != src2.
    pub fn emit_ne(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Ne, dst, src1, src2));
    }

    /// Greater than: dst = src1 > src2.
    pub fn emit_gt(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Gt, dst, src1, src2));
    }

    /// Greater than or equal: dst = src1 >= src2.
    pub fn emit_ge(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Ge, dst, src1, src2));
    }

    // --- Bitwise ---

    /// Bitwise and: dst = src1 & src2.
    pub fn emit_bitwise_and(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseAnd, dst, src1, src2));
    }

    /// Bitwise or: dst = src1 | src2.
    pub fn emit_bitwise_or(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseOr, dst, src1, src2));
    }

    /// Bitwise xor: dst = src1 ^ src2.
    pub fn emit_bitwise_xor(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::BitwiseXor, dst, src1, src2));
    }

    /// Bitwise not: dst = ~src.
    pub fn emit_bitwise_not(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::BitwiseNot, dst, src));
    }

    /// Left shift: dst = src1 << src2.
    pub fn emit_shl(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Shl, dst, src1, src2));
    }

    /// Right shift: dst = src1 >> src2.
    pub fn emit_shr(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::Shr, dst, src1, src2));
    }

    /// In-place bitwise and: dst = src1; dst &= src2.
    pub fn emit_inplace_bitwise_and(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(
            Opcode::InPlaceBitwiseAnd,
            dst,
            src1,
            src2,
        ));
    }

    /// In-place bitwise or: dst = src1; dst |= src2.
    pub fn emit_inplace_bitwise_or(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(
            Opcode::InPlaceBitwiseOr,
            dst,
            src1,
            src2,
        ));
    }

    /// In-place bitwise xor: dst = src1; dst ^= src2.
    pub fn emit_inplace_bitwise_xor(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(
            Opcode::InPlaceBitwiseXor,
            dst,
            src1,
            src2,
        ));
    }

    /// In-place left shift: dst = src1; dst <<= src2.
    pub fn emit_inplace_shl(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceShl, dst, src1, src2));
    }

    /// In-place right shift: dst = src1; dst >>= src2.
    pub fn emit_inplace_shr(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceShr, dst, src1, src2));
    }

    /// In-place matrix multiply: dst = src1; dst @= src2.
    pub fn emit_inplace_matmul(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(Instruction::op_dss(Opcode::InPlaceMatMul, dst, src1, src2));
    }

    /// Logical not: dst = not src.
    pub fn emit_not(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Not, dst, src));
    }

    // --- Control Flow ---

    /// Return value from register.
    pub fn emit_return(&mut self, src: Register) {
        self.emit(Instruction::op_d(Opcode::Return, src));
    }

    /// Return None.
    pub fn emit_return_none(&mut self) {
        self.emit(Instruction::op(Opcode::ReturnNone));
    }

    /// Unconditional jump to label.
    pub fn emit_jump(&mut self, label: Label) {
        let inst_idx = self.instructions.len();
        // Emit placeholder, will be patched later
        self.emit(Instruction::op(Opcode::Jump));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// Jump if register is falsy.
    pub fn emit_jump_if_false(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::JumpIfFalse, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// Jump if register is truthy.
    pub fn emit_jump_if_true(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::JumpIfTrue, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    // --- Object Operations ---

    /// Get attribute: dst = obj.attr.
    pub fn emit_get_attr(&mut self, dst: Register, obj: Register, name_idx: u16) {
        let inline_name = self.emit_attr_name_operand(name_idx);
        self.emit(Instruction::new(Opcode::GetAttr, dst.0, obj.0, inline_name));
        self.emit_attr_name_extension_if_needed(inline_name, name_idx);
    }

    /// Set attribute: obj.attr = value.
    pub fn emit_set_attr(&mut self, obj: Register, name_idx: u16, value: Register) {
        let inline_name = self.emit_attr_name_operand(name_idx);
        self.emit(Instruction::new(
            Opcode::SetAttr,
            obj.0,
            inline_name,
            value.0,
        ));
        self.emit_attr_name_extension_if_needed(inline_name, name_idx);
    }

    /// Delete attribute: del obj.attr.
    pub fn emit_del_attr(&mut self, obj: Register, name_idx: u16) {
        let inline_name = self.emit_attr_name_operand(name_idx);
        self.emit(Instruction::new(Opcode::DelAttr, 0, obj.0, inline_name));
        self.emit_attr_name_extension_if_needed(inline_name, name_idx);
    }

    #[inline]
    fn emit_attr_name_operand(&self, name_idx: u16) -> u8 {
        if name_idx < Self::EXTENDED_ATTR_NAME_SENTINEL as u16 {
            name_idx as u8
        } else {
            Self::EXTENDED_ATTR_NAME_SENTINEL
        }
    }

    #[inline]
    fn emit_attr_name_extension_if_needed(&mut self, inline_name: u8, name_idx: u16) {
        if inline_name == Self::EXTENDED_ATTR_NAME_SENTINEL {
            self.emit(Instruction::op_di(
                Opcode::AttrName,
                Register::new(0),
                name_idx,
            ));
        }
    }

    /// Get item: dst = obj[key].
    pub fn emit_get_item(&mut self, dst: Register, obj: Register, key: Register) {
        self.emit(Instruction::op_dss(Opcode::GetItem, dst, obj, key));
    }

    /// Set item: obj[key] = value.
    pub fn emit_set_item(&mut self, obj: Register, key: Register, value: Register) {
        self.emit(Instruction::op_dss(Opcode::SetItem, key, obj, value));
    }

    /// Delete item: del obj[key].
    pub fn emit_del_item(&mut self, obj: Register, key: Register) {
        self.emit(Instruction::op_dss(
            Opcode::DelItem,
            Register::new(0),
            obj,
            key,
        ));
    }

    // --- Function Calls ---

    /// Call function: dst = func(args...).
    /// Args should be in registers dst+1, dst+2, etc.
    pub fn emit_call(&mut self, dst: Register, func: Register, argc: u8) {
        self.emit(Instruction::new(Opcode::Call, dst.0, func.0, argc));
    }

    /// Load method for optimized method calls: dst = obj.method (with self).
    ///
    /// This is the first half of the LoadMethod/CallMethod optimization pair.
    /// It performs method lookup and stores the method in `dst` and `self` in `dst+1`.
    ///
    /// Register layout after LoadMethod:
    /// - [dst]: method/function object
    /// - [dst+1]: self instance (or NULL marker for unbound)
    ///
    /// # Arguments
    /// * `dst` - Register to store method (dst+1 gets self automatically)
    /// * `obj` - Register containing the object to look up method on
    /// * `name_idx` - Index into names table for the method name
    #[inline]
    pub fn emit_load_method(&mut self, dst: Register, obj: Register, name_idx: u16) {
        let inline_name = self.emit_attr_name_operand(name_idx);
        self.emit(Instruction::new(
            Opcode::LoadMethod,
            dst.0,
            obj.0,
            inline_name,
        ));
        self.emit_attr_name_extension_if_needed(inline_name, name_idx);
    }

    /// Call method using result from LoadMethod: dst = method(self, args...).
    ///
    /// This is the second half of the LoadMethod/CallMethod optimization pair.
    /// It expects the method in `method_reg` and self in `method_reg+1`.
    ///
    /// Register layout expected (from LoadMethod):
    /// - [method_reg]: method/function object
    /// - [method_reg+1]: self instance
    /// - [method_reg+2..]: explicit arguments
    ///
    /// # Arguments
    /// * `dst` - Register to store the return value
    /// * `method_reg` - Register containing method (from LoadMethod)
    /// * `argc` - Number of explicit arguments (in method_reg+2, method_reg+3, etc.)
    #[inline]
    pub fn emit_call_method(&mut self, dst: Register, method_reg: Register, argc: u8) {
        self.emit(Instruction::new(
            Opcode::CallMethod,
            dst.0,
            method_reg.0,
            argc,
        ));
    }

    /// Call function with keyword arguments: dst = func(pos_args..., kw_args...).
    ///
    /// Uses a two-instruction sequence for encoding:
    /// - Instruction 1: [CallKw][dst][func][posargc]
    /// - Instruction 2: [CallKwEx][kwargc][kwnames_idx_lo][kwnames_idx_hi]
    ///
    /// Arguments layout in registers:
    /// - dst+1 .. dst+posargc: positional arguments
    /// - dst+posargc+1 .. dst+posargc+kwargc: keyword argument values
    ///
    /// Keyword names are stored as a tuple in the constant pool at kwnames_idx.
    pub fn emit_call_kw(
        &mut self,
        dst: Register,
        func: Register,
        posargc: u8,
        kwargc: u8,
        kwnames_idx: u16,
    ) {
        // First instruction: opcode, dst, func, posargc
        self.emit(Instruction::new(Opcode::CallKw, dst.0, func.0, posargc));
        // Second instruction: kwargc, kwnames_idx (split into two bytes)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            kwargc,
            (kwnames_idx & 0xFF) as u8,
            (kwnames_idx >> 8) as u8,
        ));
    }

    /// Add a tuple of keyword argument names to the constant pool.
    ///
    /// This is used for CallKw instructions to efficiently pass keyword names
    /// without dictionary allocation. Names are interned strings for fast comparison.
    ///
    /// Returns the constant pool index of the tuple.
    pub fn add_kwnames_tuple(&mut self, names: Vec<Arc<str>>) -> u16 {
        // Create deduplication key
        let key = ConstantKey::KwNamesTuple(names.clone().into_boxed_slice());

        if let Some(&idx) = self.constant_map.get(&key) {
            return idx.0;
        }

        // Create a tuple value containing the keyword names
        // For now, we use a packed representation stored as object pointer
        // The VM will interpret this as a keyword names tuple
        let tuple = KwNamesTuple::new(names);
        let tuple_ptr = Box::into_raw(Box::new(tuple)) as *const ();
        let idx = ConstIndex::new(self.constants.len() as u16);
        self.constants
            .push(Constant::Value(Value::object_ptr(tuple_ptr)));
        self.constant_map.insert(key, idx);
        idx.0
    }

    /// Call function with unpacked arguments: dst = func(*args_tuple, **kwargs_dict).
    ///
    /// Used when call site contains *args or **kwargs unpacking. Uses two instructions:
    /// - Instruction 1: [CallEx][dst][func][args_tuple_reg]
    /// - Instruction 2: [CallKwEx][kwargs_dict_reg][0][0] (or 0xFF for no kwargs)
    ///
    /// The args_tuple_reg should contain a tuple of positional arguments.
    /// The kwargs_dict_reg should contain a dict of keyword arguments (or None).
    pub fn emit_call_ex(
        &mut self,
        dst: Register,
        func: Register,
        args_tuple: Register,
        kwargs_dict: Option<Register>,
    ) {
        // First instruction: CallEx with args tuple register
        self.emit(Instruction::new(
            Opcode::CallEx,
            dst.0,
            func.0,
            args_tuple.0,
        ));
        // Second instruction as extension: kwargs dict register (0xFF = no kwargs)
        let kwargs_reg = kwargs_dict.map_or(0xFF, |r| r.0);
        self.emit(Instruction::new(Opcode::CallKwEx, kwargs_reg, 0, 0));
    }

    /// Attach default metadata to a function object.
    ///
    /// - `func`: function object register
    /// - `pos_defaults`: tuple of positional defaults or `None`
    /// - `kw_defaults`: dict of keyword-only defaults or `None`
    pub fn emit_set_function_defaults(
        &mut self,
        func: Register,
        pos_defaults: Register,
        kw_defaults: Register,
    ) {
        self.emit(Instruction::new(
            Opcode::SetFunctionDefaults,
            func.0,
            pos_defaults.0,
            kw_defaults.0,
        ));
    }

    /// Build a tuple from multiple values/iterables with unpacking.
    ///
    /// This is used for combining static positional args with *args unpacking.
    /// The unpack_flags is a bitmap where bit i indicates src+i should be unpacked.
    /// Values are sourced from consecutive registers starting at `base`.
    ///
    /// Format: [BuildTupleUnpack][dst][base][count] + [extension with flags]
    pub fn emit_build_tuple_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildTupleUnpack,
            dst.0,
            base.0,
            count,
        ));
        // Extension instruction with unpack flags (lower 24 bits)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    /// Build a dict from multiple values/mappings with unpacking.
    ///
    /// This is used for combining static keyword args with **kwargs unpacking.
    /// The unpack_flags is a bitmap where bit i indicates src+i should be unpacked.
    /// Values are sourced from consecutive registers starting at `base`.
    ///
    /// Format: [BuildDictUnpack][dst][base][count] + [extension with flags]
    pub fn emit_build_dict_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildDictUnpack,
            dst.0,
            base.0,
            count,
        ));
        // Extension instruction with unpack flags (lower 24 bits)
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    /// Build a list from multiple values/iterables with unpacking.
    ///
    /// Values are sourced from consecutive registers starting at `base`.
    /// The `unpack_flags` bitmap marks registers that should be expanded.
    pub fn emit_build_list_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildListUnpack,
            dst.0,
            base.0,
            count,
        ));
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    /// Build a set from multiple values/iterables with unpacking.
    ///
    /// Values are sourced from consecutive registers starting at `base`.
    /// The `unpack_flags` bitmap marks registers that should be expanded.
    pub fn emit_build_set_unpack(
        &mut self,
        dst: Register,
        base: Register,
        count: u8,
        unpack_flags: u32,
    ) {
        self.emit(Instruction::new(
            Opcode::BuildSetUnpack,
            dst.0,
            base.0,
            count,
        ));
        self.emit(Instruction::new(
            Opcode::CallKwEx,
            (unpack_flags & 0xFF) as u8,
            ((unpack_flags >> 8) & 0xFF) as u8,
            ((unpack_flags >> 16) & 0xFF) as u8,
        ));
    }

    // --- Container Operations ---

    /// Build list from registers.
    pub fn emit_build_list(&mut self, dst: Register, start: Register, count: u8) {
        self.emit(Instruction::new(Opcode::BuildList, dst.0, start.0, count));
    }

    /// Build tuple from registers.
    pub fn emit_build_tuple(&mut self, dst: Register, start: Register, count: u8) {
        self.emit(Instruction::new(Opcode::BuildTuple, dst.0, start.0, count));
    }

    /// Get iterator: dst = iter(src).
    pub fn emit_get_iter(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::GetIter, dst, src));
    }

    /// For iteration: dst = next(iter), jump to label on StopIteration.
    ///
    /// The iterator source is encoded implicitly in the preceding register
    /// (`dst - 1`), preserving the full 16-bit jump offset for large loop bodies.
    pub fn emit_for_iter(&mut self, dst: Register, label: Label) {
        debug_assert!(
            dst.0 > 0,
            "ForIter requires the destination register to follow the iterator register",
        );
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::ForIter, dst));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    /// End async for: check if register contains StopAsyncIteration, jump to label if so.
    ///
    /// This is used at the end of each async for iteration to check if the awaited
    /// result indicates StopAsyncIteration. If so, the exception is cleared and
    /// execution jumps to the label (typically the else clause or loop end).
    ///
    /// # Arguments
    /// * `src` - Register containing the awaited result to check
    /// * `label` - Label to jump to if StopAsyncIteration was raised
    pub fn emit_end_async_for(&mut self, src: Register, label: Label) {
        let inst_idx = self.instructions.len();
        // Emit EndAsyncFor with src register; imm16 will be patched with jump offset
        self.emit(Instruction::op_d(Opcode::EndAsyncFor, src));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
    }

    // =========================================================================
    // Import Operations
    // =========================================================================

    /// Import a module by name index.
    ///
    /// Emits ImportName opcode: dst = import(names[name_idx])
    ///
    /// # Arguments
    /// * `dst` - Register to store the imported module object
    /// * `name_idx` - Index into the names table for the module name
    #[inline]
    pub fn emit_import_name(&mut self, dst: Register, name_idx: u16) {
        self.emit(Instruction::op_di(Opcode::ImportName, dst, name_idx));
    }

    /// Import an attribute from a module.
    ///
    /// Emits ImportFrom opcode: dst = from module import names[name_idx]
    ///
    /// # Arguments
    /// * `dst` - Register to store the imported attribute value
    /// * `module_reg` - Register containing the source module object
    /// * `name_idx` - Index into the names table for the attribute name
    ///
    /// # Instruction Encoding
    /// Uses the same compact-plus-extension name encoding as attribute
    /// opcodes: inline indices below 255 stay in the primary instruction, and
    /// larger indices use a trailing `AttrName` metadata instruction.
    #[inline]
    pub fn emit_import_from(&mut self, dst: Register, module_reg: Register, name_idx: u16) {
        let inline_name = self.emit_attr_name_operand(name_idx);
        self.emit(Instruction::new(
            Opcode::ImportFrom,
            dst.0,
            module_reg.0,
            inline_name,
        ));
        self.emit_attr_name_extension_if_needed(inline_name, name_idx);
    }

    /// Import all public names from a module.
    ///
    /// Emits ImportStar opcode: from module import *
    ///
    /// # Arguments
    /// * `dst` - Unused (set to 0), but required by instruction format
    /// * `module_reg` - Register containing the source module object
    ///
    /// Note: The VM handler will inject all public names from the module
    /// into the current scope's global namespace.
    #[inline]
    pub fn emit_import_star(&mut self, dst: Register, module_reg: Register) {
        self.emit(Instruction::op_ds(Opcode::ImportStar, dst, module_reg));
    }

    // =========================================================================
    // Exception Handling
    // =========================================================================

    /// Adds an exception entry to the exception table.
    ///
    /// This is used by the exception compiler to build the zero-cost exception
    /// table. Entries should be added in order of start_pc for efficient binary
    /// search during runtime exception handling.
    ///
    /// # Arguments
    /// * `entry` - The exception entry describing a try block and its handlers
    #[inline]
    pub fn add_exception_entry(&mut self, entry: ExceptionEntry) {
        self.exception_entries.push(entry);
    }

    /// Returns the current instruction count (program counter).
    ///
    /// This is used during exception compilation to record PC values for
    /// exception table entries.
    #[inline]
    pub fn current_pc(&self) -> u32 {
        self.instructions.len() as u32
    }

    /// Returns the current stack depth for exception handling.
    ///
    /// This is used to record the stack depth at try block entry for proper
    /// stack unwinding during exception handling.
    #[inline]
    pub fn current_stack_depth(&self) -> u8 {
        self.next_register
    }

    // =========================================================================
    // Finalization
    // =========================================================================

    /// Finish building and return the code object.
    pub fn finish(mut self) -> CodeObject {
        // Finalize line table
        let final_pc = self.instructions.len() as u32;
        if final_pc > self.line_start_pc {
            self.line_table.push(LineTableEntry {
                start_pc: self.line_start_pc,
                end_pc: final_pc,
                line: self.current_line,
            });
        }

        // Patch forward references
        for fwd in self.forward_refs {
            let target = self.labels.get(&fwd.label).expect("unbound label");
            let offset = (*target as i32) - (fwd.instruction_index as i32) - 1;

            // Replace instruction with patched version
            let old = self.instructions[fwd.instruction_index];
            let opcode = Opcode::from_u8(old.opcode()).unwrap();
            let dst = old.dst();

            // Encode offset as signed 16-bit
            let offset_u16 = offset as i16 as u16;

            let patched = Instruction::op_di(opcode, dst, offset_u16);

            self.instructions[fwd.instruction_index] = patched;
        }

        CodeObject {
            name: self.name,
            qualname: self.qualname,
            filename: self.filename,
            first_lineno: self.first_lineno,
            instructions: self.instructions.into_boxed_slice(),
            constants: self.constants.into_boxed_slice(),
            locals: self.locals.into_boxed_slice(),
            names: self.names.into_boxed_slice(),
            freevars: self.freevars.into_boxed_slice(),
            cellvars: self.cellvars.into_boxed_slice(),
            arg_count: self.arg_count,
            posonlyarg_count: self.posonlyarg_count,
            kwonlyarg_count: self.kwonlyarg_count,
            register_count: self.max_registers as u16,
            flags: if self.separate_locals {
                self.flags | CodeFlags::SEPARATE_LOCALS
            } else {
                self.flags
            },
            line_table: self.line_table.into_boxed_slice(),
            exception_table: self.exception_entries.into_boxed_slice(),
            nested_code_objects: self.nested_code_objects.into_boxed_slice(),
        }
    }
}
