//! Function builder for bytecode emission.
//!
//! The `FunctionBuilder` provides a high-level API for constructing bytecode
//! with automatic register allocation and label resolution.

use super::code_object::{CodeFlags, CodeObject, LineTableEntry};
use super::instruction::{ConstIndex, Instruction, LocalSlot, Opcode, Register};
use prism_core::Value;
use std::collections::HashMap;
use std::sync::Arc;

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
    constants: Vec<Value>,
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
}

/// Key type for constant deduplication.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ConstantKey {
    None,
    Bool(bool),
    Int(i64),
    /// Float bits for exact comparison.
    Float(u64),
    String(Arc<str>),
    /// Code object by name (simplified).
    Code(Arc<str>),
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
            next_register: 0,
            max_registers: 0,
            free_registers: Vec::new(),
            next_label: 0,
            labels: HashMap::new(),
            forward_refs: Vec::new(),
            line_table: Vec::new(),
            line_start_pc: 0,
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

    /// Set the first line number.
    pub fn set_first_lineno(&mut self, line: u32) {
        self.first_lineno = line;
        self.current_line = line;
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
        // Only track if it was the most recently allocated
        // This is a simple strategy; could be improved with more sophisticated tracking
        self.free_registers.push(reg);
    }

    /// Reserve registers for function parameters.
    pub fn reserve_parameters(&mut self, count: u16) {
        for _ in 0..count {
            self.alloc_register();
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
            self.constants.push(value);
            self.constant_map.insert(key, idx);
            idx
        } else {
            // No deduplication for complex types
            let idx = ConstIndex::new(self.constants.len() as u16);
            self.constants.push(value);
            idx
        }
    }

    /// Add an integer constant.
    pub fn add_int(&mut self, value: i64) -> ConstIndex {
        self.add_constant(Value::int(value).unwrap_or_else(|| Value::none()))
    }

    /// Add a float constant.
    pub fn add_float(&mut self, value: f64) -> ConstIndex {
        self.add_constant(Value::float(value))
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

    /// Store a register into a global variable.
    pub fn emit_store_global(&mut self, name_idx: u16, src: Register) {
        self.emit(Instruction::op_di(Opcode::StoreGlobal, src, name_idx));
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

    /// Generic negate: dst = -src.
    pub fn emit_neg(&mut self, dst: Register, src: Register) {
        self.emit(Instruction::op_ds(Opcode::Neg, dst, src));
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
        self.emit(Instruction::new(
            Opcode::GetAttr,
            dst.0,
            obj.0,
            (name_idx >> 8) as u8,
        ));
        // Note: This uses a compressed format. For full 16-bit name indices,
        // we'd need a different encoding.
    }

    /// Get item: dst = obj[key].
    pub fn emit_get_item(&mut self, dst: Register, obj: Register, key: Register) {
        self.emit(Instruction::op_dss(Opcode::GetItem, dst, obj, key));
    }

    /// Set item: obj[key] = value.
    pub fn emit_set_item(&mut self, obj: Register, key: Register, value: Register) {
        self.emit(Instruction::op_dss(Opcode::SetItem, obj, key, value));
    }

    // --- Function Calls ---

    /// Call function: dst = func(args...).
    /// Args should be in registers dst+1, dst+2, etc.
    pub fn emit_call(&mut self, dst: Register, func: Register, argc: u8) {
        self.emit(Instruction::new(Opcode::Call, dst.0, func.0, argc));
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
    pub fn emit_for_iter(&mut self, dst: Register, label: Label) {
        let inst_idx = self.instructions.len();
        self.emit(Instruction::op_d(Opcode::ForIter, dst));
        self.forward_refs.push(ForwardRef {
            instruction_index: inst_idx,
            label,
        });
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
            let opcode = old.opcode();
            let dst = old.dst();

            // Encode offset as signed 16-bit
            let offset_u16 = offset as i16 as u16;
            self.instructions[fwd.instruction_index] =
                Instruction::op_di(Opcode::from_u8(opcode).unwrap(), dst, offset_u16);
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
            flags: self.flags,
            line_table: self.line_table.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_function() {
        let mut builder = FunctionBuilder::new("add");
        builder.set_filename("test.py");
        builder.set_arg_count(2);

        let x = builder.alloc_register();
        let y = builder.alloc_register();
        let result = builder.alloc_register();

        builder.emit_load_local(x, LocalSlot::new(0));
        builder.emit_load_local(y, LocalSlot::new(1));
        builder.emit_add(result, x, y);
        builder.emit_return(result);

        let code = builder.finish();

        assert_eq!(&*code.name, "add");
        assert_eq!(code.instructions.len(), 4);
        assert_eq!(code.register_count, 3);
    }

    #[test]
    fn test_constant_deduplication() {
        let mut builder = FunctionBuilder::new("test");

        let idx1 = builder.add_int(42);
        let idx2 = builder.add_int(42);
        let idx3 = builder.add_int(100);

        assert_eq!(idx1.0, idx2.0); // Same constant, same index
        assert_ne!(idx1.0, idx3.0); // Different constant, different index
    }

    #[test]
    fn test_labels() {
        let mut builder = FunctionBuilder::new("loop");

        let loop_start = builder.create_label();
        let loop_end = builder.create_label();

        let r0 = builder.alloc_register();

        builder.bind_label(loop_start);
        builder.emit_jump_if_false(r0, loop_end);
        builder.emit_nop();
        builder.emit_jump(loop_start);
        builder.bind_label(loop_end);
        builder.emit_return_none();

        let code = builder.finish();
        assert_eq!(code.instructions.len(), 4);
    }

    #[test]
    fn test_register_allocation() {
        let mut builder = FunctionBuilder::new("test");

        let r0 = builder.alloc_register();
        let r1 = builder.alloc_register();
        builder.free_register(r0);
        let r2 = builder.alloc_register(); // Should reuse r0

        assert_eq!(r0.0, 0);
        assert_eq!(r1.0, 1);
        assert_eq!(r2.0, 0); // Reused
    }
}
