//! Code object representation for compiled functions.
//!
//! A `CodeObject` contains all the compiled bytecode and metadata needed
//! to execute a Python function. This is the fundamental unit of compilation.

use super::instruction::{Instruction, Opcode};
use num_bigint::BigInt;
use prism_core::Value;
use std::fmt;
use std::sync::Arc;

const REGISTER_FILE_SIZE: u16 = 256;
const EXTENDED_NAME_SENTINEL: u8 = u8::MAX;

/// Constant pool entry stored in compiled bytecode.
#[derive(Debug, Clone)]
pub enum Constant {
    /// Fully materialized runtime value.
    Value(Value),
    /// Arbitrary-precision integer literal materialized by the VM at load time.
    BigInt(BigInt),
}

/// A compiled code object representing a function or module.
///
/// Code objects are immutable once created and can be shared across threads.
/// They contain:
/// - Bytecode instructions
/// - Constant pool
/// - Name tables
/// - Debug information
/// - Execution metadata
/// - Exception handling table (for zero-cost exceptions)
#[derive(Debug, Clone)]
pub struct CodeObject {
    /// Function name (or `<module>` for module-level code).
    pub name: Arc<str>,

    /// Qualified name (includes enclosing class/function names).
    pub qualname: Arc<str>,

    /// Filename where this code was defined.
    pub filename: Arc<str>,

    /// First line number in source.
    pub first_lineno: u32,

    /// Bytecode instructions (32-bit each).
    pub instructions: Box<[Instruction]>,

    /// Constant pool (indexed by LoadConst).
    pub constants: Box<[Constant]>,

    /// Local variable names (for debugging and closures).
    pub locals: Box<[Arc<str>]>,

    /// Global/attribute name strings (indexed by LoadGlobal, GetAttr, etc).
    pub names: Box<[Arc<str>]>,

    /// Free variable names (captured from enclosing scope).
    pub freevars: Box<[Arc<str>]>,

    /// Cell variable names (captured by nested functions).
    pub cellvars: Box<[Arc<str>]>,

    /// Number of positional parameters.
    pub arg_count: u16,

    /// Number of positional-only parameters.
    pub posonlyarg_count: u16,

    /// Number of keyword-only parameters.
    pub kwonlyarg_count: u16,

    /// Number of virtual registers used.
    pub register_count: u16,

    /// Code flags.
    pub flags: CodeFlags,

    /// Line number table (instruction index -> line number).
    /// Stored as delta-encoded pairs for compactness.
    pub line_table: Box<[LineTableEntry]>,

    /// Exception handling table for zero-cost exceptions.
    /// Sorted by start_pc for binary search during unwinding.
    pub exception_table: Box<[ExceptionEntry]>,

    /// Nested code objects (functions, classes, comprehensions defined in this code).
    /// Stored separately from constants for test accessibility and debugging.
    pub nested_code_objects: Box<[Arc<CodeObject>]>,
}

/// Code object flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CodeFlags(u32);

impl CodeFlags {
    /// No flags.
    pub const NONE: CodeFlags = CodeFlags(0);
    /// Function uses *args.
    pub const VARARGS: CodeFlags = CodeFlags(1 << 0);
    /// Function uses **kwargs.
    pub const VARKEYWORDS: CodeFlags = CodeFlags(1 << 1);
    /// Function is a generator.
    pub const GENERATOR: CodeFlags = CodeFlags(1 << 2);
    /// Function is a coroutine.
    pub const COROUTINE: CodeFlags = CodeFlags(1 << 3);
    /// Function is an async generator.
    pub const ASYNC_GENERATOR: CodeFlags = CodeFlags(1 << 4);
    /// Function is nested.
    pub const NESTED: CodeFlags = CodeFlags(1 << 5);
    /// Function has free variables.
    pub const HAS_FREEVARS: CodeFlags = CodeFlags(1 << 6);
    /// Function has cell variables.
    pub const HAS_CELLVARS: CodeFlags = CodeFlags(1 << 7);
    /// This is module-level code.
    pub const MODULE: CodeFlags = CodeFlags(1 << 8);
    /// This is class body code.
    pub const CLASS: CodeFlags = CodeFlags(1 << 9);

    /// Bitmask of all currently defined flags.
    pub const ALL_BITS: u32 = Self::VARARGS.bits()
        | Self::VARKEYWORDS.bits()
        | Self::GENERATOR.bits()
        | Self::COROUTINE.bits()
        | Self::ASYNC_GENERATOR.bits()
        | Self::NESTED.bits()
        | Self::HAS_FREEVARS.bits()
        | Self::HAS_CELLVARS.bits()
        | Self::MODULE.bits()
        | Self::CLASS.bits();

    /// Check if a flag is set.
    #[inline]
    pub const fn contains(self, other: CodeFlags) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Combine flags.
    #[inline]
    pub const fn union(self, other: CodeFlags) -> CodeFlags {
        CodeFlags(self.0 | other.0)
    }

    /// Get raw value.
    #[inline]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Construct flags from raw bits when every bit is known.
    #[inline]
    pub const fn from_bits(bits: u32) -> Option<Self> {
        if bits & !Self::ALL_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }
}

impl std::ops::BitOr for CodeFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for CodeFlags {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Line table entry mapping instruction ranges to source lines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineTableEntry {
    /// Starting instruction index (inclusive).
    pub start_pc: u32,
    /// Ending instruction index (exclusive).
    pub end_pc: u32,
    /// Source line number.
    pub line: u32,
}

/// Exception handler entry for zero-cost exception handling.
///
/// During exception unwinding, the VM performs a binary search on these
/// entries (sorted by start_pc) to find a matching handler. This enables
/// true zero-cost exceptions: no overhead on the happy path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExceptionEntry {
    /// Starting PC of the try block (inclusive).
    pub start_pc: u32,
    /// Ending PC of the try block (exclusive).
    pub end_pc: u32,
    /// Handler PC (start of except clause).
    pub handler_pc: u32,
    /// Finally block PC (if present).
    /// A value of u32::MAX indicates no finally block.
    pub finally_pc: u32,
    /// Handler depth for nested handlers.
    pub depth: u16,
    /// Exception type constant index (for type matching).
    /// A value of u16::MAX indicates bare `except:`.
    pub exception_type_idx: u16,
}

/// Error produced by bytecode validation.
///
/// The VM deliberately uses unchecked indexing in hot opcode handlers. A code
/// object must therefore pass this verifier before execution so malformed or
/// stale bytecode cannot reach those fast paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeValidationError {
    /// Instruction index where validation failed.
    pub pc: usize,
    /// Specific validation failure.
    pub kind: CodeValidationErrorKind,
}

/// Specific bytecode validation failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodeValidationErrorKind {
    /// Opcode byte is not assigned to a known instruction.
    InvalidOpcode {
        /// Raw opcode byte.
        opcode: u8,
    },
    /// A constant/name/local/closure index points outside its table.
    PoolIndexOutOfBounds {
        /// Pool/table name.
        pool: &'static str,
        /// Referenced index.
        index: u16,
        /// Pool/table length.
        len: usize,
    },
    /// A local register slot would be truncated by the current register VM.
    LocalSlotTooWide {
        /// Referenced local slot.
        slot: u16,
    },
    /// A jump target leaves the instruction stream.
    JumpTargetOutOfBounds {
        /// Computed absolute target pc.
        target: isize,
        /// Instruction stream length.
        len: usize,
    },
    /// A metadata extension is missing after an opcode that requires it.
    MissingExtension {
        /// Opcode that requires metadata.
        opcode: Opcode,
        /// Required extension opcode.
        extension: Opcode,
    },
    /// A metadata extension would execute as a standalone opcode.
    UnexpectedExtension {
        /// Detached extension opcode.
        extension: Opcode,
    },
    /// A line-table range is malformed or outside the instruction stream.
    InvalidLineRange {
        /// Inclusive start pc.
        start_pc: u32,
        /// Exclusive end pc.
        end_pc: u32,
    },
    /// An exception-table range or handler target is malformed.
    InvalidExceptionRange {
        /// Inclusive protected-range start pc.
        start_pc: u32,
        /// Exclusive protected-range end pc.
        end_pc: u32,
        /// Exception handler pc.
        handler_pc: u32,
        /// Finally handler pc, or `u32::MAX` when absent.
        finally_pc: u32,
    },
    /// The code object requires more registers than the frame layout supports.
    RegisterCountTooLarge {
        /// Requested register count.
        count: u16,
        /// Maximum register count supported by the frame layout.
        max: u16,
    },
}

impl CodeValidationError {
    #[inline]
    fn new(pc: usize, kind: CodeValidationErrorKind) -> Self {
        Self { pc, kind }
    }
}

impl fmt::Display for CodeValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use CodeValidationErrorKind::*;

        write!(f, "invalid bytecode at pc {}: ", self.pc)?;
        match &self.kind {
            InvalidOpcode { opcode } => write!(f, "unknown opcode 0x{opcode:02x}"),
            PoolIndexOutOfBounds { pool, index, len } => {
                write!(f, "{pool} index {index} out of bounds for length {len}")
            }
            LocalSlotTooWide { slot } => {
                write!(f, "local slot {slot} exceeds the 256-register frame limit")
            }
            JumpTargetOutOfBounds { target, len } => {
                write!(f, "jump target {target} out of bounds for length {len}")
            }
            MissingExtension { opcode, extension } => {
                write!(f, "{opcode:?} missing required {extension:?} extension")
            }
            UnexpectedExtension { extension } => {
                write!(f, "{extension:?} extension is not attached to a consumer")
            }
            InvalidLineRange { start_pc, end_pc } => {
                write!(f, "invalid line-table range {start_pc}..{end_pc}")
            }
            InvalidExceptionRange {
                start_pc,
                end_pc,
                handler_pc,
                finally_pc,
            } => write!(
                f,
                "invalid exception-table range {start_pc}..{end_pc}, handler={handler_pc}, finally={finally_pc}"
            ),
            RegisterCountTooLarge { count, max } => {
                write!(f, "register count {count} exceeds frame limit {max}")
            }
        }
    }
}

impl std::error::Error for CodeValidationError {}

impl CodeObject {
    /// Create a new empty code object.
    pub fn new(name: impl Into<Arc<str>>, filename: impl Into<Arc<str>>) -> Self {
        let name = name.into();
        CodeObject {
            qualname: name.clone(),
            name,
            filename: filename.into(),
            first_lineno: 1,
            instructions: Box::new([]),
            constants: Box::new([]),
            locals: Box::new([]),
            names: Box::new([]),
            freevars: Box::new([]),
            cellvars: Box::new([]),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 0,
            flags: CodeFlags::NONE,
            line_table: Box::new([]),
            exception_table: Box::new([]),
            nested_code_objects: Box::new([]),
        }
    }

    /// Get the line number for a given instruction index.
    pub fn line_for_pc(&self, pc: u32) -> Option<u32> {
        for entry in self.line_table.iter() {
            if entry.start_pc <= pc && pc < entry.end_pc {
                return Some(entry.line);
            }
        }
        None
    }

    /// Get the CPython-compatible source position tuple for an instruction.
    ///
    /// Prism currently tracks line granularity but not column offsets, so the
    /// start/end columns remain `None`. When a line number is available we
    /// mirror it into both the start and end line slots, matching CPython's
    /// `co_positions()` shape.
    #[inline]
    pub fn position_for_pc(&self, pc: u32) -> (Option<u32>, Option<u32>, Option<u32>, Option<u32>) {
        match self.line_for_pc(pc) {
            Some(line) => (Some(line), Some(line), None, None),
            None => (None, None, None, None),
        }
    }

    /// Iterate over CPython-compatible source positions for each instruction.
    #[inline]
    pub fn positions(
        &self,
    ) -> impl ExactSizeIterator<Item = (Option<u32>, Option<u32>, Option<u32>, Option<u32>)> + '_
    {
        (0..self.instructions.len() as u32).map(|pc| self.position_for_pc(pc))
    }

    /// Check if this is a generator function.
    #[inline]
    pub fn is_generator(&self) -> bool {
        self.flags.contains(CodeFlags::GENERATOR)
    }

    /// Check if this is a coroutine.
    #[inline]
    pub fn is_coroutine(&self) -> bool {
        self.flags.contains(CodeFlags::COROUTINE)
    }

    /// Check if this is an async generator.
    #[inline]
    pub fn is_async_generator(&self) -> bool {
        self.flags.contains(CodeFlags::ASYNC_GENERATOR)
    }

    /// Get the total parameter count (positional + keyword-only).
    #[inline]
    pub fn total_params(&self) -> u16 {
        self.arg_count + self.kwonlyarg_count
    }

    /// Get number of closure variables (free + cell).
    #[inline]
    pub fn closure_size(&self) -> usize {
        self.freevars.len() + self.cellvars.len()
    }

    /// Validate bytecode operands and metadata before execution.
    ///
    /// This is intentionally conservative around table indices and extension
    /// opcodes. It keeps the VM's hot opcode handlers free to use unchecked
    /// array access after the code object has crossed the execution boundary.
    pub fn validate(&self) -> Result<(), CodeValidationError> {
        if self.register_count > REGISTER_FILE_SIZE {
            return Err(CodeValidationError::new(
                0,
                CodeValidationErrorKind::RegisterCountTooLarge {
                    count: self.register_count,
                    max: REGISTER_FILE_SIZE,
                },
            ));
        }

        self.validate_line_table()?;
        self.validate_exception_table()?;

        let mut pc = 0usize;
        while pc < self.instructions.len() {
            let inst = self.instructions[pc];
            let opcode = Opcode::from_u8(inst.opcode()).ok_or_else(|| {
                CodeValidationError::new(
                    pc,
                    CodeValidationErrorKind::InvalidOpcode {
                        opcode: inst.opcode(),
                    },
                )
            })?;

            match opcode {
                Opcode::LoadConst
                | Opcode::MakeFunction
                | Opcode::MakeClosure
                | Opcode::BuildClass
                | Opcode::BuildClassWithMeta => {
                    self.validate_pool_index(pc, "constant", inst.imm16(), self.constants.len())?;
                    pc = self.skip_class_metadata(pc, opcode)?;
                }
                Opcode::LoadGlobal
                | Opcode::StoreGlobal
                | Opcode::DeleteGlobal
                | Opcode::LoadBuiltin
                | Opcode::ImportName => {
                    self.validate_pool_index(pc, "name", inst.imm16(), self.names.len())?;
                    pc += 1;
                }
                Opcode::GetAttr
                | Opcode::SetAttr
                | Opcode::DelAttr
                | Opcode::LoadMethod
                | Opcode::ImportFrom => {
                    pc = self.validate_extended_name_operand(pc, opcode, inst)?;
                }
                Opcode::LoadLocal | Opcode::StoreLocal | Opcode::DeleteLocal => {
                    self.validate_local_slot(pc, inst.imm16())?;
                    pc += 1;
                }
                Opcode::LoadClosure | Opcode::StoreClosure | Opcode::DeleteClosure => {
                    self.validate_pool_index(pc, "closure", inst.imm16(), self.closure_size())?;
                    pc += 1;
                }
                Opcode::Jump
                | Opcode::JumpIfFalse
                | Opcode::JumpIfTrue
                | Opcode::JumpIfNone
                | Opcode::JumpIfNotNone
                | Opcode::ForIter
                | Opcode::EndAsyncFor => {
                    self.validate_jump_target(pc, inst.imm16() as i16)?;
                    pc += 1;
                }
                Opcode::AttrName => {
                    return Err(CodeValidationError::new(
                        pc,
                        CodeValidationErrorKind::UnexpectedExtension {
                            extension: Opcode::AttrName,
                        },
                    ));
                }
                _ => pc += 1,
            }
        }

        Ok(())
    }

    fn validate_line_table(&self) -> Result<(), CodeValidationError> {
        let len = self.instructions.len() as u32;
        for entry in self.line_table.iter() {
            if entry.start_pc > entry.end_pc || entry.end_pc > len {
                return Err(CodeValidationError::new(
                    entry.start_pc as usize,
                    CodeValidationErrorKind::InvalidLineRange {
                        start_pc: entry.start_pc,
                        end_pc: entry.end_pc,
                    },
                ));
            }
        }
        Ok(())
    }

    fn validate_exception_table(&self) -> Result<(), CodeValidationError> {
        let len = self.instructions.len() as u32;
        for entry in self.exception_table.iter() {
            let finally_ok = entry.finally_pc == u32::MAX || entry.finally_pc < len;
            if entry.start_pc > entry.end_pc
                || entry.end_pc > len
                || entry.handler_pc >= len
                || !finally_ok
            {
                return Err(CodeValidationError::new(
                    entry.start_pc as usize,
                    CodeValidationErrorKind::InvalidExceptionRange {
                        start_pc: entry.start_pc,
                        end_pc: entry.end_pc,
                        handler_pc: entry.handler_pc,
                        finally_pc: entry.finally_pc,
                    },
                ));
            }

            if entry.exception_type_idx != u16::MAX {
                self.validate_pool_index(
                    entry.start_pc as usize,
                    "constant",
                    entry.exception_type_idx,
                    self.constants.len(),
                )?;
            }
        }
        Ok(())
    }

    fn validate_pool_index(
        &self,
        pc: usize,
        pool: &'static str,
        index: u16,
        len: usize,
    ) -> Result<(), CodeValidationError> {
        if usize::from(index) < len {
            Ok(())
        } else {
            Err(CodeValidationError::new(
                pc,
                CodeValidationErrorKind::PoolIndexOutOfBounds { pool, index, len },
            ))
        }
    }

    fn validate_local_slot(&self, pc: usize, slot: u16) -> Result<(), CodeValidationError> {
        if slot >= REGISTER_FILE_SIZE {
            return Err(CodeValidationError::new(
                pc,
                CodeValidationErrorKind::LocalSlotTooWide { slot },
            ));
        }

        Ok(())
    }

    fn validate_jump_target(&self, pc: usize, offset: i16) -> Result<(), CodeValidationError> {
        let target = pc as isize + 1 + offset as isize;
        let len = self.instructions.len();
        if (0..=len as isize).contains(&target) {
            Ok(())
        } else {
            Err(CodeValidationError::new(
                pc,
                CodeValidationErrorKind::JumpTargetOutOfBounds { target, len },
            ))
        }
    }

    fn validate_extended_name_operand(
        &self,
        pc: usize,
        opcode: Opcode,
        inst: Instruction,
    ) -> Result<usize, CodeValidationError> {
        let inline = match opcode {
            Opcode::GetAttr | Opcode::DelAttr | Opcode::LoadMethod | Opcode::ImportFrom => {
                inst.src2().0
            }
            Opcode::SetAttr => inst.src1().0,
            _ => unreachable!("non-name opcode passed to validate_extended_name_operand"),
        };

        if inline != EXTENDED_NAME_SENTINEL {
            self.validate_pool_index(pc, "name", inline as u16, self.names.len())?;
            return Ok(pc + 1);
        }

        let extension_pc = pc + 1;
        let Some(extension) = self.instructions.get(extension_pc).copied() else {
            return Err(CodeValidationError::new(
                pc,
                CodeValidationErrorKind::MissingExtension {
                    opcode,
                    extension: Opcode::AttrName,
                },
            ));
        };

        if Opcode::from_u8(extension.opcode()) != Some(Opcode::AttrName) {
            return Err(CodeValidationError::new(
                pc,
                CodeValidationErrorKind::MissingExtension {
                    opcode,
                    extension: Opcode::AttrName,
                },
            ));
        }

        self.validate_pool_index(extension_pc, "name", extension.imm16(), self.names.len())?;
        Ok(pc + 2)
    }

    fn skip_class_metadata(&self, pc: usize, opcode: Opcode) -> Result<usize, CodeValidationError> {
        if !matches!(opcode, Opcode::BuildClass | Opcode::BuildClassWithMeta) {
            return Ok(pc + 1);
        }

        let mut next_pc = pc + 1;
        if self
            .instructions
            .get(next_pc)
            .is_some_and(|inst| Opcode::from_u8(inst.opcode()) == Some(Opcode::ClassMeta))
        {
            next_pc += 1;
        }
        if self
            .instructions
            .get(next_pc)
            .is_some_and(|inst| Opcode::from_u8(inst.opcode()) == Some(Opcode::CallKwEx))
        {
            next_pc += 1;
        }
        Ok(next_pc)
    }
}

/// Disassemble a code object to a string.
pub fn disassemble(code: &CodeObject) -> String {
    use std::fmt::Write;

    let mut output = String::new();

    writeln!(output, "Code object: {}", code.name).unwrap();
    writeln!(output, "  File: {}", code.filename).unwrap();
    writeln!(output, "  First line: {}", code.first_lineno).unwrap();
    writeln!(
        output,
        "  Args: {} (pos-only: {}, kw-only: {})",
        code.arg_count, code.posonlyarg_count, code.kwonlyarg_count
    )
    .unwrap();
    writeln!(output, "  Registers: {}", code.register_count).unwrap();
    writeln!(output, "  Flags: {:08x}", code.flags.bits()).unwrap();

    if !code.constants.is_empty() {
        writeln!(output, "\nConstants:").unwrap();
        for (i, c) in code.constants.iter().enumerate() {
            writeln!(output, "  {:4}: {:?}", i, c).unwrap();
        }
    }

    if !code.names.is_empty() {
        writeln!(output, "\nNames:").unwrap();
        for (i, n) in code.names.iter().enumerate() {
            writeln!(output, "  {:4}: {}", i, n).unwrap();
        }
    }

    if !code.locals.is_empty() {
        writeln!(output, "\nLocals:").unwrap();
        for (i, l) in code.locals.iter().enumerate() {
            writeln!(output, "  {:4}: {}", i, l).unwrap();
        }
    }

    writeln!(output, "\nDisassembly:").unwrap();
    for (i, inst) in code.instructions.iter().enumerate() {
        let line = code.line_for_pc(i as u32);
        let line_str = line.map_or("    ".to_string(), |l| format!("{:4}", l));
        writeln!(output, "{} {:4}: {}", line_str, i, inst).unwrap();
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instruction::{Opcode, Register};

    #[test]
    fn test_code_flags() {
        let flags = CodeFlags::GENERATOR | CodeFlags::NESTED;
        assert!(flags.contains(CodeFlags::GENERATOR));
        assert!(flags.contains(CodeFlags::NESTED));
        assert!(!flags.contains(CodeFlags::COROUTINE));
    }

    #[test]
    fn test_code_flags_from_bits_accepts_known_flags() {
        let bits = (CodeFlags::GENERATOR | CodeFlags::MODULE).bits();
        let flags = CodeFlags::from_bits(bits).expect("known flag set should decode");
        assert!(flags.contains(CodeFlags::GENERATOR));
        assert!(flags.contains(CodeFlags::MODULE));
    }

    #[test]
    fn test_code_flags_from_bits_rejects_unknown_flags() {
        assert!(CodeFlags::from_bits(CodeFlags::ALL_BITS | (1 << 31)).is_none());
    }

    #[test]
    fn test_code_object_new() {
        let code = CodeObject::new("test_func", "test.py");
        assert_eq!(&*code.name, "test_func");
        assert_eq!(&*code.filename, "test.py");
        assert_eq!(code.instructions.len(), 0);
    }

    #[test]
    fn test_validate_accepts_well_formed_extended_name_operand() {
        let mut code = CodeObject::new("attr", "test.py");
        code.names = (0..=0x0123)
            .map(|index| Arc::<str>::from(format!("name_{index}")))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        code.instructions = vec![
            Instruction::new(Opcode::GetAttr, 0, 1, u8::MAX),
            Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123),
        ]
        .into_boxed_slice();

        assert!(code.validate().is_ok());
    }

    #[test]
    fn test_validate_rejects_invalid_opcode() {
        let mut code = CodeObject::new("bad", "test.py");
        code.instructions = vec![Instruction::from_raw(0xFF00_0000)].into_boxed_slice();

        assert!(matches!(
            code.validate(),
            Err(CodeValidationError {
                kind: CodeValidationErrorKind::InvalidOpcode { opcode: 0xFF },
                ..
            })
        ));
    }

    #[test]
    fn test_validate_rejects_constant_index_out_of_bounds() {
        let mut code = CodeObject::new("bad_const", "test.py");
        code.instructions =
            vec![Instruction::op_di(Opcode::LoadConst, Register::new(0), 7)].into_boxed_slice();

        assert!(matches!(
            code.validate(),
            Err(CodeValidationError {
                kind: CodeValidationErrorKind::PoolIndexOutOfBounds {
                    pool: "constant",
                    index: 7,
                    len: 0,
                },
                ..
            })
        ));
    }

    #[test]
    fn test_validate_rejects_missing_attr_name_extension() {
        let mut code = CodeObject::new("bad_attr", "test.py");
        code.instructions =
            vec![Instruction::new(Opcode::GetAttr, 0, 1, u8::MAX)].into_boxed_slice();

        assert!(matches!(
            code.validate(),
            Err(CodeValidationError {
                kind: CodeValidationErrorKind::MissingExtension {
                    opcode: Opcode::GetAttr,
                    extension: Opcode::AttrName,
                },
                ..
            })
        ));
    }

    #[test]
    fn test_validate_rejects_jump_out_of_bounds() {
        let mut code = CodeObject::new("bad_jump", "test.py");
        code.instructions =
            vec![Instruction::op_di(Opcode::Jump, Register::new(0), 10u16)].into_boxed_slice();

        assert!(matches!(
            code.validate(),
            Err(CodeValidationError {
                kind: CodeValidationErrorKind::JumpTargetOutOfBounds { .. },
                ..
            })
        ));
    }

    #[test]
    fn test_line_table_lookup() {
        let mut code = CodeObject::new("test", "test.py");
        code.line_table = vec![
            LineTableEntry {
                start_pc: 0,
                end_pc: 5,
                line: 10,
            },
            LineTableEntry {
                start_pc: 5,
                end_pc: 10,
                line: 15,
            },
        ]
        .into_boxed_slice();

        assert_eq!(code.line_for_pc(0), Some(10));
        assert_eq!(code.line_for_pc(4), Some(10));
        assert_eq!(code.line_for_pc(5), Some(15));
        assert_eq!(code.line_for_pc(9), Some(15));
        assert_eq!(code.line_for_pc(10), None);
    }

    #[test]
    fn test_code_positions_follow_instruction_line_ranges() {
        let mut code = CodeObject::new("test", "test.py");
        code.instructions = vec![
            Instruction::op(Opcode::Nop),
            Instruction::op(Opcode::Nop),
            Instruction::op(Opcode::Nop),
            Instruction::op(Opcode::Nop),
        ]
        .into_boxed_slice();
        code.line_table = vec![
            LineTableEntry {
                start_pc: 0,
                end_pc: 1,
                line: 10,
            },
            LineTableEntry {
                start_pc: 1,
                end_pc: 4,
                line: 14,
            },
        ]
        .into_boxed_slice();

        let positions: Vec<_> = code.positions().collect();
        assert_eq!(
            positions,
            vec![
                (Some(10), Some(10), None, None),
                (Some(14), Some(14), None, None),
                (Some(14), Some(14), None, None),
                (Some(14), Some(14), None, None),
            ]
        );
    }

    #[test]
    fn test_code_position_defaults_to_unknown_when_line_is_missing() {
        let mut code = CodeObject::new("test", "test.py");
        code.instructions = vec![Instruction::op(Opcode::Nop)].into_boxed_slice();

        assert_eq!(code.position_for_pc(0), (None, None, None, None));
    }
}
