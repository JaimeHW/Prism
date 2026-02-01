//! Code object representation for compiled functions.
//!
//! A `CodeObject` contains all the compiled bytecode and metadata needed
//! to execute a Python function. This is the fundamental unit of compilation.

use super::instruction::Instruction;
use prism_core::Value;
use std::sync::Arc;

/// A compiled code object representing a function or module.
///
/// Code objects are immutable once created and can be shared across threads.
/// They contain:
/// - Bytecode instructions
/// - Constant pool
/// - Name tables
/// - Debug information
/// - Execution metadata
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
    pub constants: Box<[Value]>,

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

    #[test]
    fn test_code_flags() {
        let flags = CodeFlags::GENERATOR | CodeFlags::NESTED;
        assert!(flags.contains(CodeFlags::GENERATOR));
        assert!(flags.contains(CodeFlags::NESTED));
        assert!(!flags.contains(CodeFlags::COROUTINE));
    }

    #[test]
    fn test_code_object_new() {
        let code = CodeObject::new("test_func", "test.py");
        assert_eq!(&*code.name, "test_func");
        assert_eq!(&*code.filename, "test.py");
        assert_eq!(code.instructions.len(), 0);
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
}
