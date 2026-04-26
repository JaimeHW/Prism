//! Exception type identifiers.
//!
//! This module defines the `ExceptionTypeId` enum which provides O(1) type
//! identification for Python exception types. The u8 representation enables
//! single-instruction comparisons in JIT-compiled code.
//!
//! # Performance Design
//!
//! - **u8 packed**: Single byte comparison = 1 CPU instruction
//! - **Contiguous IDs**: Enables efficient lookup tables
//! - **IC-friendly**: `except TypeError` compiles to `cmp al, ID; je handler`
//!
//! # Hierarchy
//!
//! The type IDs are organized to match Python's exception hierarchy:
//! - 0-15: Core exceptions (BaseException, Exception, etc.)
//! - 16-47: Common runtime exceptions
//! - 48-63: System/OS exceptions
//! - 64-127: Reserved for stdlib
//! - 128-255: User-defined (dynamic allocation)

use std::fmt;

/// Exception type identifier.
///
/// Packed into a single byte for optimal JIT code generation.
/// The ordering matches Python's exception hierarchy for efficient
/// subclass checking via range comparisons.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExceptionTypeId {
    // ═══════════════════════════════════════════════════════════════════════
    // Core Exceptions (0-15) - Exception hierarchy roots
    // ═══════════════════════════════════════════════════════════════════════
    /// Root of all exceptions.
    BaseException = 0,

    /// Base class for keyboard interrupt and system exit.
    SystemExit = 1,

    /// Raised when the user interrupts execution.
    KeyboardInterrupt = 2,

    /// Request that a generator exit.
    GeneratorExit = 3,

    /// Base class for all built-in exceptions (except system-exiting ones).
    Exception = 4,

    /// Iterator has no more items.
    StopIteration = 5,

    /// Async iterator has no more items.
    StopAsyncIteration = 6,

    /// Base class for arithmetic errors.
    ArithmeticError = 7,

    /// Numeric overflow.
    OverflowError = 8,

    /// Division by zero.
    ZeroDivisionError = 9,

    /// Floating point operation failed.
    FloatingPointError = 10,

    /// Assertion failed.
    AssertionError = 11,

    /// Attribute not found.
    AttributeError = 12,

    /// Buffer-related operation error.
    BufferError = 13,

    /// End of file reached.
    EOFError = 14,

    /// Import failed.
    ImportError = 15,

    // ═══════════════════════════════════════════════════════════════════════
    // Lookup Errors (16-19)
    // ═══════════════════════════════════════════════════════════════════════
    /// Base class for lookup errors.
    LookupError = 16,

    /// Sequence index out of range.
    IndexError = 17,

    /// Mapping key not found.
    KeyError = 18,

    /// Module not found.
    ModuleNotFoundError = 19,

    // ═══════════════════════════════════════════════════════════════════════
    // Memory/Name Errors (20-23)
    // ═══════════════════════════════════════════════════════════════════════
    /// Out of memory.
    MemoryError = 20,

    /// Name not found in local or global scope.
    NameError = 21,

    /// Unbound local variable.
    UnboundLocalError = 22,

    /// Reserved for future use.
    _Reserved23 = 23,

    // ═══════════════════════════════════════════════════════════════════════
    // OS/IO Errors (24-35)
    // ═══════════════════════════════════════════════════════════════════════
    /// Base class for OS-related errors.
    OSError = 24,

    /// File or directory not found.
    FileNotFoundError = 25,

    /// File already exists.
    FileExistsError = 26,

    /// Not a directory.
    NotADirectoryError = 27,

    /// Is a directory (when file expected).
    IsADirectoryError = 28,

    /// Permission denied.
    PermissionError = 29,

    /// Process lookup failed.
    ProcessLookupError = 30,

    /// Connection error base.
    ConnectionError = 31,

    /// Connection refused.
    ConnectionRefusedError = 32,

    /// Connection reset.
    ConnectionResetError = 33,

    /// Connection aborted.
    ConnectionAbortedError = 34,

    /// Broken pipe.
    BrokenPipeError = 35,

    // ═══════════════════════════════════════════════════════════════════════
    // More OS Errors (36-39)
    // ═══════════════════════════════════════════════════════════════════════
    /// Timeout expired.
    TimeoutError = 36,

    /// Interrupted system call.
    InterruptedError = 37,

    /// Child process error.
    ChildProcessError = 38,

    /// Blocking I/O error.
    BlockingIOError = 39,

    // ═══════════════════════════════════════════════════════════════════════
    // Reference/Runtime Errors (40-47)
    // ═══════════════════════════════════════════════════════════════════════
    /// Weak reference target was collected.
    ReferenceError = 40,

    /// Unhandled error in interpreter.
    RuntimeError = 41,

    /// Maximum recursion depth exceeded.
    RecursionError = 42,

    /// Method not implemented.
    NotImplementedError = 43,

    /// Base class for grouped exceptions.
    BaseExceptionGroup = 44,

    /// Grouped exceptions that are also regular exceptions.
    ExceptionGroup = 45,

    /// Reserved for future use.
    _Reserved46 = 46,
    _Reserved47 = 47,

    // ═══════════════════════════════════════════════════════════════════════
    // Syntax/Value Errors (48-55)
    // ═══════════════════════════════════════════════════════════════════════
    /// Syntax error in source code.
    SyntaxError = 48,

    /// Indentation error.
    IndentationError = 49,

    /// Inconsistent use of tabs and spaces.
    TabError = 50,

    /// System error (interpreter bug).
    SystemError = 51,

    /// Inappropriate argument type.
    TypeError = 52,

    /// Inappropriate argument value.
    ValueError = 53,

    /// Unicode encoding error.
    UnicodeError = 54,

    /// Unicode decoding error.
    UnicodeDecodeError = 55,

    // ═══════════════════════════════════════════════════════════════════════
    // More Unicode/Warning (56-63)
    // ═══════════════════════════════════════════════════════════════════════
    /// Unicode encoding error (specific).
    UnicodeEncodeError = 56,

    /// Unicode translation error.
    UnicodeTranslateError = 57,

    /// Base class for warnings.
    Warning = 58,

    /// Deprecation warning.
    DeprecationWarning = 59,

    /// Pending deprecation warning.
    PendingDeprecationWarning = 60,

    /// Runtime warning.
    RuntimeWarning = 61,

    /// Syntax warning.
    SyntaxWarning = 62,

    /// User warning.
    UserWarning = 63,
}

impl ExceptionTypeId {
    /// Total number of built-in exception types.
    pub const COUNT: usize = 64;

    /// First user-definable exception ID.
    pub const USER_DEFINED_START: u8 = 128;

    /// Maximum exception ID.
    pub const MAX_ID: u8 = 255;

    /// Returns the numeric ID of this exception type.
    #[inline(always)]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Creates an ExceptionTypeId from a raw u8.
    ///
    /// Returns None if the ID is reserved or invalid.
    #[inline]
    pub const fn from_u8(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::BaseException),
            1 => Some(Self::SystemExit),
            2 => Some(Self::KeyboardInterrupt),
            3 => Some(Self::GeneratorExit),
            4 => Some(Self::Exception),
            5 => Some(Self::StopIteration),
            6 => Some(Self::StopAsyncIteration),
            7 => Some(Self::ArithmeticError),
            8 => Some(Self::OverflowError),
            9 => Some(Self::ZeroDivisionError),
            10 => Some(Self::FloatingPointError),
            11 => Some(Self::AssertionError),
            12 => Some(Self::AttributeError),
            13 => Some(Self::BufferError),
            14 => Some(Self::EOFError),
            15 => Some(Self::ImportError),
            16 => Some(Self::LookupError),
            17 => Some(Self::IndexError),
            18 => Some(Self::KeyError),
            19 => Some(Self::ModuleNotFoundError),
            20 => Some(Self::MemoryError),
            21 => Some(Self::NameError),
            22 => Some(Self::UnboundLocalError),
            24 => Some(Self::OSError),
            25 => Some(Self::FileNotFoundError),
            26 => Some(Self::FileExistsError),
            27 => Some(Self::NotADirectoryError),
            28 => Some(Self::IsADirectoryError),
            29 => Some(Self::PermissionError),
            30 => Some(Self::ProcessLookupError),
            31 => Some(Self::ConnectionError),
            32 => Some(Self::ConnectionRefusedError),
            33 => Some(Self::ConnectionResetError),
            34 => Some(Self::ConnectionAbortedError),
            35 => Some(Self::BrokenPipeError),
            36 => Some(Self::TimeoutError),
            37 => Some(Self::InterruptedError),
            38 => Some(Self::ChildProcessError),
            39 => Some(Self::BlockingIOError),
            40 => Some(Self::ReferenceError),
            41 => Some(Self::RuntimeError),
            42 => Some(Self::RecursionError),
            43 => Some(Self::NotImplementedError),
            44 => Some(Self::BaseExceptionGroup),
            45 => Some(Self::ExceptionGroup),
            48 => Some(Self::SyntaxError),
            49 => Some(Self::IndentationError),
            50 => Some(Self::TabError),
            51 => Some(Self::SystemError),
            52 => Some(Self::TypeError),
            53 => Some(Self::ValueError),
            54 => Some(Self::UnicodeError),
            55 => Some(Self::UnicodeDecodeError),
            56 => Some(Self::UnicodeEncodeError),
            57 => Some(Self::UnicodeTranslateError),
            58 => Some(Self::Warning),
            59 => Some(Self::DeprecationWarning),
            60 => Some(Self::PendingDeprecationWarning),
            61 => Some(Self::RuntimeWarning),
            62 => Some(Self::SyntaxWarning),
            63 => Some(Self::UserWarning),
            _ => None,
        }
    }

    /// Returns the Python name of this exception type.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::BaseException => "BaseException",
            Self::SystemExit => "SystemExit",
            Self::KeyboardInterrupt => "KeyboardInterrupt",
            Self::GeneratorExit => "GeneratorExit",
            Self::Exception => "Exception",
            Self::StopIteration => "StopIteration",
            Self::StopAsyncIteration => "StopAsyncIteration",
            Self::ArithmeticError => "ArithmeticError",
            Self::OverflowError => "OverflowError",
            Self::ZeroDivisionError => "ZeroDivisionError",
            Self::FloatingPointError => "FloatingPointError",
            Self::AssertionError => "AssertionError",
            Self::AttributeError => "AttributeError",
            Self::BufferError => "BufferError",
            Self::EOFError => "EOFError",
            Self::ImportError => "ImportError",
            Self::LookupError => "LookupError",
            Self::IndexError => "IndexError",
            Self::KeyError => "KeyError",
            Self::ModuleNotFoundError => "ModuleNotFoundError",
            Self::MemoryError => "MemoryError",
            Self::NameError => "NameError",
            Self::UnboundLocalError => "UnboundLocalError",
            Self::_Reserved23 => "_Reserved23",
            Self::OSError => "OSError",
            Self::FileNotFoundError => "FileNotFoundError",
            Self::FileExistsError => "FileExistsError",
            Self::NotADirectoryError => "NotADirectoryError",
            Self::IsADirectoryError => "IsADirectoryError",
            Self::PermissionError => "PermissionError",
            Self::ProcessLookupError => "ProcessLookupError",
            Self::ConnectionError => "ConnectionError",
            Self::ConnectionRefusedError => "ConnectionRefusedError",
            Self::ConnectionResetError => "ConnectionResetError",
            Self::ConnectionAbortedError => "ConnectionAbortedError",
            Self::BrokenPipeError => "BrokenPipeError",
            Self::TimeoutError => "TimeoutError",
            Self::InterruptedError => "InterruptedError",
            Self::ChildProcessError => "ChildProcessError",
            Self::BlockingIOError => "BlockingIOError",
            Self::ReferenceError => "ReferenceError",
            Self::RuntimeError => "RuntimeError",
            Self::RecursionError => "RecursionError",
            Self::NotImplementedError => "NotImplementedError",
            Self::BaseExceptionGroup => "BaseExceptionGroup",
            Self::ExceptionGroup => "ExceptionGroup",
            Self::_Reserved46 => "_Reserved46",
            Self::_Reserved47 => "_Reserved47",
            Self::SyntaxError => "SyntaxError",
            Self::IndentationError => "IndentationError",
            Self::TabError => "TabError",
            Self::SystemError => "SystemError",
            Self::TypeError => "TypeError",
            Self::ValueError => "ValueError",
            Self::UnicodeError => "UnicodeError",
            Self::UnicodeDecodeError => "UnicodeDecodeError",
            Self::UnicodeEncodeError => "UnicodeEncodeError",
            Self::UnicodeTranslateError => "UnicodeTranslateError",
            Self::Warning => "Warning",
            Self::DeprecationWarning => "DeprecationWarning",
            Self::PendingDeprecationWarning => "PendingDeprecationWarning",
            Self::RuntimeWarning => "RuntimeWarning",
            Self::SyntaxWarning => "SyntaxWarning",
            Self::UserWarning => "UserWarning",
        }
    }

    /// Returns the parent type in the exception hierarchy.
    ///
    /// This enables O(log N) subclass checking by walking up the hierarchy.
    #[inline]
    pub const fn parent(self) -> Option<Self> {
        match self {
            // BaseException is the root
            Self::BaseException => None,

            // Direct children of BaseException
            Self::SystemExit => Some(Self::BaseException),
            Self::KeyboardInterrupt => Some(Self::BaseException),
            Self::GeneratorExit => Some(Self::BaseException),
            Self::Exception => Some(Self::BaseException),

            // StopIteration hierarchy
            Self::StopIteration => Some(Self::Exception),
            Self::StopAsyncIteration => Some(Self::Exception),

            // ArithmeticError hierarchy
            Self::ArithmeticError => Some(Self::Exception),
            Self::OverflowError => Some(Self::ArithmeticError),
            Self::ZeroDivisionError => Some(Self::ArithmeticError),
            Self::FloatingPointError => Some(Self::ArithmeticError),

            // Direct Exception subclasses
            Self::AssertionError => Some(Self::Exception),
            Self::AttributeError => Some(Self::Exception),
            Self::BufferError => Some(Self::Exception),
            Self::EOFError => Some(Self::Exception),
            Self::ImportError => Some(Self::Exception),

            // LookupError hierarchy
            Self::LookupError => Some(Self::Exception),
            Self::IndexError => Some(Self::LookupError),
            Self::KeyError => Some(Self::LookupError),
            Self::ModuleNotFoundError => Some(Self::ImportError),

            // Memory/Name
            Self::MemoryError => Some(Self::Exception),
            Self::NameError => Some(Self::Exception),
            Self::UnboundLocalError => Some(Self::NameError),
            Self::_Reserved23 => Some(Self::Exception),

            // OSError hierarchy
            Self::OSError => Some(Self::Exception),
            Self::FileNotFoundError => Some(Self::OSError),
            Self::FileExistsError => Some(Self::OSError),
            Self::NotADirectoryError => Some(Self::OSError),
            Self::IsADirectoryError => Some(Self::OSError),
            Self::PermissionError => Some(Self::OSError),
            Self::ProcessLookupError => Some(Self::OSError),
            Self::ConnectionError => Some(Self::OSError),
            Self::ConnectionRefusedError => Some(Self::ConnectionError),
            Self::ConnectionResetError => Some(Self::ConnectionError),
            Self::ConnectionAbortedError => Some(Self::ConnectionError),
            Self::BrokenPipeError => Some(Self::ConnectionError),
            Self::TimeoutError => Some(Self::OSError),
            Self::InterruptedError => Some(Self::OSError),
            Self::ChildProcessError => Some(Self::OSError),
            Self::BlockingIOError => Some(Self::OSError),

            // Reference/Runtime hierarchy
            Self::ReferenceError => Some(Self::Exception),
            Self::RuntimeError => Some(Self::Exception),
            Self::RecursionError => Some(Self::RuntimeError),
            Self::NotImplementedError => Some(Self::RuntimeError),
            Self::BaseExceptionGroup => Some(Self::BaseException),
            Self::ExceptionGroup => Some(Self::BaseExceptionGroup),
            Self::_Reserved46 => Some(Self::Exception),
            Self::_Reserved47 => Some(Self::Exception),

            // Syntax hierarchy
            Self::SyntaxError => Some(Self::Exception),
            Self::IndentationError => Some(Self::SyntaxError),
            Self::TabError => Some(Self::IndentationError),
            Self::SystemError => Some(Self::Exception),

            // Type/Value hierarchy
            Self::TypeError => Some(Self::Exception),
            Self::ValueError => Some(Self::Exception),
            Self::UnicodeError => Some(Self::ValueError),
            Self::UnicodeDecodeError => Some(Self::UnicodeError),
            Self::UnicodeEncodeError => Some(Self::UnicodeError),
            Self::UnicodeTranslateError => Some(Self::UnicodeError),

            // Warning hierarchy
            Self::Warning => Some(Self::Exception),
            Self::DeprecationWarning => Some(Self::Warning),
            Self::PendingDeprecationWarning => Some(Self::Warning),
            Self::RuntimeWarning => Some(Self::Warning),
            Self::SyntaxWarning => Some(Self::Warning),
            Self::UserWarning => Some(Self::Warning),
        }
    }

    /// Returns an additional direct parent for built-in types that model a
    /// small amount of multiple inheritance in CPython's exception hierarchy.
    #[inline]
    pub const fn secondary_parent(self) -> Option<Self> {
        match self {
            Self::ExceptionGroup => Some(Self::Exception),
            _ => None,
        }
    }

    /// Checks if this exception type is a subclass of another.
    ///
    /// This walks up the hierarchy, so worst case is O(depth).
    /// For Python's hierarchy, max depth is 4 (TabError → Indentation → Syntax → Exception → Base).
    #[inline]
    pub fn is_subclass_of(self, base: Self) -> bool {
        if self == base {
            return true;
        }

        if let Some(secondary_parent) = self.secondary_parent()
            && secondary_parent.is_subclass_of(base)
        {
            return true;
        }

        let mut current = self;
        while let Some(parent) = current.parent() {
            if parent == base {
                return true;
            }
            current = parent;
        }
        false
    }

    /// Returns the depth of this exception in the hierarchy.
    ///
    /// BaseException has depth 0.
    #[inline]
    pub fn depth(self) -> u8 {
        let mut depth = 0u8;
        let mut current = self;
        while let Some(parent) = current.parent() {
            depth += 1;
            current = parent;
        }

        if let Some(secondary_parent) = self.secondary_parent() {
            depth = depth.min(secondary_parent.depth().saturating_add(1));
        }

        depth
    }

    /// Returns true if this is a control-flow exception (not a real error).
    ///
    /// These exceptions are used for control flow and should be handled
    /// specially for performance (e.g., flyweight instances).
    #[inline]
    pub const fn is_control_flow(self) -> bool {
        matches!(
            self,
            Self::StopIteration | Self::StopAsyncIteration | Self::GeneratorExit
        )
    }

    /// Returns true if this is a system-exiting exception.
    ///
    /// These bypass normal exception handling.
    #[inline]
    pub const fn is_system_exit(self) -> bool {
        matches!(self, Self::SystemExit | Self::KeyboardInterrupt)
    }

    /// Returns true if this is a warning, not an error.
    #[inline]
    pub const fn is_warning(self) -> bool {
        matches!(
            self,
            Self::Warning
                | Self::DeprecationWarning
                | Self::PendingDeprecationWarning
                | Self::RuntimeWarning
                | Self::SyntaxWarning
                | Self::UserWarning
        )
    }

    /// Returns true if this is an OSError or subclass.
    #[inline]
    pub const fn is_os_error(self) -> bool {
        matches!(self.as_u8(), 24..=39)
    }
}

impl fmt::Display for ExceptionTypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl Default for ExceptionTypeId {
    #[inline]
    fn default() -> Self {
        Self::Exception
    }
}
