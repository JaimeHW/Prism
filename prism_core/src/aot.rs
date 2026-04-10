//! Shared AOT bootstrap ABI definitions.
//!
//! These types define the stable interface between Prism's native AOT stubs
//! and the runtime helpers that execute module initialization work.

/// Runtime symbol used by native stubs for `import module` operations.
pub const AOT_IMPORT_MODULE_SYMBOL: &str = "prism_aot_op_import_module";
/// Runtime symbol used by native stubs for `from module import name` operations.
pub const AOT_IMPORT_FROM_SYMBOL: &str = "prism_aot_op_import_from";
/// Runtime symbol used by native stubs for top-level assignment operations.
pub const AOT_STORE_EXPR_SYMBOL: &str = "prism_aot_op_store_expr";
/// Symbol exported at the beginning of the native-init registry.
pub const AOT_NATIVE_INIT_TABLE_START_SYMBOL: &str = "prism_aot_native_init_table_start";
/// Symbol exported immediately after the native-init registry.
pub const AOT_NATIVE_INIT_TABLE_END_SYMBOL: &str = "prism_aot_native_init_table_end";

/// Result status returned from native AOT helper calls.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotOpStatus {
    /// The operation completed successfully.
    Ok = 0,
    /// The operation failed and the VM recorded a runtime error.
    Error = 1,
}

impl AotOpStatus {
    /// Whether the status indicates success.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// Borrowed UTF-8 string used by the AOT bootstrap ABI.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotStringRef {
    /// Pointer to UTF-8 bytes.
    pub data: *const u8,
    /// Length in bytes.
    pub len: usize,
}

impl AotStringRef {
    /// Create an empty string reference.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            data: core::ptr::null(),
            len: 0,
        }
    }
}

/// One native module-init entry exported by an AOT artifact.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotNativeModuleInitEntry {
    /// Canonical module name associated with the init stub.
    pub module_name: AotStringRef,
    /// Function pointer for the module init stub.
    pub init_fn: *const (),
}

/// Immediate value category used by assignment operands.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotImmediateKind {
    /// `bits` holds a fully encoded [`crate::Value`] payload.
    ValueBits = 0,
    /// `string` holds a UTF-8 string literal that should be interned at runtime.
    String = 1,
}

/// Immediate operand payload for native module initialization.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotImmediate {
    /// Immediate encoding kind.
    pub kind: AotImmediateKind,
    /// Reserved for future ABI extension and alignment.
    pub reserved: [u8; 7],
    /// Encoded [`crate::Value`] bits for non-string immediates.
    pub bits: u64,
    /// UTF-8 string literal storage for string immediates.
    pub string: AotStringRef,
}

impl AotImmediate {
    /// Create an immediate from raw [`crate::Value`] bits.
    #[must_use]
    pub const fn value_bits(bits: u64) -> Self {
        Self {
            kind: AotImmediateKind::ValueBits,
            reserved: [0; 7],
            bits,
            string: AotStringRef::empty(),
        }
    }

    /// Create a string-literal immediate.
    #[must_use]
    pub const fn string(string: AotStringRef) -> Self {
        Self {
            kind: AotImmediateKind::String,
            reserved: [0; 7],
            bits: 0,
            string,
        }
    }
}

/// Operand category used by native assignments.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotOperandKind {
    /// Operand is an immediate literal.
    Immediate = 0,
    /// Operand loads an existing module/global name.
    Name = 1,
}

/// One operand consumed by an assignment expression.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotOperand {
    /// Operand kind.
    pub kind: AotOperandKind,
    /// Reserved for future ABI extension and alignment.
    pub reserved: [u8; 7],
    /// Immediate payload when `kind == Immediate`.
    pub immediate: AotImmediate,
    /// Source name when `kind == Name`.
    pub name: AotStringRef,
}

impl AotOperand {
    /// Create an immediate operand.
    #[must_use]
    pub const fn immediate(immediate: AotImmediate) -> Self {
        Self {
            kind: AotOperandKind::Immediate,
            reserved: [0; 7],
            immediate,
            name: AotStringRef::empty(),
        }
    }

    /// Create a name operand.
    #[must_use]
    pub const fn name(name: AotStringRef) -> Self {
        Self {
            kind: AotOperandKind::Name,
            reserved: [0; 7],
            immediate: AotImmediate::value_bits(0),
            name,
        }
    }
}

/// Binding mode for `import module` statements.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotImportBinding {
    /// Bind the exact imported module object.
    Exact = 0,
    /// Bind the top-level package object for dotted imports.
    TopLevel = 1,
}

/// Descriptor for a native `import module` top-level statement.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotImportModuleOp {
    /// Target name written into module scope.
    pub target: AotStringRef,
    /// Raw import spec, including any leading dots for relative imports.
    pub module_spec: AotStringRef,
    /// Whether to bind the exact module or the top-level package.
    pub binding: AotImportBinding,
    /// Reserved for future ABI extension and alignment.
    pub reserved: [u8; 7],
}

/// Descriptor for a native `from module import name` top-level statement.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotImportFromOp {
    /// Target name written into module scope.
    pub target: AotStringRef,
    /// Raw module spec, including any leading dots for relative imports.
    pub module_spec: AotStringRef,
    /// Imported attribute or submodule name.
    pub attribute: AotStringRef,
}

/// Expression shape supported by the initial native module-init pipeline.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AotStoreExprKind {
    /// Store a single operand directly.
    Operand = 0,
    /// Store the result of `left + right`.
    Add = 1,
}

/// Descriptor for a native top-level assignment.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AotStoreExprOp {
    /// Target name written into module scope.
    pub target: AotStringRef,
    /// Expression kind.
    pub kind: AotStoreExprKind,
    /// Reserved for future ABI extension and alignment.
    pub reserved: [u8; 7],
    /// Left operand, or the only operand when `kind == Operand`.
    pub left: AotOperand,
    /// Right operand when `kind == Add`.
    pub right: AotOperand,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aot_status_ok() {
        assert!(AotOpStatus::Ok.is_ok());
        assert!(!AotOpStatus::Error.is_ok());
    }

    #[test]
    fn test_aot_string_ref_empty() {
        let string = AotStringRef::empty();
        assert!(string.data.is_null());
        assert_eq!(string.len, 0);
    }

    #[test]
    fn test_native_module_init_entry_layout() {
        let entry = AotNativeModuleInitEntry {
            module_name: AotStringRef::empty(),
            init_fn: core::ptr::null(),
        };
        assert_eq!(entry.module_name.len, 0);
        assert!(entry.init_fn.is_null());
    }

    #[test]
    fn test_aot_immediate_constructors() {
        let bits = AotImmediate::value_bits(123);
        assert_eq!(bits.kind, AotImmediateKind::ValueBits);
        assert_eq!(bits.bits, 123);

        let string = AotImmediate::string(AotStringRef::empty());
        assert_eq!(string.kind, AotImmediateKind::String);
    }

    #[test]
    fn test_aot_operand_constructors() {
        let immediate = AotOperand::immediate(AotImmediate::value_bits(7));
        assert_eq!(immediate.kind, AotOperandKind::Immediate);

        let name = AotOperand::name(AotStringRef::empty());
        assert_eq!(name.kind, AotOperandKind::Name);
    }
}
