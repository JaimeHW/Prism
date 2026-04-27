//! Type objects and type system.
//!
//! Implements Python's type object infrastructure for method dispatch.

use crate::object::{ObjectHeader, PyObject};
use prism_core::intern::InternedString;
use std::fmt;

// =============================================================================
// Type ID
// =============================================================================

/// Compact type identifier for fast type checking.
///
/// Uses a u32 to avoid pointer chasing. Well-known types have
/// fixed IDs; user-defined types get IDs from a counter.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct TypeId(pub u32);

impl TypeId {
    // Built-in type IDs (0-255 reserved)
    pub const NONE: Self = Self(0);
    pub const BOOL: Self = Self(1);
    pub const INT: Self = Self(2);
    pub const FLOAT: Self = Self(3);
    pub const STR: Self = Self(4);
    pub const BYTES: Self = Self(5);
    pub const LIST: Self = Self(6);
    pub const TUPLE: Self = Self(7);
    pub const DICT: Self = Self(8);
    pub const SET: Self = Self(9);
    pub const FROZENSET: Self = Self(10);
    pub const FUNCTION: Self = Self(11);
    pub const METHOD: Self = Self(12);
    pub const CLOSURE: Self = Self(13);
    pub const CODE: Self = Self(14);
    pub const MODULE: Self = Self(15);
    pub const TYPE: Self = Self(16);
    pub const OBJECT: Self = Self(17);
    pub const SLICE: Self = Self(18);
    pub const RANGE: Self = Self(19);
    pub const ITERATOR: Self = Self(20);
    pub const GENERATOR: Self = Self(21);
    pub const EXCEPTION: Self = Self(22);
    pub const BUILTIN_FUNCTION: Self = Self(23);
    pub const SUPER: Self = Self(24);
    pub const CELL: Self = Self(25);
    pub const MODULE_OBJECT: Self = Self(26);
    pub const EXCEPTION_TYPE: Self = Self(27);
    pub const BYTEARRAY: Self = Self(28);
    pub const CLASSMETHOD: Self = Self(29);
    pub const STATICMETHOD: Self = Self(30);
    pub const PROPERTY: Self = Self(31);
    pub const CELL_VIEW: Self = Self(32);
    pub const GENERIC_ALIAS: Self = Self(33);
    pub const UNION: Self = Self(34);
    pub const MAPPING_PROXY: Self = Self(35);
    pub const WRAPPER_DESCRIPTOR: Self = Self(36);
    pub const METHOD_WRAPPER: Self = Self(37);
    pub const METHOD_DESCRIPTOR: Self = Self(38);
    pub const CLASSMETHOD_DESCRIPTOR: Self = Self(39);
    pub const GETSET_DESCRIPTOR: Self = Self(40);
    pub const MEMBER_DESCRIPTOR: Self = Self(41);
    pub const TRACEBACK: Self = Self(42);
    pub const FRAME: Self = Self(43);
    pub const ELLIPSIS: Self = Self(44);
    pub const NOT_IMPLEMENTED: Self = Self(45);
    pub const DICT_KEYS: Self = Self(46);
    pub const DICT_VALUES: Self = Self(47);
    pub const DICT_ITEMS: Self = Self(48);
    pub const MEMORYVIEW: Self = Self(49);
    pub const DEQUE: Self = Self(50);
    pub const REGEX_PATTERN: Self = Self(51);
    pub const REGEX_MATCH: Self = Self(52);
    pub const COMPLEX: Self = Self(53);
    pub const ENUMERATE: Self = Self(54);

    /// First ID available for user-defined types.
    pub const FIRST_USER_TYPE: u32 = 256;

    /// Get raw value.
    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Create a TypeId from a raw value.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Check if this is a built-in type.
    #[inline]
    pub const fn is_builtin(self) -> bool {
        self.0 < Self::FIRST_USER_TYPE
    }

    /// Get type name for debugging.
    pub fn name(self) -> &'static str {
        match self.0 {
            0 => "NoneType",
            1 => "bool",
            2 => "int",
            3 => "float",
            4 => "str",
            5 => "bytes",
            6 => "list",
            7 => "tuple",
            8 => "dict",
            9 => "set",
            10 => "frozenset",
            11 => "function",
            12 => "method",
            13 => "closure",
            14 => "code",
            15 => "module",
            16 => "type",
            17 => "object",
            18 => "slice",
            19 => "range",
            20 => "iterator",
            21 => "generator",
            22 => "BaseException",
            23 => "builtin_function",
            24 => "super",
            25 => "cell",
            26 => "module",
            27 => "exception_type",
            28 => "bytearray",
            29 => "classmethod",
            30 => "staticmethod",
            31 => "property",
            32 => "cell",
            33 => "generic_alias",
            34 => "union",
            35 => "mappingproxy",
            36 => "wrapper_descriptor",
            37 => "method-wrapper",
            38 => "method_descriptor",
            39 => "classmethod_descriptor",
            40 => "getset_descriptor",
            41 => "member_descriptor",
            42 => "traceback",
            43 => "frame",
            44 => "ellipsis",
            45 => "NotImplementedType",
            46 => "dict_keys",
            47 => "dict_values",
            48 => "dict_items",
            49 => "memoryview",
            50 => "deque",
            51 => "Pattern",
            52 => "Match",
            53 => "complex",
            54 => "enumerate",
            _ => "<unknown>",
        }
    }
}

impl fmt::Debug for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TypeId({}: {})", self.0, self.name())
    }
}

// =============================================================================
// Type Flags
// =============================================================================

bitflags::bitflags! {
    /// Flags describing type capabilities.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TypeFlags: u32 {
        /// Type is a heap-allocated object (not built-in).
        const HEAPTYPE = 1 << 0;
        /// Type is abstract (cannot be instantiated).
        const ABSTRACT = 1 << 1;
        /// Type has __dict__ slot.
        const HAS_DICT = 1 << 2;
        /// Type is immutable (hash can be cached).
        const IMMUTABLE = 1 << 3;
        /// Type supports sequence protocol.
        const SEQUENCE = 1 << 4;
        /// Type supports mapping protocol.
        const MAPPING = 1 << 5;
        /// Type supports number protocol.
        const NUMBER = 1 << 6;
        /// Type is iterable.
        const ITERABLE = 1 << 7;
        /// Type is an iterator.
        const ITERATOR = 1 << 8;
        /// Type is callable.
        const CALLABLE = 1 << 9;
        /// Type has custom __hash__.
        const HASHABLE = 1 << 10;
        /// Type uses gc (contains references).
        const GC = 1 << 11;
    }
}

// =============================================================================
// Slot Function Signatures
// =============================================================================

use prism_core::{PrismResult, Value};

/// Call slot: (self, args, kwargs) -> result
pub type CallSlot = fn(Value, &[Value], Option<&[(InternedString, Value)]>) -> PrismResult<Value>;

/// Unary operation slot: (self) -> result
pub type UnarySlot = fn(Value) -> PrismResult<Value>;

/// Binary operation slot: (self, other) -> result
pub type BinarySlot = fn(Value, Value) -> PrismResult<Value>;

/// Ternary operation slot: (self, a, b) -> result
pub type TernarySlot = fn(Value, Value, Value) -> PrismResult<Value>;

/// Get attribute slot: (self, name) -> result
pub type GetattrSlot = fn(Value, &str) -> PrismResult<Value>;

/// Set attribute slot: (self, name, value) -> result
pub type SetattrSlot = fn(Value, &str, Value) -> PrismResult<()>;

/// Delete attribute slot: (self, name) -> result
pub type DelattrSlot = fn(Value, &str) -> PrismResult<()>;

/// Hash slot: (self) -> hash
pub type HashSlot = fn(Value) -> PrismResult<u64>;

/// Length slot: (self) -> length
pub type LenSlot = fn(Value) -> PrismResult<usize>;

/// Iterator slot: (self) -> iterator
pub type IterSlot = fn(Value) -> PrismResult<Value>;

/// Next slot: (self) -> next item or StopIteration
pub type NextSlot = fn(Value) -> PrismResult<Option<Value>>;

/// Get item slot: (self, key) -> value
pub type GetItemSlot = fn(Value, Value) -> PrismResult<Value>;

/// Set item slot: (self, key, value) -> ()
pub type SetItemSlot = fn(Value, Value, Value) -> PrismResult<()>;

/// Delete item slot: (self, key) -> ()
pub type DelItemSlot = fn(Value, Value) -> PrismResult<()>;

/// Contains slot: (self, item) -> bool
pub type ContainsSlot = fn(Value, Value) -> PrismResult<bool>;

/// Repr slot: (self) -> string
pub type ReprSlot = fn(Value) -> PrismResult<String>;

/// Bool slot: (self) -> bool (truthiness)
pub type BoolSlot = fn(Value) -> PrismResult<bool>;

// =============================================================================
// Type Slots
// =============================================================================

/// Slot table for type dispatch.
///
/// All slots are optional; None means the type doesn't support that operation.
#[derive(Default)]
pub struct TypeSlots {
    // Basic slots
    pub tp_call: Option<CallSlot>,
    pub tp_hash: Option<HashSlot>,
    pub tp_repr: Option<ReprSlot>,
    pub tp_str: Option<ReprSlot>,
    pub tp_bool: Option<BoolSlot>,

    // Attribute access
    pub tp_getattr: Option<GetattrSlot>,
    pub tp_setattr: Option<SetattrSlot>,
    pub tp_delattr: Option<DelattrSlot>,

    // Iteration
    pub tp_iter: Option<IterSlot>,
    pub tp_next: Option<NextSlot>,

    // Sequence protocol
    pub sq_length: Option<LenSlot>,
    pub sq_item: Option<GetItemSlot>,
    pub sq_ass_item: Option<SetItemSlot>,
    pub sq_contains: Option<ContainsSlot>,

    // Mapping protocol
    pub mp_length: Option<LenSlot>,
    pub mp_subscript: Option<GetItemSlot>,
    pub mp_ass_subscript: Option<SetItemSlot>,

    // Number protocol
    pub nb_add: Option<BinarySlot>,
    pub nb_sub: Option<BinarySlot>,
    pub nb_mul: Option<BinarySlot>,
    pub nb_truediv: Option<BinarySlot>,
    pub nb_floordiv: Option<BinarySlot>,
    pub nb_mod: Option<BinarySlot>,
    pub nb_pow: Option<TernarySlot>,
    pub nb_neg: Option<UnarySlot>,
    pub nb_pos: Option<UnarySlot>,
    pub nb_abs: Option<UnarySlot>,
    pub nb_bool: Option<BoolSlot>,
    pub nb_int: Option<UnarySlot>,
    pub nb_float: Option<UnarySlot>,

    // Bitwise
    pub nb_and: Option<BinarySlot>,
    pub nb_or: Option<BinarySlot>,
    pub nb_xor: Option<BinarySlot>,
    pub nb_invert: Option<UnarySlot>,
    pub nb_lshift: Option<BinarySlot>,
    pub nb_rshift: Option<BinarySlot>,
}

impl std::fmt::Debug for TypeSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeSlots")
            .field("tp_call", &self.tp_call.map(|_| "fn(...)"))
            .field("tp_hash", &self.tp_hash.map(|_| "fn(...)"))
            .field("tp_iter", &self.tp_iter.map(|_| "fn(...)"))
            .finish_non_exhaustive()
    }
}

// =============================================================================
// Type Object
// =============================================================================

/// Type object - describes a Python type.
///
/// Contains the type's name, base class, method slots, and metadata.
pub struct TypeObject {
    /// Object header (type of type is TYPE).
    pub header: ObjectHeader,
    /// Runtime type identifier represented by this type object.
    type_id: TypeId,
    /// Type name.
    pub name: InternedString,
    /// Base type (None for object).
    pub base: Option<&'static TypeObject>,
    /// Method slots for dispatch.
    pub slots: TypeSlots,
    /// Instance size in bytes (for allocation).
    pub instance_size: usize,
    /// Type capability flags.
    pub flags: TypeFlags,
}

impl TypeObject {
    /// Create a new type object.
    pub fn new(
        type_id: TypeId,
        name: InternedString,
        base: Option<&'static TypeObject>,
        instance_size: usize,
        flags: TypeFlags,
    ) -> Self {
        Self {
            header: ObjectHeader::new(TypeId::TYPE),
            type_id,
            name,
            base,
            slots: TypeSlots::default(),
            instance_size,
            flags,
        }
    }

    /// Get the type ID.
    pub fn type_id(&self) -> TypeId {
        self.type_id
    }

    /// Check if type is callable.
    #[inline]
    pub fn is_callable(&self) -> bool {
        self.slots.tp_call.is_some() || self.flags.contains(TypeFlags::CALLABLE)
    }

    /// Check if type is iterable.
    #[inline]
    pub fn is_iterable(&self) -> bool {
        self.slots.tp_iter.is_some() || self.flags.contains(TypeFlags::ITERABLE)
    }

    /// Check if type is hashable.
    #[inline]
    pub fn is_hashable(&self) -> bool {
        self.slots.tp_hash.is_some() || self.flags.contains(TypeFlags::HASHABLE)
    }

    /// Check if type supports sequence protocol.
    #[inline]
    pub fn is_sequence(&self) -> bool {
        self.slots.sq_length.is_some() || self.flags.contains(TypeFlags::SEQUENCE)
    }

    /// Check if type supports mapping protocol.
    #[inline]
    pub fn is_mapping(&self) -> bool {
        self.slots.mp_subscript.is_some() || self.flags.contains(TypeFlags::MAPPING)
    }
}

impl PyObject for TypeObject {
    fn header(&self) -> &ObjectHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut ObjectHeader {
        &mut self.header
    }
}
