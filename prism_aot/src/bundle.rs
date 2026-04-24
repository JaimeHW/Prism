use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use prism_code::KwNamesTuple;
use prism_code::{CodeObject, Constant, Opcode};
use prism_core::Value;
use prism_core::intern::interned_by_ptr;

use crate::error::AotError;
use crate::planner::{BuildPlan, ModuleKind};

const BUNDLE_MAGIC: &[u8; 8] = b"PRMBNDL1";
const CONSTANT_NONE: u8 = 0;
const CONSTANT_BOOL: u8 = 1;
const CONSTANT_INT: u8 = 2;
const CONSTANT_FLOAT_BITS: u8 = 3;
const CONSTANT_STRING: u8 = 4;
const CONSTANT_NESTED_CODE: u8 = 5;
const CONSTANT_KW_NAMES: u8 = 6;
const CONSTANT_BIGINT: u8 = 7;

/// Serialized frozen module bundle for downstream native link steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenModuleBundle {
    /// Schema version for the frozen bundle format.
    pub format_version: u32,
    /// Target triple or identifier selected for compilation.
    pub target: String,
    /// Entrypoint metadata for the native bootstrap.
    pub entry: FrozenEntryImage,
    /// Deterministic frozen module list.
    pub modules: Vec<FrozenModuleImage>,
}

/// Serialized entrypoint metadata for the frozen bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenEntryImage {
    /// Canonical module name in the build graph.
    pub canonical_module: String,
    /// Runtime execution name for the entrypoint.
    pub execution_name: String,
    /// Package context for relative imports.
    pub package_name: String,
}

/// One frozen module in the bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenModuleImage {
    /// Canonical module name.
    pub name: String,
    /// Module origin.
    pub kind: ModuleKind,
    /// Package context.
    pub package_name: String,
    /// Whether the source module comes from `__init__.py`.
    pub is_package: bool,
    /// Source path when present.
    pub source_path: Option<String>,
    /// Serialized code image for source-backed modules.
    pub code: Option<CodeImage>,
}

/// Deterministic serialized form of a Prism code object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodeImage {
    /// Unqualified code object name.
    pub name: String,
    /// Qualified code object name.
    pub qualname: String,
    /// Source filename recorded in the code object.
    pub filename: String,
    /// First source line number.
    pub first_lineno: u32,
    /// Raw 32-bit instructions.
    pub instructions: Vec<u32>,
    /// Serialized constant pool.
    pub constants: Vec<ConstantImage>,
    /// Local names.
    pub locals: Vec<String>,
    /// Global/attribute names.
    pub names: Vec<String>,
    /// Free variable names.
    pub freevars: Vec<String>,
    /// Cell variable names.
    pub cellvars: Vec<String>,
    /// Positional parameter count.
    pub arg_count: u16,
    /// Positional-only parameter count.
    pub posonlyarg_count: u16,
    /// Keyword-only parameter count.
    pub kwonlyarg_count: u16,
    /// Virtual register count.
    pub register_count: u16,
    /// Raw code flags.
    pub flags: u32,
    /// Line table entries.
    pub line_table: Vec<LineTableImage>,
    /// Exception table entries.
    pub exception_table: Vec<ExceptionTableImage>,
    /// Nested code objects referenced by this code unit.
    pub nested_code_objects: Vec<CodeImage>,
}

/// Serialized constant pool entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstantImage {
    /// `None`.
    None,
    /// Boolean literal.
    Bool(bool),
    /// Inline integer literal.
    Int(i64),
    /// IEEE-754 float bits.
    FloatBits(u64),
    /// Interned string literal.
    String(String),
    /// Reference into `nested_code_objects`.
    NestedCode(u32),
    /// Keyword names tuple used by `CallKw`.
    KwNamesTuple(Vec<String>),
    /// Arbitrary-precision integer literal encoded in base 10.
    BigInt(String),
}

/// Serialized line table entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LineTableImage {
    /// Starting instruction offset.
    pub start_pc: u32,
    /// Ending instruction offset.
    pub end_pc: u32,
    /// Source line number.
    pub line: u32,
}

/// Serialized exception handler entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionTableImage {
    /// Start of try region.
    pub start_pc: u32,
    /// End of try region.
    pub end_pc: u32,
    /// Handler entry.
    pub handler_pc: u32,
    /// Finally entry, or `u32::MAX` if absent.
    pub finally_pc: u32,
    /// Handler nesting depth.
    pub depth: u16,
    /// Exception type index, or `u16::MAX` for bare `except`.
    pub exception_type_idx: u16,
}

impl FrozenModuleBundle {
    /// Construct a frozen module bundle from a whole-program build plan.
    pub fn from_build_plan(plan: &BuildPlan) -> Result<Self, AotError> {
        let modules = plan
            .modules
            .iter()
            .map(|module| {
                let code = match module.kind {
                    ModuleKind::Source => Some(module.code_image.clone().ok_or_else(|| {
                        AotError::InvalidArtifact {
                            message: format!(
                                "source module '{}' is missing a serialized code image",
                                module.name
                            ),
                        }
                    })?),
                    ModuleKind::Stdlib => None,
                };

                Ok(FrozenModuleImage {
                    name: module.name.clone(),
                    kind: module.kind,
                    package_name: module.package_name.clone(),
                    is_package: module.is_package,
                    source_path: module
                        .source_path
                        .as_ref()
                        .map(|path| path.display().to_string()),
                    code,
                })
            })
            .collect::<Result<Vec<_>, AotError>>()?;

        Ok(Self {
            format_version: 1,
            target: plan.target.clone(),
            entry: FrozenEntryImage {
                canonical_module: plan.entry.canonical_module.clone(),
                execution_name: plan.entry.execution_name.clone(),
                package_name: plan.entry.package_name.clone(),
            },
            modules,
        })
    }

    /// Encode the frozen bundle into a deterministic binary artifact.
    pub fn to_bytes(&self) -> Result<Vec<u8>, AotError> {
        let mut writer = BundleWriter::default();
        writer.write_bytes(BUNDLE_MAGIC);
        writer.write_u32(self.format_version);
        writer.write_string(&self.target)?;
        writer.write_string(&self.entry.canonical_module)?;
        writer.write_string(&self.entry.execution_name)?;
        writer.write_string(&self.entry.package_name)?;
        writer.write_len(self.modules.len())?;
        for module in &self.modules {
            writer.write_module(module)?;
        }
        Ok(writer.finish())
    }

    /// Decode a frozen bundle from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AotError> {
        let mut reader = BundleReader::new(bytes);
        let magic = reader.read_array::<8>()?;
        if &magic != BUNDLE_MAGIC {
            return Err(AotError::InvalidArtifact {
                message: format!(
                    "invalid frozen bundle magic: expected {:?}, got {:?}",
                    BUNDLE_MAGIC, magic
                ),
            });
        }

        let format_version = reader.read_u32()?;
        let target = reader.read_string()?;
        let entry = FrozenEntryImage {
            canonical_module: reader.read_string()?,
            execution_name: reader.read_string()?,
            package_name: reader.read_string()?,
        };
        let modules = reader.read_vec(|reader| reader.read_module())?;
        reader.finish()?;

        Ok(Self {
            format_version,
            target,
            entry,
            modules,
        })
    }

    /// Write the binary bundle to disk.
    pub fn write_to_path(&self, path: &Path) -> Result<(), AotError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| AotError::Io {
                path: parent.to_path_buf(),
                message: err.to_string(),
            })?;
        }

        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(|err| AotError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })
    }

    /// Read a binary bundle from disk.
    pub fn read_from_path(path: &Path) -> Result<Self, AotError> {
        let bytes = std::fs::read(path).map_err(|err| AotError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        Self::from_bytes(&bytes)
    }
}

impl CodeImage {
    /// Convert a Prism code object into a deterministic serialized image.
    pub fn from_code_object(module_name: &str, code: &CodeObject) -> Result<Self, AotError> {
        let nested_code_indices = code
            .nested_code_objects
            .iter()
            .enumerate()
            .map(|(index, nested)| (Arc::as_ptr(nested) as usize, index as u32))
            .collect::<HashMap<_, _>>();
        let kw_name_indices = kw_names_constant_indices(code);

        let constants = code
            .constants
            .iter()
            .enumerate()
            .map(|(index, constant)| {
                ConstantImage::from_constant(
                    module_name,
                    &code.qualname,
                    index,
                    constant,
                    &nested_code_indices,
                    &kw_name_indices,
                )
            })
            .collect::<Result<Vec<_>, AotError>>()?;

        let nested_code_objects = code
            .nested_code_objects
            .iter()
            .map(|nested| Self::from_code_object(module_name, nested))
            .collect::<Result<Vec<_>, AotError>>()?;

        Ok(Self {
            name: code.name.to_string(),
            qualname: code.qualname.to_string(),
            filename: code.filename.to_string(),
            first_lineno: code.first_lineno,
            instructions: code.instructions.iter().map(|inst| inst.raw()).collect(),
            constants,
            locals: code.locals.iter().map(|name| name.to_string()).collect(),
            names: code.names.iter().map(|name| name.to_string()).collect(),
            freevars: code.freevars.iter().map(|name| name.to_string()).collect(),
            cellvars: code.cellvars.iter().map(|name| name.to_string()).collect(),
            arg_count: code.arg_count,
            posonlyarg_count: code.posonlyarg_count,
            kwonlyarg_count: code.kwonlyarg_count,
            register_count: code.register_count,
            flags: code.flags.bits(),
            line_table: code
                .line_table
                .iter()
                .map(|entry| LineTableImage {
                    start_pc: entry.start_pc,
                    end_pc: entry.end_pc,
                    line: entry.line,
                })
                .collect(),
            exception_table: code
                .exception_table
                .iter()
                .map(|entry| ExceptionTableImage {
                    start_pc: entry.start_pc,
                    end_pc: entry.end_pc,
                    handler_pc: entry.handler_pc,
                    finally_pc: entry.finally_pc,
                    depth: entry.depth,
                    exception_type_idx: entry.exception_type_idx,
                })
                .collect(),
            nested_code_objects,
        })
    }
}

impl ConstantImage {
    fn from_constant(
        module_name: &str,
        code_name: &str,
        constant_index: usize,
        constant: &Constant,
        nested_code_indices: &HashMap<usize, u32>,
        kw_name_indices: &BTreeSet<usize>,
    ) -> Result<Self, AotError> {
        match constant {
            Constant::Value(value) => Self::from_materialized_value(
                module_name,
                code_name,
                constant_index,
                *value,
                nested_code_indices,
                kw_name_indices,
            ),
            Constant::BigInt(value) => Ok(Self::BigInt(value.to_string())),
        }
    }

    fn from_materialized_value(
        module_name: &str,
        code_name: &str,
        constant_index: usize,
        value: Value,
        nested_code_indices: &HashMap<usize, u32>,
        kw_name_indices: &BTreeSet<usize>,
    ) -> Result<Self, AotError> {
        if value.is_none() {
            return Ok(Self::None);
        }
        if let Some(boolean) = value.as_bool() {
            return Ok(Self::Bool(boolean));
        }
        if let Some(integer) = value.as_int() {
            return Ok(Self::Int(integer));
        }
        if let Some(float) = value.as_float() {
            return Ok(Self::FloatBits(float.to_bits()));
        }
        if let Some(string_ptr) = value.as_string_object_ptr() {
            let interned = interned_by_ptr(string_ptr as *const u8).ok_or_else(|| {
                AotError::InvalidArtifact {
                    message: format!(
                        "unable to resolve interned string constant {} in '{}' for module '{}'",
                        constant_index, code_name, module_name
                    ),
                }
            })?;
            return Ok(Self::String(interned.as_str().to_string()));
        }
        if let Some(object_ptr) = value.as_object_ptr() {
            if let Some(index) = nested_code_indices.get(&(object_ptr as usize)) {
                return Ok(Self::NestedCode(*index));
            }
            if kw_name_indices.contains(&constant_index) {
                let tuple = unsafe { &*(object_ptr as *const KwNamesTuple) };
                return Ok(Self::KwNamesTuple(
                    tuple.names.iter().map(|name| name.to_string()).collect(),
                ));
            }
        }

        Err(AotError::InvalidArtifact {
            message: format!(
                "unsupported constant at index {} in '{}' for module '{}'",
                constant_index, code_name, module_name
            ),
        })
    }
}

fn kw_names_constant_indices(code: &CodeObject) -> BTreeSet<usize> {
    let mut indices = BTreeSet::new();

    for window in code.instructions.windows(2) {
        let previous = Opcode::from_u8(window[0].opcode());
        let current = Opcode::from_u8(window[1].opcode());
        if previous == Some(Opcode::CallKw) && current == Some(Opcode::CallKwEx) {
            indices.insert((window[1].src1().0 as usize) | ((window[1].src2().0 as usize) << 8));
        }
    }

    indices
}

#[derive(Default)]
struct BundleWriter {
    bytes: Vec<u8>,
}

impl BundleWriter {
    fn finish(self) -> Vec<u8> {
        self.bytes
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        self.bytes.extend_from_slice(bytes);
    }

    fn write_u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn write_bool(&mut self, value: bool) {
        self.write_u8(u8::from(value));
    }

    fn write_u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_len(&mut self, len: usize) -> Result<(), AotError> {
        let len = u32::try_from(len).map_err(|_| AotError::InvalidArtifact {
            message: format!("artifact section exceeds u32 length limit: {}", len),
        })?;
        self.write_u32(len);
        Ok(())
    }

    fn write_string(&mut self, value: &str) -> Result<(), AotError> {
        self.write_len(value.len())?;
        self.write_bytes(value.as_bytes());
        Ok(())
    }

    fn write_optional_string(&mut self, value: Option<&str>) -> Result<(), AotError> {
        match value {
            Some(value) => {
                self.write_bool(true);
                self.write_string(value)?;
            }
            None => self.write_bool(false),
        }
        Ok(())
    }

    fn write_module(&mut self, module: &FrozenModuleImage) -> Result<(), AotError> {
        self.write_string(&module.name)?;
        self.write_u8(module_kind_tag(module.kind));
        self.write_string(&module.package_name)?;
        self.write_bool(module.is_package);
        self.write_optional_string(module.source_path.as_deref())?;
        match &module.code {
            Some(code) => {
                self.write_bool(true);
                self.write_code(code)?;
            }
            None => self.write_bool(false),
        }
        Ok(())
    }

    fn write_code(&mut self, code: &CodeImage) -> Result<(), AotError> {
        self.write_string(&code.name)?;
        self.write_string(&code.qualname)?;
        self.write_string(&code.filename)?;
        self.write_u32(code.first_lineno);

        self.write_len(code.instructions.len())?;
        for instruction in &code.instructions {
            self.write_u32(*instruction);
        }

        self.write_len(code.constants.len())?;
        for constant in &code.constants {
            self.write_constant(constant)?;
        }

        self.write_string_vec(&code.locals)?;
        self.write_string_vec(&code.names)?;
        self.write_string_vec(&code.freevars)?;
        self.write_string_vec(&code.cellvars)?;

        self.write_u16(code.arg_count);
        self.write_u16(code.posonlyarg_count);
        self.write_u16(code.kwonlyarg_count);
        self.write_u16(code.register_count);
        self.write_u32(code.flags);

        self.write_len(code.line_table.len())?;
        for entry in &code.line_table {
            self.write_u32(entry.start_pc);
            self.write_u32(entry.end_pc);
            self.write_u32(entry.line);
        }

        self.write_len(code.exception_table.len())?;
        for entry in &code.exception_table {
            self.write_u32(entry.start_pc);
            self.write_u32(entry.end_pc);
            self.write_u32(entry.handler_pc);
            self.write_u32(entry.finally_pc);
            self.write_u16(entry.depth);
            self.write_u16(entry.exception_type_idx);
        }

        self.write_len(code.nested_code_objects.len())?;
        for nested in &code.nested_code_objects {
            self.write_code(nested)?;
        }

        Ok(())
    }

    fn write_string_vec(&mut self, values: &[String]) -> Result<(), AotError> {
        self.write_len(values.len())?;
        for value in values {
            self.write_string(value)?;
        }
        Ok(())
    }

    fn write_constant(&mut self, constant: &ConstantImage) -> Result<(), AotError> {
        match constant {
            ConstantImage::None => self.write_u8(CONSTANT_NONE),
            ConstantImage::Bool(value) => {
                self.write_u8(CONSTANT_BOOL);
                self.write_bool(*value);
            }
            ConstantImage::Int(value) => {
                self.write_u8(CONSTANT_INT);
                self.write_i64(*value);
            }
            ConstantImage::FloatBits(value) => {
                self.write_u8(CONSTANT_FLOAT_BITS);
                self.write_u64(*value);
            }
            ConstantImage::String(value) => {
                self.write_u8(CONSTANT_STRING);
                self.write_string(value)?;
            }
            ConstantImage::NestedCode(index) => {
                self.write_u8(CONSTANT_NESTED_CODE);
                self.write_u32(*index);
            }
            ConstantImage::KwNamesTuple(names) => {
                self.write_u8(CONSTANT_KW_NAMES);
                self.write_len(names.len())?;
                for name in names {
                    self.write_string(name)?;
                }
            }
            ConstantImage::BigInt(value) => {
                self.write_u8(CONSTANT_BIGINT);
                self.write_string(value)?;
            }
        }
        Ok(())
    }
}

struct BundleReader<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> BundleReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn finish(&self) -> Result<(), AotError> {
        if self.cursor == self.bytes.len() {
            Ok(())
        } else {
            Err(AotError::InvalidArtifact {
                message: format!(
                    "unexpected trailing bytes in frozen bundle: {}",
                    self.bytes.len() - self.cursor
                ),
            })
        }
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], AotError> {
        if self.cursor + N > self.bytes.len() {
            return Err(AotError::InvalidArtifact {
                message: "unexpected end of frozen bundle".to_string(),
            });
        }

        let mut array = [0u8; N];
        array.copy_from_slice(&self.bytes[self.cursor..self.cursor + N]);
        self.cursor += N;
        Ok(array)
    }

    fn read_u8(&mut self) -> Result<u8, AotError> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_bool(&mut self) -> Result<bool, AotError> {
        match self.read_u8()? {
            0 => Ok(false),
            1 => Ok(true),
            value => Err(AotError::InvalidArtifact {
                message: format!("invalid boolean tag in frozen bundle: {}", value),
            }),
        }
    }

    fn read_u16(&mut self) -> Result<u16, AotError> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32, AotError> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    fn read_u64(&mut self) -> Result<u64, AotError> {
        Ok(u64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_i64(&mut self) -> Result<i64, AotError> {
        Ok(i64::from_le_bytes(self.read_array::<8>()?))
    }

    fn read_len(&mut self) -> Result<usize, AotError> {
        Ok(self.read_u32()? as usize)
    }

    fn read_string(&mut self) -> Result<String, AotError> {
        let len = self.read_len()?;
        if self.cursor + len > self.bytes.len() {
            return Err(AotError::InvalidArtifact {
                message: "unexpected end of frozen bundle while reading string".to_string(),
            });
        }

        let bytes = &self.bytes[self.cursor..self.cursor + len];
        self.cursor += len;
        String::from_utf8(bytes.to_vec()).map_err(|err| AotError::InvalidArtifact {
            message: format!("invalid UTF-8 in frozen bundle: {}", err),
        })
    }

    fn read_optional_string(&mut self) -> Result<Option<String>, AotError> {
        if self.read_bool()? {
            Ok(Some(self.read_string()?))
        } else {
            Ok(None)
        }
    }

    fn read_vec<T, F>(&mut self, mut read_item: F) -> Result<Vec<T>, AotError>
    where
        F: FnMut(&mut Self) -> Result<T, AotError>,
    {
        let len = self.read_len()?;
        let mut values = Vec::with_capacity(len);
        for _ in 0..len {
            values.push(read_item(self)?);
        }
        Ok(values)
    }

    fn read_module(&mut self) -> Result<FrozenModuleImage, AotError> {
        let name = self.read_string()?;
        let kind = module_kind_from_tag(self.read_u8()?)?;
        let package_name = self.read_string()?;
        let is_package = self.read_bool()?;
        let source_path = self.read_optional_string()?;
        let code = if self.read_bool()? {
            Some(self.read_code()?)
        } else {
            None
        };

        Ok(FrozenModuleImage {
            name,
            kind,
            package_name,
            is_package,
            source_path,
            code,
        })
    }

    fn read_code(&mut self) -> Result<CodeImage, AotError> {
        let name = self.read_string()?;
        let qualname = self.read_string()?;
        let filename = self.read_string()?;
        let first_lineno = self.read_u32()?;

        let instructions = self.read_vec(|reader| reader.read_u32())?;
        let constants = self.read_vec(|reader| reader.read_constant())?;
        let locals = self.read_vec(|reader| reader.read_string())?;
        let names = self.read_vec(|reader| reader.read_string())?;
        let freevars = self.read_vec(|reader| reader.read_string())?;
        let cellvars = self.read_vec(|reader| reader.read_string())?;

        let arg_count = self.read_u16()?;
        let posonlyarg_count = self.read_u16()?;
        let kwonlyarg_count = self.read_u16()?;
        let register_count = self.read_u16()?;
        let flags = self.read_u32()?;

        let line_table = self.read_vec(|reader| {
            Ok(LineTableImage {
                start_pc: reader.read_u32()?,
                end_pc: reader.read_u32()?,
                line: reader.read_u32()?,
            })
        })?;

        let exception_table = self.read_vec(|reader| {
            Ok(ExceptionTableImage {
                start_pc: reader.read_u32()?,
                end_pc: reader.read_u32()?,
                handler_pc: reader.read_u32()?,
                finally_pc: reader.read_u32()?,
                depth: reader.read_u16()?,
                exception_type_idx: reader.read_u16()?,
            })
        })?;

        let nested_code_objects = self.read_vec(|reader| reader.read_code())?;

        Ok(CodeImage {
            name,
            qualname,
            filename,
            first_lineno,
            instructions,
            constants,
            locals,
            names,
            freevars,
            cellvars,
            arg_count,
            posonlyarg_count,
            kwonlyarg_count,
            register_count,
            flags,
            line_table,
            exception_table,
            nested_code_objects,
        })
    }

    fn read_constant(&mut self) -> Result<ConstantImage, AotError> {
        match self.read_u8()? {
            CONSTANT_NONE => Ok(ConstantImage::None),
            CONSTANT_BOOL => Ok(ConstantImage::Bool(self.read_bool()?)),
            CONSTANT_INT => Ok(ConstantImage::Int(self.read_i64()?)),
            CONSTANT_FLOAT_BITS => Ok(ConstantImage::FloatBits(self.read_u64()?)),
            CONSTANT_STRING => Ok(ConstantImage::String(self.read_string()?)),
            CONSTANT_NESTED_CODE => Ok(ConstantImage::NestedCode(self.read_u32()?)),
            CONSTANT_KW_NAMES => {
                let names = self.read_vec(|reader| reader.read_string())?;
                Ok(ConstantImage::KwNamesTuple(names))
            }
            CONSTANT_BIGINT => Ok(ConstantImage::BigInt(self.read_string()?)),
            tag => Err(AotError::InvalidArtifact {
                message: format!("invalid constant tag in frozen bundle: {}", tag),
            }),
        }
    }
}

fn module_kind_tag(kind: ModuleKind) -> u8 {
    match kind {
        ModuleKind::Source => 0,
        ModuleKind::Stdlib => 1,
    }
}

fn module_kind_from_tag(tag: u8) -> Result<ModuleKind, AotError> {
    match tag {
        0 => Ok(ModuleKind::Source),
        1 => Ok(ModuleKind::Stdlib),
        _ => Err(AotError::InvalidArtifact {
            message: format!("invalid module kind tag in frozen bundle: {}", tag),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{BuildEntry, BuildOptions, BuildPlanner};
    use prism_compiler::OptimizationLevel;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestTempDir {
        path: PathBuf,
    }

    impl TestTempDir {
        fn new() -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time went backwards")
                .as_nanos();

            let mut path = std::env::temp_dir();
            path.push(format!(
                "prism_aot_bundle_tests_{}_{}_{}",
                std::process::id(),
                nanos,
                unique
            ));

            std::fs::create_dir_all(&path).expect("failed to create temp dir");
            Self { path }
        }
    }

    impl Drop for TestTempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    fn write_file(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("failed to create parent dir");
        }
        std::fs::write(path, content).expect("failed to write test file");
    }

    fn planner_for(path: &Path) -> BuildPlanner {
        BuildPlanner::new(BuildOptions {
            search_paths: vec![path.to_path_buf()],
            optimize: OptimizationLevel::Basic,
            target: "x86_64-windows".to_string(),
        })
    }

    #[test]
    fn test_bundle_roundtrip_preserves_nested_code_images() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(
            &main_path,
            "def outer(x):\n    def inner(y):\n        return x + y\n    return inner(4)\nVALUE = outer(3)\nTEXT = 'hello'\n",
        );

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Script(main_path))
            .expect("plan should succeed");
        let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle build should work");
        let decoded =
            FrozenModuleBundle::from_bytes(&bundle.to_bytes().expect("bundle bytes should exist"))
                .expect("bundle should round-trip");

        assert_eq!(bundle, decoded);
        let main_module = decoded
            .modules
            .iter()
            .find(|module| module.name == "__main__")
            .expect("main module should exist");
        let code = main_module
            .code
            .as_ref()
            .expect("source module should have code");
        assert!(
            code.constants
                .iter()
                .any(|constant| matches!(constant, ConstantImage::NestedCode(_)))
        );
        assert!(
            code.constants.iter().any(
                |constant| matches!(constant, ConstantImage::String(value) if value == "hello")
            )
        );
    }

    #[test]
    fn test_bundle_serializes_keyword_name_tuples() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(
            &main_path,
            "def build(x, y):\n    return x + y\nRESULT = build(x=1, y=2)\n",
        );

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Script(main_path))
            .expect("plan should succeed");
        let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle build should work");
        let main_module = bundle
            .modules
            .iter()
            .find(|module| module.name == "__main__")
            .expect("main module should exist");
        let code = main_module
            .code
            .as_ref()
            .expect("source module should have code");

        assert!(code.constants.iter().any(|constant| {
            matches!(
                constant,
                ConstantImage::KwNamesTuple(names) if names == &vec!["x".to_string(), "y".to_string()]
            )
        }));
    }

    #[test]
    fn test_bundle_write_and_read_path_roundtrip() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(&main_path, "VALUE = 42\n");

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Script(main_path))
            .expect("plan should succeed");
        let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle build should work");
        let output_path = temp.path.join("out").join("frozen-modules.prism");
        bundle
            .write_to_path(&output_path)
            .expect("bundle should write");

        let decoded =
            FrozenModuleBundle::read_from_path(&output_path).expect("bundle should read back");
        assert_eq!(bundle, decoded);
    }

    #[test]
    fn test_bundle_roundtrip_preserves_bigint_constants() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        let literal = "12345678901234567890123456789012345678901234567890";
        write_file(&main_path, &format!("VALUE = {literal}\n"));

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Script(main_path))
            .expect("plan should succeed");
        let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle build should work");
        let decoded =
            FrozenModuleBundle::from_bytes(&bundle.to_bytes().expect("bundle bytes should exist"))
                .expect("bundle should round-trip");

        let main_module = decoded
            .modules
            .iter()
            .find(|module| module.name == "__main__")
            .expect("main module should exist");
        let code = main_module
            .code
            .as_ref()
            .expect("source module should have code");

        assert!(code.constants.iter().any(|constant| {
            matches!(constant, ConstantImage::BigInt(value) if value == literal)
        }));
    }
}
