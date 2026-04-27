use std::sync::Arc;

use num_bigint::BigInt;
use prism_code::{
    CodeFlags, CodeObject, Constant, ExceptionEntry, Instruction, KwNamesTuple, LineTableEntry,
};
use prism_core::Value;
use prism_core::intern::intern;

use crate::bundle::{CodeImage, ConstantImage, FrozenModuleBundle};
use crate::error::AotError;

/// Runtime-ready entrypoint metadata decoded from a frozen bundle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFrozenEntry {
    /// Canonical module name in the build graph.
    pub canonical_module: Arc<str>,
    /// Runtime execution name for the entrypoint.
    pub execution_name: Arc<str>,
    /// Package context used for relative imports.
    pub package_name: Arc<str>,
}

/// Runtime-ready source module decoded from a frozen bundle.
#[derive(Debug, Clone)]
pub struct RuntimeFrozenModule {
    /// Canonical module name.
    pub name: Arc<str>,
    /// Compiled Prism bytecode for the module.
    pub code: Arc<CodeObject>,
    /// Source filename recorded for diagnostics and metadata.
    pub filename: Arc<str>,
    /// Package context used for relative imports.
    pub package_name: Arc<str>,
    /// Whether the module originated from a package `__init__.py`.
    pub is_package: bool,
}

/// Runtime view of a frozen bundle suitable for VM installation.
#[derive(Debug, Clone)]
pub struct RuntimeFrozenBundle {
    /// Entrypoint metadata.
    pub entry: RuntimeFrozenEntry,
    /// Runtime-loadable source modules.
    pub modules: Vec<RuntimeFrozenModule>,
}

impl RuntimeFrozenBundle {
    /// Look up a decoded module by canonical name.
    pub fn module(&self, name: &str) -> Option<&RuntimeFrozenModule> {
        self.modules
            .iter()
            .find(|module| module.name.as_ref() == name)
    }

    /// Resolve the decoded entry module.
    pub fn entry_module(&self) -> Result<&RuntimeFrozenModule, AotError> {
        self.module(self.entry.canonical_module.as_ref())
            .ok_or_else(|| AotError::InvalidArtifact {
                message: format!(
                    "frozen bundle entry module '{}' is missing from decoded modules",
                    self.entry.canonical_module
                ),
            })
    }
}

impl FrozenModuleBundle {
    /// Decode source-backed modules into runtime code objects.
    pub fn decode_runtime_bundle(&self) -> Result<RuntimeFrozenBundle, AotError> {
        let modules = self
            .modules
            .iter()
            .filter_map(|module| module.code.as_ref().map(|code| (module, code)))
            .map(|(module, code)| {
                let code = Arc::new(code.to_code_object()?);
                let filename = module
                    .source_path
                    .as_deref()
                    .map(Arc::<str>::from)
                    .unwrap_or_else(|| Arc::clone(&code.filename));
                Ok(RuntimeFrozenModule {
                    name: Arc::from(module.name.as_str()),
                    code,
                    filename,
                    package_name: Arc::from(module.package_name.as_str()),
                    is_package: module.is_package,
                })
            })
            .collect::<Result<Vec<_>, AotError>>()?;

        Ok(RuntimeFrozenBundle {
            entry: RuntimeFrozenEntry {
                canonical_module: Arc::from(self.entry.canonical_module.as_str()),
                execution_name: Arc::from(self.entry.execution_name.as_str()),
                package_name: Arc::from(self.entry.package_name.as_str()),
            },
            modules,
        })
    }
}

impl CodeImage {
    /// Decode a serialized code image back into a Prism code object.
    pub fn to_code_object(&self) -> Result<CodeObject, AotError> {
        let nested_code_objects = self
            .nested_code_objects
            .iter()
            .map(CodeImage::to_code_object)
            .map(|result| result.map(Arc::new))
            .collect::<Result<Vec<_>, AotError>>()?;

        let constants = self
            .constants
            .iter()
            .enumerate()
            .map(|(index, constant)| {
                constant.to_constant(&nested_code_objects).map_err(|err| {
                    AotError::InvalidArtifact {
                        message: format!(
                            "failed to decode constant {} for '{}': {}",
                            index, self.qualname, err
                        ),
                    }
                })
            })
            .collect::<Result<Vec<_>, AotError>>()?;

        let flags = CodeFlags::from_bits(self.flags).ok_or_else(|| AotError::InvalidArtifact {
            message: format!(
                "invalid code flags {:#010x} in '{}'",
                self.flags, self.qualname
            ),
        })?;

        Ok(CodeObject {
            name: Arc::from(self.name.as_str()),
            qualname: Arc::from(self.qualname.as_str()),
            filename: Arc::from(self.filename.as_str()),
            first_lineno: self.first_lineno,
            instructions: self
                .instructions
                .iter()
                .copied()
                .map(Instruction::from_raw)
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            constants: constants.into_boxed_slice(),
            locals: arc_str_box(&self.locals),
            names: arc_str_box(&self.names),
            freevars: arc_str_box(&self.freevars),
            cellvars: arc_str_box(&self.cellvars),
            arg_count: self.arg_count,
            posonlyarg_count: self.posonlyarg_count,
            kwonlyarg_count: self.kwonlyarg_count,
            register_count: self.register_count,
            flags,
            line_table: self
                .line_table
                .iter()
                .map(|entry| LineTableEntry {
                    start_pc: entry.start_pc,
                    end_pc: entry.end_pc,
                    line: entry.line,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            exception_table: self
                .exception_table
                .iter()
                .map(|entry| ExceptionEntry {
                    start_pc: entry.start_pc,
                    end_pc: entry.end_pc,
                    handler_pc: entry.handler_pc,
                    finally_pc: entry.finally_pc,
                    depth: entry.depth,
                    exception_type_idx: entry.exception_type_idx,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice(),
            nested_code_objects: nested_code_objects.into_boxed_slice(),
        })
    }
}

impl ConstantImage {
    fn to_constant(&self, nested_code_objects: &[Arc<CodeObject>]) -> Result<Constant, AotError> {
        match self {
            ConstantImage::None => Ok(Constant::Value(Value::none())),
            ConstantImage::Bool(value) => Ok(Constant::Value(Value::bool(*value))),
            ConstantImage::Int(value) => Ok(match Value::int(*value) {
                Some(value) => Constant::Value(value),
                None => Constant::BigInt(BigInt::from(*value)),
            }),
            ConstantImage::FloatBits(bits) => {
                Ok(Constant::Value(Value::float(f64::from_bits(*bits))))
            }
            ConstantImage::String(value) => Ok(Constant::Value(Value::string(intern(value)))),
            ConstantImage::NestedCode(index) => {
                let code = nested_code_objects.get(*index as usize).ok_or_else(|| {
                    AotError::InvalidArtifact {
                        message: format!(
                            "nested code constant references missing nested code object {}",
                            index
                        ),
                    }
                })?;
                Ok(Constant::Value(Value::object_ptr(
                    Arc::as_ptr(code) as *const ()
                )))
            }
            ConstantImage::KwNamesTuple(names) => {
                let tuple = KwNamesTuple::new(
                    names
                        .iter()
                        .map(|name| Arc::<str>::from(name.as_str()))
                        .collect(),
                );
                let tuple_ptr = Box::into_raw(Box::new(tuple)) as *const ();
                Ok(Constant::Value(Value::object_ptr(tuple_ptr)))
            }
            ConstantImage::BigInt(value) => {
                let parsed = BigInt::parse_bytes(value.as_bytes(), 10).ok_or_else(|| {
                    AotError::InvalidArtifact {
                        message: format!("invalid bigint constant '{}'", value),
                    }
                })?;
                Ok(Constant::BigInt(parsed))
            }
        }
    }
}

fn arc_str_box(values: &[String]) -> Box<[Arc<str>]> {
    values
        .iter()
        .map(|value| Arc::<str>::from(value.as_str()))
        .collect::<Vec<_>>()
        .into_boxed_slice()
}
