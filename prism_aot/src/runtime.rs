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
                "prism_aot_runtime_tests_{}_{}_{}",
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
            target: "x86_64-pc-windows-msvc".to_string(),
        })
    }

    #[test]
    fn test_code_image_roundtrip_restores_nested_code_and_strings() {
        let temp = TestTempDir::new();
        let main_path = temp.path.join("main.py");
        write_file(
            &main_path,
            "def outer(x):\n    def inner(y):\n        return x + y\n    return inner(5)\nTEXT = 'hello'\nVALUE = outer(7)\n",
        );

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Script(main_path))
            .expect("plan should succeed");
        let image = plan
            .modules
            .iter()
            .find(|module| module.name == "__main__")
            .and_then(|module| module.code_image.as_ref())
            .expect("main module image should exist");
        let code = image.to_code_object().expect("code image should decode");

        assert_eq!(code.qualname.as_ref(), "<module>");
        assert_eq!(code.filename.as_ref(), image.filename.as_str());
        assert_eq!(
            code.nested_code_objects.len(),
            image.nested_code_objects.len()
        );
        assert!(
            code.constants
                .iter()
                .any(|constant| matches!(constant, Constant::Value(value) if value.is_string()))
        );
        assert!(
            code.constants
                .iter()
                .any(|constant| matches!(constant, Constant::Value(value) if value.is_object()))
        );
    }

    #[test]
    fn test_decode_runtime_bundle_preserves_entry_metadata_and_modules() {
        let temp = TestTempDir::new();
        write_file(&temp.path.join("pkg").join("__init__.py"), "");
        write_file(
            &temp.path.join("pkg").join("__main__.py"),
            "from .helper import VALUE\nRESULT = VALUE\n",
        );
        write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 42\n");

        let plan = planner_for(&temp.path)
            .plan(BuildEntry::Module("pkg".to_string()))
            .expect("package entry plan should succeed");
        let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle should build");
        let runtime = bundle
            .decode_runtime_bundle()
            .expect("runtime bundle should decode");

        assert_eq!(runtime.entry.canonical_module.as_ref(), "pkg.__main__");
        assert_eq!(runtime.entry.execution_name.as_ref(), "__main__");
        assert_eq!(runtime.entry.package_name.as_ref(), "pkg");
        assert!(runtime.module("pkg").is_some());
        assert!(runtime.module("pkg.helper").is_some());
        assert_eq!(
            runtime
                .entry_module()
                .expect("entry module should exist")
                .name
                .as_ref(),
            "pkg.__main__"
        );
    }

    #[test]
    fn test_decode_runtime_bundle_rejects_invalid_nested_code_index() {
        let image = CodeImage {
            name: "mod".to_string(),
            qualname: "<module>".to_string(),
            filename: "<frozen>".to_string(),
            first_lineno: 1,
            instructions: Vec::new(),
            constants: vec![ConstantImage::NestedCode(1)],
            locals: Vec::new(),
            names: Vec::new(),
            freevars: Vec::new(),
            cellvars: Vec::new(),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 0,
            flags: CodeFlags::MODULE.bits(),
            line_table: Vec::new(),
            exception_table: Vec::new(),
            nested_code_objects: Vec::new(),
        };

        let err = image
            .to_code_object()
            .expect_err("invalid nested code reference should fail");
        assert!(err.to_string().contains("nested code"));
    }

    #[test]
    fn test_code_image_decodes_bigint_constants() {
        let image = CodeImage {
            name: "mod".to_string(),
            qualname: "<module>".to_string(),
            filename: "<frozen>".to_string(),
            first_lineno: 1,
            instructions: Vec::new(),
            constants: vec![ConstantImage::BigInt(
                "12345678901234567890123456789012345678901234567890".to_string(),
            )],
            locals: Vec::new(),
            names: Vec::new(),
            freevars: Vec::new(),
            cellvars: Vec::new(),
            arg_count: 0,
            posonlyarg_count: 0,
            kwonlyarg_count: 0,
            register_count: 0,
            flags: CodeFlags::MODULE.bits(),
            line_table: Vec::new(),
            exception_table: Vec::new(),
            nested_code_objects: Vec::new(),
        };

        let code = image
            .to_code_object()
            .expect("bigint constants should decode");
        assert!(code.constants.iter().any(|constant| {
            matches!(constant, Constant::BigInt(value) if value == &BigInt::parse_bytes(b"12345678901234567890123456789012345678901234567890", 10).expect("valid bigint"))
        }));
    }
}
