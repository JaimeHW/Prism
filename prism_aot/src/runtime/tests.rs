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
