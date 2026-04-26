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
        code.constants
            .iter()
            .any(|constant| matches!(constant, ConstantImage::String(value) if value == "hello"))
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

    assert!(
        code.constants.iter().any(|constant| {
            matches!(constant, ConstantImage::BigInt(value) if value == literal)
        })
    );
}
