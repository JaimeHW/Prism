use super::*;
use crate::manifest::BuildManifest;
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
            "prism_aot_tests_{}_{}_{}",
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
fn test_plan_script_collects_source_and_stdlib_dependencies() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "import helper\nimport math\n");
    write_file(&temp.path.join("helper.py"), "VALUE = 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path.clone()))
        .expect("plan should succeed");

    assert_eq!(plan.entry.canonical_module, "__main__");
    assert_eq!(plan.modules.len(), 3);
    assert!(plan.modules.iter().any(|module| module.name == "__main__"));
    assert!(plan.modules.iter().any(|module| module.name == "helper"));
    assert!(
        plan.modules
            .iter()
            .any(|module| { module.name == "math" && module.kind == ModuleKind::Stdlib })
    );
}

#[test]
fn test_plan_script_uses_entry_parent_as_import_root() {
    let temp = TestTempDir::new();
    let app_dir = temp.path.join("app");
    let main_path = app_dir.join("main.py");
    write_file(&main_path, "import helper\n");
    write_file(&app_dir.join("helper.py"), "VALUE = 1\n");

    let planner = BuildPlanner::new(BuildOptions {
        search_paths: Vec::new(),
        optimize: OptimizationLevel::Basic,
        target: "x86_64-windows".to_string(),
    });

    let plan = planner
        .plan(BuildEntry::Script(main_path))
        .expect("script plan should succeed without explicit search paths");

    assert!(plan.modules.iter().any(|module| module.name == "__main__"));
    assert!(plan.modules.iter().any(|module| module.name == "helper"));
}

#[test]
fn test_plan_module_collects_relative_package_dependencies() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "import math\n");
    write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 7\n");
    write_file(
        &temp.path.join("pkg").join("main.py"),
        "from .helper import VALUE\nassert VALUE == 7\n",
    );

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Module("pkg.main".to_string()))
        .expect("module plan should succeed");

    assert_eq!(plan.entry.canonical_module, "pkg.main");
    assert!(plan.modules.iter().any(|module| module.name == "pkg"));
    assert!(
        plan.modules
            .iter()
            .any(|module| module.name == "pkg.helper")
    );
    assert!(plan.modules.iter().any(|module| module.name == "pkg.main"));
    assert!(plan.modules.iter().any(|module| module.name == "math"));
}

#[test]
fn test_plan_package_entry_uses_dunder_main_module() {
    let temp = TestTempDir::new();
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(
        &temp.path.join("pkg").join("__main__.py"),
        "from . import helper\n",
    );
    write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Module("pkg".to_string()))
        .expect("package entry plan should succeed");

    assert_eq!(plan.entry.canonical_module, "pkg.__main__");
    assert_eq!(plan.entry.execution_name, "__main__");
    assert_eq!(plan.entry.package_name, "pkg");
    assert!(
        plan.modules
            .iter()
            .any(|module| module.name == "pkg.__main__")
    );
    assert!(
        plan.modules
            .iter()
            .any(|module| module.name == "pkg.helper")
    );
}

#[test]
fn test_plan_from_import_attribute_does_not_require_fake_submodule() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "from helper import VALUE\n");
    write_file(&temp.path.join("helper.py"), "VALUE = 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");

    assert!(plan.modules.iter().any(|module| module.name == "helper"));
    assert!(
        !plan
            .modules
            .iter()
            .any(|module| module.name == "helper.VALUE")
    );
}

#[test]
fn test_plan_from_import_includes_stdlib_submodule_candidate() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "from os import path\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");

    assert!(plan.modules.iter().any(|module| module.name == "os"));
    assert!(plan.modules.iter().any(|module| module.name == "os.path"));
}

#[test]
fn test_plan_from_import_includes_source_submodule_candidate() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "from pkg import submodule\n");
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(&temp.path.join("pkg").join("submodule.py"), "VALUE = 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");

    assert!(plan.modules.iter().any(|module| module.name == "pkg"));
    assert!(
        plan.modules
            .iter()
            .any(|module| module.name == "pkg.submodule")
    );
}

#[test]
fn test_plan_reports_unresolved_module() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "import missing_dependency\n");

    let err = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect_err("missing module should fail planning");

    match err {
        AotError::ModuleNotFound { module, .. } => {
            assert_eq!(module, "missing_dependency");
        }
        other => panic!("unexpected error: {other}"),
    }
}

#[test]
fn test_manifest_writer_emits_deterministic_json() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "import helper\n");
    write_file(&temp.path.join("helper.py"), "VALUE = 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");
    let manifest = BuildManifest::from(&plan);
    let output_path = temp.path.join("out").join("build-plan.json");
    manifest
        .write_to_path(&output_path)
        .expect("manifest write should succeed");

    let json = std::fs::read_to_string(&output_path).expect("manifest should exist");
    let parsed: serde_json::Value =
        serde_json::from_str(&json).expect("manifest should be valid JSON");

    assert_eq!(parsed["formatVersion"], 1);
    assert_eq!(parsed["entry"]["canonicalModule"], "__main__");
    assert!(parsed["modules"].is_array());
    assert_eq!(
        parsed["modules"][0]["compilationMode"].as_str(),
        Some("frozen-bytecode-plus-native-init")
    );
}

#[test]
fn test_plan_tracks_native_init_support_diagnostics() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "def helper():\n    return 1\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should still succeed for non-lowerable source");
    let main = plan
        .modules
        .iter()
        .find(|module| module.name == "__main__")
        .expect("main module should exist");

    assert!(main.native_init.is_none());
    assert!(
        main.native_init_diagnostic
            .as_deref()
            .unwrap_or_default()
            .contains("cannot lower statement")
    );
}
