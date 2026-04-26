use super::*;
use object::{Object as _, ObjectSymbol as _};
use std::path::Path;
use std::sync::Arc;
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
            "prismc_tests_{}_{}_{}",
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

fn execute_frozen_bundle(
    bundle: &FrozenModuleBundle,
) -> Result<Arc<prism_vm::imports::ModuleObject>, String> {
    let runtime = bundle
        .decode_runtime_bundle()
        .map_err(|err| err.to_string())?;
    let entry = runtime.entry_module().map_err(|err| err.to_string())?;

    let mut vm = prism_vm::VirtualMachine::new();
    for module in &runtime.modules {
        vm.insert_frozen_module(
            module.name.as_ref(),
            prism_vm::imports::FrozenModuleSource::new(
                Arc::clone(&module.code),
                Arc::clone(&module.filename),
                Arc::clone(&module.package_name),
                module.is_package,
            ),
        );
    }

    let main_module = Arc::new(prism_vm::imports::ModuleObject::with_metadata(
        Arc::clone(&runtime.entry.execution_name),
        None,
        Some(Arc::clone(&entry.filename)),
        Some(Arc::clone(&runtime.entry.package_name)),
    ));
    vm.execute_in_module(Arc::clone(&entry.code), Arc::clone(&main_module))
        .map_err(|err| err.to_string())?;
    Ok(main_module)
}

#[test]
fn test_parse_script_build_args() {
    let parsed = parse_args(&[
        "build".to_string(),
        "app.py".to_string(),
        "-o".to_string(),
        "out.json".to_string(),
        "-O".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.entry, BuildEntry::Script(PathBuf::from("app.py")));
    assert_eq!(parsed.output, PathBuf::from("out.json"));
    assert_eq!(parsed.bundle_output, PathBuf::from("frozen-modules.prism"));
    assert_eq!(
        parsed.object_output,
        Some(PathBuf::from("frozen-modules.obj"))
    );
    assert_eq!(parsed.optimize, OptimizationLevel::Basic);
    assert_eq!(parsed.target, default_target());
}

#[test]
fn test_parse_module_build_args() {
    let parsed = parse_args(&[
        "build".to_string(),
        "--module".to_string(),
        "pkg.tool".to_string(),
        "-I".to_string(),
        "vendor".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.entry, BuildEntry::Module("pkg.tool".to_string()));
    assert_eq!(parsed.search_paths, vec![PathBuf::from("vendor")]);
    assert_eq!(
        parsed.output,
        PathBuf::from("prism-build").join("build-plan.json")
    );
    assert_eq!(
        parsed.bundle_output,
        PathBuf::from("prism-build").join("frozen-modules.prism")
    );
    assert_eq!(
        parsed.object_output,
        Some(PathBuf::from("prism-build").join("frozen-modules.obj"))
    );
    assert_eq!(parsed.target, default_target());
}

#[test]
fn test_parse_rejects_unknown_option() {
    let err = parse_args(&["build".to_string(), "--wat".to_string()])
        .expect_err("unknown option should fail");
    assert!(err.contains("unknown option"));
}

#[test]
fn test_parse_bundle_output_override() {
    let parsed = parse_args(&[
        "build".to_string(),
        "app.py".to_string(),
        "--emit-bundle".to_string(),
        "bundle.prism".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.bundle_output, PathBuf::from("bundle.prism"));
}

#[test]
fn test_parse_object_output_override() {
    let parsed = parse_args(&[
        "build".to_string(),
        "app.py".to_string(),
        "--emit-object".to_string(),
        "bundle.obj".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.object_output, Some(PathBuf::from("bundle.obj")));
}

#[test]
fn test_parse_target_override() {
    let parsed = parse_args(&[
        "build".to_string(),
        "app.py".to_string(),
        "--target".to_string(),
        "x86_64-pc-windows-msvc".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.target, "x86_64-pc-windows-msvc");
    assert_eq!(
        parsed.object_output,
        Some(PathBuf::from("prism-build").join("frozen-modules.obj"))
    );
}

#[test]
fn test_non_windows_target_skips_default_object_output() {
    let parsed = parse_args(&[
        "build".to_string(),
        "app.py".to_string(),
        "--target".to_string(),
        "x86_64-unknown-linux-gnu".to_string(),
    ])
    .expect("parse should succeed");

    assert_eq!(parsed.target, "x86_64-unknown-linux-gnu");
    assert_eq!(parsed.object_output, None);
}

#[test]
fn test_execute_build_writes_manifest_bundle_and_object() {
    let temp = TestTempDir::new();
    let entry = temp.path.join("app.py");
    let manifest_output = temp.path.join("out").join("build-plan.json");
    let bundle_output = temp.path.join("out").join("frozen-modules.prism");
    let object_output = temp.path.join("out").join("frozen-modules.obj");
    write_file(&entry, "import helper\n");
    write_file(&temp.path.join("helper.py"), "VALUE = 1\n");

    let args = CompilerArgs {
        entry: BuildEntry::Script(entry),
        output: manifest_output.clone(),
        bundle_output: bundle_output.clone(),
        object_output: Some(object_output.clone()),
        optimize: OptimizationLevel::Basic,
        target: "x86_64-pc-windows-msvc".to_string(),
        search_paths: vec![temp.path.clone()],
    };

    let outputs = execute_build(&args).expect("build execution should succeed");

    assert_eq!(outputs.planned_modules, 2);
    assert!(manifest_output.is_file());
    assert!(bundle_output.is_file());
    assert!(object_output.is_file());
    assert!(
        std::fs::metadata(&object_output)
            .expect("object metadata should exist")
            .len()
            > 0
    );

    let manifest_json =
        std::fs::read_to_string(&manifest_output).expect("manifest output should exist");
    let manifest: serde_json::Value =
        serde_json::from_str(&manifest_json).expect("manifest should be valid JSON");
    let entry_module = manifest["modules"]
        .as_array()
        .and_then(|modules| {
            modules
                .iter()
                .find(|module| module["name"].as_str() == Some("__main__"))
        })
        .expect("manifest should contain the entry module");
    let expected_symbol = prism_aot::native_init_symbol("__main__");
    assert_eq!(
        entry_module["compilationMode"].as_str(),
        Some("frozen-bytecode-plus-native-init")
    );
    assert_eq!(entry_module["nativeInitSupported"], true);
    assert_eq!(
        entry_module["nativeInitSymbol"].as_str(),
        Some(expected_symbol.as_str())
    );

    let object_bytes = std::fs::read(&object_output).expect("object output should exist");
    let object_file =
        object::File::parse(object_bytes.as_slice()).expect("object output should parse");
    let symbol_names = object_file
        .symbols()
        .filter_map(|symbol| symbol.name().ok().map(str::to_string))
        .collect::<Vec<_>>();
    assert!(symbol_names.iter().any(|name| name == &expected_symbol));
}

#[test]
fn test_execute_built_bundle_runs_package_entrypoint() {
    let temp = TestTempDir::new();
    let manifest_output = temp.path.join("out").join("build-plan.json");
    let bundle_output = temp.path.join("out").join("frozen-modules.prism");
    write_file(&temp.path.join("pkg").join("__init__.py"), "");
    write_file(
        &temp.path.join("pkg").join("__main__.py"),
        "from .helper import VALUE\nRESULT = VALUE + 5\n",
    );
    write_file(&temp.path.join("pkg").join("helper.py"), "VALUE = 37\n");

    let args = CompilerArgs {
        entry: BuildEntry::Module("pkg".to_string()),
        output: manifest_output,
        bundle_output: bundle_output.clone(),
        object_output: None,
        optimize: OptimizationLevel::Basic,
        target: "x86_64-pc-windows-msvc".to_string(),
        search_paths: vec![temp.path.clone()],
    };

    execute_build(&args).expect("build execution should succeed");
    let bundle = FrozenModuleBundle::read_from_path(&bundle_output)
        .expect("frozen bundle should round-trip from disk");
    let main_module = execute_frozen_bundle(&bundle).expect("bundle should execute");

    assert_eq!(
        main_module
            .get_attr("RESULT")
            .and_then(|value| value.as_int()),
        Some(42)
    );
}

#[test]
fn test_build_options_extend_default_search_paths() {
    let args = CompilerArgs {
        entry: BuildEntry::Module("pkg.tool".to_string()),
        output: PathBuf::from("out.json"),
        bundle_output: PathBuf::from("bundle.prism"),
        object_output: Some(PathBuf::from("bundle.obj")),
        optimize: OptimizationLevel::Full,
        target: "x86_64-pc-windows-msvc".to_string(),
        search_paths: vec![PathBuf::from("vendor")],
    };

    let options = build_options(&args);

    assert_eq!(options.optimize, OptimizationLevel::Full);
    assert_eq!(options.target, "x86_64-pc-windows-msvc");
    assert!(
        options
            .search_paths
            .iter()
            .any(|path| path == &PathBuf::from("vendor"))
    );
    assert!(
        options
            .search_paths
            .iter()
            .any(|path| path == &std::env::current_dir().expect("cwd should resolve"))
    );
}
