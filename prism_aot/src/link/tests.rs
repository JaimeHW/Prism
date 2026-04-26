use super::*;
use crate::native::native_init_symbol;
use crate::planner::{BuildEntry, BuildOptions, BuildPlanner};
use object::{Object as _, ObjectSection as _, ObjectSymbol as _, RelocationTarget, SymbolIndex};
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
            "prism_aot_link_tests_{}_{}_{}",
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

fn symbol_names(file: &object::File<'_>) -> Vec<(SymbolIndex, String)> {
    file.symbols()
        .filter_map(|symbol| {
            symbol
                .name()
                .ok()
                .map(|name| (symbol.index(), name.to_string()))
        })
        .collect()
}

#[test]
fn test_linkable_bundle_artifact_emits_windows_coff_with_bundle_symbols() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "VALUE = 42\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");
    let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle should build");
    let expected_payload = bundle.to_bytes().expect("bundle bytes should exist");
    let artifact =
        LinkableBundleArtifact::from_bundle(&bundle).expect("artifact emission should work");

    assert_eq!(artifact.format, LinkArtifactFormat::Coff);

    let file = object::File::parse(artifact.bytes.as_slice()).expect("COFF parse should work");
    assert_eq!(file.format(), BinaryFormat::Coff);

    let section = file
        .sections()
        .find(|section| section.name().ok() == Some(".rdata$prism$bundle"))
        .expect("bundle section should exist");
    assert_eq!(
        section.data().expect("section data should be readable"),
        expected_payload.as_slice()
    );

    let symbol_names = file
        .symbols()
        .filter_map(|symbol| symbol.name().ok().map(str::to_string))
        .collect::<Vec<_>>();
    assert!(
        symbol_names
            .iter()
            .any(|name| name == FROZEN_BUNDLE_START_SYMBOL)
    );
    assert!(
        symbol_names
            .iter()
            .any(|name| name == FROZEN_BUNDLE_END_SYMBOL)
    );
}

#[test]
fn test_linkable_bundle_artifact_from_build_plan_emits_native_init_stubs() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(
        &main_path,
        "import helper\nfrom helper import VALUE\nRESULT = VALUE + 1\n",
    );
    write_file(&temp.path.join("helper.py"), "VALUE = 41\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");
    let artifact = LinkableBundleArtifact::from_build_plan(&plan)
        .expect("plan-aware artifact emission should work");

    let file = object::File::parse(artifact.bytes.as_slice()).expect("COFF parse should work");
    let names = symbol_names(&file);
    assert!(
        names
            .iter()
            .any(|(_, name)| name == AOT_NATIVE_INIT_TABLE_START_SYMBOL)
    );
    assert!(
        names
            .iter()
            .any(|(_, name)| name == AOT_NATIVE_INIT_TABLE_END_SYMBOL)
    );
    assert!(
        names
            .iter()
            .any(|(_, name)| name == &native_init_symbol("__main__"))
    );
    assert!(
        names
            .iter()
            .any(|(_, name)| name == &native_init_symbol("helper"))
    );

    let text = file
        .sections()
        .find(|section| section.name().ok() == Some(".text$prism$native"))
        .expect("native text section should exist");
    let relocation_targets = text
        .relocations()
        .filter_map(|(_, relocation)| match relocation.target() {
            RelocationTarget::Symbol(index) => names
                .iter()
                .find(|(candidate, _)| *candidate == index)
                .map(|(_, name)| name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert!(
        relocation_targets
            .iter()
            .any(|target| target == AOT_IMPORT_MODULE_SYMBOL)
    );
    assert!(
        relocation_targets
            .iter()
            .any(|target| target == AOT_IMPORT_FROM_SYMBOL)
    );
    assert!(
        relocation_targets
            .iter()
            .any(|target| target == AOT_STORE_EXPR_SYMBOL)
    );
}

#[test]
fn test_linkable_bundle_artifact_writes_to_disk() {
    let temp = TestTempDir::new();
    let main_path = temp.path.join("main.py");
    write_file(&main_path, "VALUE = 42\n");

    let plan = planner_for(&temp.path)
        .plan(BuildEntry::Script(main_path))
        .expect("plan should succeed");
    let bundle = FrozenModuleBundle::from_build_plan(&plan).expect("bundle should build");
    let artifact =
        LinkableBundleArtifact::from_bundle(&bundle).expect("artifact emission should work");
    let output_path = temp.path.join("out").join("frozen-modules.obj");
    artifact
        .write_to_path(&output_path)
        .expect("artifact write should work");

    let written = std::fs::read(&output_path).expect("object file should exist");
    let file = object::File::parse(written.as_slice()).expect("written object should parse");
    assert_eq!(file.format(), BinaryFormat::Coff);
}

#[test]
fn test_linkable_bundle_artifact_rejects_unsupported_targets() {
    let bundle = FrozenModuleBundle {
        format_version: 1,
        target: "x86_64-linux".to_string(),
        entry: crate::bundle::FrozenEntryImage {
            canonical_module: "__main__".to_string(),
            execution_name: "__main__".to_string(),
            package_name: String::new(),
        },
        modules: Vec::new(),
    };

    let err =
        LinkableBundleArtifact::from_bundle(&bundle).expect_err("unsupported target should fail");
    match err {
        AotError::UnsupportedTarget { target, .. } => assert_eq!(target, "x86_64-linux"),
        other => panic!("unexpected error: {other}"),
    }
}
