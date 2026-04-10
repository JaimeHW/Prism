//! Dedicated compiler driver for Prism's standalone AOT pipeline.

use prism_aot::{
    BuildEntry, BuildManifest, BuildOptions, BuildPlanner, FrozenModuleBundle,
    LinkableBundleArtifact,
};
use prism_compiler::OptimizationLevel;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompilerArgs {
    entry: BuildEntry,
    output: PathBuf,
    bundle_output: PathBuf,
    object_output: Option<PathBuf>,
    optimize: OptimizationLevel,
    target: String,
    search_paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BuildOutputs {
    planned_modules: usize,
    target: String,
    manifest_output: PathBuf,
    bundle_output: PathBuf,
    object_output: Option<PathBuf>,
}

fn main() -> ExitCode {
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let parsed = match parse_args(&raw_args) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("prismc: {message}");
            eprintln!("{}", help_text());
            return ExitCode::from(2);
        }
    };

    let outputs = match execute_build(&parsed) {
        Ok(outputs) => outputs,
        Err(err) => {
            eprintln!("prismc: {err}");
            return ExitCode::from(1);
        }
    };

    let artifact_suffix = outputs
        .object_output
        .as_ref()
        .map(|path| format!(", {}", path.display()))
        .unwrap_or_default();
    println!(
        "planned {} modules for target {} -> {}, {}{}",
        outputs.planned_modules,
        outputs.target,
        outputs.manifest_output.display(),
        outputs.bundle_output.display(),
        artifact_suffix
    );
    ExitCode::SUCCESS
}

fn execute_build(args: &CompilerArgs) -> Result<BuildOutputs, prism_aot::AotError> {
    let planner = BuildPlanner::new(build_options(args));
    let plan = planner.plan(args.entry.clone())?;

    let manifest = BuildManifest::from(&plan);
    manifest.write_to_path(&args.output)?;

    let bundle = FrozenModuleBundle::from_build_plan(&plan)?;
    bundle.write_to_path(&args.bundle_output)?;

    if let Some(object_output) = &args.object_output {
        let artifact = LinkableBundleArtifact::from_bundle(&bundle)?;
        artifact.write_to_path(object_output)?;
    }

    Ok(BuildOutputs {
        planned_modules: plan.modules.len(),
        target: plan.target,
        manifest_output: args.output.clone(),
        bundle_output: args.bundle_output.clone(),
        object_output: args.object_output.clone(),
    })
}

fn parse_args(args: &[String]) -> Result<CompilerArgs, String> {
    if args.is_empty() {
        return Err("missing command".to_string());
    }

    if args[0] != "build" {
        return Err(format!("unsupported command '{}'", args[0]));
    }

    let mut entry_value: Option<String> = None;
    let mut module_mode = false;
    let mut output: Option<PathBuf> = None;
    let mut bundle_output: Option<PathBuf> = None;
    let mut object_output: Option<PathBuf> = None;
    let mut optimize = OptimizationLevel::None;
    let mut target: Option<String> = None;
    let mut search_paths = Vec::new();

    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "-m" | "--module" => {
                module_mode = true;
                index += 1;
            }
            "-o" | "--output" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected path after --output".to_string());
                };
                output = Some(PathBuf::from(value));
                index += 1;
            }
            "--emit-bundle" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected path after --emit-bundle".to_string());
                };
                bundle_output = Some(PathBuf::from(value));
                index += 1;
            }
            "--emit-object" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected path after --emit-object".to_string());
                };
                object_output = Some(PathBuf::from(value));
                index += 1;
            }
            "-I" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected path after -I".to_string());
                };
                search_paths.push(PathBuf::from(value));
                index += 1;
            }
            "--target" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected value after --target".to_string());
                };
                target = Some(value.to_string());
                index += 1;
            }
            "-O" => {
                optimize = OptimizationLevel::Basic;
                index += 1;
            }
            "-OO" => {
                optimize = OptimizationLevel::Full;
                index += 1;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown option '{value}'"));
            }
            value => {
                if entry_value.is_some() {
                    return Err("only one entrypoint may be specified".to_string());
                }
                entry_value = Some(value.to_string());
                index += 1;
            }
        }
    }

    let entry_value = entry_value.ok_or_else(|| "missing build entrypoint".to_string())?;
    let entry = if module_mode {
        BuildEntry::Module(entry_value)
    } else {
        BuildEntry::Script(PathBuf::from(entry_value))
    };
    let output = output.unwrap_or_else(|| PathBuf::from("prism-build").join("build-plan.json"));
    let bundle_output = bundle_output.unwrap_or_else(|| default_bundle_output(&output));
    let target = target.unwrap_or_else(default_target);
    let object_output = object_output.or_else(|| default_object_output(&output, &target));

    Ok(CompilerArgs {
        entry,
        output,
        bundle_output,
        object_output,
        optimize,
        target,
        search_paths,
    })
}

fn help_text() -> &'static str {
    "Usage: prismc build [-m|--module] <entry> [-o <file>] [--emit-bundle <file>] [--emit-object <file>] [--target <triple>] [-I <path>] [-O|-OO]"
}

fn build_options(args: &CompilerArgs) -> BuildOptions {
    let mut options = BuildOptions {
        optimize: args.optimize,
        target: args.target.clone(),
        ..BuildOptions::default()
    };

    for path in &args.search_paths {
        if !options.search_paths.iter().any(|existing| existing == path) {
            options.search_paths.push(path.clone());
        }
    }

    options
}

fn default_bundle_output(manifest_output: &std::path::Path) -> PathBuf {
    if let Some(parent) = manifest_output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        parent.join("frozen-modules.prism")
    } else {
        PathBuf::from("frozen-modules.prism")
    }
}

fn default_object_output(manifest_output: &std::path::Path, target: &str) -> Option<PathBuf> {
    if !target.contains("windows") {
        return None;
    }

    let filename = "frozen-modules.obj";
    Some(
        manifest_output
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .map(|parent| parent.join(filename))
            .unwrap_or_else(|| PathBuf::from(filename)),
    )
}

fn default_target() -> String {
    BuildOptions::default().target
}

#[cfg(test)]
mod tests {
    use super::*;
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
    ) -> Result<Arc<prism_vm::import::ModuleObject>, String> {
        let runtime = bundle
            .decode_runtime_bundle()
            .map_err(|err| err.to_string())?;
        let entry = runtime.entry_module().map_err(|err| err.to_string())?;

        let mut vm = prism_vm::VirtualMachine::new();
        for module in &runtime.modules {
            vm.import_resolver.insert_frozen_module(
                module.name.as_ref(),
                prism_vm::import::FrozenModuleSource::new(
                    Arc::clone(&module.code),
                    Arc::clone(&module.filename),
                    Arc::clone(&module.package_name),
                    module.is_package,
                ),
            );
        }

        let main_module = Arc::new(prism_vm::import::ModuleObject::with_metadata(
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
}
