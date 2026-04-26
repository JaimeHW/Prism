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
        let artifact = LinkableBundleArtifact::from_build_plan(&plan)?;
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
#[path = "prismc/tests.rs"]
mod tests;
