//! Dedicated compiler driver for Prism's standalone AOT pipeline.

use prism_aot::{BuildEntry, BuildManifest, BuildOptions, BuildPlanner, FrozenModuleBundle};
use prism_compiler::OptimizationLevel;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CompilerArgs {
    entry: BuildEntry,
    output: PathBuf,
    bundle_output: PathBuf,
    optimize: OptimizationLevel,
    search_paths: Vec<PathBuf>,
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

    let planner = BuildPlanner::new(build_options(&parsed));

    let plan = match planner.plan(parsed.entry) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("prismc: {err}");
            return ExitCode::from(1);
        }
    };

    let manifest = BuildManifest::from(&plan);
    if let Err(err) = manifest.write_to_path(&parsed.output) {
        eprintln!("prismc: {err}");
        return ExitCode::from(1);
    }

    let bundle = match FrozenModuleBundle::from_build_plan(&plan) {
        Ok(bundle) => bundle,
        Err(err) => {
            eprintln!("prismc: {err}");
            return ExitCode::from(1);
        }
    };
    if let Err(err) = bundle.write_to_path(&parsed.bundle_output) {
        eprintln!("prismc: {err}");
        return ExitCode::from(1);
    }

    println!(
        "planned {} modules for target {} -> {} ({})",
        plan.modules.len(),
        plan.target,
        parsed.output.display(),
        parsed.bundle_output.display()
    );
    ExitCode::SUCCESS
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
    let mut optimize = OptimizationLevel::None;
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
            "-I" => {
                index += 1;
                let Some(value) = args.get(index) else {
                    return Err("expected path after -I".to_string());
                };
                search_paths.push(PathBuf::from(value));
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

    Ok(CompilerArgs {
        entry,
        output,
        bundle_output,
        optimize,
        search_paths,
    })
}

fn help_text() -> &'static str {
    "Usage: prismc build [-m|--module] <entry> [-o <file>] [--emit-bundle <file>] [-I <path>] [-O|-OO]"
}

fn build_options(args: &CompilerArgs) -> BuildOptions {
    let mut options = BuildOptions {
        optimize: args.optimize,
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

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(parsed.optimize, OptimizationLevel::Basic);
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
    fn test_build_options_extend_default_search_paths() {
        let args = CompilerArgs {
            entry: BuildEntry::Module("pkg.tool".to_string()),
            output: PathBuf::from("out.json"),
            bundle_output: PathBuf::from("bundle.prism"),
            optimize: OptimizationLevel::Full,
            search_paths: vec![PathBuf::from("vendor")],
        };

        let options = build_options(&args);

        assert_eq!(options.optimize, OptimizationLevel::Full);
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
