use std::path::Path;

use object::write::{Object, Symbol, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};

use crate::bundle::FrozenModuleBundle;
use crate::error::AotError;

/// Symbol exported at the beginning of the frozen module payload.
pub const FROZEN_BUNDLE_START_SYMBOL: &str = "prism_frozen_bundle_start";
/// Symbol exported immediately after the frozen module payload.
pub const FROZEN_BUNDLE_END_SYMBOL: &str = "prism_frozen_bundle_end";

const PRISM_BUNDLE_SECTION: &[u8] = b".rdata$prism$bundle";

/// Native object format emitted for downstream linking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinkArtifactFormat {
    /// Windows COFF object file.
    Coff,
}

/// Linkable native artifact containing Prism's frozen module image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkableBundleArtifact {
    /// Target identifier selected for the build.
    pub target: String,
    /// Native object format emitted for the target.
    pub format: LinkArtifactFormat,
    /// Serialized object-file bytes.
    pub bytes: Vec<u8>,
}

impl LinkableBundleArtifact {
    /// Emit a linkable object from a frozen module bundle.
    pub fn from_bundle(bundle: &FrozenModuleBundle) -> Result<Self, AotError> {
        let spec = target_spec(&bundle.target)?;
        let bytes = match spec.format {
            LinkArtifactFormat::Coff => emit_windows_coff_bundle(bundle, spec.architecture)?,
        };

        Ok(Self {
            target: bundle.target.clone(),
            format: spec.format,
            bytes,
        })
    }

    /// Write the native object artifact to disk.
    pub fn write_to_path(&self, path: &Path) -> Result<(), AotError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| AotError::Io {
                path: parent.to_path_buf(),
                message: err.to_string(),
            })?;
        }

        std::fs::write(path, &self.bytes).map_err(|err| AotError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TargetSpec {
    format: LinkArtifactFormat,
    architecture: Architecture,
}

fn target_spec(target: &str) -> Result<TargetSpec, AotError> {
    if !target.contains("windows") {
        return Err(AotError::UnsupportedTarget {
            target: target.to_string(),
            feature: "native object emission".to_string(),
        });
    }

    let architecture = if target.starts_with("x86_64-") {
        Architecture::X86_64
    } else if target.starts_with("aarch64-") {
        Architecture::Aarch64
    } else {
        return Err(AotError::UnsupportedTarget {
            target: target.to_string(),
            feature: "Windows COFF emission for this architecture".to_string(),
        });
    };

    Ok(TargetSpec {
        format: LinkArtifactFormat::Coff,
        architecture,
    })
}

fn emit_windows_coff_bundle(
    bundle: &FrozenModuleBundle,
    architecture: Architecture,
) -> Result<Vec<u8>, AotError> {
    let bundle_bytes = bundle.to_bytes()?;
    let mut object = Object::new(BinaryFormat::Coff, architecture, Endianness::Little);
    let section = object.add_section(
        Vec::new(),
        PRISM_BUNDLE_SECTION.to_vec(),
        SectionKind::ReadOnlyData,
    );
    object.append_section_data(section, &bundle_bytes, 1);

    let bundle_size = u64::try_from(bundle_bytes.len()).map_err(|_| AotError::InvalidArtifact {
        message: format!(
            "frozen bundle exceeds supported object symbol size: {} bytes",
            bundle_bytes.len()
        ),
    })?;

    object.add_symbol(Symbol {
        name: FROZEN_BUNDLE_START_SYMBOL.as_bytes().to_vec(),
        value: 0,
        size: bundle_size,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });
    object.add_symbol(Symbol {
        name: FROZEN_BUNDLE_END_SYMBOL.as_bytes().to_vec(),
        value: bundle_size,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(section),
        flags: SymbolFlags::None,
    });

    object.write().map_err(|err| AotError::EmitArtifact {
        artifact: "Windows COFF object".to_string(),
        message: err.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::{BuildEntry, BuildOptions, BuildPlanner};
    use object::{Object as _, ObjectSection as _, ObjectSymbol as _};
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

        let err = LinkableBundleArtifact::from_bundle(&bundle)
            .expect_err("unsupported target should fail");
        match err {
            AotError::UnsupportedTarget { target, .. } => assert_eq!(target, "x86_64-linux"),
            other => panic!("unexpected error: {other}"),
        }
    }
}
