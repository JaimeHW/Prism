use std::collections::BTreeMap;
use std::mem::{offset_of, size_of};
use std::path::Path;

use object::write::{Object, Relocation, SectionId, Symbol, SymbolId, SymbolSection};
use object::{
    Architecture, BinaryFormat, Endianness, RelocationEncoding, RelocationFlags, RelocationKind,
    SectionKind, SymbolFlags, SymbolKind, SymbolScope,
};
use prism_core::aot::{
    AOT_IMPORT_FROM_SYMBOL, AOT_IMPORT_MODULE_SYMBOL, AOT_NATIVE_INIT_TABLE_END_SYMBOL,
    AOT_NATIVE_INIT_TABLE_START_SYMBOL, AOT_STORE_EXPR_SYMBOL, AotImmediate, AotImmediateKind,
    AotImportFromOp, AotImportModuleOp, AotNativeModuleInitEntry, AotOperand, AotOperandKind,
    AotStoreExprKind, AotStoreExprOp, AotStringRef,
};

use crate::bundle::FrozenModuleBundle;
use crate::error::AotError;
use crate::native::{
    NativeExpr, NativeImmediate, NativeInitOperation, NativeModuleInitPlan, NativeOperand,
};
use crate::planner::BuildPlan;

/// Symbol exported at the beginning of the frozen module payload.
pub const FROZEN_BUNDLE_START_SYMBOL: &str = "prism_frozen_bundle_start";
/// Symbol exported immediately after the frozen module payload.
pub const FROZEN_BUNDLE_END_SYMBOL: &str = "prism_frozen_bundle_end";

const PRISM_BUNDLE_SECTION: &[u8] = b".rdata$prism$bundle";
const PRISM_NATIVE_DATA_SECTION: &[u8] = b".rdata$prism$native";
const PRISM_NATIVE_TEXT_SECTION: &[u8] = b".text$prism$native";

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

    /// Emit a linkable object from a whole-program build plan.
    ///
    /// The emitted object always contains the frozen bundle payload. On Windows
    /// x64 targets it additionally contains native module-init stubs for source
    /// modules that fit Prism's current lowering subset.
    pub fn from_build_plan(plan: &BuildPlan) -> Result<Self, AotError> {
        let spec = target_spec(&plan.target)?;
        let bundle = FrozenModuleBundle::from_build_plan(plan)?;
        let bytes = match spec.format {
            LinkArtifactFormat::Coff => {
                emit_windows_coff_build_plan(plan, &bundle, spec.architecture)?
            }
        };

        Ok(Self {
            target: plan.target.clone(),
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

#[derive(Debug)]
struct NativeCoffState {
    data_section: SectionId,
    text_section: SectionId,
    string_symbols: BTreeMap<Vec<u8>, SymbolId>,
    helper_symbols: BTreeMap<&'static str, SymbolId>,
    next_string_id: usize,
}

#[derive(Debug, Clone, Copy)]
struct PendingRelocation {
    offset: u64,
    symbol: SymbolId,
    addend: i64,
    kind: RelocationKind,
    encoding: RelocationEncoding,
    size: u8,
}

#[derive(Debug, Clone)]
struct NativeInitRegistryEntry {
    module_name: String,
    init_symbol: SymbolId,
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
    let mut object = Object::new(BinaryFormat::Coff, architecture, Endianness::Little);
    emit_bundle_section(&mut object, bundle)?;
    write_object(object)
}

fn emit_windows_coff_build_plan(
    plan: &BuildPlan,
    bundle: &FrozenModuleBundle,
    architecture: Architecture,
) -> Result<Vec<u8>, AotError> {
    let mut object = Object::new(BinaryFormat::Coff, architecture, Endianness::Little);
    emit_bundle_section(&mut object, bundle)?;

    if plan
        .modules
        .iter()
        .any(|module| module.native_init.is_some())
    {
        match architecture {
            Architecture::X86_64 => emit_x86_64_native_init_stubs(&mut object, plan)?,
            _ => {
                return Err(AotError::UnsupportedTarget {
                    target: plan.target.clone(),
                    feature: "native module init emission for this architecture".to_string(),
                });
            }
        }
    }

    write_object(object)
}

fn emit_bundle_section(
    object: &mut Object<'_>,
    bundle: &FrozenModuleBundle,
) -> Result<(), AotError> {
    let bundle_bytes = bundle.to_bytes()?;
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

    Ok(())
}

fn emit_x86_64_native_init_stubs(
    object: &mut Object<'_>,
    plan: &BuildPlan,
) -> Result<(), AotError> {
    let mut state = NativeCoffState {
        data_section: object.add_section(
            Vec::new(),
            PRISM_NATIVE_DATA_SECTION.to_vec(),
            SectionKind::ReadOnlyData,
        ),
        text_section: object.add_section(
            Vec::new(),
            PRISM_NATIVE_TEXT_SECTION.to_vec(),
            SectionKind::Text,
        ),
        string_symbols: BTreeMap::new(),
        helper_symbols: BTreeMap::new(),
        next_string_id: 0,
    };

    let mut registry_entries = Vec::new();
    for module in &plan.modules {
        if let Some(native_init) = &module.native_init {
            let init_symbol = emit_native_init_stub(object, &mut state, native_init)?;
            registry_entries.push(NativeInitRegistryEntry {
                module_name: native_init.module_name.clone(),
                init_symbol,
            });
        }
    }

    emit_native_init_registry(object, &mut state, &registry_entries)?;
    Ok(())
}

fn emit_native_init_stub(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    native_init: &NativeModuleInitPlan,
) -> Result<SymbolId, AotError> {
    let mut operations = Vec::with_capacity(native_init.operations.len());
    for (index, operation) in native_init.operations.iter().enumerate() {
        let descriptor_symbol =
            emit_operation_descriptor(object, state, native_init, operation, index)?;
        let helper_symbol = helper_symbol_id(object, state, operation_helper_symbol(operation));
        operations.push((descriptor_symbol, helper_symbol));
    }

    let mut code = Vec::new();
    let mut relocations = Vec::new();

    if operations.is_empty() {
        code.extend_from_slice(&[0x31, 0xC0, 0xC3]);
    } else {
        code.extend_from_slice(&[0x48, 0x83, 0xEC, 0x38]);
        code.extend_from_slice(&[0x48, 0x89, 0x4C, 0x24, 0x20]);
        code.extend_from_slice(&[0x48, 0x89, 0x54, 0x24, 0x28]);

        let mut failure_jumps = Vec::with_capacity(operations.len());
        for (descriptor_symbol, helper_symbol) in operations {
            code.extend_from_slice(&[0x48, 0x8B, 0x4C, 0x24, 0x20]);
            code.extend_from_slice(&[0x48, 0x8B, 0x54, 0x24, 0x28]);

            let lea_disp_offset =
                u64::try_from(code.len() + 3).map_err(|_| AotError::InvalidArtifact {
                    message: format!(
                        "native init stub '{}' exceeds supported code size",
                        native_init.symbol_name
                    ),
                })?;
            code.extend_from_slice(&[0x4C, 0x8D, 0x05, 0, 0, 0, 0]);
            relocations.push(PendingRelocation {
                offset: lea_disp_offset,
                symbol: descriptor_symbol,
                addend: 0,
                kind: RelocationKind::Relative,
                encoding: RelocationEncoding::Generic,
                size: 32,
            });

            let call_disp_offset =
                u64::try_from(code.len() + 1).map_err(|_| AotError::InvalidArtifact {
                    message: format!(
                        "native init stub '{}' exceeds supported code size",
                        native_init.symbol_name
                    ),
                })?;
            code.extend_from_slice(&[0xE8, 0, 0, 0, 0]);
            relocations.push(PendingRelocation {
                offset: call_disp_offset,
                symbol: helper_symbol,
                addend: 0,
                kind: RelocationKind::Relative,
                encoding: RelocationEncoding::Generic,
                size: 32,
            });

            code.extend_from_slice(&[0x85, 0xC0]);
            let jump_disp_offset = code.len() + 2;
            code.extend_from_slice(&[0x0F, 0x85, 0, 0, 0, 0]);
            failure_jumps.push(jump_disp_offset);
        }

        code.extend_from_slice(&[0x31, 0xC0]);
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38]);
        code.push(0xC3);

        let failure_offset = code.len();
        code.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]);
        code.extend_from_slice(&[0x48, 0x83, 0xC4, 0x38]);
        code.push(0xC3);

        for jump_disp_offset in failure_jumps {
            let displacement = i32::try_from(failure_offset as i64 - jump_disp_offset as i64)
                .map_err(|_| AotError::InvalidArtifact {
                    message: format!(
                        "native init stub '{}' exceeds supported branch range",
                        native_init.symbol_name
                    ),
                })?;
            write_i32(&mut code, jump_disp_offset, displacement);
        }
    }

    let symbol_id = object.add_symbol(Symbol {
        name: native_init.symbol_name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Text,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    let section_offset = object.add_symbol_data(symbol_id, state.text_section, &code, 16);
    apply_relocations(object, state.text_section, section_offset, &relocations)?;
    Ok(symbol_id)
}

fn emit_native_init_registry(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    entries: &[NativeInitRegistryEntry],
) -> Result<(), AotError> {
    if entries.is_empty() {
        return Ok(());
    }

    let entry_size = size_of::<AotNativeModuleInitEntry>();
    let total_size =
        entry_size
            .checked_mul(entries.len())
            .ok_or_else(|| AotError::InvalidArtifact {
                message: "native init registry exceeds supported size".to_string(),
            })?;
    let mut bytes = vec![0_u8; total_size];
    let mut relocations = Vec::new();

    for (index, entry) in entries.iter().enumerate() {
        let base = index * entry_size;
        encode_string_ref(
            &mut bytes,
            &mut relocations,
            object,
            state,
            base + offset_of!(AotNativeModuleInitEntry, module_name),
            entry.module_name.as_bytes(),
        )?;
        relocations.push(PendingRelocation {
            offset: u64::try_from(base + offset_of!(AotNativeModuleInitEntry, init_fn)).map_err(
                |_| AotError::InvalidArtifact {
                    message: "native init registry offset exceeds supported range".to_string(),
                },
            )?,
            symbol: entry.init_symbol,
            addend: 0,
            kind: RelocationKind::Absolute,
            encoding: RelocationEncoding::Generic,
            size: 64,
        });
    }

    let table_symbol = object.add_symbol(Symbol {
        name: b"prism_aot_native_init_table".to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Compilation,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    let section_offset = object.add_symbol_data(table_symbol, state.data_section, &bytes, 8);
    apply_relocations(object, state.data_section, section_offset, &relocations)?;

    let table_size = u64::try_from(bytes.len()).map_err(|_| AotError::InvalidArtifact {
        message: "native init registry exceeds supported symbol size".to_string(),
    })?;
    object.add_symbol(Symbol {
        name: AOT_NATIVE_INIT_TABLE_START_SYMBOL.as_bytes().to_vec(),
        value: section_offset,
        size: table_size,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(state.data_section),
        flags: SymbolFlags::None,
    });
    object.add_symbol(Symbol {
        name: AOT_NATIVE_INIT_TABLE_END_SYMBOL.as_bytes().to_vec(),
        value: section_offset + table_size,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Section(state.data_section),
        flags: SymbolFlags::None,
    });

    Ok(())
}

fn emit_operation_descriptor(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    native_init: &NativeModuleInitPlan,
    operation: &NativeInitOperation,
    index: usize,
) -> Result<SymbolId, AotError> {
    let symbol_name = format!("{}_op_{index}", native_init.symbol_name);
    match operation {
        NativeInitOperation::ImportModule {
            target,
            module_spec,
            binding,
        } => emit_named_descriptor(
            object,
            state,
            &symbol_name,
            size_of::<AotImportModuleOp>(),
            |bytes, relocations, object, state| {
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotImportModuleOp, target),
                    target.as_bytes(),
                )?;
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotImportModuleOp, module_spec),
                    module_spec.as_bytes(),
                )?;
                bytes[offset_of!(AotImportModuleOp, binding)] = *binding as u8;
                Ok(())
            },
        ),
        NativeInitOperation::ImportFrom {
            target,
            module_spec,
            attribute,
        } => emit_named_descriptor(
            object,
            state,
            &symbol_name,
            size_of::<AotImportFromOp>(),
            |bytes, relocations, object, state| {
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotImportFromOp, target),
                    target.as_bytes(),
                )?;
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotImportFromOp, module_spec),
                    module_spec.as_bytes(),
                )?;
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotImportFromOp, attribute),
                    attribute.as_bytes(),
                )?;
                Ok(())
            },
        ),
        NativeInitOperation::StoreExpr { target, expr } => emit_named_descriptor(
            object,
            state,
            &symbol_name,
            size_of::<AotStoreExprOp>(),
            |bytes, relocations, object, state| {
                encode_string_ref(
                    bytes,
                    relocations,
                    object,
                    state,
                    offset_of!(AotStoreExprOp, target),
                    target.as_bytes(),
                )?;
                match expr {
                    NativeExpr::Operand(operand) => {
                        bytes[offset_of!(AotStoreExprOp, kind)] = AotStoreExprKind::Operand as u8;
                        encode_operand(
                            bytes,
                            relocations,
                            object,
                            state,
                            offset_of!(AotStoreExprOp, left),
                            operand,
                        )?;
                    }
                    NativeExpr::Add { left, right } => {
                        bytes[offset_of!(AotStoreExprOp, kind)] = AotStoreExprKind::Add as u8;
                        encode_operand(
                            bytes,
                            relocations,
                            object,
                            state,
                            offset_of!(AotStoreExprOp, left),
                            left,
                        )?;
                        encode_operand(
                            bytes,
                            relocations,
                            object,
                            state,
                            offset_of!(AotStoreExprOp, right),
                            right,
                        )?;
                    }
                }
                Ok(())
            },
        ),
    }
}

fn emit_named_descriptor(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    name: &str,
    size: usize,
    encode: impl FnOnce(
        &mut [u8],
        &mut Vec<PendingRelocation>,
        &mut Object<'_>,
        &mut NativeCoffState,
    ) -> Result<(), AotError>,
) -> Result<SymbolId, AotError> {
    let mut bytes = vec![0_u8; size];
    let mut relocations = Vec::new();
    encode(&mut bytes, &mut relocations, object, state)?;

    let symbol_id = object.add_symbol(Symbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Compilation,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    let section_offset = object.add_symbol_data(symbol_id, state.data_section, &bytes, 8);
    apply_relocations(object, state.data_section, section_offset, &relocations)?;
    Ok(symbol_id)
}

fn encode_operand(
    bytes: &mut [u8],
    relocations: &mut Vec<PendingRelocation>,
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    base: usize,
    operand: &NativeOperand,
) -> Result<(), AotError> {
    match operand {
        NativeOperand::Immediate(immediate) => {
            bytes[base + offset_of!(AotOperand, kind)] = AotOperandKind::Immediate as u8;
            let immediate_base = base + offset_of!(AotOperand, immediate);
            match immediate {
                NativeImmediate::ValueBits(bits) => {
                    bytes[immediate_base + offset_of!(AotImmediate, kind)] =
                        AotImmediateKind::ValueBits as u8;
                    write_u64(
                        bytes,
                        immediate_base + offset_of!(AotImmediate, bits),
                        *bits,
                    );
                }
                NativeImmediate::String(string) => {
                    bytes[immediate_base + offset_of!(AotImmediate, kind)] =
                        AotImmediateKind::String as u8;
                    encode_string_ref(
                        bytes,
                        relocations,
                        object,
                        state,
                        immediate_base + offset_of!(AotImmediate, string),
                        string.as_bytes(),
                    )?;
                }
            }
        }
        NativeOperand::Name(name) => {
            bytes[base + offset_of!(AotOperand, kind)] = AotOperandKind::Name as u8;
            encode_string_ref(
                bytes,
                relocations,
                object,
                state,
                base + offset_of!(AotOperand, name),
                name.as_bytes(),
            )?;
        }
    }

    Ok(())
}

fn encode_string_ref(
    bytes: &mut [u8],
    relocations: &mut Vec<PendingRelocation>,
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    base: usize,
    string_bytes: &[u8],
) -> Result<(), AotError> {
    let len = u64::try_from(string_bytes.len()).map_err(|_| AotError::InvalidArtifact {
        message: format!(
            "native init string literal exceeds supported size: {} bytes",
            string_bytes.len()
        ),
    })?;
    write_u64(bytes, base + offset_of!(AotStringRef, len), len);

    if !string_bytes.is_empty() {
        let symbol = intern_string_symbol(object, state, string_bytes);
        relocations.push(PendingRelocation {
            offset: u64::try_from(base + offset_of!(AotStringRef, data)).map_err(|_| {
                AotError::InvalidArtifact {
                    message: "native init descriptor offset exceeds supported range".to_string(),
                }
            })?,
            symbol,
            addend: 0,
            kind: RelocationKind::Absolute,
            encoding: RelocationEncoding::Generic,
            size: 64,
        });
    }

    Ok(())
}

fn intern_string_symbol(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    string_bytes: &[u8],
) -> SymbolId {
    if let Some(symbol) = state.string_symbols.get(string_bytes) {
        return *symbol;
    }

    let name = format!("prism_aot_str_{}", state.next_string_id);
    state.next_string_id += 1;
    let symbol_id = object.add_symbol(Symbol {
        name: name.into_bytes(),
        value: 0,
        size: 0,
        kind: SymbolKind::Data,
        scope: SymbolScope::Compilation,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    object.add_symbol_data(symbol_id, state.data_section, string_bytes, 1);
    state
        .string_symbols
        .insert(string_bytes.to_vec(), symbol_id);
    symbol_id
}

fn helper_symbol_id(
    object: &mut Object<'_>,
    state: &mut NativeCoffState,
    name: &'static str,
) -> SymbolId {
    if let Some(symbol) = state.helper_symbols.get(name) {
        return *symbol;
    }

    let symbol_id = object.add_symbol(Symbol {
        name: name.as_bytes().to_vec(),
        value: 0,
        size: 0,
        kind: SymbolKind::Text,
        scope: SymbolScope::Linkage,
        weak: false,
        section: SymbolSection::Undefined,
        flags: SymbolFlags::None,
    });
    state.helper_symbols.insert(name, symbol_id);
    symbol_id
}

fn operation_helper_symbol(operation: &NativeInitOperation) -> &'static str {
    match operation {
        NativeInitOperation::ImportModule { .. } => AOT_IMPORT_MODULE_SYMBOL,
        NativeInitOperation::ImportFrom { .. } => AOT_IMPORT_FROM_SYMBOL,
        NativeInitOperation::StoreExpr { .. } => AOT_STORE_EXPR_SYMBOL,
    }
}

fn apply_relocations(
    object: &mut Object<'_>,
    section: SectionId,
    section_offset: u64,
    relocations: &[PendingRelocation],
) -> Result<(), AotError> {
    for relocation in relocations {
        object
            .add_relocation(
                section,
                Relocation {
                    offset: section_offset + relocation.offset,
                    symbol: relocation.symbol,
                    addend: relocation.addend,
                    flags: RelocationFlags::Generic {
                        kind: relocation.kind,
                        encoding: relocation.encoding,
                        size: relocation.size,
                    },
                },
            )
            .map_err(|err| AotError::EmitArtifact {
                artifact: "Windows COFF relocation".to_string(),
                message: err.to_string(),
            })?;
    }

    Ok(())
}

fn write_object(object: Object<'_>) -> Result<Vec<u8>, AotError> {
    object.write().map_err(|err| AotError::EmitArtifact {
        artifact: "Windows COFF object".to_string(),
        message: err.to_string(),
    })
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}
