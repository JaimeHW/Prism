//! OSR Entry Stub Generator.
//!
//! Generates descriptors for OSR entry stubs that materialize interpreter
//! state into JIT register/stack layout. The actual machine code generation
//! integrates with the existing codegen infrastructure.
//!
//! # Design
//!
//! Rather than duplicate the assembler logic, this module provides:
//! 1. **OsrStubInfo** - Describes what code to generate
//! 2. Integration with the tier1 template system for actual emission

use crate::tier2::osr::{OsrEntry, OsrStateDescriptor, ValueLocation};
use std::collections::BTreeMap;

// =============================================================================
// OSR Stub Info
// =============================================================================

/// Information needed to generate an OSR entry stub.
#[derive(Debug, Clone)]
pub struct OsrStubInfo {
    /// Target JIT code offset.
    pub jit_target_offset: u32,
    /// Frame size required.
    pub frame_size: u32,
    /// Number of callee-saved registers.
    pub callee_saved_count: u8,
    /// Value materialization instructions.
    pub materializations: Vec<Materialization>,
}

/// A single value materialization instruction.
#[derive(Debug, Clone, Copy)]
pub struct Materialization {
    /// Source: interpreter local index.
    pub source_local_idx: u16,
    /// Destination in JIT frame.
    pub destination: ValueLocation,
}

impl OsrStubInfo {
    /// Create from an OSR entry descriptor.
    pub fn from_entry(entry: &OsrEntry) -> Self {
        let descriptor = &entry.state_descriptor;

        let materializations: Vec<_> = descriptor
            .local_mappings()
            .iter()
            .enumerate()
            .filter(|(_, loc)| loc.is_live())
            .map(|(idx, loc)| Materialization {
                source_local_idx: idx as u16,
                destination: *loc,
            })
            .collect();

        Self {
            jit_target_offset: entry.jit_offset,
            frame_size: descriptor.frame_size(),
            callee_saved_count: descriptor.callee_saved_count(),
            materializations,
        }
    }

    /// Estimate the size of the generated stub in bytes.
    pub fn estimated_size(&self) -> usize {
        // Prologue: ~10 bytes
        // Frame allocation: ~7 bytes
        // Each materialization: ~15 bytes
        // Jump: ~12 bytes
        20 + self.materializations.len() * 15 + 12
    }

    /// Get live register destinations.
    pub fn register_destinations(&self) -> impl Iterator<Item = (u16, u8)> + '_ {
        self.materializations.iter().filter_map(|m| {
            if let ValueLocation::Register(reg) = m.destination {
                Some((m.source_local_idx, reg))
            } else {
                None
            }
        })
    }

    /// Get live stack destinations.
    pub fn stack_destinations(&self) -> impl Iterator<Item = (u16, i32)> + '_ {
        self.materializations.iter().filter_map(|m| {
            if let ValueLocation::Stack(offset) = m.destination {
                Some((m.source_local_idx, offset))
            } else {
                None
            }
        })
    }
}

// =============================================================================
// OSR Stub Cache
// =============================================================================

/// Cache of generated OSR stub information.
#[derive(Debug, Default)]
pub struct OsrStubCache {
    /// Stubs indexed by (code_id, bytecode_offset).
    stubs: BTreeMap<(u64, u32), OsrStubInfo>,
}

impl OsrStubCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store stub info for a code/offset pair.
    pub fn insert(&mut self, code_id: u64, bc_offset: u32, info: OsrStubInfo) {
        self.stubs.insert((code_id, bc_offset), info);
    }

    /// Retrieve stub info.
    pub fn get(&self, code_id: u64, bc_offset: u32) -> Option<&OsrStubInfo> {
        self.stubs.get(&(code_id, bc_offset))
    }

    /// Remove stub info.
    pub fn remove(&mut self, code_id: u64, bc_offset: u32) -> Option<OsrStubInfo> {
        self.stubs.remove(&(code_id, bc_offset))
    }

    /// Clear all stubs for a code object.
    pub fn clear_for_code(&mut self, code_id: u64) {
        self.stubs.retain(|(c, _), _| *c != code_id);
    }

    /// Total number of cached stubs.
    pub fn len(&self) -> usize {
        self.stubs.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.stubs.is_empty()
    }

    /// Clear all stubs.
    pub fn clear(&mut self) {
        self.stubs.clear()
    }
}

// =============================================================================
// OSR Exit Builder
// =============================================================================

/// Builds state for transitioning from JIT back to interpreter.
#[derive(Debug, Default)]
pub struct OsrExitBuilder {
    /// Values to restore to interpreter frame.
    values: Vec<CapturedValue>,
}

/// A value captured from JIT state.
#[derive(Debug, Clone, Copy)]
pub struct CapturedValue {
    /// Destination interpreter local index.
    pub local_idx: u16,
    /// Source location in JIT frame.
    pub source: ValueLocation,
}

impl OsrExitBuilder {
    /// Create a new exit builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value to capture.
    pub fn capture(&mut self, local_idx: u16, source: ValueLocation) {
        self.values.push(CapturedValue { local_idx, source });
    }

    /// Build from state descriptor (captures all live values).
    pub fn from_descriptor(descriptor: &OsrStateDescriptor) -> Self {
        let values: Vec<_> = descriptor
            .local_mappings()
            .iter()
            .enumerate()
            .filter(|(_, loc)| loc.is_live())
            .map(|(idx, loc)| CapturedValue {
                local_idx: idx as u16,
                source: *loc,
            })
            .collect();

        Self { values }
    }

    /// Get all captured values.
    pub fn values(&self) -> &[CapturedValue] {
        &self.values
    }

    /// Number of values to capture.
    pub fn count(&self) -> usize {
        self.values.len()
    }
}
