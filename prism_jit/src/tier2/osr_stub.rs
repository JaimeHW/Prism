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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_osr_stub_info_from_entry() {
        let mut desc = OsrStateDescriptor::new();
        desc.set_frame_size(64);
        desc.set_callee_saved_count(2);
        desc.add_local_mapping(ValueLocation::register(0));
        desc.add_local_mapping(ValueLocation::stack(-8));
        desc.add_local_mapping(ValueLocation::dead());
        desc.add_local_mapping(ValueLocation::constant(42));

        let entry = OsrEntry::new(100, 200, desc);
        let info = OsrStubInfo::from_entry(&entry);

        assert_eq!(info.jit_target_offset, 200);
        assert_eq!(info.frame_size, 64);
        assert_eq!(info.callee_saved_count, 2);
        // 3 live values (constant is also considered live for capture)
        assert_eq!(info.materializations.len(), 3);
    }

    #[test]
    fn test_osr_stub_info_estimated_size() {
        let info = OsrStubInfo {
            jit_target_offset: 100,
            frame_size: 32,
            callee_saved_count: 1,
            materializations: vec![Materialization {
                source_local_idx: 0,
                destination: ValueLocation::register(0),
            }],
        };

        let size = info.estimated_size();
        assert!(size > 0);
        assert!(size < 1000); // Reasonable bound
    }

    #[test]
    fn test_osr_stub_cache() {
        let mut cache = OsrStubCache::new();
        assert!(cache.is_empty());

        let info = OsrStubInfo {
            jit_target_offset: 100,
            frame_size: 32,
            callee_saved_count: 0,
            materializations: vec![],
        };

        cache.insert(1, 50, info);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(1, 50).is_some());
        assert!(cache.get(1, 60).is_none());

        cache.remove(1, 50);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_osr_stub_cache_clear_for_code() {
        let mut cache = OsrStubCache::new();

        let info = OsrStubInfo {
            jit_target_offset: 100,
            frame_size: 32,
            callee_saved_count: 0,
            materializations: vec![],
        };

        cache.insert(1, 50, info.clone());
        cache.insert(1, 100, info.clone());
        cache.insert(2, 50, info);

        assert_eq!(cache.len(), 3);

        cache.clear_for_code(1);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(2, 50).is_some());
    }

    #[test]
    fn test_osr_exit_builder() {
        let mut builder = OsrExitBuilder::new();
        builder.capture(0, ValueLocation::register(0));
        builder.capture(1, ValueLocation::stack(-8));

        assert_eq!(builder.count(), 2);
    }

    #[test]
    fn test_osr_exit_builder_from_descriptor() {
        let mut desc = OsrStateDescriptor::new();
        desc.add_local_mapping(ValueLocation::register(0));
        desc.add_local_mapping(ValueLocation::dead());
        desc.add_local_mapping(ValueLocation::stack(-16));

        let builder = OsrExitBuilder::from_descriptor(&desc);
        assert_eq!(builder.count(), 2); // Only 2 live values
    }

    #[test]
    fn test_register_and_stack_destinations() {
        let info = OsrStubInfo {
            jit_target_offset: 100,
            frame_size: 64,
            callee_saved_count: 0,
            materializations: vec![
                Materialization {
                    source_local_idx: 0,
                    destination: ValueLocation::register(0),
                },
                Materialization {
                    source_local_idx: 1,
                    destination: ValueLocation::stack(-8),
                },
                Materialization {
                    source_local_idx: 2,
                    destination: ValueLocation::register(1),
                },
            ],
        };

        let regs: Vec<_> = info.register_destinations().collect();
        assert_eq!(regs.len(), 2);
        assert_eq!(regs[0], (0, 0));
        assert_eq!(regs[1], (2, 1));

        let stacks: Vec<_> = info.stack_destinations().collect();
        assert_eq!(stacks.len(), 1);
        assert_eq!(stacks[0], (1, -8));
    }
}
