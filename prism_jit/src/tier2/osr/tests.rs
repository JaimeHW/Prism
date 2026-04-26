use super::*;

#[test]
fn test_value_location_creation() {
    let reg = ValueLocation::register(0);
    assert!(reg.is_register());
    assert!(reg.is_live());

    let stack = ValueLocation::stack(-8);
    assert!(stack.is_stack());
    assert!(stack.is_live());

    let dead = ValueLocation::dead();
    assert!(!dead.is_live());
}

#[test]
fn test_osr_state_descriptor() {
    let mut desc = OsrStateDescriptor::new();
    desc.set_frame_size(64);
    desc.set_callee_saved_count(4);
    desc.add_local_mapping(ValueLocation::register(0));
    desc.add_local_mapping(ValueLocation::stack(-8));
    desc.add_local_mapping(ValueLocation::dead());

    assert_eq!(desc.frame_size(), 64);
    assert_eq!(desc.callee_saved_count(), 4);
    assert_eq!(desc.local_count(), 3);
    assert!(desc.local_location(0).unwrap().is_register());
    assert!(desc.local_location(1).unwrap().is_stack());
    assert!(!desc.local_location(2).unwrap().is_live());
}

#[test]
fn test_osr_entry() {
    let mut desc = OsrStateDescriptor::new();
    desc.set_frame_size(48);

    let entry = OsrEntry::new(100, 200, desc);
    assert_eq!(entry.bc_offset, 100);
    assert_eq!(entry.jit_offset, 200);
    assert_eq!(entry.state_descriptor.frame_size(), 48);
}

#[test]
fn test_osr_compiled_code() {
    let mut code = OsrCompiledCode::new(0x10000, 0x1000);

    let desc = OsrStateDescriptor::new();
    code.add_entry(OsrEntry::new(0, 100, desc.clone()));
    code.add_entry(OsrEntry::new(50, 200, desc));

    assert_eq!(code.entry_count(), 2);
    assert_eq!(code.entry_address(0), Some(0x10000 + 100));
    assert_eq!(code.entry_address(50), Some(0x10000 + 200));
    assert!(code.entry_address(25).is_none());

    assert!(code.contains_address(0x10500));
    assert!(!code.contains_address(0x20000));
}

#[test]
fn test_deopt_info() {
    let desc = OsrStateDescriptor::new();
    let info = DeoptInfo::new(100, 50, desc, DeoptReason::TypeGuard);

    assert_eq!(info.jit_offset, 100);
    assert_eq!(info.bc_offset, 50);
    assert_eq!(info.reason, DeoptReason::TypeGuard);
    assert_eq!(info.reason.description(), "type guard failed");
}

#[test]
fn test_osr_state_builder() {
    let mut builder = OsrStateBuilder::new();
    builder.set_register(0, 42);
    builder.set_register(1, 100);
    builder.set_stack(-8, 0xDEADBEEF);
    builder.set_stack(-16, 0xCAFEBABE);

    assert_eq!(builder.register_count(), 2);
    assert_eq!(builder.stack_count(), 2);

    // Overwrite should work
    builder.set_register(0, 99);
    assert_eq!(builder.register_count(), 2);
}

#[test]
fn test_osr_stats() {
    let mut stats = OsrStats::new();
    stats.record_entry();
    stats.record_entry();
    stats.record_deopt(DeoptReason::TypeGuard);
    stats.record_deopt(DeoptReason::TypeGuard);
    stats.record_deopt(DeoptReason::Overflow);

    assert_eq!(stats.entry_count, 2);
    assert_eq!(stats.deopt_count, 3);
    assert_eq!(stats.deopt_count_for(DeoptReason::TypeGuard), 2);
    assert_eq!(stats.deopt_count_for(DeoptReason::Overflow), 1);
    assert_eq!(stats.deopt_count_for(DeoptReason::BoundsCheck), 0);
}
