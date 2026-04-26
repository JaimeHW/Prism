use super::*;

#[test]
fn test_state_point_creation() {
    let sp = StatePoint::new(100, 50);
    assert_eq!(sp.jit_offset, 100);
    assert_eq!(sp.bc_offset, 50);
    assert_eq!(sp.local_count(), 0);
    assert_eq!(sp.stack_count(), 0);
}

#[test]
fn test_state_point_with_values() {
    let mut sp = StatePoint::with_capacity(100, 50, 4, 2);
    sp.add_local(ValueLocation::register(0));
    sp.add_local(ValueLocation::stack(-8));
    sp.add_local(ValueLocation::dead());
    sp.add_stack(ValueLocation::register(1));

    assert_eq!(sp.local_count(), 3);
    assert_eq!(sp.stack_count(), 1);
    assert_eq!(sp.live_local_count(), 2);
    assert_eq!(sp.live_stack_count(), 1);
}

#[test]
fn test_frame_info() {
    let info = StatePointFrameInfo::new(64, 48)
        .with_callee_saved(4)
        .with_spills();

    assert_eq!(info.jit_frame_size, 64);
    assert_eq!(info.interp_frame_size, 48);
    assert_eq!(info.callee_saved_count, 4);
    assert!(info.has_spills);
}

#[test]
fn test_state_point_table() {
    let mut table = StatePointTable::new();

    // Add out of order
    table.add(StatePoint::new(200, 100));
    table.add(StatePoint::new(100, 50));
    table.add(StatePoint::new(150, 75));

    assert_eq!(table.len(), 3);

    // Should be sorted
    assert_eq!(table.points()[0].jit_offset, 100);
    assert_eq!(table.points()[1].jit_offset, 150);
    assert_eq!(table.points()[2].jit_offset, 200);

    // Lookup exact
    assert!(table.lookup_exact(150).is_some());
    assert!(table.lookup_exact(125).is_none());

    // Lookup at or before
    let sp = table.lookup_at_or_before(175).unwrap();
    assert_eq!(sp.jit_offset, 150);
}

#[test]
fn test_state_point_builder() {
    let mut builder = StatePointBuilder::new(100, 50);
    builder.local_in_reg(0);
    builder.local_on_stack(-8);
    builder.local_constant(42);
    builder.local_dead();
    builder.stack_in_reg(1);
    builder.frame_info(StatePointFrameInfo::new(64, 48));
    let sp = builder.build();

    assert_eq!(sp.jit_offset, 100);
    assert_eq!(sp.bc_offset, 50);
    assert_eq!(sp.local_count(), 4);
    assert_eq!(sp.stack_count(), 1);
    assert_eq!(sp.frame_info().jit_frame_size, 64);
}

#[test]
fn test_state_reconstructor_constants() {
    let mut sp = StatePoint::new(100, 50);
    sp.add_local(ValueLocation::constant(42));
    sp.add_local(ValueLocation::dead());

    let regs = [0u64; 16];
    let reconstructor = StateReconstructor::new(&sp, &regs, 0);

    // Constants and dead values don't need memory access
    unsafe {
        assert_eq!(
            reconstructor.read_location(&ValueLocation::constant(42)),
            Some(42)
        );
        assert_eq!(reconstructor.read_location(&ValueLocation::dead()), None);
    }
}

#[test]
fn test_state_reconstructor_registers() {
    let mut sp = StatePoint::new(100, 50);
    sp.add_local(ValueLocation::register(0));
    sp.add_local(ValueLocation::register(5));

    let mut regs = [0u64; 16];
    regs[0] = 100;
    regs[5] = 500;

    let reconstructor = StateReconstructor::new(&sp, &regs, 0);

    unsafe {
        let locals = reconstructor.reconstruct_locals();
        assert_eq!(locals[0], Some(100));
        assert_eq!(locals[1], Some(500));
    }
}
