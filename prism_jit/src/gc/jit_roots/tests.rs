use super::*;

#[test]
fn test_jit_roots_basics() {
    let mut roots = JitRoots::new();
    assert!(roots.is_empty());

    roots.stack_values.push((
        Value::int(42).unwrap(),
        RootLocation::Stack {
            frame_base: 0x1000,
            offset: -8,
        },
    ));
    assert_eq!(roots.len(), 1);
    assert!(!roots.is_empty());

    roots.clear();
    assert!(roots.is_empty());
}

#[test]
fn test_root_location_address() {
    let stack_loc = RootLocation::Stack {
        frame_base: 0x1000,
        offset: -16,
    };
    assert_eq!(stack_loc.address(), 0x1000 - 16);

    let reg_loc = RootLocation::Register {
        reg_index: 0,
        saved_at: 0x2000,
    };
    assert_eq!(reg_loc.address(), 0x2000);
}

#[test]
fn test_saved_registers() {
    let mut regs = SavedRegisters::new();

    regs.set(0, Value::int(42).unwrap());
    assert_eq!(regs.get(0).unwrap().as_int(), Some(42));

    regs.set_raw(1, 0x12345678);
    assert_eq!(regs.gprs[1], 0x12345678);

    // Out of bounds
    assert!(regs.get(20).is_none());
}

#[test]
fn test_conservative_scanner() {
    let scanner = ConservativeScanner::new(0x10000, 0x20000);

    // Valid pointer
    assert!(scanner.is_potential_pointer(0x15000));

    // Unaligned
    assert!(!scanner.is_potential_pointer(0x15001));

    // Below heap
    assert!(!scanner.is_potential_pointer(0x5000));

    // Above heap
    assert!(!scanner.is_potential_pointer(0x25000));
}
