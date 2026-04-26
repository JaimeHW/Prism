use super::*;

#[test]
fn test_exit_reason_from_u8() {
    assert_eq!(ExitReason::from_u8(0), Some(ExitReason::Return));
    assert_eq!(ExitReason::from_u8(1), Some(ExitReason::Exception));
    assert_eq!(ExitReason::from_u8(2), Some(ExitReason::Deoptimize));
    assert_eq!(ExitReason::from_u8(99), None);
}

#[test]
fn test_entry_stub_generation() {
    let stub = EntryStub::new();
    // Stub should have some code
    assert!(!stub.code().is_empty());
    assert!(stub.code_size() > 10); // Should have prologue + call + epilogue
}

#[test]
fn test_deopt_stub_generation() {
    let stub = DeoptStub::new();
    assert!(!stub.code().is_empty());
}

#[test]
fn test_exception_stub_generation() {
    let stub = ExceptionStub::new();
    assert!(!stub.code().is_empty());
}

#[test]
fn test_stub_generator() {
    let generator = StubGenerator::new();
    assert!(generator.total_size() > 0);
    assert_eq!(
        generator.total_size(),
        generator.entry.code_size() + generator.deopt.code_size() + generator.exception.code_size()
    );
}
