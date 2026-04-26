use super::*;

#[test]
fn test_module_exports() {
    // Verify all public types are accessible
    let _ = GeneratorState::Created;
    let header = GeneratorHeader::new();
    assert_eq!(header.state(), GeneratorState::Created);
}
