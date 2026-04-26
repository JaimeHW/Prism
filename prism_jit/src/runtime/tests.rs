use super::*;

#[test]
fn test_default_config() {
    let config = RuntimeConfig::default();
    assert_eq!(config.max_code_size, 64 * 1024 * 1024);
    assert_eq!(config.compiler_threads, 1);
    assert!(config.enable_osr);
}

#[test]
fn test_testing_config() {
    let config = RuntimeConfig::for_testing();
    assert_eq!(config.max_code_size, 1024 * 1024);
    assert_eq!(config.compiler_threads, 0);
    assert!(!config.enable_osr);
}
