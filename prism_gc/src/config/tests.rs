use super::*;

#[test]
fn test_default_config_is_valid() {
    assert!(GcConfig::default().validate().is_ok());
}

#[test]
fn test_preset_configs_are_valid() {
    assert!(GcConfig::low_memory().validate().is_ok());
    assert!(GcConfig::high_throughput().validate().is_ok());
    assert!(GcConfig::low_latency().validate().is_ok());
}

#[test]
fn test_invalid_nursery_size() {
    let config = GcConfig {
        nursery_size: 1024,
        ..Default::default()
    };
    assert_eq!(config.validate(), Err(ConfigError::NurseryTooSmall));
}

#[test]
fn test_invalid_card_size() {
    let config = GcConfig {
        card_size: 100, // Not power of two
        ..Default::default()
    };
    assert_eq!(config.validate(), Err(ConfigError::InvalidCardSize));
}
