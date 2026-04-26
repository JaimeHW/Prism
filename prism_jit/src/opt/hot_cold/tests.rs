use super::*;

// =========================================================================
// HotColdConfig Tests
// =========================================================================

#[test]
fn test_config_default() {
    let config = HotColdConfig::default();
    assert!((config.hot_threshold - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.max_cold_regions, 16);
}

#[test]
fn test_config_aggressive() {
    let config = HotColdConfig::aggressive();
    assert!(config.hot_threshold < HotColdConfig::default().hot_threshold);
}

#[test]
fn test_config_conservative() {
    let config = HotColdConfig::conservative();
    assert!(config.hot_threshold > HotColdConfig::default().hot_threshold);
}

#[test]
fn test_config_testing() {
    let config = HotColdConfig::for_testing();
    assert_eq!(config.min_execution_count, 0);
}

// =========================================================================
// BlockTemperature Tests
// =========================================================================

#[test]
fn test_temperature_is_inline() {
    assert!(BlockTemperature::Hot.is_inline());
    assert!(BlockTemperature::Warm.is_inline());
    assert!(!BlockTemperature::Cold.is_inline());
    assert!(!BlockTemperature::Frozen.is_inline());
}

#[test]
fn test_temperature_is_out_of_line() {
    assert!(!BlockTemperature::Hot.is_out_of_line());
    assert!(!BlockTemperature::Warm.is_out_of_line());
    assert!(BlockTemperature::Cold.is_out_of_line());
    assert!(BlockTemperature::Frozen.is_out_of_line());
}

// =========================================================================
// SplitRegion Tests
// =========================================================================

#[test]
fn test_region_new() {
    let r = SplitRegion::new(BlockTemperature::Hot);
    assert!(r.is_empty());
    assert_eq!(r.block_count(), 0);
    assert_eq!(r.temperature(), BlockTemperature::Hot);
}

#[test]
fn test_region_add_block() {
    let mut r = SplitRegion::new(BlockTemperature::Hot);
    r.add_block(1, 5.0);
    r.add_block(2, 3.0);
    assert_eq!(r.block_count(), 2);
    assert!(!r.is_empty());
    assert!((r.frequency() - 8.0).abs() < f64::EPSILON);
}

// =========================================================================
// SplitLayout Tests
// =========================================================================

#[test]
fn test_layout_new() {
    let layout = SplitLayout::new();
    assert_eq!(layout.total_blocks(), 0);
    assert!(layout.hot_regions().is_empty());
    assert!(layout.cold_regions().is_empty());
}

// =========================================================================
// HotColdSplitter Tests
// =========================================================================

#[test]
fn test_splitter_new() {
    let s = HotColdSplitter::new();
    assert!((s.config.hot_threshold - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_classify_all_hot() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    for i in 0..5u32 {
        ann.set_block_freq(i, BlockFrequency::new(10.0));
    }
    let layout = s.classify_blocks(&[0, 1, 2, 3, 4], &ann);
    assert_eq!(layout.stats().hot_blocks, 5);
    assert_eq!(layout.stats().cold_blocks, 0);
    assert_eq!(layout.hot_regions().len(), 1);
    assert_eq!(layout.cold_regions().len(), 0);
}

#[test]
fn test_classify_all_cold() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    for i in 0..5u32 {
        ann.set_block_freq(i, BlockFrequency::new(0.005));
    }
    let layout = s.classify_blocks(&[0, 1, 2, 3, 4], &ann);
    assert_eq!(layout.stats().hot_blocks, 0);
    assert_eq!(layout.stats().cold_blocks, 5);
    assert_eq!(layout.hot_regions().len(), 0);
    assert_eq!(layout.cold_regions().len(), 1);
}

#[test]
fn test_classify_mixed() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(0, BlockFrequency::new(10.0));
    ann.set_block_freq(1, BlockFrequency::new(10.0));
    ann.set_block_freq(2, BlockFrequency::new(0.005));
    ann.set_block_freq(3, BlockFrequency::new(10.0));
    ann.set_block_freq(4, BlockFrequency::new(0.005));

    let layout = s.classify_blocks(&[0, 1, 2, 3, 4], &ann);
    assert_eq!(layout.stats().hot_blocks, 3);
    // Check layout order: hot first, then cold
    let order = layout.layout_order();
    assert_eq!(order.len(), 5);
}

#[test]
fn test_classify_with_warm() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(0, BlockFrequency::new(10.0)); // hot
    ann.set_block_freq(1, BlockFrequency::new(0.5)); // warm
    ann.set_block_freq(2, BlockFrequency::new(0.005)); // cold

    let layout = s.classify_blocks(&[0, 1, 2], &ann);
    assert_eq!(layout.stats().hot_blocks, 1);
    assert_eq!(layout.stats().warm_blocks, 1);
    assert_eq!(layout.stats().cold_blocks, 1);
    // Warm stays inline
    assert!(layout.is_hot(0));
    assert!(!layout.is_cold(1)); // warm is inline
    assert!(layout.is_cold(2));
}

#[test]
fn test_classify_frozen_block() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(0, BlockFrequency::new(10.0));
    ann.set_block_freq(1, BlockFrequency::new(0.0)); // frozen

    let layout = s.classify_blocks(&[0, 1], &ann);
    assert_eq!(layout.stats().frozen_blocks, 1);
    assert_eq!(layout.temperature(1), Some(BlockTemperature::Frozen));
}

#[test]
fn test_layout_order_hot_first() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(0, BlockFrequency::new(0.005)); // cold
    ann.set_block_freq(1, BlockFrequency::new(10.0)); // hot
    ann.set_block_freq(2, BlockFrequency::new(10.0)); // hot
    ann.set_block_freq(3, BlockFrequency::new(0.005)); // cold

    let layout = s.classify_blocks(&[0, 1, 2, 3], &ann);
    let order = layout.layout_order();
    // Hot blocks (1, 2) should come before cold blocks (0, 3)
    assert_eq!(order.len(), 4);
    // First two should be the hot blocks
    let hot_idx_1 = order.iter().position(|&b| b == 1).unwrap();
    let cold_idx_0 = order.iter().position(|&b| b == 0).unwrap();
    assert!(hot_idx_1 < cold_idx_0);
}

#[test]
fn test_max_cold_regions_limit() {
    let config = HotColdConfig {
        max_cold_regions: 2,
        ..HotColdConfig::for_testing()
    };
    let s = HotColdSplitter::with_config(config);
    let mut ann = BranchAnnotations::new();
    // Create alternating hot/cold to force many cold regions
    for i in 0..10u32 {
        if i % 2 == 0 {
            ann.set_block_freq(i, BlockFrequency::new(10.0));
        } else {
            ann.set_block_freq(i, BlockFrequency::new(0.005));
        }
    }
    let blocks: Vec<u32> = (0..10).collect();
    let layout = s.classify_blocks(&blocks, &ann);
    assert!(layout.cold_regions().len() <= 2);
}

#[test]
fn test_identify_cold_branches() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_branch(10, BranchProbability::from_f64(0.01)); // cold
    ann.set_branch(20, BranchProbability::from_f64(0.9)); // hot
    ann.set_branch(30, BranchProbability::from_f64(0.03)); // cold

    let cold = s.identify_cold_branches(&ann);
    assert!(cold.contains(&10));
    assert!(!cold.contains(&20));
    assert!(cold.contains(&30));
}

#[test]
fn test_empty_blocks() {
    let s = HotColdSplitter::new();
    let ann = BranchAnnotations::new();
    let layout = s.classify_blocks(&[], &ann);
    assert_eq!(layout.total_blocks(), 0);
    assert!(layout.layout_order().is_empty());
}

#[test]
fn test_single_block() {
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let mut ann = BranchAnnotations::new();
    ann.set_block_freq(0, BlockFrequency::new(5.0));
    let layout = s.classify_blocks(&[0], &ann);
    assert_eq!(layout.total_blocks(), 1);
    assert!(layout.is_hot(0));
}

#[test]
fn test_default_frequency_is_hot() {
    // Blocks without explicit frequency default to ENTRY (1.0)
    let s = HotColdSplitter::with_config(HotColdConfig::for_testing());
    let ann = BranchAnnotations::new();
    let layout = s.classify_blocks(&[0, 1, 2], &ann);
    assert_eq!(layout.stats().hot_blocks, 3);
}
