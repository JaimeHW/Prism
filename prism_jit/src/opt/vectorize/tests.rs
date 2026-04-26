    use super::*;

    #[test]
    fn test_config_default() {
        let config = VectorizeConfig::default();
        assert_eq!(config.simd_level, SimdLevel::Avx2);
        assert_eq!(config.min_trip_count, 8);
        assert!(config.enable_slp);
        assert!(config.enable_loop_vec);
    }

    #[test]
    fn test_config_sse42() {
        let config = VectorizeConfig::sse42();
        assert_eq!(config.simd_level, SimdLevel::Sse42);
        assert_eq!(config.max_vector_width, 2);
        assert!(!config.enable_gather_scatter);
    }

    #[test]
    fn test_config_avx2() {
        let config = VectorizeConfig::avx2();
        assert_eq!(config.simd_level, SimdLevel::Avx2);
        assert_eq!(config.max_vector_width, 4);
    }

    #[test]
    fn test_config_avx512() {
        let config = VectorizeConfig::avx512();
        assert_eq!(config.simd_level, SimdLevel::Avx512);
        assert_eq!(config.max_vector_width, 8);
        assert!(config.enable_gather_scatter);
    }

    #[test]
    fn test_config_aggressive() {
        let config = VectorizeConfig::aggressive();
        assert_eq!(config.simd_level, SimdLevel::Avx512);
        assert_eq!(config.max_vector_width, 16);
        assert_eq!(config.min_trip_count, 4);
        assert!(config.enable_gather_scatter);
    }

    #[test]
    fn test_stats_default() {
        let stats = VectorizeStats::default();
        assert_eq!(stats.loops_analyzed, 0);
        assert_eq!(stats.loops_vectorized, 0);
        assert_eq!(stats.success_rate(), 0.0);
    }

    #[test]
    fn test_stats_merge() {
        let mut stats1 = VectorizeStats {
            loops_analyzed: 10,
            loops_vectorized: 5,
            ..Default::default()
        };
        let stats2 = VectorizeStats {
            loops_analyzed: 20,
            loops_vectorized: 15,
            ..Default::default()
        };
        stats1.merge(&stats2);
        assert_eq!(stats1.loops_analyzed, 30);
        assert_eq!(stats1.loops_vectorized, 20);
    }

    #[test]
    fn test_stats_success_rate() {
        let stats = VectorizeStats {
            loops_analyzed: 10,
            loops_vectorized: 7,
            ..Default::default()
        };
        assert!((stats.success_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_vectorize_pass_new() {
        let pass = Vectorize::new();
        assert_eq!(pass.name(), "vectorize");
        assert_eq!(pass.loops_vectorized(), 0);
    }

    #[test]
    fn test_vectorize_pass_with_config() {
        let config = VectorizeConfig::avx512();
        let pass = Vectorize::with_config(config.clone());
        assert_eq!(pass.config().simd_level, SimdLevel::Avx512);
    }
