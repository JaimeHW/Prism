    use super::*;

    // -------------------------------------------------------------------------
    // SimdLevel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simd_level_ordering() {
        assert!(SimdLevel::Sse42 < SimdLevel::Avx);
        assert!(SimdLevel::Avx < SimdLevel::Avx2);
        assert!(SimdLevel::Avx2 < SimdLevel::Avx512);
    }

    #[test]
    fn test_simd_level_max_vector_bytes() {
        assert_eq!(SimdLevel::Sse42.max_vector_bytes(), 16);
        assert_eq!(SimdLevel::Avx.max_vector_bytes(), 32);
        assert_eq!(SimdLevel::Avx2.max_vector_bytes(), 32);
        assert_eq!(SimdLevel::Avx512.max_vector_bytes(), 64);
    }

    #[test]
    fn test_simd_level_max_vector_bits() {
        assert_eq!(SimdLevel::Sse42.max_vector_bits(), 128);
        assert_eq!(SimdLevel::Avx.max_vector_bits(), 256);
        assert_eq!(SimdLevel::Avx512.max_vector_bits(), 512);
    }

    #[test]
    fn test_simd_level_max_lanes() {
        assert_eq!(SimdLevel::Sse42.max_lanes(ValueType::Int64), 2);
        assert_eq!(SimdLevel::Avx2.max_lanes(ValueType::Int64), 4);
        assert_eq!(SimdLevel::Avx512.max_lanes(ValueType::Int64), 8);
        assert_eq!(SimdLevel::Sse42.max_lanes(ValueType::Float64), 2);
        assert_eq!(SimdLevel::Avx2.max_lanes(ValueType::Float64), 4);
    }

    #[test]
    fn test_simd_level_has_fma() {
        assert!(!SimdLevel::Sse42.has_fma());
        assert!(!SimdLevel::Avx.has_fma());
        assert!(SimdLevel::Avx2.has_fma());
        assert!(SimdLevel::Avx512.has_fma());
    }

    #[test]
    fn test_simd_level_has_gather() {
        assert!(!SimdLevel::Sse42.has_gather());
        assert!(!SimdLevel::Avx.has_gather());
        assert!(SimdLevel::Avx2.has_gather());
        assert!(SimdLevel::Avx512.has_gather());
    }

    #[test]
    fn test_simd_level_has_scatter() {
        assert!(!SimdLevel::Sse42.has_scatter());
        assert!(!SimdLevel::Avx.has_scatter());
        assert!(!SimdLevel::Avx2.has_scatter());
        assert!(SimdLevel::Avx512.has_scatter());
    }

    #[test]
    fn test_simd_level_has_masking() {
        assert!(!SimdLevel::Sse42.has_masking());
        assert!(!SimdLevel::Avx2.has_masking());
        assert!(SimdLevel::Avx512.has_masking());
    }

    #[test]
    fn test_simd_level_best_vector_op() {
        let vop = SimdLevel::Avx2.best_vector_op(ValueType::Int64);
        assert_eq!(vop.lanes, 4);
        assert_eq!(vop.element, ValueType::Int64);
    }

    #[test]
    fn test_simd_level_default() {
        assert_eq!(SimdLevel::default(), SimdLevel::Avx2);
    }

    // -------------------------------------------------------------------------
    // OpCost Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_op_cost_new() {
        let cost = OpCost::new(5, 2.0);
        assert_eq!(cost.latency, 5);
        assert!((cost.throughput - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_op_cost_free() {
        let cost = OpCost::free();
        assert_eq!(cost.latency, 0);
        assert!((cost.throughput - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_op_cost_presets() {
        assert!(
            OpCost::trivial().latency < OpCost::alu().latency
                || OpCost::trivial().throughput < OpCost::alu().throughput
        );
        assert!(OpCost::alu().total_cost() < OpCost::mul().total_cost());
        assert!(OpCost::mul().total_cost() < OpCost::div().total_cost());
    }

    #[test]
    fn test_op_cost_scale() {
        let cost = OpCost::alu();
        let scaled = cost.scale(2.0);
        assert!((scaled.throughput - cost.throughput * 2.0).abs() < 0.001);
        assert_eq!(scaled.latency, cost.latency);
    }

    #[test]
    fn test_op_cost_with_penalty() {
        let cost = OpCost::load();
        let penalized = cost.with_penalty(0.5);
        assert!((penalized.throughput - cost.throughput - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_op_cost_chain() {
        let c1 = OpCost::new(3, 1.0);
        let c2 = OpCost::new(2, 0.5);
        let chained = c1.chain(c2);
        assert_eq!(chained.latency, 5);
        assert!((chained.throughput - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_op_cost_total_cost() {
        let cost = OpCost::new(5, 2.0);
        // throughput + latency * 0.2
        let expected = 2.0 + 5.0 * 0.2;
        assert!((cost.total_cost() - expected).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // VectorCostModel Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cost_model_new() {
        let model = VectorCostModel::new(SimdLevel::Avx2);
        assert_eq!(model.level(), SimdLevel::Avx2);
    }

    #[test]
    fn test_cost_model_arith_cost() {
        let model = VectorCostModel::new(SimdLevel::Avx2);
        let vop = VectorOp::V4I64;

        let add_cost = model.arith_cost(VectorArithKind::Add, vop);
        let div_cost = model.arith_cost(VectorArithKind::Div, vop);

        // Division should be more expensive than addition
        assert!(div_cost.total_cost() > add_cost.total_cost());
    }

    #[test]
    fn test_cost_model_fma_cost_with_native() {
        let model = VectorCostModel::new(SimdLevel::Avx2); // Has FMA
        let fma_cost = model.fma_cost(VectorOp::V4F64);

        // Native FMA should be reasonably cheap
        assert!(fma_cost.latency <= 5);
    }

    #[test]
    fn test_cost_model_fma_cost_emulated() {
        let model = VectorCostModel::new(SimdLevel::Sse42); // No FMA
        let fma_cost = model.fma_cost(VectorOp::V2F64);

        // Emulated FMA = mul + add, should be more expensive
        let mul_cost = model.arith_cost(VectorArithKind::Mul, VectorOp::V2F64);
        assert!(fma_cost.throughput >= mul_cost.throughput);
    }

    #[test]
    fn test_cost_model_memory_cost_aligned() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let aligned = model.memory_cost(VectorMemoryKind::LoadAligned, VectorOp::V4I64, true);
        let unaligned = model.memory_cost(VectorMemoryKind::LoadUnaligned, VectorOp::V4I64, false);

        // Unaligned should have penalty
        assert!(unaligned.total_cost() > aligned.total_cost());
    }

    #[test]
    fn test_cost_model_gather_cost() {
        let model_avx2 = VectorCostModel::new(SimdLevel::Avx2);
        let model_sse = VectorCostModel::new(SimdLevel::Sse42);

        let gather_avx2 = model_avx2.memory_cost(VectorMemoryKind::Gather, VectorOp::V4I64, true);
        let gather_sse = model_sse.memory_cost(VectorMemoryKind::Gather, VectorOp::V2I64, true);

        // SSE emulated gather should be more expensive
        assert!(gather_sse.total_cost() > gather_avx2.total_cost());
    }

    #[test]
    fn test_cost_model_scatter_cost() {
        let model_512 = VectorCostModel::new(SimdLevel::Avx512);
        let model_avx2 = VectorCostModel::new(SimdLevel::Avx2);

        let scatter_512 = model_512.memory_cost(VectorMemoryKind::Scatter, VectorOp::V8I64, true);
        let scatter_avx2 = model_avx2.memory_cost(VectorMemoryKind::Scatter, VectorOp::V4I64, true);

        // AVX2 emulated scatter should be more expensive
        assert!(scatter_avx2.total_cost() > scatter_512.total_cost());
    }

    #[test]
    fn test_cost_model_extract_cost() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let extract_0 = model.extract_cost(VectorOp::V4I64, 0);
        let extract_3 = model.extract_cost(VectorOp::V4I64, 3);

        // Lane 0 extraction is cheapest
        assert!(extract_0.total_cost() <= extract_3.total_cost());
    }

    #[test]
    fn test_cost_model_hadd_cost() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let hadd_2 = model.hadd_cost(VectorOp::V2I64);
        let hadd_4 = model.hadd_cost(VectorOp::V4I64);
        let hadd_8 = model.hadd_cost(VectorOp::V8I64);

        // More lanes = more reduction steps = more expensive
        assert!(hadd_2.total_cost() < hadd_4.total_cost());
        assert!(hadd_4.total_cost() < hadd_8.total_cost());
    }

    #[test]
    fn test_cost_model_blend_cost() {
        let model_512 = VectorCostModel::new(SimdLevel::Avx512);
        let model_avx2 = VectorCostModel::new(SimdLevel::Avx2);

        let blend_512 = model_512.blend_cost(VectorOp::V8I64);
        let blend_avx2 = model_avx2.blend_cost(VectorOp::V4I64);

        // AVX-512 masking makes blend cheaper
        assert!(blend_512.total_cost() <= blend_avx2.total_cost());
    }

    #[test]
    fn test_cost_model_scalar_costs() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let add = model.scalar_arith_cost(ArithOp::Add);
        let div = model.scalar_arith_cost(ArithOp::TrueDiv);
        let pow = model.scalar_arith_cost(ArithOp::Pow);

        assert!(add.total_cost() < div.total_cost());
        assert!(div.total_cost() < pow.total_cost());
    }

    #[test]
    fn test_cost_model_estimate_vector_cost() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let ops = vec![
            (VectorArithKind::Add, VectorOp::V4I64),
            (VectorArithKind::Mul, VectorOp::V4I64),
        ];

        let cost = model.estimate_vector_cost(&ops, 2, 1, true);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_model_estimate_scalar_cost() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        let ops = vec![ArithOp::Add, ArithOp::Mul];
        let cost = model.estimate_scalar_cost(&ops, 2, 1);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_cost_model_is_profitable_yes() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        // Scalar: 4 cycles, Vector: 2 cycles for 4 elements = 0.5/elem
        // Speedup = 4 / 0.5 = 8x
        assert!(model.is_profitable(4.0, 2.0, 4, Some(100)));
    }

    #[test]
    fn test_cost_model_is_profitable_no_small_speedup() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        // Scalar: 1 cycle, Vector: 3.5 cycles for 4 elements = 0.875/elem
        // Speedup = 1 / 0.875 = 1.14x (below threshold)
        assert!(!model.is_profitable(1.0, 3.5, 4, Some(100)));
    }

    #[test]
    fn test_cost_model_is_profitable_no_small_trip_count() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        // Good speedup but trip count < vector width
        assert!(!model.is_profitable(4.0, 2.0, 4, Some(2)));
    }

    #[test]
    fn test_cost_model_best_vector_width() {
        let model = VectorCostModel::new(SimdLevel::Avx2);
        assert_eq!(model.best_vector_width(ValueType::Int64), 4);

        let model_512 = VectorCostModel::new(SimdLevel::Avx512);
        assert_eq!(model_512.best_vector_width(ValueType::Int64), 8);
    }

    #[test]
    fn test_cost_model_default() {
        let model = VectorCostModel::default();
        assert_eq!(model.level(), SimdLevel::Avx2);
    }

    #[test]
    fn test_cost_model_debug() {
        let model = VectorCostModel::new(SimdLevel::Avx512);
        let debug_str = format!("{:?}", model);
        assert!(debug_str.contains("VectorCostModel"));
        assert!(debug_str.contains("Avx512"));
    }

    // -------------------------------------------------------------------------
    // CostAnalysis Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_cost_analysis_new() {
        let analysis = CostAnalysis::new(8.0, 4.0, 4, Some(100));

        assert!((analysis.scalar_cost - 8.0).abs() < 0.001);
        assert!((analysis.vector_cost - 4.0).abs() < 0.001);
        assert_eq!(analysis.vector_width, 4);
        assert_eq!(analysis.trip_count, Some(100));
    }

    #[test]
    fn test_cost_analysis_speedup() {
        let analysis = CostAnalysis::new(8.0, 4.0, 4, Some(100));

        // Vector cost per scalar iter = 4.0 / 4 = 1.0
        // Speedup = 8.0 / 1.0 = 8.0
        assert!((analysis.speedup - 8.0).abs() < 0.001);
        assert!(analysis.profitable);
    }

    #[test]
    fn test_cost_analysis_vector_cost_per_iter() {
        let analysis = CostAnalysis::new(8.0, 4.0, 4, Some(100));
        assert!((analysis.vector_cost_per_iter() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_analysis_savings() {
        let analysis = CostAnalysis::new(8.0, 4.0, 4, Some(100));

        // Savings per iter = 8.0 - 1.0 = 7.0
        assert!((analysis.savings_per_iter() - 7.0).abs() < 0.001);

        // Total savings = 7.0 * 100 = 700.0
        assert!((analysis.total_savings(100) - 700.0).abs() < 0.001);
    }

    #[test]
    fn test_cost_analysis_not_profitable() {
        // Vector cost too high relative to scalar
        let analysis = CostAnalysis::new(1.0, 4.0, 4, Some(100));

        // Speedup = 1.0 / 1.0 = 1.0 (below 1.25 threshold)
        assert!(analysis.speedup <= 1.25);
        assert!(!analysis.profitable);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vectorization_profitability_comparison() {
        let model = VectorCostModel::new(SimdLevel::Avx2);

        // Simulate a[i] = b[i] + c[i] loop
        let scalar_ops = vec![ArithOp::Add];
        let vector_ops = vec![(VectorArithKind::Add, VectorOp::V4I64)];

        let scalar_cost = model.estimate_scalar_cost(&scalar_ops, 2, 1);
        let vector_cost = model.estimate_vector_cost(&vector_ops, 2, 1, true);

        // Vector should be profitable for large enough trip count
        assert!(model.is_profitable(scalar_cost, vector_cost, 4, Some(1000),));
    }

    #[test]
    fn test_different_simd_levels_profitability() {
        let model_sse = VectorCostModel::new(SimdLevel::Sse42);
        let model_avx2 = VectorCostModel::new(SimdLevel::Avx2);
        let model_512 = VectorCostModel::new(SimdLevel::Avx512);

        // More capable SIMD = better vectorization
        assert!(
            model_avx2.best_vector_width(ValueType::Int64)
                > model_sse.best_vector_width(ValueType::Int64)
        );
        assert!(
            model_512.best_vector_width(ValueType::Int64)
                > model_avx2.best_vector_width(ValueType::Int64)
        );
    }
