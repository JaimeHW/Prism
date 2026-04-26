    use super::*;

    // -------------------------------------------------------------------------
    // LoopVecAnalysis Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analysis_not_vectorizable() {
        let analysis = LoopVecAnalysis::not_vectorizable(VecRejectReason::ComplexControlFlow);
        assert!(!analysis.vectorizable);
        assert!(analysis.rejection_reason.is_some());
        assert_eq!(analysis.vector_width, 1);
    }

    #[test]
    fn test_analysis_vectorizable() {
        let analysis = LoopVecAnalysis::vectorizable(4, Some(100));
        assert!(analysis.vectorizable);
        assert!(analysis.rejection_reason.is_none());
        assert_eq!(analysis.vector_width, 4);
        assert_eq!(analysis.trip_count, Some(100));
    }

    #[test]
    fn test_analysis_needs_epilog() {
        // Trip count not divisible by width
        let analysis = LoopVecAnalysis::vectorizable(4, Some(17));
        assert!(analysis.needs_epilog(4));

        // Trip count divisible by width
        let analysis = LoopVecAnalysis::vectorizable(4, Some(16));
        assert!(!analysis.needs_epilog(4));

        // Unknown trip count always needs epilog
        let analysis = LoopVecAnalysis::vectorizable(4, None);
        assert!(analysis.needs_epilog(4));
    }

    #[test]
    fn test_analysis_vector_iterations() {
        let analysis = LoopVecAnalysis::vectorizable(4, Some(17));
        assert_eq!(analysis.vector_iterations(4), Some(4));
    }

    #[test]
    fn test_analysis_epilog_iterations() {
        let analysis = LoopVecAnalysis::vectorizable(4, Some(17));
        assert_eq!(analysis.epilog_iterations(4), Some(1));

        let analysis = LoopVecAnalysis::vectorizable(4, Some(20));
        assert_eq!(analysis.epilog_iterations(4), Some(0));
    }

    // -------------------------------------------------------------------------
    // VecRejectReason Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reject_reason_description() {
        assert!(
            VecRejectReason::ComplexControlFlow
                .description()
                .contains("control flow")
        );
        assert!(
            VecRejectReason::TripCountTooLow(4)
                .description()
                .contains("4")
        );
        assert!(
            VecRejectReason::NotProfitable(0.5)
                .description()
                .contains("0.50")
        );
    }

    #[test]
    fn test_reject_reason_is_hard_blocker() {
        assert!(VecRejectReason::ComplexControlFlow.is_hard_blocker());
        assert!(VecRejectReason::ContainsCalls.is_hard_blocker());
        assert!(!VecRejectReason::TripCountTooLow(4).is_hard_blocker());
        assert!(!VecRejectReason::NotProfitable(0.5).is_hard_blocker());
    }

    // -------------------------------------------------------------------------
    // RuntimeCheck Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_runtime_check_no_alias() {
        let check = RuntimeCheck::no_alias(NodeId::new(1), NodeId::new(2));
        assert_eq!(check.kind, RuntimeCheckKind::NoAlias);
        assert_eq!(check.nodes.len(), 2);
    }

    #[test]
    fn test_runtime_check_min_trip_count() {
        let check = RuntimeCheck::min_trip_count(16);
        assert_eq!(check.kind, RuntimeCheckKind::MinTripCount(16));
    }

    #[test]
    fn test_runtime_check_alignment() {
        let check = RuntimeCheck::alignment(NodeId::new(1), 32);
        assert_eq!(check.kind, RuntimeCheckKind::Alignment(32));
        assert_eq!(check.nodes.len(), 1);
    }

    // -------------------------------------------------------------------------
    // InductionStep Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_induction_step_constant_value() {
        assert_eq!(InductionStep::Constant(5).constant_value(), Some(5));
        assert_eq!(
            InductionStep::Dynamic(NodeId::new(1)).constant_value(),
            None
        );
        assert_eq!(InductionStep::Unknown.constant_value(), None);
    }

    #[test]
    fn test_induction_step_is_unit() {
        assert!(InductionStep::Constant(1).is_unit());
        assert!(InductionStep::Constant(-1).is_unit());
        assert!(!InductionStep::Constant(2).is_unit());
        assert!(!InductionStep::Unknown.is_unit());
    }

    // -------------------------------------------------------------------------
    // ReductionKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_reduction_kind_identity() {
        assert_eq!(ReductionKind::Sum.identity(), 0);
        assert_eq!(ReductionKind::Product.identity(), 1);
        assert_eq!(ReductionKind::Min.identity(), i64::MAX);
        assert_eq!(ReductionKind::Max.identity(), i64::MIN);
        assert_eq!(ReductionKind::Or.identity(), 0);
    }

    #[test]
    fn test_reduction_kind_is_associative() {
        assert!(ReductionKind::Sum.is_associative());
        assert!(ReductionKind::Product.is_associative());
        assert!(ReductionKind::Min.is_associative());
        assert!(ReductionKind::And.is_associative());
    }

    #[test]
    fn test_reduction_kind_is_commutative() {
        assert!(ReductionKind::Sum.is_commutative());
        assert!(ReductionKind::Product.is_commutative());
        assert!(ReductionKind::Xor.is_commutative());
    }

    // -------------------------------------------------------------------------
    // LoopVectorizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_vectorizer_new() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        assert_eq!(vectorizer.stats().loops_analyzed, 0);
    }

    #[test]
    fn test_vectorizer_with_min_trip_count() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default()).with_min_trip_count(16);
        assert_eq!(vectorizer.min_trip_count, 16);
    }

    #[test]
    fn test_vectorizer_with_target_width() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default()).with_target_width(8);
        assert_eq!(vectorizer.target_width, 8);
    }

    #[test]
    fn test_vectorizer_default() {
        let vectorizer = LoopVectorizer::default();
        assert!(vectorizer.target_width >= 2);
    }

    #[test]
    fn test_vectorizer_analyze_trip_count_too_low() {
        let mut vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let graph = Graph::new();
        let deps = DependenceGraph::new(1);

        let analysis = vectorizer.analyze(&graph, &[], &deps, Some(2));
        assert!(!analysis.vectorizable);
        assert!(matches!(
            analysis.rejection_reason,
            Some(VecRejectReason::TripCountTooLow(_))
        ));
    }

    #[test]
    fn test_vectorizer_stats() {
        let mut vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let graph = Graph::new();
        let deps = DependenceGraph::new(1);

        let _ = vectorizer.analyze(&graph, &[], &deps, Some(2));
        assert_eq!(vectorizer.stats().loops_analyzed, 1);
    }

    // -------------------------------------------------------------------------
    // LoopVecResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_result_failure() {
        let result = LoopVecResult::failure();
        assert!(!result.success);
        assert_eq!(result.vector_width, 1);
        assert!((result.speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_result_success() {
        let result = LoopVecResult::success(4, Some(100));
        assert!(result.success);
        assert_eq!(result.vector_width, 4);
        assert!(result.speedup > 1.0);
    }

    #[test]
    fn test_result_success_unknown_trip_count() {
        let result = LoopVecResult::success(4, None);
        assert!(result.success);
        // Lower efficiency without known trip count
        assert!(result.speedup < 4.0);
    }

    // -------------------------------------------------------------------------
    // VectorTransformer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_transformer_new_width_2() {
        let mut graph = Graph::new();
        let transformer = VectorTransformer::new(&mut graph, 2);
        assert_eq!(transformer.width, 2);
        assert_eq!(transformer.int_vop, VectorOp::V2I64);
        assert_eq!(transformer.float_vop, VectorOp::V2F64);
    }

    #[test]
    fn test_transformer_new_width_4() {
        let mut graph = Graph::new();
        let transformer = VectorTransformer::new(&mut graph, 4);
        assert_eq!(transformer.width, 4);
        assert_eq!(transformer.int_vop, VectorOp::V4I64);
        assert_eq!(transformer.float_vop, VectorOp::V4F64);
    }

    #[test]
    fn test_transformer_new_width_8() {
        let mut graph = Graph::new();
        let transformer = VectorTransformer::new(&mut graph, 8);
        assert_eq!(transformer.width, 8);
        assert_eq!(transformer.int_vop, VectorOp::V8I64);
        assert_eq!(transformer.float_vop, VectorOp::V8F64);
    }

    #[test]
    fn test_transformer_widen_induction_unit_step() {
        let mut graph = Graph::new();
        let init = graph.const_int(0);
        let phi = graph.const_int(0); // Placeholder for phi node

        let induction = Induction {
            phi,
            init,
            step: InductionStep::Constant(1),
            is_primary: true,
        };

        let mut transformer = VectorTransformer::new(&mut graph, 4);
        transformer.widen_induction(&induction);

        // Should have created vector nodes
        assert!(transformer.scalar_to_vector.contains_key(&phi));
    }

    #[test]
    fn test_transformer_widen_induction_stride_2() {
        let mut graph = Graph::new();
        let init = graph.const_int(0);
        let phi = graph.const_int(0);

        let induction = Induction {
            phi,
            init,
            step: InductionStep::Constant(2),
            is_primary: true,
        };

        let mut transformer = VectorTransformer::new(&mut graph, 4);
        transformer.widen_induction(&induction);

        assert!(transformer.scalar_to_vector.contains_key(&phi));

        // Vector should contain <0, 2, 4, 6> offsets added to broadcast(init)
        let widened = transformer.scalar_to_vector[&phi];
        assert!(graph.get(widened).is_some());
    }

    #[test]
    fn test_transformer_dynamic_step_skipped() {
        let mut graph = Graph::new();
        let init = graph.const_int(0);
        let phi = graph.const_int(0);
        let step_node = graph.const_int(1);

        let induction = Induction {
            phi,
            init,
            step: InductionStep::Dynamic(step_node),
            is_primary: true,
        };

        let mut transformer = VectorTransformer::new(&mut graph, 4);
        transformer.widen_induction(&induction);

        // Dynamic step should be skipped
        assert!(!transformer.scalar_to_vector.contains_key(&phi));
    }

    #[test]
    fn test_transformer_create_lane_offsets() {
        let mut graph = Graph::new();
        let mut transformer = VectorTransformer::new(&mut graph, 4);

        let offsets = transformer.create_lane_offsets(vec![0, 2, 4, 6]);

        // Should have created insert nodes for non-zero offsets
        assert!(graph.get(offsets).is_some());
    }

    #[test]
    fn test_transformer_generate_epilog() {
        let mut graph = Graph::new();
        let mut transformer = VectorTransformer::new(&mut graph, 4);

        transformer.generate_epilog();

        // For now, epilog is a placeholder
        assert!(transformer.epilog_loop.is_none());
    }

    #[test]
    fn test_vectorize_not_vectorizable() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let mut graph = Graph::new();

        let analysis = LoopVecAnalysis::not_vectorizable(VecRejectReason::ComplexControlFlow);
        let result = vectorizer.vectorize(&mut graph, &analysis);

        assert!(!result.success);
        assert_eq!(result.vector_width, 1);
    }

    #[test]
    fn test_vectorize_simple_loop() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let mut graph = Graph::new();

        let analysis = LoopVecAnalysis::vectorizable(4, Some(100));
        let result = vectorizer.vectorize(&mut graph, &analysis);

        assert!(result.success);
        assert_eq!(result.vector_width, 4);
        assert!(result.speedup > 1.0);
    }

    #[test]
    fn test_vectorize_with_inductions() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let mut graph = Graph::new();

        let init = graph.const_int(0);
        let phi = graph.const_int(0);

        let mut analysis = LoopVecAnalysis::vectorizable(4, Some(100));
        analysis.inductions.push(Induction {
            phi,
            init,
            step: InductionStep::Constant(1),
            is_primary: true,
        });

        let result = vectorizer.vectorize(&mut graph, &analysis);

        assert!(result.success);
        // Graph should have new vector nodes from widening
        assert!(graph.len() > 3); // start + end + constants + vector ops
    }

    #[test]
    fn test_vectorize_needs_epilog() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let mut graph = Graph::new();

        // Trip count 17 not divisible by width 4
        let analysis = LoopVecAnalysis::vectorizable(4, Some(17));
        let result = vectorizer.vectorize(&mut graph, &analysis);

        assert!(result.success);
        // Epilog generation was called
    }

    #[test]
    fn test_vectorize_no_epilog_needed() {
        let vectorizer = LoopVectorizer::new(VectorCostModel::default());
        let mut graph = Graph::new();

        // Trip count 16 divisible by width 4
        let analysis = LoopVecAnalysis::vectorizable(4, Some(16));
        let result = vectorizer.vectorize(&mut graph, &analysis);

        assert!(result.success);
    }
