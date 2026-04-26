    use super::*;

    // =========================================================================
    // ReductionKind Tests
    // =========================================================================

    #[test]
    fn test_reduction_kind_identity_int() {
        assert_eq!(ReductionKind::Sum.identity_int(), 0);
        assert_eq!(ReductionKind::Product.identity_int(), 1);
        assert_eq!(ReductionKind::Min.identity_int(), i64::MAX);
        assert_eq!(ReductionKind::Max.identity_int(), i64::MIN);
        assert_eq!(ReductionKind::And.identity_int(), -1);
        assert_eq!(ReductionKind::Or.identity_int(), 0);
        assert_eq!(ReductionKind::Xor.identity_int(), 0);
    }

    #[test]
    fn test_reduction_kind_identity_float() {
        assert_eq!(ReductionKind::Sum.identity_float(), 0.0);
        assert_eq!(ReductionKind::Product.identity_float(), 1.0);
        assert_eq!(ReductionKind::Min.identity_float(), f64::INFINITY);
        assert_eq!(ReductionKind::Max.identity_float(), f64::NEG_INFINITY);
        assert!(ReductionKind::And.identity_float().is_nan());
        assert!(ReductionKind::Or.identity_float().is_nan());
        assert!(ReductionKind::Xor.identity_float().is_nan());
    }

    #[test]
    fn test_reduction_kind_associative() {
        assert!(ReductionKind::Sum.is_associative());
        assert!(ReductionKind::Product.is_associative());
        assert!(ReductionKind::Min.is_associative());
        assert!(ReductionKind::Max.is_associative());
        assert!(ReductionKind::And.is_associative());
        assert!(ReductionKind::Or.is_associative());
        assert!(ReductionKind::Xor.is_associative());
    }

    #[test]
    fn test_reduction_kind_commutative() {
        assert!(ReductionKind::Sum.is_commutative());
        assert!(ReductionKind::Product.is_commutative());
        assert!(ReductionKind::Min.is_commutative());
        assert!(ReductionKind::Max.is_commutative());
        assert!(ReductionKind::And.is_commutative());
        assert!(ReductionKind::Or.is_commutative());
        assert!(ReductionKind::Xor.is_commutative());
    }

    #[test]
    fn test_reduction_kind_supports_integer() {
        assert!(ReductionKind::Sum.supports_integer());
        assert!(ReductionKind::Product.supports_integer());
        assert!(ReductionKind::Min.supports_integer());
        assert!(ReductionKind::Max.supports_integer());
        assert!(ReductionKind::And.supports_integer());
        assert!(ReductionKind::Or.supports_integer());
        assert!(ReductionKind::Xor.supports_integer());
    }

    #[test]
    fn test_reduction_kind_supports_float() {
        assert!(ReductionKind::Sum.supports_float());
        assert!(ReductionKind::Product.supports_float());
        assert!(ReductionKind::Min.supports_float());
        assert!(ReductionKind::Max.supports_float());
        assert!(!ReductionKind::And.supports_float());
        assert!(!ReductionKind::Or.supports_float());
        assert!(!ReductionKind::Xor.supports_float());
    }

    #[test]
    fn test_reduction_kind_horizontal_intrinsic() {
        assert_eq!(ReductionKind::Sum.horizontal_intrinsic(), "horizontal_add");
        assert_eq!(
            ReductionKind::Product.horizontal_intrinsic(),
            "horizontal_mul"
        );
        assert_eq!(ReductionKind::Min.horizontal_intrinsic(), "horizontal_min");
        assert_eq!(ReductionKind::Max.horizontal_intrinsic(), "horizontal_max");
        assert_eq!(ReductionKind::And.horizontal_intrinsic(), "horizontal_and");
        assert_eq!(ReductionKind::Or.horizontal_intrinsic(), "horizontal_or");
        assert_eq!(ReductionKind::Xor.horizontal_intrinsic(), "horizontal_xor");
    }

    // =========================================================================
    // Reduction Tests
    // =========================================================================

    fn make_test_reduction(kind: ReductionKind, is_float: bool) -> Reduction {
        Reduction::new(
            NodeId::new(0),
            NodeId::new(1),
            kind,
            NodeId::new(2),
            NodeId::new(3),
            is_float,
        )
    }

    #[test]
    fn test_reduction_new() {
        let r = make_test_reduction(ReductionKind::Sum, false);
        assert_eq!(r.phi, NodeId::new(0));
        assert_eq!(r.op_node, NodeId::new(1));
        assert_eq!(r.kind, ReductionKind::Sum);
        assert_eq!(r.init, NodeId::new(2));
        assert_eq!(r.value_operand, NodeId::new(3));
        assert!(!r.is_float);
    }

    #[test]
    fn test_reduction_identity_int() {
        assert_eq!(
            make_test_reduction(ReductionKind::Sum, false).identity_int(),
            0
        );
        assert_eq!(
            make_test_reduction(ReductionKind::Product, false).identity_int(),
            1
        );
        assert_eq!(
            make_test_reduction(ReductionKind::And, false).identity_int(),
            -1
        );
    }

    #[test]
    fn test_reduction_identity_float() {
        assert_eq!(
            make_test_reduction(ReductionKind::Sum, true).identity_float(),
            0.0
        );
        assert_eq!(
            make_test_reduction(ReductionKind::Product, true).identity_float(),
            1.0
        );
    }

    #[test]
    fn test_reduction_is_sum() {
        assert!(make_test_reduction(ReductionKind::Sum, false).is_sum());
        assert!(!make_test_reduction(ReductionKind::Product, false).is_sum());
        assert!(!make_test_reduction(ReductionKind::Min, false).is_sum());
    }

    #[test]
    fn test_reduction_is_product() {
        assert!(make_test_reduction(ReductionKind::Product, false).is_product());
        assert!(!make_test_reduction(ReductionKind::Sum, false).is_product());
        assert!(!make_test_reduction(ReductionKind::Max, false).is_product());
    }

    #[test]
    fn test_reduction_is_vectorizable() {
        assert!(make_test_reduction(ReductionKind::Sum, false).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::Sum, true).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::Product, false).is_vectorizable());
        assert!(make_test_reduction(ReductionKind::And, false).is_vectorizable());
    }

    // =========================================================================
    // ReductionAnalysis Tests
    // =========================================================================

    #[test]
    fn test_reduction_analysis_empty() {
        let analysis = ReductionAnalysis::empty();
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
        assert!(!analysis.is_reduction(0, NodeId::new(0)));
    }

    #[test]
    fn test_reduction_analysis_with_capacity() {
        let analysis = ReductionAnalysis::with_capacity(5);
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
    }

    #[test]
    fn test_reduction_analysis_add_loop() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, false),
        );

        analysis.add_loop(loop_reds);

        assert_eq!(analysis.total(), 2);
        assert_eq!(analysis.num_loops(), 1);
        assert!(analysis.is_reduction(0, NodeId::new(0)));
        assert!(analysis.is_reduction(0, NodeId::new(1)));
        assert!(!analysis.is_reduction(0, NodeId::new(2)));
        assert!(!analysis.is_reduction(1, NodeId::new(0)));
    }

    #[test]
    fn test_reduction_analysis_get() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop_reds);

        assert!(analysis.get(0).is_some());
        assert!(analysis.get(1).is_none());
    }

    #[test]
    fn test_reduction_analysis_get_reduction() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(5),
            make_test_reduction(ReductionKind::Max, false),
        );
        analysis.add_loop(loop_reds);

        let red = analysis.get_reduction(0, NodeId::new(5));
        assert!(red.is_some());
        assert_eq!(red.unwrap().kind, ReductionKind::Max);

        assert!(analysis.get_reduction(0, NodeId::new(0)).is_none());
        assert!(analysis.get_reduction(1, NodeId::new(5)).is_none());
    }

    #[test]
    fn test_reduction_analysis_iter_all() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop0 = FxHashMap::default();
        loop0.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop0);

        let mut loop1 = FxHashMap::default();
        loop1.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, true),
        );
        analysis.add_loop(loop1);

        let all: Vec<_> = analysis.iter_all().collect();
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_sum() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Sum, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Product, false),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_sum(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_product() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Product, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_product(), 2);
    }

    #[test]
    fn test_reduction_analysis_count_float() {
        let mut analysis = ReductionAnalysis::empty();

        let mut loop_reds = FxHashMap::default();
        loop_reds.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        loop_reds.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Sum, true),
        );
        loop_reds.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Product, true),
        );
        analysis.add_loop(loop_reds);

        assert_eq!(analysis.count_float(), 2);
    }

    #[test]
    fn test_reduction_analysis_default() {
        let analysis = ReductionAnalysis::default();
        assert_eq!(analysis.total(), 0);
        assert_eq!(analysis.num_loops(), 0);
    }

    #[test]
    fn test_reduction_analysis_multiple_loops() {
        let mut analysis = ReductionAnalysis::with_capacity(3);

        let mut loop0 = FxHashMap::default();
        loop0.insert(
            NodeId::new(0),
            make_test_reduction(ReductionKind::Sum, false),
        );
        analysis.add_loop(loop0);

        let mut loop1 = FxHashMap::default();
        loop1.insert(
            NodeId::new(1),
            make_test_reduction(ReductionKind::Product, false),
        );
        loop1.insert(
            NodeId::new(2),
            make_test_reduction(ReductionKind::Max, false),
        );
        analysis.add_loop(loop1);

        let loop2 = FxHashMap::default(); // Empty loop
        analysis.add_loop(loop2);

        assert_eq!(analysis.num_loops(), 3);
        assert_eq!(analysis.total(), 3);
        assert!(analysis.is_reduction(0, NodeId::new(0)));
        assert!(analysis.is_reduction(1, NodeId::new(1)));
        assert!(analysis.is_reduction(1, NodeId::new(2)));
    }
