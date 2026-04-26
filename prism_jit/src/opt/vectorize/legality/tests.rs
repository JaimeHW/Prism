    use super::*;
    use crate::ir::operators::ArithOp;

    // -------------------------------------------------------------------------
    // LegalityResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_legality_result_legal() {
        let result = LegalityResult::legal(8);
        assert!(result.legal);
        assert!(result.primary_reason.is_none());
        assert!(result.violations.is_empty());
        assert_eq!(result.max_legal_width, 8);
    }

    #[test]
    fn test_legality_result_illegal() {
        let result = LegalityResult::illegal(ViolationKind::ComplexControlFlow);
        assert!(!result.legal);
        assert!(result.primary_reason.is_some());
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.max_legal_width, 1);
    }

    #[test]
    fn test_legality_result_add_violation() {
        let mut result = LegalityResult::legal(8);
        assert!(result.legal);

        result.add_violation(NodeId::new(1), ViolationKind::MayThrow);
        assert!(!result.legal);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn test_legality_result_mark_scalar() {
        let mut result = LegalityResult::legal(8);
        result.mark_scalar(NodeId::new(1));
        result.mark_scalar(NodeId::new(1)); // Duplicate should not add

        assert_eq!(result.scalar_ops.len(), 1);
    }

    #[test]
    fn test_legality_result_mark_vectorizable() {
        let mut result = LegalityResult::legal(8);
        result.mark_vectorizable(NodeId::new(1), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(2), VectorOp::V4F64);

        assert_eq!(result.vectorizable_ops.len(), 2);
    }

    #[test]
    fn test_legality_result_vectorizable_percentage() {
        let mut result = LegalityResult::legal(8);

        // 3 vectorizable, 1 scalar = 75%
        result.mark_vectorizable(NodeId::new(1), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(2), VectorOp::V4I64);
        result.mark_vectorizable(NodeId::new(3), VectorOp::V4I64);
        result.mark_scalar(NodeId::new(4));

        assert!((result.vectorizable_percentage() - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_legality_result_vectorizable_percentage_empty() {
        let result = LegalityResult::legal(8);
        assert!((result.vectorizable_percentage() - 0.0).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // ViolationKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_violation_kind_is_hard_blocker() {
        assert!(
            ViolationKind::BackwardDependence(NodeId::new(1), NodeId::new(2)).is_hard_blocker()
        );
        assert!(ViolationKind::ComplexControlFlow.is_hard_blocker());
        assert!(ViolationKind::MayThrow.is_hard_blocker());
        assert!(ViolationKind::MultipleExits.is_hard_blocker());

        assert!(!ViolationKind::NonContiguousAccess.is_hard_blocker());
        assert!(!ViolationKind::UnknownBounds.is_hard_blocker());
    }

    #[test]
    fn test_violation_kind_resolvable_with_runtime_check() {
        assert!(
            ViolationKind::UnknownDependence(NodeId::new(1), NodeId::new(2))
                .resolvable_with_runtime_check()
        );
        assert!(ViolationKind::NonContiguousAccess.resolvable_with_runtime_check());
        assert!(ViolationKind::UnknownBounds.resolvable_with_runtime_check());

        assert!(
            !ViolationKind::BackwardDependence(NodeId::new(1), NodeId::new(2))
                .resolvable_with_runtime_check()
        );
        assert!(!ViolationKind::MayThrow.resolvable_with_runtime_check());
    }

    #[test]
    fn test_violation_kind_description() {
        let desc = ViolationKind::ComplexControlFlow.description();
        assert!(desc.contains("Complex control flow"));

        let desc = ViolationKind::TripCountTooSmall(4).description();
        assert!(desc.contains("4"));
    }

    // -------------------------------------------------------------------------
    // Widening Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_widening_creation() {
        let widening = Widening {
            scalar: NodeId::new(1),
            vector_type: VectorOp::V4I64,
            kind: WideningKind::Broadcast,
        };

        assert_eq!(widening.scalar, NodeId::new(1));
        assert_eq!(widening.vector_type.lanes, 4);
        assert_eq!(widening.kind, WideningKind::Broadcast);
    }

    // -------------------------------------------------------------------------
    // LegalityAnalyzer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyzer_new() {
        let analyzer = LegalityAnalyzer::new(8);
        assert_eq!(analyzer.max_width, 8);
        assert!(!analyzer.allow_gather_scatter);
        assert!(analyzer.allow_partial);
    }

    #[test]
    fn test_analyzer_with_gather_scatter() {
        let analyzer = LegalityAnalyzer::new(8).with_gather_scatter();
        assert!(analyzer.allow_gather_scatter);
    }

    #[test]
    fn test_analyzer_without_partial() {
        let analyzer = LegalityAnalyzer::new(8).without_partial();
        assert!(!analyzer.allow_partial);
    }

    #[test]
    fn test_is_vectorizable_op() {
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::IntOp(
            ArithOp::Add
        )));
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::FloatOp(
            ArithOp::Mul
        )));
        assert!(LegalityAnalyzer::is_vectorizable_op(&Operator::Memory(
            MemoryOp::Load
        )));

        assert!(!LegalityAnalyzer::is_vectorizable_op(&Operator::Call(
            CallKind::Direct
        )));
        assert!(!LegalityAnalyzer::is_vectorizable_op(&Operator::GetItem));
    }

    #[test]
    fn test_is_inherently_scalar() {
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::Call(
            CallKind::Direct
        )));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::GetItem));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::SetAttr));
        assert!(LegalityAnalyzer::is_inherently_scalar(&Operator::GetIter));

        assert!(!LegalityAnalyzer::is_inherently_scalar(&Operator::IntOp(
            ArithOp::Add
        )));
        assert!(!LegalityAnalyzer::is_inherently_scalar(&Operator::Memory(
            MemoryOp::Load
        )));
    }

    #[test]
    fn test_analyzer_default() {
        let analyzer = LegalityAnalyzer::default();
        assert_eq!(analyzer.max_width, 8);
    }

    // -------------------------------------------------------------------------
    // Dependence Checking Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_check_dependences_forward_ok() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Forward);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(result.legal);
    }

    #[test]
    fn test_check_dependences_backward_blocks() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Backward);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(!result.legal);
        assert!(matches!(
            result.primary_reason,
            Some(ViolationKind::BackwardDependence(_, _))
        ));
    }

    #[test]
    fn test_check_dependences_unknown_blocks() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let mut dep = super::super::dependence::Dependence::new(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        dep.set_direction(0, Direction::Unknown);
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(!result.legal);
        assert!(matches!(
            result.primary_reason,
            Some(ViolationKind::UnknownDependence(_, _))
        ));
    }

    #[test]
    fn test_check_dependences_loop_independent_ok() {
        let analyzer = LegalityAnalyzer::new(8);
        let mut result = LegalityResult::legal(8);

        let mut deps = super::super::dependence::DependenceGraph::new(1);
        let dep = super::super::dependence::Dependence::loop_independent(
            NodeId::new(1),
            NodeId::new(2),
            super::super::dependence::DependenceKind::RAW,
        );
        deps.add_dependence(dep);

        analyzer.check_dependences(&deps, &mut result);
        assert!(result.legal);
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_full_analysis_empty() {
        let analyzer = LegalityAnalyzer::new(8);
        let graph = crate::ir::graph::Graph::new();
        let deps = super::super::dependence::DependenceGraph::new(1);

        let result = analyzer.analyze(&graph, &[], &deps);
        assert!(result.legal);
    }

    #[test]
    fn test_max_width_from_deps() {
        let analyzer = LegalityAnalyzer::new(8);

        let deps = super::super::dependence::DependenceGraph::new(1);
        // Empty deps = fully vectorizable
        let width = analyzer.max_width_from_deps(&deps);
        assert_eq!(width, Some(usize::MAX));
    }
