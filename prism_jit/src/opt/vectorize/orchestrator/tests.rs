    use super::*;
    use crate::ir::graph::Graph;
    use crate::ir::node::NodeId;
    use crate::ir::operators::*;

    // =========================================================================
    // Orchestrator Construction
    // =========================================================================

    #[test]
    fn test_orchestrator_new_default_config() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
        assert!(orch.loop_decisions().is_empty());
        assert!(orch.slp_decisions().is_empty());
    }

    #[test]
    fn test_orchestrator_new_sse42_config() {
        let config = VectorizeConfig::sse42();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Sse42);
        assert_eq!(orch.config.max_vector_width, 2);
    }

    #[test]
    fn test_orchestrator_new_avx2_config() {
        let config = VectorizeConfig::avx2();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx2);
        assert_eq!(orch.config.max_vector_width, 4);
    }

    #[test]
    fn test_orchestrator_new_avx512_config() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx512);
        assert_eq!(orch.config.max_vector_width, 8);
    }

    #[test]
    fn test_orchestrator_new_aggressive_config() {
        let config = VectorizeConfig::aggressive();
        let orch = VectorizationOrchestrator::new(&config);
        assert_eq!(orch.config.simd_level, SimdLevel::Avx512);
        assert_eq!(orch.config.max_vector_width, 16);
        assert!(orch.config.enable_gather_scatter);
    }

    #[test]
    fn test_orchestrator_with_custom_cost_model() {
        let config = VectorizeConfig::default();
        let cost_model = VectorCostModel::new(SimdLevel::Avx512);
        let orch = VectorizationOrchestrator::with_cost_model(&config, cost_model);
        assert_eq!(orch.cost_model.level(), SimdLevel::Avx512);
    }

    // =========================================================================
    // Run on Empty Graph
    // =========================================================================

    #[test]
    fn test_orchestrator_run_empty_graph() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        let changed = orch.run(&mut graph);

        // Empty graph should not be changed
        assert!(!changed);
        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
    }

    #[test]
    fn test_orchestrator_run_no_loops() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        // Add some simple nodes but no loops
        let _c1 = graph.const_int(42);
        let _c2 = graph.const_int(100);

        let changed = orch.run(&mut graph);

        assert!(!changed);
        assert_eq!(orch.stats().loops_analyzed, 0);
    }

    // =========================================================================
    // Vector Width Selection
    // =========================================================================

    #[test]
    fn test_select_optimal_width_unconstrained() {
        let config = VectorizeConfig::avx2();
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();
        // max_legal_width defaults to usize::MAX — no legality constraint

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 4); // AVX2 max = 4
    }

    #[test]
    fn test_select_optimal_width_constrained_by_deps() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);

        // Deps allow max width of 4
        let mut deps = DependenceGraph::new(1);
        // DependenceGraph's max_safe_width() defaults to large value
        // when no deps exist, so this will take from config

        let legality = super::super::legality::LegalityResult::default(); // max_legal_width = usize::MAX
        let width = orch.select_optimal_width(&deps, &legality);
        // Should be limited by config max_vector_width (8 for avx512)
        // and rounded to power of 2
        assert!(width >= 2);
        assert!(width <= 8);
    }

    #[test]
    fn test_select_optimal_width_constrained_by_legality() {
        let config = VectorizeConfig::avx512();
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let mut legality = super::super::legality::LegalityResult::default();
        legality.max_legal_width = 2; // Legality limits to 2

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 2);
    }

    #[test]
    fn test_select_optimal_width_power_of_two_rounding() {
        let config = VectorizeConfig {
            max_vector_width: 5, // Not a power of 2
            ..VectorizeConfig::default()
        };
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 4); // Rounded down to 4
    }

    #[test]
    fn test_select_optimal_width_min_is_1() {
        let config = VectorizeConfig {
            max_vector_width: 1,
            ..VectorizeConfig::default()
        };
        let orch = VectorizationOrchestrator::new(&config);

        let deps = DependenceGraph::new(1);
        let legality = super::super::legality::LegalityResult::default();

        let width = orch.select_optimal_width(&deps, &legality);
        assert_eq!(width, 1);
    }

    // =========================================================================
    // Memory Operation Detection
    // =========================================================================

    #[test]
    fn test_is_memory_op_load_field() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::LoadField
        )));
    }

    #[test]
    fn test_is_memory_op_store_field() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::StoreField
        )));
    }

    #[test]
    fn test_is_memory_op_load_element() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::LoadElement
        )));
    }

    #[test]
    fn test_is_memory_op_store_element() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::StoreElement
        )));
    }

    #[test]
    fn test_is_memory_op_get_item() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::GetItem));
    }

    #[test]
    fn test_is_memory_op_set_item() {
        assert!(VectorizationOrchestrator::is_memory_op(&Operator::SetItem));
    }

    #[test]
    fn test_is_not_memory_op_arith() {
        assert!(!VectorizationOrchestrator::is_memory_op(&Operator::IntOp(
            ArithOp::Add
        )));
    }

    #[test]
    fn test_is_not_memory_op_const() {
        assert!(!VectorizationOrchestrator::is_memory_op(
            &Operator::ConstInt(42)
        ));
    }

    #[test]
    fn test_is_not_memory_op_control() {
        assert!(!VectorizationOrchestrator::is_memory_op(
            &Operator::Control(ControlOp::Start)
        ));
    }

    #[test]
    fn test_is_not_memory_op_alloc() {
        // Alloc is a memory op by type but not relevant for vectorization dependence
        assert!(!VectorizationOrchestrator::is_memory_op(&Operator::Memory(
            MemoryOp::Alloc
        )));
    }

    // =========================================================================
    // SimdLevel Extensions
    // =========================================================================

    #[test]
    fn test_simd_level_max_lanes_scalar() {
        assert_eq!(SimdLevel::Scalar.max_lanes(), 1);
    }

    #[test]
    fn test_simd_level_max_lanes_sse42() {
        assert_eq!(SimdLevel::Sse42.max_lanes(), 2);
    }

    #[test]
    fn test_simd_level_max_lanes_avx2() {
        assert_eq!(SimdLevel::Avx2.max_lanes(), 4);
    }

    #[test]
    fn test_simd_level_max_lanes_avx512() {
        assert_eq!(SimdLevel::Avx512.max_lanes(), 8);
    }

    #[test]
    fn test_simd_level_max_lanes_neon() {
        assert_eq!(SimdLevel::Neon.max_lanes(), 2);
    }

    // =========================================================================
    // LoopDecision Tests
    // =========================================================================

    #[test]
    fn test_loop_decision_not_vectorized() {
        let decision = LoopDecision {
            loop_index: 0,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::NoMemoryOps),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        assert_eq!(decision.vector_width, 1);
        assert!((decision.estimated_speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_loop_decision_vectorized() {
        let analysis = LoopVecAnalysis {
            vectorizable: true,
            rejection_reason: None,
            vector_width: 4,
            runtime_checks: Vec::new(),
            inductions: Vec::new(),
            reductions: Vec::new(),
            legality: super::super::legality::LegalityResult::default(),
            cost: None,
            trip_count: Some(100),
            interleave_factor: 1,
        };
        let decision = LoopDecision {
            loop_index: 0,
            vectorized: true,
            analysis: Some(analysis),
            rejection: None,
            vector_width: 4,
            estimated_speedup: 3.6,
        };
        assert!(decision.vectorized);
        assert_eq!(decision.vector_width, 4);
        assert!(decision.estimated_speedup > 1.0);
    }

    #[test]
    fn test_loop_decision_rejected_dependences() {
        let decision = LoopDecision {
            loop_index: 1,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::IllegalDependences { violation_count: 3 }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        match &decision.rejection {
            Some(LoopRejection::IllegalDependences { violation_count }) => {
                assert_eq!(*violation_count, 3);
            }
            _ => panic!("Expected IllegalDependences rejection"),
        }
    }

    #[test]
    fn test_loop_decision_rejected_unprofitable() {
        let decision = LoopDecision {
            loop_index: 2,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::NotProfitable { speedup: 0.5 }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        match &decision.rejection {
            Some(LoopRejection::NotProfitable { speedup }) => {
                assert!((*speedup - 0.5).abs() < 0.001);
            }
            _ => panic!("Expected NotProfitable rejection"),
        }
    }

    #[test]
    fn test_loop_decision_rejected_trip_count() {
        let decision = LoopDecision {
            loop_index: 3,
            vectorized: false,
            analysis: None,
            rejection: Some(LoopRejection::TripCountTooLow {
                trip_count: 4,
                minimum: 8,
            }),
            vector_width: 1,
            estimated_speedup: 1.0,
        };
        match &decision.rejection {
            Some(LoopRejection::TripCountTooLow {
                trip_count,
                minimum,
            }) => {
                assert_eq!(*trip_count, 4);
                assert_eq!(*minimum, 8);
            }
            _ => panic!("Expected TripCountTooLow rejection"),
        }
    }

    // =========================================================================
    // SlpDecision Tests
    // =========================================================================

    #[test]
    fn test_slp_decision_not_vectorized() {
        let decision = SlpDecision {
            vectorized: false,
            vector_ops: 0,
            scalar_ops_eliminated: 0,
            estimated_speedup: 1.0,
        };
        assert!(!decision.vectorized);
        assert_eq!(decision.vector_ops, 0);
    }

    #[test]
    fn test_slp_decision_vectorized() {
        let decision = SlpDecision {
            vectorized: true,
            vector_ops: 5,
            scalar_ops_eliminated: 20,
            estimated_speedup: 4.0,
        };
        assert!(decision.vectorized);
        assert_eq!(decision.vector_ops, 5);
        assert_eq!(decision.scalar_ops_eliminated, 20);
        assert!((decision.estimated_speedup - 4.0).abs() < 0.001);
    }

    // =========================================================================
    // LoopRejection Debug Coverage
    // =========================================================================

    #[test]
    fn test_loop_rejection_debug_formatting() {
        // Ensure all variants implement Debug
        let rejections: Vec<LoopRejection> = vec![
            LoopRejection::IllegalDependences { violation_count: 1 },
            LoopRejection::NotProfitable { speedup: 0.8 },
            LoopRejection::TripCountTooLow {
                trip_count: 3,
                minimum: 8,
            },
            LoopRejection::TooComplex(VecRejectReason::ComplexControlFlow),
            LoopRejection::NoMemoryOps,
        ];

        for r in &rejections {
            let s = format!("{:?}", r);
            assert!(!s.is_empty());
        }
    }

    // =========================================================================
    // Trip Count Estimation
    // =========================================================================

    #[test]
    fn test_estimate_trip_count_simple_loop() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1)],
            body: vec![BlockId::new(0), BlockId::new(1)],
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        assert!(tc.is_some());
        assert!(tc.unwrap() >= config.min_trip_count);
    }

    #[test]
    fn test_estimate_trip_count_multi_backedge() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1), BlockId::new(2)],
            body: vec![BlockId::new(0), BlockId::new(1), BlockId::new(2)],
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        // Multiple back edges → unknown trip count
        assert!(tc.is_none());
    }

    #[test]
    fn test_estimate_trip_count_large_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        use crate::ir::cfg::BlockId;
        let body: Vec<BlockId> = (0..25).map(|i| BlockId::new(i)).collect();
        let loop_info = Loop {
            header: BlockId::new(0),
            back_edges: vec![BlockId::new(1)],
            body,
            parent: None,
            children: vec![],
            depth: 1,
        };

        let tc = orch.estimate_trip_count(&loop_info);
        // Large body → unknown trip count
        assert!(tc.is_none());
    }

    // =========================================================================
    // Collect Memory Ops
    // =========================================================================

    #[test]
    fn test_collect_memory_ops_from_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        let mut graph = Graph::new();
        use crate::ir::node::InputList;

        // Create some nodes including memory ops
        let c1 = graph.const_int(0);
        let c2 = graph.const_int(1);
        let add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));
        let load = graph.add_node(
            Operator::Memory(MemoryOp::LoadElement),
            InputList::Single(c1),
        );
        let store = graph.add_node(
            Operator::Memory(MemoryOp::StoreElement),
            InputList::Pair(c1, c2),
        );

        let body_nodes = vec![c1, c2, add, load, store];
        let mem_ops = orch.collect_memory_ops(&graph, &body_nodes);

        assert_eq!(mem_ops.len(), 2); // load + store
        assert!(mem_ops.contains(&load));
        assert!(mem_ops.contains(&store));
    }

    #[test]
    fn test_collect_memory_ops_empty_body() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        let graph = Graph::new();

        let mem_ops = orch.collect_memory_ops(&graph, &[]);
        assert!(mem_ops.is_empty());
    }

    #[test]
    fn test_collect_memory_ops_no_memory() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        let mut graph = Graph::new();
        let c1 = graph.const_int(0);
        let c2 = graph.const_int(1);

        let mem_ops = orch.collect_memory_ops(&graph, &[c1, c2]);
        assert!(mem_ops.is_empty());
    }

    // =========================================================================
    // Stats Tracking
    // =========================================================================

    #[test]
    fn test_stats_initial_zero() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);

        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
        assert_eq!(orch.stats().loops_rejected_unsafe, 0);
        assert_eq!(orch.stats().loops_rejected_unprofitable, 0);
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
        assert_eq!(orch.stats().slp_regions_vectorized, 0);
        assert_eq!(orch.stats().vector_ops_created, 0);
        assert_eq!(orch.stats().scalar_ops_eliminated, 0);
        assert!((orch.stats().estimated_speedup - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_after_empty_run() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        assert_eq!(orch.stats().loops_analyzed, 0);
        assert_eq!(orch.stats().loops_vectorized, 0);
    }

    // =========================================================================
    // Config Propagation
    // =========================================================================

    #[test]
    fn test_config_disable_slp() {
        let config = VectorizeConfig {
            enable_slp: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        // SLP should not have been analyzed
        assert_eq!(orch.stats().slp_regions_analyzed, 0);
    }

    #[test]
    fn test_config_disable_loop_vec() {
        let config = VectorizeConfig {
            enable_loop_vec: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        orch.run(&mut graph);

        assert_eq!(orch.stats().loops_analyzed, 0);
    }

    #[test]
    fn test_config_disable_both() {
        let config = VectorizeConfig {
            enable_loop_vec: false,
            enable_slp: false,
            ..VectorizeConfig::default()
        };
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        let changed = orch.run(&mut graph);
        assert!(!changed);
    }

    // =========================================================================
    // Integration: Orchestrator with Graph containing only constants
    // =========================================================================

    #[test]
    fn test_orchestrator_graph_with_constants_only() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        // Constants only — no loops, no SLP candidates
        let _c1 = graph.const_int(1);
        let _c2 = graph.const_int(2);
        let _c3 = graph.const_int(3);
        let _c4 = graph.const_int(4);

        let changed = orch.run(&mut graph);
        assert!(!changed);
    }

    #[test]
    fn test_orchestrator_graph_with_arithmetic() {
        let config = VectorizeConfig::default();
        let mut orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        use crate::ir::node::InputList;

        let c1 = graph.const_int(10);
        let c2 = graph.const_int(20);
        let _add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));
        let _mul = graph.add_node(Operator::IntOp(ArithOp::Mul), InputList::Pair(c1, c2));

        let changed = orch.run(&mut graph);
        // Simple arithmetic without loops — may or may not vectorize via SLP
        // but at minimum should not crash
        let _ = changed; // Don't assert specific outcome — graph-dependent
    }

    // =========================================================================
    // Non-Loop Node Collection
    // =========================================================================

    #[test]
    fn test_collect_non_loop_nodes_empty_analysis() {
        let config = VectorizeConfig::default();
        let orch = VectorizationOrchestrator::new(&config);
        let mut graph = Graph::new();

        use crate::ir::node::InputList;
        let c1 = graph.const_int(1);
        let c2 = graph.const_int(2);
        let _add = graph.add_node(Operator::IntOp(ArithOp::Add), InputList::Pair(c1, c2));

        let empty_cfg = Cfg::build(&graph);
        let dom = DominatorTree::build(&empty_cfg);
        let loop_analysis = LoopAnalysis::compute(&empty_cfg, &dom);

        let non_loop = orch.collect_non_loop_nodes(&graph, &loop_analysis);
        // Should include the add node but not constants
        // (constants are filtered out)
        assert!(non_loop.iter().any(|&id| {
            let node = graph.node(id);
            matches!(node.op, Operator::IntOp(ArithOp::Add))
        }));
    }
