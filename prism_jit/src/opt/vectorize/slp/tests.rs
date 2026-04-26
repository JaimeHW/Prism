    use super::*;

    // -------------------------------------------------------------------------
    // Pack Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_pack_new() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            NodeId::new(4),
        ];
        let pack = Pack::new(scalars.clone(), PackOpKind::Arith(ArithOp::Add));

        assert_eq!(pack.lanes(), 4);
        assert!(!pack.is_empty());
        assert!(pack.is_complete());
        assert_eq!(pack.op_kind, PackOpKind::Arith(ArithOp::Add));
    }

    #[test]
    fn test_pack_empty() {
        let pack = Pack::empty();
        assert!(pack.is_empty());
        assert!(!pack.is_complete());
        assert_eq!(pack.lanes(), 0);
    }

    #[test]
    fn test_pack_scalar_at() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![NodeId::new(1), NodeId::new(2),];
        let pack = Pack::new(scalars, PackOpKind::Load);

        assert_eq!(pack.scalar_at(0), Some(NodeId::new(1)));
        assert_eq!(pack.scalar_at(1), Some(NodeId::new(2)));
        assert_eq!(pack.scalar_at(2), None);
    }

    #[test]
    fn test_pack_set_scalar() {
        let mut pack = Pack::empty();
        pack.set_scalar(0, NodeId::new(1));
        pack.set_scalar(2, NodeId::new(3));

        assert_eq!(pack.lanes(), 3);
        assert_eq!(pack.scalar_at(0), Some(NodeId::new(1)));
        assert_eq!(pack.scalar_at(1), None); // Invalid node
        assert_eq!(pack.scalar_at(2), Some(NodeId::new(3)));
    }

    #[test]
    fn test_pack_contains() {
        let scalars: SmallVec<[NodeId; 8]> = smallvec::smallvec![NodeId::new(1), NodeId::new(2),];
        let pack = Pack::new(scalars, PackOpKind::Load);

        assert!(pack.contains(NodeId::new(1)));
        assert!(pack.contains(NodeId::new(2)));
        assert!(!pack.contains(NodeId::new(3)));
    }

    #[test]
    fn test_pack_lane_of() {
        let scalars: SmallVec<[NodeId; 8]> =
            smallvec::smallvec![NodeId::new(10), NodeId::new(20), NodeId::new(30),];
        let pack = Pack::new(scalars, PackOpKind::Store);

        assert_eq!(pack.lane_of(NodeId::new(10)), Some(0));
        assert_eq!(pack.lane_of(NodeId::new(20)), Some(1));
        assert_eq!(pack.lane_of(NodeId::new(30)), Some(2));
        assert_eq!(pack.lane_of(NodeId::new(99)), None);
    }

    #[test]
    fn test_pack_savings() {
        let mut pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );

        // Not profitable
        pack.profitable = false;
        pack.scalar_cost = 4.0;
        pack.vector_cost = 5.0;
        assert!((pack.savings() - 0.0).abs() < 0.001);

        // Profitable
        pack.profitable = true;
        assert!((pack.savings() - (-1.0)).abs() < 0.001);
    }

    #[test]
    fn test_pack_set_element_type() {
        let mut pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );

        pack.set_element_type(ValueType::Float64);
        assert_eq!(pack.vector_type.element, ValueType::Float64);
    }

    // -------------------------------------------------------------------------
    // PackOpKind Tests
    // -------------------------------------------------------------------------

    // -------------------------------------------------------------------------
    // SlpTree Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_tree_new() {
        let tree = SlpTree::new();
        assert!(tree.is_empty());
        assert_eq!(tree.num_packs(), 0);
    }

    #[test]
    fn test_slp_tree_add_pack() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Load,
        );

        let idx = tree.add_pack(pack);
        assert_eq!(idx, 0);
        assert_eq!(tree.num_packs(), 1);
        assert!(!tree.is_empty());
    }

    #[test]
    fn test_slp_tree_find_pack() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(
            smallvec::smallvec![NodeId::new(1), NodeId::new(2)],
            PackOpKind::Load,
        );
        let idx = tree.add_pack(pack);

        assert_eq!(tree.find_pack(NodeId::new(1)), Some(idx));
        assert_eq!(tree.find_pack(NodeId::new(2)), Some(idx));
        assert_eq!(tree.find_pack(NodeId::new(99)), None);
    }

    #[test]
    fn test_slp_tree_is_packed() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Constant);
        tree.add_pack(pack);

        assert!(tree.is_packed(NodeId::new(1)));
        assert!(!tree.is_packed(NodeId::new(2)));
    }

    #[test]
    fn test_slp_tree_mark_root() {
        let mut tree = SlpTree::new();
        let pack = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Store);
        let idx = tree.add_pack(pack);

        tree.mark_root(idx);
        tree.mark_root(idx); // Duplicate should not add

        assert_eq!(tree.roots().len(), 1);
        assert_eq!(tree.roots()[0], idx);
    }

    #[test]
    fn test_slp_tree_add_dependency() {
        let mut tree = SlpTree::new();
        let pack1 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        let pack2 = Pack::new(smallvec::smallvec![NodeId::new(2)], PackOpKind::Store);

        let idx1 = tree.add_pack(pack1);
        let idx2 = tree.add_pack(pack2);

        tree.add_dependency(idx1, idx2);
        assert_eq!(tree.pack_deps.len(), 1);
    }

    #[test]
    fn test_slp_tree_topological_order() {
        let mut tree = SlpTree::new();

        // Create chain: 0 -> 1 -> 2
        let pack0 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        let pack1 = Pack::new(
            smallvec::smallvec![NodeId::new(2)],
            PackOpKind::Arith(ArithOp::Add),
        );
        let pack2 = Pack::new(smallvec::smallvec![NodeId::new(3)], PackOpKind::Store);

        tree.add_pack(pack0);
        tree.add_pack(pack1);
        tree.add_pack(pack2);
        tree.add_dependency(0, 1);
        tree.add_dependency(1, 2);

        let order = tree.topological_order();
        assert_eq!(order.len(), 3);

        // 0 should come before 1, and 1 before 2
        let pos_0 = order.iter().position(|&x| x == 0).unwrap();
        let pos_1 = order.iter().position(|&x| x == 1).unwrap();
        let pos_2 = order.iter().position(|&x| x == 2).unwrap();

        assert!(pos_0 < pos_1);
        assert!(pos_1 < pos_2);
    }

    #[test]
    fn test_slp_tree_clear() {
        let mut tree = SlpTree::new();
        tree.add_pack(Pack::new(
            smallvec::smallvec![NodeId::new(1)],
            PackOpKind::Load,
        ));
        tree.mark_root(0);

        tree.clear();
        assert!(tree.is_empty());
        assert!(tree.roots().is_empty());
    }

    #[test]
    fn test_slp_tree_vector_ops_count() {
        let mut tree = SlpTree::new();

        let mut pack1 = Pack::new(smallvec::smallvec![NodeId::new(1)], PackOpKind::Load);
        pack1.profitable = true;
        tree.add_pack(pack1);

        let mut pack2 = Pack::new(smallvec::smallvec![NodeId::new(2)], PackOpKind::Load);
        pack2.profitable = false;
        tree.add_pack(pack2);

        assert_eq!(tree.vector_ops_count(), 1);
    }

    #[test]
    fn test_slp_tree_debug() {
        let tree = SlpTree::new();
        let debug_str = format!("{:?}", tree);
        assert!(debug_str.contains("SlpTree"));
    }

    // -------------------------------------------------------------------------
    // SlpVectorizer Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_vectorizer_new() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        assert!(vectorizer.tree().is_empty());
        assert_eq!(vectorizer.statistics().seeds_found, 0);
    }

    impl SlpVectorizer<'_> {
        fn statistics(&self) -> &SlpStats {
            &self.stats
        }
    }

    #[test]
    fn test_slp_find_seeds_empty() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        let seeds = vectorizer.find_seeds(&[]);
        assert!(seeds.is_empty());
    }

    #[test]
    fn test_slp_are_isomorphic_empty() {
        let graph = Graph::new();
        let cost_model = VectorCostModel::default();
        let vectorizer = SlpVectorizer::new(&graph, &cost_model, 4);

        assert!(!vectorizer.are_isomorphic(&[]));
        assert!(!vectorizer.are_isomorphic(&[NodeId::new(1)]));
    }

    #[test]
    fn test_slp_classify_op() {
        assert_eq!(
            SlpVectorizer::classify_op(&Operator::IntOp(ArithOp::Add)),
            Some(PackOpKind::Arith(ArithOp::Add))
        );
        assert_eq!(
            SlpVectorizer::classify_op(&Operator::ConstInt(42)),
            Some(PackOpKind::Constant)
        );
        assert_eq!(SlpVectorizer::classify_op(&Operator::Phi), None);
    }

    #[test]
    fn test_slp_is_store_op() {
        assert!(SlpVectorizer::is_store_op(&Operator::SetItem));
        assert!(SlpVectorizer::is_store_op(&Operator::Memory(
            crate::ir::operators::MemoryOp::Store
        )));
        assert!(!SlpVectorizer::is_store_op(&Operator::IntOp(ArithOp::Add)));
    }

    // -------------------------------------------------------------------------
    // SlpResult Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_result_unchanged() {
        let result = SlpResult::unchanged();
        assert!(!result.changed);
        assert_eq!(result.vector_ops, 0);
        assert_eq!(result.scalar_ops, 0);
        assert!((result.speedup - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slp_result_success() {
        let result = SlpResult::success(4, 16, 3.5);
        assert!(result.changed);
        assert_eq!(result.vector_ops, 4);
        assert_eq!(result.scalar_ops, 16);
        assert!((result.speedup - 3.5).abs() < 0.001);
    }

    // -------------------------------------------------------------------------
    // SlpStats Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_slp_stats_default() {
        let stats = SlpStats::default();
        assert_eq!(stats.seeds_found, 0);
        assert_eq!(stats.packs_built, 0);
        assert_eq!(stats.profitable_packs, 0);
    }
