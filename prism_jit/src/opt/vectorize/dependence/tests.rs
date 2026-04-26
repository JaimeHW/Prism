    use super::*;

    // -------------------------------------------------------------------------
    // Direction Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_direction_is_safe() {
        assert!(Direction::Forward.is_safe());
        assert!(Direction::Equal.is_safe());
        assert!(!Direction::Backward.is_safe());
        assert!(!Direction::Unknown.is_safe());
    }

    #[test]
    fn test_direction_prevents_vectorization() {
        assert!(Direction::Backward.prevents_vectorization());
        assert!(!Direction::Forward.prevents_vectorization());
        assert!(!Direction::Equal.prevents_vectorization());
        assert!(!Direction::Unknown.prevents_vectorization());
    }

    #[test]
    fn test_direction_merge() {
        // Equal merges to other
        assert_eq!(
            Direction::Equal.merge(Direction::Forward),
            Direction::Forward
        );
        assert_eq!(
            Direction::Equal.merge(Direction::Backward),
            Direction::Backward
        );
        assert_eq!(
            Direction::Forward.merge(Direction::Equal),
            Direction::Forward
        );

        // Same direction stays same
        assert_eq!(
            Direction::Forward.merge(Direction::Forward),
            Direction::Forward
        );
        assert_eq!(
            Direction::Backward.merge(Direction::Backward),
            Direction::Backward
        );

        // Opposite directions -> Unknown
        assert_eq!(
            Direction::Forward.merge(Direction::Backward),
            Direction::Unknown
        );
        assert_eq!(
            Direction::Backward.merge(Direction::Forward),
            Direction::Unknown
        );

        // Unknown dominates
        assert_eq!(
            Direction::Unknown.merge(Direction::Forward),
            Direction::Unknown
        );
        assert_eq!(
            Direction::Forward.merge(Direction::Unknown),
            Direction::Unknown
        );
    }

    #[test]
    fn test_direction_default() {
        assert_eq!(Direction::default(), Direction::Unknown);
    }

    // -------------------------------------------------------------------------
    // Distance Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_distance_min_distance() {
        assert_eq!(Distance::Constant(5).min_distance(), Some(5));
        assert_eq!(Distance::Constant(-3).min_distance(), Some(-3));
        assert_eq!(Distance::Constant(0).min_distance(), Some(0));
        assert_eq!(Distance::Positive.min_distance(), Some(1));
        assert_eq!(Distance::Negative.min_distance(), None);
        assert_eq!(Distance::Unknown.min_distance(), None);
        assert_eq!(Distance::Infinity.min_distance(), None);
    }

    #[test]
    fn test_distance_stride_min() {
        let dist = Distance::Stride { base: 2, stride: 4 };
        assert_eq!(dist.min_distance(), Some(2));

        let neg_stride = Distance::Stride {
            base: 2,
            stride: -4,
        };
        assert_eq!(neg_stride.min_distance(), None);
    }

    #[test]
    fn test_distance_allows_vector_width() {
        assert!(Distance::Constant(4).allows_vector_width(4));
        assert!(Distance::Constant(8).allows_vector_width(4));
        assert!(!Distance::Constant(2).allows_vector_width(4));
        assert!(!Distance::Constant(-1).allows_vector_width(4));
        assert!(!Distance::Unknown.allows_vector_width(4));
        assert!(Distance::Constant(0).allows_vector_width(0)); // Edge case
    }

    #[test]
    fn test_distance_merge() {
        // Same constants stay same
        assert_eq!(
            Distance::Constant(5).merge(Distance::Constant(5)),
            Distance::Constant(5)
        );

        // Infinity is absorbed
        assert_eq!(
            Distance::Constant(5).merge(Distance::Infinity),
            Distance::Constant(5)
        );
        assert_eq!(
            Distance::Infinity.merge(Distance::Constant(5)),
            Distance::Constant(5)
        );

        // Unknown dominates
        assert_eq!(
            Distance::Constant(5).merge(Distance::Unknown),
            Distance::Unknown
        );

        // Positive stays positive
        assert_eq!(
            Distance::Positive.merge(Distance::Positive),
            Distance::Positive
        );
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(17, 13), 1);
        assert_eq!(gcd(100, 25), 25);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(5, 0), 5);
    }

    // -------------------------------------------------------------------------
    // DependenceKind Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dependence_kind_source_is_write() {
        assert!(DependenceKind::RAW.source_is_write());
        assert!(!DependenceKind::WAR.source_is_write());
        assert!(DependenceKind::WAW.source_is_write());
    }

    #[test]
    fn test_dependence_kind_dest_is_write() {
        assert!(!DependenceKind::RAW.dest_is_write());
        assert!(DependenceKind::WAR.dest_is_write());
        assert!(DependenceKind::WAW.dest_is_write());
    }

    #[test]
    fn test_dependence_kind_removable() {
        assert!(!DependenceKind::RAW.removable_by_renaming());
        assert!(DependenceKind::WAR.removable_by_renaming());
        assert!(DependenceKind::WAW.removable_by_renaming());
    }

    // -------------------------------------------------------------------------
    // AccessPattern Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_access_pattern_new() {
        let base = NodeId::new(1);
        let node = NodeId::new(2);
        let pattern = AccessPattern::new(base, 16, 8, true, node);

        assert_eq!(pattern.base, base);
        assert_eq!(pattern.offset, 16);
        assert_eq!(pattern.element_size, 8);
        assert!(pattern.is_store);
        assert_eq!(pattern.node, node);
        assert!(pattern.coefficients.is_empty());
    }

    #[test]
    fn test_access_pattern_coefficients() {
        let mut pattern = AccessPattern::new(NodeId::new(1), 0, 8, false, NodeId::new(2));

        // Initially all zero
        assert_eq!(pattern.coefficient(0), 0);
        assert_eq!(pattern.coefficient(1), 0);

        // Set coefficients
        pattern.set_coefficient(0, 8); // Stride 8 for innermost loop
        pattern.set_coefficient(1, 64); // Stride 64 for outer loop

        assert_eq!(pattern.coefficient(0), 8);
        assert_eq!(pattern.coefficient(1), 64);
        assert_eq!(pattern.coefficient(2), 0); // Unset level
    }

    #[test]
    fn test_access_pattern_invariant() {
        let mut pattern = AccessPattern::new(NodeId::new(1), 0, 8, false, NodeId::new(2));
        pattern.set_coefficient(0, 8);

        assert!(!pattern.is_invariant_at(0));
        assert!(pattern.is_invariant_at(1)); // No coefficient set
    }

    #[test]
    fn test_access_pattern_same_base() {
        let base = NodeId::new(1);
        let p1 = AccessPattern::new(base, 0, 8, false, NodeId::new(2));
        let p2 = AccessPattern::new(base, 16, 8, true, NodeId::new(3));
        let p3 = AccessPattern::new(NodeId::new(99), 0, 8, false, NodeId::new(4));

        assert!(p1.same_base(&p2));
        assert!(!p1.same_base(&p3));
    }

    #[test]
    fn test_access_pattern_definitely_before() {
        let base = NodeId::new(1);
        let mut p1 = AccessPattern::new(base, 0, 8, false, NodeId::new(2));
        let mut p2 = AccessPattern::new(base, 16, 8, false, NodeId::new(3));
        p1.set_coefficient(0, 8);
        p2.set_coefficient(0, 8);

        assert!(p1.definitely_before(&p2)); // [0..8) before [16..24)
        assert!(!p2.definitely_before(&p1));
    }

    // -------------------------------------------------------------------------
    // Dependence Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dependence_new() {
        let src = NodeId::new(1);
        let dst = NodeId::new(2);
        let dep = Dependence::new(src, dst, DependenceKind::RAW);

        assert_eq!(dep.src, src);
        assert_eq!(dep.dst, dst);
        assert_eq!(dep.kind, DependenceKind::RAW);
        assert!(!dep.loop_independent);
        assert!(dep.direction.is_empty());
        assert!(dep.distance.is_empty());
    }

    #[test]
    fn test_dependence_loop_independent() {
        let dep = Dependence::loop_independent(NodeId::new(1), NodeId::new(2), DependenceKind::WAR);

        assert!(dep.loop_independent);
        assert_eq!(dep.confidence, DependenceConfidence::Proven);
    }

    #[test]
    fn test_dependence_set_direction() {
        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);

        dep.set_direction(0, Direction::Forward);
        dep.set_direction(1, Direction::Equal);

        assert_eq!(dep.direction_at(0), Direction::Forward);
        assert_eq!(dep.direction_at(1), Direction::Equal);
        assert_eq!(dep.direction_at(2), Direction::Unknown); // Unset
    }

    #[test]
    fn test_dependence_set_distance() {
        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);

        dep.set_distance(0, Distance::Constant(4));
        dep.set_distance(1, Distance::Positive);

        assert_eq!(dep.distance_at(0), Distance::Constant(4));
        assert_eq!(dep.distance_at(1), Distance::Positive);
        assert_eq!(dep.distance_at(2), Distance::Unknown);
    }

    #[test]
    fn test_dependence_prevents_vectorization_at() {
        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Forward);
        dep.set_direction(1, Direction::Backward);

        assert!(!dep.prevents_vectorization_at(0)); // Forward is ok
        assert!(dep.prevents_vectorization_at(1)); // Backward blocks
    }

    #[test]
    fn test_dependence_loop_independent_allows_vectorization() {
        let dep = Dependence::loop_independent(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);

        // Loop-independent deps never prevent vectorization
        assert!(!dep.prevents_vectorization_at(0));
        assert!(!dep.prevents_vectorization_at(1));
    }

    #[test]
    fn test_dependence_allows_vector_width() {
        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Forward);
        dep.set_distance(0, Distance::Constant(4));

        assert!(dep.allows_vector_width(0, 4));
        assert!(dep.allows_vector_width(0, 2));
        assert!(!dep.allows_vector_width(0, 8)); // Distance too small
    }

    // -------------------------------------------------------------------------
    // DependenceGraph Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dependence_graph_new() {
        let dg = DependenceGraph::new(2);

        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), usize::MAX);
        assert_eq!(dg.depth(), 2);
        assert!(dg.memory_ops().is_empty());
        assert_eq!(dg.num_dependences(), 0);
    }

    #[test]
    fn test_dependence_graph_add_dependence() {
        let mut dg = DependenceGraph::new(1);
        let src = NodeId::new(1);
        let dst = NodeId::new(2);

        dg.memory_ops.push(src);
        dg.memory_ops.push(dst);
        dg.stores.push(src);
        dg.loads.push(dst);

        let dep = Dependence::new(src, dst, DependenceKind::RAW);
        dg.add_dependence(dep);

        assert_eq!(dg.num_dependences(), 1);
        assert!(dg.has_dependence(src, dst));
        assert!(!dg.has_dependence(dst, src));
    }

    #[test]
    fn test_dependence_graph_dependences_from() {
        let mut dg = DependenceGraph::new(1);
        let src = NodeId::new(1);
        let dst = NodeId::new(2);

        let dep = Dependence::new(src, dst, DependenceKind::RAW);
        dg.add_dependence(dep);

        assert_eq!(dg.dependences_from(src).len(), 1);
        assert!(dg.dependences_from(dst).is_empty());
    }

    #[test]
    fn test_dependence_graph_dependences_to() {
        let mut dg = DependenceGraph::new(1);
        let src = NodeId::new(1);
        let dst = NodeId::new(2);

        let dep = Dependence::new(src, dst, DependenceKind::RAW);
        dg.add_dependence(dep);

        assert!(dg.dependences_to(src).is_empty());
        assert_eq!(dg.dependences_to(dst).len(), 1);
    }

    #[test]
    fn test_dependence_graph_backward_dependences() {
        let mut dg = DependenceGraph::new(1);

        // Forward dependence
        let mut dep1 = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep1.set_direction(0, Direction::Forward);
        dg.add_dependence(dep1);

        // Backward dependence
        let mut dep2 = Dependence::new(NodeId::new(3), NodeId::new(4), DependenceKind::WAR);
        dep2.set_direction(0, Direction::Backward);
        dg.add_dependence(dep2);

        let backward = dg.backward_dependences();
        assert_eq!(backward.len(), 1);
        assert_eq!(backward[0].src, NodeId::new(3));
    }

    #[test]
    fn test_dependence_graph_loop_independent_dependences() {
        let mut dg = DependenceGraph::new(1);

        // Loop-carried
        let dep1 = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dg.add_dependence(dep1);

        // Loop-independent
        let dep2 =
            Dependence::loop_independent(NodeId::new(3), NodeId::new(4), DependenceKind::RAW);
        dg.add_dependence(dep2);

        let loop_ind = dg.loop_independent_dependences();
        assert_eq!(loop_ind.len(), 1);
        assert_eq!(loop_ind[0].src, NodeId::new(3));
    }

    #[test]
    fn test_dependence_graph_vectorizability_forward() {
        let mut dg = DependenceGraph::new(1);

        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Forward);
        dep.set_distance(0, Distance::Constant(4));
        dg.add_dependence(dep);

        dg.compute_vectorizability();

        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), 4);
    }

    #[test]
    fn test_dependence_graph_vectorizability_backward() {
        let mut dg = DependenceGraph::new(1);

        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Backward);
        dg.add_dependence(dep);

        dg.compute_vectorizability();

        assert!(!dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), 1);
    }

    #[test]
    fn test_dependence_graph_vectorizability_equal() {
        let mut dg = DependenceGraph::new(1);

        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Equal);
        dg.add_dependence(dep);

        dg.compute_vectorizability();

        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), usize::MAX);
    }

    #[test]
    fn test_dependence_graph_vectorizability_loop_independent() {
        let mut dg = DependenceGraph::new(1);

        let dep = Dependence::loop_independent(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dg.add_dependence(dep);

        dg.compute_vectorizability();

        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), usize::MAX);
    }

    #[test]
    fn test_dependence_graph_debug() {
        let dg = DependenceGraph::new(2);
        let debug_str = format!("{:?}", dg);
        assert!(debug_str.contains("DependenceGraph"));
        assert!(debug_str.contains("vectorizable: true"));
    }

    // -------------------------------------------------------------------------
    // Integration Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_multi_level_dependence() {
        let mut dg = DependenceGraph::new(2);

        // Outer loop: equal, Inner loop: forward with distance 1
        let mut dep = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep.set_direction(0, Direction::Forward);
        dep.set_distance(0, Distance::Constant(1));
        dep.set_direction(1, Direction::Equal);
        dep.set_distance(1, Distance::Constant(0));
        dg.add_dependence(dep);

        dg.compute_vectorizability();

        // Can vectorize inner loop with width 1
        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), 1);
    }

    #[test]
    fn test_multiple_dependences_min_distance() {
        let mut dg = DependenceGraph::new(1);

        // First dependence: distance 8
        let mut dep1 = Dependence::new(NodeId::new(1), NodeId::new(2), DependenceKind::RAW);
        dep1.set_direction(0, Direction::Forward);
        dep1.set_distance(0, Distance::Constant(8));
        dg.add_dependence(dep1);

        // Second dependence: distance 4 (more restrictive)
        let mut dep2 = Dependence::new(NodeId::new(3), NodeId::new(4), DependenceKind::WAW);
        dep2.set_direction(0, Direction::Forward);
        dep2.set_distance(0, Distance::Constant(4));
        dg.add_dependence(dep2);

        dg.compute_vectorizability();

        // Max safe width is minimum of all distances
        assert!(dg.is_vectorizable());
        assert_eq!(dg.max_safe_vector_width(), 4);
    }
