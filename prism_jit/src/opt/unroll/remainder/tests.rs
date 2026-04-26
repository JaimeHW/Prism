use super::*;

// =========================================================================
// RemainderStrategy Tests
// =========================================================================

#[test]
fn test_remainder_strategy_default() {
    assert_eq!(RemainderStrategy::default(), RemainderStrategy::EpilogLoop);
}

#[test]
fn test_remainder_strategy_adds_loop() {
    assert!(RemainderStrategy::EpilogLoop.adds_loop());
    assert!(RemainderStrategy::PrologLoop.adds_loop());
    assert!(!RemainderStrategy::UnrolledRemainder.adds_loop());
    assert!(!RemainderStrategy::DuffsDevice.adds_loop());
    assert!(!RemainderStrategy::None.adds_loop());
}

#[test]
fn test_remainder_strategy_adds_straight_line() {
    assert!(!RemainderStrategy::EpilogLoop.adds_straight_line());
    assert!(!RemainderStrategy::PrologLoop.adds_straight_line());
    assert!(RemainderStrategy::UnrolledRemainder.adds_straight_line());
    assert!(RemainderStrategy::DuffsDevice.adds_straight_line());
    assert!(!RemainderStrategy::None.adds_straight_line());
}

#[test]
fn test_remainder_strategy_code_growth_none() {
    assert_eq!(RemainderStrategy::None.code_growth(4, 10), 0);
}

#[test]
fn test_remainder_strategy_code_growth_epilog() {
    let growth = RemainderStrategy::EpilogLoop.code_growth(4, 10);
    assert_eq!(growth, 20); // body + overhead
}

#[test]
fn test_remainder_strategy_code_growth_prolog() {
    let growth = RemainderStrategy::PrologLoop.code_growth(4, 10);
    assert_eq!(growth, 20);
}

#[test]
fn test_remainder_strategy_code_growth_unrolled() {
    let growth = RemainderStrategy::UnrolledRemainder.code_growth(4, 10);
    // 3 copies * (10 body + 2 guards)
    assert_eq!(growth, 36);
}

#[test]
fn test_remainder_strategy_code_growth_duffs() {
    let growth = RemainderStrategy::DuffsDevice.code_growth(4, 10);
    // 3 copies * 10 body + 10 jump table
    assert_eq!(growth, 40);
}

// =========================================================================
// RemainderResult Tests
// =========================================================================

#[test]
fn test_remainder_result_none() {
    let result = RemainderResult::none();
    assert!(result.entry.is_none());
    assert!(result.exit.is_none());
    assert_eq!(result.nodes_added, 0);
    assert!(!result.has_remainder());
}

#[test]
fn test_remainder_result_has_remainder() {
    let result = RemainderResult {
        entry: Some(NodeId::new(5)),
        exit: Some(NodeId::new(10)),
        nodes_added: 15,
    };
    assert!(result.has_remainder());
}

// =========================================================================
// RemainderGenerator Tests
// =========================================================================

#[test]
fn test_remainder_generator_generate_none() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let mut remainder_gen = RemainderGenerator::new(&mut graph, &mut cfg);
    let result = remainder_gen.generate(
        RemainderStrategy::None,
        4,
        &[],
        NodeId::new(0),
        NodeId::new(1),
    );

    assert!(!result.has_remainder());
}

#[test]
fn test_remainder_generator_epilog_placeholder() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let mut remainder_gen = RemainderGenerator::new(&mut graph, &mut cfg);
    let result = remainder_gen.generate(
        RemainderStrategy::EpilogLoop,
        4,
        &[],
        NodeId::new(0),
        NodeId::new(1),
    );

    // Currently returns placeholder
    assert!(!result.has_remainder());
}

#[test]
fn test_remainder_generator_prolog_placeholder() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let mut remainder_gen = RemainderGenerator::new(&mut graph, &mut cfg);
    let result = remainder_gen.generate(
        RemainderStrategy::PrologLoop,
        4,
        &[],
        NodeId::new(0),
        NodeId::new(1),
    );

    assert!(!result.has_remainder());
}

#[test]
fn test_remainder_generator_unrolled_placeholder() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let mut remainder_gen = RemainderGenerator::new(&mut graph, &mut cfg);
    let result = remainder_gen.generate(
        RemainderStrategy::UnrolledRemainder,
        4,
        &[],
        NodeId::new(0),
        NodeId::new(1),
    );

    assert!(!result.has_remainder());
}

#[test]
fn test_remainder_generator_duffs_placeholder() {
    use crate::ir::builder::GraphBuilder;

    let builder = GraphBuilder::new(4, 0);
    let mut graph = builder.finish();
    let mut cfg = Cfg::build(&graph);

    let mut remainder_gen = RemainderGenerator::new(&mut graph, &mut cfg);
    let result = remainder_gen.generate(
        RemainderStrategy::DuffsDevice,
        4,
        &[],
        NodeId::new(0),
        NodeId::new(1),
    );

    assert!(!result.has_remainder());
}
