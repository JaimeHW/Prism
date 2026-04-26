use super::*;
use crate::ir::graph::Graph;

fn empty_analysis() -> (AnticipationAnalysis, AvailabilityAnalysis, ExpressionTable) {
    let graph = Graph::new();
    let expr_table = ExpressionTable::build(&graph);
    let antic = AnticipationAnalysis::compute(&graph, &expr_table);
    let avail = AvailabilityAnalysis::compute(&graph, &expr_table);
    (antic, avail, expr_table)
}

#[test]
fn test_placement_empty() {
    let (antic, avail, expr_table) = empty_analysis();
    let placement = PlacementAnalysis::compute(&antic, &avail, &expr_table);

    assert!(!placement.has_changes());
    assert_eq!(placement.total_insertions(), 0);
    assert_eq!(placement.total_deletions(), 0);
}

#[test]
fn test_placement_should_insert() {
    let mut placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 5],
        delete_at: vec![FxHashSet::default(); 5],
        total_insertions: 0,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    let expr_id = ExprId::new(0);
    placement.insert_at[2].insert(expr_id);

    assert!(placement.should_insert(2, expr_id));
    assert!(!placement.should_insert(3, expr_id));
}

#[test]
fn test_placement_should_delete() {
    let mut placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 5],
        delete_at: vec![FxHashSet::default(); 5],
        total_insertions: 0,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    let expr_id = ExprId::new(0);
    placement.delete_at[3].insert(expr_id);

    assert!(placement.should_delete(3, expr_id));
    assert!(!placement.should_delete(2, expr_id));
}

#[test]
fn test_placement_insertions_at() {
    let mut placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 5],
        delete_at: vec![FxHashSet::default(); 5],
        total_insertions: 0,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    placement.insert_at[1].insert(ExprId::new(0));
    placement.insert_at[1].insert(ExprId::new(1));

    let insertions = placement.insertions_at(1);
    assert_eq!(insertions.len(), 2);
}

#[test]
fn test_placement_deletions_at() {
    let mut placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 5],
        delete_at: vec![FxHashSet::default(); 5],
        total_insertions: 0,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    placement.delete_at[2].insert(ExprId::new(0));

    let deletions = placement.deletions_at(2);
    assert_eq!(deletions.len(), 1);
}

#[test]
fn test_placement_out_of_bounds() {
    let placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 3],
        delete_at: vec![FxHashSet::default(); 3],
        total_insertions: 0,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    assert!(!placement.should_insert(100, ExprId::new(0)));
    assert!(!placement.should_delete(100, ExprId::new(0)));
    assert!(placement.insertions_at(100).is_empty());
    assert!(placement.deletions_at(100).is_empty());
}

#[test]
fn test_placement_has_changes() {
    let mut placement = PlacementAnalysis {
        insert_at: vec![FxHashSet::default(); 3],
        delete_at: vec![FxHashSet::default(); 3],
        total_insertions: 1,
        total_deletions: 0,
        empty: FxHashSet::default(),
    };

    assert!(placement.has_changes());

    placement.total_insertions = 0;
    placement.total_deletions = 1;
    assert!(placement.has_changes());

    placement.total_deletions = 0;
    assert!(!placement.has_changes());
}

#[test]
fn test_estimate_node_count() {
    let (antic, avail, _) = empty_analysis();
    let count = PlacementAnalysis::estimate_node_count(&antic, &avail);
    assert!(count >= 2);
}
