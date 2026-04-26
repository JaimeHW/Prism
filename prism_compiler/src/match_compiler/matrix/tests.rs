use super::*;

fn wildcard_row(action: usize) -> PatternRow {
    PatternRow {
        patterns: vec![FlatPattern::Wildcard],
        bindings: vec![],
        guard: None,
        action,
    }
}

fn literal_row(val: i64, action: usize) -> PatternRow {
    PatternRow {
        patterns: vec![FlatPattern::Literal(LiteralValue::Int(val))],
        bindings: vec![],
        guard: None,
        action,
    }
}

#[test]
fn test_empty_matrix() {
    let matrix = PatternMatrix::new(vec![]);
    assert!(matrix.is_empty());
    assert_eq!(matrix.row_count(), 0);
}

#[test]
fn test_first_match_all() {
    let matrix = PatternMatrix::new(vec![wildcard_row(0)]);
    assert!(matrix.first_match_all().is_some());
}

#[test]
fn test_first_match_all_none() {
    let matrix = PatternMatrix::new(vec![literal_row(42, 0)]);
    assert!(matrix.first_match_all().is_none());
}

#[test]
fn test_select_column_prefers_refutable() {
    let row = PatternRow {
        patterns: vec![
            FlatPattern::Wildcard,
            FlatPattern::Literal(LiteralValue::Int(1)),
        ],
        bindings: vec![],
        guard: None,
        action: 0,
    };
    let matrix = PatternMatrix::new(vec![row]);
    // Column 1 (literal) should be selected over column 0 (wildcard)
    assert_eq!(matrix.select_column(), 1);
}

#[test]
fn test_distinct_constructors() {
    let matrix = PatternMatrix::new(vec![
        literal_row(1, 0),
        literal_row(2, 1),
        literal_row(1, 2), // Duplicate
    ]);
    let ctors = matrix.distinct_constructors(0);
    assert_eq!(ctors.len(), 2); // Only 2 distinct literals
}

#[test]
fn test_specialize_literal() {
    let matrix = PatternMatrix::new(vec![literal_row(1, 0), literal_row(2, 1), wildcard_row(2)]);

    let ctor = Constructor::Literal(LiteralValue::Int(1));
    let specialized = matrix.specialize(0, &ctor);

    // Should contain row 0 (matches) and row 2 (wildcard matches anything)
    assert_eq!(specialized.row_count(), 2);
    assert_eq!(specialized.rows[0].action, 0);
    assert_eq!(specialized.rows[1].action, 2);
}

#[test]
fn test_default_matrix() {
    let matrix = PatternMatrix::new(vec![literal_row(1, 0), wildcard_row(1)]);

    let default = matrix.default_matrix(0);

    // Only wildcard row included
    assert_eq!(default.row_count(), 1);
    assert_eq!(default.rows[0].action, 1);
}
