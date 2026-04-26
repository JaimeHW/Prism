use super::*;
use crate::ir::graph::Graph;
use smallvec::SmallVec;

// -------------------------------------------------------------------------
// ScalarReplacementResult Tests
// -------------------------------------------------------------------------

#[test]
fn test_scalar_replacement_result_failure() {
    let result = ScalarReplacementResult::failure(NodeId::new(5));

    assert!(!result.success);
    assert_eq!(result.allocation, NodeId::new(5));
    assert_eq!(result.loads_eliminated, 0);
    assert_eq!(result.stores_eliminated, 0);
    assert!(result.killed_nodes.is_empty());
}

#[test]
fn test_scalar_replacement_result_success() {
    let result = ScalarReplacementResult {
        success: true,
        loads_eliminated: 5,
        stores_eliminated: 3,
        phis_created: 1,
        allocation: NodeId::new(10),
        killed_nodes: vec![NodeId::new(1), NodeId::new(2)],
    };

    assert!(result.success);
    assert_eq!(result.loads_eliminated, 5);
    assert_eq!(result.stores_eliminated, 3);
    assert_eq!(result.phis_created, 1);
}

// -------------------------------------------------------------------------
// ScalarReplacementConfig Tests
// -------------------------------------------------------------------------

#[test]
fn test_config_default() {
    let config = ScalarReplacementConfig::default();

    assert_eq!(config.max_fields, 64);
    assert_eq!(config.max_accesses, 256);
    assert!(config.create_defaults);
    assert!(!config.allow_partial);
}

#[test]
fn test_config_custom() {
    let config = ScalarReplacementConfig {
        max_fields: 32,
        max_accesses: 128,
        create_defaults: false,
        allow_partial: true,
    };

    assert_eq!(config.max_fields, 32);
    assert!(!config.create_defaults);
    assert!(config.allow_partial);
}

// -------------------------------------------------------------------------
// FieldValueTracker Tests
// -------------------------------------------------------------------------

#[test]
fn test_field_value_tracker_new() {
    let tracker = FieldValueTracker::new();
    assert!(tracker.values.is_empty());
    assert!(tracker.types.is_empty());
}

#[test]
fn test_field_value_tracker_set_get() {
    let mut tracker = FieldValueTracker::new();

    tracker.set(0, NodeId::new(10), ValueType::Int64);
    tracker.set(1, NodeId::new(11), ValueType::Float64);

    assert_eq!(tracker.get(0), Some(NodeId::new(10)));
    assert_eq!(tracker.get(1), Some(NodeId::new(11)));
    assert_eq!(tracker.get(2), None);
}

#[test]
fn test_field_value_tracker_get_or_default_existing() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    tracker.set(0, NodeId::new(10), ValueType::Int64);

    let value = tracker.get_or_default(&mut graph, 0, ValueType::Int64);
    assert_eq!(value, NodeId::new(10));
}

#[test]
fn test_field_value_tracker_get_or_default_int() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    let value = tracker.get_or_default(&mut graph, 0, ValueType::Int64);

    // Should create a const int 0
    let node = graph.get(value).unwrap();
    assert!(matches!(node.op, Operator::ConstInt(0)));
}

#[test]
fn test_field_value_tracker_get_or_default_float() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    let value = tracker.get_or_default(&mut graph, 0, ValueType::Float64);

    // Should create a const float 0.0
    let node = graph.get(value).unwrap();
    assert!(matches!(node.op, Operator::ConstFloat(_)));
}

#[test]
fn test_field_value_tracker_get_or_default_bool() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    let value = tracker.get_or_default(&mut graph, 0, ValueType::Bool);

    let node = graph.get(value).unwrap();
    assert!(matches!(node.op, Operator::ConstBool(false)));
}

#[test]
fn test_field_value_tracker_get_or_default_none() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    let value = tracker.get_or_default(&mut graph, 0, ValueType::None);

    let node = graph.get(value).unwrap();
    assert!(matches!(node.op, Operator::ConstNone));
}

#[test]
fn test_field_value_tracker_caches_defaults() {
    let mut tracker = FieldValueTracker::new();
    let mut graph = Graph::new();

    let value1 = tracker.get_or_default(&mut graph, 0, ValueType::Int64);
    let value2 = tracker.get_or_default(&mut graph, 0, ValueType::Int64);

    // Should return the same node
    assert_eq!(value1, value2);
}

// -------------------------------------------------------------------------
// ScalarReplacer Tests
// -------------------------------------------------------------------------

#[test]
fn test_scalar_replacer_new() {
    let replacer = ScalarReplacer::new();
    assert_eq!(replacer.config.max_fields, 64);
}

#[test]
fn test_scalar_replacer_with_config() {
    let config = ScalarReplacementConfig {
        max_fields: 16,
        ..Default::default()
    };
    let replacer = ScalarReplacer::with_config(config);
    assert_eq!(replacer.config.max_fields, 16);
}

#[test]
fn test_scalar_replacer_empty_graph() {
    let replacer = ScalarReplacer::new();
    let mut graph = Graph::new();

    // Try to replace a non-existent allocation
    let result = replacer.replace(&mut graph, NodeId::new(999));

    // Should fail (no accesses)
    assert!(!result.success);
}

// -------------------------------------------------------------------------
// AdvancedScalarReplacer Tests
// -------------------------------------------------------------------------

#[test]
fn test_advanced_replacer_new() {
    let replacer = AdvancedScalarReplacer::new();
    assert_eq!(replacer.base.config.max_fields, 64);
}

#[test]
fn test_advanced_replacer_empty_graph() {
    let replacer = AdvancedScalarReplacer::new();
    let mut graph = Graph::new();

    let result = replacer.replace(&mut graph, NodeId::new(999));

    assert!(!result.success);
}

// -------------------------------------------------------------------------
// Integration Tests
// -------------------------------------------------------------------------

#[test]
fn test_create_default_value_int64() {
    let mut graph = Graph::new();
    let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Int64);

    let node = graph.get(value).unwrap();
    assert!(matches!(node.op, Operator::ConstInt(0)));
    assert_eq!(node.ty, ValueType::Int64);
}

#[test]
fn test_create_default_value_float64() {
    let mut graph = Graph::new();
    let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Float64);

    let node = graph.get(value).unwrap();
    // Float 0.0 stored as bits
    assert!(matches!(node.op, Operator::ConstFloat(_)));
}

#[test]
fn test_create_default_value_unknown_type() {
    let mut graph = Graph::new();
    let value = FieldValueTracker::create_default_value(&mut graph, ValueType::Object);

    let node = graph.get(value).unwrap();
    // Unknown types default to None
    assert!(matches!(node.op, Operator::ConstNone));
}

#[test]
fn test_can_replace_dynamic_access() {
    let replacer = ScalarReplacer::new();

    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.has_dynamic_access = true;

    assert!(!replacer.can_replace(&tracker));
}

#[test]
fn test_can_replace_unknown_uses() {
    let replacer = ScalarReplacer::new();

    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.has_unknown_uses = true;

    assert!(!replacer.can_replace(&tracker));
}

#[test]
fn test_can_replace_no_accesses() {
    let replacer = ScalarReplacer::new();
    let tracker = FieldTracker::new(NodeId::new(0));

    assert!(!replacer.can_replace(&tracker));
}

#[test]
fn test_can_replace_too_many_fields() {
    let config = ScalarReplacementConfig {
        max_fields: 2,
        ..Default::default()
    };
    let replacer = ScalarReplacer::with_config(config);

    let mut tracker = FieldTracker::new(NodeId::new(0));
    // Add accesses for 3 fields
    tracker.stores_by_field.insert(0, SmallVec::new());
    tracker.stores_by_field.insert(1, SmallVec::new());
    tracker.stores_by_field.insert(2, SmallVec::new());
    tracker
        .accesses
        .push(super::super::field_tracking::FieldAccess {
            node: NodeId::new(1),
            object: NodeId::new(0),
            field: FieldIndex::Constant(0),
            kind: FieldAccessKind::Store,
            value: Some(NodeId::new(2)),
            value_type: ValueType::Int64,
            control: None,
        });

    assert!(!replacer.can_replace(&tracker));
}

#[test]
fn test_can_replace_too_many_accesses() {
    let config = ScalarReplacementConfig {
        max_accesses: 2,
        ..Default::default()
    };
    let replacer = ScalarReplacer::with_config(config);

    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.stores_by_field.insert(0, SmallVec::new());

    // Add 3 accesses
    for i in 0..3 {
        tracker
            .accesses
            .push(super::super::field_tracking::FieldAccess {
                node: NodeId::new(i + 1),
                object: NodeId::new(0),
                field: FieldIndex::Constant(0),
                kind: FieldAccessKind::Store,
                value: Some(NodeId::new(10)),
                value_type: ValueType::Int64,
                control: None,
            });
    }

    assert!(!replacer.can_replace(&tracker));
}

#[test]
fn test_can_replace_valid() {
    let replacer = ScalarReplacer::new();

    let mut tracker = FieldTracker::new(NodeId::new(0));
    tracker.stores_by_field.insert(0, SmallVec::new());
    tracker
        .accesses
        .push(super::super::field_tracking::FieldAccess {
            node: NodeId::new(1),
            object: NodeId::new(0),
            field: FieldIndex::Constant(0),
            kind: FieldAccessKind::Store,
            value: Some(NodeId::new(2)),
            value_type: ValueType::Int64,
            control: None,
        });

    assert!(replacer.can_replace(&tracker));
}
