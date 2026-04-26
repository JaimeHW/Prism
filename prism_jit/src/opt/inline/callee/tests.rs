use super::*;
use crate::ir::builder::{ControlBuilder, GraphBuilder};

fn make_test_graph(param_count: usize) -> Graph {
    let mut builder = GraphBuilder::new(8, param_count);
    if param_count > 0 {
        let p0 = builder.parameter(0).unwrap();
        builder.return_value(p0);
    }
    builder.finish()
}

#[test]
fn test_callee_graph_new() {
    let graph = make_test_graph(2);
    let callee = CalleeGraph::new(graph, 2);

    assert_eq!(callee.param_count, 2);
    assert_eq!(callee.inline_hint, InlineHint::Default);
    assert!(!callee.is_intrinsic);
}

#[test]
fn test_callee_graph_with_hint() {
    let graph = make_test_graph(1);
    let callee = CalleeGraph::new(graph, 1).with_hint(InlineHint::Always);

    assert_eq!(callee.inline_hint, InlineHint::Always);
}

#[test]
fn test_callee_graph_as_intrinsic() {
    let graph = make_test_graph(1);
    let callee = CalleeGraph::new(graph, 1).as_intrinsic();

    assert!(callee.is_intrinsic);
}

#[test]
fn test_callee_registry_register() {
    let registry = CalleeRegistry::new();
    let graph = make_test_graph(2);
    let callee = CalleeGraph::new(graph, 2);

    registry.register(42, callee);

    assert!(registry.has_function(42));
    assert!(!registry.has_function(99));
}

#[test]
fn test_callee_registry_get_graph() {
    let registry = CalleeRegistry::new();
    let graph = make_test_graph(3);
    let callee = CalleeGraph::new(graph, 3);

    registry.register(1, callee);

    let retrieved = registry.get_graph(1);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().param_count, 3);
}

#[test]
fn test_callee_registry_unregister() {
    let registry = CalleeRegistry::new();
    let graph = make_test_graph(1);
    let callee = CalleeGraph::new(graph, 1);

    registry.register(5, callee);
    assert!(registry.has_function(5));

    let removed = registry.unregister(5);
    assert!(removed.is_some());
    assert!(!registry.has_function(5));
}

#[test]
fn test_callee_registry_clear() {
    let registry = CalleeRegistry::new();

    for i in 0..5 {
        let graph = make_test_graph(1);
        registry.register(i, CalleeGraph::new(graph, 1));
    }

    assert_eq!(registry.len(), 5);

    registry.clear();
    assert!(registry.is_empty());
}

#[test]
fn test_callee_provider_param_count() {
    let registry = CalleeRegistry::new();
    let graph = make_test_graph(4);
    registry.register(10, CalleeGraph::new(graph, 4));

    assert_eq!(registry.param_count(10), Some(4));
    assert_eq!(registry.param_count(99), None);
}

#[test]
fn test_callee_provider_inline_hint() {
    let registry = CalleeRegistry::new();
    let graph = make_test_graph(1);
    let callee = CalleeGraph::new(graph, 1).with_hint(InlineHint::Hot);
    registry.register(20, callee);

    assert_eq!(registry.inline_hint(20), InlineHint::Hot);
    assert_eq!(registry.inline_hint(99), InlineHint::Default);
}

#[test]
fn test_intrinsic_provider() {
    let mut provider = IntrinsicProvider::new();
    let graph = make_test_graph(1);
    provider.register(100, CalleeGraph::new(graph, 1));

    assert!(provider.has_function(100));
    assert!(provider.is_intrinsic(100));
}

#[test]
fn test_composite_provider() {
    let registry1 = Arc::new(CalleeRegistry::new());
    let registry2 = Arc::new(CalleeRegistry::new());

    let graph1 = make_test_graph(1);
    registry1.register(1, CalleeGraph::new(graph1, 1));

    let graph2 = make_test_graph(2);
    registry2.register(2, CalleeGraph::new(graph2, 2));

    let composite = CompositeProvider::new()
        .add_provider(registry1 as Arc<dyn CalleeProvider>)
        .add_provider(registry2 as Arc<dyn CalleeProvider>);

    assert!(composite.has_function(1));
    assert!(composite.has_function(2));
    assert!(!composite.has_function(3));

    assert_eq!(composite.param_count(1), Some(1));
    assert_eq!(composite.param_count(2), Some(2));
}

#[test]
fn test_composite_provider_priority() {
    let registry1 = Arc::new(CalleeRegistry::new());
    let registry2 = Arc::new(CalleeRegistry::new());

    // Both have the same function ID
    let graph1 = make_test_graph(1);
    registry1.register(1, CalleeGraph::new(graph1, 1).with_hint(InlineHint::Always));

    let graph2 = make_test_graph(2);
    registry2.register(1, CalleeGraph::new(graph2, 2).with_hint(InlineHint::Never));

    let composite = CompositeProvider::new()
        .add_provider(registry1 as Arc<dyn CalleeProvider>)
        .add_provider(registry2 as Arc<dyn CalleeProvider>);

    // Should get the first one (registry1)
    let graph = composite.get_graph(1).unwrap();
    assert_eq!(graph.param_count, 1);
    assert_eq!(graph.inline_hint, InlineHint::Always);
}
