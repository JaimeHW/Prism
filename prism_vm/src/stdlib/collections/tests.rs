//! Integration tests for the collections module.

#[cfg(test)]
mod integration_tests {
    use crate::stdlib::collections::counter::Counter;
    use crate::stdlib::collections::defaultdict::{DefaultDict, DefaultFactory};
    use crate::stdlib::collections::deque::Deque;
    use crate::stdlib::collections::ordereddict::OrderedDict;
    use prism_core::Value;
    use prism_core::intern::intern;

    fn int(i: i64) -> Value {
        Value::int_unchecked(i)
    }

    fn str_val(s: &str) -> Value {
        Value::string(intern(s))
    }

    // =========================================================================
    // Cross-Type Interaction Tests
    // =========================================================================

    #[test]
    fn test_counter_from_deque() {
        let mut d = Deque::new();
        d.append(str_val("a"));
        d.append(str_val("b"));
        d.append(str_val("a"));
        d.append(str_val("c"));
        d.append(str_val("a"));

        let c = Counter::from_iter(d.into_iter());
        assert_eq!(c.get(&str_val("a")), 3);
        assert_eq!(c.get(&str_val("b")), 1);
        assert_eq!(c.get(&str_val("c")), 1);
    }

    #[test]
    fn test_deque_from_ordereddict_keys() {
        let mut od = OrderedDict::new();
        od.set(int(1), str_val("a"));
        od.set(int(2), str_val("b"));
        od.set(int(3), str_val("c"));

        let d: Deque = od.keys().cloned().collect();
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn test_counter_most_common_in_deque() {
        let c = Counter::from_iter(vec![int(1), int(1), int(1), int(2), int(2), int(3)]);

        let mc = c.most_common();
        let mut d = Deque::new();
        for (val, count) in mc {
            d.append(val);
            d.append(int(count));
        }

        assert_eq!(d.len(), 6); // 3 pairs
    }

    // =========================================================================
    // Pattern Tests - Common Python Patterns
    // =========================================================================

    #[test]
    fn test_word_frequency_counter() {
        // Simulating: Counter(words)
        let words = vec![
            str_val("the"),
            str_val("quick"),
            str_val("brown"),
            str_val("fox"),
            str_val("the"),
            str_val("the"),
        ];

        let c = Counter::from_iter(words);
        assert_eq!(c.get(&str_val("the")), 3);
        assert_eq!(c.get(&str_val("quick")), 1);
    }

    #[test]
    fn test_defaultdict_grouping() {
        // Simulating: d = defaultdict(list); d[key].append(value)
        let mut d = DefaultDict::with_factory(DefaultFactory::Int);

        // In Python: for word in words: d[len(word)] += 1
        let words = vec!["a", "bb", "ccc", "dd", "e", "fff"];
        for word in words {
            let len_key = int(word.len() as i64);
            let count = d.get_or_insert(&len_key).unwrap().as_int().unwrap();
            d.set(len_key, int(count + 1));
        }

        assert_eq!(d.get(&int(1)).and_then(|v| v.as_int()), Some(2)); // "a", "e"
        assert_eq!(d.get(&int(2)).and_then(|v| v.as_int()), Some(2)); // "bb", "dd"
        assert_eq!(d.get(&int(3)).and_then(|v| v.as_int()), Some(2)); // "ccc", "fff"
    }

    #[test]
    fn test_ordereddict_lru_cache_pattern() {
        // LRU cache pattern with maxsize
        let mut cache = OrderedDict::new();
        let maxsize = 3;

        let access_order = vec![(int(1), int(100)), (int(2), int(200)), (int(3), int(300))];

        for (key, value) in access_order {
            cache.set(key, value);
        }

        // Access key 1 - move to end
        cache.move_to_end(&int(1), true);

        // Add new key - should evict first (key 2)
        if cache.len() >= maxsize {
            cache.popitem(false);
        }
        cache.set(int(4), int(400));

        assert!(!cache.contains(&int(2))); // Evicted
        assert!(cache.contains(&int(1))); // Kept (accessed recently)
        assert!(cache.contains(&int(3)));
        assert!(cache.contains(&int(4)));
    }

    #[test]
    fn test_deque_sliding_window() {
        // Sliding window pattern with maxlen
        let mut window = Deque::with_maxlen(3);

        for i in 0..5 {
            window.append(int(i));
        }

        // Should have [2, 3, 4]
        assert_eq!(window.len(), 3);
        assert_eq!(window.get(0).and_then(|v| v.as_int()), Some(2));
        assert_eq!(window.get(-1).and_then(|v| v.as_int()), Some(4));
    }

    #[test]
    fn test_deque_undo_redo_stack() {
        // Undo/redo pattern with two deques
        let mut undo_stack = Deque::new();
        let mut redo_stack = Deque::new();

        // Actions
        undo_stack.append(str_val("action1"));
        undo_stack.append(str_val("action2"));
        undo_stack.append(str_val("action3"));

        // Undo
        if let Some(action) = undo_stack.pop() {
            redo_stack.append(action);
        }

        // Redo
        if let Some(action) = redo_stack.pop() {
            undo_stack.append(action);
        }

        assert_eq!(undo_stack.len(), 3);
        assert!(redo_stack.is_empty());
    }

    // =========================================================================
    // Performance Pattern Tests
    // =========================================================================

    #[test]
    fn test_deque_fifo_queue() {
        let mut queue = Deque::new();

        // Enqueue
        for i in 0..100 {
            queue.append(int(i));
        }

        // Dequeue
        for i in 0..100 {
            let val = queue.popleft().unwrap();
            assert_eq!(val.as_int(), Some(i));
        }

        assert!(queue.is_empty());
    }

    #[test]
    fn test_deque_lifo_stack() {
        let mut stack = Deque::new();

        // Push
        for i in 0..100 {
            stack.append(int(i));
        }

        // Pop
        for i in (0..100).rev() {
            let val = stack.pop().unwrap();
            assert_eq!(val.as_int(), Some(i));
        }

        assert!(stack.is_empty());
    }

    #[test]
    fn test_counter_subtract_pattern() {
        // Available - required pattern
        let available = Counter::from_pairs(vec![
            (str_val("a"), 5),
            (str_val("b"), 3),
            (str_val("c"), 2),
        ]);

        let required = Counter::from_pairs(vec![(str_val("a"), 3), (str_val("b"), 1)]);

        let remaining = available.sub(&required);

        assert_eq!(remaining.get(&str_val("a")), 2);
        assert_eq!(remaining.get(&str_val("b")), 2);
        assert_eq!(remaining.get(&str_val("c")), 2);
    }

    // =========================================================================
    // Empty/Edge Case Tests
    // =========================================================================

    #[test]
    fn test_empty_deque_operations() {
        let d = Deque::new();
        assert!(d.is_empty());
        assert_eq!(d.front(), None);
        assert_eq!(d.back(), None);
        assert_eq!(d.get(0), None);
    }

    #[test]
    fn test_empty_counter_operations() {
        let c = Counter::new();
        assert!(c.is_empty());
        assert_eq!(c.get(&int(1)), 0);
        assert_eq!(c.total(), 0);
        assert!(c.most_common().is_empty());
    }

    #[test]
    fn test_empty_defaultdict_operations() {
        let d = DefaultDict::new();
        assert!(d.is_empty());
        assert_eq!(d.get(&int(1)), None);
    }

    #[test]
    fn test_empty_ordereddict_operations() {
        let mut od = OrderedDict::new();
        assert!(od.is_empty());
        assert_eq!(od.get(&int(1)), None);
        assert_eq!(od.popitem(true), None);
    }
}
