use super::*;
use crate::types::range::RangeObject;

fn leak_value<T>(obj: T) -> Value {
    let ptr = Box::into_raw(Box::new(obj)) as *const ();
    Value::object_ptr(ptr)
}

fn expect_interned_string(value: Value, expected: &str) {
    let ptr = value
        .as_string_object_ptr()
        .expect("iterator should yield an interned string") as *const u8;
    let actual = prism_core::intern::interned_by_ptr(ptr)
        .expect("iterator string should resolve through interner");
    assert_eq!(actual.as_str(), expected);
}

#[test]
fn test_empty_iterator() {
    let mut iter = IteratorObject::empty();
    assert!(iter.is_exhausted());
    assert!(iter.next().is_none());
}

#[test]
fn test_range_iterator() {
    let range = RangeObject::from_stop(5);
    let mut iter = IteratorObject::from_range(range.iter());

    let mut values = Vec::new();
    while let Some(v) = iter.next() {
        values.push(v.as_int().unwrap());
    }
    assert_eq!(values, vec![0, 1, 2, 3, 4]);
    assert!(iter.is_exhausted());
}

#[test]
fn test_list_iterator() {
    let list = leak_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));
    let mut iter = IteratorObject::from_list(list);

    let mut values = Vec::new();
    while let Some(v) = iter.next() {
        values.push(v.as_int().unwrap());
    }
    assert_eq!(values, vec![1, 2, 3]);
}

#[test]
fn test_list_iterator_observes_growth_after_creation() {
    let list = Box::into_raw(Box::new(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ])));
    let list_value = Value::object_ptr(list as *const ());
    let mut iter = IteratorObject::from_list(list_value);

    assert_eq!(iter.next().unwrap().as_int(), Some(1));
    unsafe { &mut *list }.push(Value::int(3).unwrap());

    assert_eq!(iter.size_hint(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
}

#[test]
fn test_tuple_iterator() {
    let tuple = leak_value(TupleObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let mut iter = IteratorObject::from_tuple(tuple);

    let mut values = Vec::new();
    while let Some(v) = iter.next() {
        values.push(v.as_int().unwrap());
    }
    assert_eq!(values, vec![10, 20, 30]);
}

#[test]
fn test_tagged_string_iterator_yields_unicode_characters() {
    let mut iter =
        IteratorObject::from_string_chars(Value::string(prism_core::intern::intern("aé🙂")));

    expect_interned_string(iter.next().unwrap(), "a");
    expect_interned_string(iter.next().unwrap(), "é");
    expect_interned_string(iter.next().unwrap(), "🙂");
    assert!(iter.next().is_none());
}

#[test]
fn test_values_iterator() {
    let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
    let mut iter = IteratorObject::from_values(values);

    assert_eq!(iter.size_hint(), Some(2));
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 100);
    assert_eq!(iter.size_hint(), Some(1));
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 200);
    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
}

#[test]
fn test_shared_iterator_proxies_remaining_items() {
    let source = leak_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));
    let iterator_value = leak_value(IteratorObject::from_list(source));
    let mut proxy = IteratorObject::from_existing_iterator(iterator_value);

    assert_eq!(proxy.size_hint(), Some(3));
    assert_eq!(proxy.next().unwrap().as_int(), Some(1));
    assert_eq!(proxy.size_hint(), Some(2));
    assert_eq!(proxy.next().unwrap().as_int(), Some(2));
    assert_eq!(proxy.next().unwrap().as_int(), Some(3));
    assert!(proxy.next().is_none());
    assert!(proxy.is_exhausted());
}

#[test]
fn test_shared_iterator_observes_underlying_progress() {
    let source = leak_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let iterator_value = leak_value(IteratorObject::from_list(source));
    let underlying = iterator_from_value_mut(iterator_value);
    assert_eq!(underlying.next().unwrap().as_int(), Some(10));
    let mut proxy = IteratorObject::from_existing_iterator(iterator_value);

    assert_eq!(proxy.next().unwrap().as_int(), Some(20));
    assert_eq!(proxy.next().unwrap().as_int(), Some(30));
    assert!(proxy.next().is_none());
}

#[test]
fn test_count_iterator_yields_unbounded_progression() {
    let mut iter = IteratorObject::count(Value::int(3).unwrap(), Value::int(2).unwrap())
        .expect("count iterator should construct");

    assert_eq!(format!("{:?}", iter), "<count>");
    assert_eq!(iter.size_hint(), None);
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert_eq!(iter.next().unwrap().as_int(), Some(5));
    assert_eq!(iter.next().unwrap().as_int(), Some(7));
    assert!(!iter.is_exhausted());
}

#[test]
fn test_repeat_iterator_tracks_bounded_remaining_length() {
    let mut iter = IteratorObject::repeat(Value::string(prism_core::intern::intern("x")), Some(2));

    assert_eq!(format!("{:?}", iter), "<repeat>");
    assert_eq!(iter.size_hint(), Some(2));
    expect_interned_string(iter.next().unwrap(), "x");
    assert_eq!(iter.size_hint(), Some(1));
    expect_interned_string(iter.next().unwrap(), "x");
    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
    assert!(iter.is_exhausted());
}

#[test]
fn test_repeat_iterator_can_be_unbounded() {
    let mut iter = IteratorObject::repeat(Value::int(9).unwrap(), None);

    assert_eq!(iter.size_hint(), None);
    assert_eq!(iter.next().unwrap().as_int(), Some(9));
    assert_eq!(iter.next().unwrap().as_int(), Some(9));
    assert!(!iter.is_exhausted());
}

#[test]
fn test_chain_iterator_yields_sources_in_order() {
    let first = IteratorObject::from_values(vec![Value::int(1).unwrap()]);
    let second = IteratorObject::from_values(vec![Value::int(2).unwrap(), Value::int(3).unwrap()]);
    let mut iter = IteratorObject::chain(vec![first, second]);

    assert_eq!(format!("{:?}", iter), "<chain>");
    assert_eq!(iter.size_hint(), Some(3));
    assert_eq!(iter.next().unwrap().as_int(), Some(1));
    assert_eq!(iter.size_hint(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(3));
    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
    assert!(iter.is_exhausted());
}

#[test]
fn test_chain_iterator_skips_empty_sources() {
    let first = IteratorObject::empty();
    let second = IteratorObject::from_values(vec![Value::int(42).unwrap()]);
    let third = IteratorObject::empty();
    let mut iter = IteratorObject::chain(vec![first, second, third]);

    assert_eq!(iter.next().unwrap().as_int(), Some(42));
    assert!(iter.next().is_none());
    assert!(iter.is_exhausted());
}

#[test]
fn test_bytes_iterator_yields_ints_and_updates_hint() {
    let bytes = leak_value(BytesObject::from_slice(&[0, 65, 255]));
    let mut iter = IteratorObject::from_bytes(bytes);

    assert_eq!(iter.size_hint(), Some(3));
    assert_eq!(iter.next().unwrap().as_int(), Some(0));
    assert_eq!(iter.size_hint(), Some(2));
    assert_eq!(iter.next().unwrap().as_int(), Some(65));
    assert_eq!(iter.next().unwrap().as_int(), Some(255));
    assert_eq!(iter.size_hint(), Some(0));
    assert!(iter.next().is_none());
}

#[test]
fn test_collect_remaining() {
    let range = RangeObject::new(0, 5, 1);
    let mut iter = IteratorObject::from_range(range.iter());

    // Consume first two
    iter.next();
    iter.next();

    // Collect remaining
    let remaining = iter.collect_remaining();
    assert_eq!(remaining.len(), 3);
    assert_eq!(remaining[0].as_int().unwrap(), 2);
    assert_eq!(remaining[1].as_int().unwrap(), 3);
    assert_eq!(remaining[2].as_int().unwrap(), 4);
}

#[test]
fn test_iterator_debug() {
    let iter = IteratorObject::empty();
    let debug = format!("{:?}", iter);
    assert!(debug.contains("empty_iterator"));
}

// =========================================================================
// Composite iterator tests (Phase 3.4)
// =========================================================================

#[test]
fn test_enumerate_basic() {
    let list = leak_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let inner = IteratorObject::from_list(list);
    let mut enumerate = IteratorObject::enumerate(inner, 0);

    assert_eq!(format!("{:?}", enumerate), "<enumerate>");

    // First: (0, 10)
    let pair1 = enumerate.next().unwrap();
    assert!(!pair1.is_none());

    // Second: (1, 20)
    let pair2 = enumerate.next().unwrap();
    assert!(!pair2.is_none());

    // Third: (2, 30)
    let pair3 = enumerate.next().unwrap();
    assert!(!pair3.is_none());

    // Exhausted
    assert!(enumerate.next().is_none());
    assert!(enumerate.is_exhausted());
}

#[test]
fn test_enumerate_with_start() {
    let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
    let inner = IteratorObject::from_values(values);
    let mut enumerate = IteratorObject::enumerate(inner, 5);

    // First: (5, 100)
    let pair1 = enumerate.next().unwrap();
    assert!(!pair1.is_none());

    // Second: (6, 200)
    let pair2 = enumerate.next().unwrap();
    assert!(!pair2.is_none());

    assert!(enumerate.next().is_none());
}

#[test]
fn test_enumerate_empty() {
    let inner = IteratorObject::empty();
    let mut enumerate = IteratorObject::enumerate(inner, 0);
    assert!(enumerate.next().is_none());
}

#[test]
fn test_zip_two_iterators() {
    let list1 = leak_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ]));
    let list2 = leak_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let iter1 = IteratorObject::from_list(list1);
    let iter2 = IteratorObject::from_list(list2);
    let mut zip_iter = IteratorObject::zip(vec![iter1, iter2]);

    assert_eq!(format!("{:?}", zip_iter), "<zip>");

    // Should yield 3 tuples
    let t1 = zip_iter.next();
    assert!(t1.is_some());
    let t2 = zip_iter.next();
    assert!(t2.is_some());
    let t3 = zip_iter.next();
    assert!(t3.is_some());

    assert!(zip_iter.next().is_none());
    assert!(zip_iter.is_exhausted());
}

#[test]
fn test_zip_unequal_lengths() {
    // Short iterator
    let list1 = leak_value(ListObject::from_slice(&[Value::int(1).unwrap()]));
    // Long iterator
    let list2 = leak_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let iter1 = IteratorObject::from_list(list1);
    let iter2 = IteratorObject::from_list(list2);
    let mut zip_iter = IteratorObject::zip(vec![iter1, iter2]);

    // Only 1 element because first iterator has only 1
    assert!(zip_iter.next().is_some());
    assert!(zip_iter.next().is_none());
}

#[test]
fn test_zip_empty() {
    let mut zip_iter = IteratorObject::zip(vec![]);
    assert!(zip_iter.next().is_none());
    assert!(zip_iter.is_exhausted());
}

#[test]
fn test_zip_size_hint() {
    let list1 = leak_value(ListObject::from_slice(&[
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
    ]));
    let list2 = leak_value(ListObject::from_slice(&[
        Value::int(10).unwrap(),
        Value::int(20).unwrap(),
        Value::int(30).unwrap(),
    ]));
    let iter1 = IteratorObject::from_list(list1);
    let iter2 = IteratorObject::from_list(list2);
    let zip_iter = IteratorObject::zip(vec![iter1, iter2]);

    // Should be minimum of the two
    assert_eq!(zip_iter.size_hint(), Some(2));
}

#[test]
fn test_reversed_basic() {
    let values = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let mut reversed = IteratorObject::reversed(values);

    assert_eq!(format!("{:?}", reversed), "<reversed>");
    assert_eq!(reversed.size_hint(), Some(3));

    assert_eq!(reversed.next().unwrap().as_int().unwrap(), 3);
    assert_eq!(reversed.next().unwrap().as_int().unwrap(), 2);
    assert_eq!(reversed.next().unwrap().as_int().unwrap(), 1);
    assert!(reversed.next().is_none());
    assert!(reversed.is_exhausted());
}

#[test]
fn test_reversed_empty() {
    let mut reversed = IteratorObject::reversed(vec![]);
    assert!(reversed.is_exhausted());
    assert!(reversed.next().is_none());
    assert_eq!(reversed.size_hint(), Some(0));
}

#[test]
fn test_reversed_single() {
    let values = vec![Value::int(42).unwrap()];
    let mut reversed = IteratorObject::reversed(values);

    assert_eq!(reversed.next().unwrap().as_int().unwrap(), 42);
    assert!(reversed.next().is_none());
}

#[test]
fn test_dict_keys_iterator() {
    let keys = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let mut iter = IteratorObject::dict_keys(keys);

    assert_eq!(format!("{:?}", iter), "<dict_keys>");
    assert_eq!(iter.size_hint(), Some(3));

    assert_eq!(iter.next().unwrap().as_int().unwrap(), 1);
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 2);
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 3);
    assert!(iter.next().is_none());
}

#[test]
fn test_dict_values_iterator() {
    let values = vec![Value::int(100).unwrap(), Value::int(200).unwrap()];
    let mut iter = IteratorObject::dict_values(values);

    assert_eq!(format!("{:?}", iter), "<dict_values>");

    assert_eq!(iter.next().unwrap().as_int().unwrap(), 100);
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 200);
    assert!(iter.next().is_none());
}

#[test]
fn test_dict_items_iterator() {
    let items = vec![
        (Value::int(1).unwrap(), Value::int(100).unwrap()),
        (Value::int(2).unwrap(), Value::int(200).unwrap()),
    ];
    let mut iter = IteratorObject::dict_items(items);

    assert_eq!(format!("{:?}", iter), "<dict_items>");
    assert_eq!(iter.size_hint(), Some(2));

    // Returns tuples
    let item1 = iter.next();
    assert!(item1.is_some());
    let item2 = iter.next();
    assert!(item2.is_some());
    assert!(iter.next().is_none());
}

#[test]
fn test_set_iterator() {
    let values = vec![
        Value::int(5).unwrap(),
        Value::int(10).unwrap(),
        Value::int(15).unwrap(),
    ];
    let mut iter = IteratorObject::set_iter(values);

    assert_eq!(format!("{:?}", iter), "<set_iterator>");

    assert_eq!(iter.next().unwrap().as_int().unwrap(), 5);
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 10);
    assert_eq!(iter.next().unwrap().as_int().unwrap(), 15);
    assert!(iter.next().is_none());
}

#[test]
fn test_filter_identity() {
    // Identity filter: filters out falsy values
    let values = vec![
        Value::int(0).unwrap(), // falsy
        Value::int(1).unwrap(), // truthy
        Value::int(0).unwrap(), // falsy
        Value::int(2).unwrap(), // truthy
        Value::none(),          // falsy
        Value::int(3).unwrap(), // truthy
    ];
    let inner = IteratorObject::from_values(values);
    let mut filter = IteratorObject::filter(None, inner);

    assert_eq!(format!("{:?}", filter), "<filter>");

    // Should only yield truthy values: 1, 2, 3
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 1);
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 2);
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 3);
    assert!(filter.next().is_none());
}

#[test]
fn test_filter_all_falsy() {
    let values = vec![
        Value::int(0).unwrap(),
        Value::none(),
        Value::int(0).unwrap(),
    ];
    let inner = IteratorObject::from_values(values);
    let mut filter = IteratorObject::filter(None, inner);

    // All falsy, should yield nothing
    assert!(filter.next().is_none());
}

#[test]
fn test_filter_all_truthy() {
    let values = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let inner = IteratorObject::from_values(values);
    let mut filter = IteratorObject::filter(None, inner);

    // All truthy
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 1);
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 2);
    assert_eq!(filter.next().unwrap().as_int().unwrap(), 3);
    assert!(filter.next().is_none());
}

#[test]
fn test_islice_with_finite_stop_skips_and_steps() {
    let values = vec![
        Value::int(0).unwrap(),
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
        Value::int(4).unwrap(),
        Value::int(5).unwrap(),
    ];
    let inner = IteratorObject::from_values(values);
    let mut islice = IteratorObject::islice(inner, 1, Some(6), 2);

    assert_eq!(format!("{:?}", islice), "<islice>");
    assert_eq!(islice.size_hint(), Some(3));
    assert_eq!(islice.next().unwrap().as_int(), Some(1));
    assert_eq!(islice.next().unwrap().as_int(), Some(3));
    assert_eq!(islice.next().unwrap().as_int(), Some(5));
    assert_eq!(islice.size_hint(), Some(0));
    assert!(islice.next().is_none());
}

#[test]
fn test_islice_without_stop_consumes_remaining_source() {
    let values = vec![
        Value::int(10).unwrap(),
        Value::int(11).unwrap(),
        Value::int(12).unwrap(),
        Value::int(13).unwrap(),
    ];
    let inner = IteratorObject::from_values(values);
    let mut islice = IteratorObject::islice(inner, 2, None, 1);

    assert_eq!(islice.next().unwrap().as_int(), Some(12));
    assert_eq!(islice.next().unwrap().as_int(), Some(13));
    assert!(islice.next().is_none());
}

#[test]
fn test_map_basic() {
    // Map iterator stores func but returns raw values for VM to process
    let values = vec![Value::int(1).unwrap(), Value::int(2).unwrap()];
    let inner = IteratorObject::from_values(values);
    let map_iter = IteratorObject::map(Value::none(), inner);

    assert_eq!(format!("{:?}", map_iter), "<map>");
}

#[test]
fn test_enumerate_size_hint() {
    let values = vec![
        Value::int(1).unwrap(),
        Value::int(2).unwrap(),
        Value::int(3).unwrap(),
    ];
    let inner = IteratorObject::from_values(values);
    let enumerate = IteratorObject::enumerate(inner, 0);

    assert_eq!(enumerate.size_hint(), Some(3));
}

#[test]
fn test_composite_iterator_debug_formats() {
    // Verify all debug formats are correct
    let empty_vals: Vec<Value> = vec![];
    let single_val = vec![Value::int(1).unwrap()];

    assert!(
        format!(
            "{:?}",
            IteratorObject::enumerate(IteratorObject::empty(), 0)
        )
        .contains("enumerate")
    );
    assert!(format!("{:?}", IteratorObject::zip(vec![])).contains("zip"));
    assert!(
        format!(
            "{:?}",
            IteratorObject::map(Value::none(), IteratorObject::empty())
        )
        .contains("map")
    );
    assert!(
        format!(
            "{:?}",
            IteratorObject::filter(None, IteratorObject::empty())
        )
        .contains("filter")
    );
    assert!(format!("{:?}", IteratorObject::reversed(empty_vals.clone())).contains("reversed"));
    assert!(format!("{:?}", IteratorObject::dict_keys(empty_vals.clone())).contains("dict_keys"));
    assert!(
        format!("{:?}", IteratorObject::dict_values(empty_vals.clone())).contains("dict_values")
    );
    assert!(format!("{:?}", IteratorObject::dict_items(vec![])).contains("dict_items"));
    assert!(format!("{:?}", IteratorObject::set_iter(single_val)).contains("set_iterator"));
}
