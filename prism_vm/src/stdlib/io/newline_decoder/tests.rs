use super::*;

#[test]
fn test_incremental_newline_state_translates_and_tracks_all_newlines() {
    let mut state = IncrementalNewlineState {
        translate: true,
        ..Default::default()
    };

    let output = state.decode_text("a\r\nb\rc\n", true);

    assert_eq!(output, "a\nb\nc\n");
    assert_eq!(state.seen_nl, LF_MASK | CR_MASK | CRLF_MASK);
    assert!(!state.pending_cr);
}

#[test]
fn test_incremental_newline_state_preserves_pending_cr_between_chunks() {
    let mut state = IncrementalNewlineState {
        translate: true,
        ..Default::default()
    };

    let first = state.decode_text("line\r", false);
    let second = state.decode_text("\nnext", true);

    assert_eq!(first, "line");
    assert_eq!(second, "\nnext");
    assert_eq!(state.seen_nl, CRLF_MASK);
    assert!(!state.pending_cr);
}

#[test]
fn test_newlines_value_renders_expected_tuple_order() {
    let value = newlines_value(LF_MASK | CR_MASK | CRLF_MASK).expect("value should exist");
    let ptr = value.as_object_ptr().expect("tuple value should be boxed");
    let header = unsafe { &*(ptr as *const ObjectHeader) };
    assert_eq!(header.type_id, TypeId::TUPLE);

    let tuple = unsafe { &*(ptr as *const TupleObject) };
    let rendered: Vec<String> = tuple
        .as_slice()
        .iter()
        .map(|item| string_from_value(*item, "tuple entry").expect("entry should be str"))
        .collect();
    assert_eq!(rendered, vec!["\r", "\n", "\r\n"]);
}
