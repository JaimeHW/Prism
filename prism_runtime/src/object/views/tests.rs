use super::*;

#[test]
fn test_code_object_view_uses_code_type_id() {
    let code = Arc::new(CodeObject::new("demo", "<test>"));
    let view = CodeObjectView::new(Arc::clone(&code));
    assert_eq!(view.header().type_id, TypeId::CODE);
    assert!(Arc::ptr_eq(view.code(), &code));
}

#[test]
fn test_frame_view_uses_frame_type_id() {
    let code = Arc::new(CodeObject::new("demo", "<test>"));
    let globals = Value::int(11).unwrap();
    let locals = Value::int(13).unwrap();
    let view = FrameViewObject::new(Some(Arc::clone(&code)), globals, locals, 12, 7, None);
    assert_eq!(view.header().type_id, TypeId::FRAME);
    assert!(Arc::ptr_eq(
        view.code().expect("code should be present"),
        &code
    ));
    assert_eq!(view.globals(), globals);
    assert_eq!(view.locals(), locals);
    assert_eq!(view.line_number(), 12);
    assert_eq!(view.lasti(), 7);
    assert_eq!(view.back(), None);
}

#[test]
fn test_traceback_view_uses_traceback_type_id() {
    let frame = Value::int(7).unwrap();
    let mut view = TracebackViewObject::new(frame, Some(Value::none()), 12, 3);
    assert_eq!(view.header().type_id, TypeId::TRACEBACK);
    assert_eq!(view.frame(), frame);
    assert_eq!(view.next(), Some(Value::none()));
    assert_eq!(view.line_number(), 12);
    assert_eq!(view.lasti(), 3);

    view.set_next(None);
    assert_eq!(view.next(), None);
}

#[test]
fn test_cell_view_uses_internal_cell_view_type_id() {
    let cell = Arc::new(Cell::new(Value::int_unchecked(42)));
    let view = CellViewObject::new(Arc::clone(&cell));
    assert_eq!(view.header().type_id, TypeId::CELL_VIEW);
    assert_eq!(view.cell().get_or_none().as_int(), Some(42));
}

#[test]
fn test_generic_alias_preserves_origin_and_args() {
    let alias = GenericAliasObject::new(
        Value::int_unchecked(7),
        vec![Value::int_unchecked(1), Value::int_unchecked(2)],
    );
    assert_eq!(alias.header().type_id, TypeId::GENERIC_ALIAS);
    assert_eq!(alias.origin().as_int(), Some(7));
    assert_eq!(alias.args().len(), 2);
}

#[test]
fn test_union_type_preserves_member_order() {
    let union = UnionTypeObject::new(vec![TypeId::INT, TypeId::STR]);
    assert_eq!(union.header().type_id, TypeId::UNION);
    assert_eq!(union.members(), &[TypeId::INT, TypeId::STR]);
}

#[test]
fn test_singleton_object_supports_ellipsis_type_id() {
    let singleton = SingletonObject::new(TypeId::ELLIPSIS);
    assert_eq!(singleton.header().type_id, TypeId::ELLIPSIS);
}

#[test]
fn test_singleton_object_supports_not_implemented_type_id() {
    let singleton = SingletonObject::new(TypeId::NOT_IMPLEMENTED);
    assert_eq!(singleton.header().type_id, TypeId::NOT_IMPLEMENTED);
}

#[test]
fn test_mapping_proxy_records_source() {
    let proxy = MappingProxyObject::for_builtin_type(TypeId::TYPE);
    assert_eq!(proxy.header().type_id, TypeId::MAPPING_PROXY);
    assert_eq!(
        proxy.source(),
        MappingProxySource::BuiltinType(TypeId::TYPE)
    );
}

#[test]
fn test_dict_view_preserves_kind_and_backing_dict_value() {
    let dict = Value::int_unchecked(7);
    let view = DictViewObject::new(DictViewKind::Keys, dict);
    assert_eq!(view.header().type_id, TypeId::DICT_KEYS);
    assert_eq!(view.dict(), dict);
    assert_eq!(view.kind(), DictViewKind::Keys);
}

#[test]
fn test_descriptor_view_preserves_owner_and_name() {
    let view = DescriptorViewObject::new(
        TypeId::METHOD_DESCRIPTOR,
        TypeId::STR,
        prism_core::intern::intern("join"),
    );
    assert_eq!(view.header().type_id, TypeId::METHOD_DESCRIPTOR);
    assert_eq!(view.owner(), TypeId::STR);
    assert_eq!(view.name().as_str(), "join");
}

#[test]
fn test_method_wrapper_preserves_receiver() {
    let view = MethodWrapperObject::new(
        TypeId::OBJECT,
        prism_core::intern::intern("__str__"),
        Value::int_unchecked(7),
    );
    assert_eq!(view.header().type_id, TypeId::METHOD_WRAPPER);
    assert_eq!(view.owner(), TypeId::OBJECT);
    assert_eq!(view.name().as_str(), "__str__");
    assert_eq!(view.receiver().as_int(), Some(7));
}
