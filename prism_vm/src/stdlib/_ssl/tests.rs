use super::*;
use prism_runtime::types::bytes::BytesObject;

fn builtin_from_value(value: Value) -> &'static BuiltinFunctionObject {
    let ptr = value.as_object_ptr().expect("expected builtin function");
    unsafe { &*(ptr as *const BuiltinFunctionObject) }
}

#[test]
fn test_ssl_module_exposes_import_time_surface() {
    let module = SslModule::new();
    assert!(module.get_attr("SSLError").is_ok());
    assert!(module.get_attr("SSLWantReadError").is_ok());
    assert!(module.get_attr("_SSLContext").is_ok());
    assert_eq!(module.get_attr("HAS_SNI").unwrap().as_bool(), Some(true));
    assert_eq!(
        module.get_attr("PROTOCOL_TLS_CLIENT").unwrap().as_int(),
        Some(PROTOCOL_TLS_CLIENT)
    );
}

#[test]
fn test_ssl_txt2obj_supports_purpose_oids() {
    let txt2obj = builtin_from_value(SslModule::new().get_attr("txt2obj").unwrap());
    let result = txt2obj
        .call_with_keywords(
            &[Value::string(intern("1.3.6.1.5.5.7.3.1"))],
            &[("name", Value::bool(false))],
        )
        .expect("txt2obj should resolve server auth");
    let tuple = unsafe { &*(result.as_object_ptr().unwrap() as *const TupleObject) };
    assert_eq!(tuple.len(), 4);
    assert_eq!(tuple.get(0).and_then(|value| value.as_int()), Some(129));
}

#[test]
fn test_ssl_rand_bytes_returns_requested_size() {
    let rand_bytes = builtin_from_value(SslModule::new().get_attr("RAND_bytes").unwrap());
    let result = rand_bytes
        .call(&[Value::int(8).unwrap()])
        .expect("RAND_bytes should succeed");
    let bytes = unsafe { &*(result.as_object_ptr().unwrap() as *const BytesObject) };
    assert_eq!(bytes.len(), 8);
}

#[test]
fn test_ssl_context_new_allocates_shaped_context_instances() {
    let class_value = SslModule::new().get_attr("_SSLContext").unwrap();
    let value = ssl_context_new(&[class_value, Value::int(PROTOCOL_TLS_CLIENT).unwrap()])
        .expect("_SSLContext.__new__ should allocate");
    let object = expect_ssl_context_instance(value).expect("context instance");
    assert_eq!(
        object
            .get_property(SSL_CONTEXT_PROTOCOL_ATTR)
            .and_then(|value| value.as_int()),
        Some(PROTOCOL_TLS_CLIENT)
    );
}
