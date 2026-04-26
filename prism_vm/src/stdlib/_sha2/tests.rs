use super::*;
use prism_runtime::types::string::value_as_string_ref;

fn bytes_value(bytes: &[u8]) -> Value {
    to_object_value(BytesObject::from_slice(bytes))
}

fn string_value(value: Value) -> String {
    value_as_string_ref(value)
        .expect("value should be a string")
        .as_str()
        .to_string()
}

#[test]
fn test_module_exposes_sha512_surface() {
    let module = Sha2Module::new();
    assert!(module.get_attr("sha512").is_ok());

    let class_value = module
        .get_attr("SHA512Type")
        .expect("SHA512Type should exist");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("SHA512Type should be a class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };
    assert_eq!(class.name().as_str(), "SHA512Type");
}

#[test]
fn test_sha512_constructor_hashes_initial_bytes() {
    let value = hash_constructor(Sha2Kind::Sha512, &[bytes_value(b"prism")], &[])
        .expect("sha512 should construct");
    let digest = hash_digest(&[value]).expect("digest should succeed");
    let bytes = bytes_argument(digest, "digest").expect("digest should return bytes");

    assert_eq!(bytes, Sha512::digest(b"prism").as_slice());
}

#[test]
fn test_update_and_copy_are_independent() {
    let value = hash_constructor(Sha2Kind::Sha256, &[], &[]).expect("sha256 should construct");
    hash_update(&[value, bytes_value(b"abc")]).expect("update should succeed");

    let copied = hash_copy(&[value]).expect("copy should succeed");
    hash_update(&[value, bytes_value(b"def")]).expect("second update should succeed");

    let original_hex = string_value(hash_hexdigest(&[value]).expect("hexdigest should succeed"));
    let copied_hex = string_value(hash_hexdigest(&[copied]).expect("hexdigest should succeed"));

    assert_eq!(
        original_hex,
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    );
    assert_eq!(
        copied_hex,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn test_constructor_accepts_usedforsecurity_keyword() {
    let value = sha512_constructor(&[], &[("usedforsecurity", Value::bool(false))])
        .expect("usedforsecurity keyword should be accepted");
    let digest = hash_hexdigest(&[value]).expect("hexdigest should succeed");
    assert_eq!(
        string_value(digest),
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce\
             47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e"
    );
}
