use super::*;

fn test_bytes(data: &[u8]) -> Value {
    bytes_value(data.to_vec())
}

fn value_to_bytes(value: Value) -> Vec<u8> {
    let ptr = value
        .as_object_ptr()
        .expect("binascii functions should return bytes");
    unsafe { &*(ptr as *const BytesObject) }.to_vec()
}

#[test]
fn test_module_exposes_cpython_surface() {
    let module = BinasciiModule::new();
    for name in [
        "Error",
        "Incomplete",
        "a2b_base64",
        "b2a_base64",
        "a2b_hex",
        "hexlify",
        "unhexlify",
        "a2b_qp",
        "b2a_qp",
        "a2b_uu",
        "b2a_uu",
        "crc32",
        "crc_hqx",
    ] {
        assert!(module.get_attr(name).is_ok(), "missing {name}");
    }
}

#[test]
fn test_base64_round_trips_and_respects_newline_keyword() {
    let encoded = b2a_base64_builtin(&[test_bytes(b"hello")], &[("newline", Value::bool(false))])
        .expect("base64 encode should succeed");
    assert_eq!(value_to_bytes(encoded), b"aGVsbG8=");

    let decoded = a2b_base64_builtin(&[encoded], &[]).expect("base64 decode should succeed");
    assert_eq!(value_to_bytes(decoded), b"hello");

    let with_newline = b2a_base64_builtin(&[test_bytes(b"hello")], &[]).expect("newline default");
    assert_eq!(value_to_bytes(with_newline), b"aGVsbG8=\n");
}

#[test]
fn test_base64_strict_mode_rejects_invalid_input() {
    let err = a2b_base64_builtin(
        &[test_bytes(b"a\nb==")],
        &[("strict_mode", Value::bool(true))],
    )
    .expect_err("strict base64 should reject non alphabet data");
    assert!(err.to_string().contains("Only base64 data"));

    let len_err = a2b_base64_builtin(&[test_bytes(b"a")], &[])
        .expect_err("invalid base64 length should fail");
    assert!(len_err.to_string().contains("number of data characters"));
}

#[test]
fn test_hexlify_and_unhexlify() {
    let encoded =
        b2a_hex_builtin(&[test_bytes(&[0xb9, 0x01, 0xef])], &[]).expect("hexlify should succeed");
    assert_eq!(value_to_bytes(encoded), b"b901ef");

    let separated = b2a_hex_builtin(
        &[
            test_bytes(&[0xb9, 0x01, 0xef]),
            test_bytes(b":"),
            Value::int(1).unwrap(),
        ],
        &[],
    )
    .expect("hexlify with separator should succeed");
    assert_eq!(value_to_bytes(separated), b"b9:01:ef");

    let decoded = a2b_hex_builtin(&[test_bytes(b"b901EF")]).expect("unhexlify should succeed");
    assert_eq!(value_to_bytes(decoded), &[0xb9, 0x01, 0xef]);
}

#[test]
fn test_quoted_printable_matches_core_edge_cases() {
    assert_eq!(a2b_qp_bytes(b"=00\r\n=00", false), b"\x00\r\n\x00");
    assert_eq!(a2b_qp_bytes(b"=\rAB\nCD", false), b"CD");
    assert_eq!(a2b_qp_bytes(b"_", true), b" ");

    let encoded = b2a_qp_builtin(
        &[test_bytes(b"x y\tz")],
        &[("quotetabs", Value::bool(true))],
    )
    .expect("quoted-printable encode should succeed");
    assert_eq!(value_to_bytes(encoded), b"x=20y=09z");

    let header = b2a_qp_builtin(&[test_bytes(b"x y")], &[("header", Value::bool(true))])
        .expect("header quoted-printable encode should succeed");
    assert_eq!(value_to_bytes(header), b"x_y");
}

#[test]
fn test_uu_codec_matches_cpython_edge_cases() {
    let encoded = b2a_uu_builtin(&[test_bytes(b"x")], &[]).expect("uuencode should succeed");
    assert_eq!(value_to_bytes(encoded), b"!>   \n");

    let decoded = a2b_uu_builtin(&[test_bytes(b"!>   \n")]).expect("uudecode should succeed");
    assert_eq!(value_to_bytes(decoded), b"x");

    assert_eq!(
        value_to_bytes(a2b_uu_builtin(&[test_bytes(b"\x7f")]).expect("missing data pads")),
        vec![0; 31]
    );
    assert_eq!(
        value_to_bytes(
            b2a_uu_builtin(&[test_bytes(b"")], &[("backtick", Value::bool(true))])
                .expect("empty backtick uuencode")
        ),
        b"`\n"
    );
}

#[test]
fn test_crc_functions_are_incremental() {
    let full = crc32_builtin(&[test_bytes(b"Test the CRC-32 of this string.")])
        .expect("crc32 should succeed");
    let first = crc32_builtin(&[test_bytes(b"Test the CRC-32 of")]).expect("crc32 first chunk");
    let second = crc32_builtin(&[test_bytes(b" this string."), first]).expect("crc32 second chunk");
    assert_eq!(value_to_bigint(full), value_to_bigint(second));
    assert_eq!(
        value_to_bigint(second),
        Some(BigInt::from(1_571_220_330_u32))
    );

    let hqx_first = crc_hqx_builtin(&[test_bytes(b"Test the CRC-32 of"), Value::int(0).unwrap()])
        .expect("crc_hqx first chunk");
    let hqx_second =
        crc_hqx_builtin(&[test_bytes(b" this string."), hqx_first]).expect("crc_hqx second chunk");
    assert_eq!(value_to_bigint(hqx_second), Some(BigInt::from(14_290_u32)));
}
