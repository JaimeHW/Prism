use super::*;

#[test]
fn test_module_exposes_tokenizer_iter_placeholder() {
    let module = TokenizeModule::new();
    assert!(
        module
            .get_attr("TokenizerIter")
            .expect("TokenizerIter should exist")
            .as_object_ptr()
            .is_some()
    );
}
