use super::*;
use crate::builtins::allocate_heap_instance_for_class;

fn new_random_instance() -> Value {
    let instance = allocate_heap_instance_for_class(random_class());
    Value::object_ptr(Box::into_raw(Box::new(instance)) as *const ())
}

#[test]
fn test_random_module_exposes_random_class() {
    let module = RandomModule::new();
    let class_value = module.get_attr("Random").expect("Random should exist");
    let class_ptr = class_value
        .as_object_ptr()
        .expect("Random should be exposed as a class object");
    let class = unsafe { &*(class_ptr as *const PyClassObject) };

    assert_eq!(class.name().as_str(), "Random");
    assert_eq!(
        class
            .get_attr(&intern("__module__"))
            .and_then(string_from_value),
        Some("_random".to_string())
    );
}

#[test]
fn test_random_seed_state_roundtrip_replays_sequence() {
    let instance = new_random_instance();
    random_seed(&[instance, Value::int(123456).expect("seed should fit")])
        .expect("seed should succeed");

    let state = random_getstate(&[instance]).expect("getstate should succeed");
    let first = random_random(&[instance])
        .expect("random should succeed")
        .as_float()
        .expect("random should return float");
    let second = random_random(&[instance])
        .expect("random should succeed")
        .as_float()
        .expect("random should return float");

    random_setstate(&[instance, state]).expect("setstate should succeed");
    let replay_first = random_random(&[instance])
        .expect("random should succeed")
        .as_float()
        .expect("random should return float");
    let replay_second = random_random(&[instance])
        .expect("random should succeed")
        .as_float()
        .expect("random should return float");

    assert_eq!(replay_first, first);
    assert_eq!(replay_second, second);
}

#[test]
fn test_random_getrandbits_obeys_requested_width() {
    let instance = new_random_instance();
    random_seed(&[instance, Value::int(7).expect("seed should fit")]).expect("seed should work");

    let zero = random_getrandbits(&[instance, Value::int(0).unwrap()])
        .expect("getrandbits(0) should succeed");
    assert_eq!(zero.as_int(), Some(0));

    let sixty_five = random_getrandbits(&[instance, Value::int(65).unwrap()])
        .expect("getrandbits(65) should succeed");
    let bigint = value_to_bigint(sixty_five).expect("result should be an integer");
    assert!(bigint.sign() != Sign::Minus);
    assert!(bigint.bits() <= 65);
}

#[test]
fn test_random_init_accepts_none_seed_and_produces_state() {
    let instance = new_random_instance();
    random_init(&[instance]).expect("__init__ without seed should succeed");
    let state = random_getstate(&[instance]).expect("getstate should succeed");
    assert!(value_to_bigint(state).is_some());
}
