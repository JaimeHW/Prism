use super::*;
use crate::import::ModuleObject;
use prism_code::{CodeObject, Opcode, Register};
use std::sync::Arc;

fn vm_with_frame() -> VirtualMachine {
    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(CodeObject::new("containers", "<test>")), 0)
        .expect("frame push should succeed");
    vm
}

fn exhaust_nursery(vm: &VirtualMachine) {
    for _ in 0..200_000 {
        if vm.allocator().alloc(DictObject::new()).is_none() {
            return;
        }
    }
    panic!("test setup should fill the nursery");
}

fn names_with_extended_import() -> Vec<Arc<str>> {
    (0..=0x0123)
        .map(|index| {
            if index == 0x0123 {
                Arc::from("extended")
            } else {
                Arc::from(format!("unused_{index}"))
            }
        })
        .collect()
}

#[test]
fn test_build_list_allocates_after_full_nursery() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(0, Value::int(1).unwrap());
    vm.current_frame_mut().set_reg(1, Value::int(2).unwrap());
    exhaust_nursery(&vm);

    let inst = Instruction::op_dss(
        Opcode::BuildList,
        Register::new(4),
        Register::new(0),
        Register::new(2),
    );
    assert!(matches!(build_list(&mut vm, inst), ControlFlow::Continue));
    assert!(vm.current_frame().get_reg(4).as_object_ptr().is_some());
}

#[test]
fn test_build_tuple_allocates_after_full_nursery() {
    let mut vm = vm_with_frame();
    vm.current_frame_mut().set_reg(0, Value::int(1).unwrap());
    vm.current_frame_mut().set_reg(1, Value::int(2).unwrap());
    exhaust_nursery(&vm);

    let inst = Instruction::op_dss(
        Opcode::BuildTuple,
        Register::new(4),
        Register::new(0),
        Register::new(2),
    );
    assert!(matches!(build_tuple(&mut vm, inst), ControlFlow::Continue));
    assert!(vm.current_frame().get_reg(4).as_object_ptr().is_some());
}

#[test]
fn test_import_from_consumes_extended_name_index() {
    let primary = Instruction::new(Opcode::ImportFrom, 2, 1, u8::MAX);
    let extension = Instruction::op_di(Opcode::AttrName, Register::new(0), 0x0123);
    let mut code = CodeObject::new("import_from", "<test>");
    code.names = names_with_extended_import().into_boxed_slice();
    code.instructions = vec![primary, extension].into_boxed_slice();

    let mut vm = VirtualMachine::new();
    vm.push_frame(Arc::new(code), 0)
        .expect("frame push should succeed");
    vm.current_frame_mut().ip = 1;

    let module = Arc::new(ModuleObject::new("import_ext"));
    let value = Value::int(42).unwrap();
    module.set_attr("extended", value);
    vm.import_resolver
        .insert_module("import_ext", module.clone());
    vm.current_frame_mut()
        .set_reg(1, Value::object_ptr(Arc::as_ptr(&module) as *const ()));

    assert!(matches!(
        import_from(&mut vm, primary),
        ControlFlow::Continue
    ));
    assert_eq!(vm.current_frame().get_reg(2), value);
    assert_eq!(vm.current_frame().ip, 2);
}
