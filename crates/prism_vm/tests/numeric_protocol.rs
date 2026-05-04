use prism_compiler::{OptimizationLevel, compile_source_code};
use prism_vm::VirtualMachine;

fn execute(source: &str) {
    let code = compile_source_code(source, "<test>", OptimizationLevel::None)
        .expect("source should compile");
    let mut vm = VirtualMachine::new();
    vm.execute_runtime(code).expect("source should execute");
}

#[test]
fn int_binds_class_descriptor_special_method() {
    execute(
        r#"
class IntDescriptor:
    def __get__(self, instance, owner):
        def converted():
            return 11
        return converted

class Target:
    pass

Target.__int__ = IntDescriptor()

if int(Target()) != 11:
    raise RuntimeError("descriptor-backed __int__ was not bound")
"#,
    );
}

#[test]
fn hash_binds_class_descriptor_special_method() {
    execute(
        r#"
class HashDescriptor:
    def __get__(self, instance, owner):
        def converted():
            return 123
        return converted

class Target:
    pass

Target.__hash__ = HashDescriptor()

if hash(Target()) != 123:
    raise RuntimeError("descriptor-backed __hash__ was not bound")
"#,
    );
}
