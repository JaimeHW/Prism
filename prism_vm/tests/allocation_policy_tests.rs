use std::fs;
use std::path::{Path, PathBuf};

use prism_runtime::allocation_context::alloc_value_in_current_heap_or_box;
use prism_runtime::types::list::ListObject;
use prism_vm::VirtualMachine;

#[test]
fn vm_lifetime_binds_runtime_heap_for_helper_allocations() {
    let vm = VirtualMachine::new();
    let value = alloc_value_in_current_heap_or_box(ListObject::new());
    let ptr = value
        .as_object_ptr()
        .expect("managed helper should allocate an object value");

    assert!(
        vm.heap().heap().contains(ptr),
        "runtime helper allocations should land in the active VM heap"
    );
}

#[test]
fn production_vm_code_uses_managed_value_allocation() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let mut violations = Vec::new();

    visit_rs_files(&src_dir, &mut |path| {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            return;
        }

        let source = fs::read_to_string(path).expect("source file should be readable");
        let mut brace_depth = 0usize;
        let mut test_module_depth = None;
        let mut saw_cfg_test = false;

        for (index, line) in source.lines().enumerate() {
            let trimmed = line.trim();

            if trimmed.starts_with("#[cfg(test)]") {
                saw_cfg_test = true;
                continue;
            }

            let opens = line.bytes().filter(|byte| *byte == b'{').count();
            let closes = line.bytes().filter(|byte| *byte == b'}').count();
            let next_depth = brace_depth.saturating_add(opens).saturating_sub(closes);

            if saw_cfg_test && is_inline_module_decl(trimmed) {
                test_module_depth = Some(next_depth);
                saw_cfg_test = false;
                brace_depth = next_depth;
                continue;
            }
            saw_cfg_test = false;

            if test_module_depth.is_none() && line.contains("Box::into_raw(Box::new") {
                let relative = path
                    .strip_prefix(&manifest_dir)
                    .expect("path should be under manifest dir");
                violations.push(format!("{}:{}", relative.display(), index + 1));
            }

            brace_depth = next_depth;
            if test_module_depth.is_some_and(|depth| brace_depth < depth) {
                test_module_depth = None;
            }
        }
    });

    assert!(
        violations.is_empty(),
        "production VM code must allocate Python objects through managed heap helpers, not raw boxes: {violations:?}"
    );
}

fn is_inline_module_decl(line: &str) -> bool {
    line.contains('{')
        && (line.starts_with("mod ")
            || line.starts_with("pub mod ")
            || line.starts_with("pub(crate) mod "))
}

fn visit_rs_files(dir: &Path, visit: &mut impl FnMut(&Path)) {
    for entry in fs::read_dir(dir).expect("source directory should be readable") {
        let entry = entry.expect("directory entry should be readable");
        let path = entry.path();
        if path.is_dir() {
            visit_rs_files(&path, visit);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            visit(&path);
        }
    }
}
