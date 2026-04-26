use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn production_runtime_code_confines_raw_box_allocations() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let src_dir = manifest_dir.join("src");
    let mut violations = Vec::new();

    visit_rs_files(&src_dir, &mut |path| {
        if path.file_name().is_some_and(|name| name == "tests.rs") {
            return;
        }

        let relative = path
            .strip_prefix(&manifest_dir)
            .expect("path should be under manifest dir");
        if is_allowed_raw_allocation_file(relative) {
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
        "production runtime code must use managed heap allocation for Python objects; raw boxes are confined to the explicit fallback allocator and non-graph sentinels: {violations:?}"
    );
}

fn is_allowed_raw_allocation_file(relative: &Path) -> bool {
    relative == Path::new("src").join("allocation_context.rs")
        || relative == Path::new("src").join("object").join("shaped_object.rs")
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
