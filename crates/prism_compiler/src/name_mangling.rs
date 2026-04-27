//! CPython-compatible private name mangling.
//!
//! Names beginning with two underscores inside a class body are rewritten to
//! include the current class name. This happens at compile time so runtime
//! attribute lookup and local-variable resolution stay on their normal fast
//! paths.

use std::borrow::Cow;

/// Mangle a source identifier using CPython's private-name rules.
#[inline]
pub(crate) fn mangle_private_name<'a>(
    private_context: Option<&str>,
    name: &'a str,
) -> Cow<'a, str> {
    let Some(class_name) = private_context else {
        return Cow::Borrowed(name);
    };
    if !requires_private_mangle(name) {
        return Cow::Borrowed(name);
    }

    let class_name = class_name.trim_start_matches('_');
    if class_name.is_empty() {
        return Cow::Borrowed(name);
    }

    Cow::Owned(format!("_{class_name}{name}"))
}

#[inline]
fn requires_private_mangle(name: &str) -> bool {
    name.starts_with("__") && !name.ends_with("__") && !name.contains('.')
}

#[cfg(test)]
mod tests {
    use super::mangle_private_name;

    #[test]
    fn mangles_private_names_with_stripped_class_prefix() {
        assert_eq!(mangle_private_name(Some("Rat"), "__num"), "_Rat__num");
        assert_eq!(mangle_private_name(Some("_Rat"), "__num"), "_Rat__num");
    }

    #[test]
    fn preserves_dunder_and_non_private_names() {
        assert_eq!(mangle_private_name(Some("Rat"), "__class__"), "__class__");
        assert_eq!(mangle_private_name(Some("Rat"), "_num"), "_num");
        assert_eq!(mangle_private_name(Some("Rat"), "num"), "num");
        assert_eq!(mangle_private_name(None, "__num"), "__num");
    }
}
