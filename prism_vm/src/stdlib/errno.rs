//! Native `errno` module bootstrap surface.
//!
//! CPython's stdlib and regression helpers import `errno` for symbolic error
//! constants and the reverse `errorcode` mapping. Prism provides the same
//! surface natively so that extension-backed imports keep working even when
//! the CPython source tree does not ship a pure-Python fallback.

use super::{Module, ModuleError, ModuleResult};
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::dict::DictObject;
use rustc_hash::FxHashMap;
use std::sync::Arc;

#[cfg(windows)]
const PLATFORM_ERRNO_ENTRIES: &[(&str, i64)] = &[
    ("EPERM", 1),
    ("ENOENT", 2),
    ("ESRCH", 3),
    ("EINTR", 10004),
    ("EIO", 5),
    ("ENXIO", 6),
    ("E2BIG", 7),
    ("ENOEXEC", 8),
    ("EBADF", 10009),
    ("ECHILD", 10),
    ("EAGAIN", 11),
    ("ENOMEM", 12),
    ("EACCES", 13),
    ("EFAULT", 14),
    ("EBUSY", 16),
    ("EEXIST", 17),
    ("EXDEV", 18),
    ("ENODEV", 19),
    ("ENOTDIR", 20),
    ("EISDIR", 21),
    ("EINVAL", 22),
    ("ENFILE", 23),
    ("EMFILE", 24),
    ("ENOTTY", 25),
    ("EFBIG", 27),
    ("ENOSPC", 28),
    ("ESPIPE", 29),
    ("EROFS", 30),
    ("EMLINK", 31),
    ("EPIPE", 32),
    ("EDOM", 33),
    ("ERANGE", 34),
    ("EDEADLK", 36),
    ("ENAMETOOLONG", 38),
    ("ENOLCK", 39),
    ("ENOSYS", 40),
    ("ENOTEMPTY", 41),
    ("EILSEQ", 42),
    ("EWOULDBLOCK", 10035),
    ("EINPROGRESS", 10036),
    ("EALREADY", 10037),
    ("ENOTSOCK", 10038),
    ("EDESTADDRREQ", 10039),
    ("EMSGSIZE", 10040),
    ("EPROTOTYPE", 10041),
    ("ENOPROTOOPT", 10042),
    ("EPROTONOSUPPORT", 10043),
    ("EOPNOTSUPP", 10045),
    ("EAFNOSUPPORT", 10047),
    ("EADDRINUSE", 10048),
    ("EADDRNOTAVAIL", 10049),
    ("ENETDOWN", 10050),
    ("ENETUNREACH", 10051),
    ("ENETRESET", 10052),
    ("ECONNABORTED", 10053),
    ("ECONNRESET", 10054),
    ("ENOBUFS", 10055),
    ("EISCONN", 10056),
    ("ENOTCONN", 10057),
    ("ETIMEDOUT", 10060),
    ("ECONNREFUSED", 10061),
    ("ELOOP", 10062),
    ("EHOSTUNREACH", 10065),
    ("EUSERS", 10068),
];

#[cfg(unix)]
const PLATFORM_ERRNO_ENTRIES: &[(&str, i64)] = &[
    ("EPERM", libc::EPERM as i64),
    ("ENOENT", libc::ENOENT as i64),
    ("ESRCH", libc::ESRCH as i64),
    ("EINTR", libc::EINTR as i64),
    ("EIO", libc::EIO as i64),
    ("ENXIO", libc::ENXIO as i64),
    ("E2BIG", libc::E2BIG as i64),
    ("ENOEXEC", libc::ENOEXEC as i64),
    ("EBADF", libc::EBADF as i64),
    ("ECHILD", libc::ECHILD as i64),
    ("EAGAIN", libc::EAGAIN as i64),
    ("ENOMEM", libc::ENOMEM as i64),
    ("EACCES", libc::EACCES as i64),
    ("EFAULT", libc::EFAULT as i64),
    ("EBUSY", libc::EBUSY as i64),
    ("EEXIST", libc::EEXIST as i64),
    ("EXDEV", libc::EXDEV as i64),
    ("ENODEV", libc::ENODEV as i64),
    ("ENOTDIR", libc::ENOTDIR as i64),
    ("EISDIR", libc::EISDIR as i64),
    ("EINVAL", libc::EINVAL as i64),
    ("ENFILE", libc::ENFILE as i64),
    ("EMFILE", libc::EMFILE as i64),
    ("ENOTTY", libc::ENOTTY as i64),
    ("EFBIG", libc::EFBIG as i64),
    ("ENOSPC", libc::ENOSPC as i64),
    ("ESPIPE", libc::ESPIPE as i64),
    ("EROFS", libc::EROFS as i64),
    ("EMLINK", libc::EMLINK as i64),
    ("EPIPE", libc::EPIPE as i64),
    ("EDOM", libc::EDOM as i64),
    ("ERANGE", libc::ERANGE as i64),
    ("EDEADLK", libc::EDEADLK as i64),
    ("ENAMETOOLONG", libc::ENAMETOOLONG as i64),
    ("ENOLCK", libc::ENOLCK as i64),
    ("ENOSYS", libc::ENOSYS as i64),
    ("ENOTEMPTY", libc::ENOTEMPTY as i64),
    ("EILSEQ", libc::EILSEQ as i64),
    ("EWOULDBLOCK", libc::EWOULDBLOCK as i64),
    ("EINPROGRESS", libc::EINPROGRESS as i64),
    ("EALREADY", libc::EALREADY as i64),
    ("ENOTSOCK", libc::ENOTSOCK as i64),
    ("EDESTADDRREQ", libc::EDESTADDRREQ as i64),
    ("EMSGSIZE", libc::EMSGSIZE as i64),
    ("EPROTOTYPE", libc::EPROTOTYPE as i64),
    ("ENOPROTOOPT", libc::ENOPROTOOPT as i64),
    ("EPROTONOSUPPORT", libc::EPROTONOSUPPORT as i64),
    ("EOPNOTSUPP", libc::EOPNOTSUPP as i64),
    ("EAFNOSUPPORT", libc::EAFNOSUPPORT as i64),
    ("EADDRINUSE", libc::EADDRINUSE as i64),
    ("EADDRNOTAVAIL", libc::EADDRNOTAVAIL as i64),
    ("ENETDOWN", libc::ENETDOWN as i64),
    ("ENETUNREACH", libc::ENETUNREACH as i64),
    ("ENETRESET", libc::ENETRESET as i64),
    ("ECONNABORTED", libc::ECONNABORTED as i64),
    ("ECONNRESET", libc::ECONNRESET as i64),
    ("ENOBUFS", libc::ENOBUFS as i64),
    ("EISCONN", libc::EISCONN as i64),
    ("ENOTCONN", libc::ENOTCONN as i64),
    ("ETIMEDOUT", libc::ETIMEDOUT as i64),
    ("ECONNREFUSED", libc::ECONNREFUSED as i64),
    ("ELOOP", libc::ELOOP as i64),
    ("EHOSTUNREACH", libc::EHOSTUNREACH as i64),
];

#[cfg(not(any(windows, unix)))]
const PLATFORM_ERRNO_ENTRIES: &[(&str, i64)] = &[
    ("EPERM", 1),
    ("ENOENT", 2),
    ("EINTR", 4),
    ("EIO", 5),
    ("EBADF", 9),
    ("EACCES", 13),
    ("EEXIST", 17),
    ("EINVAL", 22),
    ("ENOSYS", 40),
];

#[inline]
fn leak_object_value<T: prism_runtime::Trace>(object: T) -> Value {
    crate::alloc_managed_value(object)
}

/// Native `errno` module descriptor.
#[derive(Debug, Clone)]
pub struct ErrnoModule {
    attrs: Vec<Arc<str>>,
    values: FxHashMap<Arc<str>, Value>,
}

impl ErrnoModule {
    /// Create a new `errno` module descriptor.
    pub fn new() -> Self {
        let mut attrs = Vec::with_capacity(PLATFORM_ERRNO_ENTRIES.len() + 1);
        let mut values = FxHashMap::default();
        let mut errorcode = DictObject::with_capacity(PLATFORM_ERRNO_ENTRIES.len());

        for &(name, code) in PLATFORM_ERRNO_ENTRIES {
            let value = Value::int(code).expect("errno constant should fit in Value::int");
            values.insert(Arc::from(name), value);
            attrs.push(Arc::from(name));
            errorcode.set(value, Value::string(intern(name)));
        }

        values.insert(Arc::from("errorcode"), leak_object_value(errorcode));
        attrs.push(Arc::from("errorcode"));
        attrs.sort_unstable();

        Self { attrs, values }
    }
}

impl Default for ErrnoModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ErrnoModule {
    fn name(&self) -> &str {
        "errno"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        self.values.get(name).copied().ok_or_else(|| {
            ModuleError::AttributeError(format!("module 'errno' has no attribute '{}'", name))
        })
    }

    fn dir(&self) -> Vec<Arc<str>> {
        self.attrs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn errorcode_mapping(module: &ErrnoModule) -> &'static DictObject {
        let value = module
            .get_attr("errorcode")
            .expect("errorcode should be exported");
        let ptr = value
            .as_object_ptr()
            .expect("errorcode should be a dict object");
        unsafe { &*(ptr as *const DictObject) }
    }

    #[test]
    fn test_errno_module_exports_expected_constants() {
        let module = ErrnoModule::new();
        assert!(module.get_attr("ENOENT").is_ok());
        assert!(module.get_attr("EBADF").is_ok());
        assert!(module.get_attr("EINVAL").is_ok());
        assert!(module.get_attr("errorcode").is_ok());
    }

    #[test]
    fn test_errno_errorcode_maps_back_to_symbol_names() {
        let module = ErrnoModule::new();
        let errorcode = errorcode_mapping(&module);

        for name in ["ENOENT", "EBADF", "EINVAL"] {
            let code = module.get_attr(name).expect("constant should be exported");
            assert_eq!(
                errorcode.get(code),
                Some(Value::string(intern(name))),
                "errorcode should map {} back to its symbol",
                name
            );
        }
    }
}
