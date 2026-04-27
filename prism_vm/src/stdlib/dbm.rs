//! Native `dbm` module bootstrap surface.
//!
//! CPython exposes `dbm.whichdb()` as a small, filesystem-oriented detector used
//! by pickle compatibility paths and regression helpers. Prism keeps this
//! module native so imports do not depend on a source stdlib tree, while actual
//! database backends can be added independently.

use super::{Module, ModuleError, ModuleResult};
use crate::builtins::{BuiltinError, BuiltinFunctionObject, exception_type_value_for_id};
use crate::stdlib::exceptions::ExceptionTypeId;
use prism_core::Value;
use prism_core::intern::intern;
use prism_runtime::types::bytes::value_as_bytes_ref;
use prism_runtime::types::list::ListObject;
use prism_runtime::types::string::value_as_string_ref;
use prism_runtime::types::tuple::TupleObject;
use std::ffi::OsString;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

const EXPORTS: &[&str] = &["open", "whichdb", "error"];
const DBM_BACKENDS: &[&str] = &["dbm.gnu", "dbm.ndbm", "dbm.dumb"];
const GDBM_MAGIC: [i32; 3] = [0x13579ace, 0x13579acd, 0x13579acf];

static OPEN_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dbm.open"), dbm_open));
static WHICHDB_FUNCTION: LazyLock<BuiltinFunctionObject> =
    LazyLock::new(|| BuiltinFunctionObject::new(Arc::from("dbm.whichdb"), whichdb));

/// Native `dbm` module descriptor.
#[derive(Debug, Clone)]
pub struct DbmModule {
    attrs: Vec<Arc<str>>,
    all: Value,
    error: Value,
}

impl DbmModule {
    /// Create a new native `dbm` module.
    pub fn new() -> Self {
        Self {
            attrs: EXPORTS.iter().copied().map(Arc::from).collect(),
            all: string_list_value(EXPORTS),
            error: dbm_error_value(),
        }
    }
}

impl Default for DbmModule {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for DbmModule {
    fn name(&self) -> &str {
        "dbm"
    }

    fn get_attr(&self, name: &str) -> ModuleResult {
        match name {
            "__all__" => Ok(self.all),
            "open" => Ok(builtin_value(&OPEN_FUNCTION)),
            "whichdb" => Ok(builtin_value(&WHICHDB_FUNCTION)),
            "error" => Ok(self.error),
            _ => Err(ModuleError::AttributeError(format!(
                "module 'dbm' has no attribute '{}'",
                name
            ))),
        }
    }

    fn dir(&self) -> Vec<Arc<str>> {
        let mut attrs = self.attrs.clone();
        attrs.push(Arc::from("__all__"));
        attrs
    }
}

#[inline]
fn builtin_value(function: &'static BuiltinFunctionObject) -> Value {
    Value::object_ptr(function as *const BuiltinFunctionObject as *const ())
}

fn string_list_value(items: &[&str]) -> Value {
    let values = items
        .iter()
        .copied()
        .map(|item| Value::string(intern(item)))
        .collect::<Vec<_>>();
    crate::alloc_managed_value(ListObject::from_iter(values))
}

fn dbm_error_value() -> Value {
    let os_error = exception_type_value_for_id(ExceptionTypeId::OSError as u16)
        .expect("OSError should be registered in the builtin exception table");
    crate::alloc_managed_value(TupleObject::from_vec(vec![os_error, os_error]))
}

fn dbm_open(args: &[Value]) -> Result<Value, BuiltinError> {
    if !(1..=3).contains(&args.len()) {
        return Err(BuiltinError::TypeError(format!(
            "open() takes from 1 to 3 positional arguments but {} were given",
            args.len()
        )));
    }

    Err(BuiltinError::ImportError(format!(
        "no dbm clone found; tried {}",
        DBM_BACKENDS.join(", ")
    )))
}

fn whichdb(args: &[Value]) -> Result<Value, BuiltinError> {
    if args.len() != 1 {
        return Err(BuiltinError::TypeError(format!(
            "whichdb() takes exactly 1 argument ({} given)",
            args.len()
        )));
    }

    let path = dbm_path_arg(args[0])?;
    Ok(match detect_dbm_kind(&path) {
        DbmKind::Missing => Value::none(),
        DbmKind::Unknown => Value::string(intern("")),
        DbmKind::Ndbm => Value::string(intern("dbm.ndbm")),
        DbmKind::Dumb => Value::string(intern("dbm.dumb")),
        DbmKind::Gnu => Value::string(intern("dbm.gnu")),
    })
}

fn dbm_path_arg(value: Value) -> Result<PathBuf, BuiltinError> {
    if let Some(text) = value_as_string_ref(value) {
        return Ok(PathBuf::from(text.as_str()));
    }

    if let Some(bytes) = value_as_bytes_ref(value) {
        return Ok(PathBuf::from(
            String::from_utf8_lossy(bytes.as_bytes()).into_owned(),
        ));
    }

    Err(BuiltinError::TypeError(
        "whichdb() argument must be str, bytes, or os.PathLike".to_string(),
    ))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DbmKind {
    Missing,
    Unknown,
    Ndbm,
    Dumb,
    Gnu,
}

fn detect_dbm_kind(filename: &Path) -> DbmKind {
    if readable_file(&path_with_suffix(filename, ".pag"))
        && readable_file(&path_with_suffix(filename, ".dir"))
    {
        return DbmKind::Ndbm;
    }

    if readable_file(&path_with_suffix(filename, ".db")) {
        return DbmKind::Ndbm;
    }

    let dat_path = path_with_suffix(filename, ".dat");
    let dir_path = path_with_suffix(filename, ".dir");
    if readable_file(&dat_path) {
        if let Ok(metadata) = std::fs::metadata(&dir_path) {
            if metadata.len() == 0 {
                return DbmKind::Dumb;
            }
            if first_byte(&dir_path).is_some_and(|byte| matches!(byte, b'\'' | b'"')) {
                return DbmKind::Dumb;
            }
        }
    }

    let Some(header) = read_prefix(filename, 16) else {
        return DbmKind::Missing;
    };
    if header.len() < 4 {
        return DbmKind::Unknown;
    }

    if is_gdbm_magic(&header[0..4]) || (header.len() >= 16 && is_gdbm_magic(&header[12..16])) {
        return DbmKind::Gnu;
    }

    DbmKind::Unknown
}

#[inline]
fn path_with_suffix(base: &Path, suffix: &str) -> PathBuf {
    let mut path = OsString::from(base.as_os_str());
    path.push(suffix);
    PathBuf::from(path)
}

#[inline]
fn readable_file(path: &Path) -> bool {
    File::open(path).is_ok()
}

fn first_byte(path: &Path) -> Option<u8> {
    read_prefix(path, 1).and_then(|bytes| bytes.first().copied())
}

fn read_prefix(path: &Path, max_len: usize) -> Option<Vec<u8>> {
    let mut file = File::open(path).ok()?;
    let mut bytes = vec![0; max_len];
    let read = file.read(&mut bytes).ok()?;
    bytes.truncate(read);
    Some(bytes)
}

#[inline]
fn is_gdbm_magic(bytes: &[u8]) -> bool {
    let Ok(bytes) = <&[u8; 4]>::try_from(bytes) else {
        return false;
    };
    GDBM_MAGIC.contains(&i32::from_ne_bytes(*bytes))
}

#[cfg(test)]
mod tests {
    use super::{DbmKind, detect_dbm_kind, path_with_suffix};
    use std::fs::{File, remove_file};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn detects_missing_database() {
        let fixture = TempDbmFixture::new();
        assert_eq!(detect_dbm_kind(&fixture.base), DbmKind::Missing);
    }

    #[test]
    fn detects_dumb_database() {
        let fixture = TempDbmFixture::new();
        fixture.write_suffix(".dat", b"data");
        fixture.write_suffix(".dir", b"'key', (0, 4)\n");
        assert_eq!(detect_dbm_kind(&fixture.base), DbmKind::Dumb);
    }

    #[test]
    fn detects_gnu_database_magic() {
        let fixture = TempDbmFixture::new();
        fixture.write_base(&0x13579ace_i32.to_ne_bytes());
        assert_eq!(detect_dbm_kind(&fixture.base), DbmKind::Gnu);
    }

    struct TempDbmFixture {
        base: PathBuf,
    }

    impl TempDbmFixture {
        fn new() -> Self {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock should be after unix epoch")
                .as_nanos();
            Self {
                base: std::env::temp_dir()
                    .join(format!("prism-dbm-test-{}-{unique}", std::process::id())),
            }
        }

        fn write_base(&self, bytes: &[u8]) {
            write_file(&self.base, bytes);
        }

        fn write_suffix(&self, suffix: &str, bytes: &[u8]) {
            write_file(&path_with_suffix(&self.base, suffix), bytes);
        }
    }

    impl Drop for TempDbmFixture {
        fn drop(&mut self) {
            for suffix in ["", ".pag", ".dir", ".db", ".dat"] {
                let path = if suffix.is_empty() {
                    self.base.clone()
                } else {
                    path_with_suffix(&self.base, suffix)
                };
                let _ = remove_file(path);
            }
        }
    }

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut file = File::create(path).expect("test fixture should be writable");
        file.write_all(bytes)
            .expect("test fixture bytes should be writable");
    }
}
