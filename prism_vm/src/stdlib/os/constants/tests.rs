use super::*;

// =========================================================================
// Platform Constants Tests
// =========================================================================

#[test]
fn test_os_name_valid() {
    assert!(OS_NAME == "nt" || OS_NAME == "posix");
}

#[cfg(windows)]
#[test]
fn test_os_name_windows() {
    assert_eq!(OS_NAME, "nt");
}

#[cfg(not(windows))]
#[test]
fn test_os_name_posix() {
    assert_eq!(OS_NAME, "posix");
}

// =========================================================================
// Path Separator Tests
// =========================================================================

#[cfg(windows)]
#[test]
fn test_sep_windows() {
    assert_eq!(SEP, '\\');
    assert_eq!(SEP_STR, "\\");
}

#[cfg(not(windows))]
#[test]
fn test_sep_unix() {
    assert_eq!(SEP, '/');
    assert_eq!(SEP_STR, "/");
}

#[cfg(windows)]
#[test]
fn test_altsep_windows() {
    assert_eq!(ALTSEP, Some('/'));
}

#[cfg(not(windows))]
#[test]
fn test_altsep_unix() {
    assert_eq!(ALTSEP, None);
}

#[cfg(windows)]
#[test]
fn test_pathsep_windows() {
    assert_eq!(PATHSEP, ';');
    assert_eq!(PATHSEP_STR, ";");
}

#[cfg(not(windows))]
#[test]
fn test_pathsep_unix() {
    assert_eq!(PATHSEP, ':');
    assert_eq!(PATHSEP_STR, ":");
}

#[cfg(windows)]
#[test]
fn test_linesep_windows() {
    assert_eq!(LINESEP, "\r\n");
}

#[cfg(not(windows))]
#[test]
fn test_linesep_unix() {
    assert_eq!(LINESEP, "\n");
}

// =========================================================================
// Directory Reference Tests
// =========================================================================

#[test]
fn test_curdir() {
    assert_eq!(CURDIR, ".");
}

#[test]
fn test_pardir() {
    assert_eq!(PARDIR, "..");
}

#[test]
fn test_extsep() {
    assert_eq!(EXTSEP, '.');
}

// =========================================================================
// Devnull Tests
// =========================================================================

#[cfg(windows)]
#[test]
fn test_devnull_windows() {
    assert_eq!(DEVNULL, "nul");
}

#[cfg(not(windows))]
#[test]
fn test_devnull_unix() {
    assert_eq!(DEVNULL, "/dev/null");
}

// =========================================================================
// O_* Flag Tests
// =========================================================================

#[test]
fn test_o_rdonly() {
    assert_eq!(O_RDONLY, 0);
}

#[test]
fn test_o_wronly() {
    assert_eq!(O_WRONLY, 1);
}

#[test]
fn test_o_rdwr() {
    assert_eq!(O_RDWR, 2);
}

#[test]
fn test_o_creat_nonzero() {
    assert!(O_CREAT > 0);
}

#[test]
fn test_o_trunc_nonzero() {
    assert!(O_TRUNC > 0);
}

#[test]
fn test_o_append_nonzero() {
    assert!(O_APPEND > 0);
}

#[test]
fn test_o_excl_nonzero() {
    assert!(O_EXCL > 0);
}

#[test]
fn test_o_flags_distinct() {
    // Ensure common flags don't overlap (except RDONLY which is 0)
    assert_ne!(O_WRONLY, O_RDWR);
    assert_ne!(O_CREAT, O_TRUNC);
    assert_ne!(O_APPEND, O_EXCL);
}

// =========================================================================
// Access Mode Tests
// =========================================================================

#[test]
fn test_f_ok() {
    assert_eq!(F_OK, 0);
}

#[test]
fn test_r_ok() {
    assert_eq!(R_OK, 4);
}

#[test]
fn test_w_ok() {
    assert_eq!(W_OK, 2);
}

#[test]
fn test_x_ok() {
    assert_eq!(X_OK, 1);
}

#[test]
fn test_access_flags_can_combine() {
    let combined = R_OK | W_OK | X_OK;
    assert_eq!(combined, 7);
}

// =========================================================================
// Seek Whence Tests
// =========================================================================

#[test]
fn test_seek_set() {
    assert_eq!(SEEK_SET, 0);
}

#[test]
fn test_seek_cur() {
    assert_eq!(SEEK_CUR, 1);
}

#[test]
fn test_seek_end() {
    assert_eq!(SEEK_END, 2);
}

#[test]
fn test_seek_whence_distinct() {
    assert_ne!(SEEK_SET, SEEK_CUR);
    assert_ne!(SEEK_CUR, SEEK_END);
    assert_ne!(SEEK_SET, SEEK_END);
}
