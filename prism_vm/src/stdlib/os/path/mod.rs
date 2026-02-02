//! os.path module - Path manipulation operations.

mod join;
mod normalize;
mod query;
mod split;

pub use join::*;
pub use normalize::*;
pub use query::*;
pub use split::*;

use super::constants::SEP;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exports() {
        // Ensure all submodules export correctly
        let _ = exists(".");
        let _ = join("a", "b");
        let _ = basename("/foo/bar");
        let _ = abspath(".");
    }
}
