//! Path helpers, port of the used parts of `core/tools/path-utils.ts`.

use std::path::{Path, PathBuf};

/// Resolve `path` against `cwd` for absolute-path tools (read/edit/write/...).
pub fn resolve_to_cwd(path: &str, cwd: &Path) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        cwd.join(p)
    }
}

/// Convert a path to a posix-style display string.
pub fn to_posix(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_relative() {
        let cwd = Path::new("/work/dir");
        assert_eq!(resolve_to_cwd("a/b.rs", cwd), PathBuf::from("/work/dir/a/b.rs"));
        assert_eq!(resolve_to_cwd("/abs/x", cwd), PathBuf::from("/abs/x"));
    }
}
