//! Per-path mutation serialization, port of `core/tools/file-mutation-queue.ts`.
//!
//! Operations targeting the same canonical file path run one at a time;
//! operations on different files run in parallel.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use tokio::sync::Mutex as AsyncMutex;

fn queues() -> &'static Mutex<HashMap<String, Arc<AsyncMutex<()>>>> {
    static QUEUES: OnceLock<Mutex<HashMap<String, Arc<AsyncMutex<()>>>>> = OnceLock::new();
    QUEUES.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Canonical key for a path: resolved absolute path (lexically; symlinks are
/// not re-resolved per call, matching the common case).
fn queue_key(path: &std::path::Path) -> String {
    if let Ok(canon) = path.canonicalize() {
        canon.to_string_lossy().into_owned()
    } else {
        // Missing file (new file case): fall back to lexical resolution.
        match std::fs::canonicalize(path.parent().unwrap_or(path)) {
            Ok(parent) => parent
                .join(path.file_name().unwrap_or_default())
                .to_string_lossy()
                .into_owned(),
            Err(_) => path.to_string_lossy().into_owned(),
        }
    }
}

/// Run `op` while holding the mutation lock for `path`.
pub async fn with_file_mutation_queue<F, T>(path: &std::path::Path, op: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let key = queue_key(path);
    let lock = {
        let mut map = queues().lock().unwrap();
        map.entry(key).or_default().clone()
    };
    let _guard = lock.lock().await;
    op.await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn serializes_same_path() {
        let counter = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));
        let path = std::path::Path::new("/tmp/pi-rs-test-same-file");

        let mut tasks = Vec::new();
        for _ in 0..8 {
            let counter = counter.clone();
            let max = max_concurrent.clone();
            tasks.push(tokio::spawn(async move {
                with_file_mutation_queue(path, async {
                    let now = counter.fetch_add(1, Ordering::SeqCst) + 1;
                    max.fetch_max(now, Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    counter.fetch_sub(1, Ordering::SeqCst);
                })
                .await;
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }
        assert_eq!(max_concurrent.load(Ordering::SeqCst), 1);
    }

    use std::time::Duration;
}
