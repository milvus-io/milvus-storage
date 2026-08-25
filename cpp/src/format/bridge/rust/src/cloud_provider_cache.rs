use std::future::Future;
use std::num::NonZeroUsize;

use lru::LruCache;
use tokio::sync::Mutex;

pub(crate) const CACHE_KEY: &str = "milvus_fs_cache_key";
pub(crate) const CACHE_CAPACITY: usize = 64;

pub(crate) struct GlobalLruCache<V> {
    entries: Mutex<LruCache<String, V>>,
}

impl<V: Clone> GlobalLruCache<V> {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(LruCache::new(
                NonZeroUsize::new(capacity)
                    .expect("cloud provider cache capacity must be non-zero"),
            )),
        }
    }

    pub(crate) async fn get<E, F, Fut>(&self, key: &str, create: F) -> Result<V, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<V, E>>,
    {
        let mut entries = self.entries.lock().await;
        if let Some(value) = entries.get(key).cloned() {
            return Ok(value);
        }

        let value = create().await?;
        entries.put(key.to_string(), value.clone());
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::future::join_all;

    use super::*;

    #[tokio::test]
    async fn concurrent_same_key_initializes_once() {
        let cache = Arc::new(GlobalLruCache::new(2));
        let initializations = Arc::new(AtomicUsize::new(0));
        let tasks = (0..100).map(|_| {
            let cache = cache.clone();
            let initializations = initializations.clone();
            async move {
                cache
                    .get("same", || async move {
                        initializations.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        Ok::<_, ()>(Arc::new(42_u64))
                    })
                    .await
                    .unwrap()
            }
        });

        let values = join_all(tasks).await;
        assert_eq!(initializations.load(Ordering::SeqCst), 1);
        assert!(values.iter().all(|value| Arc::ptr_eq(value, &values[0])));
    }

    #[tokio::test]
    async fn different_keys_initialize_serially() {
        let cache = Arc::new(GlobalLruCache::new(32));
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let tasks = (0..16).map(|index| {
            let cache = cache.clone();
            let active = active.clone();
            let max_active = max_active.clone();
            async move {
                cache
                    .get(&format!("key-{index}"), || async move {
                        let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(current, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok::<_, ()>(Arc::new(index))
                    })
                    .await
                    .unwrap()
            }
        });

        join_all(tasks).await;
        assert_eq!(max_active.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn failed_initialization_is_not_cached() {
        let cache = GlobalLruCache::new(2);
        let initializations = AtomicUsize::new(0);

        let first = cache
            .get("key", || async {
                initializations.fetch_add(1, Ordering::SeqCst);
                Err::<Arc<u64>, _>("failed")
            })
            .await;
        assert_eq!(first.unwrap_err(), "failed");

        let second = cache
            .get("key", || async {
                initializations.fetch_add(1, Ordering::SeqCst);
                Ok::<_, &str>(Arc::new(42_u64))
            })
            .await
            .unwrap();

        assert_eq!(*second, 42);
        assert_eq!(initializations.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn least_recently_used_entry_is_evicted() {
        let cache = GlobalLruCache::new(2);
        let first = cache
            .get("first", || async { Ok::<_, ()>(Arc::new(1)) })
            .await
            .unwrap();
        cache
            .get("second", || async { Ok::<_, ()>(Arc::new(2)) })
            .await
            .unwrap();
        let first_hit = cache
            .get("first", || async { Ok::<_, ()>(Arc::new(10)) })
            .await
            .unwrap();
        cache
            .get("third", || async { Ok::<_, ()>(Arc::new(3)) })
            .await
            .unwrap();
        let second_reloaded = cache
            .get("second", || async { Ok::<_, ()>(Arc::new(20)) })
            .await
            .unwrap();

        assert!(Arc::ptr_eq(&first, &first_hit));
        assert_eq!(*second_reloaded, 20);
    }
}
