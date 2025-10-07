import time
import threading
from typing import Any, Optional, Dict, Callable
from dataclasses import dataclass
from bot.core.config import config
from bot.core.logging import get_logger
logger = get_logger(__name__)
@dataclass
class CacheItem:
    value: Any
    expires_at: float
    created_at: float
class TTLCache:
    def __init__(self, default_ttl: int = None):
        self.default_ttl = default_ttl or config.cache_ttl
        self._cache: Dict[str, CacheItem] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()
    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            self._cleanup_if_needed()
            if key not in self._cache:
                return None
            item = self._cache[key]
            if time.time() > item.expires_at:
                del self._cache[key]
                return None
            return item.value
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            ttl = ttl or self.default_ttl
            expires_at = time.time() + ttl
            created_at = time.time()
            self._cache[key] = CacheItem(
                value=value,
                expires_at=expires_at,
                created_at=created_at
            )
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
    def size(self) -> int:
        with self._lock:
            self._cleanup_if_needed()
            return len(self._cache)
    def _cleanup_if_needed(self) -> None:
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            self._cleanup()
            self._last_cleanup = current_time
    def _cleanup(self) -> None:
        current_time = time.time()
        expired_keys = [
            key for key, item in self._cache.items()
            if current_time > item.expires_at
        ]
        for key in expired_keys:
            del self._cache[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value, ttl)
        return value
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            current_time = time.time()
            total_items = len(self._cache)
            expired_items = sum(
                1 for item in self._cache.values()
                if current_time > item.expires_at
            )
            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "active_items": total_items - expired_items,
                "default_ttl": self.default_ttl
            }
class UserScopedCache:
    def __init__(self, default_ttl: int = None):
        self.default_ttl = default_ttl or config.cache_ttl
        self._user_caches: Dict[int, TTLCache] = {}
        self._lock = threading.RLock()
    def _get_user_cache(self, user_id: int) -> TTLCache:
        with self._lock:
            if user_id not in self._user_caches:
                self._user_caches[user_id] = TTLCache(self.default_ttl)
            return self._user_caches[user_id]
    def get(self, user_id: int, key: str) -> Optional[Any]:
        user_cache = self._get_user_cache(user_id)
        return user_cache.get(key)
    def set(self, user_id: int, key: str, value: Any, ttl: Optional[int] = None) -> None:
        user_cache = self._get_user_cache(user_id)
        user_cache.set(key, value, ttl)
    def delete(self, user_id: int, key: str) -> bool:
        user_cache = self._get_user_cache(user_id)
        return user_cache.delete(key)
    def clear_user(self, user_id: int) -> None:
        with self._lock:
            if user_id in self._user_caches:
                self._user_caches[user_id].clear()
    def clear_all(self) -> None:
        with self._lock:
            for user_cache in self._user_caches.values():
                user_cache.clear()
            self._user_caches.clear()
    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_users = len(self._user_caches)
            total_items = sum(cache.size() for cache in self._user_caches.values())
            return {
                "total_users": total_users,
                "total_items": total_items,
                "default_ttl": self.default_ttl
            }
market_data_cache = TTLCache(default_ttl=config.moex_iss_cache_ttl)
user_cache = UserScopedCache(default_ttl=config.cache_ttl)
def cache_key(*args, **kwargs) -> str:
    key_parts = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        else:
            key_parts.append(str(hash(str(arg))))
    for key, value in sorted(kwargs.items()):
        if isinstance(value, (str, int, float, bool)):
            key_parts.append(f"{key}:{value}")
        else:
            key_parts.append(f"{key}:{hash(str(value))}")
    return "|".join(key_parts)
def cached(ttl: Optional[int] = None, key_func: Optional[Callable] = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if key_func:
                cache_key_str = key_func(*args, **kwargs)
            else:
                cache_key_str = cache_key(*args, **kwargs)
            cached_result = market_data_cache.get(cache_key_str)
            if cached_result is not None:
                return cached_result
            result = func(*args, **kwargs)
            market_data_cache.set(cache_key_str, result, ttl)
            return result
        return wrapper
    return decorator