"""
Кэш для данных облигаций
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from bot.core.logging import get_logger
logger = get_logger(__name__)
@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: float
class BondCache:
    """Простой кэш для данных облигаций"""
    def __init__(self, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
    def get(self, key: str) -> Optional[Any]:
        """Получить данные из кэша"""
        if key not in self.cache:
            return None
        entry = self.cache[key]
        current_time = time.time()
        if current_time - entry.timestamp > entry.ttl:
            del self.cache[key]
            logger.debug(f"Cache entry expired for key: {key}")
            return None
        logger.debug(f"Cache hit for key: {key}")
        return entry.data
    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Сохранить данные в кэш"""
        if ttl is None:
            ttl = self.default_ttl
        self.cache[key] = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=ttl
        )
        logger.debug(f"Cache set for key: {key}, ttl: {ttl}s")
    def clear(self) -> None:
        """Очистить весь кэш"""
        self.cache.clear()
        logger.info("Cache cleared")
    def clear_expired(self) -> None:
        """Очистить истекшие записи"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if current_time - entry.timestamp > entry.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        if expired_keys:
            logger.debug(f"Cleared {len(expired_keys)} expired cache entries")
    def size(self) -> int:
        """Размер кэша"""
        return len(self.cache)
    def stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        current_time = time.time()
        total_entries = len(self.cache)
        expired_entries = sum(
            1 for entry in self.cache.values()
            if current_time - entry.timestamp > entry.ttl
        )
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries
        }
bond_cache = BondCache(default_ttl=1800)