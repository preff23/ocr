"""
High-performance caching system for speed profile optimizations
"""
import json
import hashlib
import time
from typing import Any, Optional, Dict, Union
from dataclasses import dataclass
from bot.core.logging import get_logger
from bot.core.config import config
logger = get_logger(__name__)
@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    value: Any
    timestamp: float
    ttl: int
class SpeedCache:
    """High-performance in-memory cache with TTL"""
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return time.time() - entry.timestamp > entry.ttl
    def _cleanup_expired(self):
        """Remove expired entries"""
        if not config.feature_speed_profile:
            return
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > entry.ttl
        ]
        for key in expired_keys:
            del self._cache[key]
            self._stats['evictions'] += 1
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not config.feature_speed_profile:
            return None
        self._cleanup_expired()
        if key in self._cache:
            entry = self._cache[key]
            if not self._is_expired(entry):
                self._stats['hits'] += 1
                return entry.value
            else:
                del self._cache[key]
                self._stats['evictions'] += 1
        self._stats['misses'] += 1
        return None
    def set(self, key: str, value: Any, ttl: int) -> None:
        """Set value in cache with TTL"""
        if not config.feature_speed_profile:
            return
        self._cache[key] = CacheEntry(
            value=value,
            timestamp=time.time(),
            ttl=ttl
        )
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = (self._stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        return {
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'hit_rate': round(hit_rate, 2),
            'size': len(self._cache)
        }
class ImageHashCache:
    """Cache for image hashes to avoid duplicate OCR processing"""
    def __init__(self):
        self.cache = SpeedCache()
    def get_image_hash(self, image_bytes: bytes) -> str:
        """Generate SHA256 hash for image"""
        return hashlib.sha256(image_bytes).hexdigest()
    def get_ocr_result(self, image_bytes: bytes) -> Optional[Any]:
        """Get cached OCR result for image"""
        if not config.feature_speed_profile:
            return None
        image_hash = self.get_image_hash(image_bytes)
        return self.cache.get(f"ocr_{image_hash}")
    def set_ocr_result(self, image_bytes: bytes, result: Any) -> None:
        """Cache OCR result for image"""
        if not config.feature_speed_profile:
            return
        image_hash = self.get_image_hash(image_bytes)
        self.cache.set(f"ocr_{image_hash}", result, config.cache_ocr_ttl_s)
class MarketDataCache:
    """Cache for market data with different TTLs"""
    def __init__(self):
        self.cache = SpeedCache()
    def get_market_data(self, key: str, source: str) -> Optional[Any]:
        """Get cached market data"""
        if not config.feature_speed_profile:
            return None
        cache_key = f"md_{source}_{key}"
        return self.cache.get(cache_key)
    def set_market_data(self, key: str, source: str, data: Any) -> None:
        """Cache market data"""
        if not config.feature_speed_profile:
            return
        cache_key = f"md_{source}_{key}"
        self.cache.set(cache_key, data, config.cache_md_ttl_s)
    def get_calendar_data(self, key: str) -> Optional[Any]:
        """Get cached calendar data"""
        if not config.feature_speed_profile:
            return None
        cache_key = f"calendar_{key}"
        return self.cache.get(cache_key)
    def set_calendar_data(self, key: str, data: Any) -> None:
        """Cache calendar data"""
        if not config.feature_speed_profile:
            return
        cache_key = f"calendar_{key}"
        self.cache.set(cache_key, data, config.cache_calendar_ttl_s)
image_hash_cache = ImageHashCache()
market_data_cache = MarketDataCache()
def get_cache_stats() -> Dict[str, Any]:
    """Get combined cache statistics"""
    if not config.feature_speed_profile:
        return {}
    return {
        'image_cache': image_hash_cache.cache.get_stats(),
        'market_data_cache': market_data_cache.cache.get_stats()
    }