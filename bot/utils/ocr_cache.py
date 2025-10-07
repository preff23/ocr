"""
Кэширование результатов OCR для ускорения обработки
"""
import hashlib
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from bot.core.logging import get_logger
from bot.ai.vision import OCRResult
logger = get_logger(__name__)
@dataclass
class CachedOCRResult:
    """Кэшированный результат OCR"""
    result: Dict[str, Any]
    timestamp: float
    image_hash: str
    ttl: float = 3600
    def is_expired(self) -> bool:
        """Проверяет, истек ли срок действия кэша"""
        return time.time() - self.timestamp > self.ttl
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для сериализации"""
        return asdict(self)
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedOCRResult':
        """Создает из словаря"""
        return cls(**data)
class OCRCache:
    """Кэш для результатов OCR"""
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.cache: Dict[str, CachedOCRResult] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.access_times: Dict[str, float] = {}
        self.hit_count = 0
        self.miss_count = 0
    def _get_image_hash(self, image_bytes: bytes) -> str:
        """Вычисляет хэш изображения"""
        return hashlib.md5(image_bytes).hexdigest()
    def _is_lru_eviction_needed(self) -> bool:
        """Проверяет, нужно ли удаление по LRU"""
        return len(self.cache) >= self.max_size
    def _evict_lru(self):
        """Удаляет наименее недавно использованный элемент"""
        if not self.access_times:
            return
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        if lru_key in self.cache:
            del self.cache[lru_key]
        if lru_key in self.access_times:
            del self.access_times[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    def _cleanup_expired(self):
        """Удаляет истекшие записи"""
        current_time = time.time()
        expired_keys = []
        for key, cached_result in self.cache.items():
            if cached_result.is_expired():
                expired_keys.append(key)
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    def get(self, image_bytes: bytes) -> Optional[OCRResult]:
        """
        Получает результат OCR из кэша
        Args:
            image_bytes: Байты изображения
        Returns:
            OCRResult или None, если не найдено в кэше
        """
        image_hash = self._get_image_hash(image_bytes)
        if image_hash not in self.cache:
            self.miss_count += 1
            return None
        cached_result = self.cache[image_hash]
        if cached_result.is_expired():
            del self.cache[image_hash]
            if image_hash in self.access_times:
                del self.access_times[image_hash]
            self.miss_count += 1
            return None
        self.access_times[image_hash] = time.time()
        self.hit_count += 1
        logger.debug(f"OCR cache hit for hash: {image_hash[:8]}...")
        try:
            return self._dict_to_ocr_result(cached_result.result)
        except Exception as e:
            logger.error(f"Error deserializing cached OCR result: {e}")
            del self.cache[image_hash]
            if image_hash in self.access_times:
                del self.access_times[image_hash]
            self.miss_count += 1
            return None
    def put(self, image_bytes: bytes, ocr_result: OCRResult, ttl: Optional[float] = None) -> None:
        """
        Сохраняет результат OCR в кэш
        Args:
            image_bytes: Байты изображения
            ocr_result: Результат OCR
            ttl: Время жизни в секундах (по умолчанию default_ttl)
        """
        image_hash = self._get_image_hash(image_bytes)
        self._cleanup_expired()
        if self._is_lru_eviction_needed():
            self._evict_lru()
        try:
            result_dict = self._ocr_result_to_dict(ocr_result)
        except Exception as e:
            logger.error(f"Error serializing OCR result: {e}")
            return
        cached_result = CachedOCRResult(
            result=result_dict,
            timestamp=time.time(),
            image_hash=image_hash,
            ttl=ttl or self.default_ttl
        )
        self.cache[image_hash] = cached_result
        self.access_times[image_hash] = time.time()
        logger.debug(f"OCR result cached with hash: {image_hash[:8]}...")
    def _ocr_result_to_dict(self, ocr_result: OCRResult) -> Dict[str, Any]:
        """Преобразует OCRResult в словарь для сериализации"""
        return {
            "accounts": [
                {
                    "account_id": account.account_id,
                    "positions": [
                        {
                            "raw_name": pos.raw_name,
                            "quantity": pos.quantity,
                            "hints": pos.hints or {}
                        }
                        for pos in (account.positions or [])
                    ]
                }
                for account in (ocr_result.accounts or [])
            ],
            "cash_positions": [
                {
                    "raw_name": cash.raw_name,
                    "amount": cash.amount,
                    "currency": cash.currency,
                    "account_id": cash.account_id
                }
                for cash in (ocr_result.cash_positions or [])
            ],
            "reason": ocr_result.reason,
            "is_portfolio": ocr_result.is_portfolio,
            "warnings": ocr_result.warnings,
            "portfolio_name": getattr(ocr_result, "portfolio_name", None),
            "portfolio_value": getattr(ocr_result, "portfolio_value", None),
            "currency": getattr(ocr_result, "currency", None),
            "positions_count": getattr(ocr_result, "positions_count", None)
        }
    def _dict_to_ocr_result(self, data: Dict[str, Any]) -> OCRResult:
        """Восстанавливает OCRResult из словаря"""
        from bot.ai.vision import ExtractedPosition, ExtractedAccount, ExtractedCashPosition
        accounts = []
        for account_data in data.get("accounts", []):
            positions = []
            for pos_data in account_data.get("positions", []):
                position = ExtractedPosition(
                    raw_name=pos_data["raw_name"],
                    quantity=pos_data["quantity"],
                    hints=pos_data.get("hints", {})
                )
                positions.append(position)
            account = ExtractedAccount(
                account_id=account_data.get("account_id"),
                positions=positions
            )
            accounts.append(account)
        cash_positions = []
        for cash_data in data.get("cash_positions", []):
            cash = ExtractedCashPosition(
                raw_name=cash_data["raw_name"],
                amount=cash_data["amount"],
                currency=cash_data["currency"],
                account_id=cash_data.get("account_id")
            )
            cash_positions.append(cash)
        ocr_result = OCRResult(
            accounts=accounts,
            cash_positions=cash_positions,
            reason=data.get("reason", "cached"),
            is_portfolio=data.get("is_portfolio", False),
            warnings=data.get("warnings")
        )
        if "portfolio_name" in data:
            ocr_result.portfolio_name = data["portfolio_name"]
        if "portfolio_value" in data:
            ocr_result.portfolio_value = data["portfolio_value"]
        if "currency" in data:
            ocr_result.currency = data["currency"]
        if "positions_count" in data:
            ocr_result.positions_count = data["positions_count"]
        return ocr_result
    def clear(self):
        """Очищает весь кэш"""
        self.cache.clear()
        self.access_times.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("OCR cache cleared")
    def get_stats(self) -> Dict[str, Any]:
        """Возвращает статистику кэша"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }
    def cleanup_expired(self):
        """Публичный метод для очистки истекших записей"""
        self._cleanup_expired()
ocr_cache = OCRCache(max_size=1000, default_ttl=3600)