"""
Оптимизированный Portfolio Ingest Pipeline с Speed Profile
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from bot.core.config import Config
from bot.core.logging import get_logger
from bot.pipeline.portfolio_ingest_pipeline import PortfolioIngestPipeline, IngestResult, ExtractedPosition
from bot.utils.timing import timing_decorator
from bot.utils.speed_cache import market_data_cache
logger = get_logger(__name__)
@dataclass
class BondSearchResult:
    """Результат поиска облигации"""
    isin: str
    found: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    search_time: float = 0.0
class SpeedPortfolioIngestPipeline(PortfolioIngestPipeline):
    """Оптимизированный Portfolio Ingest Pipeline"""
    def __init__(self):
        super().__init__()
        self.config = Config.from_env()
        self.aggregator = MarketDataAggregator()
        self.semaphore = asyncio.Semaphore(self.config.md_concurrency)
        self.executor = ThreadPoolExecutor(max_workers=self.config.md_concurrency)
    @timing_decorator("ingest_total", "ingest")
    async def ingest_from_ocr_batch(
        self, 
        ocr_results: List[Any], 
        user_id: int
    ) -> IngestResult:
        """Оптимизированный ингест с параллельным поиском и кэшированием"""
        if not self.config.feature_speed_profile:
            return await super().ingest_from_ocr_batch(ocr_results, user_id)
        logger.info(f"Speed Ingest: Processing {len(ocr_results)} OCR results for user {user_id}")
        start_time = time.time()
        all_positions = []
        for ocr_result in ocr_results:
            if ocr_result.is_portfolio and ocr_result.accounts:
                for account in ocr_result.accounts:
                    all_positions.extend(account.positions)
        if not all_positions:
            logger.warning("No positions found in OCR results")
            return IngestResult(
                added=0,
                positions=[],
                raw_detected=0,
                normalized=0,
                resolved=0
            )
        logger.info(f"Found {len(all_positions)} positions to process")
        search_results = await self._parallel_bond_search(all_positions)
        resolved_positions = []
        for position, search_result in zip(all_positions, search_results):
            if search_result.found and search_result.data:
                resolved_pos = self._create_resolved_position(position, search_result.data)
                if resolved_pos:
                    resolved_positions.append(resolved_pos)
        added_count = await self._bulk_save_positions(resolved_positions, user_id)
        total_time = time.time() - start_time
        logger.info(f"Speed Ingest completed: {added_count} positions added in {total_time:.2f}s")
        return IngestResult(
            added=added_count,
            positions=resolved_positions,
            raw_detected=len(all_positions),
            normalized=len(all_positions),
            resolved=len(resolved_positions)
        )
    @timing_decorator("parallel_bond_search", "ingest")
    async def _parallel_bond_search(self, positions: List[ExtractedPosition]) -> List[BondSearchResult]:
        """Параллельный поиск облигаций с кэшированием и таймаутами"""
        logger.info(f"Starting parallel search for {len(positions)} positions")
        tasks = []
        for position in positions:
            task = self._search_single_bond_with_timeout(position)
            tasks.append(task)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        search_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Search failed for position {i}: {result}")
                search_results.append(BondSearchResult(
                    isin=positions[i].raw_ticker or "unknown",
                    found=False,
                    error=str(result)
                ))
            else:
                search_results.append(result)
        found_count = sum(1 for r in search_results if r.found)
        cache_hits = sum(1 for r in search_results if hasattr(r, 'from_cache') and r.from_cache)
        logger.info(f"Search completed: {found_count}/{len(positions)} found, {cache_hits} cache hits")
        return search_results
    async def _search_single_bond_with_timeout(self, position: ExtractedPosition) -> BondSearchResult:
        """Поиск одной облигации с таймаутом и кэшированием"""
        search_key = position.raw_ticker or position.raw_name or "unknown"
        start_time = time.time()
        cached_result = market_data_cache.get(search_key)
        if cached_result:
            logger.debug(f"Cache hit for {search_key}")
            return BondSearchResult(
                isin=search_key,
                found=True,
                data=cached_result,
                search_time=time.time() - start_time
            )
        try:
            async with asyncio.timeout(self.config.request_timeout_ms / 1000):
                async with self.semaphore:
                    search_data = await self._try_multiple_search_methods(position)
                    if search_data:
                        market_data_cache.set(search_key, search_data)
                        return BondSearchResult(
                            isin=search_key,
                            found=True,
                            data=search_data,
                            search_time=time.time() - start_time
                        )
                    else:
                        return BondSearchResult(
                            isin=search_key,
                            found=False,
                            search_time=time.time() - start_time
                        )
        except asyncio.TimeoutError:
            logger.warning(f"Search timeout for {search_key}")
            return BondSearchResult(
                isin=search_key,
                found=False,
                error="Timeout",
                search_time=time.time() - start_time
            )
        except Exception as e:
            logger.warning(f"Search error for {search_key}: {e}")
            return BondSearchResult(
                isin=search_key,
                found=False,
                error=str(e),
                search_time=time.time() - start_time
            )
    async def _try_multiple_search_methods(self, position: ExtractedPosition) -> Optional[Dict[str, Any]]:
        """Пробуем разные методы поиска облигации"""
        search_terms = [
            position.raw_ticker,
            position.raw_name,
            position.normalized_name,
            position.normalized_key
        ]
        search_terms = list(dict.fromkeys([term for term in search_terms if term]))
        for search_term in search_terms:
            try:
                if search_term:
                    snapshots = await self.aggregator.get_snapshot_for(search_term)
                    if snapshots:
                        result = snapshots[0]
                        logger.debug(f"Found security for '{search_term}': {result.name}")
                        return {
                            'name': result.name,
                            'ticker': result.ticker,
                            'isin': getattr(result, 'isin', None),
                            'sector': getattr(result, 'sector', None),
                            'rating': getattr(result, 'rating', None),
                            'search_term': search_term
                        }
            except Exception as e:
                logger.debug(f"Search failed for '{search_term}': {e}")
                continue
        return None
    def _create_resolved_position(self, position: ExtractedPosition, search_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Создает резолвед позицию из данных поиска"""
        try:
            return {
                'raw_name': position.raw_name,
                'raw_ticker': position.raw_ticker,
                'raw_quantity': position.quantity,
                'raw_type': position.quantity_unit,
                'normalized_name': search_data.get('name', position.raw_name),
                'normalized_key': search_data.get('ticker', position.raw_ticker),
                'isin': search_data.get('isin'),
                'sector': search_data.get('sector'),
                'rating': search_data.get('rating'),
                'search_term': search_data.get('search_term')
            }
        except Exception as e:
            logger.warning(f"Failed to create resolved position: {e}")
            return None
    @timing_decorator("bulk_save_positions", "ingest")
    async def _bulk_save_positions(self, positions: List[Dict[str, Any]], user_id: int) -> int:
        """Массовое сохранение позиций в БД"""
        if not positions:
            return 0
        try:
            db_manager.clear_user_holdings(user_id)
            added_count = 0
            for position_data in positions:
                try:
                    db_manager.add_holding(
                        user_id=user_id,
                        raw_name=position_data.get('raw_name', ''),
                        raw_ticker=position_data.get('raw_ticker', ''),
                        raw_quantity=position_data.get('raw_quantity', 0),
                        raw_type=position_data.get('raw_type', ''),
                        normalized_name=position_data.get('normalized_name', ''),
                        normalized_key=position_data.get('normalized_key', '')
                    )
                    added_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save position: {e}")
                    continue
            logger.info(f"Bulk save completed: {added_count} positions saved")
            return added_count
        except Exception as e:
            logger.error(f"Bulk save failed: {e}")
            return 0
speed_ingest_pipeline = SpeedPortfolioIngestPipeline()