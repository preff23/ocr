"""
Timing decorators and utilities for performance monitoring
"""
import time
import asyncio
import hashlib
from functools import wraps
from typing import Dict, Any, Optional, Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from bot.core.logging import get_logger
logger = get_logger(__name__)
request_context: ContextVar[Dict[str, Any]] = ContextVar('request_context', default={})
@dataclass
class TimingMetrics:
    """Container for timing metrics"""
    user_id: Optional[int] = None
    batch_id: Optional[str] = None
    n_images: int = 0
    timings: Dict[str, float] = field(default_factory=dict)
    cache_hits: Dict[str, int] = field(default_factory=dict)
    cache_misses: Dict[str, int] = field(default_factory=dict)
    api_calls: Dict[str, Dict[str, Any]] = field(default_factory=dict)
def get_request_context() -> Dict[str, Any]:
    """Get current request context"""
    return request_context.get({})
def set_request_context(context: Dict[str, Any]) -> None:
    """Set request context"""
    request_context.set(context)
def generate_batch_id() -> str:
    """Generate unique batch ID"""
    return hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8]
def timing_decorator(operation_name: str, category: str = "general"):
    """Decorator for timing operations"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not config.feature_speed_profile:
                return await func(*args, **kwargs)
            start_time = time.time()
            context = get_request_context()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                context.setdefault('timings', {})
                context['timings'][f"{category}_{operation_name}"] = elapsed
                logger.info(
                    "Operation completed",
                    operation=operation_name,
                    category=category,
                    duration_ms=round(elapsed * 1000, 2),
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    category=category,
                    duration_ms=round(elapsed * 1000, 2),
                    error=str(e),
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                raise
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not config.feature_speed_profile:
                return func(*args, **kwargs)
            start_time = time.time()
            context = get_request_context()
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                context.setdefault('timings', {})
                context['timings'][f"{category}_{operation_name}"] = elapsed
                logger.info(
                    "Operation completed",
                    operation=operation_name,
                    category=category,
                    duration_ms=round(elapsed * 1000, 2),
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(
                    "Operation failed",
                    operation=operation_name,
                    category=category,
                    duration_ms=round(elapsed * 1000, 2),
                    error=str(e),
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                raise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator
def api_call_timing(service_name: str):
    """Decorator for timing API calls"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not config.feature_speed_profile:
                return await func(*args, **kwargs)
            start_time = time.time()
            context = get_request_context()
            try:
                result = await func(*args, **kwargs)
                elapsed = time.time() - start_time
                context.setdefault('api_calls', {})
                context['api_calls'][service_name] = {
                    'latency_ms': round(elapsed * 1000, 2),
                    'status': 'success',
                    'timestamp': time.time()
                }
                logger.info(
                    "API call completed",
                    service=service_name,
                    latency_ms=round(elapsed * 1000, 2),
                    status='success',
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                context.setdefault('api_calls', {})
                context['api_calls'][service_name] = {
                    'latency_ms': round(elapsed * 1000, 2),
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                logger.error(
                    "API call failed",
                    service=service_name,
                    latency_ms=round(elapsed * 1000, 2),
                    status='error',
                    error=str(e),
                    user_id=context.get('user_id'),
                    batch_id=context.get('batch_id')
                )
                raise
        return async_wrapper
    return decorator
def cache_hit(operation: str):
    """Log cache hit"""
    if not config.feature_speed_profile:
        return
    context = get_request_context()
    context.setdefault('cache_hits', {})
    context['cache_hits'][operation] = context['cache_hits'].get(operation, 0) + 1
def cache_miss(operation: str):
    """Log cache miss"""
    if not config.feature_speed_profile:
        return
    context = get_request_context()
    context.setdefault('cache_misses', {})
    context['cache_misses'][operation] = context['cache_misses'].get(operation, 0) + 1
def log_final_metrics():
    """Log final timing metrics for the request"""
    if not config.feature_speed_profile:
        return
    context = get_request_context()
    timings = context.get('timings', {})
    cache_hits = context.get('cache_hits', {})
    cache_misses = context.get('cache_misses', {})
    api_calls = context.get('api_calls', {})
    total_cache_ops = sum(cache_hits.values()) + sum(cache_misses.values())
    cache_hit_rate = (sum(cache_hits.values()) / total_cache_ops * 100) if total_cache_ops > 0 else 0
    ocr_total = timings.get('ocr_total', 0)
    ingest_total = timings.get('ingest_total', 0)
    analysis_total = timings.get('analysis_total', 0)
    end_to_end = ocr_total + ingest_total + analysis_total
    logger.info(
        "Request completed - Final metrics",
        user_id=context.get('user_id'),
        batch_id=context.get('batch_id'),
        n_images=context.get('n_images', 0),
        ocr_total_ms=round(ocr_total * 1000, 2),
        ingest_total_ms=round(ingest_total * 1000, 2),
        analysis_total_ms=round(analysis_total * 1000, 2),
        end_to_end_ms=round(end_to_end * 1000, 2),
        cache_hit_rate=round(cache_hit_rate, 2),
        sla_met=end_to_end <= 40.0,
        timings=timings,
        api_calls=api_calls
    )
from bot.core.config import config