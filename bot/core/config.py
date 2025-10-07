import os
from typing import Optional
from dataclasses import dataclass

@dataclass
class Config:
    # Vision and AI models
    vision_provider: str = "openai"
    vision_model: str = "gpt-4o-mini"
    analysis_model: str = "gpt-5-mini"
    
    # OCR optimization flags
    feature_ocr_v10: bool = True
    feature_ocr_v11: bool = False
    feature_ocr_v12: bool = False
    feature_ocr_v13: bool = False
    
    # Speed Profile optimization flags
    feature_speed_profile: bool = False
    ocr_image_max_dim_px: int = 1024
    ocr_image_jpeg_quality: int = 75
    ocr_precheck_min_text_chars: int = 12
    md_concurrency: int = 8
    request_timeout_ms: int = 12000
    retry_max_attempts: int = 2
    retry_base_delay_ms: int = 250
    cache_md_ttl_s: int = 1800
    cache_ocr_ttl_s: int = 259200
    cache_calendar_ttl_s: int = 3600
    analysis_token_budget: int = 1200
    analysis_model_routing: str = "4o"
    news_fetch_enabled: bool = True
    
    # OCR concurrency
    ocr_concurrency: int = 4
    
    # Cache settings
    cache_ttl: int = 300
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            vision_provider=os.getenv("VISION_PROVIDER", "openai"),
            vision_model=os.getenv("VISION_MODEL", "gpt-4o-mini"),
            analysis_model=os.getenv("ANALYSIS_MODEL", "gpt-5-mini"),
            feature_ocr_v10=os.getenv("FEATURE_OCR_V10", "1") == "1",
            feature_speed_profile=os.getenv("FEATURE_SPEED_PROFILE", "0") == "1",
            ocr_image_max_dim_px=int(os.getenv("OCR_IMAGE_MAX_DIM_PX", "1024")),
            ocr_image_jpeg_quality=int(os.getenv("OCR_IMAGE_JPEG_QUALITY", "75")),
            ocr_precheck_min_text_chars=int(os.getenv("OCR_PRECHECK_MIN_TEXT_CHARS", "12")),
            md_concurrency=int(os.getenv("MD_CONCURRENCY", "8")),
            request_timeout_ms=int(os.getenv("REQUEST_TIMEOUT_MS", "12000")),
            retry_max_attempts=int(os.getenv("RETRY_MAX_ATTEMPTS", "2")),
            retry_base_delay_ms=int(os.getenv("RETRY_BASE_DELAY_MS", "250")),
            cache_md_ttl_s=int(os.getenv("CACHE_MD_TTL_S", "1800")),
            cache_ocr_ttl_s=int(os.getenv("CACHE_OCR_TTL_S", "259200")),
            cache_calendar_ttl_s=int(os.getenv("CACHE_CALENDAR_TTL_S", "3600")),
            analysis_token_budget=int(os.getenv("ANALYSIS_TOKEN_BUDGET", "1200")),
            analysis_model_routing=os.getenv("ANALYSIS_MODEL_ROUTING", "4o"),
            news_fetch_enabled=os.getenv("NEWS_FETCH_ENABLED", "1") == "1",
            ocr_concurrency=int(os.getenv("OCR_CONCURRENCY", "4")),
            cache_ttl=int(os.getenv("CACHE_TTL", "300")),
        )

config = Config.from_env()