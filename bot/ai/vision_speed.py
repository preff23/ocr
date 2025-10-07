"""
Optimized OCR processing for speed profile
"""
import base64
import io
import json
import re
import hashlib
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from PIL import Image
import cv2
import numpy as np
from bot.core.config import config
from bot.core.logging import get_logger
from bot.utils.timing import timing_decorator, cache_hit, cache_miss
from bot.utils.speed_cache import image_hash_cache
from bot.ai.vision import ExtractedPosition, OCRAccount, CashPosition, OCRResult
logger = get_logger(__name__)
def preprocess_image_speed(image_bytes: bytes) -> bytes:
    """Optimized image preprocessing for speed"""
    if not config.feature_speed_profile:
        from bot.ai.vision import compress_image
        return compress_image(image_bytes, max_width=1024, quality=75)
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        width, height = image.size
        max_dim = config.ocr_image_max_dim_px
        if width > max_dim or height > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int((height * max_dim) / width)
            else:
                new_height = max_dim
                new_width = int((width * max_dim) / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=config.ocr_image_jpeg_quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}")
        return image_bytes
def quick_text_check(image_bytes: bytes) -> bool:
    """Quick check if image contains text without LLM call"""
    if not config.feature_speed_profile:
        return True
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return False
        small_img = cv2.resize(img, (256, 256))
        gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        has_text = edge_density > 0.01
        logger.debug(f"Quick text check: edge_density={edge_density:.4f}, has_text={has_text}")
        return has_text
    except Exception as e:
        logger.warning(f"Quick text check failed: {e}")
        return True
class SpeedVisionProcessor:
    """Optimized vision processor for speed profile"""
    def __init__(self):
        self.client = openai.AsyncOpenAI(api_key=config.openai_api_key)
        self.model = config.vision_model
        self._semaphore = asyncio.Semaphore(config.ocr_concurrency)
    @timing_decorator("total", "ocr")
    async def extract_positions_for_ingest(self, image_bytes: bytes) -> OCRResult:
        """Extract positions with speed optimizations"""
        if not config.feature_speed_profile:
            from bot.ai.vision import vision_processor
            return await vision_processor.extract_positions_for_ingest(image_bytes)
        cached_result = image_hash_cache.get_ocr_result(image_bytes)
        if cached_result:
            cache_hit("ocr")
            logger.info("OCR cache hit")
            return cached_result
        cache_miss("ocr")
        if not quick_text_check(image_bytes):
            logger.info("Quick text check failed, skipping OCR")
            result = OCRResult(
                accounts=[],
                cash_positions=[],
                reason="not_portfolio",
                is_portfolio=False
            )
            image_hash_cache.set_ocr_result(image_bytes, result)
            return result
        async with self._semaphore:
            result = await self._process_with_llm(image_bytes)
            image_hash_cache.set_ocr_result(image_bytes, result)
            return result
    @timing_decorator("call", "ocr")
    async def _process_with_llm(self, image_bytes: bytes) -> OCRResult:
        """Process image with LLM"""
        compressed_image = preprocess_image_speed(image_bytes)
        image_base64 = base64.b64encode(compressed_image).decode('utf-8')
        prompt = self._load_ocr_prompt()
        request_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0,
            "top_p": 1
        }
        if hasattr(openai, 'types') and hasattr(openai.types, 'ChatCompletionResponseFormat'):
            request_params["response_format"] = {"type": "json_object"}
        try:
            response = await asyncio.wait_for(
                self.client.chat.completions.create(**request_params),
                timeout=config.request_timeout_ms / 1000
            )
        except asyncio.TimeoutError:
            logger.error("OCR API call timeout")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
        except Exception as e:
            logger.error(f"OCR API call failed: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
        content = response.choices[0].message.content
        return self._parse_ocr_response(content)
    @timing_decorator("parse", "ocr")
    def _parse_ocr_response(self, content: str) -> OCRResult:
        """Parse OCR response with optimized JSON parsing"""
        try:
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    logger.error("No valid JSON found in OCR response")
                    return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
            accounts = []
            if "accounts" in data and data["accounts"]:
                for account_data in data["accounts"]:
                    positions = []
                    for pos_data in account_data.get("positions", []):
                        position = ExtractedPosition(
                            raw_name=pos_data.get("raw_name", ""),
                            raw_ticker=pos_data.get("raw_ticker"),
                            raw_isin=pos_data.get("raw_isin"),
                            raw_type=pos_data.get("raw_type"),
                            quantity=pos_data.get("quantity"),
                            quantity_unit=pos_data.get("quantity_unit"),
                            confidence=pos_data.get("confidence", 0.9)
                        )
                        positions.append(position)
                    account = OCRAccount(
                        account_id=account_data.get("account_id", "default"),
                        account_name=account_data.get("account_name"),
                        portfolio_value=account_data.get("portfolio_value"),
                        currency=account_data.get("currency"),
                        positions=positions
                    )
                    accounts.append(account)
            cash_positions = []
            if "cash_positions" in data and data["cash_positions"]:
                for cash_data in data["cash_positions"]:
                    cash_position = CashPosition(
                        account_id=cash_data.get("account_id", "default"),
                        raw_name=cash_data.get("raw_name", ""),
                        amount=cash_data.get("amount"),
                        currency=cash_data.get("currency")
                    )
                    cash_positions.append(cash_position)
            return OCRResult(
                accounts=accounts,
                cash_positions=cash_positions,
                reason=data.get("reason", "ok"),
                is_portfolio=data.get("is_portfolio", True),
                portfolio_name=data.get("portfolio_name"),
                portfolio_value=data.get("portfolio_value"),
                currency=data.get("currency"),
                positions_count=data.get("positions_count")
            )
        except Exception as e:
            logger.error(f"Failed to parse OCR response: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
    def _load_ocr_prompt(self) -> str:
        """Load OCR prompt"""
        try:
            with open("bot/ai/ocr_prompt_portfolio_v10_minimal.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error("OCR prompt not found")
            return "{}"
speed_vision_processor = SpeedVisionProcessor()