import base64
import io
import json
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from PIL import Image
from bot.core.config import config
from bot.core.logging import get_logger
logger = get_logger(__name__)
def compress_image(image_bytes: bytes, max_width: int = 1024, quality: int = 85) -> bytes:
    """Сжимает изображение для ускорения OCR"""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        width, height = image.size
        if width > max_width:
            new_height = int((height * max_width) / width)
            image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception as e:
        logger.warning(f"Failed to compress image: {e}")
        return image_bytes
@dataclass
class ExtractedPosition:
    raw_name: str
    raw_ticker: Optional[str] = None
    raw_isin: Optional[str] = None
    raw_type: Optional[str] = None
    quantity: Optional[float] = None
    quantity_unit: Optional[str] = None
    confidence: float = 0.0
    hints: Optional[Dict[str, Any]] = None
@dataclass
class OCRAccount:
    account_id: str
    account_name: Optional[str]
    portfolio_value: Optional[float]
    currency: Optional[str]
    positions: List[ExtractedPosition]
    positions_count: Optional[int] = None
    daily_change_value: Optional[float] = None
    daily_change_percent: Optional[float] = None
    cash_balance: Optional["CashPosition"] = None
@dataclass
class CashPosition:
    account_id: str
    raw_name: str
    amount: Optional[float]
    currency: Optional[str]
@dataclass
class OCRResult:
    accounts: List[OCRAccount]
    cash_positions: List[CashPosition]
    reason: str
    is_portfolio: bool = True
    portfolio_name: Optional[str] = None
    portfolio_value: Optional[float] = None
    currency: Optional[str] = None
    positions_count: Optional[int] = None
    warnings: Optional[List[str]] = None
class VisionProcessor:
    def __init__(self):
        self.client = openai.AsyncOpenAI(
            api_key=config.openai_api_key
        )
        self.model = config.vision_model
    async def extract_positions_for_ingest(self, image_bytes: bytes) -> OCRResult:
        """Extract positions from portfolio image for ingest only.
        Args:
            image_bytes: Image bytes
        Returns:
            OCRResult with extracted positions
        """
        try:
            compressed_image = compress_image(image_bytes, max_width=1024, quality=75)
            image_base64 = base64.b64encode(compressed_image).decode('utf-8')
            prompt = self._load_ocr_prompt_v13()
            parser = self._parse_ocr_response_v13
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
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
                max_tokens=4000,
                temperature=0.1
            )
            content = response.choices[0].message.content
            return parser(content)
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
    def _load_ocr_prompt_v10(self) -> str:
        try:
            with open("bot/ai/ocr_prompt_portfolio_v10_minimal.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            logger.error("OCR prompt v10 not found!")
            return "{}"
    def _parse_ocr_response_v10(self, content: str) -> OCRResult:
        """Parse OCR response for v10 minimal prompt"""
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
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        logger.error("Failed to parse JSON even with regex extraction")
                        return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
                else:
                    logger.error("No JSON block found in response")
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
            logger.error(f"Failed to parse OCR response v10: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
    def _load_ocr_prompt_v12(self) -> str:
        """Load OCR prompt v12 - умный промпт с контекстным пониманием"""
        try:
            with open("bot/ai/ocr_prompt_portfolio_v12_smart.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load OCR prompt v12: {e}")
            return self._load_ocr_prompt_v10()
    def _parse_ocr_response_v12(self, response_text: str) -> OCRResult:
        """Parse OCR response v12 - умный парсинг с контекстным пониманием"""
        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            data = json.loads(response_text)
            accounts = []
            if "accounts" in data and data["accounts"]:
                for account_data in data["accounts"]:
                    positions = []
                    if "positions" in account_data and account_data["positions"]:
                        for pos_data in account_data["positions"]:
                            position = ExtractedPosition(
                                raw_name=pos_data.get("raw_name", ""),
                                raw_ticker=pos_data.get("raw_ticker"),
                                raw_isin=pos_data.get("raw_isin"),
                                raw_type=pos_data.get("raw_type"),
                                quantity=pos_data.get("quantity"),
                                quantity_unit=pos_data.get("quantity_unit"),
                                confidence=pos_data.get("confidence", 0.8)
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
            logger.error(f"Failed to parse OCR response v12: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
    def _load_ocr_prompt_v13(self) -> str:
        """Load OCR prompt v13 - контекстный промпт с фокусом на облигации"""
        try:
            with open("bot/ai/ocr_prompt_portfolio_v13_contextual.txt", "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load OCR prompt v13: {e}")
            return self._load_ocr_prompt_v10()
    def _parse_ocr_response_v13(self, response_text: str) -> OCRResult:
        """Parse OCR response v13 - контекстный парсинг с фокусом на облигации"""
        try:
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            data = json.loads(response_text)
            accounts = []
            if "accounts" in data and data["accounts"]:
                for account_data in data["accounts"]:
                    positions = []
                    if "positions" in account_data and account_data["positions"]:
                        for pos_data in account_data["positions"]:
                            position = ExtractedPosition(
                                raw_name=pos_data.get("raw_name", ""),
                                raw_ticker=None,
                                raw_isin=None,
                                raw_type=pos_data.get("raw_type", "облигация"),
                                quantity=pos_data.get("quantity"),
                                quantity_unit=None,
                                confidence=pos_data.get("confidence", 0.9)
                            )
                            positions.append(position)
                    account = OCRAccount(
                        account_id=account_data.get("account_id", "default"),
                        account_name=None,
                        portfolio_value=None,
                        currency=None,
                        positions=positions
                    )
                    accounts.append(account)
            return OCRResult(
                accounts=accounts,
                cash_positions=[],
                reason=data.get("reason", "ok"),
                is_portfolio=data.get("is_portfolio", True),
                portfolio_name=None,
                portfolio_value=None,
                currency=None,
                positions_count=data.get("positions_count")
            )
        except Exception as e:
            logger.error(f"Failed to parse OCR response v13: {e}")
            return OCRResult(accounts=[], cash_positions=[], reason="error", is_portfolio=False)
vision_processor = VisionProcessor()