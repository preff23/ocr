import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import re
from bot.core.logging import get_logger
from bot.utils.normalize import normalize_security_name, normalize_security_series
ALIAS_STOPWORDS = [
    'КОМПАНИЯ', 'КОМПАНИ', 'КОМП',
    'ТРАНСПОРТНАЯ', 'ТРАНСПОРТН', 'ТРАНСПОРТ', 'ТРАНС',
    'ТОРГОВЫЙ ДОМ', 'ТОРГОВЫЙДОМ', 'ТОРГОВЫЙ', 'ТД',
    'ГОСУДАРСТВЕННАЯ', 'ГОС', 'ГКЛ', 'ГТЛК',
]
logger = get_logger(__name__)
@dataclass
class BondReferenceEntry:
    isin: str
    name: Optional[str] = None
    full_name: Optional[str] = None
    issuer_name: Optional[str] = None
    ticker: Optional[str] = None
    board: Optional[str] = None
    sources: Optional[List[str]] = None
    normalized_name: Optional[str] = None
    normalized_full_name: Optional[str] = None
    alias_keys: Optional[List[str]] = None
FALLBACK_ENTRIES = [
    BondReferenceEntry(
        isin="BYM000002154",
        name="РесБел 340",
        full_name="РесБел 340 29.06.2027",
        issuer_name="РесБел",
        ticker="BYM000002154",
        normalized_name=normalize_security_name("РесБел 340"),
        normalized_full_name=normalize_security_name("РесБел 340 29.06.2027"),
        alias_keys=[normalize_security_name("РЕСБЕЛ340"), normalize_security_name("РесБел 340"), normalize_security_name("РесБел340")],
        sources=["fallback"],
    ),
]
class BondReference:
    def __init__(self, entries: List[BondReferenceEntry]):
        self.entries = entries
        self.by_isin: Dict[str, BondReferenceEntry] = {}
        self.by_ticker: Dict[str, BondReferenceEntry] = {}
        self.by_normalized_name: Dict[str, BondReferenceEntry] = {}
        self.by_alias: Dict[str, BondReferenceEntry] = {}
        self._normalized_pairs: List[Tuple[str, BondReferenceEntry]] = []
        for entry in entries:
            if entry.isin:
                self.by_isin.setdefault(entry.isin.upper(), entry)
            if entry.ticker:
                self.by_ticker.setdefault(entry.ticker.upper(), entry)
            if entry.normalized_name:
                self.by_normalized_name.setdefault(entry.normalized_name, entry)
            if entry.normalized_full_name and entry.normalized_full_name not in self.by_normalized_name:
                self.by_normalized_name[entry.normalized_full_name] = entry
            for alias in entry.alias_keys or []:
                self.by_alias.setdefault(alias, entry)
        self._normalized_pairs = list(self.by_alias.items())
    def match(self, *, isin: Optional[str], ticker: Optional[str], name: Optional[str]) -> Optional[BondReferenceEntry]:
        if isin:
            entry = self.by_isin.get(isin.upper())
            if entry:
                return entry
        if ticker:
            entry = self.by_ticker.get(ticker.upper())
            if entry:
                return entry
        if name:
            normalized = normalize_security_name(name)
            if normalized in self.by_normalized_name:
                return self.by_normalized_name[normalized]
            variants = {
                normalized,
                normalized.replace(' ', ''),
                normalized.replace('-', ''),
                normalized.replace(' ', '').replace('-', ''),
            }
            normalized_series = normalize_security_name(normalize_security_series(name))
            if normalized_series and normalized_series != normalized:
                variants.update({
                    normalized_series,
                    normalized_series.replace(' ', ''),
                    normalized_series.replace('-', ''),
                    normalized_series.replace(' ', '').replace('-', ''),
                })
            variants_with_bo = set(variants)
            for variant in list(variants_with_bo):
                cleaned = re.sub(r'\b[БB][ОO]\b', '', variant)
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
                if cleaned:
                    variants.update({
                        cleaned,
                        cleaned.replace(' ', ''),
                        cleaned.replace('-', ''),
                        cleaned.replace(' ', '').replace('-', ''),
                    })
            for variant in variants:
                if variant and variant in self.by_alias:
                    return self.by_alias[variant]
        return None
    def match_fuzzy(self, name: Optional[str], threshold: float = 0.82) -> Optional[BondReferenceEntry]:
        if not name or not self._normalized_pairs:
            return None
        normalized = normalize_security_name(name)
        if not normalized:
            return None
        normalized_compact = normalized.replace(' ', '')
        best_entry: Optional[BondReferenceEntry] = None
        best_score = threshold
        for key, entry in self._normalized_pairs:
            score = SequenceMatcher(None, normalized, key).ratio()
            if score > best_score:
                best_score = score
                best_entry = entry
                continue
            if normalized_compact:
                score_compact = SequenceMatcher(None, normalized_compact, key).ratio()
                if score_compact > best_score:
                    best_score = score_compact
                    best_entry = entry
        return best_entry
    def enrich_position(self, position: Dict[str, any]) -> bool:
        entry = self.match(
            isin=position.get("isin"),
            ticker=position.get("ticker") or position.get("raw_ticker"),
            name=position.get("normalized_name") or position.get("raw_name"),
        )
        if not entry and position.get("raw_name"):
            entry = self.match_fuzzy(position.get("raw_name"))
        if not entry:
            return False
        changed = False
        if entry.isin and position.get("isin") != entry.isin:
            position["isin"] = entry.isin
            changed = True
        if entry.ticker and position.get("ticker") != entry.ticker:
            position["ticker"] = entry.ticker
            changed = True
        if entry.name and position.get("normalized_name") not in (entry.name, entry.full_name):
            position["normalized_name"] = entry.name
            changed = True
        if entry.full_name and entry.full_name not in position.get("aliases", []):
            position.setdefault("aliases", []).append(entry.full_name)
            changed = True
        hints = position.setdefault("hints", {})
        hints.setdefault("reference_sources", entry.sources or [])
        hints.setdefault("reference_name", entry.name)
        hints.setdefault("reference_full_name", entry.full_name)
        hints.setdefault("reference_issuer", entry.issuer_name)
        return changed
def _add_alias_variants(bucket: set, value: str) -> None:
    if not value:
        return
    variants = {value}
    def _cleanup(text: str) -> str:
        cleaned = re.sub(r'\b[БB][ОO]\b', '', text)
        for stopword in ALIAS_STOPWORDS:
            cleaned = cleaned.replace(stopword, ' ')
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    cleaned = _cleanup(value)
    if cleaned:
        variants.add(cleaned)
    stripped = _cleanup(value)
    if stripped and stripped != value:
        variants.add(stripped)
    for variant in list(variants):
        compact = variant.replace(' ', '')
        dashless = variant.replace('-', '')
        if compact:
            variants.add(compact)
        if dashless:
            variants.add(dashless)
    bucket.update(filter(None, variants))
def _generate_aliases(entry: BondReferenceEntry) -> List[str]:
    aliases = set()
    for candidate in [entry.name, entry.full_name, entry.ticker]:
        if not candidate:
            continue
        normalized = normalize_security_name(candidate)
        if normalized:
            _add_alias_variants(aliases, normalized)
        normalized_series = normalize_security_name(normalize_security_series(candidate))
        if normalized_series and normalized_series != normalized:
            _add_alias_variants(aliases, normalized_series)
    return sorted({alias for alias in aliases if alias})
def _load_entries(path: Path) -> List[BondReferenceEntry]:
    if not path.exists():
        logger.warning(f"Bond reference file not found: {path}")
        return FALLBACK_ENTRIES.copy()
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load bond reference from {path}: {e}")
        return FALLBACK_ENTRIES.copy()
    entries: List[BondReferenceEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        isin = item.get("isin")
        if not isin:
            continue
        entry = BondReferenceEntry(
            isin=isin.strip(),
            name=item.get("name"),
            full_name=item.get("full_name"),
            issuer_name=item.get("issuer_name"),
            ticker=item.get("ticker"),
            board=item.get("board"),
            sources=item.get("sources") if isinstance(item.get("sources"), list) else [],
        )
        entry.normalized_name = normalize_security_name(entry.name) if entry.name else None
        entry.normalized_full_name = normalize_security_name(entry.full_name) if entry.full_name else None
        entry.alias_keys = _generate_aliases(entry)
        entries.append(entry)
    logger.info(f"Loaded {len(entries)} bond reference entries from {path}")
    if entries:
        for fallback in FALLBACK_ENTRIES:
            fallback.alias_keys = _generate_aliases(fallback)
        entries.extend(FALLBACK_ENTRIES)
    return entries
@lru_cache(maxsize=1)
def load_bond_reference(path: Optional[Path] = None) -> Optional[BondReference]:
    target_path = path or (Path("data") / "bonds_reference.json")
    entries = _load_entries(target_path)
    if not entries:
        return None
    return BondReference(entries)