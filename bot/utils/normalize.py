import re
from typing import List, Set
NON_SECURITY_PATTERNS = [
    r'золот[оа]',
    r'рубл[ьи]',
    r'доллар',
    r'евро',
    r'индекс',
    r'индексы',
    r'фонд\s+индекс',
    r'etf',
    r'spdr',
    r'ishares',
    r'валюта',
    r'валютн',
    r'металл',
    r'нефть\s+сырая',
    r'газ\s+природный',
    r'commodity',
    r'товар',
    r'крипто',
    r'bitcoin',
    r'эфир',
    r'ethereum',
    r'crypto',
    r'депозит',
    r'вклад',
    r'счет',
    r'наличн',
    r'кэш',
    r'cash',
    r'деньги',
    r'money',
    r'баланс',
    r'итого',
    r'сумма',
    r'total',
    r'портфель',
    r'portfolio',
    r'актив',
    r'пассив',
    r'активы',
    r'пассивы',
    r'имущество',
    r'недвижимость',
    r'real estate',
    r'property',
    r'земля',
    r'здание',
    r'сооружение',
    r'оборудование',
    r'техника',
    r'автомобиль',
    r'машина',
    r'транспорт',
    r'мебель',
    r'бытовая техника',
    r'электроника',
    r'компьютер',
    r'телефон',
    r'планшет',
    r'ноутбук',
    r'часы',
    r'украшения',
    r'драгоценности',
    r'ювелирные',
    r'антиквариат',
    r'коллекция',
    r'картины',
    r'скульптуры',
    r'книги',
    r'документы',
    r'ценные бумаги',
    r'сертификаты',
    r'патенты',
    r'лицензии',
    r'авторские права',
    r'торговые марки',
    r'бренды',
    r'goodwill',
    r'деловая репутация',
    r'нематериальные активы',
    r'интеллектуальная собственность',
    r'права требования',
    r'дебиторская задолженность',
    r'кредиторская задолженность',
    r'займы',
    r'кредиты',
    r'ссуды',
    r'обязательства',
    r'долги',
    r'задолженность',
    r'расчеты',
    r'платежи',
    r'поступления',
    r'выплаты',
    r'дивиденды',
    r'проценты',
    r'купон',
    r'доход',
    r'прибыль',
    r'убыток',
    r'результат',
    r'финансовый результат',
    r'операционный результат',
    r'внереализационный результат',
    r'налог',
    r'сбор',
    r'пошлина',
    r'штраф',
    r'пеня',
    r'неустойка',
    r'комиссия',
    r'плата',
    r'вознаграждение',
    r'гонорар',
    r'зарплата',
    r'заработная плата',
    r'оплата труда',
    r'социальные взносы',
    r'пенсионные взносы',
    r'медицинские взносы',
    r'страховые взносы',
    r'страхование',
    r'страховка',
    r'полис',
    r'страховой случай',
    r'выплата по страховке',
    r'страховое возмещение',
    r'страховая премия',
    r'страховой взнос',
    r'страховой тариф',
    r'страховой коэффициент',
    r'страховой риск',
    r'страховой интерес',
    r'страховое покрытие',
    r'страховая сумма',
    r'страховая стоимость',
    r'страховое возмещение',
    r'страховая выплата',
    r'страховой случай',
    r'страховой договор',
    r'страховой полис',
    r'страховая компания',
    r'страховщик',
    r'страхователь',
    r'застрахованное лицо',
    r'выгодоприобретатель',
    r'страховой агент',
    r'страховой брокер',
    r'страховой посредник',
    r'страховой рынок',
    r'страховая деятельность',
    r'страховые услуги',
    r'страховые продукты',
    r'страховые программы',
    r'страховые тарифы',
    r'страховые премии',
    r'страховые взносы',
    r'страховые выплаты',
    r'страховые возмещения',
    r'страховые убытки',
    r'страховые резервы',
    r'страховые фонды',
    r'страховые капиталы',
    r'страховые активы',
    r'страховые пассивы',
    r'страховой баланс',
    r'страховая отчетность',
    r'страховой учет',
    r'страховой аудит',
    r'страховой надзор',
    r'страховое регулирование',
    r'страховое законодательство',
    r'страховое право',
    r'страховые стандарты',
    r'страховые правила',
    r'страховые условия',
    r'страховые тарифы',
    r'страховые премии',
    r'страховые взносы',
    r'страховые выплаты',
    r'страховые возмещения',
    r'страховые убытки',
    r'страховые резервы',
    r'страховые фонды',
    r'страховые капиталы',
    r'страховые активы',
    r'страховые пассивы',
    r'страховой баланс',
    r'страховая отчетность',
    r'страховой учет',
    r'страховой аудит',
    r'страховой надзор',
    r'страховое регулирование',
    r'страховое законодательство',
    r'страховое право',
    r'страховые стандарты',
    r'страховые правила',
    r'страховые условия',
]
def normalize_security_name(name: str) -> str:
    """Normalize security name for better matching."""
    if not name or not isinstance(name, str):
        return ""
    normalized = name.upper().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    replacements = [
        (r'А', 'A'),
        (r'Б', 'B'),
        (r'В', 'V'),
        (r'Г', 'G'),
        (r'Д', 'D'),
        (r'Е', 'E'),
        (r'Ж', 'ZH'),
        (r'З', 'Z'),
        (r'И', 'I'),
        (r'Й', 'J'),
        (r'К', 'K'),
        (r'Л', 'L'),
        (r'М', 'M'),
        (r'Н', 'N'),
        (r'О', 'O'),
        (r'П', 'P'),
        (r'Р', 'R'),
        (r'С', 'S'),
        (r'Т', 'T'),
        (r'У', 'U'),
        (r'Ф', 'F'),
        (r'Х', 'H'),
        (r'Ц', 'TS'),
        (r'Ч', 'CH'),
        (r'Ш', 'SH'),
        (r'Щ', 'SCH'),
        (r'Ъ', ''),
        (r'Ы', 'Y'),
        (r'Ь', ''),
        (r'Э', 'E'),
        (r'Ю', 'YU'),
        (r'Я', 'YA'),
        (r'БО(\d+)', r'BO\1'),
        (r'(\d+)Р', r'\1P'),
        (r'О(\d+)', r'O\1'),
        (r'об6|o62', 'обл'),
        (r'0Б', 'OB'),
        (r'Б0', 'BO'),
        (r'ОБ(\d+)', r'OB\1'),
        (r'BО', 'BO'),
        (r'OБ', 'OB'),
        (r'O{2}(?=P)', '00'),
        (r'O{2}(?=\d)', '00'),
        (r'(?<=\b)O(?=\d)', '0'),
        (r'(?<=\d)O(?=\d)', '0'),
        (r'(?<=\b)I(?=P)', '1'),
        (r'(?<=\b)I(?=\d)', '1'),
        (r'(?<=\d)I(?=\d)', '1'),
        (r'[^\w\s\-\.]', ''),
        (r'\.+', '.'),
        (r'-+', '-'),
    ]
    for pattern, replacement in replacements:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = normalized.strip(' .-')
    return normalized
def is_security_name(name: str) -> bool:
    """Check if the name looks like a security (not gold, currency, etc.)."""
    if not name or not isinstance(name, str):
        return False
    normalized = name.lower().strip()
    for pattern in NON_SECURITY_PATTERNS:
        if re.search(pattern, normalized, re.IGNORECASE):
            return False
    if not re.search(r'[a-zA-Zа-яА-Я0-9]', normalized):
        return False
    if len(normalized) < 2:
        return False
    return True
def extract_ticker_from_name(name: str) -> str:
    """Try to extract ticker from security name."""
    if not name:
        return ""
    ticker_patterns = [
        r'\b([A-Z]{3,6})\b',
        r'\b([A-Z]{2,4}\d{1,2})\b',
        r'\b(\d{3,6}[A-Z]{1,3})\b',
    ]
    for pattern in ticker_patterns:
        matches = re.findall(pattern, name.upper())
        if matches:
            for match in matches:
                if len(match) >= 3 and len(match) <= 8:
                    return match
    return ""
def extract_isin_from_name(name: str) -> str:
    """Try to extract ISIN from security name."""
    if not name:
        return ""
    isin_pattern = r'\b([A-Z]{2}[A-Z0-9]{9}[0-9])\b'
    matches = re.findall(isin_pattern, name.upper())
    if matches:
        return matches[0]
    return ""
def generate_normalized_key(name: str, ticker: str = "", isin: str = "") -> str:
    """Generate normalized key for deduplication."""
    parts = []
    if isin:
        parts.append(f"ISIN:{isin.upper()}")
    if ticker:
        parts.append(f"TICKER:{ticker.upper()}")
    if name:
        normalized_name = normalize_security_name(name)
        if normalized_name:
            parts.append(f"NAME:{normalized_name}")
    return "|".join(parts) if parts else ""
def filter_securities(names: List[str]) -> List[str]:
    """Filter list of names to keep only securities."""
    return [name for name in names if is_security_name(name)]
def deduplicate_securities(securities: List[dict]) -> List[dict]:
    """Remove duplicate securities based on normalized key."""
    seen_keys = set()
    result = []
    for security in securities:
        key = generate_normalized_key(
            security.get('normalized_name', security.get('name', '')),
            security.get('ticker', ''),
            security.get('isin', '')
        )
        if key and key not in seen_keys:
            seen_keys.add(key)
            result.append(security)
    return result
def normalize_security_series(name: str) -> str:
    """Normalize security series (БО->BO, 001Р->001P, etc.)."""
    if not name or not isinstance(name, str):
        return ""
    normalized = name
    series_fixes = [
        (r'БО(\d+)', r'BO\1'),
        (r'бо(\d+)', r'BO\1'),
        (r'(\d+)Р', r'\1P'),
        (r'(\d+)р', r'\1P'),
        (r'О6|o62', 'обл'),
        (r'об6', 'обл'),
        (r'обл', '001'),
        (r'обб', '001'),
        (r'0662', '001'),
        (r'06{2}2', '001'),
        (r'ГOCT', 'ГOС'),
        (r'00162', '001P'),
        (r'0012', '001P'),
        (r'001П', '001P'),
        (r'001PП', '001P'),
        (r'001PP', '001P'),
        (r'001P(\d+)', r'001P-\1'),
        (r'ob662П(\d+)', r'облигации П\1'),
        (r'(\d+)Р$', r'\1P'),
        (r'(\d+)р$', r'\1P'),
    ]
    for pattern, replacement in series_fixes:
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    return normalized
def normalize_security_type(raw_type: str) -> str:
    """Normalize security type to standard values."""
    if not raw_type or not isinstance(raw_type, str):
        return None
    normalized = raw_type.lower().strip()
    type_mapping = {
        "акция": "акция",
        "акции": "акция",
        "share": "акция",
        "shares": "акция",
        "stock": "акция",
        "stocks": "акция",
        "облигация": "облигация",
        "облигации": "облигация",
        "бонд": "облигация",
        "bond": "облигация",
        "bonds": "облигация",
        "офз": "облигация",
        "ofz": "облигация",
        "etf": "фонд",
        "бпиф": "фонд",
        "пиф": "фонд",
        "фонд": "фонд",
        "fund": "фонд",
        "funds": "фонд",
    }
    return type_mapping.get(normalized)