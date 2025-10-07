from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bot.core.logging import get_logger
logger = get_logger(__name__)
def format_trading_status(status: Optional[str]) -> str:
    if not status:
        return "—"
    if status == "SECURITY_TRADING_STATUS_NORMAL_TRADING":
        return "✅ Торги активны"
    elif status == "SECURITY_TRADING_STATUS_NOT_AVAILABLE_FOR_TRADING":
        return "🚨 Торги недоступны"
    elif status == "SECURITY_TRADING_STATUS_CLOSING_AUCTION":
        return "🟠 Закрытие торгов"
    elif status == "SECURITY_TRADING_STATUS_OPENING_AUCTION":
        return "🟠 Открытие торгов"
    elif status == "SECURITY_TRADING_STATUS_BREAK_IN_TRADING":
        return "⏸️ Перерыв в торгах"
    elif status == "SECURITY_TRADING_STATUS_CLOSED":
        return "🔒 Торги закрыты"
    elif status == "SECURITY_TRADING_STATUS_SUSPENDED":
        return "🚨 Торги приостановлены"
    elif status == "SECURITY_TRADING_STATUS_UNKNOWN":
        return "❓ Статус неизвестен"
    elif status == "T":
        return "✅ Торги активны"
    elif status == "S":
        return "🚨 Торги приостановлены"
    elif status == "N":
        return "🚨 Торги недоступны"
    elif status == "C":
        return "🔒 Торги закрыты"
    else:
        return f"🟠 {status}"
def format_percentage(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value > 0:
        return f"+{value:.2f}%"
    elif value < 0:
        return f"{value:.2f}%"
    else:
        return "0.00%"
def format_price(value: Optional[float], currency: str = "RUB", security_type: str = None, face_value: Optional[float] = None) -> str:
    if value is None or value == "None" or str(value).lower() == "none":
        return "—"
    try:
        float_value = float(value)
        if currency is None or currency == "None" or str(currency).lower() == "none":
            currency = "RUB"
        if security_type == "bond" and 50 <= float_value <= 200:
            if face_value and face_value > 0:
                price_in_rubles = (float_value * face_value) / 100
                return f"{float_value:.2f}% ({price_in_rubles:.2f} ₽)"
            else:
                return f"{float_value:.2f}%"
        if currency == "RUB":
            return f"{float_value:.2f} ₽"
        elif currency == "USD":
            return f"${float_value:.2f}"
        elif currency == "EUR":
            return f"€{float_value:.2f}"
        else:
            return f"{float_value:.2f} {currency}"
    except (ValueError, TypeError):
        return "—"
def format_date(value: Optional[datetime]) -> str:
    if value is None:
        return "—"
    return value.strftime("%d.%m.%Y")
def get_risk_level(snapshot: MarketSnapshot) -> str:
    if snapshot.security_type == "bond":
        return get_bond_risk_level(snapshot)
    elif snapshot.security_type == "share":
        return get_share_risk_level(snapshot)
    elif snapshot.security_type == "fund":
        return "🟢 низкий (фонд)"
    else:
        return "—"
def get_bond_risk_level(snapshot: MarketSnapshot) -> str:
    if snapshot.trading_status:
        if "NOT_AVAILABLE" in snapshot.trading_status or "SUSPENDED" in snapshot.trading_status:
            return "🚨 критический (торги ограничены)"
        elif snapshot.trading_status not in ["T", "SECURITY_TRADING_STATUS_NORMAL_TRADING"]:
            return "🟠 мониторинг (торги ограничены)"
    if snapshot.ytm is None:
        return "🟡 умеренный (нет YTM)"
    if snapshot.ytm > 20:
        return "🔴 высокий"
    elif snapshot.ytm >= 12:
        return "🟠 мониторинг"
    else:
        return "🟢 низкий"
def get_share_risk_level(snapshot: MarketSnapshot) -> str:
    if snapshot.change_day_pct is None:
        return "🟡 умеренный (нет данных)"
    if snapshot.change_day_pct > 5:
        return "🟡 умеренный (событие/волатильность)"
    elif snapshot.change_day_pct < -5:
        return "🟠 мониторинг"
    else:
        return "🟢 низкий"
def get_trend(snapshot: MarketSnapshot) -> str:
    if snapshot.change_day_pct is None:
        return "—"
    if snapshot.change_day_pct > 0:
        return "Рост"
    elif snapshot.change_day_pct < 0:
        return "Падение"
    else:
        return "Без изменений"
def get_sector(snapshot: MarketSnapshot) -> str:
    if snapshot.sector:
        return snapshot.sector
    if snapshot.security_type == "bond":
        name = snapshot.name or ""
        return _get_sector_from_keywords(name)
    return "—"
async def get_sector_async(snapshot: MarketSnapshot) -> str:
    if snapshot.sector:
        return snapshot.sector
    if snapshot.security_type == "bond":
        name = snapshot.name or ""
        if name:
            from bot.utils.sector_detector import bond_sector_detector
            cached_sector = bond_sector_detector.get_sector_from_cache(
                name, snapshot.ticker or "", snapshot.isin or ""
            )
            if cached_sector:
                return cached_sector
            try:
                sector = await bond_sector_detector.detect_sector(
                    name, snapshot.ticker or "", snapshot.isin or ""
                )
                return sector or _get_sector_from_keywords(name)
            except Exception as e:
                print(f"AI sector detection failed for '{name}': {e}")
                return _get_sector_from_keywords(name)
        return "Определяется"
    return "—"
def _get_sector_from_keywords(name: str) -> str:
    name_lower = name.lower()
    if any(word in name_lower for word in ["банк", "bank"]):
        return "Банки"
    elif any(word in name_lower for word in ["лизинг", "leasing"]):
        return "Лизинг"
    elif any(word in name_lower for word in ["нефть", "газ", "oil", "газпром", "лукойл", "роснефть"]):
        return "Энергетика"
    elif any(word in name_lower for word in ["транспорт", "авто", "железнодорож", "авиа"]):
        return "Транспорт"
    elif any(word in name_lower for word in ["связь", "телеком", "мтс", "мегафон", "билайн"]):
        return "Телекоммуникации"
    elif any(word in name_lower for word in ["строитель", "недвижимость", "девелоп"]):
        return "Строительство"
    elif any(word in name_lower for word in ["металл", "сталь", "алюмин", "медь", "никель"]):
        return "Металлургия"
    elif any(word in name_lower for word in ["химия", "фарма", "медицин"]):
        return "Химия"
    elif any(word in name_lower for word in ["сельск", "агро", "зерно", "мясо", "молоч"]):
        return "Сельское хозяйство"
    elif any(word in name_lower for word in ["торгов", "ритейл", "магазин", "сеть"]):
        return "Розничная торговля"
    elif any(word in name_lower for word in ["пищев", "продукт", "еда", "напитк"]):
        return "Пищевая промышленность"
    elif any(word in name_lower for word in ["машин", "автомоб", "оборудован", "станок"]):
        return "Машиностроение"
    elif any(word in name_lower for word in ["мфк", "финанс", "кредит", "инвест"]):
        return "Финансы"
    elif any(word in name_lower for word in ["государств", "муниципаль", "облигац"]):
        return "Государственные"
    else:
        return "Другое"
async def render_signals_as_cards(snapshots: List[MarketSnapshot], has_shares: bool = False) -> str:
    if not snapshots:
        return "📊 **Таблица сигналов**\n\n❌ Нет данных для отображения"
    filtered_snapshots = []
    for snapshot in snapshots:
        if snapshot.security_type == "bond":
            filtered_snapshots.append(snapshot)
        elif snapshot.security_type == "share" and has_shares:
            filtered_snapshots.append(snapshot)
        elif snapshot.security_type not in ["share", "bond"]:
            filtered_snapshots.append(snapshot)
    if not filtered_snapshots:
        return "📊 **Таблица сигналов**\n\n❌ Нет данных для отображения"
    header = "📊 **ТАБЛИЦА СИГНАЛОВ**\n"
    header += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    cards = []
    for i, snapshot in enumerate(filtered_snapshots, 1):
        card = await render_single_signal_card(snapshot)
        cards.append(card)
    cards_text = "\n\n".join(cards)
    return header + cards_text
async def render_single_signal_card(snapshot: MarketSnapshot) -> str:
    name = snapshot.name or snapshot.ticker or snapshot.secid
    ticker = snapshot.ticker or "—"
    isin = snapshot.isin or "—"
    def escape_markdown(text: str) -> str:
        if not text:
            return text
        return text.replace('*', '\\*').replace('_', '\\_').replace('`', '\\`').replace('[', '\\[').replace(']', '\\]')
    name = escape_markdown(name)
    ticker = escape_markdown(ticker)
    isin = escape_markdown(isin)
    if snapshot.security_type == "share":
        header_emoji = "📈"
        type_name = "Акция"
    elif snapshot.security_type == "bond":
        header_emoji = "🏛️"
        type_name = "Облигация"
    elif snapshot.security_type == "fund":
        header_emoji = "📊"
        type_name = "Фонд"
    else:
        header_emoji = "📋"
        type_name = "Инструмент"
    card = f"{header_emoji} **{name}**\n"
    card += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    card += f"🏷️ **{ticker}**"
    if isin != "—" and isin != ticker:
        card += f" • {isin}"
    card += f"\n📋 {type_name}\n"
    price = format_price(snapshot.last_price, snapshot.currency, snapshot.security_type, snapshot.face_value)
    change = format_percentage(snapshot.change_day_pct)
    if snapshot.change_day_pct is not None:
        if snapshot.change_day_pct > 0:
            change_emoji = "📈"
        elif snapshot.change_day_pct < 0:
            change_emoji = "📉"
        else:
            change_emoji = "➡️"
    else:
        change_emoji = "❓"
    card += f"💰 {price} {change_emoji} {change}\n"
    risk = get_risk_level(snapshot)
    trend = get_trend(snapshot)
    card += f"⚠️ {risk}\n"
    card += f"📊 Тренд: {trend}\n"
    sector = await get_sector_async(snapshot)
    if sector != "—":
        card += f"🏢 {sector}\n"
    if snapshot.trading_status:
        status_display = format_trading_status(snapshot.trading_status)
        card += f"{status_display}\n"
    if snapshot.security_type == "bond":
        if snapshot.ytm is not None:
            card += f"📊 YTM: {snapshot.ytm:.2f}%\n"
        if snapshot.next_coupon_date and snapshot.coupon_value is not None:
            coupon_date = format_date(snapshot.next_coupon_date)
            coupon_value = format_price(snapshot.coupon_value, snapshot.currency, snapshot.security_type, snapshot.face_value)
            card += f"📅 Купон: {coupon_date} • {coupon_value}\n"
        elif snapshot.next_coupon_date:
            coupon_date = format_date(snapshot.next_coupon_date)
            card += f"📅 Купон: {coupon_date}\n"
        if snapshot.maturity_date:
            maturity_date = format_date(snapshot.maturity_date)
            card += f"🏁 Погашение: {maturity_date}\n"
    elif snapshot.security_type == "share":
        if snapshot.next_dividend_date and snapshot.dividend_value is not None:
            div_date = format_date(snapshot.next_dividend_date)
            div_value = format_price(snapshot.dividend_value, snapshot.currency, snapshot.security_type, snapshot.face_value)
            card += f"💰 Дивиденд: {div_date} • {div_value}\n"
        elif snapshot.next_dividend_date:
            div_date = format_date(snapshot.next_dividend_date)
            card += f"💰 Дивиденд: {div_date}\n"
    card += f"ℹ️ История дефолтов: не зафиксировано"
    return card
def render_calendar_30d(calendar_data: List[Dict[str, Any]]) -> str:
    if not calendar_data:
        return "📅 •КАЛЕНДАРЬ ВЫПЛАТ (7 ДНЕЙ)•\n\n❌ Нет предстоящих выплат в ближайшие 7 дней"
    current_date = datetime.now()
    end_date = current_date + timedelta(days=7)
    logger.info(f"Rendering calendar with {len(calendar_data)} events")
    logger.info(f"Current date: {current_date}")
    logger.info(f"End date (30 days): {end_date}")
    future_events = []
    for event in calendar_data:
        event_date = event.get("date")
        logger.info(f"Event: {event.get('secid', 'unknown')} - date: {event_date}, type: {event.get('type', 'unknown')}")
        if event_date:
            is_within_range = current_date <= event_date <= end_date
            logger.info(f"  Date check: {current_date} <= {event_date} <= {end_date} = {is_within_range}")
            if is_within_range:
                future_events.append(event)
                logger.info(f"  -> Added to future events (within 30 days)")
            else:
                logger.info(f"  -> Skipped (outside 30-day range)")
        else:
            logger.info(f"  -> Skipped (no date)")
    logger.info(f"Found {len(future_events)} future events within 30 days")
    if not future_events:
        return "📅 •КАЛЕНДАРЬ ВЫПЛАТ (30 ДНЕЙ)•\n\n❌ Нет предстоящих выплат в ближайшие 30 дней"
    calendar_text = "📅 •КАЛЕНДАРЬ ВЫПЛАТ (30 ДНЕЙ)•\n"
    calendar_text += "═══════════════════════════════════\n\n"
    sorted_events = sorted(future_events, key=lambda x: x["date"])
    for i, event in enumerate(sorted_events, 1):
        date_str = event["date"].strftime("%d.%m.%Y")
        event_type = "Купон" if event["type"] == "coupon" else "Амортизация"
        value = format_price(event["value"], "RUB", "bond", None)
        secid = event.get("secid", "")
        moex_url = f"https://www.moex.com/ru/issue.aspx?board=TQCB&code={secid}" if secid else ""
        event_emoji = "💰" if event["type"] == "coupon" else "🏦"
        calendar_text += f"📅 •{date_str}•\n"
        calendar_text += f"┌─────────────────────────────────\n"
        calendar_text += f"│ {event_emoji} {event_type}: {value}\n"
        calendar_text += f"│ 🏷️ Выпуск: {event.get('name', '—')}\n"
        if moex_url:
            calendar_text += f"│ 🔗 MOEX: {moex_url}\n"
        calendar_text += f"└─────────────────────────────────\n"
        if i < len(sorted_events):
            calendar_text += "\n"
    return calendar_text
def render_payment_history(payment_history: Dict[str, Any]) -> str:
    if not payment_history or not payment_history.get("history"):
        return "📊 •ИСТОРИЯ ВЫПЛАТ•\n\n❌ Нет данных по истории выплат"
    history_data = payment_history["history"]
    risk_signals = payment_history.get("risk_signals", [])
    history_text = "📊 •ИСТОРИЯ ВЫПЛАТ•\n"
    history_text += "═══════════════════════════════════\n\n"
    total_bonds = payment_history.get("total_bonds_analyzed", 0)
    bonds_with_history = payment_history.get("bonds_with_history", 0)
    history_text += f"📈 **Статистика анализа:**\n"
    history_text += f"┌─────────────────────────────────\n"
    history_text += f"│ 📊 Всего облигаций: {total_bonds}\n"
    history_text += f"│ 📋 С историей выплат: {bonds_with_history}\n"
    history_text += f"└─────────────────────────────────\n\n"
    if risk_signals:
        history_text += f"⚠️ **Сигналы риска:**\n"
        for signal in risk_signals[:5]:
            history_text += f"• {signal}\n"
        history_text += "\n"
    for secid, history in list(history_data.items())[:3]:
        history_text += f"🏷️ **{secid}**\n"
        history_text += f"┌─────────────────────────────────\n"
        if history.total_events == 0:
            history_text += f"│ 📊 Событий: Нет данных за период\n"
            history_text += f"│ ℹ️ Статус: Новая облигация\n"
        else:
            history_text += f"│ 📊 Событий: {history.total_events}\n"
            history_text += f"│ ✅ Выплачено: {history.paid_events}\n"
            history_text += f"│ ❌ Отменено: {history.cancelled_events}\n"
            history_text += f"│ ⚠️ Задержек: {history.delayed_events}\n"
        history_text += f"│ 📈 Надежность: {history.reliability_score:.1f}%\n"
        history_text += f"└─────────────────────────────────\n\n"
    return history_text
def render_news_summary(news_items: List[Dict[str, Any]], holdings: List[Any] = None) -> str:
    if not news_items:
        return "📰 **НОВОСТНОЙ ФОН**\n\n❌ Нет новостей по эмитентам"
    news_text = "📰 **НОВОСТНОЙ ФОН**\n"
    news_text += "═══════════════════════════════════\n\n"
    for i, item in enumerate(news_items[:5], 1):
        title = item.get("title", "")
        link = item.get("link", "")
        source = item.get("source", "")
        published_at = item.get("published_at")
        related_tickers = item.get("related_tickers", [])
        related_issuers = item.get("related_issuers", [])
        matched_terms = item.get("matched_terms", [])
        if published_at:
            date_str = published_at.strftime("%d.%m.%Y %H:%M")
        else:
            date_str = "—"
        if "rbc" in source.lower():
            source_emoji = "🔴"
        elif "smart-lab" in source.lower():
            source_emoji = "🧠"
        else:
            source_emoji = "📰"
        news_text += f"📰 **{title}**\n"
        news_text += f"┌─────────────────────────────────\n"
        news_text += f"│ {source_emoji} Источник: {source}\n"
        news_text += f"│ 📅 Дата: {date_str}\n"
        affected_positions = []
        if holdings:
            for ticker in related_tickers:
                for holding in holdings:
                    if holding.ticker == ticker:
                        affected_positions.append(f"{holding.normalized_name} ({holding.ticker})")
            for issuer in related_issuers:
                for holding in holdings:
                    if holding.normalized_name == issuer:
                        affected_positions.append(f"{holding.normalized_name} ({holding.ticker})")
            for term in matched_terms:
                if term in ['СБЕРБАНК', 'БАНК', 'БАНКОВСКИЙ']:
                    for holding in holdings:
                        if holding.ticker == 'SBER' or 'БАНК' in holding.normalized_name.upper():
                            affected_positions.append(f"{holding.normalized_name} ({holding.ticker})")
                elif term in ['ГАЗПРОМ', 'ГАЗ', 'ГАЗОВЫЙ']:
                    for holding in holdings:
                        if holding.ticker == 'GAZP' or 'ГАЗ' in holding.normalized_name.upper():
                            affected_positions.append(f"{holding.normalized_name} ({holding.ticker})")
                elif term in ['ЛУКОЙЛ', 'НЕФТЬ', 'НЕФТЯНОЙ']:
                    for holding in holdings:
                        if holding.ticker == 'LKOH' or 'НЕФТЬ' in holding.normalized_name.upper():
                            affected_positions.append(f"{holding.normalized_name} ({holding.ticker})")
        affected_positions = list(set(affected_positions))
        if affected_positions:
            news_text += f"│ 📈 Затрагивает: {', '.join(affected_positions)}\n"
        else:
            news_text += f"│ 📈 Затрагивает: общие рыночные новости\n"
        if link:
            news_text += f"│ 🔗 [Читать]({link})\n"
        news_text += f"└─────────────────────────────────\n"
        if i < min(len(news_items), 5):
            news_text += "\n"
    return news_text
def render_portfolio_summary(
    snapshots: List[MarketSnapshot],
    ocr_meta: Optional[Dict[str, Any]] = None,
    accounts: Optional[List[Any]] = None,
    cash_by_account: Optional[Dict[int, List[Any]]] = None
) -> str:
    summary = "📊 **ОБЩАЯ ОЦЕНКА ПОРТФЕЛЯ**\n"
    summary += "═══════════════════════════════════\n\n"
    portfolio_name = ocr_meta.get("portfolio_name") if ocr_meta else None
    portfolio_value = ocr_meta.get("portfolio_value") if ocr_meta else None
    portfolio_currency = ocr_meta.get("currency") if ocr_meta else "RUB"
    positions_count = ocr_meta.get("positions_count") if ocr_meta else None
    if portfolio_name:
        summary += f"📁 **Название:** {portfolio_name}\n"
    if portfolio_value is not None:
        if portfolio_currency == "RUB":
            value_str = f"{portfolio_value:,.2f} ₽"
        elif portfolio_currency == "USD":
            value_str = f"${portfolio_value:,.2f}"
        elif portfolio_currency == "EUR":
            value_str = f"€{portfolio_value:,.2f}"
        else:
            value_str = f"{portfolio_value:,.2f} {portfolio_currency}"
        summary += f"💰 **Общая стоимость:** {value_str}\n"
    if positions_count is not None:
        summary += f"📈 **Количество позиций:** {positions_count}\n"
    total_value = sum(snapshot.last_price for snapshot in snapshots if snapshot.last_price)
    change_values = [snapshot.change_day_pct for snapshot in snapshots if snapshot.change_day_pct is not None]
    if total_value and portfolio_value is None:
        summary += f"💰 **Оценка совокупной стоимости:** {total_value:.2f} ₽\n"
    if change_values:
        avg_change = sum(change_values) / len(change_values)
        summary += f"📉 **Среднее изменение за день:** {avg_change:.2f}%\n"
    summary += "\n📊 **Распределение по типам активов:**\n"
    summary += "┌─────────────────────────────────\n"
    distribution = {"Акции": 0, "Облигации": 0, "Фонды": 0, "Прочее": 0}
    for snapshot in snapshots:
        if snapshot.security_type == "share":
            distribution["Акции"] += 1
        elif snapshot.security_type == "bond":
            distribution["Облигации"] += 1
        elif snapshot.security_type == "fund":
            distribution["Фонды"] += 1
        else:
            distribution["Прочее"] += 1
    total_count = len(snapshots) if snapshots else 1
    for asset_type, count in distribution.items():
        percentage = (count / total_count) * 100
        summary += f"│ {asset_type}: {count} ({percentage:.0f}%)\n"
    summary += "└─────────────────────────────────\n"
    if accounts:
        summary += "\n🏦 **Счета портфеля:**\n"
        summary += "┌─────────────────────────────────\n"
        for account in accounts:
            account_name = getattr(account, "account_name", None) or getattr(account, "account_id", None) or "Неизвестный счёт"
            currency = getattr(account, "currency", None) or "RUB"
            value = getattr(account, "portfolio_value", None)
            if value is not None:
                if currency == "RUB":
                    value_str = f"{value:,.2f} ₽"
                elif currency == "USD":
                    value_str = f"${value:,.2f}"
                elif currency == "EUR":
                    value_str = f"€{value:,.2f}"
                else:
                    value_str = f"{value:,.2f} {currency}"
            else:
                value_str = "—"
            summary += f"│ {account_name}: {value_str}\n"
            daily_change_value = getattr(account, "daily_change_value", None)
            daily_change_percent = getattr(account, "daily_change_percent", None)
            if daily_change_value is not None or daily_change_percent is not None:
                change_parts = []
                if daily_change_value is not None:
                    change_value_str = format_price(daily_change_value, currency)
                    change_parts.append(change_value_str)
                if daily_change_percent is not None:
                    perc = f"{daily_change_percent:+.2f}%"
                    change_parts.append(perc)
                summary += f"│   Δ {', '.join(change_parts)}\n"
            cash_entries = (cash_by_account or {}).get(getattr(account, "id", None), [])
            if cash_entries:
                for cash in cash_entries:
                    cash_value = cash.amount or 0
                    cash_currency = cash.currency or currency
                    if cash_currency == "RUB":
                        cash_str = f"{cash_value:,.2f} ₽"
                    elif cash_currency == "USD":
                        cash_str = f"${cash_value:,.2f}"
                    elif cash_currency == "EUR":
                        cash_str = f"€{cash_value:,.2f}"
                    else:
                        cash_str = f"{cash_value:,.2f} {cash_currency}"
                    summary += f"│   💵 {cash.raw_name or 'Cash'}: {cash_str}\n"
        summary += "└─────────────────────────────────\n"
    elif cash_by_account and cash_by_account.get(None):
        summary += "\n💵 **Денежные средства:**\n"
        summary += "┌─────────────────────────────────\n"
        for cash in cash_by_account.get(None, []):
            cash_currency = cash.currency or "RUB"
            if cash_currency == "RUB":
                cash_str = f"{cash.amount:,.2f} ₽"
            elif cash_currency == "USD":
                cash_str = f"${cash.amount:,.2f}"
            elif cash_currency == "EUR":
                cash_str = f"€{cash.amount:,.2f}"
            else:
                cash_str = f"{cash.amount:,.2f} {cash_currency}"
            summary += f"│ {cash.raw_name or 'Cash'}: {cash_str}\n"
        summary += "└─────────────────────────────────\n"
    if ocr_meta and ocr_meta.get("warnings"):
        summary += "⚠️ **Предупреждения OCR:**\n"
        for warn in ocr_meta["warnings"]:
            summary += f"- {warn}\n"
        summary += "\n"
    return summary
def render_recommendations(snapshots: List[MarketSnapshot]) -> str:
    recommendations = []
    recommendations.append("💡 **РЕКОМЕНДАЦИИ**")
    recommendations.append("═══════════════════════════════════")
    recommendations.append("")
    high_risk = [s for s in snapshots if "🔴" in get_risk_level(s) or "🚨" in get_risk_level(s)]
    monitor_risk = [s for s in snapshots if "🟠" in get_risk_level(s)]
    if high_risk:
        recommendations.append("🚨 **Критические позиции требуют внимания:**")
        recommendations.append("┌─────────────────────────────────")
        for snapshot in high_risk:
            name = snapshot.name or snapshot.ticker or snapshot.secid
            risk = get_risk_level(snapshot)
            recommendations.append(f"│ ⚠️ {name} - {risk}")
        recommendations.append("└─────────────────────────────────")
        recommendations.append("")
    declining = [s for s in snapshots if get_trend(s) == "Падение"]
    if declining:
        recommendations.append("📉 **Падающие позиции:**")
        recommendations.append("┌─────────────────────────────────")
        for snapshot in declining:
            name = snapshot.name or snapshot.ticker or snapshot.secid
            change = format_percentage(snapshot.change_day_pct)
            recommendations.append(f"│ 📉 {name} ({change})")
        recommendations.append("└─────────────────────────────────")
        recommendations.append("")
    rising = [s for s in snapshots if get_trend(s) == "Рост"]
    if rising:
        recommendations.append("📈 **Растущие позиции:**")
        recommendations.append("┌─────────────────────────────────")
        for snapshot in rising:
            name = snapshot.name or snapshot.ticker or snapshot.secid
            change = format_percentage(snapshot.change_day_pct)
            recommendations.append(f"│ 📈 {name} ({change})")
        recommendations.append("└─────────────────────────────────")
        recommendations.append("")
    if not high_risk and not monitor_risk and not declining:
        recommendations.append("✅ **Портфель в стабильном состоянии**")
        recommendations.append("┌─────────────────────────────────")
        recommendations.append("│ 🎯 Все позиции в норме")
        recommendations.append("│ 📊 Риски под контролем")
        recommendations.append("└─────────────────────────────────")
    return "\n".join(recommendations)