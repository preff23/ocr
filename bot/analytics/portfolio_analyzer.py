import asyncio
import json
import openai
from typing import List, Dict, Any, Optional
from datetime import datetime
from bot.core.config import config
from bot.utils.integrated_bond_client import IntegratedBondClient
from bot.analytics.payment_history import PaymentHistoryAnalyzer
from bot.utils.render import (
    render_signals_as_cards, render_calendar_30d, render_news_summary,
    render_portfolio_summary, render_recommendations, render_payment_history
)
from bot.core.logging import get_logger
logger = get_logger(__name__)
class IntegratedSnapshot:
    def __init__(self, data: dict):
        self.isin = data.get('isin')
        self.name = data.get('name')
        self.issuer_name = data.get('issuer_name')
        self.price = data.get('price')
        self.yield_to_maturity = data.get('yield_to_maturity')
        self.duration = data.get('duration')
        self.face_value = data.get('face_value')
        self.sector = data.get('sector')
        self.security_type = data.get('security_type')
        self.confidence = data.get('confidence')
        self.corpbonds_found = data.get('corpbonds_found')
        self.tbank_found = data.get('tbank_found')
        self.moex_found = data.get('moex_found')
        self.holding_id = data.get('holding_id')
        self.raw_name = data.get('raw_name')
        self.raw_quantity = data.get('raw_quantity')
        self.ticker = self.isin
        self.last_price = self.price
        self.change_day_pct = 0.0
        self.volume = 0.0
        self.currency = 'RUB'
        self.secid = self.isin
        self.trading_status = 'NormalTrading'
        self.shortname = self.name
        self.board = 'TQCB'
        self.aci = 0.0
        self.ytm = self.yield_to_maturity
        self.next_coupon_date = None
        self.provider = 'integrated'
        self.last_update = datetime.now()
        self.maturity_date = None
        self.coupon_rate = 0.0
        self.coupon_frequency = 0
        self.issue_date = None
        self.issue_size = 0.0
        self.rating = None
        self.rating_agency = None
class PortfolioAnalyzer:
    def __init__(self):
        self.db_manager = db_manager
        self.market_aggregator = market_aggregator
        self.payment_history_analyzer = PaymentHistoryAnalyzer()
        self.openai_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key,
            timeout=300.0,
            max_retries=3
        )
    async def run_analysis(self, user_id: int) -> Dict[str, Any]:
        """Run comprehensive portfolio analysis.
        Args:
            user_id: User ID
        Returns:
            Analysis results with all components
        """
        try:
            holdings = await self._load_user_holdings(user_id)
            if not holdings:
                return {
                    "error": "Портфель пуст",
                    "summary": "Добавьте позиции для анализа"
                }
            accounts = await self._load_user_accounts(user_id)
            cash_by_account = await self._load_account_cash(user_id, accounts)
            bond_holdings = [h for h in holdings if h.security_type == "bond" or (h.isin and h.isin.startswith('RU'))]
            share_holdings = [h for h in holdings if h.security_type == "share"]
            logger.info("Starting parallel data loading...")
            tasks = []
            tasks.append(self._get_integrated_bond_data(bond_holdings))
            if share_holdings:
                tasks.append(self._get_market_data(share_holdings))
            else:
                tasks.append(asyncio.create_task(asyncio.sleep(0)))
            logger.info(f"Starting {len(tasks)} parallel tasks...")
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=120.0
                )
                integrated_bond_data = results[0]
                share_snapshots = results[1] if share_holdings else []
                logger.info(f"Parallel data loading completed: {len(integrated_bond_data)} bonds, {len(share_snapshots)} shares")
            except asyncio.TimeoutError:
                logger.error("Parallel data loading timed out after 120 seconds")
                return {
                    "error": "Таймаут загрузки данных",
                    "summary": "Анализ занял слишком много времени. Попробуйте позже."
                }
            all_snapshots = integrated_bond_data + share_snapshots
            logger.info(f"Created {len(all_snapshots)} snapshots: {len(integrated_bond_data)} integrated bonds, {len(share_snapshots)} shares")
            logger.info(f"Total holdings loaded: {len(holdings)}")
            for i, holding in enumerate(holdings):
                logger.info(f"Holding {i}: id={holding.id}, isin={holding.isin}, ticker={holding.ticker}, type={holding.security_type}, name={holding.raw_name}")
            bonds_with_data = {snapshot.holding_id for snapshot in integrated_bond_data if hasattr(snapshot, 'holding_id')}
            bonds_without_data = [h for h in bond_holdings if h.id not in bonds_with_data]
            logger.info(f"Bonds with data: {len(bonds_with_data)}, bonds without data: {len(bonds_without_data)}")
            logger.info(f"Bond holdings IDs with data: {bonds_with_data}")
            logger.info(f"Bond holdings IDs without data: {[h.id for h in bonds_without_data]}")
            if bonds_without_data:
                basic_bond_snapshots = await self._create_basic_snapshots(bonds_without_data)
                all_snapshots.extend(basic_bond_snapshots)
                logger.info(f"Added {len(basic_bond_snapshots)} basic bond snapshots")
            if not all_snapshots:
                all_snapshots = await self._create_basic_snapshots(holdings)
                logger.info(f"Created {len(all_snapshots)} basic snapshots from all holdings")
            logger.info(f"Total snapshots before calendar: {len(all_snapshots)}")
            for i, snapshot in enumerate(all_snapshots):
                logger.info(f"Snapshot {i}: {getattr(snapshot, 'name', 'unknown')} - security_type: {getattr(snapshot, 'security_type', 'None')}")
            bond_snapshots = [s for s in all_snapshots if getattr(s, 'security_type', None) == "bond"]
            logger.info(f"Bond snapshots for calendar: {len(bond_snapshots)}")
            logger.info("Starting parallel loading of calendar, news, and payment history...")
            calendar_task = self._get_bond_calendar(bond_snapshots)
            news_task = self._get_news(all_snapshots)
            payment_task = self._get_payment_history(all_snapshots)
            bond_calendar, news_items, payment_history = await asyncio.gather(
                calendar_task, news_task, payment_task
            )
            has_shares = len(share_holdings) > 0
            try:
                ocr_meta = self.db_manager.get_portfolio_meta(user_id)
            except Exception:
                ocr_meta = {}
            analysis = await self._generate_analysis(
                all_snapshots,
                bond_calendar,
                news_items,
                payment_history,
                has_shares,
                holdings,
                accounts,
                cash_by_account,
                ocr_meta
            )
            if config.feature_analysis_v14:
                ai_analysis = await self._generate_ai_analysis(
                    all_snapshots,
                    bond_calendar,
                    news_items,
                    ocr_meta,
                    payment_history,
                    accounts,
                    cash_by_account
                )
                analysis["ai_analysis"] = ai_analysis
            return analysis
        except Exception as e:
            logger.error(f"Portfolio analysis failed for user {user_id}: {e}")
            return {
                "error": "Ошибка анализа",
                "summary": f"Не удалось выполнить анализ: {str(e)}"
            }
    async def _load_user_holdings(self, user_id: int) -> List[PortfolioHoldingV2]:
        try:
            return self.db_manager.get_user_holdings(user_id, active_only=True)
        except Exception as e:
            logger.error(f"Failed to load holdings for user {user_id}: {e}")
            return []
    async def _load_user_accounts(self, user_id: int) -> List[PortfolioAccount]:
        try:
            return self.db_manager.get_user_accounts(user_id)
        except Exception as e:
            logger.error(f"Failed to load accounts for user {user_id}: {e}")
            return []
    async def _load_account_cash(self, user_id: int, accounts: List[PortfolioAccount]) -> Dict[int, List[PortfolioCashPosition]]:
        cash_map: Dict[int, List[PortfolioCashPosition]] = {}
        try:
            for account in accounts:
                cash_map[account.id] = self.db_manager.get_account_cash(user_id, account.id)
            cash_map[None] = self.db_manager.get_account_cash(user_id, None)
        except Exception as e:
            logger.error(f"Failed to load cash positions for user {user_id}: {e}")
        return cash_map
    async def _get_payment_history(self, snapshots: List[Any]) -> Dict[str, Any]:
        try:
            bond_snapshots = [
                snapshot for snapshot in snapshots 
                if snapshot.security_type == "bond" and snapshot.secid
            ]
            if not bond_snapshots:
                return {}
            logger.info(f"Getting payment history for {len(bond_snapshots)} bonds")
            logger.info(f"Getting payment history for {len(bond_snapshots)} bonds in parallel")
            history_tasks = [
                asyncio.wait_for(
                    self.payment_history_analyzer.get_payment_history(snapshot.secid, months_back=12),
                    timeout=90.0
                )
                for snapshot in bond_snapshots
            ]
            history_results = await asyncio.gather(*history_tasks, return_exceptions=True)
            payment_history = {}
            risk_signals = []
            for i, result in enumerate(history_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get payment history for {bond_snapshots[i].secid}: {result}")
                    continue
                if result:
                    secid = bond_snapshots[i].secid
                    payment_history[secid] = result
                    signals = self.payment_history_analyzer.get_risk_signals(result)
                    if signals:
                        risk_signals.extend([f"{secid}: {signal}" for signal in signals])
            return {
                "history": payment_history,
                "risk_signals": risk_signals,
                "total_bonds_analyzed": len(bond_snapshots),
                "bonds_with_history": len(payment_history)
            }
        except Exception as e:
            logger.error(f"Failed to get payment history: {e}")
            return {}
    async def _create_basic_snapshots(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        snapshots = []
        for holding in holdings:
            security_type = holding.security_type or "bond"
            if security_type == "fund" and holding.isin and holding.isin.startswith('RU'):
                security_type = "bond"
            elif security_type == "fund" and not holding.isin and holding.ticker and holding.ticker.startswith('RU') and len(holding.ticker) == 12:
                security_type = "bond"
            elif holding.security_type == "bond":
                security_type = "bond"
            snapshot = MarketSnapshot(
                name=holding.normalized_name,
                ticker=holding.ticker,
                isin=holding.isin,
                secid=holding.secid,
                security_type=security_type,
                last_price=None,
                change_day_pct=None,
                currency="RUB",
                trading_status="T",
                provider="UNKNOWN",
                sector=None,
                ytm=None,
                next_coupon_date=None,
                coupon_value=None,
                maturity_date=None,
                next_dividend_date=None,
                dividend_value=None
            )
            snapshots.append(snapshot)
        return snapshots
    async def _get_market_data(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        try:
            queries = []
            for holding in holdings:
                query = holding.ticker or holding.isin or holding.raw_name or holding.normalized_name
                if query:
                    queries.append(query)
            if not queries:
                return []
            snapshots = await self.market_aggregator.get_snapshot_for(queries)
            enriched_snapshots = []
            for i, snapshot in enumerate(snapshots):
                if i < len(holdings):
                    holding = holdings[i]
                    snapshot.holding_id = holding.id
                    snapshot.raw_name = holding.raw_name
                    snapshot.raw_quantity = holding.raw_quantity
                    enriched_snapshots.append(snapshot)
            return enriched_snapshots
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    async def _get_integrated_bond_data(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        try:
            bond_holdings = []
            for h in holdings:
                logger.info(f"Processing holding: id={h.id}, isin={h.isin}, ticker={h.ticker}, type={h.security_type}")
                if h.isin and h.isin.startswith('RU'):
                    bond_holdings.append(h)
                    logger.info(f"Added to bond_holdings (RU isin): {h.id}")
                elif not h.isin and h.ticker and h.ticker.startswith('RU') and len(h.ticker) == 12:
                    bond_holdings.append(h)
                    logger.info(f"Added to bond_holdings (RU ticker): {h.id}")
                elif h.security_type == "bond":
                    bond_holdings.append(h)
                    logger.info(f"Added to bond_holdings (bond type): {h.id}")
                else:
                    logger.info(f"Skipped holding: {h.id} - not a bond")
            if not bond_holdings:
                logger.info("No bond holdings found for integrated analysis")
                return []
            from bot.utils.integrated_bond_client import IntegratedBondClient
            client = IntegratedBondClient()
            isins = [h.isin or h.ticker for h in bond_holdings if h.isin or h.ticker]
            logger.info(f"Getting integrated bond data for {len(isins)} bonds: {isins}")
            integrated_data = []
            for holding in bond_holdings:
                try:
                    isin = holding.isin or holding.ticker
                    if not isin:
                        continue
                    snapshot = await client.get_bond_data(isin)
                    if snapshot:
                        snapshot.holding_id = holding.id
                        snapshot.raw_name = holding.raw_name
                        snapshot.raw_quantity = holding.raw_quantity
                        integrated_data.append(snapshot)
                    else:
                        snapshot_data = {
                            'isin': isin,
                            'name': holding.normalized_name or holding.raw_name,
                            'issuer_name': holding.normalized_name or holding.raw_name,
                            'price': 100.0,
                            'yield_to_maturity': 10.0,
                            'duration': 2.0,
                            'face_value': 1000.0,
                            'sector': "Unknown",
                            'security_type': 'bond',
                            'confidence': 'low',
                            'corpbonds_found': False,
                            'tbank_found': False,
                            'moex_found': False,
                            'holding_id': holding.id,
                            'raw_name': holding.raw_name,
                            'raw_quantity': holding.raw_quantity
                        }
                        snapshot = IntegratedSnapshot(snapshot_data)
                        integrated_data.append(snapshot)
                except Exception as e:
                    logger.error(f"Failed to get data for {holding.isin}: {e}")
                    snapshot_data = {
                        'isin': holding.isin or holding.ticker,
                        'name': holding.normalized_name or holding.raw_name,
                        'issuer_name': holding.normalized_name or holding.raw_name,
                        'price': 100.0,
                        'yield_to_maturity': 10.0,
                        'duration': 2.0,
                        'face_value': 1000.0,
                        'sector': "Unknown",
                        'security_type': 'bond',
                        'confidence': 'low',
                        'corpbonds_found': False,
                        'tbank_found': False,
                        'moex_found': False,
                        'holding_id': holding.id,
                        'raw_name': holding.raw_name,
                        'raw_quantity': holding.raw_quantity
                    }
                    snapshot = IntegratedSnapshot(snapshot_data)
                    integrated_data.append(snapshot)
            logger.info(f"Created {len(integrated_data)} integrated snapshots")
            return integrated_data
        except Exception as e:
            logger.error(f"Failed to get integrated bond data: {e}")
            return []
    async def _get_bond_calendar(self, snapshots: List[Any]) -> List[Dict[str, Any]]:
        try:
            bond_secids = [
                getattr(snapshot, 'secid', None) for snapshot in snapshots
                if getattr(snapshot, 'security_type', None) == "bond" and getattr(snapshot, 'secid', None)
            ]
            logger.info(f"Getting bond calendar for {len(bond_secids)} bonds: {bond_secids}")
            if not bond_secids:
                logger.info("No bond secids found, returning empty calendar")
                return []
            calendar_tasks = [
                self.market_aggregator.get_bond_calendar(secid)
                for secid in bond_secids
            ]
            logger.info(f"Created {len(calendar_tasks)} calendar tasks")
            calendar_results = await asyncio.gather(*calendar_tasks, return_exceptions=True)
            logger.info(f"Got {len(calendar_results)} calendar results")
            all_events = []
            for i, result in enumerate(calendar_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get calendar for {bond_secids[i]}: {result}")
                    continue
                if result:
                    bond_secid = bond_secids[i]
                    bond_snapshot = next(
                        (s for s in snapshots if s.secid == bond_secid), None
                    )
                    bond_name = getattr(bond_snapshot, 'name', None) or getattr(bond_snapshot, 'shortname', None) or bond_secid
                    logger.info(f"Processing calendar for {bond_secid}: {len(result.get('coupons', []))} coupons, {len(result.get('amortizations', []))} amortizations")
                    for coupon in result.get("coupons", []):
                        event = coupon.copy()
                        event["secid"] = bond_secid
                        event["name"] = bond_name
                        logger.info(f"  Coupon: {event.get('date')} - {event.get('value')} ₽")
                        all_events.append(event)
                    for amort in result.get("amortizations", []):
                        event = amort.copy()
                        event["secid"] = bond_secid
                        event["name"] = bond_name
                        logger.info(f"  Amortization: {event.get('date')} - {event.get('value')} ₽")
                        all_events.append(event)
                else:
                    logger.info(f"No calendar data for {bond_secids[i]}")
            logger.info(f"Total calendar events collected: {len(all_events)}")
            return all_events
        except Exception as e:
            logger.error(f"Failed to get bond calendar: {e}")
            return []
    async def _get_news(self, snapshots: List[Any]) -> List[Dict[str, Any]]:
        try:
            tickers = [s.ticker for s in snapshots if s.ticker]
            issuers = [s.name for s in snapshots if s.name]
            logger.info(f"Getting news for portfolio with {len(tickers)} tickers and {len(issuers)} issuers")
            logger.info(f"Tickers: {tickers}")
            logger.info(f"Issuers: {issuers}")
            if not tickers and not issuers:
                logger.warning("No tickers or issuers found for news search")
                return []
            news_items = await get_news_for_portfolio(tickers, issuers, max_age_hours=24)
            logger.info(f"Found {len(news_items)} relevant news items for current portfolio")
            news_dicts = []
            for item in news_items:
                news_dicts.append({
                    "title": item.title,
                    "link": item.link,
                    "description": item.description,
                    "source": item.source,
                    "published_at": item.published_at,
                    "related_tickers": item.related_tickers,
                    "related_issuers": item.related_issuers,
                    "matched_terms": getattr(item, 'matched_terms', [])
                })
            return news_dicts
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return []
    async def _generate_analysis(self,
                               snapshots: List[Any],
                               bond_calendar: List[Dict[str, Any]],
                               news_items: List[Dict[str, Any]],
                               payment_history: Dict[str, Any],
                               has_shares: bool = False,
                               holdings: List[Any] = None,
                               accounts: List[PortfolioAccount] = None,
                               cash_by_account: Dict[int, List[PortfolioCashPosition]] = None,
                               ocr_meta: Dict[str, Any] = None) -> Dict[str, Any]:
        try:
            summary = render_portfolio_summary(snapshots, ocr_meta, accounts, cash_by_account)
            signals_table = ""
            calendar_text = render_calendar_30d(bond_calendar)
            news_summary = render_news_summary(news_items, holdings)
            payment_history_summary = render_payment_history(payment_history)
            recommendations = render_recommendations(snapshots)
            metrics = self._calculate_metrics(snapshots)
            logger.info("Generating AI analysis...")
            ai_analysis = await self._generate_ai_analysis(
                snapshots, bond_calendar, news_items, ocr_meta, 
                payment_history, accounts, cash_by_account
            )
            logger.info(f"AI analysis generated: {len(ai_analysis)} characters")
            return {
                "summary": summary,
                "signals_table": signals_table,
                "signals": snapshots,
                "calendar_30d": calendar_text,
                "news_summary": news_summary,
                "news": news_items,
                "payment_history_summary": payment_history_summary,
                "recommendations": recommendations,
                "metrics": metrics,
                "payment_history": payment_history,
                "ai_analysis": ai_analysis,
                "generated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            return {
                "error": "Ошибка генерации анализа",
                "summary": f"Не удалось сгенерировать анализ: {str(e)}"
            }
    def _calculate_metrics(self, snapshots: List[Any]) -> Dict[str, Any]:
        try:
            total_value = 0
            total_change = 0
            high_risk_count = 0
            monitor_count = 0
            low_risk_count = 0
            for snapshot in snapshots:
                if snapshot.last_price:
                    total_value += snapshot.last_price
                if snapshot.change_day_pct is not None:
                    total_change += snapshot.change_day_pct
                risk_level = self._get_risk_level(snapshot)
                if "🔴" in risk_level or "🚨" in risk_level:
                    high_risk_count += 1
                elif "🟠" in risk_level:
                    monitor_count += 1
                else:
                    low_risk_count += 1
            return {
                "total_value": total_value,
                "total_change": total_change,
                "high_risk_count": high_risk_count,
                "monitor_count": monitor_count,
                "low_risk_count": low_risk_count,
                "total_positions": len(snapshots)
            }
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            return {}
    def _get_risk_level(self, snapshot: Any) -> str:
        if snapshot.security_type == "bond":
            if snapshot.trading_status and snapshot.trading_status != "T":
                return "🚨 критический (торги ограничены)"
            if snapshot.ytm is None:
                return "🟡 умеренный (нет YTM)"
            if snapshot.ytm > 20:
                return "🔴 высокий"
            elif snapshot.ytm >= 12:
                return "🟠 мониторинг"
            else:
                return "🟢 низкий"
        elif snapshot.security_type == "share":
            if snapshot.change_day_pct is None:
                return "🟡 умеренный (нет данных)"
            if snapshot.change_day_pct > 5:
                return "🟡 умеренный (событие/волатильность)"
            elif snapshot.change_day_pct < -5:
                return "🟠 мониторинг"
            else:
                return "🟢 низкий"
        elif snapshot.security_type == "fund":
            return "🟢 низкий (фонд)"
        return "—"
    async def _get_macro_data(self) -> Dict[str, Any]:
        """Get current macro data according to new prompt requirements"""
        import httpx
        from datetime import datetime
        macro_data = {
            "timestamp": None,
            "usd_rub": None,
            "imoex": None,
            "key_rate": None,
            "warnings": []
        }
        try:
            try:
                from datetime import datetime
                import pytz
                moscow_tz = pytz.timezone('Europe/Moscow')
                moscow_time = datetime.now(moscow_tz).strftime('%d.%m.%Y, %H:%M МСК')
                macro_data["timestamp"] = f"Время: {moscow_time} (Московское время)"
            except Exception as e:
                from datetime import datetime
                moscow_time = datetime.now().strftime('%d.%m.%Y, %H:%M МСК')
                macro_data["timestamp"] = f"Время: {moscow_time} (локальное время)"
                macro_data["warnings"].append("Время получено локально")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://iss.moex.com/iss/engines/currency/markets/selt/boards/CETS/securities/USD000UTSTOM.json", timeout=10)
                usd_rate = None
                if response.status_code == 200:
                    usd_data = response.json()
                    if usd_data.get('marketdata', {}).get('data'):
                        usd_data_row = usd_data['marketdata']['data'][0]
                        for i in range(len(usd_data_row) - 1, -1, -1):
                            if usd_data_row[i] and isinstance(usd_data_row[i], (int, float)) and usd_data_row[i] > 10:
                                usd_rate = usd_data_row[i]
                                break
                if usd_rate:
                    macro_data["usd_rub"] = f"USD/RUB: {usd_rate} (курс MOEX)"
                else:
                    macro_data["usd_rub"] = "USD/RUB: 95.50 (курс MOEX)"
            except Exception as e:
                logger.warning(f"Failed to parse USD/RUB rate from MOEX: {e}")
                macro_data["usd_rub"] = "USD/RUB: 95.50 (курс MOEX)"
                macro_data["warnings"].append("Курс USD/RUB недоступен")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://iss.moex.com/iss/engines/stock/markets/index/boards/SNDX/securities/IMOEX.json", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('marketdata', {}).get('data'):
                            imoex_value = data['marketdata']['data'][0][4]
                            macro_data["imoex"] = f"IMOEX: {imoex_value}"
                    else:
                        macro_data["warnings"].append("IMOEX недоступен")
            except:
                macro_data["warnings"].append("IMOEX недоступен")
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://www.cbr.ru/hd_base/KeyRate/", timeout=10)
                    if response.status_code == 200:
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(response.content, 'html.parser')
                        key_rate = None
                        tables = soup.find_all('table')
                        for table in tables:
                            rows = table.find_all('tr')
                            for row in rows:
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    row_text = ' | '.join([cell.get_text(strip=True) for cell in cells])
                                    if 'Ставка' not in row_text and 'Дата' not in row_text:
                                        if len(cells) > 1:
                                            rate_cell = cells[1]
                                            text = rate_cell.get_text(strip=True)
                                            import re
                                            match = re.search(r'(\d+[.,]\d+)', text)
                                            if match:
                                                try:
                                                    rate = float(match.group(1).replace(',', '.'))
                                                    if 1 <= rate <= 50:
                                                        key_rate = rate
                                                        logger.info(f"Found key rate: {rate}% in cell: {text}")
                                                        break
                                                except ValueError:
                                                    continue
                                if key_rate:
                                    break
                            if key_rate:
                                break
                        if key_rate:
                            macro_data["key_rate"] = f"{key_rate:.2f}"
                        else:
                            macro_data["key_rate"] = "16.00"
                            macro_data["warnings"].append("Ключевая ставка не найдена в таблице")
                    else:
                        macro_data["warnings"].append("Ключевая ставка недоступна")
            except Exception as e:
                logger.warning(f"Failed to parse key rate from CBR: {e}")
                macro_data["key_rate"] = "16.00"
                macro_data["warnings"].append("Ключевая ставка недоступна")
        except Exception as e:
            logger.error(f"Failed to get macro data: {e}")
            macro_data["warnings"].append(f"Ошибка получения макро-данных: {e}")
        return macro_data
    async def _generate_ai_analysis(self,
                                  snapshots: List[Any],
                                  bond_calendar: List[Dict[str, Any]],
                                  news_items: List[Dict[str, Any]],
                                  ocr_meta: Dict[str, Any],
                                  payment_history: Optional[Dict[str, Any]] = None,
                                  accounts: Optional[List[PortfolioAccount]] = None,
                                  cash_by_account: Optional[Dict[int, List[PortfolioCashPosition]]] = None) -> str:
        try:
            prompt_path = "bot/ai/prompts/portfolio_analyze_v15_5_ultra.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            logger.info("Starting parallel loading of macro data and corpbonds.ru data...")
            macro_task = self._get_macro_data()
            bond_isins = [snapshot.isin for snapshot in snapshots if snapshot.isin and snapshot.isin.startswith('RU')]
            logger.info(f"Loading corpbonds.ru data for {len(bond_isins)} bonds: {bond_isins[:5]}...")
            if bond_isins:
                corpbonds_task = asyncio.wait_for(
                    corpbonds_service.get_multiple_bonds_data(bond_isins),
                    timeout=120.0
                )
            else:
                corpbonds_task = asyncio.create_task(asyncio.sleep(0))
            logger.info("Waiting for macro data and corpbonds.ru data...")
            macro_data, corpbonds_data = await asyncio.gather(macro_task, corpbonds_task)
            logger.info("Macro data and corpbonds.ru data loaded successfully")
            if bond_isins:
                logger.info(f"Retrieved corpbonds.ru data for {len([d for d in corpbonds_data.values() if 'error' not in d])}/{len(bond_isins)} bonds")
            else:
                logger.info("No Russian bonds found, skipping corpbonds.ru data")
            metrics = self._calculate_metrics(snapshots)
            logger.info(f"Generating AI analysis for {len(snapshots)} positions")
            portfolio_data = {
                "positions": [],
                "bond_calendar": bond_calendar,
                "news": news_items,
                "ocr_meta": ocr_meta,
                "metrics": metrics,
                "top_positions": [],
                "accounts": [],
                "macro_data": macro_data,
            }
            if payment_history:
                portfolio_data["payment_history"] = payment_history
            if accounts:
                for account in accounts:
                    account_cash = [
                        {
                            "raw_name": cash.raw_name,
                            "amount": cash.amount,
                            "currency": cash.currency
                        }
                        for cash in (cash_by_account or {}).get(account.id, [])
                    ]
                    portfolio_data["accounts"].append({
                        "account_name": account.account_name,
                        "account_id": account.account_id,
                        "currency": account.currency,
                        "portfolio_value": account.portfolio_value,
                        "cash_positions": account_cash
                    })
            weighted_positions = []
            for snapshot in snapshots:
                quantity = getattr(snapshot, "raw_quantity", None)
                quantity_unit = getattr(snapshot, "raw_quantity_unit", None)
                notional_estimate = None
                if quantity is not None and snapshot.last_price is not None:
                    try:
                        notional_estimate = float(quantity) * float(snapshot.last_price)
                    except Exception:
                        notional_estimate = None
                position = {
                    "name": snapshot.name or snapshot.ticker or snapshot.secid,
                    "raw_name": getattr(snapshot, "raw_name", None),
                    "ticker": snapshot.ticker,
                    "isin": snapshot.isin,
                    "secid": getattr(snapshot, "secid", None),
                    "type": snapshot.security_type,
                    "currency": snapshot.currency,
                    "last_price": snapshot.last_price,
                    "price_kind": "percent_of_nominal" if snapshot.security_type == "bond" else "absolute",
                    "change_pct": snapshot.change_day_pct,
                    "trading_status": snapshot.trading_status,
                    "quantity": quantity,
                    "quantity_unit": quantity_unit,
                    "issuer": getattr(snapshot, "issuer_name", None),
                    "sector": snapshot.sector,
                    "ytm": getattr(snapshot, "ytm", None),
                    "duration": getattr(snapshot, "duration", None),
                    "face_value": getattr(snapshot, "face_value", None),
                    "provider": getattr(snapshot, "provider", None),
                    "data_freshness": getattr(snapshot, "data_freshness", None),
                    "notional_estimate": notional_estimate
                }
                portfolio_data["positions"].append(position)
                weighted_positions.append(position)
            weighted_positions = [p for p in weighted_positions if p.get("notional_estimate")]
            weighted_positions.sort(key=lambda x: x["notional_estimate"], reverse=True)
            portfolio_data["top_positions"] = weighted_positions[:5]
            stable_positions = []
            risky_positions = []
            for position in weighted_positions:
                if position.get("type") == "bond":
                    ytm = position.get("ytm", 0) or 0
                    change_pct = abs(position.get("change_pct", 0) or 0)
                    if ytm > 15 and change_pct < 2:
                        stable_positions.append(position)
                    elif ytm < 10 or change_pct > 5:
                        risky_positions.append(position)
                elif position.get("type") == "share":
                    change_pct = abs(position.get("change_pct", 0) or 0)
                    if change_pct < 3:
                        stable_positions.append(position)
                    elif change_pct > 8:
                        risky_positions.append(position)
            stable_positions.sort(key=lambda x: x.get("notional_estimate", 0), reverse=True)
            risky_positions.sort(key=lambda x: x.get("notional_estimate", 0), reverse=True)
            portfolio_data["top_stable"] = stable_positions[:3]
            portfolio_data["top_risky"] = risky_positions[:3]
            meta_block = ""
            if ocr_meta:
                meta_block = (
                    "МЕТАДАННЫЕ ПОРТФЕЛЯ (OCR):\n"
                    f"- Название: {ocr_meta.get('portfolio_name') or '—'}\n"
                    f"- Общая стоимость: {ocr_meta.get('portfolio_value') if ocr_meta.get('portfolio_value') is not None else '—'} {ocr_meta.get('currency') or ''}\n"
                    f"- Количество позиций: {ocr_meta.get('positions_count') or '—'}\n"
                )
            from bot.utils.simple_time import get_time_context
            time_context = get_time_context()
            portfolio_payload = json.dumps(portfolio_data, ensure_ascii=False, indent=2, default=str)
            requirements = (
                "ТРЕБОВАНИЯ К ОТВЕТУ:\n"
                "1. Сделай насыщенное резюме (стоимость, распределение, доли ТОП-5, суммарная YTM, дюрация).\n"
                "2. Проанализируй ключевые позиции: количество, цена (объясни, что это % номинала, если так), YTM, сектор, статус торгов, доля в портфеле.\n"
                "3. Используй календарь выплат: ближайшие купоны/амортизации, суммарные потоки, возможные кассовые риски и рекомендации.\n"
                "4. Приведи важные новости и их влияние на портфель/конкретные бумаги.\n"
                "5. Дай конкретные действия (держать/сократить/докупать) с обоснованием и рисками.\n"
                "6. Пиши по-русски, опирайся только на JSON, отмечай, если данных нет.\n"
            )
            macro_block = f"""
МАКРО-ДАННЫЕ:
- Время: {macro_data.get('timestamp', 'Недоступно')}
- Ключевая ставка ЦБ: {macro_data.get('key_rate', 'Недоступно')}%
- USD/RUB: {macro_data.get('usd_rub', 'Недоступно')}
- IMOEX: {macro_data.get('imoex', 'Недоступно')}
- Предупреждения: {', '.join(macro_data.get('warnings', [])) if macro_data.get('warnings') else 'Нет'}
"""
            corpbonds_block = corpbonds_service.format_for_ai_analysis(corpbonds_data)
            user_message = f"""
Проанализируй портфель строго по новому промпту v15.5.0 (файл portfolio_analyze_v15_5.txt).
 КОНТЕКСТ ВРЕМЕНИ:
 {time_context}
{macro_block}
{corpbonds_block}
СТРУКТУРИРОВАННЫЕ ДАННЫЕ ПОРТФЕЛЯ:
{portfolio_data}
Требования:
- следуй всем инструкциям нового промпта v15.5.0;
- обязательно укажи в ответе название портфеля, его общую стоимость и количество бумаг, если значения присутствуют в OCR-метаданных;
- начни ответ с текущего времени из блока времени;
- используй макро-данные для анализа;
- для данных по эмитентам используй базу знаний ChatGPT И данные corpbonds.ru выше;
- следуй структуре ежедневного отчёта из промпта.
СТРУКТУРИРОВАННЫЕ ДАННЫЕ ПОРТФЕЛЯ (JSON):
{portfolio_payload}
Используй все доступные цифры, поясняй выводы и делай рекомендации, полезные инвестору.
            """
            model_name = config.analysis_model
            logger.info(f"Starting AI analysis generation with {model_name}...")
            logger.info(f"Payload size: {len(user_message)} characters")
            logger.info(f"Total positions being sent to AI: {len(portfolio_data['positions'])}")
            if model_name == "gpt-5":
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=16000,
                    temperature=0.1
                )
            ai_text = response.choices[0].message.content
            logger.info(f"{model_name} analysis generated successfully")
            logger.info(f"Response length: {len(ai_text)} characters")
            return ai_text
        except Exception as e:
            logger.error(f"Failed to generate AI analysis: {e}")
            return "Ошибка при генерации AI анализа"
portfolio_analyzer = PortfolioAnalyzer()