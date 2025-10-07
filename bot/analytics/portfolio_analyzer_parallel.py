import asyncio
import json
import openai
from typing import List, Dict, Any, Optional
from datetime import datetime
from bot.core.config import config
from bot.analytics.portfolio_analyzer_corpbonds_full import corpbonds_full_analyzer
from bot.utils.integrated_bond_client import IntegratedBondClient
from bot.analytics.payment_history import PaymentHistoryAnalyzer
from bot.utils.render import (
    render_signals_as_cards, render_calendar_30d, render_news_summary,
    render_portfolio_summary, render_recommendations, render_payment_history
)
from bot.core.logging import get_logger
from bot.utils.bond_cache import bond_cache
logger = get_logger(__name__)
class IntegratedSnapshot:
    def __init__(self, data: dict):
        self.isin = data.get('isin')
        self.name = data.get('name')
        self.ticker = data.get('ticker')
        self.issuer_name = data.get('issuer_name')
        self.price = data.get('price')
        self.yield_to_maturity = data.get('yield_to_maturity')
        self.duration = data.get('duration')
        self.credit_rating = data.get('credit_rating')
        self.currency = data.get('currency')
        self.face_value = data.get('face_value')
        self.coupon_rate = data.get('coupon_rate')
        self.maturity_date = data.get('maturity_date')
        self.issue_date = data.get('issue_date')
        self.volume = data.get('volume')
        self.change_percent = data.get('change_percent')
        self.holding_id = data.get('holding_id')
        self.raw_name = data.get('raw_name')
        self.raw_quantity = data.get('raw_quantity')
        self.quantity = data.get('quantity')
        self.ytm = data.get('ytm')
        self.sector = data.get('sector')
        self.rating = data.get('rating')
        self.rating_agency = data.get('rating_agency')
        self.security_type = data.get('security_type')
class ParallelPortfolioAnalyzer:
    """Оптимизированный анализатор портфеля с параллельной обработкой"""
    def __init__(self):
        self.openai_client = None
        self.corpbonds_service = corpbonds_service
        self.payment_analyzer = PaymentHistoryAnalyzer()
    async def init_services(self):
        """Инициализация сервисов"""
        import openai
        self.openai_client = openai.AsyncOpenAI(
            api_key=config.openai_api_key,
            timeout=300.0,
            max_retries=3
        )
    async def run_analysis(self, user_id: int) -> Dict[str, Any]:
        """Запуск анализа портфеля с использованием только CorpBonds.ru"""
        logger.info(f"Starting CorpBonds.ru portfolio analysis for user {user_id}")
        start_time = datetime.now()
        try:
            logger.info("Using CorpBonds.ru as primary data source...")
            result = await corpbonds_full_analyzer.run_analysis(user_id)
            if not result.get("success"):
                return result
            holdings = await self._load_user_holdings(user_id)
            accounts = await self._load_user_accounts(user_id)
            cash_positions = await self._load_account_cash(user_id, accounts)
            news_task = asyncio.create_task(self._get_news_parallel(holdings))
            payment_task = asyncio.create_task(self._get_payment_history_parallel(holdings))
            calendar_task = asyncio.create_task(self._get_bond_calendar_parallel(holdings))
            macro_task = asyncio.create_task(self._get_macro_data_parallel())
            news_data, payment_data, calendar_data, macro_data = await asyncio.gather(
                news_task, payment_task, calendar_task, macro_task, return_exceptions=True
            )
            if isinstance(news_data, Exception):
                logger.error(f"News loading failed: {news_data}")
                news_data = []
            if isinstance(payment_data, Exception):
                logger.error(f"Payment history loading failed: {payment_data}")
                payment_data = None
            if isinstance(calendar_data, Exception):
                logger.error(f"Calendar loading failed: {calendar_data}")
                calendar_data = ""
            if isinstance(macro_data, Exception):
                logger.error(f"Macro data loading failed: {macro_data}")
                macro_data = {}
            result.update({
                "bond_calendar": calendar_data if calendar_data else "",
                "news": news_data,
                "payment_history": payment_data,
                "macro_data": macro_data,
                "accounts": accounts,
                "cash_by_account": cash_positions,
                "generated_at": datetime.now().isoformat(),
                "ai_analysis": result.get("analysis", "")
            })
            total_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"CorpBonds.ru analysis completed in {total_time:.2f} seconds")
            return result
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
            return {
                "error": f"Ошибка анализа: {str(e)}",
                "summary": "Произошла ошибка при анализе портфеля",
                "generated_at": datetime.now().isoformat()
            }
    async def _load_user_holdings(self, user_id: int) -> List[PortfolioHoldingV2]:
        """Загрузка позиций пользователя"""
        return db_manager.get_user_holdings(user_id)
    async def _load_user_accounts(self, user_id: int) -> List[PortfolioAccount]:
        """Загрузка аккаунтов пользователя"""
        return db_manager.get_user_accounts(user_id)
    async def _load_account_cash(self, user_id: int, accounts: List[PortfolioAccount]) -> Dict[int, List[PortfolioCashPosition]]:
        """Загрузка денежных позиций"""
        cash_positions = {}
        for account in accounts:
            try:
                cash_positions[account.id] = db_manager.get_account_cash_positions(account.id)
            except AttributeError:
                cash_positions[account.id] = []
        return cash_positions
    async def _get_market_data_parallel(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        """Параллельное получение рыночных данных"""
        try:
            queries = []
            for holding in holdings:
                query = holding.ticker or holding.isin or holding.raw_name or holding.normalized_name
                if query:
                    queries.append(query)
            if not queries:
                return []
            snapshots = []
            enriched_snapshots = []
            for i, snapshot in enumerate(snapshots):
                if i < len(holdings):
                    holding = holdings[i]
                    snapshot.holding_id = holding.id
                    snapshot.raw_name = holding.raw_name
                    snapshot.raw_quantity = holding.raw_quantity
                    enriched_snapshots.append(snapshot)
            logger.info(f"Retrieved market data for {len(enriched_snapshots)} holdings")
            return enriched_snapshots
        except Exception as e:
            logger.error(f"Failed to get market data: {e}")
            return []
    async def _get_integrated_bond_data_parallel(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        """Параллельное получение данных облигаций"""
        try:
            bond_holdings = []
            for h in holdings:
                if h.isin and h.isin.startswith('RU'):
                    bond_holdings.append(h)
                elif not h.isin and h.ticker and h.ticker.startswith('RU') and len(h.ticker) == 12:
                    bond_holdings.append(h)
                elif h.security_type == "bond":
                    bond_holdings.append(h)
            if not bond_holdings:
                return []
            client = IntegratedBondClient()
            isins = [h.isin or h.ticker for h in bond_holdings if h.isin or h.ticker]
            tasks = []
            for isin in isins:
                task = asyncio.create_task(self._get_single_bond_data(client, isin))
                tasks.append(task)
            bond_data = await asyncio.gather(*tasks, return_exceptions=True)
            valid_data = [data for data in bond_data if not isinstance(data, Exception) and data]
            logger.info(f"Retrieved bond data for {len(valid_data)} bonds")
            return valid_data
        except Exception as e:
            logger.error(f"Failed to get bond data: {e}")
            return []
    async def _get_single_bond_data(self, client, isin: str) -> Any:
        """Получение данных одной облигации с кэшированием"""
        try:
            cache_key = f"bond_data_{isin}"
            cached_data = bond_cache.get(cache_key)
            if cached_data:
                logger.debug(f"Using cached data for {isin}")
                return cached_data
            data = await client.get_bond_data(isin)
            if data:
                bond_cache.set(cache_key, data, ttl=1800)
                logger.debug(f"Cached data for {isin}")
            return data
        except Exception as e:
            logger.warning(f"Failed to get bond data for {isin}: {e}")
            return None
    async def _get_news_parallel(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        """Параллельное получение новостей"""
        try:
            tasks = []
            for holding in holdings:
                task = asyncio.create_task(self._get_news_for_holding(holding))
                tasks.append(task)
            news_data = await asyncio.gather(*tasks, return_exceptions=True)
            valid_news = [news for news in news_data if not isinstance(news, Exception) and news]
            logger.info(f"Retrieved news for {len(valid_news)} holdings")
            return valid_news
        except Exception as e:
            logger.error(f"Failed to get news: {e}")
            return []
    async def _get_news_for_holding(self, holding: PortfolioHoldingV2) -> Any:
        """Получение новостей для одной позиции"""
        try:
            tickers = [holding.ticker] if holding.ticker else []
            issuers = [holding.normalized_name] if holding.normalized_name else []
            return await get_news_for_portfolio(tickers, issuers)
        except Exception as e:
            logger.warning(f"Failed to get news for holding {holding.id}: {e}")
            return None
    async def _get_payment_history_parallel(self, holdings: List[PortfolioHoldingV2]) -> Any:
        """Параллельное получение истории выплат"""
        try:
            bond_holdings = [h for h in holdings if h.isin and h.isin.startswith('RU')]
            if not bond_holdings:
                return {}
            tasks = []
            for holding in bond_holdings:
                task = asyncio.create_task(
                    self.payment_analyzer.get_payment_history(holding.isin, months_back=12)
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            payment_history = {}
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get payment history for {bond_holdings[i].isin}: {result}")
                    continue
                if result:
                    payment_history[bond_holdings[i].isin] = result
            return payment_history
        except Exception as e:
            logger.error(f"Failed to get payment history: {e}")
            return {}
    async def _get_bond_calendar_parallel(self, holdings: List[PortfolioHoldingV2]) -> str:
        """Параллельное получение календаря выплат"""
        try:
            bond_holdings = [h for h in holdings if h.isin and h.isin.startswith('RU')]
            if not bond_holdings:
                return ""
            tasks = []
            for holding in bond_holdings:
                task = asyncio.create_task(
                    asyncio.sleep(0)
                )
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            calendar_events = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to get calendar for {bond_holdings[i].isin}: {result}")
                    continue
                if result and isinstance(result, dict):
                    for event in result.get('events', []):
                        event['bond_name'] = bond_holdings[i].normalized_name
                        event['bond_isin'] = bond_holdings[i].isin
                        calendar_events.append(event)
            calendar_events.sort(key=lambda x: x.get('date', ''))
            from datetime import datetime, timedelta
            today = datetime.now()
            thirty_days = today + timedelta(days=30)
            upcoming_events = [
                event for event in calendar_events
                if event.get('date') and datetime.fromisoformat(event['date'].replace('Z', '+00:00')).date() <= thirty_days.date()
            ]
            if upcoming_events:
                return render_calendar_30d(upcoming_events)
            else:
                return "Нет предстоящих выплат в ближайшие 30 дней"
        except Exception as e:
            logger.error(f"Failed to get bond calendar: {e}")
            return ""
    async def _get_macro_data_parallel(self) -> Dict[str, Any]:
        """Параллельное получение макроэкономических данных"""
        try:
            return {
                "key_rate": 17.0,
                "usd_rub": 83.0,
                "imoex": 2600.0
            }
        except Exception as e:
            logger.error(f"Failed to get macro data: {e}")
            return {}
    async def _create_basic_snapshots(self, holdings: List[PortfolioHoldingV2]) -> List[Any]:
        """Создание базовых снимков портфеля"""
        snapshots = []
        for holding in holdings:
            snapshot = IntegratedSnapshot({
                'isin': holding.isin,
                'name': holding.normalized_name,
                'raw_name': holding.raw_name,
                'raw_quantity': holding.raw_quantity,
                'holding_id': holding.id
            })
            snapshots.append(snapshot)
        return snapshots
    async def _enrich_snapshots_parallel(self, snapshots: List[Any], market_data: List[Any], bond_data: List[Any]) -> List[Any]:
        """Параллельное обогащение снимков данными"""
        return snapshots
    async def _generate_ai_analysis_parallel(self, snapshots: List[Any], news_data: List[Any], 
                                           payment_data: Any, calendar_data: Any, macro_data: Dict[str, Any], 
                                           user_id: int) -> Dict[str, Any]:
        """Параллельная генерация AI анализа"""
        try:
            portfolio_summary = {
                "holdings_count": len(snapshots),
                "news_count": len(news_data),
                "macro_data": macro_data
            }
            prompt_path = "bot/ai/prompts/portfolio_analyze_v15_5_ultra.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                system_prompt = f.read()
            portfolio_data = {
                "positions": [
                    {
                        "name": snapshot.name,
                        "ticker": snapshot.ticker,
                        "isin": snapshot.isin,
                        "quantity": snapshot.quantity,
                        "price": snapshot.price,
                        "ytm": snapshot.ytm,
                        "duration": snapshot.duration,
                        "sector": snapshot.sector,
                        "rating": snapshot.rating,
                        "rating_agency": snapshot.rating_agency,
                        "security_type": snapshot.security_type
                    }
                    for snapshot in snapshots
                ],
                "news": news_data,
                "payment_history": payment_data,
                "calendar_30d": calendar_data,
                "macro_data": macro_data,
                "portfolio_summary": {
                    "total_positions": len(snapshots),
                    "bond_positions": len([s for s in snapshots if s.security_type == "bond"]),
                    "stock_positions": len([s for s in snapshots if s.security_type == "stock"]),
                    "total_quantity": sum(s.quantity or 0 for s in snapshots if s.quantity),
                    "holdings_count": len(snapshots)
                }
            }
            from datetime import datetime
            import pytz
            moscow_tz = pytz.timezone('Europe/Moscow')
            current_time = datetime.now(moscow_tz).strftime('%d.%m.%Y, %H:%M МСК')
            time_context = f"⏰ {current_time} — время Москвы"
            macro_block = f"""
МАКРО-ДАННЫЕ:
- Время: {macro_data.get('timestamp', 'Недоступно')}
- Ключевая ставка ЦБ: {macro_data.get('key_rate', 'Недоступно')}%
- USD/RUB: {macro_data.get('usd_rub', 'Недоступно')}
- IMOEX: {macro_data.get('imoex', 'Недоступно')}
- Предупреждения: {', '.join(macro_data.get('warnings', [])) if macro_data.get('warnings') else 'Нет'}
"""
            corpbonds_block = ""
            if hasattr(self, 'corpbonds_service') and self.corpbonds_service:
                try:
                    bond_isins = [snapshot.isin for snapshot in snapshots if snapshot.isin and snapshot.isin.startswith('RU')]
                    if bond_isins:
                        corpbonds_data = await self.corpbonds_service.get_multiple_bonds_data(bond_isins)
                        corpbonds_block = self.corpbonds_service.format_for_ai_analysis(corpbonds_data)
                except Exception as e:
                    logger.warning(f"Failed to get corpbonds data: {e}")
                    corpbonds_block = "Данные corpbonds.ru недоступны"
            user_message = f"""
Проанализируй портфель строго по новому промпту v15.5.0 (файл portfolio_analyze_v15_5.txt).
 КОНТЕКСТ ВРЕМЕНИ:
 {time_context}
{macro_block}
{corpbonds_block}
СТРУКТУРИРОВАННЫЕ ДАННЫЕ ПОРТФЕЛЯ:
{json.dumps(portfolio_data, ensure_ascii=False, indent=2, default=str)}
Требования:
- следуй всем инструкциям нового промпта v15.5.0;
- обязательно укажи в ответе название портфеля, его общую стоимость и количество бумаг;
- начни ответ с текущего времени из блока времени;
- используй макро-данные для анализа;
- для данных по эмитентам используй базу знаний ChatGPT И данные corpbonds.ru выше;
- следуй структуре ежедневного отчёта из промпта.
Используй все доступные цифры, поясняй выводы и делай рекомендации, полезные инвестору.
            """
            if config.analysis_model in ["gpt-5", "gpt-5-mini"]:
                response = await self.openai_client.chat.completions.create(
                    model=config.analysis_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ]
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model=config.analysis_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    max_tokens=2000,
                    temperature=0.1
                )
            analysis = response.choices[0].message.content
            return {
                "analysis": analysis,
                "summary": f"Анализ портфеля из {len(snapshots)} позиций",
                "signals_table": "",
                "signals": [],
                "calendar_30d": "",
                "news_summary": "",
                "news": news_data,
                "payment_history_summary": "",
                "recommendations": "",
                "metrics": {},
                "payment_history": [],
                "ai_analysis": analysis
            }
        except Exception as e:
            logger.error(f"Failed to generate AI analysis: {e}")
            return {
                "analysis": f"Ошибка анализа: {str(e)}",
                "summary": "Ошибка при генерации анализа",
                "signals_table": "",
                "signals": [],
                "calendar_30d": "",
                "news_summary": "",
                "news": [],
                "payment_history_summary": "",
                "recommendations": "",
                "metrics": {},
                "payment_history": [],
                "ai_analysis": f"Ошибка анализа: {str(e)}"
            }
portfolio_analyzer = ParallelPortfolioAnalyzer()
PortfolioAnalyzer = ParallelPortfolioAnalyzer