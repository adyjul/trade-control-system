"""
live_bot_limit_scalper_m1.py

M1 scalping bot using LIMIT (maker) orders on Binance USDT-M Futures.
- Entry: place LIMIT order (price = reference +/- entry_offset) aiming to be maker
- If entry filled, place TP + SL as LIMIT (reduceOnly) orders
- If entry not filled within `entry_timeout_sec`, cancel it (missed trade)
- Filters: ATR minimum, cooldown between trades, min_tp_abs
- Position sizing via `max_exposure_frac` (fraction of balance*leverage)
- Testnet support and `live_mode` toggle

NOTES:
- Binance Futures behavior varies across API versions. Test thoroughly on testnet before enabling live_mode.
- Futures may not support strict POST_ONLY flags; we emulate maker intention by placing limit slightly away from market and canceling if immediate fill occurs.

Dependencies: python-binance (async), pandas, openpyxl, python-dotenv

Usage: set BINANCE_API_KEY/BINANCE_API_SECRET in .env or assign cfg.api_key / cfg.api_secret
Run: python3 live_bot_limit_scalper_m1.py
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
from openpyxl import Workbook, load_workbook
from dotenv import load_dotenv

load_dotenv()

# --------- Config ---------
@dataclass
class BotConfig:
    pair: str = "AVAXUSDT"
    interval: str = "1m"
    initial_balance: float = 15.0
    leverage: int = 10
    fee_rate: float = 0.0004
    min_atr: float = 0.0005
    atr_period: int = 14
    level_mult: float = 0.2
    tp_atr_mult: float = 0.9
    sl_atr_mult: float = 0.8
    monitor_candles: int = 3
    candles_buffer: int = 500
    min_hold_sec: int = 30
    logfile: str = "trades_log_limit.xlsx"
    live_mode: bool = False        # place orders when True
    api_key: Optional[str] = os.getenv("BINANCE_API_KEY")
    api_secret: Optional[str] = os.getenv("BINANCE_API_SECRET")
    testnet: bool = False

    # maker strategy settings
    entry_offset: float = 0.0005          # fraction of price to adjust limit to favor being maker (e.g. 0.0005 => 0.05%)
    entry_timeout_sec: int = 60           # cancel entry if not filled within X seconds
    min_tp_abs: float = 0.03              # minimal absolute TP (USDT) to avoid fee bleed
    max_exposure_frac: float = 0.1        # fraction of balance*leverage used per trade
    min_time_between_trades_sec: int = 10 # throttle openings to reduce overtrading

cfg = BotConfig()

# --------- Excel helpers ---------
def init_excel(path: str):
    if not os.path.exists(path):
        wb = Workbook()
        ws = wb.active
        ws.title = "Trades"
        ws.append([
            "pair", "watch_start", "trigger_side", "trigger_level", "atr_at_trigger",
            "entry_time", "entry_price", "exit_time", "exit_price", "pnl",
            "exit_reason", "balance_after", "executed_entry_price", "executed_exit_price", "qty", "entry_order_id", "tp_order_id", "sl_order_id"
        ])
        wb.save(path)


def append_trade_excel(path: str, row: List):
    wb = load_workbook(path)
    ws = wb["Trades"]
    clean_row = []
    for v in row:
        if isinstance(v, datetime):
            clean_row.append(v.replace(tzinfo=None))
        else:
            clean_row.append(v)
    ws.append(clean_row)
    wb.save(path)

# --------- Indicators ---------

def compute_atr_from_df(df: pd.DataFrame, period: int):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

# --------- Bot Implementation ---------
class LimitScalpBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.balance = cfg.initial_balance
        self.candles = pd.DataFrame(columns=['open','high','low','close','volume'])
        self.watches: List[Dict] = []
        self._current_position: Optional[Dict] = None
        self.client: Optional[AsyncClient] = None
        self._pending_entry_orders: Dict[str, Dict] = {}  # orderId -> metadata
        self._last_trade_time: Optional[pd.Timestamp] = None
        init_excel(self.cfg.logfile)

    async def _init_client(self):
        if self.client:
            return self.client
        if self.cfg.testnet:
            self.client = await AsyncClient.create(self.cfg.api_key, self.cfg.api_secret, testnet=True)
        else:
            self.client = await AsyncClient.create(self.cfg.api_key, self.cfg.api_secret)
        try:
            await self.client.futures_change_leverage(symbol=self.cfg.pair, leverage=self.cfg.leverage)
        except Exception as e:
            print("[WARN] set leverage failed:", e)
        return self.client

    async def _format_price(self, price: float) -> float:
        client = await self._init_client()
        try:
            info = await client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == self.cfg.pair:
                    for f in s['filters']:
                        if f['filterType'] == 'PRICE_FILTER':
                            tick_size = float(f['tickSize'])
                            precision = int(round(-np.log10(tick_size)))
                            return round(price, precision)
        except Exception as e:
            print("[WARN] format price failed:", e)
        return price

    async def _close_client(self):
        if self.client:
            await self.client.close_connection()
            self.client = None

    async def _format_quantity(self, qty: float) -> float:
        client = await self._init_client()
        try:
            info = await client.futures_exchange_info()
            for s in info['symbols']:
                if s['symbol'] == self.cfg.pair:
                    for f in s['filters']:
                        if f['filterType'] == 'LOT_SIZE':
                            step_size = float(f['stepSize'])
                            min_qty = float(f['minQty'])
                            precision = int(round(-np.log10(step_size)))
                            qty = max(min_qty, round(qty, precision))
                            return qty
        except Exception as e:
            print("[WARN] format qty failed:", e)
        return qty

    async def _place_limit_order(self, side: str, price: float, qty: float, reduce_only: bool=False):
        """Place a LIMIT order. Returns order dict or None in paper-mode."""
        if not self.cfg.live_mode:
            # emulate an order id for paper-mode
            return {"orderId": f"paper-{int(datetime.utcnow().timestamp()*1000)}", "status": "NEW", "price": str(price), "origQty": str(qty)}
        client = await self._init_client()
        try:
            # TimeInForce GTC; for some futures APIs, POST_ONLY isn't supported. We place limit slightly away to aim for maker.
            resp = await client.futures_create_order(
                symbol=self.cfg.pair,
                side=side,
                type='LIMIT',
                timeInForce='GTC',
                quantity=qty,
                price=str(price),
                reduceOnly=reduce_only
            )
            return resp
        except Exception as e:
            print('[ERROR] place_limit_order failed:', e)
            return None

    async def _cancel_order(self, order_id: str):
        if not self.cfg.live_mode:
            return True
        client = await self._init_client()
        try:
            await client.futures_cancel_order(symbol=self.cfg.pair, orderId=order_id)
            return True
        except Exception as e:
            print('[WARN] cancel order failed:', e)
            return False

    async def _get_order(self, order_id: str):
        if not self.cfg.live_mode:
            return None
        client = await self._init_client()
        try:
            resp = await client.futures_get_order(symbol=self.cfg.pair, orderId=order_id)
            return resp
        except Exception as e:
            print('[WARN] get_order failed:', e)
            return None

    def _append_candle(self, ts, o, h, l, c, v):
        self.candles.loc[ts] = [o, h, l, c, v]
        if len(self.candles) > self.cfg.candles_buffer:
            self.candles = self.candles.iloc[-self.cfg.candles_buffer:]

    async def start(self):
        # initialize client for socket even if paper-mode
        await self._init_client()
        bm = BinanceSocketManager(self.client)
        kline_socket = bm.kline_socket(self.cfg.pair, interval=self.cfg.interval)
        print(f"[INFO] starting limit scalper {self.cfg.pair} interval={self.cfg.interval} live_mode={self.cfg.live_mode}")
        async with kline_socket as stream:
            while True:
                res = await stream.recv()
                k = res.get('k', {})
                is_closed = k.get('x', False)
                ts = pd.to_datetime(k.get('t'), unit='ms', utc=True)
                o = float(k.get('o')); h = float(k.get('h')); l = float(k.get('l')); c = float(k.get('c')); v = float(k.get('v'))
                self._append_candle(ts, o, h, l, c, v)

                # periodically check pending entry orders for timeout
                await self._check_pending_entries()

                if is_closed:
                    atr_series = compute_atr_from_df(self.candles, self.cfg.atr_period)
                    current_atr = atr_series.iat[-1] if len(atr_series) >= self.cfg.atr_period else np.nan
                    if not np.isnan(current_atr) and current_atr >= self.cfg.min_atr:
                        if self._current_position is None:
                            last_close = self.candles['close'].iat[-1]
                            long_level = last_close + current_atr * self.cfg.level_mult
                            short_level = last_close - current_atr * self.cfg.level_mult
                            watch = {
                                'start_idx': len(self.candles)-1,
                                'expire_idx': len(self.candles)-1 + self.cfg.monitor_candles,
                                'long_level': long_level,
                                'short_level': short_level,
                                'atr': current_atr,
                                'trigger_time': self.candles.index[-1]
                            }
                            self.watches.append(watch)
                            print(f"[WATCH] {watch['trigger_time']} ATR={current_atr:.6f} long={long_level:.6f} short={short_level:.6f}")

                    await self._process_watches_and_place_limit_entries()
                    await self._process_current_position_and_maybe_close()

    async def _check_pending_entries(self):
        # remove / cancel pending entry orders that timed out
        if not self._pending_entry_orders:
            return
        now = datetime.utcnow()
        to_remove = []
        for oid, meta in list(self._pending_entry_orders.items()):
            created = meta['created_at']
            elapsed = (now - created).total_seconds()
            if elapsed >= self.cfg.entry_timeout_sec:
                print(f"[TIMEOUT] Cancelling entry order {oid} after {elapsed:.1f}s")
                await self._cancel_order(oid)
                to_remove.append(oid)
        for oid in to_remove:
            self._pending_entry_orders.pop(oid, None)

    async def _process_watches_and_place_limit_entries(self):
        # throttle openings
        if self._current_position is not None:
            return
        now_idx = len(self.candles)-1
        now_ts = self.candles.index[-1]
        if self._last_trade_time is not None:
            elapsed_since = (now_ts - self._last_trade_time).total_seconds()
            if elapsed_since < self.cfg.min_time_between_trades_sec:
                return

        candle_high = self.candles['high'].iat[-1]
        candle_low = self.candles['low'].iat[-1]

        new_watches = []
        for w in self.watches:
            if now_idx <= w['start_idx']:
                new_watches.append(w); continue

            triggered_side = None
            trigger_price = None
            atr_now = w['atr']

            if candle_high >= w['long_level']:
                triggered_side = 'LONG'; trigger_price = w['long_level']
            elif candle_low <= w['short_level']:
                triggered_side = 'SHORT'; trigger_price = w['short_level']

            if triggered_side:
                # compute TP/SL
                tp_price = trigger_price + atr_now * self.cfg.tp_atr_mult if triggered_side == 'LONG' else trigger_price - atr_now * self.cfg.tp_atr_mult
                sl_price = trigger_price - atr_now * self.cfg.sl_atr_mult if triggered_side == 'LONG' else trigger_price + atr_now * self.cfg.sl_atr_mult

                # ensure minimal TP absolute
                tp_abs = abs(tp_price - trigger_price)
                if tp_abs < self.cfg.min_tp_abs:
                    # bump tp outward
                    if triggered_side == 'LONG':
                        tp_price = trigger_price + self.cfg.min_tp_abs
                        sl_price = trigger_price - self.cfg.min_tp_abs
                    else:
                        tp_price = trigger_price - self.cfg.min_tp_abs
                        sl_price = trigger_price + self.cfg.min_tp_abs
                    tp_abs = abs(tp_price - trigger_price)

                if tp_abs < self.cfg.min_tp_abs:
                    print(f"[SKIP] computed TP {tp_abs:.6f} < min_tp_abs")
                    continue

                # compute quantity
                max_notional = self.balance * self.cfg.leverage * self.cfg.max_exposure_frac
                qty = max_notional / trigger_price if trigger_price > 0 else 0.000001
                qty = max(0.000001, qty)
                qty = await self._format_quantity(qty)

                # prepare limit entry price with offset to try be maker
                if triggered_side == 'LONG':
                    limit_price = trigger_price * (1 - self.cfg.entry_offset)
                    side = 'BUY'
                else:
                    limit_price = trigger_price * (1 + self.cfg.entry_offset)
                    side = 'SELL'

                # place entry limit order
                # resp = await self._place_limit_order(side, round(limit_price,8), qty, reduce_only=False)
                limit_price = await self._format_price(limit_price)
                resp = await self._place_limit_order(side, limit_price, qty, reduce_only=False)
                if resp is None:
                    print('[WARN] entry order failed to create')
                    continue

                order_id = str(resp.get('orderId') or resp.get('clientOrderId') or resp.get('orderId'))
                self._pending_entry_orders[order_id] = {
                    'side': triggered_side,
                    'trigger_price': trigger_price,
                    'limit_price': float(limit_price),
                    'qty': qty,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'created_at': datetime.utcnow()
                }
                print(f"[ENTRY_ORDER] id={order_id} side={triggered_side} limit={limit_price:.6f} qty={qty} tp={tp_price:.6f} sl={sl_price:.6f}")
                # do not re-add watch
            else:
                if now_idx < w['expire_idx']:
                    new_watches.append(w)
        self.watches = new_watches

        # after placing, check quickly if any pending entry got filled (paper-mode simulates fill immediately)
        await self._poll_pending_entries_and_handle_fills()

    async def _poll_pending_entries_and_handle_fills(self):
        # iterate over pending entry orders and see if filled
        for oid, meta in list(self._pending_entry_orders.items()):
            # in live_mode, poll order status
            order_info = await self._get_order(oid) if self.cfg.live_mode else {'status':'FILLED', 'avgPrice': str(meta['limit_price'])}
            status = (order_info.get('status') if isinstance(order_info, dict) else None) or order_info
            if isinstance(status, str) and status.upper() == 'FILLED' or (isinstance(order_info, dict) and order_info.get('status') == 'FILLED'):
                # handle fill
                executed_price = None
                if isinstance(order_info, dict):
                    executed_price = float(order_info.get('avgPrice') or order_info.get('price') or meta['limit_price'])
                else:
                    executed_price = meta['limit_price']

                # set current position metadata
                self._current_position = {
                    'side': meta['side'],
                    'entry_price': meta['trigger_price'],            # keep trigger-level for log/backtest
                    'executed_entry_price': executed_price,
                    'tp_price': meta['tp_price'],
                    'sl_price': meta['sl_price'],
                    'qty': meta['qty'],
                    'entry_time': datetime.utcnow(),
                    'entry_order_id': oid,
                    'tp_order_id': None,
                    'sl_order_id': None
                }
                # place TP and SL as limit reduceOnly orders
                # TP (take profit)
                # tp_side = 'SELL' if meta['side']=='LONG' else 'BUY'
                # sl_side = 'SELL' if meta['side']=='SHORT' else 'BUY'
                # tp_resp = await self._place_limit_order(tp_side, round(meta['tp_price'],8), meta['qty'], reduce_only=True)
                # sl_resp = await self._place_limit_order(sl_side, round(meta['sl_price'],8), meta['qty'], reduce_only=True)
                
                tp_price = await self._format_price(meta['tp_price'])
                sl_price = await self._format_price(meta['sl_price'])
                tp_resp = await self._place_limit_order(tp_side, tp_price, meta['qty'], reduce_only=True)
                sl_resp = await self._place_limit_order(sl_side, sl_price, meta['qty'], reduce_only=True)


                tp_oid = str(tp_resp.get('orderId') if isinstance(tp_resp, dict) else f"paper-tp-{int(datetime.utcnow().timestamp()*1000)}")
                sl_oid = str(sl_resp.get('orderId') if isinstance(sl_resp, dict) else f"paper-sl-{int(datetime.utcnow().timestamp()*1000)}")

                self._current_position['tp_order_id'] = tp_oid
                self._current_position['sl_order_id'] = sl_oid

                print(f"[FILLED] entry executed={executed_price:.6f} qty={meta['qty']} tp_oid={tp_oid} sl_oid={sl_oid}")

                # remove pending entry record
                self._pending_entry_orders.pop(oid, None)
                self._last_trade_time = datetime.utcnow()

    async def _process_current_position_and_maybe_close(self):
        if self._current_position is None:
            return
        pos = self._current_position
        latest_candle = self.candles.iloc[-1]
        now = self.candles.index[-1]
        elapsed = (now - pos['entry_time']).total_seconds()

        exit_reason = None
        exit_price = None
        executed_exit_price = None

        if elapsed >= self.cfg.min_hold_sec:
            if pos['side'] == 'LONG':
                if latest_candle['high'] >= pos['tp_price']:
                    exit_price = pos['tp_price']; exit_reason = 'TP'
                elif latest_candle['low'] <= pos['sl_price']:
                    exit_price = pos['sl_price']; exit_reason = 'SL'
            else:
                if latest_candle['low'] <= pos['tp_price']:
                    exit_price = pos['tp_price']; exit_reason = 'TP'
                elif latest_candle['high'] >= pos['sl_price']:
                    exit_price = pos['sl_price']; exit_reason = 'SL'

        if exit_price is not None:
            # to close, cancel the opposite order (if exists) then let TP/SL order fill or place market close (we prefer maker TP/SL)
            # For safety, we cancel both reduceOnly orders and place a market close (optional). Here we'll cancel both and simulate fill in paper-mode.
            if self.cfg.live_mode:
                # try cancel TP & SL to avoid duplicates; actual behavior depends on which was hit first
                try:
                    if pos.get('tp_order_id'):
                        await self._cancel_order(pos['tp_order_id'])
                    if pos.get('sl_order_id'):
                        await self._cancel_order(pos['sl_order_id'])
                except Exception:
                    pass

                # place market close to ensure exit (this is taker and incurs fee) -> optional; instead we try to place a LIMIT at exit_price to remain maker
                close_side = 'SELL' if pos['side']=='LONG' else 'BUY'
                # resp_close = await self._place_limit_order(close_side, round(exit_price,8), pos['qty'], reduce_only=True)
                exit_price_fmt = await self._format_price(exit_price)
                resp_close = await self._place_limit_order(close_side, exit_price_fmt, pos['qty'], reduce_only=True)
                executed_exit_price = float(resp_close.get('avgPrice') or resp_close.get('price') or exit_price) if isinstance(resp_close, dict) else exit_price
            else:
                executed_exit_price = exit_price

            used_entry_price = pos.get('executed_entry_price') or pos['entry_price']
            pnl = self._compute_pnl(used_entry_price, executed_exit_price, pos['side'], pos['qty'])
            self.balance += pnl

            trade = [
                self.cfg.pair, pos['entry_time'], pos['side'], pos['entry_price'], None,
                pos['entry_time'], used_entry_price, now, executed_exit_price, pnl,
                exit_reason, self.balance, pos.get('executed_entry_price'), executed_exit_price, pos['qty'], pos.get('entry_order_id'), pos.get('tp_order_id'), pos.get('sl_order_id')
            ]
            append_trade_excel(self.cfg.logfile, trade)
            print(f"[CLOSED] {now} {pos['side']} entry={used_entry_price:.6f} exit={executed_exit_price:.6f} pnl={pnl:.6f} balance={self.balance:.4f}")
            self._current_position = None

    def _compute_pnl(self, entry_price: float, exit_price: float, side: str, qty: float):
        if side == 'LONG':
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price
        gross_pnl = price_diff * qty
        entry_notional = entry_price * qty
        exit_notional = exit_price * qty
        total_fee = self.cfg.fee_rate * (entry_notional + exit_notional)
        pnl = gross_pnl - total_fee
        return pnl

# --------- run ---------
if __name__ == '__main__':
    cfg.testnet = False
    cfg.live_mode = True
    cfg.api_key = os.getenv('BINANCE_API_KEY')
    cfg.api_secret = os.getenv('BINANCE_API_SECRET')

    init_excel(cfg.logfile)
    bot = LimitScalpBot(cfg)
    try:
        asyncio.get_event_loop().run_until_complete(bot.start())
    except KeyboardInterrupt:
        print('[STOP] interrupted. Trades saved to', cfg.logfile)
        try:
            asyncio.get_event_loop().run_until_complete(bot._close_client())
        except:
            pass
