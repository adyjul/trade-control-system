# live_bot_dual_entry_liveclose_improved.py
import asyncio
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from binance import AsyncClient, BinanceSocketManager
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET, ORDER_TYPE_LIMIT, TIME_IN_FORCE_GTC
from dotenv import load_dotenv
load_dotenv()

# ---------------- Config ----------------
@dataclass
class BotConfig:
    api_key: str = os.getenv('BINANCE_API_KEY')
    api_secret: str = os.getenv('BINANCE_API_SECRET')
    pair: str = "AVAXUSDT"
    interval: str = "1m"
    initial_balance: float = 20.0
    leverage: float = 10.0
    fee_rate: float = 0.0004
    min_atr: float = 0.0005
    atr_period: int = 14
    level_mult: float = 0.18
    tp_atr_mult: float = 0.65
    sl_atr_mult: float = 0.8
    monitor_candles: int = 3
    candles_buffer: int = 1000
    min_hold_sec: int = 15
    logfile: str = "trades_log.xlsx"
    risk_pct: float = 0.012
    margin_type: str = "ISOLATED"
    use_testnet: bool = True  # Set True for testing
    # New parameters for improvement
    use_limit_orders: bool = True  # Use limit orders instead of market orders
    slippage_pct: float = 0.001  # Estimated slippage for market orders
    require_confirmation: bool = True  # Require candle close confirmation for entries
    adaptive_tp_sl: bool = True  # Adjust TP/SL based on current volatility

# ---------------- Excel Logger ----------------
def init_excel(path: str):
    if not os.path.exists(path):
        wb = Workbook()
        ws = wb.active
        ws.title = "Trades"
        ws.append([
            "pair", "entry_time", "side", "entry_price", "qty", "tp_price", "sl_price",
            "exit_time", "exit_price", "pnl", "fees", "exit_reason", "balance_after",
            "atr_value", "volatility_ratio"  # Added fields for analysis
        ])
        wb.save(path)

def append_trade_excel(path: str, row: List):
    wb = load_workbook(path)
    ws = wb["Trades"]
    clean_row = [v.replace(tzinfo=None) if isinstance(v, datetime) else v for v in row]
    ws.append(clean_row)
    wb.save(path)

# ---------------- Technical helpers ----------------
def compute_atr_from_df(df: pd.DataFrame, period: int):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def compute_volatility_ratio(df: pd.DataFrame, atr_period: int = 14):
    """Calculate volatility ratio to adjust TP/SL dynamically"""
    atr = compute_atr_from_df(df, atr_period)
    current_atr = atr.iat[-1] if len(atr) > 0 else 0
    price = df['close'].iat[-1] if len(df) > 0 else 1
    return current_atr / price if price > 0 else 0

def compute_qty_by_risk(balance: float, risk_pct: float, entry_price: float, sl_price: float):
    risk_amount = balance * risk_pct
    risk_per_unit = abs(entry_price - sl_price)
    if risk_per_unit == 0:
        return 0
    qty = risk_amount / risk_per_unit
    return qty

async def round_qty_to_step(client: AsyncClient, symbol: str, raw_qty: float):
    info = await client.futures_exchange_info()
    stepSize = None
    minQty = None
    for s in info['symbols']:
        if s['symbol'] == symbol:
            lot = next((f for f in s['filters'] if f['filterType'] == 'LOT_SIZE'), None)
            if lot:
                stepSize = float(lot['stepSize'])
                minQty = float(lot['minQty'])
            break
    if stepSize is None:
        raise RuntimeError("LOT_SIZE not found for symbol")
    rounded = math.floor(raw_qty / stepSize) * stepSize
    if rounded < (minQty or 0):
        return 0.0
    return float(f"{rounded:.8f}")

# ---------------- Live Bot ----------------
class ImprovedLiveDualEntryBot:
    def __init__(self, cfg: BotConfig):
        self.cfg = cfg
        self.balance = cfg.initial_balance
        self.candles = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        self._current_position: Optional[Dict] = None
        self.watches: List[Dict] = []
        self.pending_orders: List[Dict] = []  # Track pending limit orders
        init_excel(self.cfg.logfile)
        self.client: Optional[AsyncClient] = None
        self.bm: Optional[BinanceSocketManager] = None
        self.volatility_ratio = 0.0

    async def start(self):
        self.client = await AsyncClient.create(self.cfg.api_key, self.cfg.api_secret, testnet=self.cfg.use_testnet)
        try:
            await self.client.futures_change_leverage(symbol=self.cfg.pair, leverage=int(self.cfg.leverage))
            try:
                await self.client.futures_change_margin_type(symbol=self.cfg.pair, marginType=self.cfg.margin_type)
            except Exception:
                pass
        except Exception as e:
            print("[WARN] set leverage/margin:", e)

        self.bm = BinanceSocketManager(self.client)
        print(f"[INFO] Starting improved live-dual-entry bot for {self.cfg.pair} (testnet={self.cfg.use_testnet})")
        print(f"[CONFIG] Use limit orders: {self.cfg.use_limit_orders}, Require confirmation: {self.cfg.require_confirmation}")
        
        async with self.bm.kline_socket(self.cfg.pair, interval=self.cfg.interval) as stream:
            while True:
                try:
                    res = await stream.recv()
                    k = res.get('k', {})
                    is_closed = k.get('x', False)
                    ts = pd.to_datetime(k.get('t'), unit='ms', utc=True)
                    o, h, l, c, v = map(float, (k.get('o'), k.get('h'), k.get('l'), k.get('c'), k.get('v')))
                    self._append_candle(ts, o, h, l, c, v)

                    # Update volatility ratio
                    self.volatility_ratio = compute_volatility_ratio(self.candles, self.cfg.atr_period)

                    if is_closed:
                        atr_series = compute_atr_from_df(self.candles, self.cfg.atr_period)
                        current_atr = atr_series.iat[-1] if len(atr_series) >= self.cfg.atr_period else np.nan

                        if not np.isnan(current_atr) and current_atr >= self.cfg.min_atr:
                            self._create_watch(current_atr)

                        await self._process_watches()
                        await self._process_current_position()
                        await self._check_pending_orders()  # Check if pending orders need cancellation
                except Exception as e:
                    print("[ERROR] main loop:", e)
                    await asyncio.sleep(1)

    def _append_candle(self, ts, o, h, l, c, v):
        row = pd.DataFrame([[o, h, l, c, v]], index=[ts], columns=['open', 'high', 'low', 'close', 'volume'])
        if self.candles.empty:
            self.candles = row
        else:
            self.candles = pd.concat([self.candles, row])
        if len(self.candles) > self.cfg.candles_buffer:
            self.candles = self.candles.iloc[-self.cfg.candles_buffer:]

    def _create_watch(self, atr_value):
        if self._current_position is not None:
            return
        last_close = self.candles['close'].iat[-1]
        
        # Adjust levels based on volatility
        volatility_mult = 1.0
        if self.cfg.adaptive_tp_sl and self.volatility_ratio > 0.005:  # High volatility
            volatility_mult = 1.2  # Increase distance in high volatility
        elif self.cfg.adaptive_tp_sl and self.volatility_ratio < 0.002:  # Low volatility
            volatility_mult = 0.8  # Decrease distance in low volatility
            
        watch = {
            "start_idx": len(self.candles) - 1,
            "expire_idx": len(self.candles) - 1 + self.cfg.monitor_candles,
            "long_level": last_close + atr_value * self.cfg.level_mult * volatility_mult,
            "short_level": last_close - atr_value * self.cfg.level_mult * volatility_mult,
            "atr": atr_value,
            "trigger_time": self.candles.index[-1],
            "volatility_mult": volatility_mult
        }
        self.watches.append(watch)
        print(f"[WATCH CREATED] {watch['trigger_time']} ATR={atr_value:.6f} long={watch['long_level']:.6f} short={watch['short_level']:.6f} vol_mult={volatility_mult:.2f}")

    async def _process_watches(self):
        if self._current_position is not None:
            return

        latest_idx = len(self.candles) - 1
        candle_high = self.candles['high'].iat[-1]
        candle_low = self.candles['low'].iat[-1]
        candle_close = self.candles['close'].iat[-1]

        new_watches = []
        for w in self.watches:
            if latest_idx <= w['start_idx']:
                new_watches.append(w)
                continue

            triggered = False
            side = None
            entry_price = None
            tp_price = None
            sl_price = None

            # Check if we need candle close confirmation
            if self.cfg.require_confirmation:
                long_condition = candle_close >= w['long_level']
                short_condition = candle_close <= w['short_level']
            else:
                long_condition = candle_high >= w['long_level']
                short_condition = candle_low <= w['short_level']

            if long_condition:
                triggered = True
                side = 'LONG'
                entry_price = w['long_level']
                # Adjust TP/SL based on volatility
                tp_mult = self.cfg.tp_atr_mult * w['volatility_mult']
                sl_mult = self.cfg.sl_atr_mult * w['volatility_mult']
                tp_price = entry_price + w['atr'] * tp_mult
                sl_price = entry_price - w['atr'] * sl_mult
            elif short_condition:
                triggered = True
                side = 'SHORT'
                entry_price = w['short_level']
                # Adjust TP/SL based on volatility
                tp_mult = self.cfg.tp_atr_mult * w['volatility_mult']
                sl_mult = self.cfg.sl_atr_mult * w['volatility_mult']
                tp_price = entry_price - w['atr'] * tp_mult
                sl_price = entry_price + w['atr'] * sl_mult

            if triggered:
                print(f"[TRIGGER] {w['trigger_time']} side={side} entry={entry_price:.6f} tp={tp_price:.6f} sl={sl_price:.6f}")
                if self.cfg.use_limit_orders:
                    await self._place_limit_order(side, entry_price, tp_price, sl_price, w['atr'], w['volatility_mult'])
                else:
                    await self._open_market_position(side, entry_price, tp_price, sl_price, w['atr'], w['volatility_mult'])
            else:
                if latest_idx < w['expire_idx']:
                    new_watches.append(w)

        self.watches = new_watches

    async def _place_limit_order(self, side: str, entry_price: float, tp_price: float, sl_price: float, atr_value: float, vol_mult: float):
        # Add small buffer for limit orders to improve fill rate
        # if side == 'LONG':
        #     order_price = entry_price * (1 - 0.0005)  # 0.05% below trigger
        # else:
        #     order_price = entry_price * (1 + 0.0005)  # 0.05% above trigger
            
        # raw_qty = compute_qty_by_risk(self.balance, self.cfg.risk_pct, entry_price, sl_price)
        # qty = await round_qty_to_step(self.client, self.cfg.pair, raw_qty)
        # if qty <= 0:
        #     print("[ORDER SKIP] calculated qty <= 0 (minQty or too small).")
        #     return

        qty = 1  # 1 AVAX
        # Calculate actual risk percentage untuk monitoring
        risk_per_trade = abs(entry_price - sl_price) * qty
        risk_percentage = (risk_per_trade / self.balance) * 100

        if side == 'LONG':
            order_price = entry_price * (1 - 0.0005)  # 0.05% below trigger
        else:
            order_price = entry_price * (1 + 0.0005)  # 0.05% above trigger

        print(f"[DEBUG] Actual risk: {risk_percentage:.2f}% per trade")

        try:
            # Place limit order
            order = await self.client.futures_create_order(
                symbol=self.cfg.pair,
                side=SIDE_BUY if side == 'LONG' else SIDE_SELL,
                type=ORDER_TYPE_LIMIT,
                timeInForce=TIME_IN_FORCE_GTC,
                quantity=qty,
                price=order_price
            )
            
            # Store pending order
            self.pending_orders.append({
                "order_id": order['orderId'],
                "side": side,
                "entry_price": entry_price,
                "order_price": order_price,
                "qty": qty,
                "tp_price": tp_price,
                "sl_price": sl_price,
                "atr": atr_value,
                "vol_mult": vol_mult,
                "place_time": datetime.now(timezone.utc),
                "expiry_time": datetime.now(timezone.utc) + timedelta(minutes=5)  # Cancel after 5 minutes
            })
            print(f"[LIMIT ORDER PLACED] {side} {qty} @ {order_price:.6f} (orderId: {order['orderId']})")
        except Exception as e:
            print("[ERROR] place limit order:", e)

    async def _open_market_position(self, side: str, entry_price: float, tp_price: float, sl_price: float, atr_value: float, vol_mult: float):

        # raw_qty = compute_qty_by_risk(self.balance, self.cfg.risk_pct, entry_price, sl_price)
        # qty = await round_qty_to_step(self.client, self.cfg.pair, raw_qty)
        # if qty <= 0:
        #     print("[OPEN SKIP] calculated qty <= 0 (minQty or too small).")
        #     return

        qty = 1  # 1 AVAX
        # Calculate actual risk percentage untuk monitoring
        risk_per_trade = abs(entry_price - sl_price) * qty
        risk_percentage = (risk_per_trade / self.balance) * 100

        print(f"[DEBUG] Actual risk: {risk_percentage:.2f}% per trade")

        try:
            # Apply slippage estimate for market orders
            if side == 'LONG':
                exec_price = entry_price * (1 + self.cfg.slippage_pct)
            else:
                exec_price = entry_price * (1 - self.cfg.slippage_pct)
                
            # Place market order
            order = await self.client.futures_create_order(
                symbol=self.cfg.pair,
                side=SIDE_BUY if side == 'LONG' else SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=qty
            )
            
            # Get execution details
            exec_qty = 0.0
            avg_price = 0.0
            if 'fills' in order and order['fills']:
                for fill in order['fills']:
                    exec_qty += float(fill['qty'])
                    avg_price += float(fill['price']) * float(fill['qty'])
                avg_price = avg_price / exec_qty if exec_qty > 0 else exec_price
            else:
                avg_price = exec_price
                exec_qty = qty

            print(f"[POSITION OPENED] {side} {exec_qty} @ {avg_price:.6f}")

            self._current_position = {
                "side": side,
                "entry_price": avg_price,
                "qty": float(exec_qty),
                "tp_price": tp_price,
                "sl_price": sl_price,
                "entry_time": datetime.now(timezone.utc),
                "atr": atr_value,
                "vol_mult": vol_mult
            }
        except Exception as e:
            print("[ERROR] open position:", e)

    async def _check_pending_orders(self):
        # Check and cancel expired pending orders
        now = datetime.now(timezone.utc)
        for order in self.pending_orders[:]:
            if now >= order['expiry_time']:
                try:
                    await self.client.futures_cancel_order(
                        symbol=self.cfg.pair, 
                        orderId=order['order_id']
                    )
                    print(f"[ORDER CANCELED] {order['order_id']} (expired)")
                    self.pending_orders.remove(order)
                except Exception as e:
                    print(f"[ERROR] cancel order {order['order_id']}:", e)

    async def _process_current_position(self):
        if self._current_position is None:
            return

        pos = self._current_position
        latest_candle = self.candles.iloc[-1]
        now = self.candles.index[-1]
        elapsed_sec = (now.to_pydatetime() - pos['entry_time']).total_seconds()

        exit_reason = None
        exit_price = None

        if elapsed_sec >= self.cfg.min_hold_sec:
            if pos['side'] == 'LONG':
                if latest_candle['high'] >= pos['tp_price']:
                    exit_price = pos['tp_price']
                    exit_reason = 'TP'
                elif latest_candle['low'] <= pos['sl_price']:
                    exit_price = pos['sl_price']
                    exit_reason = 'SL'
            else:
                if latest_candle['low'] <= pos['tp_price']:
                    exit_price = pos['tp_price']
                    exit_reason = 'TP'
                elif latest_candle['high'] >= pos['sl_price']:
                    exit_price = pos['sl_price']
                    exit_reason = 'SL'

        if exit_price is not None:
            try:
                side = SIDE_SELL if pos['side'] == 'LONG' else SIDE_BUY
                close_order = await self.client.futures_create_order(
                    symbol=self.cfg.pair,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    reduceOnly=True,
                    quantity=pos['qty']
                )
                
                # Get execution price
                exit_exec_price = 0.0
                exit_qty = 0.0
                if 'fills' in close_order and close_order['fills']:
                    for fill in close_order['fills']:
                        exit_qty += float(fill['qty'])
                        exit_exec_price += float(fill['price']) * float(fill['qty'])
                    exit_exec_price = exit_exec_price / exit_qty if exit_qty > 0 else exit_price
                else:
                    exit_exec_price = exit_price
                    
                # Apply slippage for market exits
                if pos['side'] == 'LONG':
                    exit_exec_price = exit_exec_price * (1 - self.cfg.slippage_pct)
                else:
                    exit_exec_price = exit_exec_price * (1 + self.cfg.slippage_pct)

                # Calculate PnL
                if pos['side'] == 'LONG':
                    raw_pnl = pos['qty'] * (exit_exec_price - pos['entry_price'])
                else:
                    raw_pnl = pos['qty'] * (pos['entry_price'] - exit_exec_price)

                # Simplified fee calculation (like forward test)
                fees = self.cfg.fee_rate * self.balance
                net_pnl = raw_pnl - fees

                # Update balance
                self.balance += net_pnl

                # Log trade
                append_trade_excel(self.cfg.logfile, [
                    self.cfg.pair,
                    pos['entry_time'],
                    pos['side'],
                    pos['entry_price'],
                    pos['qty'],
                    pos['tp_price'],
                    pos['sl_price'],
                    datetime.now(timezone.utc),
                    exit_exec_price,
                    net_pnl,
                    fees,
                    exit_reason,
                    self.balance,
                    pos['atr'],
                    pos['vol_mult']
                ])

                print(f"[POSITION CLOSED] {pos['side']} exit={exit_exec_price:.6f} reason={exit_reason} pnl={net_pnl:.6f} fees={fees:.6f} balance={self.balance:.4f}")
                self._current_position = None
            except Exception as e:
                print("[ERROR] closing position:", e)

# ---------------- Run ----------------
if __name__ == "__main__":
    cfg = BotConfig()
    cfg.api_key = os.getenv('BINANCE_API_KEY')
    cfg.api_secret = os.getenv('BINANCE_API_SECRET')
    cfg.use_testnet = False  # Set to False for live trading
    
    # Improved settings
    cfg.use_limit_orders = True  # Use limit orders for better entry prices
    cfg.require_confirmation = True  # Require candle close confirmation
    cfg.adaptive_tp_sl = True  # Adjust TP/SL based on volatility
    cfg.slippage_pct = 0.001  # 0.1% estimated slippage

    init_excel(cfg.logfile)
    bot = ImprovedLiveDualEntryBot(cfg)

    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        print("[STOP] Interrupted by user. Trades saved to", cfg.logfile)
    finally:
        try:
            if bot.client:
                loop.run_until_complete(bot.client.close_connection())
        except Exception:
            pass