import os
import pandas as pd
import numpy as np
import talib
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

# ---------------- Config yang sama persis ----------------
@dataclass
class BotConfig:
    pair: str = "AVAXUSDT"
    interval: str = "5m"
    initial_balance: float = 20.0
    leverage: float = 10.0
    fee_rate: float = 0.0004
    min_atr: float = 0.01
    atr_period: int = 14
    level_mult: float = 1.0
    tp_atr_mult: float = 3.0
    sl_atr_mult: float = 2.5
    monitor_candles: int = 3
    candles_buffer: int = 1000
    risk_pct: float = 0.008  # 0.8%
    qty_precision: int = 1
    price_precision: int = 3
    daily_profit_lock_pct: float = 1.0
    require_confirmation: bool = True
    adaptive_tp_sl: bool = True

# ---------------- Helper Functions ----------------
def compute_atr_from_df(df: pd.DataFrame, period: int):
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr

def compute_volatility_ratio(df: pd.DataFrame, atr_period: int = 14):
    atr = compute_atr_from_df(df, atr_period)
    current_atr = atr.iat[-1] if len(atr) > 0 else 0
    price = df['close'].iat[-1] if len(df) > 0 else 1
    return current_atr / price if price > 0 else 0

def round_price(price: float, precision: int) -> float:
    return round(price, precision)

def round_qty(qty: float, precision: int) -> float:
    return round(qty, precision)

# ---------------- Backtest Class ----------------
class BacktestBot:
    def __init__(self, cfg: BotConfig, df: pd.DataFrame):
        self.cfg = cfg
        self.df = df.copy()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], utc=True)
        self.df.set_index('timestamp', inplace=True)
        self.df.sort_index(inplace=True)
        
        # State
        self.balance = cfg.initial_balance
        self.candles = pd.DataFrame()
        self.watches = []
        self.trades = []
        self._current_position = None
        self.daily_start_equity = cfg.initial_balance
        self.daily_realized_pct = 0.0
        self.trade_locked = False

    def _append_candle(self, row):
        self.candles = pd.concat([self.candles, pd.DataFrame([row])])
        if len(self.candles) > self.cfg.candles_buffer:
            self.candles = self.candles.iloc[-self.cfg.candles_buffer:]
        # Hitung indikator
        if len(self.candles) >= 50:
            close = self.candles['close'].values
            high = self.candles['high'].values
            low = self.candles['low'].values
            self.candles.loc[row.name, 'ema_fast'] = talib.EMA(close, 20)[-1]
            self.candles.loc[row.name, 'ema_slow'] = talib.EMA(close, 50)[-1]
            self.candles.loc[row.name, 'rsi'] = talib.RSI(close, 14)[-1] if len(close) >= 14 else 50
            self.candles.loc[row.name, 'adx'] = talib.ADX(high, low, close, 14)[-1]
            self.candles.loc[row.name, 'ATR'] = talib.ATR(high, low, close, 14)[-1]
            self.candles.loc[row.name, 'plusDI'] = talib.PLUS_DI(high, low, close, 14)[-1]
            self.candles.loc[row.name, 'minusDI'] = talib.MINUS_DI(high, low, close, 14)[-1]

    def enhanced_trend_detection(self):
        if len(self.candles) < 30:
            return {"regime": "MIXED", "confidence": 0.3, "trend_direction": "NEUTRAL"}
        try:
            plus_di = self.candles['plusDI'].iloc[-1]
            minus_di = self.candles['minusDI'].iloc[-1]
            adx = self.candles['adx'].iloc[-1]
            ema_fast = self.candles['ema_fast'].iloc[-1]
            ema_slow = self.candles['ema_slow'].iloc[-1]
        except:
            return {"regime": "MIXED", "confidence": 0.3, "trend_direction": "NEUTRAL"}

        trend_score = 0
        max_score = 6

        if adx > 22:
            trend_score += 2
        elif adx > 16:
            trend_score += 1

        if ema_fast > ema_slow:
            trend_score += 2
        elif ema_fast < ema_slow:
            trend_score += 2

        di_diff = plus_di - minus_di
        if abs(di_diff) > 12:
            trend_score += 2
        elif abs(di_diff) > 6:
            trend_score += 1

        confidence = trend_score / max_score
        if trend_score >= 4 and confidence > 0.5:
            if plus_di > minus_di:
                return {"regime": "STRONG_TREND", "confidence": confidence, "trend_direction": "UP"}
            else:
                return {"regime": "STRONG_TREND", "confidence": confidence, "trend_direction": "DOWN"}
        elif trend_score <= 2:
            return {"regime": "SIDEWAYS", "confidence": confidence, "trend_direction": "NEUTRAL"}
        else:
            return {"regime": "MIXED", "confidence": confidence, "trend_direction": "NEUTRAL"}

    def _create_watch(self, atr_value):
        if self._current_position is not None:
            return
        last_close = self.candles['close'].iat[-1]
        volatility_mult = 1.0
        if self.cfg.adaptive_tp_sl and self.volatility_ratio > 0.005:
            volatility_mult = 1.2
        elif self.cfg.adaptive_tp_sl and self.volatility_ratio < 0.002:
            volatility_mult = 0.8
        watch = {
            "start_idx": len(self.candles) - 1,
            "expire_idx": len(self.candles) - 1 + self.cfg.monitor_candles,
            "long_level": round_price(last_close + atr_value * self.cfg.level_mult * volatility_mult, self.cfg.price_precision),
            "short_level": round_price(last_close - atr_value * self.cfg.level_mult * volatility_mult, self.cfg.price_precision),
            "atr": atr_value,
            "trigger_time": self.candles.index[-1],
            "volatility_mult": volatility_mult
        }
        self.watches.append(watch)

    def _process_watches(self):
        if self._current_position is not None or self.trade_locked:
            return

        market_analysis = self.enhanced_trend_detection()
        regime = market_analysis['regime']
        trend_direction = market_analysis['trend_direction']
        confidence = market_analysis['confidence']

        size_multiplier = 1.0
        if size_multiplier == 0:
            return

        latest_idx = len(self.candles) - 1
        candle_high = self.candles['high'].iat[-1]
        candle_low = self.candles['low'].iat[-1]
        candle_close = self.candles['close'].iat[-1]

        new_watches = []
        for w in self.watches:
            if latest_idx < w['start_idx']:
                new_watches.append(w)
                continue

            triggered = False
            side = None
            entry_price = None
            tp_price = None
            sl_price = None

            # --- Konfirmasi Volume di STRONG_TREND ---
            use_confirmation = (regime == "STRONG_TREND")
            if use_confirmation:
                avg_vol = self.candles['volume'].tail(10).mean() if len(self.candles) >= 10 else 1
                current_vol = self.candles['volume'].iat[-1]
                long_condition = (candle_close >= w['long_level'] and current_vol > avg_vol * 1.2)
                short_condition = (candle_close <= w['short_level'] and current_vol > avg_vol * 1.2)
            else:
                long_condition = candle_close >= w['long_level']
                short_condition = candle_close <= w['short_level']

            if long_condition:
                triggered = True
                side = 'LONG'
                entry_price = w['long_level']
                tp_mult = self.cfg.tp_atr_mult * w['volatility_mult']
                sl_mult = self.cfg.sl_atr_mult * w['volatility_mult']
                tp_price = round_price(entry_price + w['atr'] * tp_mult, self.cfg.price_precision)
                sl_price = round_price(entry_price - w['atr'] * sl_mult, self.cfg.price_precision)
            elif short_condition:
                triggered = True
                side = 'SHORT'
                entry_price = w['short_level']
                tp_mult = self.cfg.tp_atr_mult * w['volatility_mult']
                sl_mult = self.cfg.sl_atr_mult * w['volatility_mult']
                tp_price = round_price(entry_price - w['atr'] * tp_mult, self.cfg.price_precision)
                sl_price = round_price(entry_price + w['atr'] * sl_mult, self.cfg.price_precision)

            if triggered:
                # --- Filter S/R berdasarkan regime ---
                valid_sr = False
                if regime == "STRONG_TREND":
                    if trend_direction == "DOWN" and side == "SHORT":
                        recent_high = max(
                            self.candles['high'].iloc[-1],
                            self.candles['high'].iloc[-2] if len(self.candles) > 1 else self.candles['high'].iloc[-1],
                            self.candles['high'].iloc[-3] if len(self.candles) > 2 else self.candles['high'].iloc[-1]
                        )
                        max_dist = w['atr'] * 2.0
                        if abs(entry_price - recent_high) <= max_dist:
                            valid_sr = True
                    elif trend_direction == "UP" and side == "LONG":
                        recent_low = min(
                            self.candles['low'].iloc[-1],
                            self.candles['low'].iloc[-2] if len(self.candles) > 1 else self.candles['low'].iloc[-1],
                            self.candles['low'].iloc[-3] if len(self.candles) > 2 else self.candles['low'].iloc[-1]
                        )
                        max_dist = w['atr'] * 2.0
                        if abs(entry_price - recent_low) <= max_dist:
                            valid_sr = True
                    else:
                        valid_sr = False
                else:
                    # SIDEWAYS / MIXED: skip untuk backtest sederhana
                    valid_sr = True

                if not valid_sr:
                    if latest_idx < w['expire_idx']:
                        new_watches.append(w)
                    continue

                # Hitung qty berdasarkan risk
                risk_per_unit = abs(entry_price - sl_price)
                if risk_per_unit == 0:
                    if latest_idx < w['expire_idx']:
                        new_watches.append(w)
                    continue
                qty = (self.balance * self.cfg.risk_pct) / risk_per_unit
                qty = round_qty(qty, self.cfg.qty_precision)
                if qty <= 0:
                    if latest_idx < w['expire_idx']:
                        new_watches.append(w)
                    continue

                # Simpan posisi
                self._current_position = {
                    'side': side,
                    'entry_price': entry_price,
                    'qty': qty,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'atr': w['atr'],
                    'entry_time': self.candles.index[-1],
                    'status': 'OPEN'
                }
                break
            else:
                if latest_idx < w['expire_idx']:
                    new_watches.append(w)
        self.watches = new_watches

    def _check_exit(self, current_price, current_time):
        if self._current_position is None:
            return False
        pos = self._current_position
        if pos['side'] == 'LONG':
            if current_price >= pos['tp_price']:
                self._close_position(current_price, current_time, "TP")
                return True
            elif current_price <= pos['sl_price']:
                self._close_position(current_price, current_time, "SL")
                return True
        else:
            if current_price <= pos['tp_price']:
                self._close_position(current_price, current_time, "TP")
                return True
            elif current_price >= pos['sl_price']:
                self._close_position(current_price, current_time, "SL")
                return True
        return False

    def _close_position(self, exit_price, exit_time, reason):
        pos = self._current_position
        if pos['side'] == 'LONG':
            raw_pnl = pos['qty'] * (exit_price - pos['entry_price'])
        else:
            raw_pnl = pos['qty'] * (pos['entry_price'] - exit_price)
        fees = self.cfg.fee_rate * self.balance
        net_pnl = raw_pnl - fees
        self.balance += net_pnl
        self.daily_realized_pct = (self.balance - self.daily_start_equity) / self.daily_start_equity * 100

        self.trades.append({
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'side': pos['side'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'qty': pos['qty'],
            'pnl': net_pnl,
            'fees': fees,
            'reason': reason,
            'balance_after': self.balance
        })
        self._current_position = None

    def _simulate_limit_fill(self, order_price: float, side: str, candle: pd.Series) -> bool:
        """Simulasi apakah limit order terisi berdasarkan high/low candle"""
        if side == "LONG":
            # Limit buy terisi jika low <= order_price
            return candle['low'] <= order_price
        else:
            # Limit sell terisi jika high >= order_price
            return candle['high'] >= order_price

    def _get_fill_price(self, order_price: float, side: str, candle: pd.Series) -> float:
        """Dapatkan harga eksekusi realistis"""
        if side == "LONG":
            # Eksekusi di max(order_price, open) — asumsi conservative
            return max(order_price, candle['open'])
        else:
            return min(order_price, candle['open'])

    def run(self):
        print(f"Starting backtest on {len(self.df)} candles...")
        for i, (ts, row) in enumerate(self.df.iterrows()):
            self._append_candle(row)
            if len(self.candles) < 50:
                continue

            self.volatility_ratio = compute_volatility_ratio(self.candles, self.cfg.atr_period)

            # Cek daily reset (setiap jam 7 UTC)
            if ts.hour == 7 and (i == 0 or self.df.index[i-1].date() != ts.date()):
                self.daily_start_equity = self.balance
                self.daily_realized_pct = 0.0
                self.trade_locked = False

            # Lock jika profit harian tercapai
            if self.daily_realized_pct >= self.cfg.daily_profit_lock_pct:
                self.trade_locked = True

            # Tutup posisi jika ada
            if self._current_position:
                if self._check_exit(row['close'], ts):
                    continue

            # Buat watch jika ATR cukup
            atr_series = compute_atr_from_df(self.candles, self.cfg.atr_period)
            current_atr = atr_series.iat[-1] if len(atr_series) >= self.cfg.atr_period else np.nan
            if not np.isnan(current_atr) and current_atr >= self.cfg.min_atr:
                self._create_watch(current_atr)

            # Proses watch
            self._process_watches()

        print(f"Backtest selesai. Balance akhir: ${self.balance:.4f}")
        return pd.DataFrame(self.trades)

# ---------------- Main ----------------
if __name__ == "__main__":
    df = pd.read_csv("/home/julbot/trade-control-system/backtest_by_data/AVAXUSDT_5m.csv")
    cfg = BotConfig()
    bot = BacktestBot(cfg, df)
    trades_df = bot.run()
    trades_df.to_excel("backtest_trades.xlsx", index=False)
    print(f"Total trades: {len(trades_df)}")
    if len(trades_df) > 0:
        win_rate = (trades_df['pnl'] > 0).mean() * 100
        total_pnl = trades_df['pnl'].sum()
        print(f"Win rate: {win_rate:.1f}% | Total PnL: ${total_pnl:.4f}")