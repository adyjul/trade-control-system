import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import time
import math
from functools import lru_cache
import json
import os
import openpyxl
import requests
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG UTAMA (Gunakan konfig dari backtest sebagai base) ---
INITIAL_BALANCE = 20.0
LEVERAGE = 10
# TP_ATR_MULT = 3.0
TIMEFRAME = '15m'

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '') 
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')  
ENABLE_TELEGRAM = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

if TIMEFRAME == '15m':
    ATR_WINDOW = 14
    VOLATILITY_ADJUSTMENT = 1.8  # Faktor peningkatan volatilitas
    
    # Entry & Exit
    TP_ATR_MULT = 5.5
    SL_ATR_MULT = 3.0
    ZONE_START_FACTOR = 1.0
    ZONE_END_FACTOR = 1.8
    ENTRY_ZONE_BUFFER = 0.025  # 2.5%
    ZONE_BUFFER_MULTIPLIER = 1.5
    
    # Risk Management
    BASE_RISK_PCT = 0.0075
    MAX_RISK_PER_TRADE = 0.015
    MAX_POSITION_VALUE = 0.35
    MIN_POSITION_VALUE = 1.5
    
    # MTF Confirmation
    MTF_MIN_SCORE = 0.3
    MTF_DIRECTION_THRESHOLD = 0.1
    
    # Interval Timing
    DATA_UPDATE_INTERVAL = 900 
    RESCAN_INTERVAL_MINUTES = 60
    MIN_TIME_BETWEEN_SCANS = 30
    
    # Volume & Momentum
    VOLUME_WINDOW = 10  # Lebih panjang untuk smoothing
    MIN_MOMENTUM_STRENGTH = 0.20  # Lebih longgar

    BARS_TO_FETCH = 300
    VOLATILITY_WINDOW = 75
    MIN_SCAN_SCORE = 0.45
    SLIPPAGE_RATE = 0.0007
    MAX_SLIPPAGE_RATE = 0.0015 
    ORDER_BOOK_DEPTH = 25
    OI_UPDATE_INTERVAL = 900

else:  # 5m
    TP_ATR_MULT = 4.5
    SL_ATR_MULT = 2.5
    ATR_WINDOW = 14
    BASE_RISK_PCT = 0.01
    MAX_RISK_PER_TRADE = 0.02
    MAX_POSITION_VALUE = 0.50
    MIN_POSITION_VALUE = 1.0
    ZONE_START_FACTOR = 0.8
    ZONE_END_FACTOR = 1.2
    ZONE_BUFFER_MULTIPLIER = 1.5
    RESCAN_INTERVAL_MINUTES = 30 
    BARS_TO_FETCH = 500
    VOLATILITY_WINDOW = 50
    MIN_SCAN_SCORE = 0.5
    SLIPPAGE_RATE = 0.0005  # 0.05%
    MAX_SLIPPAGE_RATE = 0.001  # 0.1%
    MIN_TIME_BETWEEN_SCANS = 15
    ORDER_BOOK_DEPTH = 20 
    OI_UPDATE_INTERVAL = 300



# --- CONFIG FORWARD TEST ---
MODE = 'simulated'  # 'live' atau 'simulated'
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')
ORDER_TYPE = 'market'
ORDER_BOOK_DEPTH = 20 # Interval scanning ulang (menit)
# MIN_TIME_BETWEEN_SCANS = 15   # Minimal waktu antar scan (menit) untuk mencegah thrashing

# --- MARKET SCANNER CLASS (diambil dari backtest) ---
class MarketScanner:
    def __init__(self):
        if MODE == 'live':
            self.exchange = ccxt.binance({
                'apiKey': API_KEY,
                'secret': API_SECRET,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        else:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        
        self.min_volume_usd = 300000
        self.min_24h_change = 1.0
        self.max_symbols = 20
        self.last_request_time = 0
        self.request_delay = 0.5
        self.default_profiles = {
            'MAJOR': {'atr_threshold': 0.20, 'volume_multiplier': 1.5, 'level_multiplier': 0.6, 'risk_pct': 0.008},
            'MID_CAP': {'atr_threshold': 0.25, 'volume_multiplier': 1.6, 'level_multiplier': 0.7, 'risk_pct': 0.010},
            'MEME': {'atr_threshold': 0.35, 'volume_multiplier': 1.8, 'level_multiplier': 0.8, 'risk_pct': 0.006},
            'SMALL_CAP': {'atr_threshold': 0.40, 'volume_multiplier': 2.0, 'level_multiplier': 0.9, 'risk_pct': 0.005},
            'DEFAULT': {'atr_threshold': 0.25, 'volume_multiplier': 1.6, 'level_multiplier': 0.7, 'risk_pct': 0.010}
        }
        self.asset_classification = {
            'BTC': 'MAJOR', 'ETH': 'MAJOR', 'BNB': 'MAJOR',
            'SOL': 'MID_CAP', 'AVAX': 'MID_CAP', 'MATIC': 'MID_CAP', 'LINK': 'MID_CAP',
            'ICP': 'MID_CAP', 'INJ': 'MID_CAP', 'TON': 'MID_CAP', 'ARB': 'MID_CAP',
            'DOGE': 'MEME', 'SHIB': 'MEME', 'PEPE': 'MEME', 'FLOKI': 'MEME',
            'SEI': 'SMALL_CAP', 'SUI': 'SMALL_CAP', 'APT': 'SMALL_CAP'
        }

    def _rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def detect_market_regime(self, df):
        if len(df) < 200:
            return "NEUTRAL"
        current_price = df['close'].iloc[-1]
        ema_200 = df['close'].rolling(200).mean().iloc[-1]
        if pd.isna(ema_200) or ema_200 == 0:
            return "NEUTRAL"
        if current_price > ema_200 * 1.15:
            return "STRONG_BULL"
        elif current_price > ema_200 * 1.05:
            return "BULL"
        elif current_price < ema_200 * 0.85:
            return "STRONG_BEAR"
        elif current_price < ema_200 * 0.95:
            return "BEAR"
        else:
            return "NEUTRAL"

    def get_asset_profile(self, symbol, df):
        if len(df) < 50:
            return self.get_default_asset_profile(symbol)
        try:
            if 'atr' not in df.columns or df['atr'].isna().all():
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            avg_atr_pct = df['atr_pct'].rolling(VOLATILITY_WINDOW).mean().iloc[-1]
            if pd.isna(avg_atr_pct) or avg_atr_pct == 0:
                return self.get_default_asset_profile(symbol)

            df['vol_ma20'] = df['volume'].rolling(20).mean()
            df['vol_ma100'] = df['volume'].rolling(100).mean()
            avg_volume_20 = df['vol_ma20'].iloc[-1]
            avg_volume_100 = df['vol_ma100'].iloc[-1]

            if avg_volume_100 == 0 or pd.isna(avg_volume_100):
                volume_consistency = 1.0
            else:
                volume_consistency = avg_volume_20 / avg_volume_100

            df['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
            avg_range_pct = df['range_pct'].rolling(50).mean().iloc[-1]
            if pd.isna(avg_range_pct):
                avg_range_pct = 0.35

            asset_profile = {
                'volatility_class': self.classify_volatility(avg_atr_pct),
                'volume_class': self.classify_volume(volume_consistency, avg_volume_20),
                'range_class': self.classify_range(avg_range_pct),
                'avg_atr_pct': avg_atr_pct,
                'avg_range_pct': avg_range_pct,
                'volume_consistency': volume_consistency,
                'symbol_base': symbol.split('/')[0].upper(),
                'source': 'DYNAMIC_CALCULATION'
            }
            print(f"📈 {symbol} Asset Profile (Dinamis): {asset_profile['volatility_class']} ({avg_atr_pct:.3f}%)")
            return asset_profile
        except Exception as e:
            print(f"🔥 Error menghitung asset profile untuk {symbol}: {e}")
            return self.get_default_asset_profile(symbol)

    def get_default_asset_profile(self, symbol):
        symbol_base = symbol.split('/')[0].upper()
        asset_class = self.asset_classification.get(symbol_base, 'DEFAULT')
        profile = self.default_profiles[asset_class].copy()
        profile.update({
            'symbol_base': symbol_base,
            'asset_class': asset_class,
            'source': 'DEFAULT_FALLBACK'
        })
        print(f"🔄 Default Profile untuk {symbol}: {asset_class}")
        return profile

    def classify_volatility(self, avg_atr_pct):
        if TIMEFRAME == '15m':
            if avg_atr_pct < 0.25: return 'ULTRA_LOW'
            elif avg_atr_pct < 0.40: return 'LOW'
            elif avg_atr_pct < 0.70: return 'MODERATE'
            elif avg_atr_pct < 1.10: return 'HIGH'
            elif avg_atr_pct < 1.50: return 'VERY_HIGH'
            else: return 'EXTREME'
        else:  # 5m
            if avg_atr_pct < 0.15: return 'ULTRA_LOW'
            elif avg_atr_pct < 0.25: return 'LOW'
            elif avg_atr_pct < 0.40: return 'MODERATE'
            elif avg_atr_pct < 0.60: return 'HIGH'
            elif avg_atr_pct < 0.85: return 'VERY_HIGH'
            else: return 'EXTREME'

    def classify_volume(self, volume_consistency, avg_volume):
        if volume_consistency < 0.7: return 'VOLATILE'
        elif volume_consistency > 1.3: return 'SPIKY'
        else:
            if avg_volume < 5000: return 'LOW_LIQUIDITY'
            elif avg_volume < 20000: return 'MEDIUM_LIQUIDITY'
            else: return 'HIGH_LIQUIDITY'

    def classify_range(self, avg_range_pct):
        if avg_range_pct < 0.15: return 'ULTRA_LOW'
        elif avg_range_pct < 0.25: return 'LOW'
        elif avg_range_pct < 0.40: return 'MODERATE'
        elif avg_range_pct < 0.60: return 'HIGH'
        elif avg_range_pct < 0.85: return 'VERY_HIGH'
        else: return 'EXTREME'

    def get_dynamic_entry_thresholds(self, asset_profile, market_regime):
        if asset_profile['source'] == 'DEFAULT_FALLBACK':
            base_thresholds = {
                'atr_threshold': asset_profile['atr_threshold'],
                'volume_multiplier': asset_profile['volume_multiplier'],
                'level_multiplier': asset_profile['level_multiplier'],
                'adx_threshold': 18,
                'risk_pct': asset_profile['risk_pct']
            }
        else:
            volatility_map = {
                'ULTRA_LOW': 0.12, 'LOW': 0.18, 'MODERATE': 0.25, 'HIGH': 0.35, 'VERY_HIGH': 0.45, 'EXTREME': 0.60
            }
            volume_map = {
                'VOLATILE': 1.4, 'SPIKY': 1.6, 'LOW_LIQUIDITY': 2.0, 'MEDIUM_LIQUIDITY': 1.7, 'HIGH_LIQUIDITY': 1.5
            }
            range_map = {
                'ULTRA_LOW': 0.5, 'LOW': 0.6, 'MODERATE': 0.7, 'HIGH': 0.8, 'VERY_HIGH': 0.9, 'EXTREME': 1.0
            }
            base_thresholds = {
                'atr_threshold': volatility_map.get(asset_profile['volatility_class'], 0.25),
                'volume_multiplier': volume_map.get(asset_profile['volume_class'], 1.7),
                'level_multiplier': range_map.get(asset_profile['range_class'], 0.7),
                'adx_threshold': 18 if asset_profile['volatility_class'] in ['LOW', 'ULTRA_LOW'] else 22,
                'risk_pct': BASE_RISK_PCT
            }

        regime_adjustment = {
            'STRONG_BULL': {'atr_threshold': 0.85, 'volume_multiplier': 0.8, 'adx_threshold': 0.85},
            'BULL': {'atr_threshold': 0.9, 'volume_multiplier': 0.85, 'adx_threshold': 0.9},
            'NEUTRAL': {'atr_threshold': 1.0, 'volume_multiplier': 1.0, 'adx_threshold': 1.0},
            'BEAR': {'atr_threshold': 1.1, 'volume_multiplier': 1.15, 'adx_threshold': 1.05},
            'STRONG_BEAR': {'atr_threshold': 1.2, 'volume_multiplier': 1.25, 'adx_threshold': 1.1}
        }.get(market_regime, {'atr_threshold': 1.0, 'volume_multiplier': 1.0, 'adx_threshold': 1.0})

        thresholds = {
            'atr_threshold': base_thresholds['atr_threshold'] * regime_adjustment['atr_threshold'],
            'volume_multiplier': base_thresholds['volume_multiplier'] * regime_adjustment['volume_multiplier'],
            'level_multiplier': base_thresholds['level_multiplier'],
            'adx_threshold': base_thresholds['adx_threshold'] * regime_adjustment['adx_threshold'],
            'risk_pct': base_thresholds['risk_pct']
        }

        print(f"⚙️ Dynamic Thresholds untuk {asset_profile['symbol_base']} di {market_regime}:")
        print(f"   ATR Threshold: {thresholds['atr_threshold']:.3f}%")
        print(f"   Volume Multiplier: {thresholds['volume_multiplier']:.2f}x")
        print(f"   Level Multiplier: {thresholds['level_multiplier']:.2f}")
        print(f"   ADX Threshold: {thresholds['adx_threshold']:.1f}")
        print(f"   Risk Percentage: {thresholds['risk_pct']:.3f}%")
        
        return thresholds
    

    def get_trending_symbols(self):
        print("🔍 MENGAMBIL DAFTAR ASET TRENDING DARI BINANCE...")
        self._rate_limit()
        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = {}
            for symbol, data in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                if ('quoteVolume' not in data or 'percentage' not in data or
                    data['quoteVolume'] is None or data['percentage'] is None):
                    continue
                if (data['quoteVolume'] > self.min_volume_usd and
                    abs(data['percentage']) > self.min_24h_change and
                    data['last'] > 0.001):
                    usdt_pairs[symbol] = data

            if not usdt_pairs:
                print("⚠️ Tidak ada aset yang memenuhi kriteria volume dan pergerakan")
                return self.get_default_symbols()

            df = pd.DataFrame.from_dict(usdt_pairs, orient='index')
            df['symbol'] = df.index
            df['volume_usd'] = df['quoteVolume']
            df['change_24h'] = df['percentage']
            df['price'] = df['last']
            df = df[abs(df['change_24h']) < 50.0]
            df = df[df['volume_usd'] > self.min_volume_usd]
            top_symbols = df.sort_values('volume_usd', ascending=False).head(self.max_symbols)
            print(f"✅ BERHASIL MENDAPATKAN {len(top_symbols)} ASET TRENDING:")
            for i, (_, row) in enumerate(top_symbols.iterrows()):
                print(f"   #{i+1} {row['symbol']} | Vol: ${row['volume_usd']:,.0f} | Chg: {row['change_24h']:+.2f}% | P: ${row['price']:.4f}")
            return top_symbols['symbol'].tolist()
        except Exception as e:
            print(f"❌ Error mengambil data dari Binance: {e}")
            print("🔄 Menggunakan daftar aset default...")
            return self.get_default_symbols()

    def get_default_symbols(self):
        default_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'AVAX/USDT',
            'DOGE/USDT', 'XRP/USDT', 'ADA/USDT', 'MATIC/USDT', 'LINK/USDT',
            'INJ/USDT', 'TON/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ARB/USDT',
            'ICP/USDT', 'NEAR/USDT', 'OP/USDT', 'APT/USDT', 'SEI/USDT'
        ]
        return default_symbols[:self.max_symbols]

    def get_orderbook_sentiment(self, symbol):
        self._rate_limit()
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
            bid_volume = sum(bid[1] for bid in ob['bids'][:10])
            ask_volume = sum(ask[1] for ask in ob['asks'][:10])
            if bid_volume + ask_volume == 0:
                return 50.0
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            sentiment_score = 50 + (imbalance * 50)
            return max(0, min(100, sentiment_score))
        except Exception as e:
            print(f"⚠️ Error order book {symbol}: {e}")
            return 50.0

    def rank_symbols_by_activity(self, symbols):
        print("⚡ MENGANALISIS AKTIVITAS PASAR TERKINI...")
        ranked_symbols = []
        for symbol in symbols:
            try:
                print(f"📊 Menganalisis {symbol}...")
                self._rate_limit()
                ohlcv = self.exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=15)
                if len(ohlcv) < 15:
                    print(f"   ✗ {symbol}: Data tidak cukup ({len(ohlcv)} candle)")
                    continue
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)

                if len(df) >= 12:
                    df['price_change_pct'] = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1) * 100
                else:
                    df['price_change_pct'] = 0
                df['volume_ma'] = df['volume'].rolling(5).mean()
                df['volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)

                ob_sentiment = self.get_orderbook_sentiment(symbol)

                price_change_score = abs(df['price_change_pct'].iloc[-1]) * 2
                price_score = min(100, max(0, price_change_score))

                if len(df) >= 5:
                    volume_score = (df['volume'].iloc[-1] / df['volume'].mean()) * 12
                    volume_score = min(100, max(0, volume_score))
                else:
                    volume_score = 50

                if len(df) >= 14:
                    try:
                        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                        adx_score = min(100, max(0, df['adx'].iloc[-1] * 3))
                    except:
                        adx_score = 50
                else:
                    adx_score = 50

                directional_bias = self.calculate_directional_bias(symbol, df)
                activity_score = (
                    ob_sentiment * 0.30 +
                    price_score * 0.20 +
                    volume_score * 0.15 +
                    adx_score * 0.15 +
                    ((directional_bias + 1) / 2) * 0.20
                )
                # activity_score = (
                #     ob_sentiment * 0.35 +
                #     price_score * 0.25 +
                #     volume_score * 0.20 +
                #     adx_score * 0.20
                # )
                spike_count = int(df['volume_spike'].sum()) if 'volume_spike' in df else 0
                ranked_symbols.append({
                    'symbol': symbol,
                    'activity_score': activity_score,
                    'price_change': df['price_change_pct'].iloc[-1],
                    'volume_total': df['volume'].iloc[-1],
                    'spike_count': spike_count,
                    'ob_sentiment': ob_sentiment,
                    'volume_score': volume_score,
                    'adx_score': adx_score
                })
                print(f"   ✓ {symbol} - Skor: {activity_score:.1f} | OB Sent: {ob_sentiment:.1f}")
            except Exception as e:
                print(f"   ✗ Error menganalisis {symbol}: {e}")
                continue

        ranked_symbols.sort(key=lambda x: x['activity_score'], reverse=True)
        print("🏆 HASIL PERINGKAT AKTIVITAS PASAR:")
        for i, asset in enumerate(ranked_symbols[:8]):
            print(f"  #{i+1} {asset['symbol']:<10} | Skor: {asset['activity_score']:5.1f} | OB Sent: {asset['ob_sentiment']:4.1f}")
        return ranked_symbols

    def get_best_asset_for_trading(self):
        trending_symbols = self.get_trending_symbols()
        ranked_assets = self.rank_symbols_by_activity(trending_symbols)
        if not ranked_assets:
            print("❌ Tidak ada aset yang bisa dianalisis!")
            return None
        qualified_assets = [asset for asset in ranked_assets if asset['activity_score'] >= 40.0]
        if qualified_assets:
            best_asset = qualified_assets[0]
            print(f"🎯 ASET TERBAIK: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']
        else:
            best_asset = ranked_assets[0]
            print(f"⚠️ Tidak ada aset qualified, menggunakan: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']

    def calculate_market_score(self, df, btc_df=None):
        if len(df) < 50:
            print("❌ Data tidak cukup untuk analisis market score")
            return 0.0
        try:
            if 'atr' not in df.columns or df['atr'].isna().all():
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            df['avg_atr'] = df['atr'].rolling(50).mean()
            current_atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            avg_atr_pct = (df['avg_atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            if 0.15 <= current_atr_pct <= 0.60:
                vol_score = 1.0
            elif current_atr_pct < 0.10 or current_atr_pct > 1.0:
                vol_score = 0.0
            else:
                vol_score = 0.5

            df['vol_ma20'] = df['volume'].rolling(20).mean()
            df['vol_ma100'] = df['volume'].rolling(100).mean()
            if df['vol_ma100'].iloc[-1] == 0 or pd.isna(df['vol_ma100'].iloc[-1]):
                vol_ma_score = 0.5
            else:
                current_vol_ratio = df['vol_ma20'].iloc[-1] / df['vol_ma100'].iloc[-1]
                vol_ma_score = min(1.0, max(0.0, current_vol_ratio / 1.5))

            if len(df) >= 14:
                df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                adx = df['adx'].iloc[-1]
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]
                trend_score = 0
                if adx > 20: trend_score += 0.3
                elif adx > 15: trend_score += 0.2
                di_diff = plus_di - minus_di
                if abs(di_diff) > 10: trend_score += 0.2
                elif abs(di_diff) > 5: trend_score += 0.1
                if (plus_di > minus_di and plus_di > 25) or (minus_di > plus_di and minus_di > 25):
                    trend_score += 0.1
            else:
                trend_score = 0.2

            total_score = (vol_score * 0.4 + vol_ma_score * 0.3 + trend_score * 0.3)
            total_score = min(1.0, max(0.0, total_score))
            return total_score
        except Exception as e:
            print(f"🔥 Error calculating market score: {e}")
            return 0.4

    def calculate_directional_bias(self, symbol, df):
        """Hitung probabilitas bullish/bearish untuk aset"""
        if len(df) < 50:
            return 0.0  # Netral
        
        # 1. Momentum dari RSI + MACD
        rsi = df['rsi'].iloc[-1]
        rsi_bias = (70 - rsi) / 40 if rsi < 50 else (70 - rsi) / 20  # -1 (bear) hingga +1 (bull)
        
        # 2. Trend dari ADX + DI
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        trend_strength = adx / 25  # Normalisasi 0-1
        di_diff = (plus_di - minus_di) / max(plus_di, minus_di, 1)
        trend_bias = trend_strength * di_diff
        
        # 3. Order Book Sentiment (existing)
        ob_sentiment = self.get_orderbook_sentiment(symbol)
        ob_bias = (ob_sentiment - 50) / 50  # Normalisasi -1 hingga +1
        
        # 4. Volume Profile
        vol_ratio = df['volume'].iloc[-1] / df['volume_ma20'].iloc[-1]
        if vol_ratio > 1.5 and df['close'].iloc[-1] > df['open'].iloc[-1]:
            vol_bias = 0.3  # Volume beli kuat
        elif vol_ratio > 1.5 and df['close'].iloc[-1] < df['open'].iloc[-1]:
            vol_bias = -0.3  # Volume jual kuat
        else:
            vol_bias = 0
        
        # Kombinasikan semua faktor
        directional_score = (
            rsi_bias * 0.25 +
            trend_bias * 0.35 +
            ob_bias * 0.25 +
            vol_bias * 0.15
        )
        return max(-1.0, min(1.0, directional_score))

    def is_market_hot(self, score):
        return score >= MIN_SCAN_SCORE

# --- FUNGSI TAMBAHAN UNTUK FORWARD TEST ---

def get_current_price(exchange, symbol):
    """Ambil harga terakhir dari ticker atau orderbook"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        # print(f"💰 Harga terakhir: {ticker['last']:.4f}")
        return ticker['last']
    except:
        try:
            ob = exchange.fetch_order_book(symbol, limit=1)
            bid = ob['bids'][0][0] if ob['bids'] else None
            ask = ob['asks'][0][0] if ob['asks'] else None
            if bid and ask:
                return (bid + ask) / 2
            else:
                return None
        except:
            return None

def get_multi_timeframe_confirmation(exchange, symbol):
    """Dapatkan konfirmasi tren dari multiple timeframe"""
    higher_timeframes = {
        '1h': '1h',
        '4h' : '4h',
    }
    
    trend_scores = []
    trend_directions = []
    
    for tf_name, tf_value in higher_timeframes.items():
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf_value, limit=50)
            if len(ohlcv) < 30:
                continue
                
            tf_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            tf_df['timestamp'] = pd.to_datetime(tf_df['timestamp'], unit='ms')
            
            # Hitung indikator dasar
            tf_df['ema20'] = talib.EMA(tf_df['close'], 20)
            tf_df['ema50'] = talib.EMA(tf_df['close'], 50)
            tf_df['adx'] = talib.ADX(tf_df['high'], tf_df['low'], tf_df['close'], 14)
            tf_df['plus_di'] = talib.PLUS_DI(tf_df['high'], tf_df['low'], tf_df['close'], 14)
            tf_df['minus_di'] = talib.MINUS_DI(tf_df['high'], tf_df['low'], tf_df['close'], 14)
            
            last_row = tf_df.iloc[-1]
            adx = last_row['adx']
            plus_di = last_row['plus_di']
            minus_di = last_row['minus_di']
            
            # Hitung score tren untuk timeframe ini
            tf_score = 0
            tf_direction = 0  # 1 = bull, -1 = bear, 0 = neutral
            
            # 1. Kekuatan tren (ADX)
            if adx > 25:
                tf_score += 0.4
            elif adx > 20:
                tf_score += 0.3
            elif adx > 15:
                tf_score += 0.2
                
            # 2. Arah tren (DI+ vs DI-)
            if plus_di > minus_di + 5 and plus_di > 20:
                tf_score += 0.3
                tf_direction = 1
            elif minus_di > plus_di + 5 and minus_di > 20:
                tf_score += 0.3
                tf_direction = -1
                
            # 3. EMA alignment
            if last_row['ema20'] > last_row['ema50']:
                tf_score += 0.2
                if tf_direction == 0:
                    tf_direction = 1
            elif last_row['ema20'] < last_row['ema50']:
                tf_score += 0.2
                if tf_direction == 0:
                    tf_direction = -1
                    
            trend_scores.append(tf_score)
            trend_directions.append(tf_direction)
            
            # print(f"   📈 {tf_name.upper()}: Score={tf_score:.2f} | Arah={tf_direction} | ADX={adx:.1f} | DI+={plus_di:.1f}/DI-={minus_di:.1f}")
            
        except Exception as e:
            # print(f"   ⚠️ Error {tf_name}: {e}")
            continue
    
    if not trend_scores:
        return 0.5, 0  # Default neutral
    
    # Hitung average score dan konsensus arah
    avg_score = sum(trend_scores) / len(trend_scores)
    direction_consensus = sum(trend_directions) / len(trend_directions) if trend_directions else 0
    
    # Normalisasi direction consensus (-1 to 1)
    direction_consensus = max(-1, min(1, direction_consensus))
    
    return avg_score, direction_consensus

def execute_order_simulated(symbol, side, qty, price, sl_price, tp_price, balance, leverage,exchange):
    """Simulasikan eksekusi order market"""
    print(f"🎯 SIMULATED {side.upper()} Order @ {price:.4f} | Qty: {qty:.4f} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
    position_value = qty * price
    max_loss = abs((price - sl_price) * qty) * leverage
    max_profit = abs((tp_price - price) * qty) * leverage

    realistic_price = get_realistic_execution_price(exchange, symbol, side, qty)
    slippage_pct = abs(realistic_price - price) / price

    # margin_required = position_value / leverage

    print(f"🎯 SIMULATED {side.upper()} Order @ {realistic_price:.4f} (Request: {price:.4f}) | Qty: {qty:.4f} | SL: {sl_price:.4f} | TP: {tp_price:.4f}")
    print(f"   📊 Slippage: {slippage_pct:.4%} | Arah: {'LONG' if side == 'LONG' else 'SHORT'}")
    print(f"   💰 Simulated Position Value: {position_value:.4f} USDT | Max Loss: {max_loss:.4f} | Max Profit: {max_profit:.4f}")


    position_info = {
        'status': 'simulated',
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'entry_price': realistic_price,
        'sl_price': sl_price,
        'tp_price': tp_price,
        'balance': balance,  # ✅ Balance TIDAK berkurang di awal
        'max_risk': max_loss,  # Simpan sebagai info saja
        'used_margin': position_value,  # Untuk perhitungan leverage
        'slippage_pct': slippage_pct
    }

    return position_info



def get_realistic_execution_price(exchange, symbol, side, qty):
    """Simulasi eksekusi harga realistis dengan slippage"""
    try:
        ob = exchange.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
        if 'asks' not in ob or 'bids' not in ob:
            print(f"⚠️ Format order book tidak valid untuk {symbol}")
            return price * 1.001 if side == 'LONG' else price * 0.999
        
        if side == 'LONG':
            asks = ob['asks']
            filled_price = 0
            remaining_qty = qty
            
            # Simulasi isi dari order book
            for price, volume in asks:
                take_qty = min(remaining_qty, volume)
                filled_price += price * take_qty
                remaining_qty -= take_qty
                if remaining_qty <= 0:
                    break
            
            if remaining_qty > 0:  # Tidak semua terisi
                filled_price += (price * 1.001) * remaining_qty  # Slippage tambahan
            
            avg_price = filled_price / qty
            return avg_price * 1.0005  # Tambahan slippage
        
        else:  # SHORT
            bids = ob['bids']
            filled_price = 0
            remaining_qty = qty
            
            for price, volume in bids:
                take_qty = min(remaining_qty, volume)
                filled_price += price * take_qty
                remaining_qty -= take_qty
                if remaining_qty <= 0:
                    break
            
            if remaining_qty > 0:
                filled_price += (price * 1.001) * remaining_qty
            
            avg_price = filled_price / qty
            return avg_price * 1.0005  # Tambahan slippage
    
    except Exception as e:
        print(f"⚠️ Error simulating execution: {e}")
        # Fallback ke harga ticker dengan slippage
        ticker = exchange.fetch_ticker(symbol)
        if side == 'LONG':
            return ticker['ask'] * 1.001
        else:
            return ticker['bid'] * 0.999

def calculate_market_impact(exchange, symbol, qty, position_value):
    """Estimasi market impact berdasarkan volume harian"""
    try:
        ticker = exchange.fetch_ticker(symbol)
        daily_volume = ticker['quoteVolume']  # Volume dalam USDT
        
        if daily_volume == 0:
            return 0.001  # Default 0.1%
        
        # Formula sederhana: impact = (position_value / daily_volume) * impact_factor
        impact_factor = 0.3  # Faktor empiris untuk spot market
        impact_pct = (position_value / daily_volume) * impact_factor
        return max(0.0005, min(0.01, impact_pct))  # Batasi 0.05% - 1%
    
    except Exception as e:
        print(f"⚠️ Error calculating market impact for {symbol}: {e}")
        return 0.001  # Default 0.1%

        

def execute_order_live(exchange, symbol, side, qty, price, sl_price, tp_price):
    """Eksekusi order live ke exchange (gunakan leverage jika perlu)"""
    print(f"⚠️ LIVE ORDER: {side.upper()} {qty:.4f} {symbol.split('/')[0]} @ {price:.4f}")
    print("   🚨 LIVE ORDER DICEGAH UNTUK KEAMANAN - HAPUS KOMENTAR UNTUK EKSEKUSI SEBENARNYA")
    return {
        'status': 'pending',
        'symbol': symbol,
        'side': side,
        'qty': qty,
        'entry_price': price,
        'sl_price': sl_price,
        'tp_price': tp_price
    }

def check_exit_conditions(position, current_price, exchange):
    """Cek apakah posisi harus exit karena SL atau TP"""
    if position['side'] == 'LONG':
        if current_price <= position['sl_price']:
            print(f"🔴 SIMULATED LONG SL HIT @ {current_price:.4f} | Entry: {position['entry_price']:.4f}")
            return 'SL_HIT'
        elif current_price >= position['tp_price']:
            print(f"🟢 SIMULATED LONG TP HIT @ {current_price:.4f} | Entry: {position['entry_price']:.4f}")
            return 'TP_HIT'
    else: # SHORT
        if current_price >= position['sl_price']:
            print(f"🔴 SIMULATED SHORT SL HIT @ {current_price:.4f} | Entry: {position['entry_price']:.4f}")
            return 'SL_HIT'
        elif current_price <= position['tp_price']:
            print(f"🟢 SIMULATED SHORT TP HIT @ {current_price:.4f} | Entry: {position['entry_price']:.4f}")
            return 'TP_HIT'
    return None

def enhanced_trend_detection(row_idx, df):
    """Deteksi regime pasar - diambil dari backtest"""
    if row_idx < 50:
        return "INSUFFICIENT_DATA"
    try:
        if 'adx' not in df.columns or 'plus_di' not in df.columns or 'minus_di' not in df.columns:
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        adx = df['adx'].iloc[row_idx]
        plus_di = df['plus_di'].iloc[row_idx]
        minus_di = df['minus_di'].iloc[row_idx]
        if 'ema_fast' not in df.columns or 'ema_slow' not in df.columns:
            df['ema_fast'] = talib.EMA(df['close'], 20)
            df['ema_slow'] = talib.EMA(df['close'], 50)
        ema_fast = df['ema_fast'].iloc[row_idx]
        ema_slow = df['ema_slow'].iloc[row_idx]
        price = df['close'].iloc[row_idx]
        trend_score = 0
        if adx > 25: trend_score += 3
        elif adx > 20: trend_score += 2
        elif adx > 15: trend_score += 1
        di_diff = plus_di - minus_di
        if abs(di_diff) > 15: trend_score += 2
        elif abs(di_diff) > 8: trend_score += 1
        if ema_fast > ema_slow and price > ema_fast: trend_score += 2
        elif ema_fast < ema_slow and price < ema_fast: trend_score += 2
        elif (ema_fast > ema_slow and price < ema_fast) or (ema_fast < ema_slow and price > ema_fast): trend_score += 1
        if trend_score >= 6: return "VERY_STRONG_TREND"
        elif trend_score >= 4: return "STRONG_TREND"
        elif trend_score >= 2: return "MODERATE_TREND"
        else: return "SIDEWAYS"
    except Exception as e:
        print(f"⚠️ Error dalam trend detection: {e}")
        return "SIDEWAYS"

def calculate_professional_position_size(balance, entry_price, sl_price, risk_pct, leverage=10):
    """Position sizing dari backtest"""
    if sl_price <= 0 or entry_price <= 0 or balance <= 0:
        return 0
    risk_per_unit = abs(entry_price - sl_price) * leverage
    if risk_per_unit <= 0:
        return 0

    risk_amount = balance * min(risk_pct, MAX_RISK_PER_TRADE)
    position_size = risk_amount / risk_per_unit
    # MAX_POSITION_VALUE = 0.30
    # MIN_POSITION_VALUE = 0.5
    
    position_value = position_size * entry_price
    max_position_value = balance * MAX_POSITION_VALUE
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
    min_position_size = MIN_POSITION_VALUE / entry_price
    if position_value < MIN_POSITION_VALUE:
        position_size = min_position_size if min_position_size <= max_position_value / entry_price else 0
    if entry_price < 1:
        position_size = round(position_size, 0)
    else:
        position_size = round(position_size, 4)
    return max(0, position_size)

# --- FUNGSI UTAMA FORWARD TEST DENGAN SCANNING ULANG ---
def run_forward_test():
    """Jalankan Forward Test Loop dengan scanning ulang otomatis"""
    print(f"🚀 MULAI FORWARD TEST - MODE: {MODE.upper()} | INTERVAL SCAN: {RESCAN_INTERVAL_MINUTES} MENIT")
    print("=" * 70)

    scanner = MarketScanner()
    balance = INITIAL_BALANCE
    active_position = None
    trade_log = []
    current_symbol = None
    last_scan_time = datetime.now()
    last_switch_time = datetime.now()
    last_exit_time = None
    last_oi = None
    last_oi_update = datetime.now()
    OI_MIN_CHANGE_THRESHOLD = 2.0
    # oi_confirmed_long = False
    # oi_confirmed_short = False
    oi_state = {
                    'last_oi': None,
                    'long': False,
                    'short': False,
                    'threshold': 2.0 
                }

    print("🔄 MENCARI ASET TERBAIK UNTUK TRADING AWAL...")
    current_symbol = scanner.get_best_asset_for_trading()
    if not current_symbol:
        print("❌ Gagal mendapatkan aset awal, keluar.")
        return

    print(f"🎯 Memilih aset awal: {current_symbol}")
    print(f"💰 Balance Awal: {balance:.4f}")

    # Inisialisasi data dan indikator
    print(f"📊 Mengambil data awal untuk {current_symbol}...")
    df = fetch_ohlcv_data(current_symbol, TIMEFRAME, BARS_TO_FETCH)
    if df is None or len(df) < BARS_TO_FETCH * 0.7:
        print("❌ Data awal tidak cukup, keluar.")
        return

    # Hitung indikator awal
    df['ema_fast'] = talib.EMA(df['close'], 20)
    df['ema_slow'] = talib.EMA(df['close'], 50)
    df['ema_200'] = talib.EMA(df['close'], 200)
    df['rsi'] = talib.RSI(df['close'], 14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
    df['vol_ma'] = df['volume'].rolling(10).mean()
    df['volume_ma20'] = df['volume'].rolling(20).mean()

    # Ambil profil aset dan regime pasar
    asset_profile = scanner.get_asset_profile(current_symbol, df)
    market_regime = scanner.detect_market_regime(df)
    dynamic_thresholds = scanner.get_dynamic_entry_thresholds(asset_profile, market_regime)

    last_update_time = datetime.now()
    last_data_time = df['timestamp'].iloc[-1]

    print(f"✅ Forward Test Siap - {current_symbol} | Regime: {market_regime} | Aset: {asset_profile.get('asset_class', 'N/A')}")
    print("🔄 Memulai Loop Forward Test dengan SCANNING ULANG OTOMATIS... (Tekan Ctrl+C untuk berhenti)")

    try:
        while True:

            current_time = datetime.now()
            time.sleep(5) # Tunggu 5 detik sebelum cek lagi
            # --- LOGIKA SCANNING ULANG OTOMATIS ---

            if TIMEFRAME == '15m':
                minute = current_time.minute
                second = current_time.second
                # Cek apakah kita berada di menit 00, 15, 30, atau 45
                if minute % 15 != 0:
                    continue  # Skip kalau bukan menit kelipatan 15
                if second < 2:  # Tunggu minimal 2 detik setelah tepat menit
                    continue
            

            oi_confirmed_long = oi_state['long']
            oi_confirmed_short = oi_state['short']

            should_rescan = False
            
            # Kondisi 1: Interval waktu tercapai
            if (current_time - last_scan_time).total_seconds() >= RESCAN_INTERVAL_MINUTES * 60:
                print(f"⏰ [{current_time.strftime('%H:%M:%S')}] WAKTU SCAN ULANG TERJANGKAU ({RESCAN_INTERVAL_MINUTES} MENIT)")
                should_rescan = True
            
            # Kondisi 2: Setelah exit posisi (tunggu minimal MIN_TIME_BETWEEN_SCANS menit)
            if last_exit_time and (current_time - last_exit_time).total_seconds() >= MIN_TIME_BETWEEN_SCANS * 60:
                print(f"✅ [{current_time.strftime('%H:%M:%S')}] SCANNING ULANG SETELAH EXIT POSISI")
                should_rescan = True
                last_exit_time = None  # Reset agar tidak terus menerus scan
            
            # Kondisi 3: Tidak ada sinyal entry dalam waktu lama (opsional, bisa ditambahkan)
            # ...

            if should_rescan:
                print("\n" + "="*50)
                print(f"🔍 MEMULAI PROSES SCANNING ULANG...")
                print("="*50)
                
                new_symbol = scanner.get_best_asset_for_trading()
                if new_symbol:
                    if new_symbol != current_symbol:
                        # Cek apakah minimal waktu antar switch terpenuhi
                        if (current_time - last_switch_time).total_seconds() < MIN_TIME_BETWEEN_SCANS * 60:
                            print(f"⏳ [{current_time.strftime('%H:%M:%S')}] TERLALU CEPAT UNTUK GANTI ASET! Tunggu {MIN_TIME_BETWEEN_SCANS} menit sejak switch terakhir")
                            print(f"   Tetap di {current_symbol} untuk sekarang")
                        else:
                            print(f"🎯 [{current_time.strftime('%H:%M:%S')}] ASET BARU DITEMUKAN: {new_symbol} (sebelumnya: {current_symbol})")
                            
                            # Simpan data jika ada posisi aktif
                            if active_position:
                                print(f"⚠️ [{current_time.strftime('%H:%M:%S')}] MASIH ADA POSISI AKTIF DI {current_symbol}, TIDAK BISA GANTI ASET")
                                print(f"   Tunggu posisi selesai atau tutup manual terlebih dahulu")
                            else:
                                # Ganti ke aset baru
                                print(f"🔄 [{current_time.strftime('%H:%M:%S')}] MENGAMBIL DATA BARU UNTUK {new_symbol}...")
                                new_df = fetch_ohlcv_data(new_symbol, TIMEFRAME, BARS_TO_FETCH)
                                if new_df is not None and len(new_df) >= BARS_TO_FETCH * 0.7:
                                    # Data baru berhasil diambil, ganti aset
                                    current_symbol = new_symbol
                                    last_switch_time = current_time
                                    df = new_df # Ganti df dengan data baru
                                    # Hitung ulang indikator
                                    df['ema_fast'] = talib.EMA(df['close'], 20)
                                    df['ema_slow'] = talib.EMA(df['close'], 50)
                                    df['ema_200'] = talib.EMA(df['close'], 200)
                                    df['rsi'] = talib.RSI(df['close'], 14)
                                    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                                    df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                                    df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                                    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                                    df['vol_ma'] = df['volume'].rolling(10).mean()
                                    df['volume_ma20'] = df['volume'].rolling(20).mean()

                                    # reset OI
                                    oi_state['last_oi'] = None
                                    oi_state['long'] = False
                                    oi_state['short'] = False

                                    symbol_base = current_symbol.split('/')[0]
                                    asset_class = scanner.asset_classification.get(symbol_base, 'DEFAULT')
                                    threshold_map = {'MAJOR': 0.8, 'MID_CAP': 1.0, 'SMALL_CAP': 1.5, 'MEME': 2.0, 'DEFAULT': 1.8}
                                    oi_state['threshold'] = threshold_map.get(asset_class, 1.5)
                                    # end reset OI 

                                    # Update profil aset dan regime
                                    asset_profile = scanner.get_asset_profile(current_symbol, df)
                                    market_regime = scanner.detect_market_regime(df)
                                    dynamic_thresholds = scanner.get_dynamic_entry_thresholds(asset_profile, market_regime)
                                    last_data_time = df['timestamp'].iloc[-1]
                                    print(f"✅ [{current_time.strftime('%H:%M:%S')}] BERHASIL GANTI KE {current_symbol} | Regime: {market_regime}")
                                else:
                                    print(f"❌ [{current_time.strftime('%H:%M:%S')}] GAGAL AMBIL DATA UNTUK {new_symbol}, TETAP DI {current_symbol}")
                                    # Jika gagal mengambil data untuk aset baru, kita tetap di aset lama
                                    # Tidak perlu mengganti current_symbol atau df
                                    # Perbarui profil dan regime untuk aset lama (current_symbol) jika perlu
                                    print(f"🔄 [{current_time.strftime('%H:%M:%S')}] Memperbarui profil dan regime untuk aset lama: {current_symbol}")
                                    asset_profile = scanner.get_asset_profile(current_symbol, df)
                                    market_regime = scanner.detect_market_regime(df)
                                    dynamic_thresholds = scanner.get_dynamic_entry_thresholds(asset_profile, market_regime)
                                    # last_data_time dan df tetap sama
                                    # last_switch_time TIDAK diupdate karena kita tidak benar-benar pindah
                    else:
                        print(f"🔄 [{current_time.strftime('%H:%M:%S')}] ASET TERBAIK MASIH SAMA: {current_symbol}")
                else:
                    print(f"❌ [{current_time.strftime('%H:%M:%S')}] GAGAL MENDAPATKAN ASET BARU, TETAP DI {current_symbol}")
                
                last_scan_time = current_time
                print("="*50 + "\n")

            # Ambil data baru jika cukup waktu
            if (current_time - last_update_time).seconds >= DATA_UPDATE_INTERVAL: # dinamis
                try:
                    new_ohlcv = scanner.exchange.fetch_ohlcv(current_symbol, TIMEFRAME, limit=2)
                    new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    new_df = new_df.sort_values('timestamp').reset_index(drop=True)

                    latest_candle_time = new_df['timestamp'].iloc[-1]

                    if latest_candle_time > last_data_time:
                        df = pd.concat([df, new_df.iloc[1:]], ignore_index=True)
                        df = df.iloc[-BARS_TO_FETCH:]

                        if df[['open', 'high', 'low', 'close', 'volume']].isna().any().any():
                            print(f"⚠️ Data baru ({current_symbol}) mengandung NaN. Melewati pembaruan indikator.")
                        
                        # memulai perhitungan OI
                        # if len(df) < 2:
                        #     return
                        
                        symbol_base = current_symbol.split('/')[0]
                        asset_class = scanner.asset_classification.get(symbol_base, 'DEFAULT')
                        threshold_map = {'MAJOR': 0.8, 'MID_CAP': 1.0, 'SMALL_CAP': 1.5, 'MEME': 2.0, 'DEFAULT': 1.8}
                        oi_state['threshold'] = threshold_map.get(asset_class, 1.5)

                        prev_close = df['close'].iloc[-2]
                        current_close = df['close'].iloc[-1]
                        current_oi = get_open_interest(scanner.exchange, current_symbol)
                        atr_value = df['atr'].iloc[-1]  
                        long_confirm, short_confirm = get_oi_confirmation(df, current_oi, oi_state['last_oi'], atr_value)

                        oi_state['long'] = long_confirm
                        oi_state['short'] = short_confirm

                        if current_oi is not None:
                            oi_state['last_oi'] = current_oi
                            print(f"OI terbaru untuk {current_symbol}: {current_oi}")
                        else:
                            print('data OI belum lengkap')
                        
                        oi_confirmed_long = long_confirm
                        oi_confirmed_short = short_confirm

                        # hasil perhitungan OI 

                        # Update indikator dengan rolling calculation yang lebih efisien
                        df['ema_fast'] = talib.EMA(df['close'], 20)
                        df['ema_slow'] = talib.EMA(df['close'], 50)
                        df['ema_200'] = talib.EMA(df['close'], 200)
                        df['rsi'] = talib.RSI(df['close'], 14)
                        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                        df['vol_ma'] = df['volume'].rolling(10).mean()
                        df['volume_ma20'] = df['volume'].rolling(20).mean()

                        last_data_time = latest_candle_time
                        last_update_time = current_time

                        print(f"[{latest_candle_time.strftime('%Y-%m-%d %H:%M:%S')}] Data {current_symbol} diperbarui.")
                    else:
                        print(f"[{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Data belum update, menunggu candle baru...")
                except Exception as e:
                    print(f"⚠️ Error mengambil data baru untuk {current_symbol}: {e}")
                    continue


            # Pastikan df tidak None sebelum melanjutkan ke logika utama
            if df is None or len(df) < BARS_TO_FETCH * 0.7:
                 print(f"❌ [{current_time.strftime('%Y-%m-%d %H:%M:%S')}] Data untuk {current_symbol} tidak valid (None atau tidak cukup), menunggu pembaruan atau scan ulang...")
                 continue # Tunggu pembaruan data atau scan ulang


            # Ambil harga real-time untuk cek exit
            current_price = get_current_price(scanner.exchange, current_symbol)
            if current_price is None:
                print("⚠️ Gagal ambil harga real-time, lewati iterasi ini.")
                continue

            # --- CEK EXIT ---
            if active_position is not None:
                exit_reason = check_exit_conditions(active_position, current_price, scanner.exchange)
                if exit_reason:
                    # Hitung PnL sesuai arah posisi
                    if active_position['side'] == 'LONG':
                        pnl = (current_price - active_position['entry_price']) * active_position['qty'] * LEVERAGE
                    else:  # SHORT
                        pnl = (active_position['entry_price'] - current_price) * active_position['qty'] * LEVERAGE

                    position_value = active_position['entry_price'] * active_position['qty']
                    fee_cost = position_value * 0.001 * 2  # 0.1% per trade × 2 (entry+exit)
                    slippage_cost = position_value * SLIPPAGE_RATE * LEVERAGE  # 0.05% slippage
                    market_impact = calculate_market_impact(scanner.exchange, active_position['symbol'], active_position['qty'], position_value)
                    impact_cost = position_value * market_impact
                    
                    net_pnl = pnl - fee_cost - slippage_cost - impact_cost
                    balance = balance + net_pnl
                    
                    print(f"💰 Gross PnL: {pnl:+.4f} | Biaya: Fee {fee_cost:.4f} + Slippage {slippage_cost:.4f} + Impact {impact_cost:.4f} | Net: {net_pnl:+.4f}")
        
                    
                    # Log trade exit
                    trade_result = {
                        'exit_time': datetime.now(),
                        'exit_reason': exit_reason,
                        'exit_price': current_price,
                        'gross_pnl': pnl,
                        'fee_cost': fee_cost,
                        'slippage_cost': slippage_cost,
                        'impact_cost': impact_cost,
                        # 'pnl': pnl,
                        'net_pnl': net_pnl,
                        'balance_after_exit': balance,
                        'hold_time': (datetime.now() - active_position['entry_time']).total_seconds() / 60  # dalam menit
                    }
                    
                    trade_log.append({
                        **active_position, 
                        **trade_result
                    })

                    log_exit_to_excel(trade_log[-1])

                    send_telegram_message(f"❌ <b>EXIT</b>\n"
                                          f"Coin: {active_position['symbol']}\n"
                                          f"Arah: {active_position['side']}\n"
                                          f"Exit Reason: {exit_reason}\n"
                                          f"Harga: {current_price:.4f}\n"
                                          f"PnL Net: {net_pnl:+.4f}\n"
                                          f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"📋 Trade Log: {active_position['side']} {active_position['symbol']} exit on {exit_reason}")
                    print(f"   💰 PnL: {pnl:+.4f} USDT | Balance: {balance:.4f} USDT | Hold Time: {trade_result['hold_time']:.1f} menit")
                    
                    active_position = None
                    last_exit_time = datetime.now()  # Trigger scanning ulang setelah exit

            # --- CEK ENTRY BARU (hanya jika tidak ada posisi aktif) ---
            if active_position is not None:
                continue  # Lewati jika ada posisi aktif

            # Gunakan bar terakhir dari df yang diperbarui
            if len(df) < 52:
                continue # Data tidak cukup untuk analisis
            
            i = len(df) - 2
            current_row = df.iloc[i]

            regime = enhanced_trend_detection(i, df)
            atr = current_row['atr']
            close = current_row['close']
            volume = current_row['volume']
            vol_ma = current_row['vol_ma']
            adx = current_row['adx']
            vol_ratio = volume / vol_ma if vol_ma > 0 else 0
            atr_pct = (atr / close) * 100 if close > 0 else 0

            # print(f"[{current_time.strftime('%H:%M:%S')}] [DEBUG] ATR%: {atr_pct:.3f}% (Thres: {dynamic_thresholds['atr_threshold']:.3f}%) | ADX: {adx:.1f} (Thres: {dynamic_thresholds['adx_threshold']:.1f}) | VolRatio: {vol_ratio:.2f}x (Thres: {dynamic_thresholds['volume_multiplier']:.2f}x) | EMA: F{current_row['ema_fast']:.4f} vs S{current_row['ema_slow']:.4f}")

            # print(f"[{current_time.strftime('%H:%M:%S')}] Cek Entry - Regime: {regime} | Market Regime: {market_regime} | ATR%: {atr_pct:.3f}% | Vol Ratio: {vol_ratio:.2f}x | ADX: {adx:.1f}")
            # ------------------------
            if regime in ["SIDEWAYS", "INSUFFICIENT_DATA"]:
                continue

            ema_slow = current_row['ema_slow']
            ema_fast = current_row['ema_fast']
            atr_threshold = dynamic_thresholds['atr_threshold']
            volume_multiplier = dynamic_thresholds['volume_multiplier']
            adx_threshold = dynamic_thresholds['adx_threshold']
            level_multiplier = dynamic_thresholds['level_multiplier']
            risk_pct = dynamic_thresholds['risk_pct']

            # print(f"[DEBUG] Volume Multiplier Dinamis: {volume_multiplier:.2f}x | Vol Ratio Saat Ini: {vol_ratio:.2f}x | Vol Confirmed: {vol_ratio >= volume_multiplier}")

            vol_confirmed = vol_ratio >= volume_multiplier
            atr_confirmed = atr_pct >= atr_threshold

            prev_close = df['close'].iloc[i-1]
            long_level = prev_close + atr * level_multiplier
            short_level = prev_close - atr * level_multiplier

            broke_long_prev = df['high'].iloc[i-1] >= long_level
            broke_short_prev = df['low'].iloc[i-1] <= short_level

            # retest_long = (broke_long_prev and
            #               df['low'].iloc[i] <= long_level * 1.003 and
            #               close > long_level * 0.997)
            # retest_short = (broke_short_prev and
            #                df['high'].iloc[i] >= short_level * 0.997 and
            #                close < short_level * 1.003)

            ENTRY_ZONE_BUFFER = 0.015 
                
            if TIMEFRAME == '15m':
                # Zona entry lebih longgar
                long_zone_start = long_level - (atr * ZONE_START_FACTOR * level_multiplier)
                long_zone_end = long_level + (atr * ZONE_END_FACTOR * level_multiplier)
                short_zone_start = short_level - (atr * ZONE_END_FACTOR * level_multiplier)
                short_zone_end = short_level + (atr * ZONE_START_FACTOR * level_multiplier)
            else:
                # Konfigurasi default untuk 5m
                long_zone_start = long_level - (atr * 0.8 * level_multiplier)
                long_zone_end = long_level + (atr * 1.2 * level_multiplier)
                short_zone_start = short_level - (atr * 1.2 * level_multiplier)
                short_zone_end = short_level + (atr * 0.8 * level_multiplier)
        

            price_in_long_zone = (df['low'].iloc[i] <= long_zone_end) and (close >= long_zone_start)
            price_in_short_zone = (df['high'].iloc[i] >= short_zone_start) and (close <= short_zone_end)

            plus_di = current_row['plus_di']
            minus_di = current_row['minus_di']
            momentum_strength = abs(plus_di - minus_di) / max(plus_di, minus_di, 1)
            strong_momentum = momentum_strength > (0.20 if TIMEFRAME == '15m' else 0.25)

            allow_long = True
            allow_short = True

            # print(f"\n🔍 [{current_time.strftime('%H:%M:%S')}] Multi-Timeframe Analysis untuk {current_symbol}...")
            mtf_score, mtf_direction = get_multi_timeframe_confirmation(scanner.exchange, current_symbol)

            if TIMEFRAME == '15m':
                MTF_MIN_SCORE = 0.3    # Lebih longgar (30%)
                MTF_DIRECTION_THRESHOLD = 0.1  # Minimal consensus
                MTF_WEIGHT = 0.8       # Bobot lebih tinggi karena lebih reliable
            else:  # 5m
                MTF_MIN_SCORE = 0.4
                MTF_DIRECTION_THRESHOLD = 0.2
                MTF_WEIGHT = 0.6

            # print(f"🎯 MTF Result: Score={mtf_score:.2f} | Direction={mtf_direction:.2f}")
            # print(f"   {'✅ BULLISH' if mtf_direction > 0.3 else '✅ BEARISH' if mtf_direction < -0.3 else '🟡 NEUTRAL'} di timeframe lebih tinggi")

            rsi = current_row['rsi']
            volume_ratio = volume / current_row['volume_ma20']

            rsi_long_ok = (rsi < 70) and (rsi > 30) and (volume_ratio > 1.5)  # RSI < 70 untuk long
            rsi_short_ok = (rsi > 30) and (rsi < 70) and (volume_ratio > 1.5) # RSI > 30 untuk short

            if market_regime in ['STRONG_BULL', 'BULL']:
                allow_short = False
            elif market_regime in ['STRONG_BEAR', 'BEAR']:
                allow_long = False

            high_quality_long = (broke_long_prev and 
                                price_in_long_zone and
                                vol_confirmed and 
                                atr_confirmed and 
                                # oi_confirmed_long and
                                ema_fast > ema_slow and 
                                adx > adx_threshold and 
                                strong_momentum and 
                                allow_long and
                                mtf_score >= MTF_MIN_SCORE and 
                                mtf_direction >= MTF_DIRECTION_THRESHOLD
                                and rsi_long_ok
                                )

            high_quality_short = (
                                broke_short_prev and 
                                price_in_short_zone and 
                                vol_confirmed and atr_confirmed and 
                                # oi_confirmed_short and
                                ema_fast < ema_slow and 
                                adx > adx_threshold and 
                                strong_momentum and 
                                allow_short and 
                                mtf_score >= MTF_MIN_SCORE and 
                                mtf_direction <= -MTF_DIRECTION_THRESHOLD
                                and rsi_short_ok
                                )
            
            # if price_in_long_zone or price_in_short_zone:
            #     print(f"[{current_time.strftime('%H:%M:%S')}] [ZONE ENTRY] {current_symbol}")
            #     print(f"   🔵 LONG Zone: {long_zone_start:.6f} - {long_zone_end:.6f} | Current: {close:.6f} | In Zone: {price_in_long_zone}")
            #     print(f"   🔴 SHORT Zone: {short_zone_start:.6f} - {short_zone_end:.6f} | Current: {close:.6f} | In Zone: {price_in_short_zone}")
            #     print(f"   💪 Momentum Strength: {momentum_strength:.3f} | Strong: {strong_momentum} | DI+: {plus_di:.1f} | DI-: {minus_di:.1f}")


            # if (retest_long and vol_confirmed and atr_confirmed and
            #     ema_fast > ema_slow and adx > adx_threshold and allow_long):
            #     sl = long_level - atr * SL_ATR_MULT
            #     tp = long_level + atr * TP_ATR_MULT
            if high_quality_long:
                sl = long_level - atr * SL_ATR_MULT
                tp = long_level + atr * TP_ATR_MULT
                if sl <= 0 or tp <= long_level or sl >= long_level:
                    continue
                
                qty = calculate_professional_position_size(balance, long_level, sl, risk_pct, LEVERAGE)
                if qty > 0:
                    print(f"🔍 [{current_time.strftime('%H:%M:%S')}] Sinyal LONG DETECTED untuk {current_symbol} @ {long_level:.4f}")
                    print(f"   📊 ATR%: {atr_pct:.3f}% (Threshold: {atr_threshold:.3f}%) | Volume Ratio: {vol_ratio:.2f}x (Threshold: {volume_multiplier:.2f}x)")
                    print(f"   📈 Market Regime: {market_regime} | Trend Regime: {regime}")
                    
                    if MODE == 'simulated':
                        active_position = execute_order_simulated(current_symbol, 'LONG', qty, long_level, sl, tp, balance, LEVERAGE, scanner.exchange)
                        balance = active_position['balance'] 
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime
                        
                        log_entry_to_excel(active_position)

                        send_telegram_message(f"📊 <b>ENTRY</b>\n"
                          f"Coin: {active_position['symbol']}\n"
                          f"Arah: {active_position['side']}\n"
                          f"Harga: {active_position['entry_price']:.4f}\n"
                          f"Qty: {active_position['qty']:.4f}\n"
                          f"SL: {active_position['sl_price']:.4f}\n"
                          f"TP: {active_position['tp_price']:.4f}\n"
                          f"Time: {active_position['entry_time'].strftime('%H:%M:%S')}")

                    elif MODE == 'live':
                        active_position = execute_order_live(scanner.exchange, current_symbol, 'LONG', qty, long_level, sl, tp)
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime

            # elif (retest_short and vol_confirmed and atr_confirmed and
            #       ema_fast < ema_slow and adx > adx_threshold and allow_short):
            #     sl = short_level + atr * SL_ATR_MULT
            #     tp = short_level - atr * TP_ATR_MULT
            elif high_quality_short:
                sl = short_level + atr * SL_ATR_MULT
                tp = short_level - atr * TP_ATR_MULT
                if sl <= short_level or tp >= short_level or sl <= 0 or tp <= 0:
                    continue
                # if sl <= short_level or tp >= short_level or sl <= 0:
                #     continue

                qty = calculate_professional_position_size(balance, short_level, sl, risk_pct, LEVERAGE)
                if qty > 0:
                    print(f"🔍 [{current_time.strftime('%H:%M:%S')}] Sinyal SHORT DETECTED untuk {current_symbol} @ {short_level:.4f}")
                    print(f"   📊 ATR%: {atr_pct:.3f}% (Threshold: {atr_threshold:.3f}%) | Volume Ratio: {vol_ratio:.2f}x (Threshold: {volume_multiplier:.2f}x)")
                    print(f"   📉 Market Regime: {market_regime} | Trend Regime: {regime}")
                    
                    if MODE == 'simulated':
                        active_position = execute_order_simulated(current_symbol, 'SHORT', qty, short_level, sl, tp, balance, LEVERAGE,scanner.exchange)
                        balance = active_position['balance'] 
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime
                    elif MODE == 'live':
                        active_position = execute_order_live(scanner.exchange, current_symbol, 'SHORT', qty, short_level, sl, tp)
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime

    except KeyboardInterrupt:
        print("\n🛑 Forward Test dihentikan oleh pengguna.")
        if active_position:
            print(f"⚠️ Masih ada posisi aktif: {active_position}")
            current_price = get_current_price(scanner.exchange, current_symbol)
            if current_price:
                exit_reason = check_exit_conditions(active_position, current_price, scanner.exchange)
                if exit_reason:
                    if active_position['side'] == 'LONG':
                        pnl = (current_price - active_position['entry_price']) * active_position['qty'] * LEVERAGE
                    else:
                        pnl = (active_position['entry_price'] - current_price) * active_position['qty'] * LEVERAGE
                    
                    position_value = active_position['entry_price'] * active_position['qty']
                    fee_cost = position_value * 0.001 * 2  # 0.1% per trade × 2 (entry+exit)
                    slippage_cost = position_value * 0.0005 * LEVERAGE  # 0.05% slippage
                    market_impact = calculate_market_impact(scanner.exchange, current_symbol, active_position['qty'], position_value)
                    impact_cost = position_value * market_impact
                    net_pnl = pnl - fee_cost - slippage_cost - impact_cost
                    balance = balance + net_pnl

                    print(f"💰 Gross PnL: {pnl:+.4f} | Biaya: Fee {fee_cost:.4f} + Slippage {slippage_cost:.4f} + Impact {impact_cost:.4f} | Net: {net_pnl:+.4f}")

                    trade_log.append(
                        {
                            **active_position, 
                            'exit_time': datetime.now(), 
                            'exit_reason': exit_reason, 
                            'exit_price': current_price, 
                            'net_pnl': net_pnl, 
                            'balance_after_exit': balance, 
                            'hold_time': (datetime.now() - active_position['entry_time']).total_seconds() / 60
                        }
                    )

                else:
                    trade_log.append({**active_position, 'exit_time': datetime.now(), 'exit_reason': 'MANUAL_STOP', 'exit_price': current_price, 'net_pnl': 'N/A', 'balance_after_exit': balance, 'hold_time': (datetime.now() - active_position['entry_time']).total_seconds() / 60})

        if trade_log:

            # log_df = pd.DataFrame(trade_log)
            # filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # log_df.to_csv(filename, index=False)
            # print(f"💾 Log Forward Test disimpan ke: {filename}")

            log_df = pd.DataFrame(trade_log)
    
            # 🔑 KONVERSI KOLOM WAKTU KE STRING YANG RAMAH EXCEL
            datetime_columns = ['entry_time', 'exit_time']
            for col in datetime_columns:
                if col in log_df.columns:
                    if pd.api.types.is_datetime64_any_dtype(log_df[col]):
                        log_df[col] = log_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                    elif log_df[col].dtype == 'object':  # Sudah string
                        # Opsional: validasi format string
                        pass
            
            # 📁 BUAT NAMA FILE YANG AMAN
            filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            
            # 💾 SIMPAN KE EXCEL DENGAN FORMAT RAPI
            log_df.to_excel(filename, index=False, engine='openpyxl')
            
            # ✅ KONFIRMASI SUKSES
            print(f"✅ Log Forward Test BERHASIL disimpan ke: {filename}")
            print(f"   📊 Total baris: {len(log_df)} | Kolom: {', '.join(log_df.columns)}")
            
            # Tampilkan ringkasan performa
            if not log_df.empty:
                # completed_trades = log_df[log_df['exit_reason'].isin(['SL_HIT', 'TP_HIT'])]
                completed_trades = log_df[
                    (log_df['exit_reason'].isin(['SL_HIT', 'TP_HIT'])) & 
                    (log_df['net_pnl'] != 'N/A') &
                    (pd.to_numeric(log_df['net_pnl'], errors='coerce').notnull())
                ]
                
                if len(completed_trades) > 0:
                    completed_trades['net_pnl'] = pd.to_numeric(completed_trades['net_pnl'])
                    win_rate = len(completed_trades[completed_trades['net_pnl'] > 0]) / len(completed_trades)
                    total_pnl = completed_trades['net_pnl'].sum()     
                else:
                    print("⚠️ Tidak ada trade yang selesai dengan SL/TP hit")

                

                if not completed_trades.empty:
                    win_rate = len(completed_trades[completed_trades['net_pnl'] > 0]) / len(completed_trades)
                    total_pnl = completed_trades['net_pnl'].sum()
                    avg_hold_time = completed_trades['hold_time'].mean()
                    
                    print("\n" + "="*50)
                    print("📊 RINGKASAN PERFORMA FORWARD TEST")
                    print("="*50)
                    print(f"   Total Trade: {len(completed_trades)}")
                    print(f"   Win Rate: {win_rate:.2%}")
                    print(f"   Total PnL: {total_pnl:+.4f} USDT")
                    print(f"   Final Balance: {balance:.4f} USDT")
                    print(f"   Rata-rata Hold Time: {avg_hold_time:.1f} menit")
                    print(f"   Aset yang diperdagangkan: {', '.join(completed_trades['symbol'].unique())}")
                    print("="*50)
        
        print(f"💰 Balance Akhir (Simulated): {balance:.4f}")

# --- FUNGSI FETCH DATA ---
def fetch_ohlcv_data(symbol, timeframe, limit):
    """Ambil data OHLCV dari exchange"""
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        # Tambahkan sleep kecil untuk membantu menghindari rate limit
        # time.sleep(0.1) 
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        if len(ohlcv) < limit * 0.7:
            print(f"⚠️ Data tidak lengkap: {len(ohlcv)}/{limit} candle untuk {symbol}")
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        return df
    except Exception as e:
        print(f"❌ Error mengambil data {symbol}: {e}")
        return None

def send_telegram_message(message):
    """Kirim pesan ke chat Telegram"""
    if not ENABLE_TELEGRAM:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML' # Untuk format HTML opsional
        }
        response = requests.post(url, data=payload)
        if response.status_code != 200:
            print(f"❌ Gagal kirim Telegram: {response.text}")
    except Exception as e:
        print(f"⚠️ Error kirim Telegram: {e}")


def log_entry_to_excel(entry_data):
    """Simpan data entry ke file Excel"""
    filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d')}.xlsx" # Gunakan tanggal hari ini
    df_entry = pd.DataFrame([{
        'entry_time': entry_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'), # Konversi ke string
        'exit_time': '', # Kosongkan untuk entry
        'symbol': entry_data['symbol'],
        'side': entry_data['side'],
        'entry_price': entry_data['entry_price'],
        'exit_price': '', # Kosongkan untuk entry
        'qty': entry_data['qty'],
        'sl_price': entry_data['sl_price'],
        'tp_price': entry_data['tp_price'],
        'exit_reason': '', # Kosongkan untuk entry
        'gross_pnl': '', # Kosongkan untuk entry
        'fee_cost': '', # Kosongkan untuk entry
        'slippage_cost': '', # Kosongkan untuk entry
        'impact_cost': '', # Kosongkan untuk entry
        'net_pnl': '', # Kosongkan untuk entry
        'balance_after_exit': entry_data['balance'], # Balance saat entry
        'hold_time': '' # Kosongkan untuk entry
    }])

    try:
        # Coba baca file yang sudah ada
        existing_df = pd.read_excel(filename, engine='openpyxl')
        # Gabungkan data baru
        updated_df = pd.concat([existing_df, df_entry], ignore_index=True)
    except FileNotFoundError:
        # Jika file belum ada, gunakan data baru sebagai awal
        updated_df = df_entry

    # Simpan kembali ke file
    updated_df.to_excel(filename, index=False, engine='openpyxl')
    print(f"✅ Entry log disimpan ke: {filename}")

def log_exit_to_excel(exit_data):
    """Simpan data exit ke file Excel yang sama dengan entry"""
    filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d')}.xlsx" # Gunakan tanggal hari ini
    df_exit = pd.DataFrame([{
        'entry_time': exit_data['entry_time'].strftime('%Y-%m-%d %H:%M:%S'), # Konversi ke string
        'exit_time': exit_data['exit_time'], # Sudah string
        'symbol': exit_data['symbol'],
        'side': exit_data['side'],
        'entry_price': exit_data['entry_price'],
        'exit_price': exit_data['exit_price'],
        'qty': exit_data['qty'],
        'sl_price': exit_data['sl_price'],
        'tp_price': exit_data['tp_price'],
        'exit_reason': exit_data['exit_reason'],
        'gross_pnl': exit_data['gross_pnl'],
        'fee_cost': exit_data['fee_cost'],
        'slippage_cost': exit_data['slippage_cost'],
        'impact_cost': exit_data['impact_cost'],
        'net_pnl': exit_data['net_pnl'],
        'balance_after_exit': exit_data['balance_after_exit'],
        'hold_time': exit_data['hold_time']
    }])

    try:
        # Coba baca file yang sudah ada
        existing_df = pd.read_excel(filename, engine='openpyxl')
        # Gabungkan data baru
        updated_df = pd.concat([existing_df, df_exit], ignore_index=True)
    except FileNotFoundError:
        # Jika file belum ada, gunakan data baru sebagai awal
        updated_df = df_exit

    # Simpan kembali ke file
    updated_df.to_excel(filename, index=False, engine='openpyxl')
    print(f"✅ Exit log disimpan ke: {filename}")

def get_open_interest(exchange, symbol):
        try:
            base = symbol.split('/')[0]
            MULTIPLIER_SYMBOLS = {
                'PEPE': '100PEPE',
                'SHIB': '1000SHIB',
                'FLOKI': '1000FLOKI',
            }
    
            if base in MULTIPLIER_SYMBOLS:
                futures_symbol = MULTIPLIER_SYMBOLS[base] + 'USDT'
            else:
                futures_symbol = base + 'USDT'
            oi_data = exchange.fapiPublicGetOpenInterest({'symbol': futures_symbol})
            oi = float(oi_data['openInterest'])
            print(f"✅ OI untuk {symbol}: {oi:,.0f}")
            return oi
        except Exception as e:
            print(f"⚠️ Error ambil OI untuk {symbol}: {e}")
            return None

def calculate_oi_change(current_oi, previous_oi):
    if previous_oi is None or previous_oi == 0:
        return 0.0
    return ((current_oi - previous_oi) / previous_oi) * 100

def get_oi_confirmation(df, current_oi, last_oi, atr):
    if last_oi is None or current_oi is None:
        return False, False

    oi_change_pct = ((current_oi - last_oi) / last_oi) * 100
    candle_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    big_candle = candle_body > (0.20 * atr)

    price_up = df['close'].iloc[-1] > df['open'].iloc[-1]
    price_down = df['close'].iloc[-1] < df['open'].iloc[-1]

    # Default -> no OI signal
    long_confirm = False
    short_confirm = False

    # RULE 1: OI naik + big candle = valid breakout
    if oi_change_pct > 0.3 and big_candle:
        if price_up:
            long_confirm = True
            print(f"🟢 OI naik {oi_change_pct:.2f}% dan big candle, konfirmasi long")
        if price_down:
            short_confirm = True
            print(f"🟢 OI naik {oi_change_pct:.2f}% dan big candle, konfirmasi short")

    # RULE 2: OI turun = jangan entry apapun (liquidation close)
    if oi_change_pct < -0.5:
        long_confirm = False
        short_confirm = False
        print(f"🔴 OI turun {oi_change_pct:.2f}%, jangan entry apapun (liquidation close)")

    return long_confirm, short_confirm

# --- MAIN ---
if __name__ == "__main__":
    run_forward_test()