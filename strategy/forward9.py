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
LEVERAGE = 20
# TP_ATR_MULT = 3.0
TIMEFRAME = '1h'

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
    RESCAN_INTERVAL_MINUTES = 180
    MIN_TIME_BETWEEN_SCANS = 45

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

elif TIMEFRAME == '1h':
    ATR_WINDOW = 14
    VOLATILITY_ADJUSTMENT = 1.5  
    
    TP_ATR_MULT = 3.5  # Dari 5.5 di 15m
    SL_ATR_MULT = 2.0  # Dari 3.0 di 15m
    ZONE_START_FACTOR = 0.8
    ZONE_END_FACTOR = 1.5
    ENTRY_ZONE_BUFFER = 0.015 
    ZONE_BUFFER_MULTIPLIER = 1.3
    
    # Risk Management (lebih konservatif)
    BASE_RISK_PCT = 0.01  # 1% per trade
    MAX_RISK_PER_TRADE = 0.02  # Max 2%
    MAX_POSITION_VALUE = 0.40  # 40% dari balance
    MIN_POSITION_VALUE = 2.0   # Minimal $2 per trade
    
    # MTF Confirmation
    MTF_MIN_SCORE = 0.5
    MTF_DIRECTION_THRESHOLD = 0.2  # Lebih ketat
    
    # Interval Timing (sesuai permintaan 6 jam)
    DATA_UPDATE_INTERVAL = 1800  # 30 menit (update data lebih jarang)
    RESCAN_INTERVAL_MINUTES = 360  # 6 jam
    MIN_TIME_BETWEEN_SCANS = 120  # Minimal 2 jam antar switch
    
    # Volume & Momentum
    VOLUME_WINDOW = 20  # Lebih panjang untuk smoothing
    MIN_MOMENTUM_STRENGTH = 0.25  # Lebih ketat
    
    BARS_TO_FETCH = 200  # 200 candle 1h = ~8 hari
    VOLATILITY_WINDOW = 50  # Lebih pendek karena data lebih sedikit
    MIN_SCAN_SCORE = 0.55  # Lebih tinggi (kualitas sinyal lebih penting)
    SLIPPAGE_RATE = 0.0005  # 0.05% (lebih rendah di timeframe tinggi)
    MAX_SLIPPAGE_RATE = 0.001  # 0.1%
    ORDER_BOOK_DEPTH = 30  # Lebih dalam
    OI_UPDATE_INTERVAL = 3600  # 1 jam (update OI lebih jarang)

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

    # RESCAN_INTERVAL_MINUTES = 2
    # MIN_TIME_BETWEEN_SCANS = 1
    # DATA_UPDATE_INTERVAL = 1 


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
                'options': {'defaultType': 'future'}
            })
        else:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
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
            # MAJOR (Market cap > $1B, likuiditas sangat tinggi)
            'BTC': 'MAJOR', 'ETH': 'MAJOR', 'BNB': 'MAJOR',
            'XRP': 'MAJOR', 'ADA': 'MAJOR', 'DOGE': 'MAJOR', 
            'LTC': 'MAJOR', 'BCH': 'MAJOR', 'DOT': 'MAJOR',
            
            # MID_CAP (Market cap $100M-$1B, likuiditas baik)
            'SOL': 'MID_CAP', 'AVAX': 'MID_CAP', 'MATIC': 'MID_CAP', 'LINK': 'MID_CAP',
            'ICP': 'MID_CAP', 'INJ': 'MID_CAP', 'TON': 'MID_CAP', 'ARB': 'MID_CAP',
            'NEAR': 'MID_CAP', 'OP': 'MID_CAP', 'APT': 'MID_CAP', 'SUI': 'MID_CAP',
            'ATOM': 'MID_CAP', 'ETC': 'MID_CAP', 'FIL': 'MID_CAP', 'RNDR': 'MID_CAP',
            'ENA': 'MID_CAP','TIA': 'MID_CAP','ORCA': 'MID_CAP','ARC': 'MID_CAP', 
            'TRX': 'MID_CAP','AAVE': 'MID_CAP',
            
            # SMALL_CAP (Market cap $10M-$100M, likuiditas sedang)
            'SEI': 'SMALL_CAP', 'IRYS': 'SMALL_CAP', 'BONK': 'SMALL_CAP', 
            'WIF': 'SMALL_CAP', 'PYTH': 'SMALL_CAP', 'JTO': 'SMALL_CAP',
            'TAO': 'SMALL_CAP', 'STRK': 'SMALL_CAP', 'ONDO': 'SMALL_CAP','ASTER': 'SMALL_CAP',
            'TRS': 'SMALL_CAP', 'ARCON': 'SMALL_CAP',
            
            # MEME (Volatilitas sangat tinggi, driven by hype)
            'SHIB': 'MEME', 'PEPE': 'MEME', 'FLOKI': 'MEME', 
            '1000PEPE': 'MEME', '1000FLOKI': 'MEME', 'BONK': 'MEME', 'WIF': 'MEME','GIG': 'MEME',
            'MOG': 'MEME', 'TURBO': 'MEME', 'BOME': 'MEME','HYPE': 'MEME', 'GIGGLE': 'MEME', 
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

            if pd.isna(avg_atr_pct) or avg_atr_pct < 0.01 or avg_atr_pct > 10.0:
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
        elif TIMEFRAME == '1h':
            if avg_atr_pct < 0.5: return 'ULTRA_LOW'
            elif avg_atr_pct < 1.0: return 'LOW'
            elif avg_atr_pct < 2.0: return 'MODERATE'
            elif avg_atr_pct < 3.5: return 'HIGH'
            elif avg_atr_pct < 5.0: return 'VERY_HIGH'
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
    
    def is_symbol_trading(self, symbol):
        try:
            market = self.exchange.market(symbol)
            return market.get('active', False) and market['info'].get('status') == 'TRADING'
        except:
            return False

    def get_trending_symbols(self):
        print("🔍 MENGAMBIL DAFTAR ASET TRENDING DARI BINANCE...")
        self._rate_limit()

        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = {}
            for symbol, data in tickers.items():
                
                if not symbol.isascii():
                    continue
                if symbol not in self.exchange.markets:
                    continue

                base_pair = symbol.split(':')[0]
                if not base_pair.endswith('/USDT'):
                    continue

                if not self.is_symbol_trading(symbol):
                    print(f"⚠️ {symbol} tidak aktif")
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
                    vol = df['volume']
                    if vol.isna().all() or vol.mean() == 0:
                        volume_score = 50
                    else:
                        latest = vol.iloc[-1]
                        mean_val = vol.mean()
                        if np.isnan(latest) or np.isnan(mean_val) or mean_val == 0:
                            volume_score = 50
                        else:
                            volume_score = min(100, max(0, (latest / mean_val) * 12))
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
        """Dapatkan aset terbaik dengan skor aktivitas TERTINGGI yang memiliki setup probabilitas tinggi"""
        trending_symbols = self.get_trending_symbols()
        ranked_assets = self.rank_symbols_by_activity(trending_symbols)
        
        if not ranked_assets:
            print("❌ Tidak ada aset yang bisa dianalisis!")
            return None
        
        scores = [asset['activity_score'] for asset in ranked_assets]
        median_score = np.median(scores) if scores else 0
        dynamic_threshold = max(22.0, median_score * 1.1)  # Minimal 22.0
        
        print(f"📊 Threshold Dinamis: {dynamic_threshold:.1f} | Median Skor: {median_score:.1f}")
        
        # Filter aset qualified
        qualified_assets = [asset for asset in ranked_assets if asset['activity_score'] >= dynamic_threshold][:5]

        if not qualified_assets:
            # Fallback: gunakan aset teratas meskipun skornya rendah
            best_asset = ranked_assets[0]
            print(f"⚠️ Tidak ada aset qualified, menggunakan: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol'],best_asset['activity_score']
        
        print(f"🔍 Menganalisis {len(qualified_assets)} aset qualified untuk high probability setup...")
        
        high_potential_assets = []
        for asset in qualified_assets:
            symbol = asset['symbol']
            print(f"   📊 Menganalisis setup {symbol}...")
            
            # Ambil data singkat (50 candle) untuk analisis setup
            df = fetch_ohlcv_data(symbol, TIMEFRAME, 50)
            if df is None or len(df) < 30:
                continue

            try:
                df['ema20'] = talib.EMA(df['close'], 20)
                df['ema50'] = talib.EMA(df['close'], 50)
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                df['rsi'] = talib.RSI(df['close'], 14)
                df['volume_ma20'] = df['volume'].rolling(20).mean()
            except Exception as e:
                print(f"   ⚠️ Error menghitung indikator untuk {symbol}: {e}")
                continue
            
            # Hitung directional bias dan setup probability
            directional_bias =self.calculate_directional_bias(symbol, df)
            confidence_score = self.calculate_setup_confidence(df, directional_bias)
            high_potential_assets.append({
                'symbol': symbol,
                'activity_score': asset['activity_score'],
                'directional_bias': directional_bias,
                'confidence_score': confidence_score,
                'setup_type': 'BULLISH' if directional_bias > 0.3 else 'BEARISH'
            })

            if high_potential_assets:
                high_potential_assets.sort(key=lambda x: x['confidence_score'], reverse=True)
                best_asset = high_potential_assets[0]
                return best_asset['symbol'], best_asset['activity_score']  # RETURN TUPLE
            
            if qualified_assets:
                best_asset = qualified_assets[0]
                return best_asset['symbol'], best_asset['activity_score']  # RETURN TUPLE
            
            best_asset = ranked_assets[0]
            return best_asset['symbol'], best_asset['activity_score']  # RETURN TUPLE


            # if has_setup:
            #     # Hitung confidence score berdasarkan multiple faktor
            #     confidence_score = self.calculate_setup_confidence(df, directional_bias)

            #     print(f"   ✅ {symbol} - SETUP DITEMUKAN! Confidence: {confidence_score:.2f} | Bias: {directional_bias:+.2f}")
            # else:
            #     print(f"   ⚪ {symbol} - Tidak ada setup probabilitas tinggi")
        
        # Pilih aset terbaik berdasarkan confidence score
        if high_potential_assets:
            # Urutkan berdasarkan confidence_score (tertinggi ke terendah)
            high_potential_assets.sort(key=lambda x: x['confidence_score'], reverse=True)
            best_asset = high_potential_assets[0]
            print(f"\n🎯 ASET TERBAIK DENGAN HIGH PROBABILITY SETUP: {best_asset['symbol']}")
            print(f"   💯 Confidence Score: {best_asset['confidence_score']:.2f}")
            print(f"   📈 Directional Bias: {best_asset['directional_bias']:+.2f} ({best_asset['setup_type']})")
            print(f"   ⚡ Activity Score: {best_asset['activity_score']:.1f}")
            return best_asset['symbol']
        
        # Fallback: jika tidak ada setup probabilitas tinggi, gunakan aset dengan activity score tertinggi
        # best_asset = qualified_assets[0]
        # print(f"\n🎯 TIDAK ADA HIGH PROBABILITY SETUP - Menggunakan aset dengan activity score tertinggi: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
        # return best_asset['symbol']
        if qualified_assets:
            best_asset = qualified_assets[0]
            print(f"🎯 ASET TERBAIK: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']
        else:
            best_asset = ranked_assets[0]
            print(f"⚠️ Tidak ada aset qualified, menggunakan: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']
        

    def has_high_probability_setup(self, df, directional_bias):
        """Cek apakah pola candle mendukung entry"""
        if len(df) < 20:
            return False
        
        # 1. Cari breakout level
        recent_high = df['high'].iloc[-5:].max()
        recent_low = df['low'].iloc[-5:].min()
        
        # 2. Konfirmasi dengan volume & momentum
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume_ma20'].iloc[-1]
        adx = df['adx'].iloc[-1]
        
        # Bullish Setup
        if directional_bias > 0.3:
            broke_resistance = df['close'].iloc[-1] > recent_high * 1.002
            volume_confirmation = current_vol > avg_vol * 1.3
            momentum_ok = adx > 18 and df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1]
            return broke_resistance and (volume_confirmation or momentum_ok)
        
        # Bearish Setup
        elif directional_bias < -0.3:
            broke_support = df['close'].iloc[-1] < recent_low * 0.998
            volume_confirmation = current_vol > avg_vol * 1.3
            momentum_ok = adx > 18 and df['minus_di'].iloc[-1] > df['plus_di'].iloc[-1]
            return broke_support and (volume_confirmation or momentum_ok)
        
        return False
    
    def calculate_setup_confidence(self, df, directional_bias):
        """Hitung confidence score untuk setup (0.0 - 1.0)"""
        score = 0.0
        max_score = 0.0
        
        # 1. Volatilitas (ATR%)
        atr = df['atr'].iloc[-1]
        current_price = df['close'].iloc[-1]
        atr_pct = (atr / current_price) * 100
        max_score += 0.3
        if 0.15 <= atr_pct <= 0.8:  # Volatilitas ideal
            score += 0.3
        elif atr_pct > 0.8:  # Terlalu volatile
            score += 0.1
        
        # 2. Volume confirmation
        current_vol = df['volume'].iloc[-1]
        avg_vol = df['volume_ma20'].iloc[-1]
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        max_score += 0.3
        if vol_ratio > 1.8:
            score += 0.3
        elif vol_ratio > 1.4:
            score += 0.2
        
        # 3. Momentum strength (ADX + DI)
        adx = df['adx'].iloc[-1]
        plus_di = df['plus_di'].iloc[-1]
        minus_di = df['minus_di'].iloc[-1]
        di_diff = abs(plus_di - minus_di)
        max_score += 0.4
        if adx > 25 and di_diff > 10:
            score += 0.4
        elif adx > 20 and di_diff > 8:
            score += 0.3
        
        # Normalisasi ke 0.0-1.0
        confidence = score / max_score if max_score > 0 else 0.0
        
        # Adjusment berdasarkan directional bias strength
        bias_strength = abs(directional_bias)
        if bias_strength > 0.5:
            confidence *= 1.2  # Booster untuk bias kuat
        elif bias_strength < 0.2:
            confidence *= 0.7  # Penalty untuk bias lemah
        
        return min(1.0, confidence)


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
        # vol_ratio = df['volume'].iloc[-1] / df['volume_ma20'].iloc[-1]
        # if vol_ratio > 1.5 and df['close'].iloc[-1] > df['open'].iloc[-1]:
        #     vol_bias = 0.3  # Volume beli kuat
        # elif vol_ratio > 1.5 and df['close'].iloc[-1] < df['open'].iloc[-1]:
        #     vol_bias = -0.3  # Volume jual kuat
        # else:
        #     vol_bias = 0

        last_vol = df['volume'].iloc[-1]
        last_ma20 = df['volume_ma20'].iloc[-1]
        last_close = df['close'].iloc[-1]
        last_open = df['open'].iloc[-1]

        if pd.isna(last_vol) or pd.isna(last_ma20) or last_ma20 == 0:
            print('ini nol kawan')
            vol_bias = 0
        else:
            vol_ratio = last_vol / last_ma20
            print(f'vol ratio: {vol_ratio}')
            if vol_ratio > 1.5:
                vol_bias = 0.3 if last_close > last_open else -0.3
            else:
                vol_bias = 0
        
        print(f"📈 RSI Bias: {rsi_bias:.2f}, Trend Bias: {trend_bias:.2f}, OB Bias: {ob_bias:.2f}, Vol Bias: {vol_bias:.2f}")
        
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

def get_multi_timeframe_confirmation(exchange, symbol,coin_age_days=None):
    """Dapatkan konfirmasi tren dari multiple timeframe"""
    # higher_timeframes = {
    #     '4h': '4h',
    #     '1d' : '1d',
    # }

    if coin_age_days is None:
        coin_age_days = get_coin_age_days(exchange, symbol)
    
    # ✅ LOGIKA DINAMIS BERDASARKAN UMUR KOIN
    if coin_age_days < 30:  # Koin baru (<30 hari)
        print(f"🟡 KOIN BARU ({coin_age_days} hari) - Gunakan MTF alternatif")
        higher_timeframes = {
            '15m': '15m',  # Lebih responsif
            '1h': '1h',    # Timeframe utama
        }
        min_candles = 20  # Minimal candle yang diperlukan
    else:  # Koin established
        higher_timeframes = {
            '4h': '4h',
            '1d': '1d'
        }
        min_candles = 30
    
    trend_scores = []
    trend_directions = []
    
    for tf_name, tf_value in higher_timeframes.items():
        try:
            limit = 100 if tf_value == '1d' else 50
            if coin_age_days < 30:  # Koin baru (<30 hari)
                limit = min_candles
            
            ohlcv = exchange.fetch_ohlcv(symbol, tf_value, limit)
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

            weight = 0.7 if tf_value == '1d' else 0.3
            
            # Hitung score tren untuk timeframe ini
            tf_score = 0
            tf_direction = 0  # 1 = bull, -1 = bear, 0 = neutral
            
            # 1. Kekuatan tren (ADX)
            # if adx > 25:
            #     tf_score += 0.4
            # elif adx > 20:
            #     tf_score += 0.3
            # elif adx > 15:
            #     tf_score += 0.2

            adx_threshold = 20 if tf_value == '1d' else 15
            if adx > adx_threshold + 10:
                tf_score += 0.5 * weight
            elif adx > adx_threshold + 5:
                tf_score += 0.3 * weight
            elif adx > adx_threshold:
                tf_score += 0.2 * weight
                
            # 2. Arah tren (DI+ vs DI-)
            # if plus_di > minus_di + 5 and plus_di > 20:
            #     tf_score += 0.3
            #     tf_direction = 1
            # elif minus_di > plus_di + 5 and minus_di > 20:
            #     tf_score += 0.3
            #     tf_direction = -1
            di_threshold = 25 if tf_value == '1d' else 20
            if plus_di > minus_di + 5 and plus_di > di_threshold:
                tf_score += 0.4 * weight
                tf_direction = 1
            elif minus_di > plus_di + 5 and minus_di > di_threshold:
                tf_score += 0.4 * weight
                tf_direction = -1
                
            # 3. EMA alignment
            # if last_row['ema20'] > last_row['ema50']:
            #     tf_score += 0.2
            #     if tf_direction == 0:
            #         tf_direction = 1
            # elif last_row['ema20'] < last_row['ema50']:
            #     tf_score += 0.2
            #     if tf_direction == 0:
            #         tf_direction = -1
            if last_row['ema20'] > last_row['ema50'] * 1.01:  # Lebih ketat untuk daily
                tf_score += 0.3 * weight
                if tf_direction == 0:
                    tf_direction = 1
            elif last_row['ema20'] < last_row['ema50'] * 0.99:
                tf_score += 0.3 * weight
                if tf_direction == 0:
                    tf_direction = -1
            
             # 4. ✅ Daily close position (HANYA UNTUK 1D)
            if tf_value == '1d':
                daily_close = last_row['close']
                daily_open = tf_df['open'].iloc[-1]
                if daily_close > daily_open * 1.02:  # Strong bullish candle
                    tf_score += 0.2 * weight
                elif daily_close < daily_open * 0.98:  # Strong bearish candle
                    tf_score += 0.2 * weight
                    
            trend_scores.append(tf_score)
            trend_directions.append(tf_direction)
            
            # print(f"   📈 {tf_name.upper()}: Score={tf_score:.2f} | Arah={tf_direction} | ADX={adx:.1f} | DI+={plus_di:.1f}/DI-={minus_di:.1f}")
            
        except Exception as e:
            # print(f"   ⚠️ Error {tf_name}: {e}")
            continue
    
    if not trend_scores:
        return 0.5, 0  # Default neutral
    
    # Hitung average score dan konsensus arah
    # avg_score = sum(trend_scores) / len(trend_scores)
    # direction_consensus = sum(trend_directions) / len(trend_directions) if trend_directions else 0
    total_weight = sum([0.7 if tf == '1d' else 0.3 for tf in higher_timeframes.keys()])
    weighted_score = sum(score * (0.7 if i == 0 else 0.3) for i, score in enumerate(trend_scores)) / total_weight
    
    # Normalisasi direction consensus (-1 to 1)
    # direction_consensus = max(-1, min(1, direction_consensus))
    direction_consensus = trend_directions[0] if len(trend_directions) > 0 else 0
    
    return weighted_score, direction_consensus

def get_coin_age_days(exchange, symbol):
    """Hitung umur koin dalam hari sejak listing pertama"""
    try:
        # Ambil data trading terlama yang tersedia
        earliest_ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=1, since=0)  # since=0 = dari awal
        
        if earliest_ohlcv and len(earliest_ohlcv) > 0:
            first_candle_time = earliest_ohlcv[0][0]  # timestamp pertama
            first_date = datetime.fromtimestamp(first_candle_time / 1000)
            current_date = datetime.now()
            age_days = (current_date - first_date).days
            return max(1, age_days)  # minimal 1 hari
        
    except Exception as e:
        print(f"Error cek umur {symbol}: {e}")
        # Fallback: cek dari metadata exchange
        market = exchange.market(symbol)
        if 'info' in market and 'listingTime' in market['info']:
            listing_time = int(market['info']['listingTime']) / 1000
            listing_date = datetime.fromtimestamp(listing_time)
            return max(1, (datetime.now() - listing_date).days)
    
    return 365  # Default: asumsi koin lama jika gagal deteksi

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

    LOG_FILENAME = f"forward_test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
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
    oi_confirmed_long = False
    oi_confirmed_short = False
    last_entry_time = None      # waktu entry terakhir (untuk cooldown)
    last_entry_symbol = None    # simbol yang terakhir di-entry

    oi_state = {
                    'last_oi': None,
                    'long': False,
                    'short': False,
                    'threshold': 2.0 
                }

    print("🔄 MENCARI ASET TERBAIK UNTUK TRADING AWAL...")
    current_symbol, current_score = scanner.get_best_asset_for_trading()
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
    df['ema20'] = talib.EMA(df['close'], 20)
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

            if TIMEFRAME == '1h':
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
                new_score = None
                new_symbol, new_score = scanner.get_best_asset_for_trading()
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
                                print(f"⚠️ Simpan data posisi aktif sebelum switch aset...")
                                old_score = active_position.get('activity_score_at_entry', 0)
                                hold_duration = (current_time - active_position['entry_time']).total_seconds() / 60
                                current_price = get_current_price(scanner.exchange, current_symbol)
                                if current_price is None:
                                    print(f"⚠️ Gagal ambil harga untuk {current_symbol}, batalkan force close")
                                    continue

                                if active_position['side'] == 'LONG':
                                    current_pnl_pct = ((current_price - active_position['entry_price']) / active_position['entry_price']) * 100
                                else:
                                    current_pnl_pct = ((active_position['entry_price'] - current_price) / active_position['entry_price']) * 100
                                
                                # new_score = ranked_assets[0]['activity_score'] if ranked_assets else 0

                                criteria_met = 0
                                force_reasons = []
                                force_val = False

                                
                                if new_score >= 30.0 and old_score <= 22.0 and (new_score - old_score) >= 8.0:
                                    criteria_met += 1
                                    force_reasons.append(f"Skor lebih tinggi ({new_score:.1f} vs {old_score:.1f})")
                                else :
                                    print('skor rendah')
                                
                                if TIMEFRAME == '1h':
                                    timeframe_minutes = 60
                                    base_hold = 240  # 4 jam (default untuk 1h)
                                    max_hold = max(base_hold, 2.0 * (60 / timeframe_minutes))  # Min 4 jam
                                elif TIMEFRAME == '15m':
                                    timeframe_minutes = 15
                                    base_hold = 25   # 25 menit (default untuk 15m)
                                    max_hold = max(base_hold, 1.5 * (60 / timeframe_minutes))
                                else:  # 5m
                                    timeframe_minutes = 5
                                    base_hold = 15   # 15 menit
                                    max_hold = max(base_hold, 1.5 * (60 / timeframe_minutes))

                                if hold_duration > max_hold:
                                    force_val = True
                                    force_reasons.append(f"Hold terlalu lama ({hold_duration:.0f}/{max_hold:.0f} menit)")
                                else :
                                    print('masih belum ges')

                                # current_regime = scanner.detect_market_regime(df)
                                temp_df = fetch_ohlcv_data(new_symbol, TIMEFRAME, 50)
                                if temp_df is not None and len(temp_df) >= 30:
                                    current_regime = scanner.detect_market_regime(temp_df)
                                else:
                                    current_regime = "NEUTRAL"
                                
                                old_regime = active_position.get('market_regime', 'NEUTRAL')
                                regime_map = {'STRONG_BULL': 1.0, 'BULL': 0.5, 'NEUTRAL': 0.0, 'BEAR': -0.5, 'STRONG_BEAR': -1.0}
                                regime_threshold = 0.6 if TIMEFRAME == '1h' else 0.4

                                if abs(regime_map[current_regime] - regime_map[old_regime]) >= regime_threshold:
                                    criteria_met += 1
                                    force_reasons.append(f"Regime berubah ({old_regime} → {current_regime})")
                                else:
                                    print('regime tidak berubah')

                                # if abs(regime_map[current_regime] - regime_map[old_regime]) >= 0.4:
                                #     criteria_met += 1
                                #     force_reasons.append(f"Regime berubah ({old_regime} → {current_regime})")
                                # else:
                                #     print('regime tidak berubah')

                                # expected_rr = TP_ATR_MULT / SL_ATR_MULT  # Contoh: 3.0 / 1.5 = 2.0
                                # expected_pnl_pct = 1.0 * expected_rr if active_position['side'] == 'LONG' else 1.0

                                risk_per_unit = abs(active_position['entry_price'] - active_position['sl_price']) * LEVERAGE
                                reward_per_unit = abs(active_position['tp_price'] - active_position['entry_price']) * LEVERAGE
                                expected_rr = reward_per_unit / risk_per_unit if risk_per_unit > 0 else 1.0
                                expected_pnl_pct = expected_rr * abs(active_position['entry_price'] - active_position['sl_price']) / active_position['entry_price'] * 100
                                
                                if abs(current_pnl_pct) >= 0.5 and abs(current_pnl_pct) >= 0.5 * expected_pnl_pct:
                                    criteria_met += 1
                                    force_reasons.append(f"PnL optimal ({current_pnl_pct:+.2f}%)")

                                if force_val :
                                    should_force_close = True
                                else:
                                    should_force_close = (criteria_met >= 2)

                                print('force close : ',should_force_close)

                                if should_force_close:
                                    print(f"🔄 [{current_time.strftime('%H:%M:%S')}] POSISI LAMA DITUTUP OTOMATIS: {current_symbol} → BERPINDAH KE {new_symbol}")
                                    # Tutup posisi lama
                                    current_price = get_current_price(scanner.exchange, current_symbol)
                                    if current_price is not None:
                                        # Hitung PnL & tutup
                                        if active_position['side'] == 'LONG':
                                            pnl = (current_price - active_position['entry_price']) * active_position['qty'] * LEVERAGE
                                        else:
                                            pnl = (active_position['entry_price'] - current_price) * active_position['qty'] * LEVERAGE

                                        position_value = active_position['entry_price'] * active_position['qty']
                                        fee_cost = position_value * 0.001 * 2
                                        slippage_cost = position_value * SLIPPAGE_RATE * LEVERAGE
                                        market_impact = calculate_market_impact(scanner.exchange, current_symbol, active_position['qty'], position_value)
                                        impact_cost = position_value * market_impact
                                        net_pnl = pnl - fee_cost - slippage_cost - impact_cost
                                        balance = balance + net_pnl

                                        # Log exit
                                        trade_result = {
                                            'exit_time': datetime.now(),
                                            'exit_reason': 'FORCED_SWITCH',
                                            'exit_price': current_price,
                                            'gross_pnl': pnl,
                                            'fee_cost': fee_cost,
                                            'slippage_cost': slippage_cost,
                                            'impact_cost': impact_cost,
                                            'net_pnl': net_pnl,
                                            'balance_after_exit': balance,
                                            'hold_time': (datetime.now() - active_position['entry_time']).total_seconds() / 60
                                        }
                                        trade_log.append({**active_position, **trade_result})
                                        log_exit_to_excel(trade_log[-1], LOG_FILENAME)
                                        send_telegram_message(f"🔄 <b>FORCED EXIT</b>\nCoin: {active_position['symbol']}\nReason: {force_reasons[0]} \nPnL Net: {net_pnl:+.4f}")

                                        print(f"💰 Forced Close PnL: {net_pnl:+.4f} | Balance: {balance:.4f}")
                                        active_position = None
                                        last_exit_time = datetime.now()

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
                                            threshold_map = {
                                                'MAJOR': 1.2,        # perubahan OI signifikan
                                                'MID_CAP': 0.5,      # volatilitas sedang
                                                'SMALL_CAP': 0.15,    # perubahan OI lebih halus
                                                'MEME': 0.3,         # OI sangat stabil
                                                'DEFAULT': 0.15       # Fallback aman untuk semua aset
                                            }
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
                                    threshold_map = {
                                        'MAJOR': 1.2,        # perubahan OI signifikan
                                        'MID_CAP': 0.5,      # volatilitas sedang
                                        'SMALL_CAP': 0.15,    # perubahan OI lebih halus
                                        'MEME': 0.3,         # OI sangat stabil
                                        'DEFAULT': 0.15       # Fallback aman untuk semua aset
                                    }
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
                                    
                                # print(f"⏳ [{current_time.strftime('%H:%M:%S')}] POSISI AKTIF MASIH DIPERTAHANKAN. Tidak force-switch ke {new_symbol} (Skor baru: {new_score:.1f})")
                                # # Tetap di aset lama, jangan ganti
                                # last_scan_time = current_time
                                # continue
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
                        threshold_map = {
                            'MAJOR': 1.2,        # perubahan OI signifikan
                            'MID_CAP': 0.5,      # volatilitas sedang
                            'SMALL_CAP': 0.15,    # perubahan OI lebih halus
                            'MEME': 0.3,         # OI sangat stabil
                            'DEFAULT': 0.15       # Fallback aman untuk semua aset
                        }
                        oi_state['threshold'] = threshold_map.get(asset_class, 1.5)

                        prev_close = df['close'].iloc[-2]
                        current_close = df['close'].iloc[-1]
                        # current_oi = get_open_interest(scanner.exchange, current_symbol)
                        atr_value = df['atr'].iloc[-1]  

                        # if asset_class == 'MAJOR':
                        #     print(f"OI terbaru untuk {current_symbol}: {current_oi} dan sebelumnya: {oi_state['last_oi']}")

                        #     long_confirm, short_confirm = get_oi_confirmation(df, current_oi, oi_state['last_oi'], atr_value,oi_state['threshold'])
                        #     oi_state['long'] = long_confirm
                        #     oi_state['short'] = short_confirm

                        #     if current_oi is not None:
                        #         oi_state['last_oi'] = current_oi
                        #     else:
                        #         print('data OI belum lengkap')
                            
                        #     oi_confirmed_long = long_confirm
                        #     oi_confirmed_short = short_confirm
                        # else:
                        #     print('OI di SKIP karena asset bukan MAJOR')
                        #     i_confirmed_long = True
                        #     oi_confirmed_short = True

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

                    log_exit_to_excel(trade_log[-1],LOG_FILENAME)

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
            ema_20 = current_row['ema_20']
            recent_atr_pct = df['atr_pct'].iloc[-100:].quantile(0.3)
            dynamic_atr_threshold = max(0.15, recent_atr_pct * 0.8)
            # atr_threshold = dynamic_thresholds['atr_threshold']
            atr_threshold = dynamic_atr_threshold
            volume_multiplier = dynamic_thresholds['volume_multiplier']
            adx_threshold = dynamic_thresholds['adx_threshold']
            level_multiplier = dynamic_thresholds['level_multiplier']
            risk_pct = dynamic_thresholds['risk_pct']

            # print(f"[DEBUG] Volume Multiplier Dinamis: {volume_multiplier:.2f}x | Vol Ratio Saat Ini: {vol_ratio:.2f}x | Vol Confirmed: {vol_ratio >= volume_multiplier}")

            vol_confirmed = vol_ratio >= (volume_multiplier * 0.9)
            atr_confirmed = atr_pct >= atr_threshold
            atr_50 = df['atr'].rolling(50).mean().iloc[i] if i >= 50 else atr  # Fallback jika data kurang
            is_sideways = (
                current_row['adx'] < 22 and  # ADX rendah = sideways
                current_row['atr'] < (atr_50 * 0.8)  # Volatilitas menyempit
            )
            is_strong_trend = current_row['adx'] > 25
            is_uptrend = (ema_fast > ema_slow * 1.003) and (close > ema_slow)
            is_downtrend = (ema_fast < ema_slow * 0.997) and (close < ema_slow)

            high_quality_long = False
            high_quality_short = False
            mtf_score, mtf_direction = get_multi_timeframe_confirmation(scanner.exchange, current_symbol)

            market_regime = "NEUTRAL"
            allow_long = True
            allow_short = True

            if current_row['adx'] > 30:
                if ema_fast > ema_slow * 1.01:  # Uptrend kuat
                    market_regime = "STRONG_BULL"
                elif ema_fast < ema_slow * 0.99:  # Downtrend kuat
                    market_regime = "STRONG_BEAR"
            elif current_row['adx'] > 20:
                if ema_fast > ema_slow:
                    market_regime = "BULL"
                elif ema_fast < ema_slow:
                    market_regime = "BEAR"

            if market_regime in ['STRONG_BULL', 'BULL']:
                allow_short = False  # Jangan short di bull market
                print(f"🚫 SHORT DISABLED - Market Regime: {market_regime}")
            elif market_regime in ['STRONG_BEAR', 'BEAR']:
                allow_long = False  # Jangan long di bear market
                print(f"🚫 LONG DISABLED - Market Regime: {market_regime}")

            
            if is_sideways :
                print('trade terindikasi sideways')
            
            if is_strong_trend:
                print('trade terindikasi strong trend')
                if is_uptrend:
                    print('trade terindikasi uptrend')
                elif is_downtrend:
                    print('trade terindikasi downtrend')
            


            if is_sideways:
                # Hitung level breakout (gunakan Bollinger Band jika tersedia, else swing high/low)
                if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
                    long_level = df['bb_upper'].iloc[i-1]
                    short_level = df['bb_lower'].iloc[i-1]
                else:
                    # Fallback ke swing levels seperti kode lama
                    lookback_bars = 8 if TIMEFRAME == '1h' else 15  # Lebih pendek untuk 1h
                    lookback_start_swing = max(0, i - lookback_bars)
                    # lookback_start_swing = max(0, i - 15)
                    lookback_end_swing = i - 2
                    
                    if lookback_end_swing > lookback_start_swing:
                        swing_high = df['high'].iloc[lookback_start_swing:lookback_end_swing].max()
                        swing_low = df['low'].iloc[lookback_start_swing:lookback_end_swing].min()
                    else:
                        swing_high = df['high'].iloc[:i-2].max()
                        swing_low = df['low'].iloc[:i-2].min()
                    
                    long_level = swing_high
                    short_level = swing_low
                
                # # Breakout confirmation
                # broke_resistance = close > long_level * 1.002  # Break minimal 0.2%
                # broke_support = close < short_level * 0.998    # Break minimal 0.2%

                buffer_pct = 0.005 if TIMEFRAME == '1h' else 0.002  # 0.5% untuk 1h
                broke_resistance = close > long_level * (1 + buffer_pct)
                broke_support = close < short_level * (1 - buffer_pct)

                # volume_confirmed = volume > (current_row['volume_ma20'] * 1.5)

                volume_multiplier = 1.8 if TIMEFRAME == '1h' else 1.5  # Lebih tinggi untuk 1h
                volume_confirmed = volume > (current_row['volume_ma20'] * volume_multiplier)
                
                # RSI filter untuk breakout
                # rsi_long_ok = current_row['rsi'] < 75  # Hindari overbought ekstrem
                # rsi_short_ok = current_row['rsi'] > 25  # Hindari oversold ekstrem
                rsi = current_row['rsi']

                # rsi_long_ok = 45 <= rsi <= 75
                # rsi_short_ok = 25 <= rsi <= 55

                rsi_long_ok = rsi <= 80 
                rsi_short_ok = rsi >= 20

                # MTF confirmation (lebih longgar untuk breakout)
                # mtf_ok_long = mtf_direction > -0.1  # Izinkan netral untuk long
                # mtf_ok_short = mtf_direction < 0.1  # Izinkan netral untuk short

                mtf_ok_long = mtf_direction > -0.05  # Lebih ketat untuk 1h
                mtf_ok_short = mtf_direction < 0.05
                
                # Entry condition untuk breakout mode
                if broke_resistance and volume_confirmed and rsi_long_ok and mtf_ok_long and allow_long:
                    high_quality_long = True
                    print(f"🔥 BREAKOUT MODE - LONG SIGNAL: {current_symbol} @ {close:.6f}")
                    
                if broke_support and volume_confirmed and rsi_short_ok and mtf_ok_short and allow_short:
                    high_quality_short = True
                    print(f"🔥 BREAKOUT MODE - SHORT SIGNAL: {current_symbol} @ {close:.6f}")
            
            elif is_strong_trend:
                # LONG di Uptrend
                buffer_pullback = 0.008 if TIMEFRAME == '1h' else 0.004  # 0.8% untuk 1h
                # ema20 = df['ema20'].iloc[i]
                ema20 = ema_fast 
                if is_uptrend and allow_long:                    
                    pullback_ok = (
                        (df['low'].iloc[i] <= ema20 * (1 + buffer_pullback)) and
                        (close >= ema20 * (1 - buffer_pullback/2))
                    )

                    volume_ok = volume < (current_row['volume_ma20'] * 0.9)

                    rsi_ok = current_row['rsi'] < 70  # Tidak terlalu overbought
                    di_ok = current_row['plus_di'] > (current_row['minus_di'] + 3)  # Momentum positif
                    
                    # MTF confirmation (lebih ketat untuk trend continuation)
                    mtf_ok = mtf_direction > 0.05  # Minimal bullish bias
                    
                    if pullback_ok and volume_ok and rsi_ok and di_ok and mtf_ok:
                        high_quality_long = True
                        print(f"📈 TREND MODE - LONG SIGNAL: {current_symbol} @ {close:.6f}")
                
                # SHORT di Downtrend
                elif is_downtrend and allow_short:   
                    pullback_ok = (
                        (df['high'].iloc[i] >= ema20 * (1 - buffer_pullback)) and
                        (close <= ema20 * (1 + buffer_pullback/2))
                    )

                    volume_ok = volume < (current_row['volume_ma20'] * 0.9)

                    rsi_ok = current_row['rsi'] > 30  # Tidak terlalu oversold
                    di_ok = current_row['minus_di'] > (current_row['plus_di'] + 3)  # Momentum negatif
                    
                    mtf_ok = mtf_direction < -0.05  # Minimal bearish bias
                    
                    if pullback_ok and volume_ok and rsi_ok and di_ok and mtf_ok:
                        high_quality_short = True
                        print(f"📉 TREND MODE - SHORT SIGNAL: {current_symbol} @ {close:.6f}")

            current_price = get_current_price(scanner.exchange, current_symbol)
            if current_price is None:
                current_price = close  # fallback ke harga close terakhir
            
            symbol_base = current_symbol.split('/')[0]
            asset_class = scanner.asset_classification.get(symbol_base, 'DEFAULT')

            TP_MULT_MAP = {
                'MAJOR': 1.8,      # Aset besar, lebih stabil
                'MID_CAP': 2.2,    # Aset menengah
                'SMALL_CAP': 3.0,  # Aset kecil, lebih volatil
                'MEME': 4.0,       # Aset meme, sangat volatil
                'DEFAULT': 2.5
            }

            SL_MULT_MAP = {
                'MAJOR': 1.2,
                'MID_CAP': 1.5,
                'SMALL_CAP': 2.0,
                'MEME': 2.5,
                'DEFAULT': 1.8
            }

            tp_mult = TP_MULT_MAP.get(asset_class, 2.5)
            sl_mult = SL_MULT_MAP.get(asset_class, 1.8)

            if active_position is None:
                time_since_last_entry = (current_time - last_entry_time).total_seconds() / 60 if last_entry_time else float('inf')
                if last_entry_symbol == current_symbol and time_since_last_entry < RESCAN_INTERVAL_MINUTES:
                    send_telegram_message(f"entry ditunda karena suda ada sesi sebelumnya. {current_symbol} baru di-trade {time_since_last_entry:.0f} menit lalu. Tunggu hingga {RESCAN_INTERVAL_MINUTES} menit.")
                    print(f"⏳ [{current_time.strftime('%H:%M:%S')}] COOLDOWN: {current_symbol} baru di-trade {time_since_last_entry:.0f} menit lalu. Tunggu hingga {RESCAN_INTERVAL_MINUTES} menit.")
                    continue
            
            if high_quality_long:
                entry_price = current_price

                sl = entry_price - atr * sl_mult
                tp = entry_price + atr * tp_mult

                entry_price = float(scanner.exchange.price_to_precision(current_symbol, entry_price))
                sl = float(scanner.exchange.price_to_precision(current_symbol, sl))
                tp = float(scanner.exchange.price_to_precision(current_symbol, tp))

                if sl >= entry_price or tp <= entry_price:
                    print(f"⚠️ Invalid SL/TP after precision: SL={sl}, TP={tp}, Entry={entry_price}")
                    continue

                # if sl <= 0 or tp <= entry_price or sl >= entry_price:
                #     print(f"⚠️ SL/TP tidak valid untuk LONG: SL={sl:.4f}, TP={tp:.4f}, Entry={entry_price:.4f}")
                #     continue
                
                qty = calculate_professional_position_size(balance, entry_price, sl, risk_pct, LEVERAGE)
                if qty > 0:
                    print(f"🔍 [{current_time.strftime('%H:%M:%S')}] Sinyal LONG DETECTED untuk {current_symbol} @ {entry_price:.4f}")
                    print(f"   📊 ATR: {atr:.4f} | SL: {sl:.4f} ({sl_mult}x) | TP: {tp:.4f} ({tp_mult}x)")
                    print(f"   📈 Kelas Aset: {asset_class} | Market Regime: {market_regime}")
                    
                    if MODE == 'simulated':
                        active_position = execute_order_simulated(current_symbol, 'LONG', qty, entry_price, sl, tp, balance, LEVERAGE, scanner.exchange)
                        last_entry_time = datetime.now()
                        last_entry_symbol = current_symbol
                        balance = active_position['balance'] 
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime
                        active_position['activity_score_at_entry'] = current_score
                        active_position['market_regime_at_entry'] = market_regime
                        active_position['expected_rr'] = TP_ATR_MULT / SL_ATR_MULT


                        send_telegram_message(f"📊 <b>ENTRY</b>\n"
                                            f"Coin: {active_position['symbol']}\n"
                                            f"Arah: {active_position['side']}\n"
                                            f"Harga: {active_position['entry_price']:.4f}\n"
                                            f"Qty: {active_position['qty']:.4f}\n"
                                            f"SL: {active_position['sl_price']:.4f}\n"
                                            f"TP: {active_position['tp_price']:.4f}\n"
                                            f"Time: {active_position['entry_time'].strftime('%H:%M:%S')}\n"
                                            f"Regime: {active_position['regime']}\n"
                                            f"RSI Current: {current_row['rsi']}\n"
                                            f"Expected RR: {active_position['expected_rr']:.2f}")
                        log_entry_to_excel(active_position, LOG_FILENAME)
                                

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
                entry_price = current_price
                sl = entry_price + atr * sl_mult
                tp = entry_price - atr * tp_mult

                entry_price = float(scanner.exchange.price_to_precision(current_symbol, entry_price))
                sl = float(scanner.exchange.price_to_precision(current_symbol, sl))
                tp = float(scanner.exchange.price_to_precision(current_symbol, tp))

                if sl <= entry_price or tp >= entry_price:
                    print(f"⚠️ Invalid SL/TP after precision: SL={sl}, TP={tp}, Entry={entry_price}")
                    continue

                # if sl <= 0 or tp <= entry_price or sl >= entry_price:
                #     print(f"⚠️ SL/TP tidak valid untuk SHORT: SL={sl:.4f}, TP={tp:.4f}, Entry={entry_price:.4f}")
                #     continue
                
                qty = calculate_professional_position_size(balance, entry_price, sl, risk_pct, LEVERAGE)
                if qty > 0:
                    print(f"🔍 [{current_time.strftime('%H:%M:%S')}] Sinyal SHORT DETECTED untuk {current_symbol} @ {entry_price:.4f}")
                    print(f"   📊 ATR: {atr:.4f} | SL: {sl:.4f} ({sl_mult}x) | TP: {tp:.4f} ({tp_mult}x)")
                    print(f"   📈 Kelas Aset: {asset_class} | Market Regime: {market_regime}")
                    
                    if MODE == 'simulated':
                        active_position = execute_order_simulated(current_symbol, 'SHORT', qty, entry_price, sl, tp, balance, LEVERAGE,scanner.exchange)
                        last_entry_time = datetime.now()
                        last_entry_symbol = current_symbol
                        balance = active_position['balance'] 
                        active_position['entry_time'] = datetime.now()
                        active_position['regime'] = regime
                        active_position['market_regime'] = market_regime
                        active_position['activity_score_at_entry'] = current_score
                        active_position['market_regime_at_entry'] = market_regime
                        active_position['expected_rr'] = TP_ATR_MULT / SL_ATR_MULT

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
    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'future'}})
    try:
        # Tambahkan sleep kecil untuk membantu menghindari rate limit
        time.sleep(0.1) 
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


def log_entry_to_excel(entry_data,filelog):
    """Simpan data entry ke file Excel"""
    # filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d')}.xlsx" # Gunakan tanggal hari ini
    filename = filelog
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            existing_df = pd.read_excel(filename, engine='openpyxl')
            updated_df = pd.concat([existing_df, df_entry], ignore_index=True)
        except FileNotFoundError:
            updated_df = df_entry

        try:
            updated_df.to_excel(filename, index=False, engine='openpyxl')
            print(f"✅ Entry log disimpan ke: {filename}")
            return  # Sukses, keluar
        except Exception as e:
            print(f"⚠️ Gagal menyimpan entry (percobaan {attempt+1}/{max_retries}): {e}")
            time.sleep(1)
    
    # Jika semua percobaan gagal
    print("❌ SEMUA PERCOBAAN GAGAL: Simpan cadangan ke file teks")
    with open(f"entry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(str(entry_data))

def log_exit_to_excel(exit_data,filelog):
    """Simpan data exit ke file Excel yang sama dengan entry"""
    # filename = f"forward_test_log_{datetime.now().strftime('%Y%m%d')}.xlsx" # Gunakan tanggal hari ini
    filename = filelog
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

def get_oi_confirmation(df, current_oi, last_oi, atr, treshold = 0.35):

    if last_oi is None or current_oi is None:
        return False, False

    oi_change_pct = ((current_oi - last_oi) / last_oi) * 100
    candle_body = abs(df['close'].iloc[-1] - df['open'].iloc[-1])
    price_up = df['close'].iloc[-1] > df['open'].iloc[-1]
    price_down = df['close'].iloc[-1] < df['open'].iloc[-1]

    # ATR lebih kecil biar sering valid
    big_candle = candle_body > (0.08 * atr)

    long_confirm = False
    short_confirm = False

    print(f"📊 Perubahan OI sekarang {oi_change_pct:.2f}% dengan treshold {treshold}")
    # Valid breakout
    if oi_change_pct > treshold and big_candle:
        if price_up:
            long_confirm = True
        if price_down:
            short_confirm = True
        print(f"🟢 OI naik {oi_change_pct:.2f}%, valid breakout → {('LONG' if price_up else 'SHORT')}")

    # If OI turun, jangan entry
    if oi_change_pct < -0.3:
        long_confirm = False
        short_confirm = False
        print(f"🔴 OI turun {oi_change_pct:.2f}% → Hindari entry")

    return long_confirm, short_confirm

# --- MAIN ---
if __name__ == "__main__":
    run_forward_test()