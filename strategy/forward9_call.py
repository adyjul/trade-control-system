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

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_LEBAKBULUS', '') 
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_GRUP_ID', '')  
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


# utamanya disini
# --- FUNGSI UTAMA FORWARD TEST DENGAN SCANNING ULANG ---
def run_signal_generator():
    print("📡 MODE: SIGNAL GENERATOR | INTERVAL: 6 JAM | TF: 1h")
    scanner = MarketScanner()
    next_scan = get_next_aligned_time(360)
    
    print(f"⏳ Scan pertama: {next_scan.strftime('%Y-%m-%d %H:%M UTC')}")
    
    while True:
        now = datetime.utcnow()
        if now >= next_scan:
            print(f"\n🔍 [{now.strftime('%H:%M')}] MEMULAI SCAN & GENERATE CALL...")
            trending = scanner.get_trending_symbols()
            ranked = scanner.rank_symbols_by_activity(trending)
            top_3 = ranked[:3]
            
            if len(top_3) == 0:
                print("⚠️ Tidak ada koin qualified. Tunggu scan berikutnya.")
                next_scan = get_next_aligned_time(360)
                time.sleep(60)
                continue
                
            signals = []
            for asset in top_3:
                sym = asset['symbol']
                df = fetch_ohlcv_data(sym, TIMEFRAME, 120)
                if df is None or len(df) < 60: continue
                
                # Hitung ATR
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
                atr = df['atr'].iloc[-1]
                close = df['close'].iloc[-1]
                
                # Tentukan arah sederhana (bisa disesuaikan dengan logika setup Anda)
                # direction = 'LONG' if close > df['ema20'].iloc[-1] else 'SHORT'
                
                # Ambil data eksternal
                ext_data = fetch_external_market_data(sym, TIMEFRAME,scanner)
               
                signal, confidence, reasons, sl_tp = professional_entry_decision(
                    sym, df, scanner, TIMEFRAME, ext_data
                )

                if signal in ['LONG', 'SHORT']:
                    # Hitung win rate kontekstual
                    symbol_base = sym.split('/')[0]
                    asset_class = scanner.asset_classification.get(symbol_base, 'DEFAULT')
                    # wr_data = calculate_contextual_win_rate(
                    #     sym, signal, sl_tp.get('rr1', 2.0), asset_class, ext_data
                    # )  
                    rr_val = sl_tp.get('rr', 2.0)  # pakai 'rr'
                    wr_data = calculate_contextual_win_rate(sym, signal, rr_val, asset_class, ext_data) 
                
                    signal_payload = {
                        'symbol': sym,
                        'timeframe': TIMEFRAME,
                        'direction': signal,
                        'entry_low': round(close * 0.999, 5),
                        'entry_high': round(close * 1.001, 5),
                        'sl': sl_tp['sl'],
                        'tp1': sl_tp['tp1'],
                        'tp2': sl_tp['tp2'],
                        'rr1': sl_tp.get('rr1', sl_tp.get('rr', 2.0)),
                        'rr2': sl_tp.get('rr2', sl_tp.get('rr', 2.0)),
                        'win_rate': wr_data['win_rate'],
                        'confidence': wr_data['confidence'],
                        'reasons': wr_data['reasons'],
                        'factors': wr_data['factors'],
                        'next_scan': get_next_aligned_time(360).strftime('%H:%M UTC')
                    }

                    send_telegram_call(signal_payload)
                else:
                    direction = 'LONG' if close > df['ema20'].iloc[-1] else 'SHORT'
                    # Generate range TP/SL
                    ranges = generate_tp_sl_ranges(close, atr, df, direction, TIMEFRAME,scanner, sym)

                    text = f"""
                    📡 <b>MODE: SIGNAL GENERATOR TANPA PENGECEKAN</b>
                    📢 <b>SIGNAL CALL [{signal.upper()}]</b>
                    🪙 <b>Coin:</b> {sym}
                    📈 <b>Direction:</b> {'🟢 LONG' if direction == 'LONG' else '🔴 SHORT'}
                    🛑 <b>SL:</b> {ranges['sl']:.5f}
                    🎯 <b>TP1:</b> {ranges['tp1']:.5f} (RR {ranges['rr1']})
                    🚀 <b>TP2:</b> {ranges['tp2']:.5f} (RR {ranges['rr2']})
                    📊 <b>Win Rate Est:</b>({confidence})
                    📜 <b>Reasons:</b> {reasons}
                    ⚠️ <i>Gunakan risk 0.5-1% per trade. Tidak ada jaminan profit.</i>
                    ⚠️ """
                    send_telegram_message(text)

                time.sleep(1.5)  # Hindari rate limit
                
            next_scan = get_next_aligned_time(360)
            print(f"⏳ Scan berikutnya: {next_scan.strftime('%Y-%m-%d %H:%M UTC')}")
            
        time.sleep(60)

def fetch_external_market_data(symbol, timeframe,scanner):
    """Ambil data makro & sentiment publik (semua gratis/public)"""
    data = {}
    try:
        # 1. Fear & Greed Index (Global)
        fg_res = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
        data['fear_greed'] = int(fg_res.json()['data'][0]['value'])
        data['fear_greed_label'] = fg_res.json()['data'][0]['value_classification']
    except:
        data['fear_greed'] = 50
        data['fear_greed_label'] = 'Neutral'

    # 2. Funding Rate (Binance Futures)
    try:
        base = symbol.split('/')[0]
        futures_sym = f"{base}USDT"
        fr_data = scanner.exchange.fapiPublicGetFundingRate({'symbol': futures_sym, 'limit': 1})
        data['funding_rate'] = float(fr_data[0]['fundingRate']) * 100
    except:
        data['funding_rate'] = 0.0

    # 3. Open Interest Change (24h proxy)
    try:
        oi_data = scanner.exchange.fapiPublicGetOpenInterest({'symbol': futures_sym})
        data['current_oi'] = float(oi_data['openInterest'])
        data['oi_change_pct'] = 0.0  # Fallback jika history OI tidak tersedia
    except:
        data['current_oi'] = 0
        data['oi_change_pct'] = 0.0

    # 4. BTC Dominance Trend (CoinGecko Free)
    try:
        btc_dom_res = requests.get('https://api.coingecko.com/api/v3/global', timeout=5)
        data['btc_dominance'] = float(btc_dom_res.json()['data']['market_cap_percentage']['btc'])
    except:
        data['btc_dominance'] = 52.0  # default historis

    # 5. Social Sentiment Proxy (Volume/Price Momentum + OB Imbalance)
    try:
        ob = scanner.exchange.fetch_order_book(symbol, limit=20)
        bid_vol = sum(b[1] for b in ob['bids'][:10])
        ask_vol = sum(a[1] for a in ob['asks'][:10])
        total = bid_vol + ask_vol
        data['ob_imbalance'] = (bid_vol - ask_vol) / total if total > 0 else 0
        data['social_sentiment_proxy'] = max(-1, min(1, data['ob_imbalance'] * 2))
    except:
        data['social_sentiment_proxy'] = 0.0

    return data


def calculate_contextual_win_rate(symbol, side, rr, asset_class, ext_data):
    """Hitung win rate realistis berdasarkan kondisi makro & mikro"""
    base_map = {'MAJOR': 0.54, 'MID_CAP': 0.51, 'SMALL_CAP': 0.48, 'MEME': 0.44, 'DEFAULT': 0.50}
    base = base_map.get(asset_class, 0.50)
    reasons = []

    # 1. R:R Adjustment
    rr_adj = max(-0.08, min(0.08, (rr - 2.0) * 0.015))
    base += rr_adj
    reasons.append(f"R:R {'+' if rr_adj>=0 else ''}{rr_adj*100:.1f}%")

    # 2. Fear & Greed
    fg = ext_data['fear_greed']
    fg_adj = (fg - 50) / 100 * 0.10 if side == 'LONG' else (50 - fg) / 100 * 0.10
    base += fg_adj
    reasons.append(f"F&G ({ext_data['fear_greed_label']}) {'+' if fg_adj>=0 else ''}{fg_adj*100:.1f}%")

    # 3. Funding Rate (Crowding Risk)
    fr = ext_data['funding_rate']
    fr_threshold = 0.10 if asset_class in ['MEME', 'SMALL_CAP'] else 0.05
    fr_adj = 0
    if side == 'LONG':
        if fr > fr_threshold: fr_adj = -0.07
        elif fr < -0.02: fr_adj = +0.05
    else:
        if fr > fr_threshold: fr_adj = +0.05
        elif fr < -0.02: fr_adj = -0.07
    base += fr_adj
    reasons.append(f"Funding {'+' if fr_adj>=0 else ''}{fr_adj*100:.1f}%")

    # 4. BTC Dominance
    dom = ext_data['btc_dominance']
    if asset_class != 'MAJOR':
        dom_adj = 0.04 if dom < 50 else -0.05 if dom > 58 else 0
        base += dom_adj
        reasons.append(f"BTC Dom {dom:.1f}% {'+' if dom_adj>=0 else ''}{dom_adj*100:.1f}%")

    # 5. Social/OB Sentiment Proxy
    sent = ext_data['social_sentiment_proxy']
    sent_adj = sent * 0.06 if side == 'LONG' else -sent * 0.06
    base += sent_adj
    reasons.append(f"OB/Sentiment {'+' if sent_adj>=0 else ''}{sent_adj*100:.1f}%")

    # Clamp realistis
    final_wr = round(max(0.32, min(0.78, base)) * 100, 1)
    confidence = "HIGH" if final_wr >= 62 else "MEDIUM" if final_wr >= 52 else "LOW"
    
    return {'win_rate': final_wr, 'confidence': confidence, 'factors': reasons}


def generate_tp_sl_ranges(entry_price, atr, df, direction, timeframe, scanner, symbol):
    """Buat range SL & TP1/TP2 berbasis struktur pasar & volatilitas"""
    lookback = 20 if timeframe == '1h' else 12
    swing_high = df['high'].iloc[-lookback:].max()
    swing_low = df['low'].iloc[-lookback:].min()
    
    if direction == 'LONG':
        sl = max(swing_low * 0.998, entry_price - atr * 1.8)
        tp1 = entry_price + atr * 1.5
        tp2 = min(swing_high * 1.01, entry_price + atr * 3.5)
    else:
        sl = min(swing_high * 1.002, entry_price + atr * 1.8)
        tp1 = entry_price - atr * 1.5
        tp2 = max(swing_low * 0.99, entry_price - atr * 3.5)
        
    # Presisi harga sesuai exchange
    sl = float(scanner.exchange.price_to_precision(symbol, sl))
    tp1 = float(scanner.exchange.price_to_precision(symbol, tp1))
    tp2 = float(scanner.exchange.price_to_precision(symbol, tp2))
    
    rr1 = abs(tp1 - entry_price) / abs(entry_price - sl) if sl != entry_price else 0
    rr2 = abs(tp2 - entry_price) / abs(entry_price - sl) if sl != entry_price else 0
    
    return {'sl': sl, 'tp1': tp1, 'tp2': tp2, 'rr1': round(rr1, 2), 'rr2': round(rr2, 2)}

def professional_entry_decision(symbol, df, scanner, timeframe='1h', ext_data=None):
    """
    Professional-grade entry decision engine
    Returns: (signal: str, confidence: float, reasons: list, sl_tp: dict)
    signal: 'LONG', 'SHORT', or 'WAIT'
    """
    if len(df) < 100:
        return 'WAIT', 0.0, ['Data insufficient'], {}
    
    reasons = []
    score = 0.0
    max_score = 0.0
    
    # ========== 1. MARKET STRUCTURE ANALYSIS ==========
    structure = analyze_market_structure(df)
    max_score += 2.0
    
    if structure['trend'] == 'BULLISH' and structure['bos_confirmed']:
        score += 1.5
        reasons.append(f"Structure: Bullish BOS @{structure['bos_level']:.4f}")
    elif structure['trend'] == 'BEARISH' and structure['bos_confirmed']:
        score -= 1.5
        reasons.append(f"Structure: Bearish BOS @{structure['bos_level']:.4f}")
    
    if structure['choch_detected']:
        score += 0.5 if structure['choch_direction'] > 0 else -0.5
        reasons.append(f"CHoCH: {'Bull' if structure['choch_direction']>0 else 'Bear'}")
    
    # ========== 2. SMART MONEY CONCEPTS ==========
    smc = detect_smc_signals(df)
    max_score += 2.0
    
    if smc['order_block_signal'] != 'NONE':
        score += 1.0 if smc['order_block_signal'] == 'BULL' else -1.0
        reasons.append(f"Order Block: {smc['order_block_signal']} @{smc['ob_price']:.4f}")
    
    if smc['fvg_signal'] != 'NONE' and smc['fvg_filled']:
        score += 0.8 if smc['fvg_signal'] == 'BULL' else -0.8
        reasons.append(f"FVG Fill: {smc['fvg_signal']}")
    
    if smc['liquidity_sweep']:
        score += 0.7 if smc['sweep_direction'] > 0 else -0.7
        reasons.append(f"Liquidity Sweep: {'Buy-side' if smc['sweep_direction']>0 else 'Sell-side'}")
    
    # ========== 3. VOLUME PROFILE & ORDER FLOW ==========
    volume_analysis = analyze_volume_profile(df)
    max_score += 1.5
    
    if volume_analysis['poc_reaction'] != 'NONE':
        score += 0.6 if volume_analysis['poc_reaction'] == 'BOUNCE_BULL' else -0.6
        reasons.append(f"POC Reaction: {volume_analysis['poc_reaction']}")
    
    if volume_analysis['absorption_detected']:
        score += 0.5 if volume_analysis['absorption_side'] == 'BULL' else -0.5
        reasons.append(f"Absorption: {volume_analysis['absorption_side']}")
    
    if volume_analysis['imbalance_ratio'] > 1.8:
        score += 0.4 if volume_analysis['imbalance_side'] == 'BULL' else -0.4
        reasons.append(f"Volume Imbalance: {volume_analysis['imbalance_side']}")
    
    # ========== 4. DIVERGENCE DETECTION ==========
    divergence = detect_divergences(df)
    max_score += 1.5
    
    if divergence['rsi_divergence'] != 'NONE':
        score += 0.7 if divergence['rsi_divergence'] == 'BULL' else -0.7
        reasons.append(f"RSI Divergence: {divergence['rsi_divergence']}")
    
    if divergence['macd_divergence'] != 'NONE' and divergence['macd_confirmed']:
        score += 0.6 if divergence['macd_divergence'] == 'BULL' else -0.6
        reasons.append(f"MACD Divergence: {divergence['macd_divergence']}")
    
    # ========== 5. MULTI-TIMEFRAME ALIGNMENT ==========
    mtf_alignment = check_mtf_alignment(scanner.exchange, symbol, df, timeframe)
    max_score += 2.0
    
    if mtf_alignment['all_bullish']:
        score += 1.8
        reasons.append("MTF: Full Bullish Alignment (15m/1h/4h)")
    elif mtf_alignment['all_bearish']:
        score -= 1.8
        reasons.append("MTF: Full Bearish Alignment (15m/1h/4h)")
    elif mtf_alignment['majority_bullish']:
        score += 0.9
        reasons.append("MTF: Majority Bullish (2/3 timeframes)")
    elif mtf_alignment['majority_bearish']:
        score -= 0.9
        reasons.append("MTF: Majority Bearish (2/3 timeframes)")
    else:
        reasons.append("MTF: Mixed/Conflicting signals")
    
    # ========== 6. CORRELATION FILTER ==========
    corr_filter = check_correlation_filter(scanner.exchange, symbol, ext_data)
    max_score += 1.0
    
    if corr_filter['btc_correlation_ok']:
        score += 0.5 if corr_filter['btc_trend'] == 'BULL' else -0.5
        reasons.append(f"BTC Correlation: {corr_filter['btc_trend']} ✓")
    else:
        score -= 0.3
        reasons.append("BTC Correlation: Conflicting ⚠️")
    
    # ========== 7. TIME/SESSION FILTER ==========
    time_filter = check_time_filter(timeframe)
    max_score += 0.5
    
    if time_filter['optimal_window']:
        score += 0.4
        reasons.append(f"Session: {time_filter['active_session']} (High Volatility)")
    elif time_filter['avoid_window']:
        score -= 0.5
        reasons.append("Session: Low liquidity / News window ⚠️")
    
    # ========== 8. DYNAMIC RISK-REWARD VALIDATION ==========
    rr_validation = validate_risk_reward(df, structure, smc)
    max_score += 1.0
    
    if rr_validation['min_rr_met']:
        score += 0.8
        reasons.append(f"RR Ratio: {rr_validation['estimated_rr']:.2f} ✓")
    else:
        score -= 0.4
        reasons.append(f"RR Ratio: {rr_validation['estimated_rr']:.2f} < 1.5 ⚠️")
    
    # ========== FINAL DECISION ==========
    normalized_score = score / max_score if max_score > 0 else 0
    confidence = round(abs(normalized_score) * 100, 1)
    
    # Thresholds profesional
    LONG_THRESHOLD = 0.35   # Score > 35% of max → LONG
    SHORT_THRESHOLD = -0.35 # Score < -35% of max → SHORT
    
    if normalized_score >= LONG_THRESHOLD and confidence >= 55:
        signal = 'LONG'
        sl_tp = calculate_structure_based_sl_tp(df, 'LONG', structure, smc)
        reasons.append(f"✅ ENTRY LONG | Confidence: {confidence}%")
    elif normalized_score <= SHORT_THRESHOLD and confidence >= 55:
        signal = 'SHORT'
        sl_tp = calculate_structure_based_sl_tp(df, 'SHORT', structure, smc)
        reasons.append(f"✅ ENTRY SHORT | Confidence: {confidence}%")
    else:
        signal = 'WAIT'
        sl_tp = {}
        reasons.append(f"⏳ WAIT | Score: {normalized_score:.2f} | Need >±0.35")
    
    return signal, confidence, reasons, sl_tp

def analyze_market_structure(df, lookback=50):
    """Deteksi Higher Highs/Lows, BOS, CHoCH"""
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    # Detect swing points
    swing_highs = []
    swing_lows = []
    for i in range(5, len(df)-5):
        if highs[i] > max(highs[i-5:i]) and highs[i] > max(highs[i+1:i+6]):
            swing_highs.append((i, highs[i]))
        if lows[i] < min(lows[i-5:i]) and lows[i] < min(lows[i+1:i+6]):
            swing_lows.append((i, lows[i]))
    
    # Trend determination
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        last_hh = swing_highs[-1][1] > swing_highs[-2][1]
        last_hl = swing_lows[-1][1] > swing_lows[-2][1]
        last_ll = swing_lows[-1][1] < swing_lows[-2][1]
        last_lh = swing_highs[-1][1] < swing_highs[-2][1]
        
        if last_hh and last_hl:
            trend = 'BULLISH'
        elif last_ll and last_lh:
            trend = 'BEARISH'
        else:
            trend = 'RANGING'
    else:
        trend = 'UNKNOWN'
    
    # BOS detection
    bos_confirmed = False
    bos_level = None
    if trend == 'BULLISH' and len(swing_highs) >= 2:
        if closes[-1] > swing_highs[-1][1]:
            bos_confirmed = True
            bos_level = swing_highs[-1][1]
    elif trend == 'BEARISH' and len(swing_lows) >= 2:
        if closes[-1] < swing_lows[-1][1]:
            bos_confirmed = True
            bos_level = swing_lows[-1][1]
    
    # CHoCH detection
    choch_detected = False
    choch_direction = 0
    if len(swing_highs) >= 3 and len(swing_lows) >= 3:
        if trend == 'BULLISH' and closes[-1] < swing_lows[-2][1]:
            choch_detected = True
            choch_direction = -1
        elif trend == 'BEARISH' and closes[-1] > swing_highs[-2][1]:
            choch_detected = True
            choch_direction = 1
    
    return {
        'trend': trend,
        'bos_confirmed': bos_confirmed,
        'bos_level': bos_level,
        'choch_detected': choch_detected,
        'choch_direction': choch_direction,
        'swing_highs': swing_highs[-3:],
        'swing_lows': swing_lows[-3:]
    }


def detect_smc_signals(df, lookback=30):
    """Deteksi Order Block, FVG, Liquidity Sweep"""
    result = {
        'order_block_signal': 'NONE',
        'ob_price': None,
        'fvg_signal': 'NONE',
        'fvg_filled': False,
        'liquidity_sweep': False,
        'sweep_direction': 0
    }
    
    # Order Block detection (simplified)
    for i in range(len(df)-lookback, len(df)-5):
        # Bullish OB: strong down candle followed by strong up move
        if df['close'].iloc[i] < df['open'].iloc[i] * 0.98:  # Strong red
            if i+3 < len(df) and df['close'].iloc[i+3] > df['open'].iloc[i]:
                result['order_block_signal'] = 'BULL'
                result['ob_price'] = df['close'].iloc[i]
                break
        # Bearish OB: strong up candle followed by strong down move
        elif df['close'].iloc[i] > df['open'].iloc[i] * 1.02:  # Strong green
            if i+3 < len(df) and df['close'].iloc[i+3] < df['open'].iloc[i]:
                result['order_block_signal'] = 'BEAR'
                result['ob_price'] = df['close'].iloc[i]
                break
    
    # Fair Value Gap detection
    for i in range(len(df)-2):
        # Bullish FVG: gap between candle 1 high and candle 3 low
        if df['low'].iloc[i+2] > df['high'].iloc[i]:
            fvg_low = df['high'].iloc[i]
            fvg_high = df['low'].iloc[i+2]
            if df['low'].iloc[-1] <= fvg_high and df['high'].iloc[-1] >= fvg_low:
                result['fvg_signal'] = 'BULL'
                result['fvg_filled'] = True
                break
        # Bearish FVG
        elif df['high'].iloc[i+2] < df['low'].iloc[i]:
            fvg_low = df['high'].iloc[i+2]
            fvg_high = df['low'].iloc[i]
            if df['low'].iloc[-1] <= fvg_high and df['high'].iloc[-1] >= fvg_low:
                result['fvg_signal'] = 'BEAR'
                result['fvg_filled'] = True
                break
    
    # Liquidity sweep detection
    recent_high = df['high'].iloc[-10:].max()
    recent_low = df['low'].iloc[-10:].min()
    if df['high'].iloc[-1] > recent_high * 1.005 and df['close'].iloc[-1] < df['high'].iloc[-1] * 0.99:
        result['liquidity_sweep'] = True
        result['sweep_direction'] = -1  # Sell-side sweep, potential reversal up
    elif df['low'].iloc[-1] < recent_low * 0.995 and df['close'].iloc[-1] > df['low'].iloc[-1] * 1.01:
        result['liquidity_sweep'] = True
        result['sweep_direction'] = 1  # Buy-side sweep, potential reversal down
    
    return result


def analyze_volume_profile(df, lookback=50):
    """Volume Profile: POC, Value Area, Absorption, Imbalance"""
    result = {
        'poc_reaction': 'NONE',
        'absorption_detected': False,
        'absorption_side': None,
        'imbalance_ratio': 1.0,
        'imbalance_side': None
    }
    
    # Simple POC (Point of Control) - highest volume price zone
    volume_by_price = df.groupby(pd.cut(df['close'], bins=20))['volume'].sum()
    poc_bin = volume_by_price.idxmax()
    poc_price = poc_bin.mid if hasattr(poc_bin, 'mid') else df['close'].iloc[-1]
    
    # Check reaction at POC
    if abs(df['close'].iloc[-1] - poc_price) / poc_price < 0.005:  # Within 0.5%
        if df['close'].iloc[-1] > df['open'].iloc[-1]:
            result['poc_reaction'] = 'BOUNCE_BULL'
        else:
            result['poc_reaction'] = 'REJECT_BEAR'
    
    # Absorption detection (high volume, small range)
    recent_candles = df.iloc[-10:]
    avg_range = recent_candles['high'].sub(recent_candles['low']).mean()
    current_range = df['high'].iloc[-1] - df['low'].iloc[-1]
    if df['volume'].iloc[-1] > recent_candles['volume'].mean() * 1.8 and current_range < avg_range * 0.6:
        result['absorption_detected'] = True
        result['absorption_side'] = 'BULL' if df['close'].iloc[-1] > df['open'].iloc[-1] else 'BEAR'
    
    # Volume imbalance (buy vs sell pressure)
    buy_vol = df[(df['close'] > df['open'])]['volume'].iloc[-20:].sum()
    sell_vol = df[(df['close'] < df['open'])]['volume'].iloc[-20:].sum()
    total = buy_vol + sell_vol
    if total > 0:
        result['imbalance_ratio'] = max(buy_vol, sell_vol) / min(buy_vol, sell_vol) if min(buy_vol, sell_vol) > 0 else 2.0
        result['imbalance_side'] = 'BULL' if buy_vol > sell_vol else 'BEAR'
    
    return result


def detect_divergences(df, period=14):
    """Detect RSI & MACD divergences"""
    result = {
        'rsi_divergence': 'NONE',
        'macd_divergence': 'NONE',
        'macd_confirmed': False
    }
    
    # RSI divergence
    df['rsi'] = talib.RSI(df['close'], period)
    for i in range(len(df)-20, len(df)-5):
        # Bullish divergence: lower low in price, higher low in RSI
        if df['low'].iloc[i] > df['low'].iloc[i-10] and df['rsi'].iloc[i] < df['rsi'].iloc[i-10]:
            if df['close'].iloc[-1] > df['close'].iloc[i]:
                result['rsi_divergence'] = 'BULL'
                break
        # Bearish divergence
        elif df['high'].iloc[i] < df['high'].iloc[i-10] and df['rsi'].iloc[i] > df['rsi'].iloc[i-10]:
            if df['close'].iloc[-1] < df['close'].iloc[i]:
                result['rsi_divergence'] = 'BEAR'
                break
    
    # MACD divergence (simplified)
    macd, signal, hist = talib.MACD(df['close'])
    for i in range(len(df)-20, len(df)-5):
        if df['low'].iloc[i] > df['low'].iloc[i-10] and hist.iloc[i] < hist.iloc[i-10]:
            if df['close'].iloc[-1] > df['close'].iloc[i] and hist.iloc[-1] > 0:
                result['macd_divergence'] = 'BULL'
                result['macd_confirmed'] = hist.iloc[-1] > hist.iloc[-3]
                break
        elif df['high'].iloc[i] < df['high'].iloc[i-10] and hist.iloc[i] > hist.iloc[i-10]:
            if df['close'].iloc[-1] < df['close'].iloc[i] and hist.iloc[-1] < 0:
                result['macd_divergence'] = 'BEAR'
                result['macd_confirmed'] = hist.iloc[-1] < hist.iloc[-3]
                break
    
    return result


def check_mtf_alignment(exchange, symbol, df, base_tf='1h'):
    """Check alignment across 3 timeframes"""
    timeframes = ['15m', '1h', '4h'] if base_tf == '1h' else ['5m', '15m', '1h']
    signals = []
    
    for tf in timeframes:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, tf, limit=50)
            tf_df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            
            # Simple trend: EMA20 vs EMA50
            tf_df['ema20'] = talib.EMA(tf_df['close'], 20)
            tf_df['ema50'] = talib.EMA(tf_df['close'], 50)
            
            last = tf_df.iloc[-1]
            if last['ema20'] > last['ema50'] * 1.002:
                signals.append(1)  # Bullish
            elif last['ema20'] < last['ema50'] * 0.998:
                signals.append(-1)  # Bearish
            else:
                signals.append(0)  # Neutral
        except:
            signals.append(0)
    
    return {
        'all_bullish': all(s == 1 for s in signals),
        'all_bearish': all(s == -1 for s in signals),
        'majority_bullish': sum(1 for s in signals if s == 1) >= 2,
        'majority_bearish': sum(1 for s in signals if s == -1) >= 2,
        'signals': signals
    }


def check_correlation_filter(exchange, symbol, ext_data=None):
    """Check BTC correlation and trend alignment"""
    result = {'btc_correlation_ok': True, 'btc_trend': 'NEUTRAL'}
    
    try:
        # Get BTC data
        btc_ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=50)
        btc_df = pd.DataFrame(btc_ohlcv, columns=['ts','o','h','l','c','v'])
        btc_df['ema20'] = talib.EMA(btc_df['c'], 20)
        btc_df['ema50'] = talib.EMA(btc_df['c'], 50)
        
        last_btc = btc_df.iloc[-1]
        if last_btc['ema20'] > last_btc['ema50'] * 1.003:
            result['btc_trend'] = 'BULL'
        elif last_btc['ema20'] < last_btc['ema50'] * 0.997:
            result['btc_trend'] = 'BEAR'
        
        # If altcoin and BTC trend is strong opposite, reduce confidence
        base = symbol.split('/')[0]
        if base not in ['BTC', 'ETH'] and ext_data:
            if ext_data.get('btc_dominance', 52) > 58 and result['btc_trend'] == 'BEAR':
                result['btc_correlation_ok'] = False
    except:
        pass
    
    return result


def check_time_filter(timeframe):
    """Check if current UTC time is in optimal trading window"""
    now = datetime.utcnow()
    hour = now.hour
    
    # Optimal windows: London open (7-10 UTC), NY open (12-15 UTC)
    optimal_sessions = [(7, 10), (12, 15)]
    avoid_sessions = [(0, 2), (5, 7)]  # Low liquidity / rollover
    
    result = {
        'optimal_window': any(start <= hour < end for start, end in optimal_sessions),
        'avoid_window': any(start <= hour < end for start, end in avoid_sessions),
        'active_session': 'London' if 7 <= hour < 12 else 'NY' if 12 <= hour < 20 else 'Asian'
    }
    return result


def validate_risk_reward(df, structure, smc, min_rr=1.5):
    """Validate that structure-based SL/TP gives minimum RR"""
    current = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'].iloc[-1] - df['low'].iloc[-1])
    
    # Estimate SL based on structure
    if structure['bos_level']:
        sl_distance = abs(current - structure['bos_level'])
    else:
        sl_distance = atr * 1.5
    
    # Estimate TP based on next structure level
    tp_distance = sl_distance * 2.2  # Conservative estimate
    
    estimated_rr = tp_distance / sl_distance if sl_distance > 0 else 0
    
    return {
        'min_rr_met': estimated_rr >= min_rr,
        'estimated_rr': round(estimated_rr, 2),
        'sl_distance': sl_distance,
        'tp_distance': tp_distance
    }


def calculate_structure_based_sl_tp(df, direction, structure, smc):
    """Calculate SL/TP based on market structure, not fixed ATR"""
    current = df['close'].iloc[-1]
    atr = df['atr'].iloc[-1] if 'atr' in df.columns else (df['high'].iloc[-1] - df['low'].iloc[-1]) * 0.5
    
    if direction == 'LONG':
        # SL: below recent swing low or order block
        # sl = min(
        #     structure['swing_lows'][-1][1] * 0.998 if structure['swing_lows'] else current - atr * 2,
        #     smc['ob_price'] * 0.995 if smc['ob_price'] and smc['order_block_signal'] == 'BULL' else current
        # )
        # # TP: above recent swing high or liquidity zone
        # tp = max(
        #     structure['swing_highs'][-1][1] * 1.005 if structure['swing_highs'] else current + atr * 4,
        #     current + atr * 3
        # )
        sl_candidates = [current - atr * 2]  # fallback
        if structure['swing_lows']:
            sl_candidates.append(structure['swing_lows'][-1][1] * 0.998)
        if smc.get('ob_price') and smc.get('order_block_signal') == 'BULL':
            sl_candidates.append(smc['ob_price'] * 0.995)
        sl = min(sl_candidates)

        tp_candidates = [current + atr * 4]
        if structure['swing_highs']:
            tp_candidates.append(structure['swing_highs'][-1][1] * 1.005)
        tp = max(tp_candidates)

    else:  # SHORT

        sl_candidates = [current + atr * 2]  # fallback
        if structure['swing_lows']:
            sl_candidates.append(structure['swing_lows'][-1][1] * 0.998)
        if smc.get('ob_price') and smc.get('order_block_signal') == 'BULL':
            sl_candidates.append(smc['ob_price'] * 0.995)
        sl = min(sl_candidates)

        tp_candidates = [current - atr * 4]
        if structure['swing_highs']:
            tp_candidates.append(structure['swing_highs'][-1][1] * 1.005)
        tp = max(tp_candidates)

        # sl = max(
        #     structure['swing_highs'][-1][1] * 1.002 if structure['swing_highs'] else current + atr * 2,
        #     smc['ob_price'] * 1.005 if smc['ob_price'] and smc['order_block_signal'] == 'BEAR' else current
        # )
        # tp = min(
        #     structure['swing_lows'][-1][1] * 0.995 if structure['swing_lows'] else current - atr * 4,
        #     current - atr * 3
        # )
    
    rr = abs(tp - current) / abs(current - sl) if sl != current else 0
    
    return {
        'sl': sl,
        'tp1': current + (tp - current) * 0.5 if direction == 'LONG' else current - (current - tp) * 0.5,
        'tp2': tp,
        'rr': round(rr, 2)
    }

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

        if len(df) >= 20:
            df['ema20'] = talib.EMA(df['close'], 20)
            df['ema50'] = talib.EMA(df['close'], 50)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)

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
def send_telegram_call(signal):
    """Kirim signal call rapi ke Telegram"""
    if not ENABLE_TELEGRAM:
        print("⚠️ Telegram dinonaktifkan atau token/kosong.")
        return
        
    text = f"""
        📢 <b>SIGNAL CALL [{signal['timeframe'].upper()}]</b>
        🪙 <b>Coin:</b> {signal['symbol'].split('/')[0]}
        📈 <b>Direction:</b> {'🟢 LONG' if signal['direction'] == 'LONG' else '🔴 SHORT'}
        💰 <b>Entry Zone:</b> {signal['entry_low']:.5f} - {signal['entry_high']:.5f}
        🛑 <b>SL:</b> {signal['sl']:.5f}
        🎯 <b>TP1:</b> {signal['tp1']:.5f} (RR {signal['rr1']})
        🚀 <b>TP2:</b> {signal['tp2']:.5f} (RR {signal['rr2']})
        📊 <b>Win Rate Est:</b> {signal['win_rate']}% ({signal['confidence']})
        🔍 <b>Context:</b> {', '.join(signal['factors'][:3])}
        ⏰ <b>Next Scan:</b> {signal['next_scan']}
        ⚠️ <i>Gunakan risk 0.5-1% per trade. Tidak ada jaminan profit.</i>
        ⚠️ <b>Confidence:</b> {signal['confidence']}</b>
        ⚠️ <b>Reason:</b> {signal['reason']}</b>
    """
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text.strip(), 'parse_mode': 'HTML'}
        r = requests.post(url, json=payload, timeout=5)
        if r.status_code == 200:
            print("✅ Call Telegram terkirim.")
        else:
            print(f"❌ Gagal kirim: {r.text}")
    except Exception as e:
        print(f"⚠️ Error Telegram: {e}")

def get_next_aligned_time(interval_minutes):
    """Hitung waktu UTC berikutnya yang merupakan kelipatan dari interval_minutes (misal 6 jam)."""
    now = datetime.utcnow()
    
    # Hitung menit sejak jam 00:00 hari ini
    minutes_since_midnight = now.hour * 60 + now.minute
    
    # Hitung berapa menit lagi sampai kelipatan berikutnya
    minutes_to_next = interval_minutes - (minutes_since_midnight % interval_minutes)
    
    # Tambahkan ke waktu sekarang, lalu reset detik/microsecond
    next_aligned = now + timedelta(minutes=minutes_to_next)
    next_aligned = next_aligned.replace(second=0, microsecond=0)
    
    # Jika hasilnya masih di masa lalu (misal karena perbedaan detik), tambah 1 interval
    if next_aligned <= now:
        next_aligned += timedelta(minutes=interval_minutes)
    
    return next_aligned


# --- MAIN ---
if __name__ == "__main__":
   run_signal_generator()