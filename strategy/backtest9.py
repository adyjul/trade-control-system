# backtest_with_market_scanner.py (PRODUCTION-GRADE COMPLETE FIX)
import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import time
import math
from functools import lru_cache

# --- CONFIG UTAMA ---
INITIAL_BALANCE = 20.0
BASE_RISK_PCT = 0.01  # 1% base risk (akan di-adjust dinamis)
LEVERAGE = 10
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.5
TIMEFRAME = '5m'
BARS_TO_FETCH = 1000
MIN_SCAN_SCORE = 0.5  # Turunkan threshold untuk lebih banyak trade
VOLATILITY_WINDOW = 50  # Window untuk kalkulasi volatilitas

# --- MARKET SCANNER CLASS (PRODUCTION-GRADE) ---
class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.min_volume_usd = 300000  # Turunkan threshold volume
        self.min_24h_change = 1.0      # Turunkan threshold pergerakan
        self.max_symbols = 20          # Perbesar max symbols
        self.last_request_time = 0
        self.request_delay = 0.5  # 0.5 detik antar request (lebih cepat tapi aman)
        
        # Default asset profiles (untuk fallback)
        self.default_profiles = {
            'MAJOR': {'atr_threshold': 0.20, 'volume_multiplier': 1.5, 'level_multiplier': 0.6, 'risk_pct': 0.008},
            'MID_CAP': {'atr_threshold': 0.25, 'volume_multiplier': 1.6, 'level_multiplier': 0.7, 'risk_pct': 0.010},
            'MEME': {'atr_threshold': 0.35, 'volume_multiplier': 1.8, 'level_multiplier': 0.8, 'risk_pct': 0.006},
            'SMALL_CAP': {'atr_threshold': 0.40, 'volume_multiplier': 2.0, 'level_multiplier': 0.9, 'risk_pct': 0.005},
            'DEFAULT': {'atr_threshold': 0.25, 'volume_multiplier': 1.6, 'level_multiplier': 0.7, 'risk_pct': 0.010}
        }
        
        # Asset classification mapping
        self.asset_classification = {
            'BTC': 'MAJOR', 'ETH': 'MAJOR', 'BNB': 'MAJOR',
            'SOL': 'MID_CAP', 'AVAX': 'MID_CAP', 'MATIC': 'MID_CAP', 'LINK': 'MID_CAP',
            'ICP': 'MID_CAP', 'INJ': 'MID_CAP', 'TON': 'MID_CAP', 'ARB': 'MID_CAP',
            'DOGE': 'MEME', 'SHIB': 'MEME', 'PEPE': 'MEME', 'FLOKI': 'MEME',
            'SEI': 'SMALL_CAP', 'SUI': 'SMALL_CAP', 'APT': 'SMALL_CAP'
        }

    def _rate_limit(self):
        """Rate limiting protection untuk exchange API"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def detect_market_regime(self, df):
        """Deteksi market regime (bull/bear/neutral)"""
        if len(df) < 200:
            return "NEUTRAL"
        
        current_price = df['close'].iloc[-1]
        ema_200 = df['close'].rolling(200).mean().iloc[-1]
        
        if pd.isna(ema_200) or ema_200 == 0:
            return "NEUTRAL"
        
        price_vs_ema = current_price / ema_200
        
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
        """Dapatkan profil karakteristik aset secara dinamis"""
        if len(df) < 50:  # Minimal 50 candle untuk analisis
            return self.get_default_asset_profile(symbol)
        
        try:
            # Hitung volatilitas historis (ATR%)
            if 'atr' not in df.columns or df['atr'].isna().all():
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            avg_atr_pct = df['atr_pct'].rolling(VOLATILITY_WINDOW).mean().iloc[-1]
            
            if pd.isna(avg_atr_pct) or avg_atr_pct == 0:
                return self.get_default_asset_profile(symbol)
            
            # Hitung volume profile
            df['vol_ma20'] = df['volume'].rolling(20).mean()
            df['vol_ma100'] = df['volume'].rolling(100).mean()
            avg_volume_20 = df['vol_ma20'].iloc[-1]
            avg_volume_100 = df['vol_ma100'].iloc[-1]
            
            if avg_volume_100 == 0 or pd.isna(avg_volume_100):
                volume_consistency = 1.0
            else:
                volume_consistency = avg_volume_20 / avg_volume_100
            
            # Hitung price range characteristics
            df['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
            avg_range_pct = df['range_pct'].rolling(50).mean().iloc[-1]
            
            if pd.isna(avg_range_pct):
                avg_range_pct = 0.35  # Default
            
            # Klasifikasi aset berdasarkan data
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
            
            print(f"📈 {symbol} Asset Profile (Dinamis):")
            print(f"   Volatilitas: {asset_profile['volatility_class']} ({avg_atr_pct:.3f}%)")
            print(f"   Volume Profile: {asset_profile['volume_class']} (Consistency: {volume_consistency:.2f}x)")
            print(f"   Rata-rata Range: {avg_range_pct:.3f}% ({asset_profile['range_class']})")
            print(f"   Symbol Class: {asset_profile['symbol_base']}")
            
            return asset_profile
            
        except Exception as e:
            print(f"🔥 Error menghitung asset profile untuk {symbol}: {e}")
            print("🔄 Falling back ke default profile...")
            return self.get_default_asset_profile(symbol)

    def get_default_asset_profile(self, symbol):
        """Fallback profile jika data historis tidak cukup"""
        symbol_base = symbol.split('/')[0].upper()
        
        # Dapatkan klasifikasi berdasarkan symbol
        asset_class = self.asset_classification.get(symbol_base, 'DEFAULT')
        profile = self.default_profiles[asset_class].copy()
        
        # Tambahkan metadata untuk debugging
        profile.update({
            'symbol_base': symbol_base,
            'asset_class': asset_class,
            'source': 'DEFAULT_FALLBACK'
        })
        
        print(f"🔄 Menggunakan DEFAULT ASSET PROFILE untuk {symbol}:")
        print(f"   Klasifikasi: {symbol_base} ({asset_class})")
        print(f"   ATR Threshold: {profile['atr_threshold']:.3f}%")
        print(f"   Volume Multiplier: {profile['volume_multiplier']:.2f}x")
        print(f"   Level Multiplier: {profile['level_multiplier']:.2f}")
        print(f"   Risk Percentage: {profile['risk_pct']:.3f}%")
        
        return profile

    def classify_volatility(self, avg_atr_pct):
        """Klasifikasi volatilitas berdasarkan data historis"""
        if avg_atr_pct < 0.15:
            return 'ULTRA_LOW'
        elif avg_atr_pct < 0.25:
            return 'LOW'
        elif avg_atr_pct < 0.40:
            return 'MODERATE'
        elif avg_atr_pct < 0.60:
            return 'HIGH'
        elif avg_atr_pct < 0.85:
            return 'VERY_HIGH'
        else:
            return 'EXTREME'

    def classify_volume(self, volume_consistency, avg_volume):
        """Klasifikasi volume berdasarkan konsistensi dan nilai absolut"""
        if volume_consistency < 0.7:
            return 'VOLATILE'
        elif volume_consistency > 1.3:
            return 'SPIKY'
        else:
            if avg_volume < 5000:
                return 'LOW_LIQUIDITY'
            elif avg_volume < 20000:
                return 'MEDIUM_LIQUIDITY'
            else:
                return 'HIGH_LIQUIDITY'

    def classify_range(self, avg_range_pct):
        """Klasifikasi karakteristik range harga"""
        if avg_range_pct < 0.15:
            return 'ULTRA_LOW'
        elif avg_range_pct < 0.25:
            return 'LOW'
        elif avg_range_pct < 0.40:
            return 'MODERATE'
        elif avg_range_pct < 0.60:
            return 'HIGH'
        elif avg_range_pct < 0.85:
            return 'VERY_HIGH'
        else:
            return 'EXTREME'

    def get_dynamic_entry_thresholds(self, asset_profile, market_regime):
        """Return optimal thresholds based on asset profile AND market regime"""
        # Dapatkan base thresholds dari asset profile
        if asset_profile['source'] == 'DEFAULT_FALLBACK':
            base_thresholds = {
                'atr_threshold': asset_profile['atr_threshold'],
                'volume_multiplier': asset_profile['volume_multiplier'],
                'level_multiplier': asset_profile['level_multiplier'],
                'adx_threshold': 18,
                'risk_pct': asset_profile['risk_pct']
            }
        else:
            # Dapatkan thresholds dari klasifikasi dinamis
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
        
        # Market regime adjustment
        regime_adjustment = {
            'STRONG_BULL': {'atr_threshold': 0.85, 'volume_multiplier': 0.8, 'adx_threshold': 0.85},
            'BULL': {'atr_threshold': 0.9, 'volume_multiplier': 0.85, 'adx_threshold': 0.9},
            'NEUTRAL': {'atr_threshold': 1.0, 'volume_multiplier': 1.0, 'adx_threshold': 1.0},
            'BEAR': {'atr_threshold': 1.1, 'volume_multiplier': 1.15, 'adx_threshold': 1.05},
            'STRONG_BEAR': {'atr_threshold': 1.2, 'volume_multiplier': 1.25, 'adx_threshold': 1.1}
        }.get(market_regime, {'atr_threshold': 1.0, 'volume_multiplier': 1.0, 'adx_threshold': 1.0})
        
        # Apply adjustment
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
        """Ambil daftar aset trending dari Binance secara otomatis"""
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
            
            # Filter tambahan: hindari aset dengan pergerakan ekstrem (>50% dalam 24h)
            df = df[abs(df['change_24h']) < 50.0]
            df = df[df['volume_usd'] > self.min_volume_usd]
            
            top_symbols = df.sort_values('volume_usd', ascending=False).head(self.max_symbols)
            
            print(f"✅ BERHASIL MENDAPATKAN {len(top_symbols)} ASET TRENDING:")
            for i, (_, row) in enumerate(top_symbols.iterrows()):
                print(f"   #{i+1} {row['symbol']} | Volume: ${row['volume_usd']:,.0f} | Change: {row['change_24h']:+.2f}% | Price: ${row['price']:.4f}")
            
            return top_symbols['symbol'].tolist()
            
        except Exception as e:
            print(f"❌ Error mengambil data dari Binance: {e}")
            print("🔄 Menggunakan daftar aset default...")
            return self.get_default_symbols()
    
    def get_default_symbols(self):
        """Fallback ke daftar aset default jika API error"""
        default_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'AVAX/USDT',
            'DOGE/USDT', 'XRP/USDT', 'ADA/USDT', 'MATIC/USDT', 'LINK/USDT',
            'INJ/USDT', 'TON/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ARB/USDT',
            'ICP/USDT', 'NEAR/USDT', 'OP/USDT', 'APT/USDT', 'SEI/USDT'
        ]
        return default_symbols[:self.max_symbols]
    
    def get_orderbook_sentiment(self, symbol):
        """Dapatkan sentimen dari order book imbalance"""
        self._rate_limit()
        
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=20)
            
            bid_volume = sum(bid[1] for bid in ob['bids'][:10])  # Top 10 bids
            ask_volume = sum(ask[1] for ask in ob['asks'][:10])  # Top 10 asks
            
            if bid_volume + ask_volume == 0:
                return 50.0
            
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            sentiment_score = 50 + (imbalance * 50)
            return max(0, min(100, sentiment_score))
            
        except Exception as e:
            print(f"⚠️ Error order book {symbol}: {e}")
            return 50.0

    def rank_symbols_by_activity(self, symbols):
        """Rangking aset berdasarkan aktivitas pasar terkini"""
        print("\n⚡ MENGANALISIS AKTIVITAS PASAR TERKINI...")
        ranked_symbols = []
        
        for symbol in symbols:
            try:
                print(f"📊 Menganalisis {symbol}...")
                self._rate_limit()
                
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=25)
                
                if len(ohlcv) < 15:
                    print(f"   ✗ {symbol}: Data tidak cukup ({len(ohlcv)} candle)")
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Hitung indikator aktivitas
                if len(df) >= 12:
                    df['price_change_pct'] = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1) * 100
                else:
                    df['price_change_pct'] = 0
                
                df['volume_ma'] = df['volume'].rolling(5).mean()
                df['volume_spike'] = df['volume'] > (df['volume_ma'] * 1.5)
                
                # Order Book Sentiment
                ob_sentiment = self.get_orderbook_sentiment(symbol)
                
                # Price Action Score
                price_change_score = abs(df['price_change_pct'].iloc[-1]) * 2
                price_score = min(100, max(0, price_change_score))
                
                # Volume Activity Score
                if len(df) >= 5:
                    volume_score = (df['volume'].iloc[-1] / df['volume'].mean()) * 12
                    volume_score = min(100, max(0, volume_score))
                else:
                    volume_score = 50
                
                # Trend Strength (ADX)
                if len(df) >= 14:
                    try:
                        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                        adx_score = min(100, max(0, df['adx'].iloc[-1] * 3))
                    except:
                        adx_score = 50
                else:
                    adx_score = 50
                
                # Total score
                activity_score = (
                    ob_sentiment * 0.35 +    # Order book (paling penting)
                    price_score * 0.25 +      # Price movement
                    volume_score * 0.20 +     # Volume activity
                    adx_score * 0.20         # Trend strength
                )
                
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
                print(f"      Pergerakan 1 Jam: {df['price_change_pct'].iloc[-1]:+.2f}% | Volume Spike: {spike_count}")
                
            except Exception as e:
                print(f"   ✗ Error menganalisis {symbol}: {e}")
                continue
        
        ranked_symbols.sort(key=lambda x: x['activity_score'], reverse=True)
        
        print("\n🏆 HASIL PERINGKAT AKTIVITAS PASAR:")
        for i, asset in enumerate(ranked_symbols[:8]):  # Tampilkan top 8
            print(f"  #{i+1} {asset['symbol']:<10} | Skor: {asset['activity_score']:5.1f} | OB Sent: {asset['ob_sentiment']:4.1f}")
            print(f"      Pergerakan: {asset['price_change']:+6.2f}% | Spike: {asset['spike_count']} | Volume Score: {asset['volume_score']:4.1f}")
        
        return ranked_symbols
    
    def get_best_asset_for_trading(self):
        """Dapatkan aset terbaik untuk trading saat ini"""
        trending_symbols = self.get_trending_symbols()
        ranked_assets = self.rank_symbols_by_activity(trending_symbols)
        
        if not ranked_assets:
            print("❌ Tidak ada aset yang bisa dianalisis!")
            return None
        
        # Filter: hanya ambil aset dengan skor minimal 40
        qualified_assets = [asset for asset in ranked_assets if asset['activity_score'] >= 40.0]
        
        if qualified_assets:
            best_asset = qualified_assets[0]
            print(f"\n🎯 ASET TERBAIK: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']
        else:
            best_asset = ranked_assets[0]
            print(f"\n⚠️ Tidak ada aset qualified, menggunakan: {best_asset['symbol']} (Skor: {best_asset['activity_score']:.1f})")
            return best_asset['symbol']

    def calculate_market_score(self, df, btc_df=None):
        """Hitung skor pasar (0.0 - 1.0)"""
        if len(df) < 50:
            print("❌ Data tidak cukup untuk analisis market score")
            return 0.0
        
        try:
            if 'atr' not in df.columns or df['atr'].isna().all():
                df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            
            df['avg_atr'] = df['atr'].rolling(50).mean()
            
            # 1. Skor Volatilitas
            if df['avg_atr'].iloc[-1] == 0 or pd.isna(df['avg_atr'].iloc[-1]):
                vol_score = 0.5
            else:
                current_atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
                avg_atr_pct = (df['avg_atr'].iloc[-1] / df['close'].iloc[-1]) * 100
                
                if 0.15 <= current_atr_pct <= 0.60:
                    vol_score = 1.0
                elif current_atr_pct < 0.10 or current_atr_pct > 1.0:
                    vol_score = 0.0
                else:
                    vol_score = 0.5
            
            # 2. Skor Volume
            df['vol_ma20'] = df['volume'].rolling(20).mean()
            df['vol_ma100'] = df['volume'].rolling(100).mean()
            
            if df['vol_ma100'].iloc[-1] == 0 or pd.isna(df['vol_ma100'].iloc[-1]):
                vol_ma_score = 0.5
            else:
                current_vol_ratio = df['vol_ma20'].iloc[-1] / df['vol_ma100'].iloc[-1]
                vol_ma_score = min(1.0, max(0.0, current_vol_ratio / 1.5))
        
            # 3. Skor Tren
            if len(df) >= 14:
                df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
                df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
                df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                
                adx = df['adx'].iloc[-1]
                plus_di = df['plus_di'].iloc[-1]
                minus_di = df['minus_di'].iloc[-1]
                
                trend_score = 0
                if adx > 20:
                    trend_score += 0.3
                elif adx > 15:
                    trend_score += 0.2
                
                di_diff = plus_di - minus_di
                if abs(di_diff) > 10:
                    trend_score += 0.2
                elif abs(di_diff) > 5:
                    trend_score += 0.1
                
                if (plus_di > minus_di and plus_di > 25) or (minus_di > plus_di and minus_di > 25):
                    trend_score += 0.1
            else:
                trend_score = 0.2
            
            # Hitung total score
            total_score = (vol_score * 0.4 + vol_ma_score * 0.3 + trend_score * 0.3)
            total_score = min(1.0, max(0.0, total_score))
            
            return total_score
            
        except Exception as e:
            print(f"🔥 Error calculating market score: {e}")
            return 0.4  # Default score yang memungkinkan trading

    def is_market_hot(self, score):
        """Apakah market layak ditrading?"""
        return score >= MIN_SCAN_SCORE

# --- FUNGSI AMBIL DATA DARI BINANCE ---
def fetch_ohlcv_data(symbol, timeframe, limit):
    """Ambil data OHLCV terbaru dari Binance"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if len(ohlcv) < limit * 0.7:  # Minimal 70% data tersedia
            print(f"⚠️ Data tidak lengkap: {len(ohlcv)}/{limit} candle")
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    except Exception as e:
        print(f"❌ Error mengambil data {symbol}: {e}")
        print("🔄 Menggunakan data dummy untuk testing...")
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='5min')
        base_price = 20.0
        prices = base_price + np.cumsum(np.random.normal(0, 0.15, limit))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0.03, 0.10, limit),
            'low': prices - np.random.uniform(0.03, 0.10, limit),
            'close': prices + np.random.uniform(-0.01, 0.01, limit),
            'volume': np.random.randint(2000, 25000, limit)
        })

# --- FUNGSI REGIME (ENHANCED) ---
def enhanced_trend_detection(row_idx, df):
    """Deteksi regime pasar - ENHANCED VERSION"""
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
        
        # 1. ADX Strength
        if adx > 25:
            trend_score += 3
        elif adx > 20:
            trend_score += 2
        elif adx > 15:
            trend_score += 1
        
        # 2. DI Crossover Strength
        di_diff = plus_di - minus_di
        if abs(di_diff) > 15:
            trend_score += 2
        elif abs(di_diff) > 8:
            trend_score += 1
        
        # 3. EMA Alignment
        if ema_fast > ema_slow and price > ema_fast:
            trend_score += 2
        elif ema_fast < ema_slow and price < ema_fast:
            trend_score += 2
        elif (ema_fast > ema_slow and price < ema_fast) or (ema_fast < ema_slow and price > ema_fast):
            trend_score += 1
        
        # Klasifikasi regime
        if trend_score >= 6:
            return "VERY_STRONG_TREND"
        elif trend_score >= 4:
            return "STRONG_TREND"
        elif trend_score >= 2:
            return "MODERATE_TREND"
        else:
            return "SIDEWAYS"
            
    except Exception as e:
        print(f"⚠️ Error dalam trend detection: {e}")
        return "SIDEWAYS"

# --- PROFESSIONAL POSITION SIZING ---
def calculate_professional_position_size(balance, entry_price, sl_price, risk_pct, leverage=10):
    """Position sizing dengan risk management profesional"""
    # Validasi input
    if sl_price <= 0 or entry_price <= 0 or balance <= 0:
        return 0
    
    # Hitung risk per unit
    risk_per_unit = abs(entry_price - sl_price) * leverage
    if risk_per_unit <= 0:
        return 0
    
    # Risk amount (max 2% per trade)
    MAX_RISK_PER_TRADE = 0.02
    risk_amount = balance * min(risk_pct, MAX_RISK_PER_TRADE)
    
    # Hitung position size
    position_size = risk_amount / risk_per_unit
    
    # Absolute position size caps
    MAX_POSITION_VALUE = 0.30  # 30% max dari balance
    MIN_POSITION_VALUE = 0.5   # $0.5 minimum
    
    position_value = position_size * entry_price
    
    # Apply caps
    max_position_value = balance * MAX_POSITION_VALUE
    if position_value > max_position_value:
        position_size = max_position_value / entry_price
    
    min_position_size = MIN_POSITION_VALUE / entry_price
    if position_value < MIN_POSITION_VALUE:
        position_size = min_position_size if min_position_size <= max_position_value / entry_price else 0
    
    # Round to reasonable precision
    if entry_price < 1:
        position_size = round(position_size, 0)  # Untuk coin murah
    else:
        position_size = round(position_size, 4)  # Untuk coin mahal
    
    return max(0, position_size)

# --- MAIN PROGRAM ---
def run_backtest_on_symbol(symbol):
    """Jalankan backtest pada satu simbol yang dipilih"""
    print(f"\n⚔️ MENJALANKAN BACKTEST PADA {symbol}...")
    print("=" * 70)
    
    df = fetch_ohlcv_data(symbol, TIMEFRAME, BARS_TO_FETCH)
    
    btc_df = None
    try:
        btc_df = fetch_ohlcv_data('BTC/USDT', TIMEFRAME, BARS_TO_FETCH)
    except Exception as e:
        print(f"⚠️ BTC data tidak tersedia: {e}")
    
    scanner = MarketScanner()
    market_score = scanner.calculate_market_score(df, btc_df)
    is_market_hot = scanner.is_market_hot(market_score)
    
    print(f"\n🎯 KEPUTUSAN MARKET SCANNER:")
    if is_market_hot:
        print(f"✅ PASAR '{symbol}' HOT! Skor: {market_score:.2f} - Strategi AKTIF")
    else:
        print(f"❌ PASAR '{symbol}' DINGIN! Skor: {market_score:.2f} - Strategi NON-AKTIF")
        print("💡 Saran: Cari pair lain atau tunggu kondisi pasar lebih baik")
    
    # Siapkan indikator
    print("\n🔧 Menyiapkan indikator...")
    try:
        df['ema_fast'] = talib.EMA(df['close'], 20)
        df['ema_slow'] = talib.EMA(df['close'], 50)
        df['ema_200'] = talib.EMA(df['close'], 200)
        df['rsi'] = talib.RSI(df['close'], 14)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        df['vol_ma'] = df['volume'].rolling(10).mean()
        
        print("✅ Semua indikator berhasil dihitung")
    except Exception as e:
        print(f"🔥 Error menghitung indikator: {e}")
        return
    
    # Dapatkan asset profile dan market regime
    asset_profile = scanner.get_asset_profile(symbol, df)
    market_regime = scanner.detect_market_regime(df)
    dynamic_thresholds = scanner.get_dynamic_entry_thresholds(asset_profile, market_regime)
    
    # Jalankan backtest
    balance = INITIAL_BALANCE
    position = None
    trades = []
    
    print(f"\n🚀 MEMULAI BACKTEST | Balance Awal: ${balance:.4f} | Leverage: {LEVERAGE}x")
    print(f"📊 Market Regime: {market_regime} | Asset Class: {asset_profile.get('asset_class', 'UNKNOWN')}")
    
    for i in range(50, len(df)):
        current_time = df['timestamp'].iloc[i]
        current_price = df['close'].iloc[i]
        
        # Exit position yang sedang berjalan
        if position is not None:
            if position['side'] == 'LONG':
                if current_price >= position['tp'] or current_price <= position['sl']:
                    pnl = (current_price - position['entry']) * position['qty'] * LEVERAGE
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': 'LONG',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'regime': position['regime'],
                        'symbol': symbol,
                        'market_regime': market_regime
                    })
                    print(f"✅ EXIT LONG @ {current_price:.4f} | PnL: {pnl:+.4f} | Balance: {balance:.4f}")
                    position = None
            else:  # SHORT
                if current_price <= position['tp'] or current_price >= position['sl']:
                    pnl = (position['entry'] - current_price) * position['qty'] * LEVERAGE
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': current_time,
                        'side': 'SHORT',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'regime': position['regime'],
                        'symbol': symbol,
                        'market_regime': market_regime
                    })
                    print(f"✅ EXIT SHORT @ {current_price:.4f} | PnL: {pnl:+.4f} | Balance: {balance:.4f}")
                    position = None
        
        # Cari entry baru
        if position is None:
            regime = enhanced_trend_detection(i, df)
            
            # Skip jika sideways atau insufficient data
            if regime in ["SIDEWAYS", "INSUFFICIENT_DATA"]:
                continue
            
            # Parameter entry
            atr = df['atr'].iloc[i]
            close = df['close'].iloc[i]
            volume = df['volume'].iloc[i]
            vol_ma = df['vol_ma'].iloc[i]
            ema_fast = df['ema_fast'].iloc[i]
            ema_slow = df['ema_slow'].iloc[i]
            adx = df['adx'].iloc[i]
            
            # Hitung ratio
            vol_ratio = volume / vol_ma if vol_ma > 0 else 0
            atr_pct = (atr / close) * 100 if close > 0 else 0
            
            # Dynamic thresholds
            atr_threshold = dynamic_thresholds['atr_threshold']
            volume_multiplier = dynamic_thresholds['volume_multiplier']
            adx_threshold = dynamic_thresholds['adx_threshold']
            level_multiplier = dynamic_thresholds['level_multiplier']
            risk_pct = dynamic_thresholds['risk_pct']
            
            # Volume dan ATR confirmation
            vol_confirmed = vol_ratio >= volume_multiplier
            atr_confirmed = atr_pct >= atr_threshold
            
            # Level breakout
            prev_close = df['close'].iloc[i-1]
            long_level = prev_close + atr * level_multiplier
            short_level = prev_close - atr * level_multiplier
            
            # Retest detection dengan toleransi
            broke_long_prev = df['high'].iloc[i-1] >= long_level
            broke_short_prev = df['low'].iloc[i-1] <= short_level
            
            retest_long = (broke_long_prev and 
                          df['low'].iloc[i] <= long_level * 1.003 and 
                          close > long_level * 0.997)
            
            retest_short = (broke_short_prev and 
                           df['high'].iloc[i] >= short_level * 0.997 and 
                           close < short_level * 1.003)
            
            # Market regime filter untuk entry direction
            allow_long = True
            allow_short = True
            
            if market_regime in ['STRONG_BULL', 'BULL']:
                allow_short = False  # Di bull market, hindari short
            elif market_regime in ['STRONG_BEAR', 'BEAR']:
                allow_long = False   # Di bear market, hindari long
            
            rsi = df['rsi'].iloc[i]
            rsi_long_ok = rsi < 70  # RSI < 70 untuk long
            rsi_short_ok = rsi > 30 # RSI > 30 untuk short
            
            # Entry long
            if (retest_long and vol_confirmed and atr_confirmed and 
                ema_fast > ema_slow and adx > adx_threshold and allow_long ):
                sl = long_level - atr * SL_ATR_MULT
                tp = long_level + atr * TP_ATR_MULT
                
                if sl <= 0 or tp <= long_level or sl >= long_level:
                    continue
                
                qty = calculate_professional_position_size(balance, long_level, sl, risk_pct, LEVERAGE)
                
                if qty > 0:
                    position = {
                        'side': 'LONG',
                        'entry': long_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': current_time,
                        'regime': regime
                    }
                    print(f"🟢 LONG entry @ {long_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f} | Regime: {regime}")
                    print(f"   📊 ATR%: {atr_pct:.3f}% (Threshold: {atr_threshold:.3f}%) | Volume Ratio: {vol_ratio:.2f}x (Threshold: {volume_multiplier:.2f}x)")
            
            # Entry short
            elif (retest_short and vol_confirmed and atr_confirmed and 
                  ema_fast < ema_slow and adx > adx_threshold and allow_short):
                sl = short_level + atr * SL_ATR_MULT
                tp = short_level - atr * TP_ATR_MULT
                
                if sl <= short_level or tp >= short_level or sl <= 0:
                    continue
                
                qty = calculate_professional_position_size(balance, short_level, sl, risk_pct, LEVERAGE)
                
                if qty > 0:
                    position = {
                        'side': 'SHORT',
                        'entry': short_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': current_time,
                        'regime': regime
                    }
                    print(f"🔴 SHORT entry @ {short_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f} | Regime: {regime}")
                    print(f"   📊 ATR%: {atr_pct:.3f}% (Threshold: {atr_threshold:.3f}%) | Volume Ratio: {vol_ratio:.2f}x (Threshold: {volume_multiplier:.2f}x)")
    
    # Tampilkan hasil
    print("\n" + "=" * 70)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    
    if not trades_df.empty:
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
        
        print(f"📈 TOTAL TRADES: {len(trades_df)}")
        print(f"✅ WIN RATE: {win_rate:.2%}")
        print(f"💰 TOTAL PnL: {total_pnl:.4f} USDT ({(total_pnl/INITIAL_BALANCE)*100:.2f}%)")
        print(f"🎯 FINAL BALANCE: {balance:.4f} USDT")
        print(f"📊 PROFIT FACTOR: {profit_factor:.2f}")
        
        print("\n--- ANALISIS PER REGIME ---")
        print(trades_df.groupby('regime').agg(
            total_trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ))
        
        filename = f"backtest_results_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n💾 Hasil disimpan ke: {filename}")
        
        return trades_df
    else:
        print("❌ TIDAK ADA TRADE DIEKSEKUSI")
        print("💡 Rekomendasi: Sistem sudah menggunakan dynamic thresholds yang adaptif")
        print("   - Coba aset lain atau waktu lain")
        print("   - Periksa apakah market benar-benar aktif")
        return None

def run_automatic_backtest():
    """Backtest otomatis dengan pemilihan aset dinamis"""
    print("🚀 SISTEM TRADING OTOMATIS - PRODUCTION MODE")
    print("=" * 70)
    
    auto_selector = MarketScanner()
    best_symbol = auto_selector.get_best_asset_for_trading()
    
    if best_symbol is None:
        print("🚫 GAGAL MENDAPATKAN ASET - MENGGUNAKAN ETH/USDT SEBAGAI DEFAULT")
        best_symbol = 'ETH/USDT'
    
    return run_backtest_on_symbol(best_symbol)

if __name__ == "__main__":
    start_time = time.time()
    results = run_automatic_backtest()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏰ Backtest selesai dalam {elapsed_time:.2f} detik")
    print("=" * 70)
    
    if results is not None and not results.empty:
        print("\n🎯 ANALISIS HASIL LENGKAP:")
        print(f"  • Rata-rata PnL per trade: ${results['pnl'].mean():.4f}")
        print(f"  • Max drawdown: ${results['pnl'].cumsum().min():.4f}")
        print(f"  • Sharpe Ratio (estimasi): {(results['pnl'].mean() / results['pnl'].std() * (len(results) ** 0.5)):.2f}" if len(results) > 1 else "  • Sharpe Ratio: Tidak dapat dihitung (hanya 1 trade)")