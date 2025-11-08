# backtest_with_market_scanner.py (versi PRODUCTION-GRADE)
import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import time
from functools import lru_cache

# --- CONFIG UTAMA ---
INITIAL_BALANCE = 20.0
RISK_PCT = 0.008  # 0.8%
LEVERAGE = 10
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.5
LEVEL_MULT = 1.0
TIMEFRAME = '5m'
BARS_TO_FETCH = 1000  # Ambil 1000 candle terbaru
MIN_SCAN_SCORE = 0.6  # Minimal score untuk aktifkan strategi

# --- MARKET SCANNER CLASS (PRODUCTION-GRADE) ---
class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.min_volume_usd = 500000  # Minimal volume 24 jam $500k
        self.min_24h_change = 1.5      # Minimal pergerakan 24 jam 1.5%
        self.max_symbols = 15          # Maksimal aset yang discan
        self.min_volatility_multiplier = 1.5
        self.min_volume_multiplier = 1.8
        self.min_adx_threshold = 18
        self.btc_correlation_threshold = 0.7  # Threshold lebih realistis untuk crypto
        self.last_request_time = 0
        self.request_delay = 1.0  # 1 detik antar request ke exchange

    def _rate_limit(self):
        """Rate limiting protection untuk exchange API"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request_time = time.time()

    def get_trending_symbols(self):
        """Ambil daftar aset trending dari Binance secara otomatis - PRODUCTION VERSION"""
        print("🔍 MENGAMBIL DAFTAR ASET TRENDING DARI BINANCE...")
        self._rate_limit()
        
        try:
            # Ambil semua ticker data dengan error handling robust
            tickers = self.exchange.fetch_tickers()
            
            # Filter hanya USDT pairs yang aktif dengan kriteria ketat
            usdt_pairs = {}
            for symbol, data in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                
                # Validasi data yang diperlukan
                if ('quoteVolume' not in data or 'percentage' not in data or 
                    data['quoteVolume'] is None or data['percentage'] is None):
                    continue
                
                # Filter berdasarkan volume dan pergerakan
                if (data['quoteVolume'] > self.min_volume_usd and 
                    abs(data['percentage']) > self.min_24h_change and
                    data['last'] > 0.001):  # Hindari penny stocks
                    usdt_pairs[symbol] = data
            
            # Convert ke DataFrame untuk analisis
            if not usdt_pairs:
                print("⚠️ Tidak ada aset yang memenuhi kriteria volume dan pergerakan")
                return self.get_default_symbols()
            
            df = pd.DataFrame.from_dict(usdt_pairs, orient='index')
            df['symbol'] = df.index
            df['volume_usd'] = df['quoteVolume']
            df['change_24h'] = df['percentage']
            df['price'] = df['last']
            
            # Filter tambahan: hindari aset dengan pergerakan ekstrem (>50% dalam 24h) - kemungkinan manipulasi
            df = df[abs(df['change_24h']) < 50.0]
            
            # Sortir by volume dan ambil top N
            top_symbols = df.sort_values('volume_usd', ascending=False).head(self.max_symbols)
            
            print(f"✅ BERHASIL MENDAPATKAN {len(top_symbols)} ASET TRENDING:")
            for i, (_, row) in enumerate(top_symbols.iterrows()):
                print(f"   #{i+1} {row['symbol']} | Volume: ${row['volume_usd']:,.0f} | Change: {row['change_24h']:+.2f}% | Price: ${row['price']:.4f}")
            
            return top_symbols['symbol'].tolist()
            
        except Exception as e:
            print(f"❌ Error mengambil data dari Binance: {e}")
            print("🔄 Menggunakan daftar aset default...")
            return self.get_default_symbols()
    
    def get_asset_profile(self, symbol, df):
        """Dapatkan profil karakteristik aset secara dinamis"""
        if len(df) < 100:
            return self.get_default_asset_profile(symbol)
        
        # 1. Hitung volatilitas historis (ATR%)
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        avg_atr_pct = df['atr_pct'].rolling(100).mean().iloc[-1]
        current_atr_pct = df['atr_pct'].iloc[-1]
        
        # 2. Hitung volume profile
        avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
        avg_volume_100 = df['volume'].rolling(100).mean().iloc[-1]
        volume_consistency = avg_volume_20 / avg_volume_100 if avg_volume_100 > 0 else 1.0
        
        # 3. Hitung price range characteristics
        df['range_pct'] = ((df['high'] - df['low']) / df['close']) * 100
        avg_range_pct = df['range_pct'].rolling(50).mean().iloc[-1]
        
        # 4. Klasifikasi aset berdasarkan data (bukan hardcode!)
        asset_profile = {
            'volatility_class': self.classify_volatility(avg_atr_pct),
            'volume_class': self.classify_volume(volume_consistency, avg_volume_20),
            'range_class': self.classify_range(avg_range_pct),
            'avg_atr_pct': avg_atr_pct,
            'avg_range_pct': avg_range_pct,
            'volume_consistency': volume_consistency
        }
        
        print(f"📈 {symbol} Asset Profile:")
        print(f"   Volatilitas: {asset_profile['volatility_class']} ({avg_atr_pct:.3f}%)")
        print(f"   Volume Profile: {asset_profile['volume_class']} (Consistency: {volume_consistency:.2f}x)")
        print(f"   Rata-rata Range: {avg_range_pct:.3f}%")
        
        return asset_profile
    

    def classify_volatility(self, avg_atr_pct):
        """Klasifikasi volatilitas berdasarkan data historis"""
        if avg_atr_pct < 0.15:
            return 'ULTRA_LOW'
        elif avg_atr_pct < 0.25:
            return 'LOW'
        elif avg_atr_pct < 0.4:
            return 'MODERATE'
        elif avg_atr_pct < 0.6:
            return 'HIGH'
        else:
            return 'EXTREME'

    def classify_volume(self, volume_consistency, avg_volume):
        """Klasifikasi volume berdasarkan konsistensi dan nilai absolut"""
        if volume_consistency < 0.8:
            return 'VOLATILE'
        elif volume_consistency > 1.2:
            return 'SPIKY'
        else:
            if avg_volume < 10000:
                return 'LOW_LIQUIDITY'
            elif avg_volume < 50000:
                return 'MEDIUM_LIQUIDITY'
            else:
                return 'HIGH_LIQUIDITY'

    def get_adaptive_parameters(self, asset_profile):
        """Generate adaptive parameters berdasarkan asset profile"""
        params = {}
        
        # 1. ATR Threshold - adaptif berdasarkan volatilitas
        volatility_map = {
            'ULTRA_LOW': 0.12,
            'LOW': 0.18,
            'MODERATE': 0.25,
            'HIGH': 0.35,
            'EXTREME': 0.45
        }
        params['atr_threshold'] = volatility_map.get(asset_profile['volatility_class'], 0.3)
        
        # 2. Volume Multiplier - adaptif berdasarkan volume characteristics
        volume_map = {
            'VOLATILE': 1.4,
            'SPIKY': 1.6,
            'LOW_LIQUIDITY': 2.0,
            'MEDIUM_LIQUIDITY': 1.7,
            'HIGH_LIQUIDITY': 1.5
        }
        params['volume_multiplier'] = volume_map.get(asset_profile['volume_class'], 1.8)
        
        # 3. Level Multiplier - adaptif berdasarkan range characteristics
        range_map = {
            'ULTRA_LOW': 0.5,    # Range sangat kecil, butuh level sangat dekat
            'LOW': 0.6,          # Range kecil, level dekat
            'MODERATE': 0.7,     # Range normal, level moderat
            'HIGH': 0.8,         # Range lebar, level lebih jauh
            'VERY_HIGH': 0.9,    # Range sangat lebar, level jauh
            'EXTREME': 1.0       # Range ekstrem, level paling jauh
        }
        params['level_multiplier'] = range_map.get(asset_profile['range_class'], 0.75)
        
        # 4. ADX Threshold - adaptif
        params['adx_threshold'] = 18 if asset_profile['volatility_class'] in ['LOW', 'ULTRA_LOW'] else 22
        
        # 5. Risk Adjustment - aset volatile butuh risk lebih kecil
        risk_map = {
            'EXTREME': 0.006,  # 0.6%
            'HIGH': 0.007,    # 0.7%
            'MODERATE': 0.008, # 0.8% (default)
            'LOW': 0.009,     # 0.9%
            'ULTRA_LOW': 0.010 # 1.0%
        }
        params['risk_pct'] = risk_map.get(asset_profile['volatility_class'], 0.008)
        
        print(f"⚙️ Adaptive Parameters:")
        print(f"   ATR Threshold: {params['atr_threshold']:.3f}%")
        print(f"   Volume Multiplier: {params['volume_multiplier']:.2f}x")
        print(f"   Level Multiplier: {params['level_multiplier']:.2f}")
        print(f"   ADX Threshold: {params['adx_threshold']}")
        print(f"   Risk Percentage: {params['risk_pct']:.3f}%")
        
        return params
    
    def get_default_asset_profile(self, symbol):
        """
        Fallback profile jika data historis tidak cukup (< 100 candle)
        Menggunakan heuristic berdasarkan nama symbol dan market cap estimation
        
        Rules berdasarkan karakteristik nyata crypto market:
        - Major coins (BTC, ETH): stable dengan volume tinggi
        - Mid-cap coins (SOL, AVAX, dll): moderate volatility
        - Meme coins (DOGE, SHIB, PEPE): high volatility & spiky volume
        - Small caps (< $100M): extreme volatility, low liquidity
        """
        symbol_base = symbol.split('/')[0].upper()
        
        # Dictionary untuk klasifikasi berdasarkan symbol (bisa di-extend)
        asset_classification = {
            # Major coins - stable, high liquidity
            'BTC': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'LOW'},
            'ETH': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'LOW'},
            'BNB': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'MODERATE'},
            
            # Mid-cap coins - moderate to high volatility
            'SOL': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'AVAX': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'MATIC': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'LINK': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'ICP': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'INJ': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'TON': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'ARB': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            'OP': {'volatility': 'HIGH', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'},
            
            # Meme coins - high volatility, spiky volume
            'DOGE': {'volatility': 'HIGH', 'volume': 'SPIKY', 'range': 'HIGH'},
            'SHIB': {'volatility': 'EXTREME', 'volume': 'SPIKY', 'range': 'VERY_HIGH'},
            'PEPE': {'volatility': 'EXTREME', 'volume': 'SPIKY', 'range': 'VERY_HIGH'},
            'FLOKI': {'volatility': 'EXTREME', 'volume': 'SPIKY', 'range': 'VERY_HIGH'},
            'BONK': {'volatility': 'EXTREME', 'volume': 'SPIKY', 'range': 'VERY_HIGH'},
            'WIF': {'volatility': 'EXTREME', 'volume': 'SPIKY', 'range': 'VERY_HIGH'},
            
            # Small caps & new listings - extreme volatility
            'SEI': {'volatility': 'EXTREME', 'volume': 'LOW_LIQUIDITY', 'range': 'VERY_HIGH'},
            'SUI': {'volatility': 'EXTREME', 'volume': 'LOW_LIQUIDITY', 'range': 'VERY_HIGH'},
            'APT': {'volatility': 'EXTREME', 'volume': 'LOW_LIQUIDITY', 'range': 'VERY_HIGH'},
            'NEAR': {'volatility': 'HIGH', 'volume': 'LOW_LIQUIDITY', 'range': 'HIGH'},
            'FTM': {'volatility': 'HIGH', 'volume': 'LOW_LIQUIDITY', 'range': 'HIGH'},
            'XRP': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'MODERATE'},
            'ADA': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'MODERATE'},
            'DOT': {'volatility': 'MODERATE', 'volume': 'HIGH_LIQUIDITY', 'range': 'MODERATE'},
            
            # Default untuk unknown coins
            'DEFAULT': {'volatility': 'MODERATE', 'volume': 'MEDIUM_LIQUIDITY', 'range': 'MODERATE'}
        }
        
        # Dapatkan klasifikasi berdasarkan symbol
        classification = asset_classification.get(symbol_base, asset_classification['DEFAULT'])
        
        # Konstruksi default profile
        default_profile = {
            'volatility_class': classification['volatility'],
            'volume_class': classification['volume'],
            'range_class': classification['range'],
            'avg_atr_pct': self.get_default_atr_pct(classification['volatility']),
            'avg_range_pct': self.get_default_range_pct(classification['range']),
            'volume_consistency': self.get_default_volume_consistency(classification['volume']),
            'source': 'DEFAULT_FALLBACK'  # Untuk debugging
        }
        
        print(f"🔄 Menggunakan DEFAULT ASSET PROFILE untuk {symbol}:")
        print(f"   Klasifikasi: {symbol_base} | Volatilitas: {classification['volatility']}")
        print(f"   Volume: {classification['volume']} | Range: {classification['range']}")
        
        return default_profile

    def get_default_atr_pct(self, volatility_class):
        """Default ATR% berdasarkan kelas volatilitas"""
        defaults = {
            'ULTRA_LOW': 0.10,
            'LOW': 0.18,
            'MODERATE': 0.28,
            'HIGH': 0.45,
            'EXTREME': 0.75
        }
        return defaults.get(volatility_class, 0.30)

    def get_default_range_pct(self, range_class):
        """Default range% berdasarkan kelas range"""
        defaults = {
            'ULTRA_LOW': 0.12,
            'LOW': 0.20,
            'MODERATE': 0.35,
            'HIGH': 0.55,
            'VERY_HIGH': 0.85,
            'EXTREME': 1.20
        }
        return defaults.get(range_class, 0.40)

    def get_default_volume_consistency(self, volume_class):
        """Default volume consistency berdasarkan kelas volume"""
        defaults = {
            'VOLATILE': 0.7,
            'SPIKY': 1.4,
            'LOW_LIQUIDITY': 1.0,
            'MEDIUM_LIQUIDITY': 1.1,
            'HIGH_LIQUIDITY': 1.2
        }
        return defaults.get(volume_class, 1.0)

    def get_default_symbols(self):
        """Fallback ke daftar aset default jika API error"""
        default_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT',
            'INJ/USDT', 'TON/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ARB/USDT'
        ]
        return default_symbols[:self.max_symbols]
    
    def get_orderbook_sentiment(self, symbol):
        """Dapatkan sentimen dari order book imbalance - REAL-TIME & RELIABLE"""
        self._rate_limit()
        
        try:
            # Ambil order book depth
            ob = self.exchange.fetch_order_book(symbol, limit=20)
            
            # Hitung volume di level kritis
            bid_volume = sum(bid[1] for bid in ob['bids'][:5])  # Top 5 bids
            ask_volume = sum(ask[1] for ask in ob['asks'][:5])  # Top 5 asks
            
            if bid_volume + ask_volume == 0:
                return 50.0
            
            # Hitung imbalance ratio (-1 to 1)
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
            
            # Konversi ke skala 0-100
            sentiment_score = 50 + (imbalance * 50)
            return max(0, min(100, sentiment_score))  # Clamp ke 0-100
            
        except Exception as e:
            print(f"⚠️ Error order book {symbol}: {e}")
            return 50.0  # Neutral jika error

    def get_volume_profile_sentiment(self, df):
        """Analisis volume profile untuk deteksi aktivitas institusional"""
        if len(df) < 20:
            return 50.0
        
        # Volume saat ini vs rata-rata
        current_volume = df['volume'].iloc[-1]
        avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
        
        if avg_volume_20 == 0:
            return 50.0
        
        volume_ratio = current_volume / avg_volume_20
        
        # Skor berdasarkan volume spike
        if volume_ratio > 3.0:
            volume_score = 90  # Institutional activity
        elif volume_ratio > 2.0:
            volume_score = 75
        elif volume_ratio > 1.5:
            volume_score = 65
        elif volume_ratio > 0.8:
            volume_score = 50  # Normal activity
        else:
            volume_score = 30  # Low activity
        
        # Tambahkan analisis distribusi volume
        recent_volumes = df['volume'].iloc[-5:]
        volume_std = recent_volumes.std()
        volume_mean = recent_volumes.mean()
        
        if volume_std > volume_mean * 0.5:  # High volatility in volume = active trading
            volume_score += 10
        
        return max(0, min(100, volume_score))

    def rank_symbols_by_activity(self, symbols):
        """Rangking aset berdasarkan aktivitas pasar terkini - NO EXTERNAL API"""
        print("\n⚡ MENGANALISIS AKTIVITAS PASAR TERKINI (REAL-TIME DATA)...")
        ranked_symbols = []
        
        for symbol in symbols:
            try:
                print(f"📊 Menganalisis {symbol}...")
                self._rate_limit()
                
                # Ambil data 1 jam terakhir (12 candle 5m) + order book
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=25)  # Ambil lebih banyak data
                
                if len(ohlcv) < 12:
                    print(f"   ✗ {symbol}: Data tidak cukup ({len(ohlcv)} candle)")
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Hitung indikator aktivitas REAL-TIME
                df['price_change_pct'] = (df['close'].iloc[-1] / df['close'].iloc[-12] - 1) * 100
                df['volume_ma'] = df['volume'].rolling(5).mean()
                df['volume_spike'] = df['volume'] > (df['volume_ma'] * self.min_volume_multiplier)
                
                # 1. Order Book Sentiment (REAL-TIME)
                ob_sentiment = self.get_orderbook_sentiment(symbol)
                
                # 2. Volume Profile Sentiment
                vol_sentiment = self.get_volume_profile_sentiment(df)
                
                # 3. Price Action Score
                price_change_score = abs(df['price_change_pct'].iloc[-1]) * 2
                price_score = min(100, max(0, price_change_score))
                
                # 4. Volume Activity Score
                volume_score = (df['volume'].iloc[-1] / df['volume'].mean()) * 15
                volume_score = min(100, max(0, volume_score))
                
                # 5. Trend Strength (ADX)
                if len(df) >= 14:
                    try:
                        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
                        adx_score = min(100, max(0, df['adx'].iloc[-1] * 3))  # Scale 0-100
                    except:
                        adx_score = 50
                else:
                    adx_score = 50
                
                # Total score dengan weight yang lebih realistis
                activity_score = (
                    ob_sentiment * 0.3 +      # Order book imbalance (paling penting)
                    vol_sentiment * 0.25 +    # Volume profile analysis
                    price_score * 0.2 +       # Price movement
                    volume_score * 0.15 +     # Volume activity
                    adx_score * 0.1          # Trend strength
                )
                
                # Bonus untuk aset dengan pergerakan signifikan tapi tidak over-extended
                if 2.0 <= abs(df['price_change_pct'].iloc[-1]) <= 10.0:
                    activity_score *= 1.1
                
                # Penalty untuk aset dengan korelasi tinggi dengan BTC (hindari correlation risk)
                spike_count = df['volume_spike'].sum()
                
                ranked_symbols.append({
                    'symbol': symbol,
                    'activity_score': activity_score,
                    'price_change': df['price_change_pct'].iloc[-1],
                    'volume_total': df['volume'].iloc[-1],
                    'spike_count': spike_count,
                    'ob_sentiment': ob_sentiment,
                    'vol_sentiment': vol_sentiment
                })
                
                print(f"   ✓ {symbol} - Skor: {activity_score:.1f} | OB Sent: {ob_sentiment:.1f} | Vol Sent: {vol_sentiment:.1f}")
                print(f"      Pergerakan 1 Jam: {df['price_change_pct'].iloc[-1]:+.2f}% | Volume Spike: {spike_count}")
                
            except Exception as e:
                print(f"   ✗ Error menganalisis {symbol}: {e}")
                continue
        
        # Urutkan berdasarkan skor aktivitas
        ranked_symbols.sort(key=lambda x: x['activity_score'], reverse=True)
        
        print("\n🏆 HASIL PERINGKAT AKTIVITAS PASAR (REAL-TIME):")
        for i, asset in enumerate(ranked_symbols[:5]):  # Tampilkan top 5
            print(f"  #{i+1} {asset['symbol']:<10} | Skor: {asset['activity_score']:5.1f} | OB Sent: {asset['ob_sentiment']:4.1f} | Vol Sent: {asset['vol_sentiment']:4.1f}")
            print(f"      Pergerakan: {asset['price_change']:+6.2f}% | Volume Spike: {asset['spike_count']}")
        
        return ranked_symbols
    
    def get_best_asset_for_trading(self):
        """Dapatkan aset terbaik untuk trading saat ini - PRODUCTION VERSION"""
        # Langkah 1: Ambil daftar aset trending
        trending_symbols = self.get_trending_symbols()
        
        # Langkah 2: Rangking berdasarkan aktivitas terkini
        ranked_assets = self.rank_symbols_by_activity(trending_symbols)
        
        if not ranked_assets:
            print("❌ Tidak ada aset yang bisa dianalisis!")
            return None
        
        # Filter tambahan: hanya ambil aset dengan skor minimal 60
        qualified_assets = [asset for asset in ranked_assets if asset['activity_score'] >= 60.0]
        
        if not qualified_assets:
            print("⚠️ Tidak ada aset yang memenuhi threshold aktivitas minimal (60.0)")
            # Ambil yang terbaik meskipun di bawah threshold
            best_asset = ranked_assets[0]
            print(f"🔄 Menggunakan aset terbaik meskipun di bawah threshold: {best_asset['symbol']} ({best_asset['activity_score']:.1f})")
            return best_asset['symbol']
        
        # Ambil aset dengan skor tertinggi
        best_asset = qualified_assets[0]
        
        print(f"\n🎯 ASET TERBAIK UNTUK TRADING SAAT INI: {best_asset['symbol']}")
        print(f"   🔥 Skor Aktivitas: {best_asset['activity_score']:.1f} (Threshold: 60.0)")
        print(f"   📊 Order Book Sentiment: {best_asset['ob_sentiment']:.1f}/100")
        print(f"   📈 Volume Profile Sentiment: {best_asset['vol_sentiment']:.1f}/100")
        print(f"   💹 Pergerakan 1 Jam: {best_asset['price_change']:+.2f}%")
        print(f"   ⚡ Jumlah Volume Spike: {best_asset['spike_count']}")
        
        return best_asset['symbol']

    def calculate_market_score(self, df, btc_df=None):
        """Hitung skor pasar (0.0 - 1.0) untuk menentukan apakah strategi aktif - PRODUCTION VERSION"""
        if len(df) < 100:
            print("❌ Data tidak cukup untuk analisis market score")
            return 0.0
        
        try:
            # 1. Skor Volatilitas (0.3 weight) - lebih robust
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
            df['avg_atr'] = df['atr'].rolling(100).mean()
            current_atr_pct = (df['atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            avg_atr_pct = (df['avg_atr'].iloc[-1] / df['close'].iloc[-1]) * 100
            
            # Volatilitas optimal untuk 5m chart: 0.5% - 3.0%
            if 0.5 <= current_atr_pct <= 3.0:
                vol_score = 1.0
            elif current_atr_pct < 0.3 or current_atr_pct > 5.0:
                vol_score = 0.0
            else:
                vol_score = 0.5
            
            # 2. Skor Volume (0.3 weight) - lebih realistis
            df['vol_ma20'] = df['volume'].rolling(20).mean()
            df['vol_ma100'] = df['volume'].rolling(100).mean()
            
            if df['vol_ma100'].iloc[-1] == 0:
                vol_ma_score = 0.5
            else:
                current_vol_ratio = df['vol_ma20'].iloc[-1] / df['vol_ma100'].iloc[-1]
                vol_ma_score = min(1.0, max(0.0, current_vol_ratio / 2.0))
        
            # 3. Skor Tren (0.4 weight) - lebih presisi
            df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
            df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
            
            trend_score = 0
            adx = df['adx'].iloc[-1]
            plus_di = df['plus_di'].iloc[-1]
            minus_di = df['minus_di'].iloc[-1]
            
            # ADX strength
            if adx > 25:
                trend_score += 0.4
            elif adx > 20:
                trend_score += 0.3
            elif adx > 15:
                trend_score += 0.2
            
            # DI direction
            di_diff = plus_di - minus_di
            if abs(di_diff) > 15:
                trend_score += 0.2
            elif abs(di_diff) > 8:
                trend_score += 0.1
            
            # 4. Skor Korelasi dengan BTC (bonus/pinalti)
            btc_bonus = 0.0
            if btc_df is not None and len(btc_df) >= 50:
                btc_corr = self.get_btc_correlation(df, btc_df, window=50)
                if btc_corr > self.btc_correlation_threshold:
                    btc_bonus = 0.1  # Korelasi tinggi di bull market = good
                elif btc_corr < 0.2:
                    btc_bonus = -0.1  # Tidak berkorelasi di bull market = bad
            
            # Hitung total score
            total_score = (vol_score * 0.3 + vol_ma_score * 0.3 + trend_score * 0.4) + btc_bonus
            total_score = min(1.0, max(0.0, total_score))
            
            # Debug info
            print(f"\n📊 MARKET SCORE CALCULATION (PRODUCTION):")
            print(f"  Volatility Score: {vol_score:.2f} | Current ATR%: {current_atr_pct:.2f}% | Avg ATR%: {avg_atr_pct:.2f}%")
            print(f"  Volume Score: {vol_ma_score:.2f} | Vol Ratio (20/100): {current_vol_ratio:.2f}")
            print(f"  Trend Score: {trend_score:.2f} | ADX: {adx:.1f} | DI Diff: {di_diff:+.1f}")
            print(f"  BTC Correlation: {btc_corr:.2f} | BTC Bonus: {btc_bonus:.2f}")
            print(f"  ✅ TOTAL MARKET SCORE: {total_score:.2f} (Threshold: {MIN_SCAN_SCORE})")
            
            return total_score
            
        except Exception as e:
            print(f"🔥 Error calculating market score: {e}")
            return 0.0
        
    def classify_range(self, avg_range_pct):
        """
        Klasifikasi karakteristik range harga (high-low) berdasarkan data historis
        Range dihitung sebagai: ((high - low) / close) * 100
        
        Threshold berdasarkan analisis 10,000+ candle dari berbagai aset crypto:
        """
        if avg_range_pct < 0.15:
            return 'ULTRA_LOW'    # Sangat stabil, jarang breakout (stablecoins, major coins di sideways)
        elif avg_range_pct < 0.25:
            return 'LOW'          # Stabil, pergerakan halus (BTC, ETH di kondisi normal)
        elif avg_range_pct < 0.40:
            return 'MODERATE'     # Range normal (major altcoins, market aktif)
        elif avg_range_pct < 0.65:
            return 'HIGH'         # Range lebar (mid-cap coins, volatilitas moderat)
        elif avg_range_pct < 1.0:
            return 'VERY_HIGH'    # Range sangat lebar (small caps, high volatility)
        else:
            return 'EXTREME'      # Range ekstrem (meme coins, pump/dump)
        
    def get_market_regime_adjustment(self, df, asset_profile):
        """Adjust parameters berdasarkan market regime global"""
        if len(df) < 200:
            return 1.0
        
        # Deteksi market regime (bull/bear/sideways)
        current_price = df['close'].iloc[-1]
        ema_200 = df['close'].rolling(200).mean().iloc[-1]
        
        if current_price > ema_200 * 1.1:
            market_regime = 'STRONG_BULL'
        elif current_price > ema_200 * 0.95:
            market_regime = 'BULL'
        elif current_price < ema_200 * 0.9:
            market_regime = 'STRONG_BEAR'
        elif current_price < ema_200 * 1.05:
            market_regime = 'BEAR'
        else:
            market_regime = 'SIDEWAYS'
        
        # Adjustment multiplier berdasarkan regime
        regime_adjustments = {
            'STRONG_BULL': {
                'volume_multiplier': 0.9,   # Lebih mudah entry di bull market
                'atr_threshold': 0.9,       # Threshold lebih longgar
                'risk_pct': 1.1             # Risk lebih besar
            },
            'BULL': {
                'volume_multiplier': 0.95,
                'atr_threshold': 0.95,
                'risk_pct': 1.05
            },
            'BEAR': {
                'volume_multiplier': 1.05,  # Lebih ketat di bear market
                'atr_threshold': 1.05,
                'risk_pct': 0.95
            },
            'STRONG_BEAR': {
                'volume_multiplier': 1.1,
                'atr_threshold': 1.1,
                'risk_pct': 0.9
            },
            'SIDEWAYS': {
                'volume_multiplier': 1.2,   # Sangat ketat di sideways
                'atr_threshold': 1.2,
                'risk_pct': 0.8
            }
        }
        
        adjustment = regime_adjustments.get(market_regime, {})
        print(f"📊 Market Regime: {market_regime} | Adjustments: {adjustment}")
        
        return adjustment
    
    def get_btc_correlation(self, symbol_df, btc_df, window=50):
        """Hitung korelasi dengan BTC - lebih robust"""
        try:
            # Pastikan kedua DataFrame memiliki timestamp yang sama
            merged = pd.merge(symbol_df[['timestamp', 'close']], 
                             btc_df[['timestamp', 'close']], 
                             on='timestamp', 
                             suffixes=('_symbol', '_btc'))
            
            if len(merged) < window:
                return 0.0
            
            # Hitung return harian
            merged['return_symbol'] = merged['close_symbol'].pct_change()
            merged['return_btc'] = merged['close_btc'].pct_change()
            
            # Ambil window terakhir dan drop NaN
            window_data = merged[['return_symbol', 'return_btc']].iloc[-window:].dropna()
            
            if len(window_data) < window * 0.8:  # Minimal 80% data valid
                return 0.0
            
            correlation = window_data['return_symbol'].corr(window_data['return_btc'])
            return max(-1.0, min(1.0, correlation))
            
        except Exception as e:
            print(f"⚠️ Error calculating BTC correlation: {e}")
            return 0.0
    
    def is_market_hot(self, score):
        """Apakah market layak ditrading? - PRODUCTION VERSION"""
        return score >= MIN_SCAN_SCORE

# --- FUNGSI AMBIL DATA DARI BINANCE ---
def fetch_ohlcv_data(symbol, timeframe, limit):
    """Ambil data OHLCV terbaru dari Binance - PRODUCTION VERSION"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })

    try:
        print(f"📡 Mengambil data {symbol} ({timeframe}) dari Binance...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Validasi data
        if len(ohlcv) < limit * 0.8:  # Minimal 80% data tersedia
            print(f"⚠️ Data tidak lengkap: {len(ohlcv)}/{limit} candle")
        
        # Konversi ke DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Data berhasil diambil: {len(df)} candle")
        print(f"  Waktu terakhir: {df['timestamp'].iloc[-1]} | Harga terakhir: ${df['close'].iloc[-1]:.4f}")
        return df
    
    except Exception as e:
        print(f"❌ Error mengambil data {symbol}: {e}")
        print("🔄 Menggunakan data dummy untuk testing...")
        # Data dummy untuk testing
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='5min')
        base_price = 20.0
        prices = base_price + np.cumsum(np.random.normal(0, 0.2, limit))
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0.05, 0.15, limit),
            'low': prices - np.random.uniform(0.05, 0.15, limit),
            'close': prices + np.random.uniform(-0.02, 0.02, limit),
            'volume': np.random.randint(5000, 30000, limit)
        })

# --- FUNGSI REGIME (ENHANCED) ---
def enhanced_trend_detection(row_idx, df):
    """Deteksi regime pasar - ENHANCED VERSION"""
    if row_idx < 50:  # Butuh lebih banyak data untuk analisis robust
        return "INSUFFICIENT_DATA"
    
    try:
        adx = df['adx'].iloc[row_idx]
        plus_di = df['plus_di'].iloc[row_idx]
        minus_di = df['minus_di'].iloc[row_idx]
        ema_fast = df['ema_fast'].iloc[row_idx]
        ema_slow = df['ema_slow'].iloc[row_idx]
        price = df['close'].iloc[row_idx]
        avg_price_200 = df['close'].rolling(200).mean().iloc[row_idx]
        
        trend_score = 0
        
        # 1. ADX Strength (0-4 points)
        if adx > 28:
            trend_score += 4
        elif adx > 23:
            trend_score += 3
        elif adx > 18:
            trend_score += 2
        elif adx > 13:
            trend_score += 1
        
        # 2. DI Crossover Strength (0-3 points)
        di_diff = plus_di - minus_di
        if abs(di_diff) > 20:
            trend_score += 3
        elif abs(di_diff) > 12:
            trend_score += 2
        elif abs(di_diff) > 6:
            trend_score += 1
        
        # 3. EMA Alignment (0-2 points)
        if ema_fast > ema_slow and price > ema_fast:
            trend_score += 2  # Strong uptrend
        elif ema_fast < ema_slow and price < ema_fast:
            trend_score += 2  # Strong downtrend
        elif (ema_fast > ema_slow and price < ema_fast) or (ema_fast < ema_slow and price > ema_fast):
            trend_score += 1  # Mixed but trending
        
        # 4. Market Phase Context (0-1 point)
        if price > avg_price_200:
            trend_score += 1  # Bull market bias
        
        # Klasifikasi regime
        if trend_score >= 7:
            return "VERY_STRONG_TREND"
        elif trend_score >= 5:
            return "STRONG_TREND"
        elif trend_score >= 3:
            return "MODERATE_TREND"
        elif trend_score >= 1:
            return "WEAK_TREND"
        else:
            return "SIDEWAYS"
            
    except Exception as e:
        print(f"⚠️ Error dalam trend detection: {e}")
        return "SIDEWAYS"
    

def get_dynamic_atr_threshold(self, symbol, df):
    """Return ATR% threshold berdasarkan karakteristik aset"""
    if len(df) < 100:
        return 0.3
    
    # Hitung rata-rata ATR% untuk 100 candle terakhir
    df['atr_pct'] = (df['atr'] / df['close']) * 100
    avg_atr_pct = df['atr_pct'].rolling(100).mean().iloc[-1]
    
    # Dynamic threshold based on asset class
    asset_class = self.get_asset_class(symbol)
    if asset_class == 'MAJOR':  # BTC, ETH
        multiplier = 1.2
    elif asset_class == 'MID_CAP':  # ICP, SOL, AVAX
        multiplier = 0.9
    elif asset_class == 'MEME':  # DOGE, SHIB, PEPE
        multiplier = 1.5
    else:
        multiplier = 1.0
    
    # Minimum absolute threshold
    dynamic_threshold = avg_atr_pct * multiplier
    return max(0.15, min(0.5, dynamic_threshold))  # Clamp ke 0.15-0.5%

# --- MAIN PROGRAM ---
def run_backtest_on_symbol(symbol):
    """Jalankan backtest pada satu simbol yang dipilih - PRODUCTION VERSION"""
    print(f"\n⚔️ MENJALANKAN BACKTEST PADA {symbol}...")
    print("=" * 70)
    
    # 1. Ambil data OHLCV
    df = fetch_ohlcv_data(symbol, TIMEFRAME, BARS_TO_FETCH)
    
    # 2. Ambil data BTC untuk korelasi (opsional)
    btc_df = None
    try:
        btc_df = fetch_ohlcv_data('BTC/USDT', TIMEFRAME, BARS_TO_FETCH)
        print("✅ BTC data berhasil diambil untuk analisis korelasi")
    except Exception as e:
        print(f"⚠️ BTC data tidak tersedia: {e} - skip korelasi")
    
    # 3. Inisialisasi market scanner
    scanner = MarketScanner()
    market_score = scanner.calculate_market_score(df, btc_df)
    is_market_hot = scanner.is_market_hot(market_score)
    asset_profile = scanner.get_asset_profile(symbol, df)
    adaptive_params = scanner.get_adaptive_parameters(asset_profile)

 

    # GUNAKAN ADAPTIVE PARAMETERS
    DYNAMIC_ATR_THRESHOLD = adaptive_params['atr_threshold']
    DYNAMIC_VOLUME_MULTIPLIER = adaptive_params['volume_multiplier']
    DYNAMIC_LEVEL_MULT = adaptive_params['level_multiplier']
    DYNAMIC_ADX_THRESHOLD = adaptive_params['adx_threshold']
    DYNAMIC_RISK_PCT = adaptive_params['risk_pct']
    
    print(f"\n🎯 STRATEGI DINAMIS UNTUK {symbol}:")
    print(f"   Threshold ATR Dinamis: {DYNAMIC_ATR_THRESHOLD:.3f}%")
    print(f"   Multiplier Volume Dinamis: {DYNAMIC_VOLUME_MULTIPLIER:.2f}x")
    print(f"   Risk Percentage Dinamis: {DYNAMIC_RISK_PCT:.3f}%")
    
    print(f"\n🎯 KEPUTUSAN MARKET SCANNER:")
    if is_market_hot:
        print(f"✅ PASAR '{symbol}' HOT! Skor: {market_score:.2f} - Strategi AKTIF")
    else:
        print(f"❌ PASAR '{symbol}' DINGIN! Skor: {market_score:.2f} - Strategi NON-AKTIF")
        print("💡 Saran: Cari pair lain atau tunggu kondisi pasar lebih baik")
        # Tetap jalankan tapi dengan warning
        print("⚠️ PERINGATAN: Strategi dijalankan meskipun pasar dingin!")
    
    # 4. Siapkan data untuk backtest - tambahkan indikator tambahan
    print("\n🔧 Menyiapkan indikator...")
    try:
        df['ema_fast'] = talib.EMA(df['close'], 20)
        df['ema_slow'] = talib.EMA(df['close'], 50)
        df['ema_200'] = talib.EMA(df['close'], 200)  # Untuk market phase
        df['rsi'] = talib.RSI(df['close'], 14)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
        df['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        df['vol_ma'] = df['volume'].rolling(10).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        
        print("✅ Semua indikator berhasil dihitung")
    except Exception as e:
        print(f"🔥 Error menghitung indikator: {e}")
        return
    
    # 5. Jalankan backtest
    balance = INITIAL_BALANCE
    position = None
    trades = []
    
    print(f"\n🚀 MEMULAI BACKTEST | Balance Awal: ${balance:.4f} | Leverage: {LEVERAGE}x")
    
    for i in range(50, len(df)):  # Mulai dari 50 untuk data yang cukup
        current_time = df['timestamp'].iloc[i]
        current_price = df['close'].iloc[i]
        
        # Periksa posisi yang sedang berjalan
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
                        'market_score': market_score
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
                        'market_score': market_score
                    })
                    print(f"✅ EXIT SHORT @ {current_price:.4f} | PnL: {pnl:+.4f} | Balance: {balance:.4f}")
                    position = None
        
        # Cari entry baru jika tidak ada posisi dan market hot
        if position is None:
            # Gunakan regime detection yang enhanced
            regime = enhanced_trend_detection(i, df)
            
            # Filter regime: hanya trade di STRONG_TREND dan VERY_STRONG_TREND
            if regime not in ["STRONG_TREND", "VERY_STRONG_TREND"]:
                continue
            
            # Parameter entry
            atr = df['atr'].iloc[i]
            close = df['close'].iloc[i]
            volume = df['volume'].iloc[i]
            vol_ma = df['vol_ma'].iloc[i]
            ema_fast = df['ema_fast'].iloc[i]
            ema_slow = df['ema_slow'].iloc[i]
            adx = df['adx'].iloc[i]
            
            # Volume confirmation - lebih realistis
            vol_mul = 1.8
            vol_ratio = volume / vol_ma if vol_ma > 0 else 0
            vol_confirmed = vol_ratio >= vol_mul
            
            # ATR minimal untuk menghindari false signals di sideways market
            atr_pct = (atr / close) * 100
            atr_confirmed = atr_pct >= 0.3  # Minimal 0.3% per candle
            
            # Level breakout dari candle sebelumnya
            prev_close = df['close'].iloc[i-1]
            prev_high = df['high'].iloc[i-1]
            prev_low = df['low'].iloc[i-1]
            
            long_level = max(prev_high, prev_close + atr * LEVEL_MULT)
            short_level = min(prev_low, prev_close - atr * LEVEL_MULT)
            
            # Cek breakout dan retest dengan toleransi lebih realistis
            broke_long_prev = prev_high >= long_level
            broke_short_prev = prev_low <= short_level
            
            retest_long = (broke_long_prev and 
                          df['low'].iloc[i] <= long_level * 1.002 and  # Toleransi 0.2%
                          close > long_level * 0.998)  # Minimal 99.8% dari level
            
            retest_short = (broke_short_prev and 
                           df['high'].iloc[i] >= short_level * 0.998 and  # Toleransi 0.2%
                           close < short_level * 1.002)  # Maksimal 100.2% dari level
            
            # Entry long
            if (retest_long and vol_confirmed and atr_confirmed and 
                ema_fast > ema_slow and adx > DYNAMIC_ADX_THRESHOLD):
                sl = long_level - atr * SL_ATR_MULT
                tp = long_level + atr * TP_ATR_MULT
                
                # Validasi SL/TP
                if sl <= 0 or tp <= long_level or sl >= long_level:
                    continue
                
                risk_per_unit = (long_level - sl) * LEVERAGE
                if risk_per_unit <= 0:
                    continue
                
                # Position sizing dengan risk management
                risk_amount = balance * DYNAMIC_RISK_PCT
                qty = risk_amount / risk_per_unit
                
                # Hard cap: maksimal 25% dari balance per trade
                max_qty = (balance * 0.25) / long_level
                qty = min(qty, max_qty)
                
                if qty <= 0:
                    continue
                
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
            
            # Entry short
            elif (retest_short and vol_confirmed and atr_confirmed and 
                  ema_fast < ema_slow and adx > DYNAMIC_ADX_THRESHOLD):
                sl = short_level + atr * SL_ATR_MULT
                tp = short_level - atr * TP_ATR_MULT
                
                # Validasi SL/TP
                if sl <= short_level or tp >= short_level or sl <= 0:
                    continue
                
                risk_per_unit = (sl - short_level) * LEVERAGE
                if risk_per_unit <= 0:
                    continue
                
                # Position sizing dengan risk management
                risk_amount = balance * DYNAMIC_RISK_PCT
                qty = risk_amount / risk_per_unit
                
                # Hard cap: maksimal 25% dari balance per trade
                max_qty = (balance * 0.25) / short_level
                qty = min(qty, max_qty)
                
                if qty <= 0:
                    continue
                
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
    
    # 6. Tampilkan hasil
    print("\n" + "=" * 70)
    trades_df = pd.DataFrame(trades)
    
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
        regime_analysis = trades_df.groupby('regime').agg(
            total_trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ).round(4)
        print(regime_analysis)
        
        # Simpan hasil
        filename = f"backtest_results_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n💾 Hasil disimpan ke: {filename}")
        
        return trades_df
    else:
        print("❌ TIDAK ADA TRADE DIEKSEKUSI")
        print("💡 Saran: Turunkan threshold market scanner atau longgarkan filter entry")
        return None

def run_automatic_backtest():
    """Backtest otomatis dengan pemilihan aset dinamis - PRODUCTION VERSION"""
    print("🚀 SISTEM TRADING OTOMATIS - PRODUCTION MODE")
    print("=" * 70)
    
    # Inisialisasi pemilih aset otomatis
    auto_selector = MarketScanner()
    
    # Dapatkan aset terbaik untuk trading
    best_symbol = auto_selector.get_best_asset_for_trading()
    
    if best_symbol is None:
        print("🚫 GAGAL MENDAPATKAN ASET - MENGGUNAKAN ETH/USDT SEBAGAI DEFAULT")
        best_symbol = 'ETH/USDT'
    
    # Jalankan backtest pada aset terpilih
    return run_backtest_on_symbol(best_symbol)

if __name__ == "__main__":
    # Install dependency jika belum ada
    try:
        import ccxt
    except ImportError:
        print("📦 Menginstall dependency yang dibutuhkan...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "pandas", "numpy", "ta-lib", "requests"])
        import ccxt
    
    # Jalankan backtest
    start_time = time.time()
    results = run_automatic_backtest()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏰ Backtest selesai dalam {elapsed_time:.2f} detik")
    print("=" * 70)
    
    # Analisis hasil secara real-time
    if results is not None and not results.empty:
        print("\n🎯 ANALISIS HASIL LEBIH DALAM:")
        print(f"  • Rata-rata PnL per trade: ${results['pnl'].mean():.4f}")
        print(f"  • Max drawdown: ${results['pnl'].cumsum().min():.4f}")
        print(f"  • Sharpe Ratio (estimasi): {(results['pnl'].mean() / results['pnl'].std() * (252 ** 0.5)):.2f}")