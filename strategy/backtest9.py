# backtest_with_market_scanner.py (versi FIX)
import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import requests
import time

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

# --- MARKET SCANNER CLASS ---
class MarketScanner:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        self.min_volume_usd = 1000000  # Minimal volume 24 jam $1 juta
        self.min_24h_change = 2.0      # Minimal pergerakan 24 jam 2%
        self.max_symbols = 10          # Maksimal aset yang discan
        self.min_volatility_multiplier = 1.5
        self.min_volume_multiplier = 1.8
        self.min_adx_threshold = 18
        self.btc_correlation_threshold = 0.25
    
    def get_trending_symbols(self):
        """Ambil daftar aset trending dari Binance secara otomatis"""
        print("🔍 MENGAMBIL DAFTAR ASET TRENDING DARI BINANCE...")
        
        try:
            # Ambil semua ticker data
            tickers = self.exchange.fetch_tickers()
            
            # Filter hanya USDT pairs yang aktif
            usdt_pairs = {}
            for symbol, data in tickers.items():
                if (symbol.endswith('/USDT') and 
                    'quoteVolume' in data and 
                    data['quoteVolume'] > self.min_volume_usd and
                    abs(data['percentage']) > self.min_24h_change):
                    usdt_pairs[symbol] = data
            
            # Convert ke DataFrame untuk analisis
            if not usdt_pairs:
                print("⚠️ Tidak ada aset yang memenuhi kriteria volume dan pergerakan")
                return self.get_default_symbols()
            
            df = pd.DataFrame.from_dict(usdt_pairs, orient='index')
            df['symbol'] = df.index
            df['volume_usd'] = df['quoteVolume']
            df['change_24h'] = df['percentage']
            
            # Sortir by volume dan ambil top N
            top_symbols = df.sort_values('volume_usd', ascending=False).head(self.max_symbols)
            
            print(f"✅ BERHASIL MENDAPATKAN {len(top_symbols)} ASET TRENDING:")
            for i, (_, row) in enumerate(top_symbols.iterrows()):
                print(f"   #{i+1} {row['symbol']} | Volume: ${row['volume_usd']:,.0f} | Change: {row['change_24h']:+.2f}%")
            
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
            'INJ/USDT', 'TON/USDT', 'PEPE/USDT', 'SHIB/USDT', 'ARB/USDT'
        ]
        return default_symbols[:self.max_symbols]
    
    def get_social_sentiment(self, symbol):
        """Ambil sentimen sosial dari API (opsional) - FIXED VERSION"""
        try:
            # Perbaikan 1: Hapus spasi di URL
            coin_id = symbol.split('/')[0].lower()
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
            
            print(f"🔍 Mengambil sentimen untuk {symbol} dari: {url}")
            response = requests.get(url, timeout=8, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Perbaikan 2: Cek status code secara eksplisit
            if response.status_code != 200:
                print(f"⚠️ API error untuk {symbol}: Status code {response.status_code}")
                print(f"   Response: {response.text[:100]}...")
                return 50  # Return default score
            
            data = response.json()
            
            # Perbaikan 3: Validasi struktur data sebelum akses
            if 'market_data' not in data or 'market_cap_rank' not in data['market_data']:
                print(f"⚠️ Data tidak lengkap untuk {symbol}: market_data tidak tersedia")
                return 50
            
            # Perbaikan 4: Tambahkan metrics lain untuk sentimen
            market_cap_rank = data['market_data'].get('market_cap_rank', 100)
            sentiment_score = max(0, 100 - market_cap_rank)
            
            # Tambahkan skor berdasarkan popularitas developer
            dev_score = 0
            if 'developer_score' in data:
                dev_score = data['developer_score'] * 10  # Skala 0-100
            
            # Tambahkan skor berdasarkan komunitas
            community_score = 0
            if 'community_score' in data:
                community_score = data['community_score'] * 10
            
            # Gabungkan skor
            final_score = min(100, sentiment_score + dev_score + community_score)
            
            print(f"✅ Berhasil mengambil sentimen {symbol}:")
            print(f"   Market Cap Rank: #{market_cap_rank}")
            print(f"   Developer Score: {dev_score:.1f}")
            print(f"   Community Score: {community_score:.1f}")
            print(f"   📊 FINAL SENTIMEN SCORE: {final_score:.1f}/100")
            
            return final_score
            
        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout mengambil sentimen untuk {symbol} (8 detik)")
            return 50
        except requests.exceptions.ConnectionError:
            print(f"🌐 Koneksi gagal untuk {symbol} - cek internet Anda")
            return 50
        except requests.exceptions.RequestException as e:
            print(f"📡 Error request untuk {symbol}: {str(e)}")
            return 50
        except ValueError as e:
            print(f"❌ Error parsing JSON untuk {symbol}: {str(e)}")
            return 50
        except Exception as e:
            print(f"🔥 Error tidak terduga untuk {symbol}: {str(e)}")
            print(f"   Tipe error: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return 50
    
    def rank_symbols_by_activity(self, symbols):
        """Rangking aset berdasarkan aktivitas pasar terkini"""
        print("\n⚡ MENGANALISIS AKTIVITAS PASAR TERKINI...")
        ranked_symbols = []
        
        for symbol in symbols:
            try:
                print(f"📊 Menganalisis {symbol}...")
                # Ambil data 1 jam terakhir (12 candle 5m)
                ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=12)
                
                if len(ohlcv) < 12:
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Hitung indikator aktivitas
                df['price_change_pct'] = (df['close'] / df['close'].iloc[0] - 1) * 100
                df['volume_ma'] = df['volume'].rolling(5).mean()
                df['volume_spike'] = df['volume'] > df['volume_ma'] * 1.5
                
                # Skor aktivitas
                price_change_score = abs(df['price_change_pct'].iloc[-1]) * 2
                volume_score = (df['volume'].sum() / df['volume'].mean()) * 10
                spike_count = df['volume_spike'].sum()
                
                # Social sentiment (opsional)
                sentiment_score = self.get_social_sentiment(symbol) / 10
                # Total score
                activity_score = (
                    price_change_score * 0.4 +
                    volume_score * 0.3 +
                    spike_count * 2 * 0.2 +
                    sentiment_score * 0.1
                )
                
                ranked_symbols.append({
                    'symbol': symbol,
                    'activity_score': activity_score,
                    'price_change': df['price_change_pct'].iloc[-1],
                    'volume_total': df['volume'].sum(),
                    'spike_count': spike_count
                })
                
                print(f"   ✓ {symbol} - Skor Aktivitas: {activity_score:.1f} | Pergerakan: {df['price_change_pct'].iloc[-1]:+.2f}%")
                
            except Exception as e:
                print(f"   ✗ Error menganalisis {symbol}: {e}")
                continue
        
        # Urutkan berdasarkan skor aktivitas
        ranked_symbols.sort(key=lambda x: x['activity_score'], reverse=True)
        
        print("\n🏆 HASIL PERINGKAT AKTIVITAS PASAR:")
        for i, asset in enumerate(ranked_symbols[:5]):  # Tampilkan top 5
            print(f"  #{i+1} {asset['symbol']:<10} | Skor: {asset['activity_score']:5.1f} | Pergerakan: {asset['price_change']:+6.2f}% | Volume Spike: {asset['spike_count']}")
        
        return ranked_symbols
    
    def get_best_asset_for_trading(self):
        """Dapatkan aset terbaik untuk trading saat ini"""
        # Langkah 1: Ambil daftar aset trending
        trending_symbols = self.get_trending_symbols()
        
        # Langkah 2: Rangking berdasarkan aktivitas terkini
        ranked_assets = self.rank_symbols_by_activity(trending_symbols)
        
        if not ranked_assets:
            print("❌ Tidak ada aset yang bisa dianalisis!")
            return None
        
        # Ambil aset dengan skor tertinggi
        best_asset = ranked_assets[0]
        
        print(f"\n🎯 ASET TERBAIK UNTUK TRADING SAAT INI: {best_asset['symbol']}")
        print(f"   Skor Aktivitas: {best_asset['activity_score']:.1f}")
        print(f"   Pergerakan 1 Jam: {best_asset['price_change']:+.2f}%")
        print(f"   Jumlah Volume Spike: {best_asset['spike_count']}")
        
        return best_asset['symbol']

    def calculate_market_score(self, df, btc_df=None):
        """Hitung skor pasar (0.0 - 1.0) untuk menentukan apakah strategi aktif"""
        if len(df) < 100:
            return 0.0
        
        # 1. Skor Volatilitas (0.3 weight)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['avg_atr'] = df['atr'].rolling(100).mean()
        vol_score = min(1.0, df['atr'].iloc[-1] / df['avg_atr'].iloc[-1] / 2.0)
        vol_score = max(0.0, vol_score)
        
        # 2. Skor Volume (0.3 weight)
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['avg_vol_ma'] = df['vol_ma'].rolling(100).mean()
        vol_ma_score = min(1.0, df['vol_ma'].iloc[-1] / df['avg_vol_ma'].iloc[-1] / 2.0)
        vol_ma_score = max(0.0, vol_ma_score)
        
        # 3. Skor Tren (0.4 weight)
        df['plus_di'], df['minus_di'], df['adx'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14), \
                                                  talib.MINUS_DI(df['high'], df['low'], df['close'], 14), \
                                                  talib.ADX(df['high'], df['low'], df['close'], 14)
        
        trend_score = 0
        if df['adx'].iloc[-1] > self.min_adx_threshold:
            trend_score += 0.4 * (df['adx'].iloc[-1] / 30)  # Max 30 = score 0.4
        
        if df['plus_di'].iloc[-1] > df['minus_di'].iloc[-1]:
            trend_score += 0.2
        elif df['minus_di'].iloc[-1] > df['plus_di'].iloc[-1]:
            trend_score += 0.2
        
        # 4. Skor Korelasi dengan BTC (bonus 0.1)
        btc_bonus = 0.0
        if btc_df is not None:
            btc_corr = self.get_btc_correlation(df, btc_df)
            if btc_corr > self.btc_correlation_threshold:
                btc_bonus = 0.1
        
        # Hitung total score
        total_score = (vol_score * 0.3 + vol_ma_score * 0.3 + trend_score * 0.4) + btc_bonus
        total_score = min(1.0, total_score)
        
        # Debug info
        print(f"\n📊 MARKET SCORE CALCULATION:")
        print(f"  Volatility Score: {vol_score:.2f} (ATR: {df['atr'].iloc[-1]:.4f}, Avg ATR: {df['avg_atr'].iloc[-1]:.4f})")
        print(f"  Volume Score: {vol_ma_score:.2f} (Vol MA: {df['vol_ma'].iloc[-1]:.0f}, Avg Vol MA: {df['avg_vol_ma'].iloc[-1]:.0f})")
        print(f"  Trend Score: {trend_score:.2f} (ADX: {df['adx'].iloc[-1]:.1f}, +DI: {df['plus_di'].iloc[-1]:.1f}, -DI: {df['minus_di'].iloc[-1]:.1f})")
        print(f"  BTC Bonus: {btc_bonus:.2f}")
        print(f"  ✅ TOTAL MARKET SCORE: {total_score:.2f} (Threshold: {MIN_SCAN_SCORE})")
        
        return total_score
    
    def get_btc_correlation(self, symbol_df, btc_df, window=50):
        """Hitung korelasi dengan BTC"""
        try:
            # Pastikan kedua DataFrame memiliki timestamp yang sama
            merged = pd.merge(symbol_df[['timestamp', 'close']], 
                             btc_df[['timestamp', 'close']], 
                             on='timestamp', 
                             suffixes=('_symbol', '_btc'))
            
            if len(merged) < window:
                return 0.0
            
            symbol_returns = merged['close_symbol'].pct_change()
            btc_returns = merged['close_btc'].pct_change()
            
            correlation = symbol_returns[-window:].corr(btc_returns[-window:])
            return correlation
        except Exception as e:
            print(f"⚠️ Error calculating BTC correlation: {e}")
            return 0.0
    
    def is_market_hot(self, score):
        """Apakah market layak ditrading?"""
        return score >= MIN_SCAN_SCORE

# --- FUNGSI AMBIL DATA DARI BINANCE ---
def fetch_ohlcv_data(symbol, timeframe, limit):
    """Ambil data OHLCV terbaru dari Binance"""
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot'
        }
    })

    try:
        print(f"📡 Mengambil data {symbol} ({timeframe}) dari Binance...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Konversi ke DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"✅ Data berhasil diambil: {len(df)} candle")
        print(f"  Waktu terakhir: {df['timestamp'].iloc[-1]}")
        return df
    
    except Exception as e:
        print(f"❌ Error mengambil data {symbol}: {e}")
        print("Fallback ke data dummy...")
        # Data dummy untuk testing
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='5min')
        prices = np.random.normal(20, 1, limit).cumsum() + 19
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices + 0.05,
            'volume': np.random.randint(10000, 50000, limit)
        })

# --- FUNGSI REGIME (dari kode asli) ---
def enhanced_trend_detection(row_idx, df):
    """Deteksi regime pasar (sama seperti kode asli)"""
    if row_idx < 30:
        return "MIXED"
    adx = df['adx'].iloc[row_idx]
    plus_di = df['plus_di'].iloc[row_idx]
    minus_di = df['minus_di'].iloc[row_idx]
    ema_fast = df['ema_fast'].iloc[row_idx]
    ema_slow = df['ema_slow'].iloc[row_idx]
    
    trend_score = 0
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

    if trend_score >= 4:
        return "STRONG_TREND"
    elif trend_score <= 2:
        return "SIDEWAYS"
    else:
        return "MIXED"

# --- MAIN PROGRAM ---
def run_backtest_on_symbol(symbol):
    """Jalankan backtest pada satu simbol yang dipilih"""
    print(f"\n⚔️ MENJALANKAN BACKTEST PADA {symbol}...")
    print("=" * 50)
    
    # 1. Ambil data OHLCV
    df = fetch_ohlcv_data(symbol, TIMEFRAME, BARS_TO_FETCH)
    
    # 2. Ambil data BTC untuk korelasi (opsional)
    try:
        btc_df = fetch_ohlcv_data('BTC/USDT', TIMEFRAME, BARS_TO_FETCH)
    except:
        btc_df = None
        print("⚠️ BTC data tidak tersedia - skip korelasi")
    
    # 3. Inisialisasi market scanner
    scanner = MarketScanner()
    market_score = scanner.calculate_market_score(df, btc_df)
    is_market_hot = scanner.is_market_hot(market_score)
    
    print(f"\n🎯 KEPUTUSAN MARKET SCANNER:")
    if is_market_hot:
        print(f"✅ PASAR '{symbol}' HOT! Skor: {market_score:.2f} - Strategi AKTIF")
    else:
        print(f"❌ PASAR '{symbol}' DINGIN! Skor: {market_score:.2f} - Strategi NON-AKTIF")
        print("💡 Saran: Cari pair lain atau tunggu kondisi pasar lebih baik")
        # Tetap jalankan tapi dengan warning
        print("🔄 Tetap menjalankan strategi karena ini aset paling happening...")
    
    # 4. Siapkan data untuk backtest
    print("\n🔧 Menyiapkan indikator...")
    df['ema_fast'] = talib.EMA(df['close'], 20)
    df['ema_slow'] = talib.EMA(df['close'], 50)
    df['rsi'] = talib.RSI(df['close'], 14)
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
    df['plus_di'], df['minus_di'], df['adx'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14), \
                                               talib.MINUS_DI(df['high'], df['low'], df['close'], 14), \
                                               talib.ADX(df['high'], df['low'], df['close'], 14)
    df['vol_ma'] = df['volume'].rolling(10).mean()
    
    # 5. Jalankan backtest
    balance = INITIAL_BALANCE
    position = None
    trades = []
    
    for i in range(30, len(df)):
        # Periksa posisi yang sedang berjalan
        if position is not None:
            current_price = df['close'].iloc[i]
            if position['side'] == 'LONG':
                if current_price >= position['tp'] or current_price <= position['sl']:
                    pnl = (current_price - position['entry']) * position['qty'] * LEVERAGE
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': df['timestamp'].iloc[i],
                        'side': 'LONG',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'regime': position['regime'],
                        'symbol': symbol
                    })
                    position = None
                    print(f"✅ EXIT LONG @ {current_price:.4f} | PnL: {pnl:.4f} | Balance: {balance:.4f}")
            else:  # SHORT
                if current_price <= position['tp'] or current_price >= position['sl']:
                    pnl = (position['entry'] - current_price) * position['qty'] * LEVERAGE
                    balance += pnl
                    trades.append({
                        'entry_time': position['entry_time'],
                        'exit_time': df['timestamp'].iloc[i],
                        'side': 'SHORT',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': balance,
                        'regime': position['regime'],
                        'symbol': symbol
                    })
                    position = None
                    print(f"✅ EXIT SHORT @ {current_price:.4f} | PnL: {pnl:.4f} | Balance: {balance:.4f}")
        
        # Cari entry baru jika tidak ada posisi
        if position is None and i >= 1:
            # Gunakan regime detection
            regime = enhanced_trend_detection(i, df)
            if regime != "STRONG_TREND":
                continue
            
            # Parameter entry
            atr = df['atr'].iloc[i]
            close = df['close'].iloc[i]
            volume = df['volume'].iloc[i]
            vol_ma = df['vol_ma'].iloc[i]
            ema_fast = df['ema_fast'].iloc[i]
            ema_slow = df['ema_slow'].iloc[i]
            
            # Level breakout dari candle sebelumnya
            prev_close = df['close'].iloc[i-1]
            long_level = prev_close + atr * LEVEL_MULT
            short_level = prev_close - atr * LEVEL_MULT
            
            # Cek breakout dan retest
            broke_long_prev = df['high'].iloc[i-1] >= long_level
            broke_short_prev = df['low'].iloc[i-1] <= short_level
            
            retest_long = broke_long_prev and (df['low'].iloc[i] <= long_level) and (close > long_level * 0.998)
            retest_short = broke_short_prev and (df['high'].iloc[i] >= short_level) and (close < short_level * 1.002)
            
            # Konfirmasi volume
            vol_confirmed = volume >= vol_ma * 1.5  # Lebih realistis daripada 2x
            
            # Entry long
            if retest_long and vol_confirmed and ema_fast > ema_slow:
                sl = long_level - atr * SL_ATR_MULT
                risk_per_unit = (long_level - sl) * LEVERAGE
                if risk_per_unit > 0 and sl > 0:
                    qty = (balance * RISK_PCT) / risk_per_unit
                    tp = long_level + atr * TP_ATR_MULT
                    position = {
                        'side': 'LONG',
                        'entry': long_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': df['timestamp'].iloc[i],
                        'regime': regime
                    }
                    print(f"🟢 LONG entry @ {long_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f}")
            
            # Entry short
            elif retest_short and vol_confirmed and ema_fast < ema_slow:
                sl = short_level + atr * SL_ATR_MULT
                risk_per_unit = (sl - short_level) * LEVERAGE
                if risk_per_unit > 0 and sl > 0:
                    qty = (balance * RISK_PCT) / risk_per_unit
                    tp = short_level - atr * TP_ATR_MULT
                    position = {
                        'side': 'SHORT',
                        'entry': short_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': df['timestamp'].iloc[i],
                        'regime': regime
                    }
                    print(f"🔴 SHORT entry @ {short_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f}")
    
    # 6. Tampilkan hasil
    print("\n" + "=" * 50)
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
        print(trades_df.groupby('regime').agg(
            total_trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ))
        
        # Simpan hasil
        filename = f"backtest_results_{symbol.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n💾 Hasil disimpan ke: {filename}")
    else:
        print("❌ TIDAK ADA TRADE DIEKSEKUSI")
        print("💡 Saran: Turunkan threshold market scanner atau longgarkan filter entry")

def run_automatic_backtest():
    """Backtest otomatis dengan pemilihan aset dinamis"""
    print("🚀 SISTEM TRADING OTOMATIS - PEMILIHAN ASET DINAMIS")
    print("=" * 70)
    
    # Inisialisasi pemilih aset otomatis
    auto_selector = MarketScanner()
    
    # Dapatkan aset terbaik untuk trading
    best_symbol = auto_selector.get_best_asset_for_trading()
    
    if best_symbol is None:
        print("🚫 GAGAL MENDAPATKAN ASET - MENGGUNAKAN AVAX/USDT SEBAGAI DEFAULT")
        best_symbol = 'AVAX/USDT'
    
    # Jalankan backtest pada aset terpilih
    run_backtest_on_symbol(best_symbol)

if __name__ == "__main__":
    # Install dependency jika belum ada
    try:
        import ccxt
    except ImportError:
        print("📦 Menginstall dependency yang dibutuhkan...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "pandas", "numpy", "TA-Lib", "requests"])
        import ccxt
    
    # Jalankan backtest
    start_time = time.time()
    run_automatic_backtest()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏰ Backtest selesai dalam {elapsed_time:.2f} detik")
    print("=" * 70)