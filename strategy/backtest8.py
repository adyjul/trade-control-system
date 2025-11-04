# backtest_with_market_scanner.py
import pandas as pd
import numpy as np
import talib
import ccxt
from datetime import datetime, timedelta
import time

# --- CONFIG UTAMA ---
INITIAL_BALANCE = 20.0
RISK_PCT = 0.008  # 0.8%
LEVERAGE = 10
TP_ATR_MULT = 3.0
SL_ATR_MULT = 2.5
LEVEL_MULT = 1.0
SYMBOL = 'AVAX/USDT'  # Bisa ganti jadi 'BTC/USDT', 'ETH/USDT', dll
TIMEFRAME = '5m'
BARS_TO_FETCH = 1000  # Ambil 1000 candle terbaru
MIN_SCAN_SCORE = 0.7  # Minimal score untuk aktifkan strategi

# --- MARKET SCANNER CLASS ---
class MarketScanner:
    def __init__(self):
        self.min_volatility_multiplier = 1.5
        self.min_volume_multiplier = 1.8
        self.min_adx_threshold = 18
        self.btc_correlation_threshold = 0.25
    
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
        print(f"❌ Error mengambil data: {e}")
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
def run_backtest():
   """Jalankan backtest multi-aset dengan market scanner"""
    print("🚀 MEMULAI BACKTEST MULTI-ASSET")
    print("=" * 50)
    
    # 1. Ambil data BTC sebagai benchmark (wajib untuk korelasi)
    try:
        btc_df = fetch_ohlcv_data('BTC/USDT', TIMEFRAME, BARS_TO_FETCH)
        print("✅ Data BTC/USDT berhasil diambil")
    except Exception as e:
        print(f"⚠️ Gagal mengambil data BTC: {e}")
        btc_df = None
    
    # 2. Daftar aset yang akan di-scan
    symbols = ['AVAX/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'DOGE/USDT']
    asset_data = {}
    hot_assets = []
    
    # 3. Ambil data dan scan semua aset
    print("\n🔍 SCANNING SEMUA ASET...")
    scanner = MarketScanner()
    
    for symbol in symbols:
        try:
            df = fetch_ohlcv_data(symbol, TIMEFRAME, BARS_TO_FETCH)
            
            # Hitung skor market
            market_score = scanner.calculate_market_score(df, btc_df)
            is_hot = scanner.is_market_hot(market_score)
            
            if is_hot:
                hot_assets.append(symbol)
                print(f"🔥 {symbol} - SKOR: {market_score:.2f} ✅ HOT (STRATEGI AKTIF)")
            else:
                print(f"❄️ {symbol} - SKOR: {market_score:.2f} ❌ COLD (STRATEGI NON-AKTIF)")
            
            # Simpan data untuk backtest nanti
            asset_data[symbol] = {
                'df': df,
                'is_hot': is_hot,
                'score': market_score
            }
            
        except Exception as e:
            print(f"❌ Error memproses {symbol}: {e}")
    
    if not hot_assets:
        print("\n🚫 TIDAK ADA ASET YANG 'HOT' - BACKTEST DIBATALKAN")
        print("💡 Saran: Turunkan MIN_SCAN_SCORE atau tunggu kondisi pasar lebih baik")
        return
    
    print(f"\n🎯 ASET YANG AKTIF: {', '.join(hot_assets)}")
    
    # 4. Persiapkan data untuk semua aset
    print("\n🔧 MENYIAPKAN INDIKATOR UNTUK SEMUA ASET...")
    for symbol in asset_data:
        df = asset_data[symbol]['df']
        
        # Hitung indikator teknikal
        df['ema_fast'] = talib.EMA(df['close'], 20)
        df['ema_slow'] = talib.EMA(df['close'], 50)
        df['rsi'] = talib.RSI(df['close'], 14)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        df['plus_di'], df['minus_di'], df['adx'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14), \
                                                   talib.MINUS_DI(df['high'], df['low'], df['close'], 14), \
                                                   talib.ADX(df['high'], df['low'], df['close'], 14)
        df['vol_ma'] = df['volume'].rolling(10).mean()
        
        asset_data[symbol]['df'] = df
    
    # 5. Simulasi multi-asset trading
    print("\n⚔️ MEMULAI SIMULASI MULTI-ASSET TRADING...")
    total_balance = INITIAL_BALANCE
    positions = {}  # Simpan posisi per aset
    all_trades = []
    risk_per_asset = 0.02  # Risk 2% dari total balance per aset
    
    # Loop melalui semua waktu (gunakan timestamp BTC sebagai referensi)
    timestamps = btc_df['timestamp'].tolist() if btc_df is not None else asset_data[hos_assets[0]]['df']['timestamp'].tolist()
    
    for i, timestamp in enumerate(timestamps):
        if i < 30:  # Skip periode awal untuk indikator
            continue
        
        # Periksa exit untuk semua posisi yang aktif
        for symbol in list(positions.keys()):
            position = positions[symbol]
            current_price = asset_data[symbol]['df'].iloc[i]['close']
            
            if position['side'] == 'LONG':
                if current_price >= position['tp'] or current_price <= position['sl']:
                    pnl = (current_price - position['entry']) * position['qty'] * LEVERAGE
                    total_balance += pnl
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'side': 'LONG',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': total_balance,
                        'regime': position['regime'],
                        'symbol': symbol
                    }
                    all_trades.append(trade_record)
                    
                    print(f"✅ EXIT LONG {symbol} @ {current_price:.4f} | PnL: {pnl:.4f} | Balance: {total_balance:.4f}")
                    del positions[symbol]
            
            elif position['side'] == 'SHORT':
                if current_price <= position['tp'] or current_price >= position['sl']:
                    pnl = (position['entry'] - current_price) * position['qty'] * LEVERAGE
                    total_balance += pnl
                    
                    trade_record = {
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'side': 'SHORT',
                        'entry': position['entry'],
                        'exit': current_price,
                        'pnl': pnl,
                        'balance': total_balance,
                        'regime': position['regime'],
                        'symbol': symbol
                    }
                    all_trades.append(trade_record)
                    
                    print(f"✅ EXIT SHORT {symbol} @ {current_price:.4f} | PnL: {pnl:.4f} | Balance: {total_balance:.4f}")
                    del positions[symbol]
        
        # Cari entry baru untuk aset yang 'hot' dan belum ada posisi
        for symbol in hot_assets:
            if symbol in positions:  # Sudah ada posisi aktif
                continue
            
            df = asset_data[symbol]['df']
            if i >= len(df):
                continue
            
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
            vol_confirmed = volume >= vol_ma * 1.5
            
            # Entry long
            if retest_long and vol_confirmed and ema_fast > ema_slow:
                sl = long_level - atr * SL_ATR_MULT
                risk_per_unit = (long_level - sl) * LEVERAGE
                
                if risk_per_unit > 0 and sl > 0:
                    # Hitung position size berdasarkan risk management per aset
                    risk_amount = total_balance * risk_per_asset
                    qty = risk_amount / risk_per_unit
                    
                    tp = long_level + atr * TP_ATR_MULT
                    positions[symbol] = {
                        'side': 'LONG',
                        'entry': long_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': timestamp,
                        'regime': regime
                    }
                    
                    print(f"🟢 ENTRY LONG {symbol} @ {long_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f}")
                    continue  # Skip ke aset berikutnya
            
            # Entry short
            elif retest_short and vol_confirmed and ema_fast < ema_slow:
                sl = short_level + atr * SL_ATR_MULT
                risk_per_unit = (sl - short_level) * LEVERAGE
                
                if risk_per_unit > 0 and sl > 0:
                    # Hitung position size berdasarkan risk management per aset
                    risk_amount = total_balance * risk_per_asset
                    qty = risk_amount / risk_per_unit
                    
                    tp = short_level - atr * TP_ATR_MULT
                    positions[symbol] = {
                        'side': 'SHORT',
                        'entry': short_level,
                        'tp': tp,
                        'sl': sl,
                        'qty': qty,
                        'entry_time': timestamp,
                        'regime': regime
                    }
                    
                    print(f"🔴 ENTRY SHORT {symbol} @ {short_level:.4f} | SL: {sl:.4f} | TP: {tp:.4f} | Qty: {qty:.4f}")
    
    # 6. Tampilkan hasil akhir
    print("\n" + "=" * 50)
    trades_df = pd.DataFrame(all_trades)
    
    if not trades_df.empty:
        win_rate = len(trades_df[trades_df['pnl'] > 0]) / len(trades_df)
        total_pnl = trades_df['pnl'].sum()
        profit_factor = trades_df[trades_df['pnl'] > 0]['pnl'].sum() / abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if len(trades_df[trades_df['pnl'] < 0]) > 0 else float('inf')
        
        print(f"📈 TOTAL TRADES: {len(trades_df)}")
        print(f"✅ WIN RATE: {win_rate:.2%}")
        print(f"💰 TOTAL PnL: {total_pnl:.4f} USDT ({(total_pnl/INITIAL_BALANCE)*100:.2f}%)")
        print(f"🎯 FINAL BALANCE: {total_balance:.4f} USDT")
        print(f"📊 PROFIT FACTOR: {profit_factor:.2f}")
        
        # Analisis per aset
        print("\n--- ANALISIS PER ASET ---")
        print(trades_df.groupby('symbol').agg(
            total_trades=('pnl', 'count'),
            win_rate=('pnl', lambda x: (x > 0).mean()),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ).sort_values('total_pnl', ascending=False))
        
        # Simpan hasil
        filename = f"multi_asset_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\n💾 Hasil disimpan ke: {filename}")
    else:
        print("❌ TIDAK ADA TRADE DIEKSEKUSI")
        print("💡 Saran: Turunkan threshold market scanner atau longgarkan filter entry")

if __name__ == "__main__":
    # Install dependency jika belum ada
    try:
        import ccxt
    except ImportError:
        print("📦 Menginstall dependency yang dibutuhkan...")
        import sys
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ccxt", "pandas", "numpy", "TA-Lib"])
        import ccxt
    
    # Jalankan backtest
    start_time = time.time()
    run_backtest()
    elapsed_time = time.time() - start_time
    
    print(f"\n⏰ Backtest selesai dalam {elapsed_time:.2f} detik")
    print("=" * 50)