# strategy/backtest.py
import os
from datetime import datetime, timezone
import pandas as pd
import ta
from binance.client import Client
import glob
from strategy.utils import calculate_support_resistance
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP
import numpy as np

# ---- Optional: set default folder kalau belum ada di config/env ----
DEFAULT_DATA_DIR = "./data_backtest"
DEFAULT_RESULT_DIR = "./backtest_result"

def evaluate_tp_sl(df: pd.DataFrame, look_ahead=7) -> pd.DataFrame:
    """
    Evaluasi TP / SL dengan melihat N (look_ahead) candle setelah sinyal keluar.
    """
    df['exit_status'] = 'NO HIT'
    idxs = list(df.index)
    for i, ts in enumerate(idxs):
        row = df.loc[ts]
        signal = row['signal']
        tp = row['tp_price']
        sl = row['sl_price']
        idx_start = df.index.get_loc(row.name)

        # ambil 6 candle ke depan
        # future_slice = df.loc[idxs[i+1:i+1+look_ahead]]
        # future_slice = df.iloc[idx_start+1:idx_start+7]
        future_slice = df.iloc[idx_start+1:idx_start+1+look_ahead]
        for _, f in future_slice.iterrows():
            if signal == 'LONG':
                if f['high'] >= tp:
                    df.at[ts, 'exit_status'] = 'TP HIT'
                    break
                elif f['low'] <= sl:
                    df.at[ts, 'exit_status'] = 'SL HIT'
                    break
            elif signal == 'SHORT':
                if f['low'] <= tp:
                    df.at[ts, 'exit_status'] = 'TP HIT'
                    break
                elif f['high'] >= sl:
                    df.at[ts, 'exit_status'] = 'SL HIT'
                    break
    return df

def is_false_reversal(row, df, atr_window=14, ma_fast=50, ma_slow=100):
    """
    Deteksi apakah sinyal reversal rawan false.
    Return True jika dianggap false reversal (harus di-filter)
    """
    idx = df.index.get_loc(row.name)
    close = row['close']
    
    # Ambil data historis sebelum current bar
    if idx < max(atr_window, ma_slow):
        return False  # data belum cukup
    
    # Moving Average filter
    ma50 = df['close'].rolling(ma_fast).mean().iloc[idx]
    ma100 = df['close'].rolling(ma_slow).mean().iloc[idx]

    # ATR filter
    high = df['high'].iloc[idx - atr_window:idx]
    low = df['low'].iloc[idx - atr_window:idx]
    close_prev = df['close'].iloc[idx - atr_window:idx]
    tr1 = high - low
    tr2 = abs(high - close_prev.shift(1))
    tr3 = abs(low - close_prev.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_window).mean().iloc[-1]

    # Konfirmasi multi-candle (2 candle searah)
    last2 = df['close'].iloc[idx-2:idx].reset_index(drop=True)
    bullish_confirm = all(last2[i] > df['open'].iloc[idx-2+i] for i in range(2))
    bearish_confirm = all(last2[i] < df['open'].iloc[idx-2+i] for i in range(2))

    # Deteksi false reversal untuk LONG
    if row['signal'] == 'LONG':
        if close < ma50 or close < ma100:  # masih di bawah tren
            return True
        if not bullish_confirm:  # belum ada 2 candle confirm
            return True
        if (df['open'].iloc[idx-1] - df['close'].iloc[idx-1]) > atr:  # candle merah terakhir besar
            return True

    # Deteksi false reversal untuk SHORT
    if row['signal'] == 'SHORT':
        if close > ma50 or close > ma100:  # masih di atas tren
            return True
        if not bearish_confirm:  # belum ada 2 candle confirm
            return True
        if (df['close'].iloc[idx-1] - df['open'].iloc[idx-1]) > atr:  # candle hijau terakhir besar
            return True

    return False

def detect_breakout(row):
    if pd.isna(row['prev_high']) or pd.isna(row['prev_close']) or pd.isna(row['prev_open']):
        return False  # tidak cukup data

    # Deteksi breakout palsu untuk LONG
    if row['signal'] == 'LONG':
        breakout_up = row['high'] > row['prev_high'] + row['atr'] * 0.3
        volume_spike = row['volume'] > row['volume_sma20'] * 1.5
        bullish_candle = row['close'] > row['open']
        
        # Kalau terlihat breakout tapi volume nggak kuat → patut dicurigai
        if breakout_up and not volume_spike:
            return True
    
    # Deteksi false breakdown untuk SHORT
    if row['signal'] == 'SHORT':
        big_prev_bearish = row['prev_close'] < row['prev_open'] and (row['prev_open'] - row['prev_close']) > row['atr'] * 1.5
        if big_prev_bearish:
            return True

    return False

def apply_filters(df):
    df['false_reversal'] = df.apply(lambda row: is_false_reversal(row, df), axis=1)
    # Filter sinyal → hapus kalau false_reversal = True
    df.loc[df['false_reversal'], 'signal'] = 'HOLD'
    return df

def detect_potential_breakout(df, atr_mult=0.2, vol_mult=1.2):

    # Kondisi LONG breakout
    long_cond = (
        (df['close'] >= df['resistance'] - df['atr'] * atr_mult) &  # harga dekat/tembus resistance
        (df['close'] > df['resistance']) &  # sudah melewati resistance
        (df['volume'] >= df['volume'].rolling(20).mean() * vol_mult)  # volume di atas rata-rata
    )

    # Kondisi SHORT breakout
    short_cond = (
        (df['close'] <= df['support'] + df['atr'] * atr_mult) &  # harga dekat/tembus support
        (df['close'] < df['support']) &  # sudah melewati support
        (df['volume'] >= df['volume'].rolling(20).mean() * vol_mult)  # volume di atas rata-rata
    )

    df['is_potential_breakout'] = np.where(long_cond | short_cond, 1, 0)

    return df


def detect_signal(row):
    # v1
    if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
        return 'HOLD'

    if row['atr'] < 0.005 * row['close']:
        return 'HOLD'

    if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
        return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
        if row['rsi'] < 35:
            return 'HOLD'
        return 'SHORT'

    return 'HOLD'

    # v2 mungkin untuk weekend
    # if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']):
    #     return 'HOLD'

    # Filter ATR kecil → tidak volatile
    # if row['atr'] < 0.005 * row['close']:
    #     return 'HOLD'

    # ========== LONG Condition ==========
    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     if row['rsi'] > 75:  # Overbought → hindari entry LONG
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakout
    #         return 'HOLD'
    #     return 'LONG'

    # ========== SHORT Condition ==========
    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:  # Oversold → hindari entry SHORT
    #         return 'HOLD'
    #     if row['volume'] < row['volume_sma20']:  # Volume rendah → hindari breakdown
    #         return 'HOLD'
    #     return 'SHORT'

    # return 'HOLD'

    # v3
    # if pd.isna(row['macd']) or pd.isna(row['macd_signal']) or pd.isna(row['rsi']) or pd.isna(row['volume_sma20']) or pd.isna(row['prev_high']):
    #     return 'HOLD'

    # if row['atr'] < 0.005 * row['close']:
    #     return 'HOLD'

    # breakout_up = row['high'] > row['prev_high'] + row['atr'] * 0.3
    # volume_spike = row['volume'] > row['volume_sma20'] * 1.5
    # bullish_candle = row['close'] > row['open']

    # # Cegah SHORT saat breakout atas
    # if breakout_up and volume_spike and bullish_candle:
    #     return 'HOLD'

    # # Cegah SHORT jika candle sebelumnya bearish ekstrem (false breakdown)
    # bearish_spike_prev = (
    #     not pd.isna(row['prev_close']) and not pd.isna(row['prev_open']) and
    #     row['prev_close'] < row['prev_open'] and
    #     (row['prev_open'] - row['prev_close']) > row['atr'] * 1.5
    # )
    # if bearish_spike_prev and row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     return 'HOLD'

    # if row['macd'] > row['macd_signal'] and row['rsi'] > 50:
    #     return 'LONG' if row['volume'] > row['volume_sma20'] else 'LONG_WEAK'

    # if row['macd'] < row['macd_signal'] and row['rsi'] < 50:
    #     if row['rsi'] < 35:
    #         return 'HOLD'
    #     return 'SHORT'

    # return 'HOLD'

def clear_folder(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*')):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def detect_sideways(
    df: pd.DataFrame,
    lookback: int = 20,
    atr_multiplier: float = 1.5,
    bb_width_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Deteksi kondisi sideways pada OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Data dengan kolom ['open','high','low','close','volume'].
    lookback : int
        Periode untuk Bollinger Bands & ATR.
    atr_multiplier : float
        Range high-low relatif kecil dibanding ATR.
    bb_width_threshold : float
        Ambang batas lebar Bollinger Band (dalam persentase harga).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom tambahan 'sideways_signal' (bool).
    """

    # ATR untuk volatilitas
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=lookback
    ).average_true_range()

    # Bollinger Band width
    bb_indicator = ta.volatility.BollingerBands(
        close=df["close"], window=lookback, window_dev=2
    )
    bb_width = (bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()) / df["close"]
    df["bb_width"] = bb_width

    # Range relatif kecil dibanding ATR
    rolling_high = df["high"].rolling(window=lookback).max()
    rolling_low = df["low"].rolling(window=lookback).min()
    range_size = (rolling_high - rolling_low)

    condition_atr = range_size < atr_multiplier * df["atr"]
    condition_bb = df["bb_width"] < bb_width_threshold

    df["sideways_signal"] = condition_atr & condition_bb

    return df
    
        
def run_full_backtest(
    pairs,
    timeframe: str,
    limit: int,
    look_ahead: int = 6,
    tp_atr_mult: float = 1.2,
   
    sl_atr_mult: float = 0.9,
    data_dir: str = DEFAULT_DATA_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    save_summary: bool = True
):
    
    """
    1) Scrape OHLCV dari Binance Futures
    2) Hitung indikator & sinyal
    3) Set TP/SL berbasis ATR
    4) Evaluasi TP/SL (TP HIT / SL HIT / NO HIT)
    5) Simpan hasil per-pair (xlsx) + summary keseluruhan (xlsx)
    """
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    client: Client = get_client()
    interval = BINANCE_INTERVAL_MAP[timeframe]

    # clear folder
    clear_folder(data_dir)
    clear_folder(result_dir)

    for pair in pairs:
        # --- scrape data ---
        klines = client.futures_klines(symbol=pair, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
        df.set_index('timestamp', inplace=True)

        # simpan raw
        raw_path = os.path.join(data_dir, f"{pair}_{timeframe}.csv")
        df.to_csv(raw_path)

        # --- indikator ---
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        df['support'], df['resistance'] = calculate_support_resistance(df)
        df['volume_sma20'] = df['volume'].rolling(window=20).mean()

        bb = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['upper_band'] = bb.bollinger_hband()
        df['lower_band'] = bb.bollinger_lband()
        df['boll_width'] = df['upper_band'] - df['lower_band']
        df['bb_percentile'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])

        df['prev_high'] = df['high'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['prev_open'] = df['open'].shift(1)
        
        df = detect_sideways(df)
        # --- sinyal ---
        if df.iloc[-1]["sideways_signal"]:
            df['signal'] = "HOLD_SIDEWAY"
        else:
            df['signal'] = df.apply(detect_signal, axis=1)
            df = apply_filters(df)
            df['is_fake_breakout'] = df.apply(detect_breakout, axis=1)
            df = detect_potential_breakout(df)
           
       

        # df['is_potential_breakout'] = (
        #     (df['high'] > df['resistance']) |
        #     (df['low'] < df['support'])
        # )

        df['is_breakout_zone'] = df['is_fake_breakout']
        df['entry_type'] = None  # 'LONG', 'SHORT', atau 'CANCELLED'
        df['entry_signal'] = df['is_potential_breakout'].astype(int)
        
        for i in range(len(df)):
            if df.iloc[i]['is_breakout_zone']:
                trigger_long = df.iloc[i]['high'] + df.iloc[i]['atr'] * 0.2
                trigger_short = df.iloc[i]['low'] - df.iloc[i]['atr'] * 0.2

                future = df.iloc[i+1:i+3]  # 2 candle ke depan

                triggered = False
                for _, fut in future.iterrows():
                    if fut['high'] >= trigger_long:
                        df.at[df.index[i], 'entry_type'] = 'LONG'
                        triggered = True
                        break
                    elif fut['low'] <= trigger_short:
                        df.at[df.index[i], 'entry_type'] = 'SHORT'
                        triggered = True
                        break

                if not triggered:
                    df.at[df.index[i], 'entry_type'] = 'CANCELLED'

        df_export = df.reset_index().rename(columns={'index': 'timestamp'})
        full_all_path = os.path.join(result_dir, f"{pair}_{timeframe}_all_signals.xlsx")
        df_export.to_excel(full_all_path, index=False)
        print(f"📄 Full signals saved: {pair} {timeframe}")

        df = df[df['signal'].isin(['LONG', 'SHORT'])].copy()   

        if df.empty:
            # tidak ada sinyal sama sekali
            out_path = os.path.join(result_dir, f"hasil_backtest_{pair.lower()}_{timeframe}.xlsx")
            pd.DataFrame(columns=[
                'signal','entry_price','tp_price','sl_price','exit_status'
            ]).to_excel(out_path, index=False)
            return {
                "pair": pair,
                "timeframe": timeframe,
                "total": 0,
                "tp": 0,
                "sl": 0,
                "no_hit": 0,
                "tp_rate": 0.0,
                "result_path": out_path
            }

        # --- TP/SL ATR ---
        df['entry_price'] = df['close']
        df['tp_price'] = df['entry_price'] + df['atr'] * tp_atr_mult
        df['sl_price'] = df['entry_price'] - df['atr'] * sl_atr_mult
        df.loc[df['signal'] == 'SHORT', 'tp_price'] = df['entry_price'] - df['atr'] * tp_atr_mult
        df.loc[df['signal'] == 'SHORT', 'sl_price'] = df['entry_price'] + df['atr'] * sl_atr_mult
        
        # --- evaluasi ---
        df = evaluate_tp_sl(df, look_ahead=look_ahead)
        df['label'] = df['exit_status'].map({'TP HIT': 1, 'SL HIT': 0, 'NO HIT': -1})
        # --- simpan hasil pair ---
        out_path = os.path.join(result_dir, f"hasil_backtest_{pair.lower()}_{timeframe}.xlsx")
        df.to_excel(out_path)

        # --- summary single pair ---
        total = len(df)
        tp = (df['exit_status'] == 'TP HIT').sum()
        sl = (df['exit_status'] == 'SL HIT').sum()
        no_hit = (df['exit_status'] == 'NO HIT').sum()
        tp_rate = round(tp / total * 100, 2) if total > 0 else 0.0

        single_summary = {
            "pair": pair,
            "timeframe": timeframe,
            "total": total,
            "tp": tp,
            "sl": sl,
            "no_hit": no_hit,
            "tp_rate": tp_rate,
            "result_path": out_path
        }

        # --- (opsional) regenerate summary keseluruhan ---
        if save_summary:
            summaries = []
            for f in os.listdir(result_dir):
                if f.startswith("hasil_backtest_") and f.endswith(".xlsx"):
                    _pair_tf = f.replace("hasil_backtest_", "").replace(".xlsx", "")
                    parts = _pair_tf.split("_")
                    if len(parts) >= 2:
                        _pair = "_".join(parts[:-1]).upper()
                        _tf = parts[-1]
                    else:
                        _pair = _pair_tf.upper()
                        _tf = timeframe

                    _df = pd.read_excel(os.path.join(result_dir, f))
                    _total = len(_df)
                    _tp = (_df['exit_status'] == 'TP HIT').sum() if _total else 0
                    _sl = (_df['exit_status'] == 'SL HIT').sum() if _total else 0
                    _no = (_df['exit_status'] == 'NO HIT').sum() if _total else 0
                    _rate = round(_tp / _total * 100, 2) if _total else 0.0

                    summaries.append({
                        "Pair": _pair,
                        "Timeframe": _tf,
                        "Total Sinyal": _total,
                        "TP": _tp,
                        "SL": _sl,
                        "NO HIT": _no,
                        "TP Rate (%)": _rate
                    })

           
            if summaries:
                summary_df = pd.DataFrame(summaries)

                # Hitung persentase per kategori
                summary_df["TP (%)"] = (summary_df["TP"] / summary_df["Total Sinyal"] * 100).round(2)
                summary_df["SL (%)"] = (summary_df["SL"] / summary_df["Total Sinyal"] * 100).round(2)
                summary_df["NO HIT (%)"] = (summary_df["NO HIT"] / summary_df["Total Sinyal"] * 100).round(2)

                summary_df = summary_df.sort_values(by="TP Rate (%)", ascending=False)

                summary_df.to_excel(os.path.join(result_dir, "summary_backtest.xlsx"), index=False)

    # return single_summary
    # print(summaries)
    return summaries
