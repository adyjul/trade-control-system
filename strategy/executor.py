import os
import math
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from binance.client import Client
import joblib
from utils.db import log_trade, get_active_bots
from utils.binance_client import get_client
from utils.timeframes import BINANCE_INTERVAL_MAP
from utils.timeframes import get_expected_time
from strategy.ml.utils import predict_ml_signal
# from strategy.ml_predict import is_signal_valid

load_dotenv()
client: Client = get_client()

PREDICT_DIR = "data_predict"
LOG_PATH = "log_entry_switching.xlsx"
MODAL = 20.0
ENTRY_PERCENT = 1.1
LEVERAGE = 25


def get_precision(symbol):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step = float(f['stepSize'])
                    qty_precision = abs(int(round(math.log10(step))))
                if f['filterType'] == 'PRICE_FILTER':
                    tick = float(f['tickSize'])
                    price_precision = abs(int(round(math.log10(tick))))
            return qty_precision, price_precision
    return 3, 2


def should_entry(symbol, atr, min_atr=0.1,filter_atr=1):
   
    positions = client.futures_position_information(symbol=symbol)
    pos = next((p for p in positions if float(p['positionAmt']) != 0), None)
    if pos:
        print(f"🚫 {symbol} sudah punya posisi terbuka.")
        return False
    
    if filter_atr == 1:
        if atr < min_atr:
            print(f"⚠️ ATR terlalu kecil ({atr:.4f})")
            return False
    return True


def cancel_open_orders(symbol):
    try:
        orders = client.futures_get_open_orders(symbol=symbol)
        for o in orders:
            if o['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET']:
                client.futures_cancel_order(symbol=symbol, orderId=o['orderId'])
    except Exception as e:
        print(f"⚠️ Gagal cancel order: {e}")

def get_tick_size(symbol):
    info = client.futures_exchange_info()
    for s in info['symbols']:
        if s['symbol'] == symbol:
            for f in s['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    return float(f['tickSize'])
    return 0.0001

def run_executor():
    if os.path.exists(LOG_PATH):
        old_log = pd.read_excel(LOG_PATH)
        old_log['signal_time_utc'] = pd.to_datetime(old_log['signal_time_utc'])
    else:
        old_log = pd.DataFrame(columns=['signal_time_utc'])

    bots = get_active_bots()
    now = datetime.now(timezone.utc)
    
    expected_time = now.replace(minute=0, second=0, microsecond=0)
    # expected_time = get_expected_time(bots[0]['timeframe'], now)

    for bot in bots:
        pairs = [p.strip() for p in bot['coin'].split(',')]
        tf = bot['timeframe']
        tf_suffix = f"_{tf}"
        tp_mult = bot.get('tp_percent', 1.2)
        sl_mult = bot.get('sl_percent', 1.0)

        # expected_time = get_expected_time(tf)
        # print(expected_time)

        signals = []
        # print(pairs)
        for pair in pairs:
            # fname = f"{pair}_{tf}_full.xlsx"
            fname = f"prediksi_entry_logic_{pair}.xlsx"
            fpath = os.path.join(PREDICT_DIR, fname)

            if not os.path.exists(fpath):
                print(f"{pair}⚠️ Tidak ada file sinyal: {fpath}")
                continue

           

            df = pd.read_excel(fpath)
            df = df[df['signal'].isin(['LONG', 'SHORT'])]
            if df.empty:
                print(f"⚠️ tidak ada sinyal long/short")
                continue

            row = df.iloc[-1]
            if pd.isna(row['atr']):
                print("tidak kolom atr")
                continue
            
            # ts_utc = pd.to_datetime(row['timestamp_utc']).tz_convert('UTC')
            ts_utc = pd.to_datetime(row['timestamp_utc'])
            if ts_utc.tzinfo is None:
                ts_utc = ts_utc.tz_localize('UTC')
            else:
                ts_utc = ts_utc.tz_convert('UTC')
            # print(expected_time, ts_utc)
            # if ts_utc.replace(minute=0, second=0, microsecond=0) != expected_time:
            #     print(f"{pair}⚠️ Waktu sinyal tidak sesuai: {ts_utc}")
            #     continue

            if tf == '1h':
                valid_time = expected_time - timedelta(hours=1)
            elif tf == '4h':
                valid_time = expected_time - timedelta(hours=4)

            if ts_utc.replace(minute=0, second=0, microsecond=0) != valid_time:
                print(f"{pair}⚠️ Waktu sinyal tidak sesuai: {ts_utc} (diharapkan {valid_time})")
                continue

            signals.append((row['atr'], row, pair))
        
        if not signals:
            print(f"⛔ Tidak ada sinyal valid untuk bot id={bot['id']}")
            continue

        # Gunakan best pair (ATR tertinggi) jika diaktifkan
        top_n = int(bot.get('best_pair') or 0)
        if top_n > 0:
            signals = sorted(signals, key=lambda x: x[0], reverse=True)[:top_n]

        for atr, row, pair in signals:
            signal = row['signal']
            price = row['entry_price']

            model_path = f"/root/trade-control-system/strategy/ml/models/breakout_rf_model_{pair}_{tf}.pkl"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                print(f"[ML] Model ditemukan: {model_path}")
            else:
                print(f"[ML] Tidak ada model untuk {pair} {tf}, lanjut tanpa filter.")

            # Apply ML filter
            if not predict_ml_signal(model, row):
                print(f"[ML FILTER] Sinyal {pair} {tf} dibatalkan oleh model ML.")
                return  # atau skip entry

            print(bot.get('filter_atr',0))
            if not should_entry(pair, atr,0.1,bot.get('filter_atr',0)):
                print('atr terlalu rendah')
                continue

            # Buat kolom identitas sinyal unik di log lama
            old_log['signal_id'] = old_log['symbol'] + "_" + old_log['signal_time_utc'].astype(str)

            # Buat ID sinyal saat ini
            current_signal_id = f"{pair}_{str(ts_utc)}"

            if current_signal_id in old_log['signal_id'].values:
                print(f"🚫 Sinyal {pair} pada {ts_utc} sudah dieksekusi. Skip.")
                continue

            cancel_open_orders(pair)

            try:
                client.futures_change_leverage(symbol=pair, leverage=LEVERAGE)
            except Exception as e:
                print(f"⚠️ Gagal set leverage: {e}")

            qty_precision, price_precision = get_precision(pair)
            entry_usd = MODAL * ENTRY_PERCENT
            qty = round(entry_usd / price, qty_precision)
            if qty <= 0:
                continue

            tp = round(price + atr * tp_mult, price_precision) if signal == 'LONG' else round(price - atr * tp_mult, price_precision)
            sl = round(price - atr * sl_mult, price_precision) if signal == 'LONG' else round(price + atr * sl_mult, price_precision)

            # tick_size = get_tick_size(pair)
            # sl_safe = sl + tick_size * 5 if signal == 'LONG' else sl - tick_size * 5

            try:
                order = client.futures_create_order(
                    symbol=pair,
                    side='BUY' if signal == 'LONG' else 'SELL',
                    type='MARKET',
                    quantity=qty
                )
            except Exception as e:
                print(f"❌ Gagal entry: {e}")
                continue

            for t, p in [('TAKE_PROFIT_MARKET', tp), ('STOP_MARKET', sl)]:
                try:
                    client.futures_create_order(
                        symbol=pair,
                        side='SELL' if signal == 'LONG' else 'BUY',
                        type=t,
                        stopPrice=p,
                        closePosition=True,
                        timeInForce='GTE_GTC'
                    )
                except Exception as e:
                    print(f"⚠️ Gagal pasang {t}: {e}")

            try:
                activation = round(price + atr * 0.5, price_precision) if signal == 'LONG' else round(price - atr * 0.5, price_precision)
                callback = max(0.2, min((atr * 100 / price), 1.0))
                client.futures_create_order(
                    symbol=pair,
                    side='SELL' if signal == 'LONG' else 'BUY',
                    type='TRAILING_STOP_MARKET',
                    activationPrice=activation,
                    callbackRate=callback,
                    quantity=qty,
                    reduceOnly=True
                )
            except Exception as e:
                print(f"⚠️ Gagal pasang trailing stop: {e}")

            log_trade({
                "bot_id": bot['id'],
                "coin": pair,
                "timeframe": tf,
                "side": signal,
                "entry_price": price,
                "tp_price": tp,
                "sl_price": sl,
                "qty": qty,
                "status": "OPEN",
                "txid_entry": order['orderId'],
                "txid_tp": None,
                "txid_sl": None,
                "result": None,
                "pnl": None,
                "error": None
            })

            
             # Simpan ke log Excel untuk anti duplikasi
            log_row = {
                "entry_time_utc": datetime.utcnow(),
                "entry_time_wib": datetime.utcnow() + timedelta(hours=7),
                "signal_time_utc": ts_utc.replace(tzinfo=None),
                # "signal_time_wib": ts_wib.replace(tzinfo=None),
                "signal": signal,
                "symbol": pair,
                "entry_price": price,
                "qty": qty,
                "tp_price": tp,
                "sl_price": sl,
                "order_id": order['orderId']
            }
            log_df = pd.DataFrame([log_row])
            if os.path.exists(LOG_PATH):
                log_df = pd.concat([old_log, log_df], ignore_index=True)
            log_df.to_excel(LOG_PATH, index=False)

            print(f"✅ Entry {pair} {signal} @ {price:.4f} | TP: {tp}, SL: {sl}")


if __name__ == "__main__":
    run_executor()
