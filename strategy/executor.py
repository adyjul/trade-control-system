# executor.py versi baru dengan log ke database
import os
import math
import pandas as pd
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from binance.client import Client

from utils.db import log_trade
from utils.binance_client import get_client

load_dotenv()

client: Client = get_client()

PREDICT_DIR = "predict_output"
MODAL = 20.0
ENTRY_PERCENT = 2.0
LEVERAGE = 25
TP_MULTIPLIER = 1.2
SL_MULTIPLIER = 1.0


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


def should_entry(symbol, new_atr, min_atr_threshold=0.1):
    positions = client.futures_position_information(symbol=symbol)
    pos = next((p for p in positions if float(p['positionAmt']) != 0), None)
    if pos:
        print(f"🚫 Sudah ada posisi terbuka di {symbol}. Skip.")
        return False
    if new_atr < min_atr_threshold:
        print(f"⚠️ ATR terlalu kecil ({new_atr:.4f}). Skip.")
        return False
    return True


def cancel_open_orders(symbol):
    try:
        open_orders = client.futures_get_open_orders(symbol=symbol)
        for order in open_orders:
            if order['type'] in ['STOP_MARKET', 'TAKE_PROFIT_MARKET', 'TRAILING_STOP_MARKET']:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                print(f"🔁 Cancel order {order['type']} | ID: {order['orderId']}")
    except Exception as e:
        print(f"⚠️ Gagal cancel orders: {e}")


def run_executor():
    best_signals = []
    for file in os.listdir(PREDICT_DIR):
        if not file.endswith(".xlsx"):
            continue
        path = os.path.join(PREDICT_DIR, file)
        df = pd.read_excel(path)
        df = df[df['signal'].isin(['LONG', 'SHORT'])]
        if df.empty:
            continue
        last = df.iloc[-1]
        if not pd.isna(last['atr']):
            best_signals.append((last['atr'], last, file))

    best_signals = sorted(best_signals, key=lambda x: x[0], reverse=True)[:3]
    if not best_signals:
        print("⛔ Tidak ada sinyal valid.")
        return

    now = datetime.now(timezone.utc)
    expected_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)

    for _, row, file in best_signals:
        pair = file.replace("prediksi_entry_logic_", "").replace(".xlsx", "")
        signal = row['signal']
        price = row['entry_price']
        atr = row['atr']
        ts_utc = pd.to_datetime(row['timestamp_utc']).tz_localize('UTC')
        ts_wib = pd.to_datetime(row['timestamp_wib'])

        if ts_utc != expected_time:
            print(f"⚠️ Sinyal bukan dari candle terakhir ({ts_utc} ≠ {expected_time})")
            continue

        if not should_entry(pair, atr):
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

        tp = round(price + atr * TP_MULTIPLIER, price_precision) if signal == 'LONG' else round(price - atr * TP_MULTIPLIER, price_precision)
        sl = round(price - atr * SL_MULTIPLIER, price_precision) if signal == 'LONG' else round(price + atr * SL_MULTIPLIER, price_precision)

        try:
            order = client.futures_create_order(
                symbol=pair,
                side='BUY' if signal == 'LONG' else 'SELL',
                type='MARKET',
                quantity=qty
            )
        except Exception as e:
            print(f"🚫 Gagal entry: {e}")
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

        # log_trade({
        #     "symbol": pair,
        #     "signal": signal,
        #     "entry_time_utc": datetime.utcnow(),
        #     "entry_time_wib": datetime.utcnow() + timedelta(hours=7),
        #     "signal_time_utc": ts_utc.replace(tzinfo=None),
        #     "signal_time_wib": ts_wib.replace(tzinfo=None),
        #     "entry_price": price,
        #     "tp_price": tp,
        #     "sl_price": sl,
        #     "qty": qty,
        #     "order_id": order['orderId']
        # })

        log_trade({
            "bot_id": 1,  # sesuaikan dengan ID bot di DB
            "coin": pair,
            "timeframe": "1h",  # atau "4h", sesuaikan sesuai scraping
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

        print(f"✅ Entry {pair} | {signal} @ {price:.4f}")

if __name__ == '__main__':
    run_executor()
