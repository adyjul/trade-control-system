import time
import os
from binance.um_futures import UMFutures
from binance.error import ClientError
from binance.websocket.um_futures.websocket_client import UMFuturesWebsocketClient
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

client = UMFutures(key=API_KEY, secret=API_SECRET)

symbol = "AVAXUSDT"
leverage = 20
quantity = 3

# simpan posisi saat ini
_current_position = None  # dict: {side, entry, tp, sl, order_id}

def place_order(side, entry_price, tp_price, sl_price):
    """Buka posisi dan simpan state"""
    global _current_position

    try:
        resp = client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"OPEN {side} @ {entry_price}")

        _current_position = {
            "side": side,
            "entry": entry_price,
            "tp": tp_price,
            "sl": sl_price,
            "order_id": resp["orderId"],
        }
    except ClientError as e:
        print("Order gagal:", e.error_message)

def close_position(reason):
    """Tutup posisi aktif"""
    global _current_position

    if _current_position is None:
        return

    side = "SELL" if _current_position["side"] == "BUY" else "BUY"
    try:
        resp = client.new_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"CLOSE {reason} @ market, entry={_current_position['entry']}, tp={_current_position['tp']}, sl={_current_position['sl']}")
    except ClientError as e:
        print("Close gagal:", e.error_message)

    _current_position = None

def on_message(msg):
    global _current_position

    if "p" not in msg:
        return
    price = float(msg["p"])

    # monitor posisi aktif
    if _current_position:
        if _current_position["side"] == "BUY":
            if price >= _current_position["tp"]:
                close_position("TP HIT")
            elif price <= _current_position["sl"]:
                close_position("SL HIT")

        elif _current_position["side"] == "SELL":
            if price <= _current_position["tp"]:
                close_position("TP HIT")
            elif price >= _current_position["sl"]:
                close_position("SL HIT")

    # contoh sinyal entry dummy (logic aslinya ganti pakai indikator kamu)
    else:
        if price % 2 < 0.01:  # kondisi entry dummy
            entry_price = price
            tp_price = entry_price + 0.05
            sl_price = entry_price - 0.05
            place_order("BUY", entry_price, tp_price, sl_price)


def main():
    # set leverage
    client.change_leverage(symbol=symbol, leverage=leverage)

    ws = UMFuturesWebsocketClient()
    ws.start()
    ws.symbol_book_ticker(symbol=symbol, id=1, callback=on_message)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ws.stop()


if __name__ == "__main__":
    main()
