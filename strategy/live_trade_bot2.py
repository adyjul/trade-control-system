import time
import asyncio
from binance.client import Client
from binance import BinanceSocketManager

API_KEY = "API_KEY"
API_SECRET = "API_SECRET"

client = Client(API_KEY, API_SECRET)

symbol = "AVAXUSDT"
leverage = 20
quantity = 3

_current_position = None  # dict: {side, entry, tp, sl}

# ---------------- Helper ----------------
def set_leverage():
    client.futures_change_leverage(symbol=symbol, leverage=leverage)
    print(f"[INFO] Leverage set to {leverage}")

def place_order(side, entry_price, tp_price, sl_price):
    global _current_position
    try:
        resp = client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"[OPEN] {side} @ {entry_price}")
        _current_position = {
            "side": side,
            "entry": entry_price,
            "tp": tp_price,
            "sl": sl_price,
            "order_id": resp['orderId']
        }
    except Exception as e:
        print("Order gagal:", e)

def close_position(reason):
    global _current_position
    if _current_position is None:
        return
    side = "SELL" if _current_position["side"] == "BUY" else "BUY"
    try:
        client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity
        )
        print(f"[CLOSE] {reason}, entry={_current_position['entry']}, tp={_current_position['tp']}, sl={_current_position['sl']}")
    except Exception as e:
        print("Close gagal:", e)
    _current_position = None

# ---------------- Bot logic ----------------
async def kline_listener():
    bm = BinanceSocketManager(client)
    async with bm.kline_socket(symbol, interval="1m") as stream:
        while True:
            res = await stream.recv()
            k = res.get('k', {})
            is_closed = k.get('x', False)
            close_price = float(k.get('c', 0))
            if not is_closed:
                continue

            # monitor posisi aktif
            global _current_position
            if _current_position:
                if _current_position["side"] == "BUY":
                    if close_price >= _current_position["tp"]:
                        close_position("TP HIT")
                    elif close_price <= _current_position["sl"]:
                        close_position("SL HIT")
                elif _current_position["side"] == "SELL":
                    if close_price <= _current_position["tp"]:
                        close_position("TP HIT")
                    elif close_price >= _current_position["sl"]:
                        close_position("SL HIT")
            # contoh entry dummy
            else:
                if close_price % 2 < 0.01:
                    entry_price = close_price
                    tp_price = entry_price + 0.05
                    sl_price = entry_price - 0.05
                    place_order("BUY", entry_price, tp_price, sl_price)

# ---------------- Run ----------------
if __name__ == "__main__":
    set_leverage()
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(kline_listener())
    except KeyboardInterrupt:
        print("[STOP] Bot dihentikan.")
