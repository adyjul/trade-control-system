from typing import Dict
from binance.client import Client
from config import PAPER_TRADING, LEVERAGE_DEFAULT, POSITION_SIZE_USDT
from utils.db import log_trade

# --- Helper untuk qty ---
def calc_qty(entry_price: float, position_usdt: float) -> float:
    return round(position_usdt / entry_price, 6)  # 6 digits umum, sesuaikan dengan precision pair


def ensure_leverage(client: Client, symbol: str, leverage: int = LEVERAGE_DEFAULT):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print(f"⚠️ Gagal set leverage: {e}")


def execute_order(client: Client, bot_id: int, payload: Dict):
    """
    payload harus memuat:
    {
        'symbol', 'timeframe', 'signal', 'entry_price', 'tp_price', 'sl_price'
        'tp_percent', 'sl_percent'
    }
    """
    symbol = payload['symbol']
    side = 'BUY' if payload['signal'] == 'LONG' else 'SELL'
    quantity = calc_qty(payload['entry_price'], POSITION_SIZE_USDT)

    # Logging awal
    trade_id = log_trade({
        'bot_id': bot_id,
        'coin': symbol,
        'timeframe': payload['timeframe'],
        'side': payload['signal'],
        'entry_price': payload['entry_price'],
        'tp_price': payload['tp_price'],
        'sl_price': payload['sl_price'],
        'qty': quantity,
        'status': 'OPEN'
    })

    if PAPER_TRADING:
        print(f"[PAPER] {symbol} {side} qty={quantity} entry={payload['entry_price']} tp={payload['tp_price']} sl={payload['sl_price']}")
        return trade_id

    try:
        ensure_leverage(client, symbol)

        # 1) Entry market
        order = client.futures_create_order(
            symbol=symbol,
            side=side,
            type='MARKET',
            quantity=quantity
        )
        entry_order_id = order.get('orderId')

        # 2) TP (reduceOnly)
        tp_side = 'SELL' if side == 'BUY' else 'BUY'
        tp_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='TAKE_PROFIT_MARKET',
            stopPrice=str(payload['tp_price']),
            closePosition=True,
            reduceOnly=True
        )

        # 3) SL (reduceOnly)
        sl_order = client.futures_create_order(
            symbol=symbol,
            side=tp_side,
            type='STOP_MARKET',
            stopPrice=str(payload['sl_price']),
            closePosition=True,
            reduceOnly=True
        )

        # Update log
        from utils.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                UPDATE trade_log SET txid_entry=?, txid_tp=?, txid_sl=? WHERE id=?
                """,
                (str(entry_order_id), str(tp_order.get('orderId')), str(sl_order.get('orderId')), trade_id)
            )
            conn.commit()

        print(f"✅ Order executed: {symbol} trade_id={trade_id}")
    except Exception as e:
        from utils.db import get_conn
        with get_conn() as conn:
            conn.execute("UPDATE trade_log SET status='ERROR', error=? WHERE id=?", (str(e), trade_id))
            conn.commit()
        print(f"❌ Error execute_order: {e}")

    return trade_id