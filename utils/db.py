import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any
from config import DB_PATH

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def dict_rows(cur):
    return [dict(r) for r in cur.fetchall()]

# ---------------------- BOT SETTINGS ----------------------

def get_all_bots() -> List[Dict[str, Any]]:
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM bot_settings ORDER BY id DESC")
        return dict_rows(cur)

def get_active_bots():
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM bot_settings WHERE active=1")
        return dict_rows(cur)

def get_bot(bot_id: int):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM bot_settings WHERE id=?", (bot_id,))
        row = cur.fetchone()
        return dict(row) if row else None

def insert_bot(data: Dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO bot_settings (coin, timeframe, tp_percent, sl_percent, atr_multiplier, active, mode, note,atr_filter, best_pair)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data['coin'], data['timeframe'], data['tp_percent'], data['sl_percent'],
                data.get('atr_multiplier', 1.0), data.get('active', 1),
                data.get('mode', 'LIVE'), data.get('note'), data.get('atr_filter', 1), data.get('best_pair', 1)
            )
        )
        conn.commit()
        return cur.lastrowid

def update_bot(bot_id: int, data: Dict[str, Any]):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE bot_settings
            SET coin=?, timeframe=?, tp_percent=?, sl_percent=?, atr_multiplier=?, active=?, mode=?, note=?, atr_filter=?, best_pair=?, updated_at=CURRENT_TIMESTAMP
            WHERE id=?
            """,
            (
                data['coin'], data['timeframe'], data['tp_percent'], data['sl_percent'],
                data.get('atr_multiplier', 1.0), data.get('active', 1),
                data.get('mode', 'LIVE'), data.get('note'), data.get('atr_filter', 1), data.get('best_pair', 1), bot_id
            )
        )
        conn.commit()

def toggle_bot(bot_id: int, active: int):
    with get_conn() as conn:
        conn.execute("UPDATE bot_settings SET active=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", (active, bot_id))
        conn.commit()

# ---------------------- TRADE LOG ----------------------

def log_trade(data: Dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO trade_log (
                bot_id, coin, timeframe, side, entry_price, tp_price, sl_price, qty,
                status, txid_entry, txid_tp, txid_sl, result, pnl, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                data.get('bot_id'), data.get('coin'), data.get('timeframe'), data.get('side'),
                data.get('entry_price'), data.get('tp_price'), data.get('sl_price'), data.get('qty'),
                data.get('status', 'OPEN'), data.get('txid_entry'), data.get('txid_tp'), data.get('txid_sl'),
                data.get('result'), data.get('pnl'), data.get('error')
            )
        )
        conn.commit()
        return cur.lastrowid

def get_logs(limit=200):
    with get_conn() as conn:
        cur = conn.execute("SELECT * FROM trade_log ORDER BY id DESC LIMIT ?", (limit,))
        return dict_rows(cur)