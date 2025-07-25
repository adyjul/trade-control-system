import sqlite3
import os
# from config import DB_PATH

os.makedirs(os.path.dirname('db/bot.db'), exist_ok=True)

schema = r"""
CREATE TABLE IF NOT EXISTS bot_settings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin TEXT NOT NULL,
    timeframe TEXT NOT NULL,         -- e.g. 15m, 1h, 4h
    tp_percent REAL NOT NULL,
    sl_percent REAL NOT NULL,
    atr_multiplier REAL DEFAULT 1.0, -- keep for future
    active INTEGER DEFAULT 1,
    mode TEXT DEFAULT 'LIVE',        -- LIVE / PAPER
    note TEXT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bot_id INTEGER,
    coin TEXT,
    timeframe TEXT,
    side TEXT,                     -- LONG / SHORT
    entry_price REAL,
    tp_price REAL,
    sl_price REAL,
    qty REAL,
    status TEXT,                   -- OPEN / CLOSED / CANCELLED / ERROR
    txid_entry TEXT,
    txid_tp TEXT,
    txid_sl TEXT,
    result TEXT,                   -- TP / SL / MANUAL / UNKNOWN
    pnl REAL,
    error TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP
);
"""

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.executescript(schema)
conn.commit()
conn.close()

print(f"✅ DB initialized at {DB_PATH}")