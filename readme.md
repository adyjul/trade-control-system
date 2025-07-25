# Trading Control Center – Flask + SQLite + Binance Futures

## 1) Setup
```bash
python -m venv venv
source venv/bin/activate  # atau venv\Scripts\activate di Windows
pip install -r requirements.txt
cp .env.example .env
python db/init_db.py
python app.py  # buka http://localhost:8000
```

## 2) Jalankan Bot via Cron (tiap 1 menit)
```bash
* * * * * /path/to/venv/bin/python /path/to/trading-control-center/run_bot.py >> /path/to/logs/bot.log 2>&1
```

## 3) Mode PAPER / LIVE
Atur `PAPER_TRADING=true` di .env untuk simulasi tanpa kirim order.

## 4) Next Steps
- Tambahkan **whitelist pair + filter volatilitas dinamis**
- Tambahkan **statistik winrate otomatis** per bot_id
- Tambahkan **result TP/SL tracking otomatis** (via websocket atau polling order status)
- Buat modul **regime detector** (sideways vs trending)
```

---

## ✅ Apa yang Sudah Siap?
- Web portal CRUD bot (coin, TF, TP/SL, dll) ✅
- DB SQLite dengan tabel `bot_settings` & `trade_log` ✅
- Bot runner yang baca DB dan eksekusi per jadwal timeframe ✅
- Auto-entry (PAPER/LIVE) dengan TP & SL reduceOnly ✅
- Cronjob friendly ✅

Kalau kamu setuju dengan struktur ini, tinggal kamu jalankan. Nanti kita lanjutkan dengan:
1. **Module filter real-time** (ATR/ADX/Volume, whitelist/hybrid) 
2. **Evaluator winrate otomatis** (TP>SL, 20 sinyal terakhir, dsb)
3. **Auto-switch pair/timeframe** berdasarkan performa terbaru