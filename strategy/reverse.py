import pandas as pd

LOG_FILE = "/root/trade-control-system/trades_log.xlsx"   # ganti ke file log kamu

def load_trades():
    try:
        df = pd.read_csv(LOG_FILE)
    except:
        df = pd.read_excel(LOG_FILE)

    # ubah koma ke titik supaya bisa dihitung
    for col in ['entry_price', 'exit_price', 'pnl', 'fees', 'qty']:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    return df

def reverse_backtest(df: pd.DataFrame, fee_rate=0.0004):
    results = []

    for _, row in df.iterrows():
        side = row['side'].upper().strip()
        entry = row['entry_price']
        exit_ = row['exit_price']
        qty = row['qty']

        if side == "LONG":
            new_side = "SHORT"
            pnl_rev = (entry - exit_) * qty
        elif side == "SHORT":
            new_side = "LONG"
            pnl_rev = (exit_ - entry) * qty
        else:
            print("Lewat baris tanpa side:", side)
            continue

        # hitung fee
        pnl_rev -= (entry + exit_) * qty * fee_rate

        results.append({
            "entry_time": row['entry_time'],
            "original_side": side,
            "reversed_side": new_side,
            "entry_price": entry,
            "exit_price": exit_,
            "pnl_reversed": pnl_rev
        })

    return pd.DataFrame(results)

def main():
    df = load_trades()
    rev_df = reverse_backtest(df)

    if rev_df.empty:
        print("❌ Tidak ada trade yang bisa dibalik. Cek kolom 'side' di log.")
        return

    total_pnl = rev_df['pnl_reversed'].sum()
    win_trades = (rev_df['pnl_reversed'] > 0).sum()
    lose_trades = (rev_df['pnl_reversed'] <= 0).sum()

    print("=== Hasil Backtest Reverse ===")
    print(f"Total PnL: {total_pnl:.4f} USDT")
    print(f"Win trades: {win_trades} | Lose trades: {lose_trades}")
    print(f"Winrate: {win_trades / len(rev_df) * 100:.2f}%")

    rev_df.to_excel("reverse_results.xlsx", index=False)
    print("Hasil lengkap disimpan di reverse_results.xlsx")

if __name__ == "__main__":
    main()
