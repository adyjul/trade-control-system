# backtest_reverse.py
import pandas as pd

LOG_FILE = "/root/trade-control-system/trades_log.xlsx"  # sesuaikan dengan nama log botmu
SHEET = "Sheet1"             # atau ganti sesuai sheet log

def load_trades():
    # Baca log trade hasil live_trade_bot6
    df = pd.read_excel(LOG_FILE, sheet_name=SHEET)
    # Pastikan kolom berikut tersedia: 'side','entry_price','exit_price','qty','pnl','sl','tp','timestamp'
    return df

def reverse_backtest(df: pd.DataFrame, fee_rate=0.0004):
    reversed_results = []

    for _, row in df.iterrows():
        original_side = row['side'].lower()
        # Balik arah
        if original_side == 'long':
            new_side = 'short'
        elif original_side == 'short':
            new_side = 'long'
        else:
            continue

        entry = row['entry_price']
        exit_ = row['exit_price']
        qty = row['qty']

        # Hitung PnL seolah-olah dibalik
        if new_side == 'long':
            pnl = (exit_ - entry) * qty
        else:  # short
            pnl = (entry - exit_) * qty

        # Potong biaya (fee entry+exit)
        pnl -= (entry + exit_) * qty * fee_rate

        reversed_results.append({
            'timestamp': row['timestamp'],
            'original_side': original_side,
            'reversed_side': new_side,
            'entry': entry,
            'exit': exit_,
            'pnl_reversed': pnl
        })

    return pd.DataFrame(reversed_results)

def main():
    df = load_trades()
    rev_df = reverse_backtest(df)

    total_pnl = rev_df['pnl_reversed'].sum()
    win_trades = (rev_df['pnl_reversed'] > 0).sum()
    lose_trades = (rev_df['pnl_reversed'] <= 0).sum()

    print("=== Hasil Backtest Reverse ===")
    print(f"Total PnL: {total_pnl:.2f} USDT")
    print(f"Win trades: {win_trades} | Lose trades: {lose_trades}")
    print(f"Winrate: {win_trades / len(rev_df) * 100:.2f}%")
    
    rev_df.to_excel("reverse_results.xlsx", index=False)
    print("Hasil lengkap disimpan ke reverse_results.xlsx")

if __name__ == "__main__":
    main()
