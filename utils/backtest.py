import pandas as pd

def evaluate_signals_in_file(filepath, look_ahead=6, tp_ratio=1.5, sl_ratio=1.0):
    df = pd.read_excel(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    df = df.dropna(subset=['signal', 'atr', 'close'])

    tp_count = 0
    sl_count = 0
    no_hit_count = 0
    total_signals = 0

    for i in range(len(df) - look_ahead):
        row = df.iloc[i]
        signal = row['signal']
        if signal not in ['LONG', 'SHORT']:
            continue

        entry_price = row['close']
        atr = row['atr']
        future = df.iloc[i+1:i+1+look_ahead]

        if signal == 'LONG':
            tp_level = entry_price + tp_ratio * atr
            sl_level = entry_price - sl_ratio * atr
            hit_tp = (future['high'] >= tp_level).any()
            hit_sl = (future['low'] <= sl_level).any()
        elif signal == 'SHORT':
            tp_level = entry_price - tp_ratio * atr
            sl_level = entry_price + sl_ratio * atr
            hit_tp = (future['low'] <= tp_level).any()
            hit_sl = (future['high'] >= sl_level).any()

        if hit_tp and not hit_sl:
            tp_count += 1
        elif hit_sl and not hit_tp:
            sl_count += 1
        elif hit_tp and hit_sl:
            # Hit dua-duanya, ambil yang kena lebih dulu
            tp_index = future[future['high'] >= tp_level].index[0] if signal == 'LONG' else future[future['low'] <= tp_level].index[0]
            sl_index = future[future['low'] <= sl_level].index[0] if signal == 'LONG' else future[future['high'] >= sl_level].index[0]
            if tp_index < sl_index:
                tp_count += 1
            else:
                sl_count += 1
        else:
            no_hit_count += 1

        total_signals += 1

    if total_signals == 0:
        return {
            'total_signal': 0,
            'tp': 0,
            'sl': 0,
            'no_hit': 0,
            'tp_rate': 0.0,
            'sl_rate': 0.0,
            'no_hit_rate': 0.0
        }

    return {
        'total_signal': total_signals,
        'tp': tp_count,
        'sl': sl_count,
        'no_hit': no_hit_count,
        'tp_rate': round(100 * tp_count / total_signals, 2),
        'sl_rate': round(100 * sl_count / total_signals, 2),
        'no_hit_rate': round(100 * no_hit_count / total_signals, 2)
    }
