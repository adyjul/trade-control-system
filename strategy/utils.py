import os
import pickle
import pandas as pd
import numpy as np

# def calculate_support_resistance(df, window=3):
#     # versi 1
#     # support = []
#     # resistance = []
    
#     # for i in range(len(df)):
#     #     if i < window or i > len(df) - window - 1:
#     #         support.append(np.nan)
#     #         resistance.append(np.nan)
#     #     else:
#     #         lows = df['low'].iloc[i - window:i + window + 1]
#     #         highs = df['high'].iloc[i - window:i + window + 1]

#     #         support.append(lows.min())
#     #         resistance.append(highs.max())

#     # return pd.Series(support, index=df.index), pd.Series(resistance, index=df.index)
#     support = []
#     resistance = []

#     for i in range(len(df)):
#         # Tentukan batas bawah dan atas rolling window
#         start = max(0, i - window)
#         end = min(len(df), i + window + 1)

#         lows = df['low'].iloc[start:end]
#         highs = df['high'].iloc[start:end]

#         # Hitung min dan max dari window aktual
#         support.append(lows.min() if not lows.empty else np.nan)
#         resistance.append(highs.max() if not highs.empty else np.nan)

#     return pd.Series(support, index=df.index), pd.Series(resistance, index=df.index)

def calculate_support_resistance(df, window=3):
    support = []
    resistance = []

    for i in range(len(df)):
        # ambil hanya data masa lalu sampai candle saat ini
        start = max(0, i - window)
        end = i + 1  # tidak ambil candle di depan

        lows = df['low'].iloc[start:end]
        highs = df['high'].iloc[start:end]

        support.append(lows.min() if not lows.empty else np.nan)
        resistance.append(highs.max() if not highs.empty else np.nan)

    return pd.Series(support, index=df.index), pd.Series(resistance, index=df.index)



