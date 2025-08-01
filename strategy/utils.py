import pandas as pd
import numpy as np

def calculate_support_resistance(df, window=3):
    support = []
    resistance = []
    
    for i in range(len(df)):
        if i < window or i > len(df) - window - 1:
            support.append(np.nan)
            resistance.append(np.nan)
        else:
            lows = df['low'].iloc[i - window:i + window + 1]
            highs = df['high'].iloc[i - window:i + window + 1]

            support.append(lows.min())
            resistance.append(highs.max())

    return pd.Series(support, index=df.index), pd.Series(resistance, index=df.index)
