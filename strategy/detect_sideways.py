# strategy/detect_sideways.py
import pandas as pd
import numpy as np
import ta

def detect_sideways(
    df: pd.DataFrame,
    lookback: int = 20,
    atr_multiplier: float = 1.5,
    bb_width_threshold: float = 0.02
) -> pd.DataFrame:
    """
    Deteksi kondisi sideways pada OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Data dengan kolom ['open','high','low','close','volume'].
    lookback : int
        Periode untuk Bollinger Bands & ATR.
    atr_multiplier : float
        Range high-low relatif kecil dibanding ATR.
    bb_width_threshold : float
        Ambang batas lebar Bollinger Band (dalam persentase harga).

    Returns
    -------
    pd.DataFrame
        DataFrame dengan kolom tambahan 'sideways_signal' (bool).
    """

    # ATR untuk volatilitas
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=lookback
    ).average_true_range()

    # Bollinger Band width
    bb_indicator = ta.volatility.BollingerBands(
        close=df["close"], window=lookback, window_dev=2
    )
    bb_width = (bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()) / df["close"]
    df["bb_width"] = bb_width

    # Range relatif kecil dibanding ATR
    rolling_high = df["high"].rolling(window=lookback).max()
    rolling_low = df["low"].rolling(window=lookback).min()
    range_size = (rolling_high - rolling_low)

    condition_atr = range_size < atr_multiplier * df["atr"]
    condition_bb = df["bb_width"] < bb_width_threshold

    df["sideways_signal"] = condition_atr & condition_bb

    return df
