def clamp_tp_sl(entry_price: float, tp_price: float, sl_price: float,
                side: str,
                tp_min=0.04, tp_max=0.08,
                sl_min=0.06, sl_max=0.10):
    """
    Clamp TP/SL berbasis harga untuk long & short.
    side: "LONG" atau "SHORT"
    Semua tp_min/sl_min dalam format 0.xx (contoh 0.08 = 8%).
    """

    if side.upper() == "LONG":
        # hitung % perubahan
        tp_percent = (tp_price - entry_price) / entry_price
        sl_percent = (entry_price - sl_price) / entry_price

        # clamp
        tp_percent = max(tp_min, min(tp_percent, tp_max))
        sl_percent = max(sl_min, min(sl_percent, sl_max))

        # konversi balik ke harga
        final_tp = entry_price * (1 + tp_percent)
        final_sl = entry_price * (1 - sl_percent)

    elif side.upper() == "SHORT":
        tp_percent = (entry_price - tp_price) / entry_price
        sl_percent = (sl_price - entry_price) / entry_price

        tp_percent = max(tp_min, min(tp_percent, tp_max))
        sl_percent = max(sl_min, min(sl_percent, sl_max))

        final_tp = entry_price * (1 - tp_percent)
        final_sl = entry_price * (1 + sl_percent)

    else:
        raise ValueError("side harus 'LONG' atau 'SHORT'")

    return final_tp, final_sl


# contoh pemakaian
entry = 34.844

# LONG contoh (raw 18% tp, 20% sl)
tp, sl = clamp_tp_sl(entry, 34.302, 33.045, side="LONG")
print(f"LONG → TP: {tp}, SL: {sl}")
# hasil → TP: 108.0, SL: 90.0

# SHORT contoh (raw 18% tp, 20% sl)
tp, sl = clamp_tp_sl(entry, 82, 120, side="SHORT")
print(f"SHORT → TP: {tp}, SL: {sl}")
# hasil → TP: 92.0, SL: 110.0
