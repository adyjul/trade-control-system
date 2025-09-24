def calc_profit_percent(entry_price: float, side: str, latest_price: float) -> float:
    """
    Hitung profit % posisi berdasarkan entry dan harga terakhir.
    - entry_price: harga entry posisi
    - side: "LONG" atau "SHORT"
    - latest_price: harga terakhir (close / bid / ask sesuai kebutuhan)

    Return: profit dalam bentuk desimal (contoh 0.05 = +5%, -0.03 = -3%)
    """
    if side.upper() == "LONG":
        return (latest_price - entry_price) / entry_price
    elif side.upper() == "SHORT":
        return (entry_price - latest_price) / entry_price
    else:
        raise ValueError("side harus 'LONG' atau 'SHORT'")
    

entry = 34.664


# Long contoh
print(calc_profit_percent(entry, "LONG", 35.15))   # +0.05 (5%)
print(calc_profit_percent(entry, "LONG",  34.072))   # -0.05 (-5%)
