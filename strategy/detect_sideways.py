from coinmarketcap import CoinMarketCap
market = CoinMarketCap()
resp = market.stats(convert="BTC")

print(resp)[0]["data"]["market_cap_usd"]