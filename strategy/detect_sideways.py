from coinmarketcap import CoinMarketCap
market = CoinMarketCap()
resp = market.stats(convert="BTC")

print(resp)[0]