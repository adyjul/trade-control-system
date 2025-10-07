from coinmarketcap import CoinMarketCap
market = CoinMarketCap()
resp = market.coin_list()

print(resp)