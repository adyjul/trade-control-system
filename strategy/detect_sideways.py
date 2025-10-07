from coinmarketcap import CoinMarketCap
market = CoinMarketCap()
resp = market.stats(convert="BTC")
{
	'data': {
		'active_cryptocurrencies': 1620,
		'active_markets': 11598,
		'bitcoin_percentage_of_market_cap': 42.11,
		'quotes': {
			'USD': {
				'total_market_cap': 277757148004.0,
				'total_volume_24h': 12371915084.0
			},
			'BTC': {
				'total_market_cap': 40697435.0,
				'total_volume_24h': 1812753.0
			}
		},
		'last_updated': 1531038676
	},
	'metadata': {
		'timestamp': 1531038328,
		'error': None
	},
	'cached': False
}