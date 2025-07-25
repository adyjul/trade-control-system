from binance.client import Client
from config import BINANCE_API_KEY, BINANCE_API_SECRET, BINANCE_TESTNET

def get_client() -> Client:
    client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, testnet=BINANCE_TESTNET)
    return client