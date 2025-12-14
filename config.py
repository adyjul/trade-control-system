import os
from dotenv import load_dotenv

load_dotenv()

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
LEVERAGE_DEFAULT = int(os.getenv("LEVERAGE_DEFAULT", 10))
POSITION_SIZE_USDT = float(os.getenv("POSITION_SIZE_USDT", 50))
DB_PATH = os.getenv("DB_PATH", "/home/julbot/trade-control-system/db/bot.db")
PREDICT_DIR = os.getenv("PREDICT_DIR", "./predict_output")
DATA_DIR = os.getenv("DATA_DIR", "./data")
TIMEZONE_OFFSET_HOURS = int(os.getenv("TIMEZONE_OFFSET_HOURS", 7))
PAPER_TRADING = os.getenv("PAPER_TRADING", "true").lower() == "true"