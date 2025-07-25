from datetime import datetime, timezone
from utils.db import get_active_bots
from utils.timeframes import should_run_this_tf
from utils.binance_client import get_client
from strategy.predictor import generate_signal
from strategy.executor import execute_order


def run_once():
    now = datetime.now(timezone.utc)
    bots = get_active_bots()
    if not bots:
        print("No active bots.")
        return

    client = get_client()

    for bot in bots:
        try:
            if not should_run_this_tf(bot['timeframe'], now):
                continue

            res = generate_signal(
                client=client,
                symbol=bot['coin'],
                tf=bot['timeframe'],
                tp_percent=bot['tp_percent'],
                sl_percent=bot['sl_percent'],
                atr_multiplier=bot.get('atr_multiplier', 1.0),
                save_excel=True
            )
            if not res:
                print(f"📭 No signal for {bot['coin']} ({bot['timeframe']})")
                continue

            payload = {
                'symbol': res['symbol'],
                'timeframe': res['timeframe'],
                'signal': res['signal'],
                'entry_price': res['entry_price'],
                'tp_price': res['tp_price'],
                'sl_price': res['sl_price'],
                'tp_percent': bot['tp_percent'],
                'sl_percent': bot['sl_percent']
            }

            execute_order(client, bot_id=bot['id'], payload=payload)

        except Exception as e:
            print(f"❌ Error bot {bot['id']} {bot['coin']} -> {e}")