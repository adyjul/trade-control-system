from datetime import datetime

def should_run_this_tf(tf: str, now: datetime) -> bool:
    tf = tf.lower()
    if tf == '1m':
        return True
    if tf == '5m':
        return now.minute % 5 == 0
    if tf == '15m':
        return now.minute % 15 == 0
    if tf == '1h':
        return now.minute == 0
    if tf == '4h':
        return now.minute == 0 and now.hour % 4 == 0
    if tf == '1d':
        return now.minute == 0 and now.hour == 0
    return False

BINANCE_INTERVAL_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}
