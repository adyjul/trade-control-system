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

def is_time_to_run(tf: str, now: datetime) -> bool:
    tf_minutes = {
        "1m": 1, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440
    }[tf.lower()]
    minute_of_day = now.hour * 60 + now.minute
    return (minute_of_day % tf_minutes) == 0

BINANCE_INTERVAL_MAP = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}
