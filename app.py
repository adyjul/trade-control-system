from functools import wraps
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify, render_template, url_for
from utils.db import get_all_bots, get_bot, insert_bot, update_bot, toggle_bot, get_logs
from db import init_db  # to ensure import side effect? we used separate file, so just run once manually
import os
import pandas as pd
from strategy.backtest import run_full_backtest
import shutil

USERNAME = os.getenv("WEB_USERNAME", "admin")
PASSWORD = os.getenv("WEB_PASSWORD", "secret")

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response(
        'Login required', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

app = Flask(__name__)

@app.route('/')
@requires_auth
def index():
    bots = get_all_bots()
    return render_template('index.html', bots=bots)

@app.route('/logs')
@requires_auth
def logs():
    logs = get_logs(limit=300)
    return render_template('logs.html', logs=logs)

@app.route('/bots/new', methods=['GET', 'POST'])
@requires_auth
def new_bot():
    if request.method == 'GET':
        return render_template('form.html', bot=None)
    if request.method == 'POST':
        data = {
            'coin': request.form['coin'].upper(),
            'timeframe': request.form['timeframe'],
            'tp_percent': float(request.form['tp_percent']),
            'sl_percent': float(request.form['sl_percent']),
            'atr_multiplier': float(request.form.get('atr_multiplier', 1.0)),
            'active': 1 if request.form.get('active') == 'on' else 0,
            'mode': request.form.get('mode', 'LIVE'),
            'note': request.form.get('note', '')
        }
        insert_bot(data)
        return redirect(url_for('index'))
    

@app.route('/bots/<int:bot_id>/edit', methods=['GET', 'POST'])
@requires_auth
def edit_bot(bot_id):
    bot = get_bot(bot_id)
    if not bot:
        return "Not found", 404
    if request.method == 'POST':
        data = {
            'coin': request.form['coin'].upper(),
            'timeframe': request.form['timeframe'],
            'tp_percent': float(request.form['tp_percent']),
            'sl_percent': float(request.form['sl_percent']),
            'atr_multiplier': float(request.form.get('atr_multiplier', 1.0)),
            'active': 1 if request.form.get('active') == 'on' else 0,
            'mode': request.form.get('mode', 'LIVE'),
            'note': request.form.get('note', '')
        }
        update_bot(bot_id, data)
        return redirect(url_for('index'))
    return render_template('form.html', bot=bot)

@app.route('/bots/<int:bot_id>/toggle', methods=['POST'])
@requires_auth
def toggle(bot_id):
    bot = get_bot(bot_id)
    if not bot:
        return "Not found", 404
    toggle_bot(bot_id, 0 if bot['active'] else 1)
    return redirect(url_for('index'))

@app.route('/api/active_bots')
@requires_auth
def api_active_bots():
    from utils.db import get_active_bots
    return jsonify(get_active_bots())

# @app.route("/backtest-summary")
# def backtest_summary():
#     result_folder = "/root/trade-control-system/backtest_result_test/"
#     summary_file = os.path.join(result_folder, "summary_backtest.xlsx")

#     if not os.path.exists(summary_file):
#         return "⚠️ Summary belum tersedia. Silakan jalankan backtest dulu."

#     df = pd.read_excel(summary_file)
#     summary = df.to_dict(orient="records")
#     return render_template("summary.html", summary=summary)

@app.route("/backtest/new", methods=["GET", "POST"])
def backtest_new():
    if request.method == "POST":
        raw_data = request.form['pair']  # "ada,dada"
        pairs = raw_data.split(',')  # ['ada', 'dada']
        # pair = request.form["pair"].upper()
        tf = request.form["timeframe"]
        limit = request.form["limit"]
        res = run_full_backtest(pairs, tf, limit)
        # shutil.copy(result_path, f"static/backtest_result/{pair.lower()}_{timeframse}.xlsx")
        # bisa diarahkan ke halaman summary / tampilkan hasil single pair
        # return render_template("summary_backtest.html", result=res)
        return redirect(url_for('backtest/summary'))

    return render_template("backtest_form.html")  # form input pair/tf


@app.route('/backtest/summary')
def backtest_summary():
    df = pd.read_excel("/root/trade-control-system/backtest_result/summary_backtest.xlsx")
    summary = df.to_dict(orient='records')
    return render_template("summary_backtest.html", summary=summary)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)