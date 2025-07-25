from functools import wraps
from flask import Flask, render_template, request, Response, redirect, url_for, jsonify
from utils.db import get_all_bots, get_bot, insert_bot, update_bot, toggle_bot, get_logs
from db import init_db  # to ensure import side effect? we used separate file, so just run once manually
import os

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)