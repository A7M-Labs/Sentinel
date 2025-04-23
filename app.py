from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Mock data for demonstration - in reality, this would come from your partner's API
mock_events = [
    {"id": 1, "type": "Unauthorized Entry", "timestamp": "2023-10-01 08:45:32", "confidence": 0.89, "camera": "Front Door"},
    {"id": 2, "type": "Unattended Object", "timestamp": "2023-10-01 09:12:05", "confidence": 0.76, "camera": "Lobby"},
    {"id": 3, "type": "Suspicious Behavior", "timestamp": "2023-10-01 10:30:18", "confidence": 0.92, "camera": "Parking Lot"}
]

@app.context_processor
def inject_now():
    return {'now': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html', events=mock_events)

@app.route('/api/events')
def get_events():
    # This would interface with your partner's backend API
    return jsonify(mock_events)

@app.route('/analytics')
def analytics():
    # This route will redirect to the Streamlit app for analytics
    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True) 