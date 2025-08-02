#!/usr/bin/env python3
import os
from flask import Flask, jsonify

# Simple Flask app for Railway
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/')
def home():
    return '<h1>PDF Content Moderator</h1>'

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port)