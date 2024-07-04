from flask import Flask, jsonify, request
from arima import tavg, information

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
  return 'Hello, World!'

@app.route('/info', methods=['GET'])
def info():
  return information()

@app.route('/arima/tavg', methods=['POST'])
def predict_tavg():
  return tavg()

if (__name__ == '__main__'):
  app.run()