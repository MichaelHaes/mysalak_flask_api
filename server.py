from flask import Flask, jsonify, request
from arima import pred_result, information
from deeplearning import tavg

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
  return 'Hello, World!'

@app.route('/info', methods=['GET'])
def info():
  return info()

@app.route('/arima', methods=['POST'])
def predict_tavg():
  return pred_result()

@app.route('/lstm', methods=['POST'])
def predict_lstm():
  return tavg()

if (__name__ == '__main__'):
  app.run()