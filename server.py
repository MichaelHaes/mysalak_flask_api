from flask import Flask, jsonify, request
from arima import pred_result, information, tavg

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

if (__name__ == '__main__'):
  app.run()