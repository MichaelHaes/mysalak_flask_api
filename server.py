from flask import Flask, jsonify, request
from arima import pred_result_arima
from deeplearning import pred_result_lstm
from waitress import serve

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
  return 'Hello, World!'

@app.route('/arima', methods=['POST'])
def predict_tavg():
  return pred_result_arima()

@app.route('/lstm', methods=['POST'])
def predict_lstm():
  return pred_result_lstm()

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8888)
    serve(app, host='0.0.0.0', port=5000, url_scheme='https')
