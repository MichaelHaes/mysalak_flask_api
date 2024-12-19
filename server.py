from flask import Flask, jsonify, request
# from arima import pred_result_arima
from lgbm import pred_result_lightgbm
from xgbm import pred_result_xgboost
# from deeplearning import pred_result_lstm
# from lalat import detect_lalat
from waitress import serve
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def hello():
  return 'Hello, World!'

# @app.route('/arima', methods=['POST'])
# def predict_tavg():
#   return pred_result_arima()

# @app.route('/lstm', methods=['POST'])
# def predict_lstm():
#   return pred_result_lstm()

@app.route('/lightgbm', methods=['POST'])
def predict_lightgbm():
  return pred_result_lightgbm()

@app.route('/xgboost', methods=['POST'])
def predict_xgboost():
  return pred_result_xgboost()

# @app.route('/yolo', methods=['POST'])
# def predict_lalat():
#     return detect_lalat()

if __name__ == '__main__':
    # serve(app, host='127.0.0.1', port=5000)
    # serve(app, host='127.0.0.1', port=5000, threads=1, url_scheme='https')
    app.run(port=8888)