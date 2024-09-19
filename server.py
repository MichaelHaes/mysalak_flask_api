from flask import Flask, jsonify, request
from arima import pred_result_arima
from deeplearning import pred_result_lstm
# from lalat import detect_lalat
from waitress import serve
from flask_cors import CORS

from yolov5test import test

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def hello():
  return 'Hello, World!'

@app.route('/arima', methods=['POST'])
def predict_tavg():
  return pred_result_arima()

@app.route('/lstm', methods=['POST'])
def predict_lstm():
  return pred_result_lstm()

# @app.route('/yolo', methods=['POST'])
# def predict_lalat():
#     return detect_lalat()
  
@app.route('/yolonew', methods=['POST'])
def testyolo():
    return test()
  
if __name__ == '__main__':
    # serve(app, host='0.0.0.0', port=8888, threads=1, url_scheme='https')
    app.run(port=8888)