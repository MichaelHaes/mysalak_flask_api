from flask import jsonify, request
import tensorflow as tf
import os, pandas as pd
import numpy as np

base_lstm_path = os.path.join(os.path.dirname(__file__), 'Model', 'DL')

tavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Tavg_LSTM.keras'))
# rhavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RH_avg_LSTM.h5'))
# precipitation_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RR_LSTM.h5'))
# luminosity_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Luminosity_LSTM.h5'))

minRH_avg = 0
maxRH_avg = 100
minTavg = 2
maxTavg = 33
minRR = 0
maxRR = 92
minLumen = 0
maxLumen = 2000

def normalize(value, minV, maxV): return (value - minV) / (maxV - minV)
def denormalize(value, minV, maxV): return value * (maxV - minV) + minV

def tavg():
  try:
    new_val = request.json
    
    inp = normalize(new_val['tavg'], minTavg, maxTavg)
    temp = tf.reshape(inp, [1, 1, 1])
    result = denormalize(tavg_model.predict(temp), minTavg, maxTavg).tolist()
    return jsonify(result)
  except Exception as e:
    return str(e), 500
