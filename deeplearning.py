from flask import jsonify, request
import tensorflow as tf
import os, pandas as pd
import numpy as np

base_lstm_path = os.path.join(os.path.dirname(__file__), 'Model', 'DL')

tavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Tavg_LSTM.keras'))
rhavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RH_avg_LSTM.keras'))
precipitation_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RR_LSTM.keras'))
luminosity_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Luminosity_LSTM.keras'))

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

def pred_result_lstm():
  res = {
      'Tavg': tavg(),
      'RH_avg': rh_avg(),
      'RR': rr(),
      'Lumen': lumen()
    }
  return res

def tavg():
  try:
    new_val = request.json
    
    inp = normalize(new_val['tavg'], minTavg, maxTavg)
    temp = tf.reshape(inp, [1, 1, 1])
    result = denormalize(tavg_model.predict(temp), minTavg, maxTavg).tolist()
    return result
  except Exception as e:
    return str(e), 500

def rh_avg():
  try:
    new_val = request.json
    
    inp = normalize(new_val['rh_avg'], minRH_avg, maxRH_avg)
    humid = tf.reshape(inp, [1, 1, 1])
    result = denormalize(rhavg_model.predict(humid), minRH_avg, maxRH_avg).tolist()
    return result
  except Exception as e:
    return str(e), 500
  
def rr():
  try:
    new_val = request.json
    
    inp = normalize(new_val['rr'], minRR, maxRR)
    rr = tf.reshape(inp, [1, 1, 1])
    result = denormalize(precipitation_model.predict(rr), minRR, maxRR).tolist()
    return result
  except Exception as e:
    return str(e), 500
  
def lumen():
  try:
    new_val = request.json
    
    inp = normalize(new_val['lumen'], minLumen, maxLumen)
    lumen = tf.reshape(inp, [1, 1, 1])
    result = denormalize(luminosity_model.predict(lumen), minLumen, maxLumen).tolist()
    return result
  except Exception as e:
    return str(e), 500