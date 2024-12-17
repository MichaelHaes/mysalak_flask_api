# from flask import jsonify, request
# import tensorflow as tf
# import os, pandas as pd
# import numpy as np

# base_lstm_path = os.path.join(os.path.dirname(__file__), 'Model', 'DL')

# tavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Tavg_LSTM.keras'))
# rhavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RH_avg_LSTM.keras'))
# precipitation_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RR_LSTM.keras'))
# luminosity_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Luminosity_LSTM.keras'))

# minRH_avg = 0
# maxRH_avg = 100
# minTavg = 2
# maxTavg = 33
# minRR = 0
# maxRR = 92
# minLumen = 0
# maxLumen = 2000

# def normalize(value, minV, maxV): return (value - minV) / (maxV - minV)
# def denormalize(value, minV, maxV): return value * (maxV - minV) + minV

# def pred_result_lstm():
#   res = {
#       'Tavg': tavg(),
#       'RH_avg': rh_avg(),
#       'RR': rr(),
#       'Lumen': lumen()
#     }
#   return res

# def tavg():
#   try:
#     new_val = request.json
#     prediction = []
    
#     inp = normalize(new_val['tavg'], minTavg, maxTavg)
#     temp = tf.reshape(inp, [1, 1, 1])
    
#     for i in range(7):
#       result = tavg_model.predict(temp)
#       prediction.append(denormalize(result, minTavg, maxTavg).item())
#       temp = np.reshape(result, (1, 1, 1))
    
#     return prediction
#   except Exception as e:
#     return str(e), 500

# def rh_avg():
#   try:
#     new_val = request.json
#     prediction = []
    
#     inp = normalize(new_val['rh_avg'], minRH_avg, maxRH_avg)
#     humid = tf.reshape(inp, [1, 1, 1])
    
#     for i in range(7):
#       result = rhavg_model.predict(humid)
#       prediction.append(denormalize(result, minRH_avg, maxRH_avg).item())
#       humid = np.reshape(result, (1, 1, 1))
    
#     return prediction
#   except Exception as e:
#     return str(e), 500
  
# def rr():
#   try:
#     new_val = request.json
#     prediction = []
    
#     inp = normalize(new_val['rr'], minRR, maxRR)
#     rr = tf.reshape(inp, [1, 1, 1])
    
#     for i in range(7):
#       result = precipitation_model.predict(rr)
#       prediction.append(denormalize(result, minRR, maxRR).item())
#       rr = np.reshape(result, (1, 1, 1))
    
#     return prediction
#   except Exception as e:
#     return str(e), 500
  
# def lumen():
#   try:
#     new_val = request.json
#     prediction = []
    
#     inp = normalize(new_val['lumen'], minLumen, maxLumen)
#     lumen = tf.reshape(inp, [1, 1, 1])
    
#     for i in range(7):
#       result = luminosity_model.predict(lumen)
#       prediction.append(denormalize(result, minLumen, maxLumen).item())
#       lumen = np.reshape(result, (1, 1, 1))
    
#     return prediction
#   except Exception as e:
#     return str(e), 500