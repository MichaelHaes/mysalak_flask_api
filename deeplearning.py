import tensorflow as tf
import os, pandas as pd

base_lstm_path = os.path.join(os.path.dirname(__file__), 'Model', 'DL')

tavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Tavg_LSTM.h5'))
rhavg_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RH_avg_LSTM.h5'))
precipitation_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'RR_LSTM.h5'))
luminosity_model = tf.keras.models.load_model(os.path.join(base_lstm_path, 'Luminosity_LSTM.h5'))

minRH_avg = 0
maxRH_avg = 100
minTavg = 2
maxTavg = 33
minRR = 0
maxRR = 92
minLumen = 0
maxLumen = 2000

def normalize(value, min, max): (value - min) / (max - min)
def denormalize(value, min, max): value * (max - min) + min

def tavg():
  try:
    new_val = request.json
    
    inputTemperatur = normalize(
      new_val['tavg'], minTavg, maxTavg
    )
    reshapedTemperatur = tf.reshape([[inputTemperatur]], [1, 1, 1])
    resultTemperatur = props.temperatureModel.predict(reshapedTemperatur)
    return resultTemperatur
  except Exception as e:
    return str(e), 500
