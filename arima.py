from flask import jsonify, request
from statsmodels.tsa.arima.model import ARIMAResults
import os, pandas as pd

base_path = os.path.join(os.path.dirname(__file__), 'Model', 'ARIMA')

temperature_model = ARIMAResults.load(os.path.join(base_path, 'temp_arima.pkl'))
humidity_model = ARIMAResults.load(os.path.join(base_path, 'humidity_arima.pkl'))
precipitation_model = ARIMAResults.load(os.path.join(base_path, 'precip_arima.pkl'))
luminosity_model = ARIMAResults.load(os.path.join(base_path, 'lux_arima.pkl'))

def pred_result_arima():
  new_val = request.json
  new_df = pd.DataFrame({
    'Tanggal': [pd.to_datetime(new_val['createdAt'])]
  })
  new_df.set_index('Tanggal', inplace=True)
  new_df = new_df.resample('h').nearest()
  end_date = pd.to_datetime(new_df.index[0]) + pd.Timedelta(weeks=1)
  steps = 168

  res = {
      'date': pd.date_range(start=new_df.index[0]+pd.Timedelta(hours=1), periods=168, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
      'temperature': temperature(steps),
      'humidity': humidity(steps),
      'precipitation': tips(steps),
      'luminosity': lux(steps)
    }
  data_serializable = {key: value.tolist() for key, value in res.items()}
  return jsonify(data_serializable)
  
def temperature(steps=168):
  try:
    forecast = temperature_model.forecast(steps=steps)
    # print(forecast[-168:])
    return forecast
    # return forecast[-168:]
  except Exception as e:
    return str(e), 500
  
def humidity(steps=168):
  try:
    forecast = humidity_model.forecast(steps=steps)
    
    # return forecast[-168:]
    return forecast
  except Exception as e:
    return str(e), 500
  
def tips(steps=168):
  try:
    forecast = precipitation_model.forecast(steps=steps)
    
    # return forecast[-168:]
    return forecast
  except Exception as e:
    return str(e), 500
  
def lux(steps=168):
  try:
    forecast = luminosity_model.forecast(steps=steps)
    
    # return forecast[-168:]
    return forecast
  except Exception as e:
    return str(e), 500
