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
  end_date = pd.to_datetime(new_val['createdAt']) + pd.Timedelta(weeks=1)
    
  res = {
      'temperature': temperature(end_date),
      'humidity': humidity(end_date),
      'precipitation': tips(end_date),
      'luminosity': lux(end_date)
    }
  return res
  
def temperature(end_date):
  try:
    forecast = temperature_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
def humidity(end_date):
  try:
    forecast = humidity_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
def tips(end_date):
  try:
    forecast = precipitation_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
def lux(end_date):
  try:
    forecast = luminosity_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
