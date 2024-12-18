from flask import jsonify, request
from statsmodels.tsa.arima.model import ARIMAResults
import os, pandas as pd

base_arima_path = os.path.join(os.path.dirname(__file__), 'Model', 'ARIMA')

tavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'temp_arima.pkl'))
rhavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'humidity_arima.pkl'))
precipitation_model = ARIMAResults.load(os.path.join(base_arima_path, 'precip_arima.pkl'))
luminosity_model = ARIMAResults.load(os.path.join(base_arima_path, 'lux_arima.pkl'))
# tavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'Tavg_ARIMA.pkl'))
# rhavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'RH_avg_ARIMA.pkl'))
# precipitation_model = ARIMAResults.load(os.path.join(base_arima_path, 'RR_ARIMA.pkl'))
# luminosity_model = ARIMAResults.load(os.path.join(base_arima_path, 'Luminosity_ARIMA.pkl'))

def pred_result_arima():
  new_val = request.json
  end_date = pd.to_datetime(new_val['createdAt']) + pd.Timedelta(weeks=1)
    
  res = {
      'temperature': tavg(end_date),
      'humidity': rh_avg(end_date),
      'precipitation': rr(end_date),
      'luminosity': lumen(end_date)
    }
  return res
  
def tavg(end_date):
  try:
    forecast = tavg_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
  
def rh_avg(end_date):
  try:
    forecast = rhavg_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
  
def rr(end_date):
  try:
    forecast = precipitation_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
  
  
def lumen(end_date):
  try:
    forecast = luminosity_model.forecast(steps=end_date)
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction[-168:]
  except Exception as e:
    return str(e), 500
