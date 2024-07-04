from flask import jsonify, request
from statsmodels.tsa.arima.model import ARIMAResults
import os, pandas as pd

base_arima_path = os.path.join(os.path.dirname(__file__), 'Model', 'ARIMA')

tavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'Tavg_ARIMA.pkl'))
rhavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'RH_avg_ARIMA.pkl'))
precipitation_model = ARIMAResults.load(os.path.join(base_arima_path, 'RR_ARIMA.pkl'))
luminosity_model = ARIMAResults.load(os.path.join(base_arima_path, 'Luminosity_ARIMA.pkl'))

def pred_result_arima():
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
    
    new = pd.DataFrame({
      'Tanggal': [pd.to_datetime(new_val['date'])],
      'Tavg': [new_val['tavg']]
    })
    new.set_index('Tanggal', inplace=True)
    
    # update = tavg_model.append(new)
    forecast = tavg_model.forecast(1)
    # update.save(os.path.join(base_arima_path, 'Tavg_ARIMA.pkl'))
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def rh_avg():
  try:
    new_val = request.json
    
    new = pd.DataFrame({
      'Tanggal': [pd.to_datetime(new_val['date'])],
      'RH_avg': [new_val['rh_avg']]
    })
    new.set_index('Tanggal', inplace=True)
    
    # update = tavg_model.append(new)
    forecast = rhavg_model.forecast(1)
    # update.save(os.path.join(base_arima_path, 'RH_avg_ARIMA.pkl'))
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def rr():
  try:
    new_val = request.json
    
    new = pd.DataFrame({
      'Tanggal': [pd.to_datetime(new_val['date'])],
      'RR': [new_val['rr']]
    })
    new.set_index('Tanggal', inplace=True)
    
    # update = tavg_model.append(new)
    forecast = precipitation_model.forecast(1)
    # update.save(os.path.join(base_arima_path, 'RR_ARIMA.pkl'))
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def lumen():
  try:
    new_val = request.json
    
    new = pd.DataFrame({
      'Tanggal': [pd.to_datetime(new_val['date'])],
      'Luminosity': [new_val['lumen']]
    })
    new.set_index('Tanggal', inplace=True)
    
    # update = tavg_model.append(new)
    forecast = luminosity_model.forecast(1)
    # update.save(os.path.join(base_arima_path, 'Luminosity_ARIMA.pkl'))
    
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast 
    return prediction
  except Exception as e:
    return str(e), 500
  
def information():
  fitted_values = tavg_model.fittedvalues
  # Convert index to string and then to dictionary
  fitted_values_dict = {str(date): value for date, value in fitted_values.items()}
  return jsonify(fitted_values_dict)
