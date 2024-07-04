from flask import jsonify, request
from statsmodels.tsa.arima.model import ARIMAResults
import os, pandas as pd

base_arima_path = os.path.join(os.path.dirname(__file__), 'Model', 'ARIMA')

tavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'Tavg_ARIMA.pkl'))
# rhavg_model = ARIMAResults.load(os.path.join(base_arima_path, 'RH_avg_ARIMA.pkl'))
# precipitation_model = ARIMAResults.load(os.path.join(base_arima_path, 'RR_ARIMA.pkl'))
# luminosity_model = ARIMAResults.load(os.path.join(base_arima_path, 'Luminosity_ARIMA.pkl'))

def tavg():
  try:
    new_val = request.json
  
    if 'date' not in new_val or 'tavg' not in new_val:
      return jsonify({'error': 'Missing date or tavg value'}), 400
    
    new = {
      'Tanggal': [pd.to_datetime(new_val['date'])],
      'Tavg': [new_val['tavg']]
    }
    new = pd.DataFrame(new)
    new.set_index('Tanggal', inplace=True)
    
    update = tavg_model.append(new)
  
    forecast = tavg_model.forecast(1)
    prediction = forecast.tolist() if hasattr(forecast, 'tolist') else forecast
    
    update.save(os.path.join(base_arima_path, 'Tavg_ARIMA.pkl'))
    return jsonify({'prediction': prediction})
  except Exception as e:
    return str(e), 500
  
def information():
  fitted_values = tavg_model.fittedvalues
  # Convert index to string and then to dictionary
  fitted_values_dict = {str(date): value for date, value in fitted_values.items()}
  return jsonify(fitted_values_dict)
