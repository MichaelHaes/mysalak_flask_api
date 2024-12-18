from flask import jsonify, request
import lightgbm as lgb
import os, pandas as pd
import json

base_arima_path = os.path.join(os.path.dirname(__file__), 'Model', 'LightGBM')

temperature_model = lgb.Booster(model_file=os.path.join(base_arima_path, 'temp_lgbm.txt'))
humidity_model = lgb.Booster(model_file=os.path.join(base_arima_path, 'humidity_lgbm.txt'))
precipitation_model = lgb.Booster(model_file=os.path.join(base_arima_path, 'precip_lgbm.txt'))
luminosity_model = lgb.Booster(model_file=os.path.join(base_arima_path, 'lux_lgbm.txt'))

def pred_result_lightgbm():
  new_val = request.json
  
  new_df = pd.DataFrame({
    'Tanggal': [pd.to_datetime(new_val['createdAt'])],
    'Temp': [new_val['temperature']],
    'RH_avg': [new_val['humidity']],
    'RR': [new_val['tips']],
    'Luminosity': [new_val['lux']]
  })
  new_df.set_index('Tanggal', inplace=True)
  new_df = new_df.resample('h').nearest()
  one_week = pd.DataFrame({'Tanggal':pd.date_range(start=new_df.index[0], periods=24, freq='h')})
  one_week.set_index('Tanggal', inplace=True)
  print(one_week)
  
  one_week["day_of_week"] = one_week.index.dayofweek
  one_week["day_of_year"] = one_week.index.dayofyear
  one_week["hour"] = one_week.index.hour
  one_week["minutes"] = one_week.index.minute
  one_week["day"] = one_week.index.day
  one_week["month"] = one_week.index.month
  one_week["quarter"] = one_week.index.quarter
  one_week["year"] = one_week.index.year
  
  cols = [col for col in one_week.columns if col not in ['RH_avg', 'Temp', 'RR', 'Luminosity', 'Tanggal']]
  X_test = one_week[cols]
  
  res = {
      'Tavg': tavg(X_test),
      'RH_avg': rh_avg(X_test),
      'RR': rr(X_test),
      'Lumen': lumen(X_test)
    }
  data_serializable = {key: value.tolist() for key, value in res.items()}
  print(json.dumps(data_serializable))
  return json.dumps(data_serializable)
  
def tavg(X_test):
  try:
    prediction = temperature_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def rh_avg(X_test):
  try:
    prediction = humidity_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def rr(X_test):
  try:
    prediction = precipitation_model.predict(X_test) 
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def lumen(X_test):
  try:
    prediction = luminosity_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
