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
  
  new_df["day_of_week"] = new_df.index.dayofweek
  new_df["day_of_year"] = new_df.index.dayofyear
  new_df["hour"] = new_df.index.hour
  new_df["minutes"] = new_df.index.minute
  new_df["day"] = new_df.index.day
  new_df["month"] = new_df.index.month
  new_df["quarter"] = new_df.index.quarter
  new_df["year"] = new_df.index.year
  
  cols = [col for col in new_df.columns if col not in ['RH_avg', 'Temp', 'RR', 'Luminosity', 'Tanggal']]
  print(cols)
  X_test = new_df[cols]
  
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
