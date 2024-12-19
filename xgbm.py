from flask import jsonify, request
import xgboost as xgb
import os, pandas as pd
import json

base_path = os.path.join(os.path.dirname(__file__), 'Model', 'XGBoost')

booster = xgb.Booster()
booster.load_model(os.path.join(base_path, 'temp_xgboost.json'))
temperature_model = xgb.XGBRegressor()
temperature_model._booster = booster

# temperature_model = xgb.XGBRegressor().load_model(os.path.join(base_path, 'temp_xgboost.json'))
humidity_model = xgb.XGBRegressor().load_model(os.path.join(base_path, 'humidity_xgboost.json'))
precipitation_model = xgb.XGBRegressor().load_model(os.path.join(base_path, 'precip_xgboost.json'))
luminosity_model = xgb.XGBRegressor().load_model(os.path.join(base_path, 'lux_xgboost.json'))

def pred_result_xgboost():
  new_val = request.json
  
  new_df = pd.DataFrame({
    'Tanggal': [pd.to_datetime(new_val['createdAt'])],
    'temperature': [new_val['temperature']],
    'humidity': [new_val['humidity']],
    'tips': [new_val['tips']],
    'lux': [new_val['lux']]
  })
  new_df.set_index('Tanggal', inplace=True)
  new_df = new_df.resample('h').nearest()
  one_week = pd.DataFrame({'Tanggal':pd.date_range(start=new_df.index[0], periods=24, freq='h')})
  one_week.set_index('Tanggal', inplace=True)
  
  one_week["day_of_week"] = one_week.index.dayofweek
  one_week["day_of_year"] = one_week.index.dayofyear
  one_week["hour"] = one_week.index.hour
  one_week["minutes"] = one_week.index.minute
  one_week["day"] = one_week.index.day
  one_week["month"] = one_week.index.month
  one_week["quarter"] = one_week.index.quarter
  one_week["year"] = one_week.index.year
  
  cols = [col for col in one_week.columns if col not in ['humidity', 'temperature', 'tips', 'lux', 'Tanggal']]
  X_test = one_week[cols]
  
  res = {
      'date': pd.date_range(start=new_df.index[0], periods=24, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
      'temperature': temperature(X_test),
      'humidity': humidity(X_test),
      'precipitation': tips(X_test),
      'luminosity': lux(X_test)
    }
  print(booster)
  data_serializable = {key: value.tolist() for key, value in res.items()}
  return json.dumps(data_serializable)
  
def temperature(X_test):
  try:
    prediction = temperature_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def humidity(X_test):
  try:
    prediction = humidity_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def tips(X_test):
  try:
    prediction = precipitation_model.predict(X_test) 
    
    return prediction
  except Exception as e:
    return str(e), 500
  
  
def lux(X_test):
  try:
    prediction = luminosity_model.predict(X_test)
    
    return prediction
  except Exception as e:
    return str(e), 500
