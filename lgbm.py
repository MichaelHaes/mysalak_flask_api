from flask import jsonify, request
import lightgbm as lgb
import os, pandas as pd
import json

base_path = os.path.join(os.path.dirname(__file__), 'Model', 'LightGBM')

temperature_model = lgb.Booster(model_file=os.path.join(base_path, 'temp_lgbm.txt'))
humidity_model = lgb.Booster(model_file=os.path.join(base_path, 'humidity_lgbm.txt'))
precipitation_model = lgb.Booster(model_file=os.path.join(base_path, 'precip_lgbm.txt'))
luminosity_model = lgb.Booster(model_file=os.path.join(base_path, 'lux_lgbm.txt'))

def pred_result_lightgbm():
  new_val = request.json
  
  new_df = pd.DataFrame({
    'Tanggal': [pd.to_datetime(new_val['createdAt'])+pd.Timedelta(hours=1)],
    # 'temperature': [new_val['temperature']],
    # 'humidity': [new_val['humidity']],
    # 'tips': [new_val['tips']],
    # 'lux': [new_val['lux']]
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
      'date': pd.date_range(start=new_df.index[0], periods=168, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
      'temperature': temperature(X_test),
      'humidity': humidity(X_test),
      'precipitation': tips(X_test),
      'luminosity': lux(X_test)
    }
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
