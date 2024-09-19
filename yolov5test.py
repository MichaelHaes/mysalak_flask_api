from flask import Flask, request
import os
import subprocess
import io
import onnxruntime
from yolov5.detectModif import run

def test():
  file = request.files['image']
  file.save(os.path.join("images", file.filename))
  # file_data = io.BytesIO(file.read()) 
  subprocess.run(["python", "./yolov5/detectModif.py", 
                  "--weights", "./Model/Lalat/best.pt",
                  "--source", os.path.join("images", file.filename), 
                  "--save-txt", 
                  "--save-conf"])
  # return result.stdout.decode('utf-8'), 200
  
  # result = run(weights="C:/Kuliah/IEEE/mysalak_flask_api/Model/Lalat/best.onnx", source=file_data, save_txt=True, save_conf=True)
  return "test"
  
  