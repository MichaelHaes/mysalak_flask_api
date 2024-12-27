from flask import request, jsonify
import numpy as np
import os
import base64
from PIL import Image
from yolov5.detectModif import run

# Add yolov5 to the Python path
base_yolo_path = os.path.join(os.path.dirname(__file__), 'Model', 'Lalat', 'yolov5_best.pt')

def base64toimage(original):
    print(original)
    image = base64.b64decode(original)
    return image

def detect_lalat():
  file = request.json['image']
  print(file)

  if "data:image" in file:
      base64_string = file.split(",")[1]
      file = base64toimage(base64_string)
  
  print(file)
  
  # file.save(os.path.join("images", file.filename))
  if not os.path.exists('images'):
        os.makedirs('images')
  image_path = os.path.join('images', 'image.jpg') 
  with open(image_path, 'wb') as file_converted:
      file_converted.write(file)
      
#   file_data = Image.open(image_path)
#   print(file_data)
  # subprocess.run(["python", "./yolov5/detectModif.py", 
  #                 "--weights", "./Model/Lalat/best.pt",
  #                 "--source", os.path.join("images", file.filename), 
  #                 "--save-txt", 
  #                 "--save-conf"])
  # return result.stdout.decode('utf-8'), 200
  
  detected = run(weights="./Model/Lalat/yolov5_best.pt", 
                 source=os.path.join("images", 'image.jpg'), 
                 imgsz=(448,448), 
                 save_txt=False, 
                 save_conf=False, 
                 nosave=False, 
                 conf_thres=0.4, 
                 hide_conf=True, 
                 hide_labels=True, 
                 line_thickness=3)
  
  print(detected)
  
  return detected