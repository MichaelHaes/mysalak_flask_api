from flask import request, jsonify
import json
import os
from yolov5.detectModif import run
import base64
import subprocess
from PIL import Image

def base64toimage(original):
    # print(original)
    image = base64.b64decode(original)
    return image

def detect_lalat():
  file = request.json['image']
#   print(file)
  
  if "data:image" in file:
      base64_string = file.split(",")[1]
      file = base64toimage(base64_string)
    #   print(file)
  
  # file.save(os.path.join("images", file.filename))
  if not os.path.exists('images'):
        os.makedirs('images')

  image_path = os.path.join('images', 'image.jpg')  # or any other desired path

  with open(image_path, 'wb') as file_converted:
      file_converted.write(file)
      
#   file_data = Image.open(image_path)
#   print(file_data)

  detected = subprocess.run(["python", "./yolov5/detectModif.py", 
                            "--weights", "./Model/Lalat/best.pt",
                            "--source", './images/image.jpg', 
                            "--save-txt",
                            "--save-conf",
                            "--hide-labels",
                            "--conf-thres", '0.5',
                            "--line-thickness", '3'
                            ],
                            capture_output=True,
                            text=True)
  
  if detected.returncode == 0:
    output = json.loads(detected.stdout) 
    return jsonify(output)
  else:
    return jsonify({"error": "Detection failed", "details": detected.stderr}), 500
  
#   detected = run(weights="./Model/Lalat/best.pt", source=os.path.join("images", 'image.jpg'), save_txt=False, save_conf=False, nosave=False, conf_thres=0.4, hide_conf=True, hide_labels=True, line_thickness=3)
  
#   print(detected)
  
#   return detected