from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import cv2

# Add yolov5 to the Python path
base_yolo_path = os.path.join(os.path.dirname(__file__), 'Model', 'Lalat', 'model_yolov5.onnx')
ort_session = ort.InferenceSession(base_yolo_path)

# Preprocess image
def preprocess_image(image):
    input_size = (640, 640)  # Adjust to your model's input size
    image = cv2.resize(image, input_size)
    image = image / 255.0  # Normalize the pixel values
    image = np.transpose(image, (2, 0, 1))  # Change to (channels, height, width)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype(np.float32)
    return image

# Define function to post-process YOLOv5 predictions
def postprocess_predictions(predictions, confidence_threshold=0.5, iou_threshold=0.5):
    # This part requires parsing of the YOLOv5 ONNX output
    # Parse output format, typically contains bounding boxes, objectness score, and class probabilities
    predictions = np.squeeze(predictions[0])  # Remove batch dimension

    boxes, scores, classes = [], [], []  # Dummy data, replace with actual post-processing logic
    for pred in predictions:
        if pred[4] > confidence_threshold:  # Check objectness score
            box = pred[:4]  # Extract bounding box
            score = pred[4]  # Extract confidence score
            class_id = np.argmax(pred[5:])  # Get the class with highest score
            boxes.append(box)
            scores.append(score)
            classes.append(class_id)
    
    return boxes, scores, classes

# # API endpoint for object detection
def detect_lalat():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    image = Image.open(file.stream)
    image = np.array(image)
    
    print(file)

    # Preprocess the image
    input_image = preprocess_image(image)

    # Run ONNX model inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_image}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # print(ort_outs)
    
    for output in ort_outs:
        print(output.shape)

    # Post-process the predictions
    boxes, scores, classes = postprocess_predictions(ort_outs)

    # Return the results as JSON
    result = {
        'boxes': boxes,
        'scores': scores,
        'classes': classes
    }
    
    return jsonify(result)
    # return jsonify({'jumlah': len(boxes)})

# def process_outputs(outputs):
#     detections = []
#     for output in outputs:
#         batch_size, num_anchors, height, width, num_classes = output.shape
#         # Flattening output to simplify processing
#         output = output.reshape((batch_size, num_anchors, -1))  # Reshape to (batch_size, num_anchors, num_predictions)
        
#         for anchor in range(num_anchors):
#             for i in range(height):
#                 for j in range(width):
#                     prediction = output[0, anchor, i * width + j]
#                     bbox = prediction[:4]
#                     objectness = prediction[4]
#                     class_scores = prediction[5:]

#                     # Apply sigmoid activation
#                     objectness = 1 / (1 + np.exp(-objectness))
#                     class_scores = 1 / (1 + np.exp(-class_scores))

#                     # Compute confidence scores
#                     confidence_scores = objectness * class_scores

#                     # Filter by confidence threshold (e.g., 0.5)
#                     if np.max(confidence_scores) > 0.5:
#                         detected_class = np.argmax(confidence_scores)
#                         detections.append({
#                             'bbox': bbox,
#                             'confidence': np.max(confidence_scores),
#                             'class': detected_class
#                         })

#     return detections

# def apply_nms(detections, iou_threshold=0.4):
#     boxes = []
#     confidences = []
#     class_ids = []

#     for detection in detections:
#         bbox = detection['bbox']
#         x_center, y_center, width, height = bbox
#         x_center = int(x_center * 640)  # Adjust to original image size
#         y_center = int(y_center * 640)
#         width = int(width * 640)
#         height = int(height * 640)
#         boxes.append([x_center - width / 2, y_center - height / 2, width, height])
#         confidences.append(detection['confidence'])
#         class_ids.append(detection['class'])

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=iou_threshold)
#     final_detections = [detections[i] for i in indices.flatten()]
#     return final_detections

# def detect_lalat():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image provided'}), 400
    
#     file = request.files['image']
#     image = Image.open(file.stream)
#     image = np.array(image)

#     print(image.shape)

#     # Preprocess the image
#     input_image = preprocess_image(image)

#     # Run ONNX model inference
#     ort_inputs = {ort_session.get_inputs()[0].name: input_image}
#     ort_outs = ort_session.run(None, ort_inputs)

#     # Process model outputs
#     detections = process_outputs(ort_outs)
#     filtered_detections = apply_nms(detections)

#     num_detected_objects = len(filtered_detections)

#     return jsonify({'num_detected_objects': num_detected_objects})