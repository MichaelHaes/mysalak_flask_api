# from flask import Flask, request, jsonify
# import numpy as np
# # import onnxruntime as ort
# from PIL import Image
# import os
# import cv2
# import io
# import uuid

# # Add yolov5 to the Python path
# base_yolo_path = os.path.join(os.path.dirname(__file__), 'Model', 'Lalat', 'best.onnx')
# model = ort.InferenceSession(base_yolo_path)

# def preprocess_image(image_path, input_size=(640, 640)):
#     img = cv2.imread(image_path)
#     img_resized = cv2.resize(img, input_size)
#     img_transposed = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
#     img_normalized = img_transposed / 255.0  # Normalize to [0, 1]
#     img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
#     return img_batch, img

# def preprocess_image_in_memory(image_data, input_size=(640, 640)):
#     # Load image from memory
#     image = Image.open(io.BytesIO(image_data))
#     image = np.array(image)
    
#     # Resize while keeping the aspect ratio
#     img_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
#     img_transposed = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
#     img_normalized = img_transposed / 255.0  # Normalize to [0, 1]
#     img_batch = np.expand_dims(img_normalized, axis=0).astype(np.float32)
#     return img_batch



# # Postprocess the output
# # def postprocess_output(output, original_img, conf_threshold=0.5):
# #     # Assuming the model's output follows the YOLO format (i.e., [x, y, w, h, conf, ...])
# #     predictions = []
# #     for det in output[0]:
# #         conf = det[4]
# #         if conf > conf_threshold:
# #             x, y, w, h = det[:4]
# #             predictions.append((x, y, w, h, conf))
# #             # Draw bounding boxes
# #             start_point = (int(x - w / 2), int(y - h / 2))
# #             end_point = (int(x + w / 2), int(y + h / 2))
# #             cv2.rectangle(original_img, start_point, end_point, (255, 0, 0), 2)
# #     return original_img, predictions

# def postprocess_output(output, original_img, conf_threshold=0.98, iou_threshold=0.1):
#     # Reshape the output to get the expected format (8400, 6)
#     output = output[0].reshape(-1, 6)  # Now the shape is (8400, 6)
    
#     h, w, _ = original_img.shape  # Get the original image dimensions
#     boxes = []
#     confidences = []

#     for det in output:
#         conf = det[4]  # Assuming the confidence score is the 5th element
#         if conf > conf_threshold:  # Filter out low-confidence detections
#             # Extract bounding box coordinates (assumed order: [x_center, y_center, width, height])
#             x_center, y_center, box_width, box_height = det[:4]
            
#             # Scale to original image dimensions (YOLO outputs are typically normalized)
#             x_center *= w
#             y_center *= h
#             box_width *= w
#             box_height *= h
            
#             # Convert center coordinates to top-left and bottom-right corner coordinates
#             x1 = int(x_center - box_width / 2)
#             y1 = int(y_center - box_height / 2)
#             boxes.append([x1, y1, int(box_width), int(box_height)])
#             confidences.append(float(conf))

#     # Apply non-max suppression to avoid overlapping boxes
#     indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

#     # Handle both cases: NMS returning list of lists or simple list
#     if len(indices) > 0 and isinstance(indices[0], list):
#         indices = [i[0] for i in indices]

#     # Draw bounding boxes and count objects
#     predictions = len(indices)
#     for i in indices:
#         box = boxes[i]
#         x1, y1, box_width, box_height = box
#         cv2.rectangle(original_img, (x1, y1), (x1 + box_width, y1 + box_height), (255, 0, 0), 2)

#     return predictions, original_img


# # Perform inference
# def run_inference(model, img_batch):
#     input_name = model.get_inputs()[0].name
#     output_name = model.get_outputs()[0].name
#     outputs = model.run([output_name], {input_name: img_batch})
    
#     print(f"Model output type: {type(outputs)}")
#     print(f"Model output shape: {outputs[0].shape if isinstance(outputs[0], np.ndarray) else outputs[0]}")

#     return outputs

# def letterbox_image(image, size=(640, 640)):
#     h, w, _ = image.shape
#     scale = min(size[0] / h, size[1] / w)
#     new_w = int(w * scale)
#     new_h = int(h * scale)
#     resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
#     # Create a blank image and paste the resized image into it (with padding)
#     new_image = np.full((size[1], size[0], 3), 128, dtype=np.uint8)  # Fills the image with grey
#     new_image[(size[1] - new_h) // 2:(size[1] - new_h) // 2 + new_h,
#               (size[0] - new_w) // 2:(size[0] - new_w) // 2 + new_w] = resized_image
#     return new_image

# # # API endpoint for object detection
# # def detect_lalat():
# #     if 'image' not in request.files:
# #         return jsonify({"error": "No image uploaded"}), 400

# #     file = request.files['image']
# #     img_path = os.path.join("uploads", file.filename)
# #     file.save(img_path)

# #     # Preprocess the image
# #     img_batch, original_img = preprocess_image(img_path)

# #     # Run inference
# #     outputs = run_inference(model, img_batch)

# #     # Post-process and return results
# #     result_img, predictions = postprocess_output(outputs, original_img)

# #     # Save and return the result image
# #     result_img_path = os.path.join("results", file.filename)
# #     cv2.imwrite(result_img_path, result_img)

# #     return send_file(result_img_path, mimetype='image/jpeg')

# # def detect_lalat():    
# #     if 'image' not in request.files:
# #         return jsonify({"error": "No image uploaded"}), 400

# #     file = request.files['image']
# #     img_path = os.path.join("uploads", file.filename)
# #     file.save(img_path)

# #     # Preprocess the image
# #     img_batch, _ = preprocess_image(img_path)

# #     # Run inference
# #     outputs = run_inference(model, img_batch)

# #     # Post-process and return the number of detected objects
# #     num_detections = postprocess_output(outputs)

# #     return jsonify({"num_detections": num_detections})

# def detect_lalat():
#     if 'image' not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     file = request.files['image']

#     # Read the image directly from memory
#     image_data = file.read()

#     # Preprocess the image in-memory
#     img_batch = preprocess_image_in_memory(image_data)
    
#     # Convert the image back to a NumPy array for drawing bounding boxes
#     original_img = np.array(Image.open(io.BytesIO(image_data)))

#     # Run inference
#     outputs = run_inference(model, img_batch)

#     # Post-process and return the number of detected objects and the modified image
#     num_detections, processed_img = postprocess_output(outputs, original_img)

#     # Save the processed image with a unique name
#     # output_filename = f"result_{uuid.uuid4()}.jpg"
#     # output_path = os.path.join("results", output_filename)
#     # cv2.imwrite(output_path, processed_img)

#     return jsonify({
#         "num_detections": num_detections
#         # "output_image": output_filename
#     })

