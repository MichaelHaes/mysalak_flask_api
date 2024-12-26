import os
import cv2

image_path = "/content/yolov5/runs/detect/exp10/DSC00567.JPG"
label_folder = "/content/yolov5/runs/detect/exp10/labels"
output_image_path = "/content/output2.jpg"

output_folder = os.path.dirname(output_image_path)
os.makedirs(output_folder, exist_ok=True)

image_name = os.path.basename(image_path).split(".")[0]
label_file = os.path.join(label_folder, f"{image_name}.txt")

num_objects = 0
if os.path.exists(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
        num_objects = sum(1 for line in lines if line.strip().split()[0] == '1')
else:
    raise FileNotFoundError(f"File label tidak ditemukan: {label_file}")

image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Gambar tidak ditemukan: {image_path}")

target_size = 448
image_resized = cv2.resize(image, (target_size, target_size))

text = f"Jumlah lalat: {num_objects}"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_color = (255, 255, 255)
thickness = 2

text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
text_x = (image_resized.shape[1] - text_size[0]) // 2
text_y = 3

border_size = 40
border_color = (0, 0, 0)
image_with_border = cv2.copyMakeBorder(
    image_resized,
    top=border_size, bottom=border_size,
    left=border_size, right=border_size,
    borderType=cv2.BORDER_CONSTANT,
    value=border_color
)


text_x = (image_with_border.shape[1] - text_size[0]) // 2
text_y = border_size // 2 + text_size[1] // 2


cv2.putText(image_with_border, text, (text_x, text_y), font, font_scale, font_color, thickness)


success = cv2.imwrite(output_image_path, image_with_border)
if success:
    print(f"Hasil: {output_image_path}")
else:
    print("Gagal menyimpan gambar. Periksa format atau path output.")