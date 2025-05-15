import torch
import cv2
import os

# Load pretrained YOLOv5 model from torch hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_on_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Unable to load image. Check the file format.")
        return

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb)

    # Render results (draw boxes)
    results.render()  # updates results.imgs with boxes and labels

    # Convert back to BGR for OpenCV display
    img_result = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)

    # Show image with detections
    cv2.imshow('YOLOv5 Detection', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Change this to your image path
 input_path = r'./data/pro2.jpg'  # Example image path

    # Check if the input path is a valid image file
if input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
        detect_on_image(input_path)
else:
        print("Please provide a valid image file (.jpg, .png, .jpeg).")
