import os
import random
from PIL import Image
from IPython.display import display
from ultralytics import YOLO


# Function to display results
def display_results(model, img_dir, num_images=10):
  images = os.listdir(img_dir)
  sample_images = random.sample(images, num_images)
  save_dir = 'runs/detect/exp'
  os.makedirs(save_dir, exist_ok=True)
  
  for image_name in sample_images:
    image_path = os.path.join(img_dir, image_name)
    results = model(image_path)
    for result in results:
      result.plot(save=True, filename=os.path.join(save_dir, os.path.basename(image_path)))
      result_image_path = os.path.join(save_dir, os.path.basename(image_path))
      display(Image.open(result_image_path))
      
model_path = f'{os.getcwd()}/yolov8n.pt'
# Load the trained model for inference
model = YOLO(model_path)

print("Displaying results from model trained on version 1 with augmentation:")
img_dir = f'{os.getcwd()}/The Welding Defect Dataset/The Welding Defect Dataset/test/images'
display_results(model, img_dir)