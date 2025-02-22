import ultralytics
import os

model = ultralytics.YOLO('yolov9m.pt')

results = model.train(
  data=f'{os.getcwd()}/The Welding Defect Dataset/The Welding Defect Dataset/data.yaml',
  epochs=1,
  imgsz=640
)

model.val(
  data=f'{os.getcwd()}/The Welding Defect Dataset/The Welding Defect Dataset/data.yaml'
)

# Save the trained model
model_path = f'{os.getcwd()}/yolov8n.pt'
model.save(model_path)
