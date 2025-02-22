import os
import io
from fastapi import FastAPI, File, UploadFile,  Request
from starlette.exceptions import HTTPException
from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from IPython.display import display
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
import base64
import numpy as np

app = FastAPI()

model_path = os.path.join(os.getcwd(), "yolov8n.pt")
model = YOLO(model_path)

# Enable CORS to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing purposes)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# class Image(BaseModel):
#     image_base64: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/detect")
async def detect(request: Request):
    body = await request.json()  # Read body as JSON
    image_base64 = body["image_base64"]
    if "," in image_base64:
        image_base64 = image_base64.split(",")[1]

    try:
        # Decode Base64 to bytes

        image_bytes = base64.b64decode(image_base64)

        image_stream = io.BytesIO(image_bytes)

        # Convert bytes to PIL Image
        image = Image.open(image_stream)

        # Run YOLO model inference
        results = model(image)

        # Process results
        for result in results:

            # Plot detections on image
            result_img = result.plot()

            # Convert result image to PIL
            img_pil = Image.fromarray(result_img)

            # Convert back to Base64 for response
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format="PNG")
            encoded_result = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

            return {"message": "Detection complete", "result_image_base64": encoded_result}
    except Exception as e:
        print("Error opening the image")
        raise HTTPException(status_code=500, detail=f"Error opening the image: {str(e)}")

# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#
#         image = Image.open(f"{os.getcwd()}/test_output.jpg")
#
#         print("debug")
#
#         # Run YOLO model inference
#         results = model(image)
#
#         # Process results
#         for result in results:
#             print("Result:", result)
#
#             # Plot detections on image
#             result_img = result.plot()
#
#             # Convert result image to PIL
#             img_pil = Image.fromarray(result_img)
#
#             # Convert back to Base64 for response
#             img_byte_arr = io.BytesIO()
#             img_pil.save(img_byte_arr, format="PNG")
#             encoded_result = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
#
#             return {"message": "Detection complete", "result_image_base64": encoded_result}
#
#     except Exception as e:
#         print("Error opening the image", e)
#         raise HTTPException(status_code=500, detail=f"Error opening the image: {str(e)}")
    
        
# def detect_service():
#     img_dir = f'{os.getcwd()}/demo/test/images'
#
#     save_dir = f'{os.getcwd()}/demo/test/results'
#
#     image_path = os.path.join(img_dir, "test.jpg")
#     results = model(image_path)
#     for result in results:
#         print("Result:", result)
#         result.plot(save=True, filename=os.path.join(save_dir, os.path.basename(image_path)))
#         result_image_path = os.path.join(save_dir, os.path.basename(image_path))
#         display(Image.open(result_image_path))
#
#
#
