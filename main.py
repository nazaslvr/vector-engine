import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import FastSAM #

app = FastAPI()

# 1. Load the lightweight, open-source FastSAM model into memory
# The first time this runs, it will automatically download the free ~23MB 'FastSAM-s.pt' weights file
model = FastSAM("FastSAM-s.pt") #

@app.post("/isolate_and_flatten")
async def isolate_and_flatten(file: UploadFile = File(...)):
    # Read the incoming photo upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Identify and isolate the design (Zero Contrast Dependency)
    # The AI predicts the semantic boundaries of the object
    # 'device="cpu"' ensures this runs cleanly on basic server hardware or your local machine
    results = model.predict(img, device="cpu", retina_masks=True) #

    # Failsafe if the AI completely misses finding an object
    if not results[0].masks:
         return Response(content=b"No pattern detected in image", status_code=400)

    # Extract the highest-confidence mask from the AI's results
    masks = results[0].masks.data.cpu().numpy()
    mask = masks[0] 
    
    # Scale the mask back to the original image dimensions
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 3. Show on a pure white background for confirmation
    # Generate a blank white canvas matching the image size
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    
    # Cookie-cutter the exact design out of the original photo using the AI mask
    isolated_design = cv2.bitwise_and(img, img, mask=mask)
    
    # Isolate the background space and replace it with pure white
    background_area = cv2.bitwise_not(mask)
    white_background_masked = cv2.bitwise_and(white_bg, white_bg, mask=background_area)
    
    # Combine them: Original physical material resting on a pure white background
    clean_preview = cv2.add(isolated_design, white_background_masked)

    # 4. Flatten it out
    # Calculate the exact rectangular bounding box of the isolated design
    x, y, w, h = cv2.boundingRect(mask)
    
    # Crop the image tight to the design boundaries, removing excess floor/background space
    flattened_output = clean_preview[y:y+h, x:x+w]

    # Return the pristine preview to the app interface
    is_success, buffer = cv2.imencode(".png", flattened_output)
    return Response(content=buffer.tobytes(), media_type="image/png")
