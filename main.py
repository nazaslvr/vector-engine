import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import FastSAM

app = FastAPI()

# Load FastSAM model into memory
model = FastSAM("FastSAM-s.pt")

# Catch ALL common routes (root, with/without trailing slashes, /upload, /process)
@app.post("/")
@app.post("/isolate_and_flatten")
@app.post("/isolate_and_flatten/")
@app.post("/upload")
@app.post("/upload/")
@app.post("/process")
@app.post("/process/")
async def isolate_and_flatten(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"Invalid image format", status_code=400)

    # Downscale image to fit Render's CPU limit
    h, w = img.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Run FastSAM segmentation
    results = model.predict(img, device="cpu", retina_masks=True, imgsz=640)

    if not results[0].masks:
        return Response(content=b"No pattern detected in image", status_code=400)

    # Extract mask
    masks = results[0].masks.data.cpu().numpy()
    mask = (masks[0] * 255).astype(np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Apply pure white background
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    isolated_design = cv2.bitwise_and(img, img, mask=mask)
    background_area = cv2.bitwise_not(mask)
    white_background_masked = cv2.bitwise_and(white_bg, white_bg, mask=background_area)
    clean_preview = cv2.add(isolated_design, white_background_masked)

    # Crop tight to the design bounding box
    x, y, w_box, h_box = cv2.boundingRect(mask)
    flattened_output = clean_preview[y:y+h_box, x:x+w_box]

    # Return output image
    is_success, buffer = cv2.imencode(".png", flattened_output)
    return Response(content=buffer.tobytes(), media_type="image/png")
