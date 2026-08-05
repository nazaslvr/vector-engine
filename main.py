import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import FastSAM

app = FastAPI()

# 1. Pre-load model on startup to avoid cold start execution delays
model = FastSAM("FastSAM-s.pt")


# Support BOTH the specific endpoint and root endpoint to resolve 404 errors
@app.post("/")
@app.post("/isolate_and_flatten")
async def isolate_and_flatten(file: UploadFile = File(...)):
    # Read the incoming photo upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return Response(content=b"Invalid image format", status_code=400)

    # 2. DOWNSCALE IMAGE TO PREVENT RENDER CPU TIMEOUTS
    # Keeps processing time under 2–4 seconds on free server hardware
    h, w = img.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )

    # 3. Identify and isolate design
    # 'imgsz=640' runs lightweight inference without quality loss
    results = model.predict(img, device="cpu", retina_masks=True, imgsz=640)

    # Failsafe if no object is isolated
    if not results[0].masks:
        return Response(
            content=b"No pattern detected in image", status_code=400
        )

    # Extract mask
    masks = results[0].masks.data.cpu().numpy()
    mask = (masks[0] * 255).astype(np.uint8)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # 4. Drop design onto pure white background
    white_bg = np.ones_like(img, dtype=np.uint8) * 255
    isolated_design = cv2.bitwise_and(img, img, mask=mask)
    background_area = cv2.bitwise_not(mask)
    white_background_masked = cv2.bitwise_and(
        white_bg, white_bg, mask=background_area
    )
    clean_preview = cv2.add(isolated_design, white_background_masked)

    # 5. Crop/Flatten around exact object bounding box
    x, y, w_box, h_box = cv2.boundingRect(mask)
    flattened_output = clean_preview[y : y + h_box, x : x + w_box]

    # Return clean preview image directly
    is_success, buffer = cv2.imencode(".png", flattened_output)
    return Response(content=buffer.tobytes(), media_type="image/png")
