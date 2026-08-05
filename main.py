import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from ultralytics import FastSAM

# Cap PyTorch to 1 CPU thread so it doesn't spike Render RAM/CPU limits
torch.set_num_threads(1)

app = FastAPI()

# Pre-load model safely
model = FastSAM("FastSAM-s.pt")


@app.post("/")
@app.post("/generate_preview")
@app.post("/isolate_and_flatten")
async def isolate_and_flatten(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return Response(content=b"Invalid image format", status_code=400)

        # 1. Downscale image aggressively for low-RAM server environments
        h, w = img.shape[:2]
        max_dim = 800  # Reduced to 800px to keep RAM spike extremely low
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            img = cv2.resize(
                img,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        # 2. Run FastSAM under torch.no_grad() to eliminate memory leaks
        with torch.no_grad():
            results = model.predict(
                img, device="cpu", retina_masks=False, imgsz=480
            )

        if not results or len(results) == 0 or results[0].masks is None:
            return Response(
                content=b"No pattern detected in image", status_code=400
            )

        # 3. Extract mask safely
        masks = results[0].masks.data.cpu().numpy()
        mask = masks[0].astype(np.uint8) * 255
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # 4. Composite isolated design onto pure white background
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        isolated_design = cv2.bitwise_and(img, img, mask=mask)
        background_area = cv2.bitwise_not(mask)
        white_background_masked = cv2.bitwise_and(
            white_bg, white_bg, mask=background_area
        )
        clean_preview = cv2.add(isolated_design, white_background_masked)

        # 5. Crop tight around bounding box
        x, y, w_box, h_box = cv2.boundingRect(mask)
        if w_box > 0 and h_box > 0:
            flattened_output = clean_preview[y : y + h_box, x : x + w_box]
        else:
            flattened_output = clean_preview

        # 6. Encode PNG
        is_success, buffer = cv2.imencode(".png", flattened_output)
        return Response(content=buffer.tobytes(), media_type="image/png")

    except Exception as e:
        # Prevent 502 Bad Gateway by catching internal errors cleanly
        return Response(
            content=f"Server Error: {str(e)}".encode("utf-8"), status_code=500
        )
