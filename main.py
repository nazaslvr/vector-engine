from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import vtracer
import ezdxf
import base64
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Laser Vector Engine API Running"}

@app.post("/process-design/")
async def process_design(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # 1. Convert to high-contrast B&W image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_img = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    temp_bw_path = "/tmp/bw_preview.jpg"
    cv2.imwrite(temp_bw_path, bw_img)

    # Encode B&W JPEG to Base64
    _, encoded_jpeg = cv2.imencode('.jpg', bw_img)
    jpeg_base64 = base64.b64encode(encoded_jpeg).decode('utf-8')

    # 2. Vectorize contours to DXF format
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
        if len(points) > 2:
            msp.add_lwpolyline(points, close=True)

    temp_dxf_path = "/tmp/output.dxf"
    doc.saveas(temp_dxf_path)

    with open(temp_dxf_path, "rb") as f:
        dxf_bytes = f.read()

    dxf_base64 = base64.b64encode(dxf_bytes).decode('utf-8')

    # Clean up
    for path in [temp_bw_path, temp_dxf_path]:
        if os.path.exists(path):
            os.remove(path)

    # Return both B&W preview and DXF payload
    return {
        "bw_image_jpeg_base64": f"data:image/jpeg;base64,{jpeg_base64}",
        "dxf_file_base64": dxf_base64
    }

