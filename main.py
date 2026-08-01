from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import numpy as np
import cv2
import ezdxf
import base64
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Laser Vector Engine API Running"}

# STEP 1: Process raw photo into clean Black & White preview JPEG
@app.post("/generate-preview/")
async def generate_preview(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smooth noise while keeping clean design edges
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding to extract high-contrast black & white pattern
    bw_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 15, 3
    )

    # Encode to Base64 JPEG for app preview
    _, encoded_jpeg = cv2.imencode('.jpg', bw_img)
    jpeg_base64 = base64.b64encode(encoded_jpeg).decode('utf-8')

    return {
        "bw_image_jpeg_base64": f"data:image/jpeg;base64,{jpeg_base64}"
    }

# STEP 2: Convert confirmed/edited B&W preview into DXF vector file
@app.post("/generate-dxf/")
async def generate_dxf(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    bw_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    if bw_img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format"})

    # Extract clean vector contours
    contours, _ = cv2.findContours(bw_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create DXF CAD file
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()

    for contour in contours:
        points = [(float(pt[0][0]), float(pt[0][1])) for pt in contour]
        if len(points) > 2:
            msp.add_lwpolyline(points, close=True)

    temp_dxf_path = "/tmp/output.dxf"
    doc.saveas(temp_dxf_path)

    with open(temp_dxf_path, "rb") as f:
        dxf_bytes = f.read()

    if os.path.exists(temp_dxf_path):
        os.remove(temp_dxf_path)

    return Response(
        content=dxf_bytes, 
        media_type="application/dxf", 
        headers={"Content-Disposition": "attachment; filename=design.dxf"}
    )
