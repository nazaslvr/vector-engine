from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
import numpy as np
import cv2
import vtracer

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "AI Laser Vector Engine API Running"}

@app.post("/recreate-design/")
async def recreate_design(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    # Isolate subject structure
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )

    temp_input_path = "/tmp/input.png"
    cv2.imwrite(temp_input_path, binary)

    # Convert pattern geometry into SVG spline paths
    svg_output = vtracer.convert_to_string(
        temp_input_path,
        colormode='binary',
        hierarchical='stacked',
        mode='spline',
        filter_speckle=4,
        corner_threshold=60
    )

    return Response(content=svg_output, media_type="image/svg+xml")
