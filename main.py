import os
import io
import cv2
import numpy as np
import replicate
import requests
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="Vector Engine AI")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")


def order_points(pts):
    """Orders 4 contour points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def perspective_warp(img, mask):
    """Finds the bounding quad of the isolated object and flattens it top-down."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2)
    else:
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect)

    rect_pts = order_points(pts)
    (tl, tr, br, bl) = rect_pts

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect_pts, dst)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped


@app.get("/")
def read_root():
    return {"status": "Vector Engine AI Running"}


@app.post("/generate_preview")
@app.post("/generate_preview/")
async def generate_preview(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image uploaded")

    # Step 1: Attempt SAM 2 Cloud Segmentation if token exists
    if REPLICATE_API_TOKEN:
        try:
            output = replicate.run(
                "meta/sam-2-realtime:1e29e925c04b4081c70e2f5ffed5c8b58a1f8db112bf8f3f88f8d689b0d625d8",
                input={"image": io.BytesIO(contents)}
            )
            if isinstance(output, list) and len(output) > 0:
                mask_url = output[0]
                mask_resp = requests.get(mask_url)
                mask_arr = np.frombuffer(mask_resp.content, np.uint8)
                mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

                # Process original image to B&W line art (black lines on white background)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                bw_img = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
                )

                # Crop and warp based on SAM 2 mask
                warped = perspective_warp(bw_img, mask_img)
                _, encoded = cv2.imencode(".png", warped)
                return Response(content=encoded.tobytes(), media_type="image/png")
        except Exception:
            pass  # Fallback smoothly to OpenCV classical processing

    # Step 2: Fallback OpenCV Processing (Black lines on clean White canvas)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # THRESH_BINARY ensures white background (255) and black drawing lines (0)
    bw_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
    )
    
    # Inverted mask strictly for finding bounding box contours
    inv_mask = cv2.bitwise_not(bw_img)
    warped = perspective_warp(bw_img, inv_mask)

    _, encoded = cv2.imencode(".png", warped)
    return Response(content=encoded.tobytes(), media_type="image/png")
