import os
import io
import requests
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
from PIL import Image

app = FastAPI(title="Vector Engine AI")

HF_TOKEN = os.getenv("HF_TOKEN")

# Free Hugging Face inference endpoint (SDXL ControlNet Canny lineart extraction)
HF_API_URL = "https://api-inference.huggingface.co/models/lllyasviel/sd-controlnet-canny"

AI_PROMPT = (
    "flat clean black vector outline pattern, pure white background, high contrast, "
    "sharp edges, cnc cutting template, 2d design, connected black lines, identical pattern"
)


@app.get("/")
def read_root():
    return {"status": "Vector Engine AI Running"}


@app.post("/generate_preview")
@app.post("/generate_preview/")
async def generate_preview(file: UploadFile = File(...)):
    contents = await file.read()

    # If HF Token isn't set, use instant local image extraction
    if not HF_TOKEN:
        return process_local_fallback(contents)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            data=contents,
            timeout=45
        )

        if response.status_code == 200:
            return Response(content=response.content, media_type="image/png")
        else:
            # If HF model is warming up or busy, fallback to clean local thresholding
            print(f"HF Notice: HTTP {response.status_code}, switching to local extractor.")
            return process_local_fallback(contents)

    except Exception as e:
        print("HF Request Error:", traceback.format_exc())
        return process_local_fallback(contents)


def process_local_fallback(image_bytes: bytes) -> Response:
    """Zero-cost engine: isolates pattern, converts to pure flat black & white for CNC preview."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    
    # High-contrast thresholding: turns design black, background white
    threshold = 128
    bw_img = img.point(lambda p: 255 if p > threshold else 0)
    
    buf = io.BytesIO()
    bw_img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
