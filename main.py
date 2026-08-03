import os
import io
import requests
import replicate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Vector Engine AI")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

AI_PROMPT = (
    "Look at this image, find the artwork in it, isolate it, make it flat on the screen, "
    "paint it all black and the background all white, this is for cnc cutting so all black lines "
    "must be connected, distinct, clear and broad enough. The result must be identical to the "
    "design in the image, do not restyle, simplify, or edit the design"
)

NEGATIVE_PROMPT = "background color, dark background, gray background, shadows, 3D render, photorealistic, noise, blur, color, text, watermark"


@app.get("/")
def read_root():
    return {"status": "Vector Engine AI Running"}


@app.post("/generate_preview")
@app.post("/generate_preview/")
async def generate_preview(file: UploadFile = File(...)):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(
            status_code=500, 
            detail="REPLICATE_API_TOKEN is missing in Render environment variables."
        )

    contents = await file.read()

    try:
        # Wrap raw bytes as a file-like object for Replicate file upload
        image_file = io.BytesIO(contents)

        # Run SDXL image-to-image synthesis using stable model reference
        output = replicate.run(
            "stability-ai/sdxl",
            input={
                "image": image_file,
                "prompt": AI_PROMPT,
                "negative_prompt": NEGATIVE_PROMPT,
                "prompt_strength": 0.65,
                "num_inference_steps": 30,
                "guidance_scale": 9.0
            }
        )

        # Retrieve generated image URL
        if isinstance(output, list) and len(output) > 0:
            generated_url = str(output[0])
        else:
            raise HTTPException(status_code=500, detail="AI model did not return an image.")

        # Download the AI generated flat artwork and return bytes
        img_resp = requests.get(generated_url)
        return Response(content=img_resp.content, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Failed: {str(e)}")
