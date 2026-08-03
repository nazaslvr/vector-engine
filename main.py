import os
import base64
import requests
import fal_client
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Vector Engine AI")

@app.get("/")
def read_root():
    return {"status": "Vector Engine AI Running"}

@app.post("/generate_preview")
@app.post("/generate_preview/")
async def generate_preview(file: UploadFile = File(...)):
    # Check if FAL_KEY is set in Render Environment Variables
    if not os.getenv("FAL_KEY"):
        raise HTTPException(status_code=500, detail="FAL_KEY environment variable is missing on Render.")

    contents = await file.read()
    
    # 1. Convert uploaded image file bytes into a base64 data URL for the AI model
    encoded_image = base64.b64encode(contents).decode("utf-8")
    mime_type = file.content_type or "image/png"
    image_url = f"data:{mime_type};base64,{encoded_image}"

    try:
        # 2. Submit image directly to fal.ai ControlNet/SDXL for AI pattern generation
        result = fal_client.subscribe(
            "fal-ai/fast-sdxl/image-to-image",
            arguments={
                "image_url": image_url,
                "prompt": (
                    "Look at this image, find the artwork in it, isolate it, make it flat on the screen, "
                    "paint it all black and the background all white, this is for cnc cutting so all black lines "
                    "must be connected, distinct, clear and broad enough. The result must be identical to the "
                    "design in the image, do not restyle, simplify, or edit the design"
                ),
                "strength": 0.65,
                "guidance_scale": 7.5
            }
        )

        # 3. Retrieve the generated AI image
        output_url = result["images"][0]["url"]
        img_data = requests.get(output_url).content
        
        return Response(content=img_data, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Failed: {str(e)}")
