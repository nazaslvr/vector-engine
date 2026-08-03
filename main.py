import os
import io
import requests
import replicate
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response

app = FastAPI(title="Vector Engine AI")

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Strict CNC design prompt
AI_PROMPT = (
    "Look at this image, find the artwork in it, isolate it, make it flat on the screen, "
    "paint it all black and the background all white, this is for cnc cutting so all black lines "
    "must be connected, distinct, clear and broad enough. The result must be identical to the "
    "design in the image, do not restyle, simplify, or edit the design"
)


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
        # Pass image and prompt to Replicate ControlNet / Lineart Redraw model
        output = replicate.run(
            "jagilley/controlnet-lineart:854e8727697a057c525cdb45ab037f64ecca770a1769cc522874d15144fa5009",
            input={
                "image": io.BytesIO(contents),
                "prompt": AI_PROMPT,
                "num_samples": "1",
                "image_resolution": "768",
                "ddim_steps": 20,
                "scale": 9.0
            }
        )

        # Output is a list containing the generated image URL
        if isinstance(output, list) and len(output) > 1:
            # Index 1 is the generated lineart result
            generated_url = str(output[1])
        elif isinstance(output, list) and len(output) > 0:
            generated_url = str(output[0])
        else:
            raise HTTPException(status_code=500, detail="AI model did not return an image.")

        # Download the AI generated result and return as raw bytes to Android
        img_resp = requests.get(generated_url)
        return Response(content=img_resp.content, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Generation Failed: {str(e)}")
