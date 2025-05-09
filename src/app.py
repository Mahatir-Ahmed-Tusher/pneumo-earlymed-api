from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from PIL import Image
import io
from .inference import run_inference
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Pneumonia Detection API")

# Add CORS middleware to allow frontend requests
logger.info("Adding CORS middleware for origins: ['http://localhost:8080', 'https://earlymed.vercel.app']")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://earlymed.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure models directory exists
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
if not os.path.exists(MODELS_DIR):
    logger.error("Models directory not found.")
    raise RuntimeError("Models directory not found.")

# Root endpoint
@app.get("/")
async def root():
    logger.info("Handling GET / request")
    return {
        "message": "Welcome to the Pneumonia Detection API",
        "endpoints": {
            "health": "/health",
            "analyze": "/analyze/ (POST)",
            "docs": "/docs"
        }
    }

# API endpoint for analyzing chest X-ray images
@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    logger.info("Received POST /analyze/ request")
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png"]:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail="Only JPEG or PNG images are supported.")

        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        result = run_inference(image)

        logger.info("Successfully processed /analyze/ request")
        # Format response to match Gradio app
        return JSONResponse(content={
            "prediction": result["prediction"],
            "confidence": f"{result['confidence']:.2f}%"
        })
    except Exception as e:
        logger.error(f"Error in /analyze/: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Handling GET /health request")
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)