"""
FastAPI Backend for ExplainMyXray.
Provides REST endpoint for X-ray image explanation.
"""
import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Global predictor instance
predictor = None


class ExplanationResponse(BaseModel):
    explanation: str
    status: str = "success"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global predictor
    
    # Import here to avoid circular imports
    import sys
    sys.path.insert(0, str(__file__).replace("/app/api.py", ""))
    
    from src.inference.predictor import MedGemmaPredictor
    
    adapter_path = "./medgemma_lora_adapters"
    
    try:
        predictor = MedGemmaPredictor(
            adapter_path=adapter_path,
            load_in_4bit=True,
        )
        print("✅ MedGemma model loaded successfully")
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        print("Running in demo mode with placeholder responses")
        predictor = None
    
    yield
    
    # Cleanup
    predictor = None


app = FastAPI(
    title="ExplainMyXray API",
    description="Convert Chest X-rays into patient-friendly explanations",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "ExplainMyXray API",
        "status": "running",
        "model_loaded": predictor is not None,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "model_ready": predictor is not None}


@app.post("/explain", response_model=ExplanationResponse)
async def explain_xray(file: UploadFile):
    """
    Upload a Chest X-ray image and get a patient-friendly explanation.
    
    - **file**: X-ray image (PNG, JPG, JPEG)
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (PNG, JPG, JPEG)"
        )
    
    # Read image bytes
    image_bytes = await file.read()
    
    if predictor is None:
        # Demo mode - return placeholder
        return ExplanationResponse(
            explanation=(
                "This is a demo response. The AI model is not loaded. "
                "To get real explanations, please ensure the LoRA adapters "
                "are downloaded and the model is properly initialized. "
                "Your chest X-ray appears to show normal lung fields with "
                "no obvious abnormalities visible."
            ),
            status="demo"
        )
    
    try:
        explanation = predictor.predict(image_bytes)
        return ExplanationResponse(explanation=explanation)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
