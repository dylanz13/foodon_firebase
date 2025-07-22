# app.py - Main FastAPI application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import asyncio
from contextlib import asynccontextmanager

# Import your existing pipeline
from langgraph_pipeline import analyze_ingredients_for_user, update_user_dietary_preferences

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lifespan event handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Ingredient Analysis API")
    yield
    # Shutdown
    logger.info("🛑 Shutting down Ingredient Analysis API")

# Create FastAPI app
app = FastAPI(
    title="Ingredient Analysis API",
    description="AI-powered ingredient analysis with dietary safety checking",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class IngredientAnalysisRequest(BaseModel):
    raw_text: str = Field(..., description="Raw menu text or dish description")
    user_id: Optional[str] = Field(None, description="User ID for personalized analysis")
    dish_name: Optional[str] = Field(None, description="Name of the dish")

class UserPreferencesRequest(BaseModel):
    user_id: str = Field(..., description="User ID")
    allergens: Optional[List[str]] = Field(None, description="List of allergens")
    dietary_restrictions: Optional[List[str]] = Field(None, description="List of dietary restrictions")
    custom_restrictions: Optional[Dict[str, List[str]]] = Field(None, description="Custom restrictions")

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str

# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ingredient Analysis API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    from datetime import datetime
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_ingredients(request: IngredientAnalysisRequest):
    """
    Analyze ingredients from menu text with optional user safety checking.
    """
    import time
    start_time = time.time()
    
    try:
        # Validate input
        if not request.raw_text.strip():
            raise HTTPException(status_code=400, detail="raw_text cannot be empty")
        
        logger.info(f"Processing analysis request for user: {request.user_id}")
        
        # Run the analysis pipeline
        result = await analyze_ingredients_for_user(
            raw_text=request.raw_text,
            user_id=request.user_id,
            dish_name=request.dish_name
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        
        return AnalysisResponse(
            success=True,
            data=result,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = f"Analysis failed: {str(e)}"
        logger.error(error_msg)
        
        return AnalysisResponse(
            success=False,
            error=error_msg,
            processing_time=processing_time
        )

@app.post("/users/{user_id}/preferences")
async def update_user_preferences(user_id: str, request: UserPreferencesRequest):
    """
    Update user dietary preferences.
    """
    try:
        if user_id != request.user_id:
            raise HTTPException(status_code=400, detail="User ID mismatch")
        
        success = await update_user_dietary_preferences(
            user_id=request.user_id,
            allergens=request.allergens,
            dietary_restrictions=request.dietary_restrictions,
            custom_restrictions=request.custom_restrictions
        )
        
        if success:
            return {"success": True, "message": "Preferences updated successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to update preferences")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update preferences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """
    Get user dietary profile.
    """
    try:
        from ingredient_analysis_pipeline import get_firebase_db
        
        db = get_firebase_db()
        if not db:
            raise HTTPException(status_code=503, detail="Database unavailable")
        
        user_doc = db.collection("user_profiles").document(user_id).get()
        
        if user_doc.exists:
            profile_data = user_doc.to_dict()
            return {
                "success": True,
                "data": {
                    "user_id": user_id,
                    "allergens": profile_data.get("allergens", []),
                    "dietary_restrictions": profile_data.get("dietary_restrictions", []),
                    "custom_restrictions": profile_data.get("custom_restrictions", {})
                }
            }
        else:
            raise HTTPException(status_code=404, detail="User profile not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user profile: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background task for batch processing
@app.post("/analyze/batch")
async def analyze_batch(requests: List[IngredientAnalysisRequest], background_tasks: BackgroundTasks):
    """
    Process multiple ingredient analyses in the background.
    """
    try:
        if len(requests) > 10:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum batch size is 10")
        
        # Start background processing
        task_id = f"batch_{len(requests)}_{int(time.time())}"
        background_tasks.add_task(process_batch, requests, task_id)
        
        return {
            "success": True,
            "message": "Batch processing started",
            "task_id": task_id,
            "batch_size": len(requests)
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_batch(requests: List[IngredientAnalysisRequest], task_id: str):
    """Background task for processing batches."""
    logger.info(f"Processing batch {task_id} with {len(requests)} items")
    
    for i, request in enumerate(requests):
        try:
            result = await analyze_ingredients_for_user(
                raw_text=request.raw_text,
                user_id=request.user_id,
                dish_name=request.dish_name
            )
            logger.info(f"Batch {task_id}: Completed item {i+1}/{len(requests)}")
        except Exception as e:
            logger.error(f"Batch {task_id}: Failed item {i+1}: {e}")
    
    logger.info(f"Batch {task_id} processing completed")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return {
        "success": False,
        "error": "Internal server error",
        "detail": str(exc) if app.debug else "An unexpected error occurred"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
