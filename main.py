from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from postgres_recommender import PostgreSQLContentBasedRecommender   
from mlops_manager import MLOpsManager
load_dotenv()  # Load .env file


# Pydantic models for API
class RecommendationRequest(BaseModel):
    article_id: str = Field(..., description="ID artikel untuk mendapatkan rekomendasi")
    top_n: int = Field(default=5, ge=1, le=20, description="Jumlah rekomendasi yang diinginkan")
    include_metadata: bool = Field(default=True, description="Sertakan metadata artikel")

class RecommendationResponse(BaseModel):
    article_id: str
    recommendations: List[Dict[str, Any]]
    total_found: int
    generated_at: datetime
    model_version: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    last_training: Optional[str]
    data_freshness: Optional[str]
    uptime_seconds: float

class ModelStatusResponse(BaseModel):
    model_loaded: bool
    last_training: Optional[str]
    total_articles: int
    model_performance: Dict[str, Any]
    mlops_status: Dict[str, Any]

class BatchRecommendationRequest(BaseModel):
    article_ids: List[str] = Field(..., max_items=50, description="List artikel IDs (max 50)")
    top_n: int = Field(default=5, ge=1, le=10)

# Global variables
recommender = None
mlops_manager = None
app_start_time = datetime.now()

# Database configuration (should be loaded from environment)


DB_CONFIG = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': int(os.getenv('DB_PORT', 5432))  # default ke 5432
}
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global recommender, mlops_manager
    
    # Startup
    try:
        logging.info("Loading recommender model...")
        recommender = PostgreSQLContentBasedRecommender(DB_CONFIG)
        
        # Try to load existing model, otherwise train new one
        try:
            recommender.load_model()
            logging.info("Existing model loaded successfully")
        except:
            logging.info("No existing model found, training new model...")
            recommender.fit()
        
        # Initialize MLOps manager
        mlops_manager = MLOpsManager(DB_CONFIG, recommender)
        mlops_manager.start_monitoring()
        
        logging.info("Recommender API is ready!")
        
    except Exception as e:
        logging.error(f"Failed to initialize recommender: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("Shutting down recommender API...")

# Initialize FastAPI app
app = FastAPI(
    title="Content-Based Recommender API",
    description="API untuk sistem rekomendasi artikel berbasis konten dengan MLOps",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure according to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dependency to ensure model is loaded
async def get_recommender():
    if recommender is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return recommender

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Content-Based Recommender API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - app_start_time).total_seconds()
    
    model_loaded = recommender is not None and recommender.model is not None
    last_training = None
    
    if model_loaded and mlops_manager:
        status = mlops_manager.get_monitoring_status()
        last_training = status.get('last_training')
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        last_training=last_training,
        data_freshness="real-time",
        uptime_seconds=uptime
    )

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    rec: PostgreSQLContentBasedRecommender = Depends(get_recommender)
):
    """Get article recommendations"""
    try:
        # Get recommendations
        recommendations = rec.recommend(
            article_id=request.article_id,
            top_n=request.top_n
        )
        
        if recommendations.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Article with ID '{request.article_id}' not found or no recommendations available"
            )
        
        # Format response
        recs_list = []
        for _, row in recommendations.iterrows():
            rec_item = {
                "id": row['id'],
                "title": row['title'],
                "similarity_score": float(row['similarity_score']),
            }
            
            if request.include_metadata:
                rec_item.update({
                    "province": row.get('province', ''),
                    "city": row.get('city', ''),
                    "engagement_score": float(row.get('engagement_score', 0))
                })
            
            recs_list.append(rec_item)
        
        return RecommendationResponse(
            article_id=request.article_id,
            recommendations=recs_list,
            total_found=len(recs_list),
            generated_at=datetime.now(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend/batch")
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    rec: PostgreSQLContentBasedRecommender = Depends(get_recommender)
):
    """Get recommendations for multiple articles"""
    try:
        batch_results = {}
        
        for article_id in request.article_ids:
            try:
                recommendations = rec.recommend(
                    article_id=article_id,
                    top_n=request.top_n
                )
                
                if not recommendations.empty:
                    recs_list = []
                    for _, row in recommendations.iterrows():
                        recs_list.append({
                            "id": row['id'],
                            "title": row['title'],
                            "similarity_score": float(row['similarity_score'])
                        })
                    batch_results[article_id] = recs_list
                else:
                    batch_results[article_id] = []
                    
            except Exception as e:
                batch_results[article_id] = {"error": str(e)}
        
        return {
            "results": batch_results,
            "processed_count": len(request.article_ids),
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error in batch recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/status", response_model=ModelStatusResponse)
async def get_model_status(rec: PostgreSQLContentBasedRecommender = Depends(get_recommender)):
    """Get model status and performance metrics"""
    try:
        # Get basic model info
        model_loaded = rec.model is not None
        total_articles = len(rec.articles_data) if rec.articles_data is not None else 0
        
        # Get performance metrics
        performance = {}
        if rec.similarity_matrix is not None:
            performance = {
                "avg_similarity": float(rec.similarity_matrix.mean()),
                "max_similarity": float(rec.similarity_matrix.max()),
                "matrix_shape": rec.similarity_matrix.shape
            }
        
        # Get MLOps status
        mlops_status = {}
        if mlops_manager:
            mlops_status = mlops_manager.get_monitoring_status()
        
        return ModelStatusResponse(
            model_loaded=model_loaded,
            last_training=mlops_status.get('last_training'),
            total_articles=total_articles,
            model_performance=performance,
            mlops_status=mlops_status
        )
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/retrain")
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    force: bool = False,
    rec: PostgreSQLContentBasedRecommender = Depends(get_recommender)
):
    """Trigger model retraining"""
    def retrain_model():
        try:
            logger.info("Manual retraining triggered")
            rec.fit(force_retrain=force)
            logger.info("Manual retraining completed")
        except Exception as e:
            logger.error(f"Manual retraining failed: {e}")
    
    background_tasks.add_task(retrain_model)
    
    return {
        "message": "Retraining triggered",
        "force": force,
        "status": "processing",
        "triggered_at": datetime.now()
    }

@app.get("/articles/{article_id}")
async def get_article_info(
    article_id: str,
    rec: PostgreSQLContentBasedRecommender = Depends(get_recommender)
):
    """Get information about a specific article"""
    try:
        if rec.articles_data is None:
            raise HTTPException(status_code=503, detail="Model data not available")
        
        article = rec.articles_data[rec.articles_data['id'] == article_id]
        
        if article.empty:
            raise HTTPException(status_code=404, detail="Article not found")
        
        article_info = article.iloc[0].to_dict()
        
        # Convert numpy types to Python types for JSON serialization
        for key, value in article_info.items():
            if hasattr(value, 'item'):
                article_info[key] = value.item()
        
        return {
            "article": article_info,
            "retrieved_at": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving article info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Get API metrics and statistics"""
    try:
        metrics = {
            "uptime_seconds": (datetime.now() - app_start_time).total_seconds(),
            "model_loaded": recommender is not None and recommender.model is not None,
            "total_articles": len(recommender.articles_data) if recommender and recommender.articles_data is not None else 0,
        }
        
        if mlops_manager:
            mlops_status = mlops_manager.get_monitoring_status()
            metrics.update({
                "data_changes_detected": mlops_status.get('data_changes_detected', 0),
                "last_training": mlops_status.get('last_training'),
                "training_history_count": len(mlops_status.get('recent_training_history', []))
            })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Run the API
    uvicorn.run(
        "inference_api:app",
        host=HOST,
        port=PORT,
        reload=True,  # Set to False in production
        log_level="info"
    )