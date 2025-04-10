import os
from fastapi import FastAPI, HTTPException, Depends # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from pydantic import BaseModel # type: ignore
from typing import List, Optional
import uvicorn # type: ignore
from hybrid_recommender import HybridRecommender

app = FastAPI(title="LMS Recommender API", description="API for course recommendations")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency to get recommender
def get_recommender():
    recommender = HybridRecommender()
    try:
        yield recommender
    finally:
        recommender.close()

# Response models
class CourseBase(BaseModel):
    _id: str
    name: str
    description: str
    categories: Optional[str] = None
    tags: Optional[str] = None
    level: Optional[str] = None
    ratings: Optional[float] = None
    purchased: Optional[int] = None

class RecommendationResponse(BaseModel):
    recommendations: List[CourseBase]

# API routes
@app.get("/")
def read_root():
    return {"message": "LMS Recommender API is running"}

@app.get("/recommend/user/{user_id}", response_model=RecommendationResponse)
def recommend_for_user(user_id: str, limit: int = 5, recommender: HybridRecommender = Depends(get_recommender)):
    """Get personalized course recommendations for a user"""
    try:
        recommendations = recommender.recommend(user_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/similar/{course_id}", response_model=RecommendationResponse)
def recommend_similar(course_id: str, limit: int = 5, recommender: HybridRecommender = Depends(get_recommender)):
    """Get courses similar to a specified course"""
    try:
        recommendations = recommender.recommend_similar_to_course(course_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend/popular", response_model=RecommendationResponse)
def recommend_popular(limit: int = 5, recommender: HybridRecommender = Depends(get_recommender)):
    """Get popular courses based on ratings and purchases"""
    try:
        recommendations = recommender.recommend_popular_courses(limit)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Run the API server
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 