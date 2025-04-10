# LMS Recommender System

This Python-based recommender system is designed to provide personalized course recommendations for the LMS (Learning Management System) platform.

## Features

- **Collaborative Filtering**: Recommends courses based on user behavior and preferences
  - User-based: Recommends courses based on similar users' preferences
  - Item-based: Recommends courses similar to ones the user has interacted with

- **Content-Based Filtering**: Recommends courses based on course features and metadata
  - Uses course categories, tags, level, and description for similarity calculation

- **Hybrid Approach**: Combines collaborative and content-based approaches for better recommendations

- **API Endpoints**: RESTful API for easy integration with the main application

## Setup

1. Make sure you have Python 3.8+ installed

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in a `.env` file:
   ```
   MONGODB_URI=mongodb://username:password@host:port/database
   PORT=8000
   ```

## Usage

### Running the API server

```
python api.py
```

This will start the FastAPI server on the specified port (default: 8000).

### API Endpoints

1. **Get personalized recommendations for a user**
   ```
   GET /recommend/user/{user_id}?limit=5
   ```

2. **Get courses similar to a specific course**
   ```
   GET /recommend/similar/{course_id}?limit=5
   ```

3. **Get popular courses**
   ```
   GET /recommend/popular?limit=5
   ```

## Integration with Node.js Server

To integrate the recommender system with the main Node.js application:

1. Add the following route to your Express server:

```javascript
// routes/recommender.route.ts
import express from "express";
import axios from "axios";
import { isAuthenticated } from "../middleware/auth";

const router = express.Router();

// Recommender API base URL
const RECOMMENDER_API_URL = process.env.RECOMMENDER_API_URL || "http://localhost:8000";

// Get personalized recommendations for the logged-in user
router.get("/recommendations", isAuthenticated, async (req, res) => {
  try {
    const userId = req.user?._id.toString();
    const limit = req.query.limit || 5;
    
    const response = await axios.get(`${RECOMMENDER_API_URL}/recommend/user/${userId}?limit=${limit}`);
    
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching recommendations:", error);
    res.status(500).json({ success: false, message: "Failed to fetch recommendations" });
  }
});

// Get similar courses
router.get("/similar/:courseId", async (req, res) => {
  try {
    const { courseId } = req.params;
    const limit = req.query.limit || 5;
    
    const response = await axios.get(`${RECOMMENDER_API_URL}/recommend/similar/${courseId}?limit=${limit}`);
    
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching similar courses:", error);
    res.status(500).json({ success: false, message: "Failed to fetch similar courses" });
  }
});

// Get popular courses
router.get("/popular", async (req, res) => {
  try {
    const limit = req.query.limit || 5;
    
    const response = await axios.get(`${RECOMMENDER_API_URL}/recommend/popular?limit=${limit}`);
    
    res.status(200).json(response.data);
  } catch (error) {
    console.error("Error fetching popular courses:", error);
    res.status(500).json({ success: false, message: "Failed to fetch popular courses" });
  }
});

export default router;
```

2. Register this router in your app.ts file:

```typescript
// Import the router
import recommenderRouter from "./routes/recommender.route";

// Add the route
app.use("/api/v1/recommender", recommenderRouter);
```

3. Make sure to install axios if not already installed:
```
npm install axios
```

## How It Works

The recommender system uses multiple approaches to generate recommendations:

1. **Collaborative Filtering**:
   - Creates a user-item interaction matrix based on course purchases and progress
   - Uses cosine similarity to find similar users or items
   - Generates recommendations based on what similar users liked or what's similar to items the user liked

2. **Content-Based Filtering**:
   - Uses TF-IDF to vectorize course features (categories, tags, level, description)
   - Calculates cosine similarity between courses
   - Recommends courses with similar content features to what the user has already liked

3. **Hybrid Approach**:
   - Combines recommendations from both methods with weighted scores
   - Allows for more diverse and relevant recommendations 