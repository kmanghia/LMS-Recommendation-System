import numpy as np
import pandas as pd
from collaborative_filtering import CollaborativeFilteringRecommender
from content_based import ContentBasedRecommender

class HybridRecommender:
    def __init__(self, collab_weight=0.6, content_weight=0.4):
        """Initialize hybrid recommender with weights for each approach"""
        self.collaborative_recommender = CollaborativeFilteringRecommender()
        self.content_recommender = ContentBasedRecommender()
        self.collab_weight = collab_weight
        self.content_weight = content_weight
        
        # Initialize data
        self.collaborative_recommender.preprocess_data()
    
    def recommend(self, user_id, n_recommendations=5):
        """Generate hybrid recommendations for a user"""
        # Get collaborative filtering recommendations
        collab_item_recommendations = self.collaborative_recommender.recommend_item_based(user_id, n_recommendations*2)
        collab_user_recommendations = self.collaborative_recommender.recommend_user_based(user_id, n_recommendations*2)
        
        # Get content-based recommendations
        content_recommendations = self.content_recommender.recommend_for_user(user_id, n_recommendations*2)
        
        # Combine recommendations with weights
        all_recommendations = {}
        
        # Process collaborative filtering (item-based) recommendations
        for i, course in enumerate(collab_item_recommendations):
            course_id = course['_id']
            if course_id not in all_recommendations:
                all_recommendations[course_id] = 0
            # Assign score based on position and weight
            score = self.collab_weight * (1.0 - (i * 0.1 if i < 10 else 0.9))
            all_recommendations[course_id] += score
        
        # Process collaborative filtering (user-based) recommendations
        for i, course in enumerate(collab_user_recommendations):
            course_id = course['_id']
            if course_id not in all_recommendations:
                all_recommendations[course_id] = 0
            # Assign score based on position and weight
            score = self.collab_weight * 0.8 * (1.0 - (i * 0.1 if i < 10 else 0.9))
            all_recommendations[course_id] += score
        
        # Process content-based recommendations
        for i, course in enumerate(content_recommendations):
            course_id = course['_id']
            if course_id not in all_recommendations:
                all_recommendations[course_id] = 0
            # Assign score based on position and weight
            score = self.content_weight * (1.0 - (i * 0.1 if i < 10 else 0.9))
            all_recommendations[course_id] += score
        
        # Sort recommendations by score
        sorted_recommendations = sorted(all_recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get top n course IDs
        top_course_ids = [course_id for course_id, _ in sorted_recommendations[:n_recommendations]]
        
        # Get course details
        courses_df = self.collaborative_recommender.courses_df
        recommended_courses = courses_df[courses_df['_id'].isin(top_course_ids)].to_dict('records')
        
        return recommended_courses
    
    def recommend_similar_to_course(self, course_id, n_recommendations=5):
        """Recommend courses similar to a given course"""
        return self.content_recommender.recommend_similar_courses(course_id, n_recommendations)
    
    def recommend_popular_courses(self, n_recommendations=5):
        """Recommend popular courses based on ratings and purchase count"""
        courses_df = self.collaborative_recommender.courses_df
        
        if 'ratings' in courses_df.columns and 'purchased' in courses_df.columns:
            # Create a popularity score based on ratings and purchase count
            courses_df['popularity_score'] = (
                courses_df['ratings'].fillna(0) * 0.7 + 
                courses_df['purchased'].fillna(0) * 0.3
            )
            
            # Sort by popularity score and get top n
            popular_courses = courses_df.sort_values(by='popularity_score', ascending=False).head(n_recommendations)
            return popular_courses.to_dict('records')
        
        return []
    
    def close(self):
        """Close recommender connections"""
        self.collaborative_recommender.close()
        self.content_recommender.close() 