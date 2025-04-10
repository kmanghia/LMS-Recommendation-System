import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from data_loader import DataLoader

class CollaborativeFilteringRecommender:
    def __init__(self):
        self.data_loader = DataLoader()
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.interaction_matrix = None
        self.courses_df = None
        self.users_df = None
    
    def preprocess_data(self):
        """Load and preprocess data"""
        # Load data
        self.courses_df = self.data_loader.load_courses()
        self.users_df = self.data_loader.load_users()
        interactions_df = self.data_loader.create_user_item_matrix()
        
        if interactions_df.empty:
            return False
        
        # Create a weighted interaction score (purchased + progress)
        interactions_df['interaction_score'] = interactions_df['purchased'] * 5 + interactions_df['progress'] * 10
        
        # Create user-item matrix
        self.interaction_matrix = interactions_df.pivot_table(
            index='user_id', 
            columns='course_id', 
            values='interaction_score',
            fill_value=0
        )
        
        return True
    
    def train_user_based(self):
        """Train user-based collaborative filtering"""
        if self.interaction_matrix is None:
            if not self.preprocess_data():
                return False
        
        # Calculate user similarity
        self.user_similarity_matrix = cosine_similarity(self.interaction_matrix)
        self.user_similarity_matrix = pd.DataFrame(
            self.user_similarity_matrix,
            index=self.interaction_matrix.index,
            columns=self.interaction_matrix.index
        )
        
        return True
    
    def train_item_based(self):
        """Train item-based collaborative filtering"""
        if self.interaction_matrix is None:
            if not self.preprocess_data():
                return False
        
        # Calculate item similarity
        self.item_similarity_matrix = cosine_similarity(self.interaction_matrix.T)
        self.item_similarity_matrix = pd.DataFrame(
            self.item_similarity_matrix,
            index=self.interaction_matrix.columns,
            columns=self.interaction_matrix.columns
        )
        
        return True
    
    def recommend_user_based(self, user_id, n_recommendations=5):
        """Generate user-based recommendations"""
        if self.user_similarity_matrix is None:
            self.train_user_based()
        
        if user_id not in self.interaction_matrix.index:
            return []
        
        # Find similar users
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False).index[1:11]  # Top 10 similar users
        
        # Courses that the user has already interacted with
        user_interactions = self.interaction_matrix.loc[user_id]
        user_courses = user_interactions[user_interactions > 0].index
        
        # Collect recommendations from similar users
        recommendations = {}
        for similar_user in similar_users:
            similar_user_interactions = self.interaction_matrix.loc[similar_user]
            
            # Consider only courses that similar user has interacted with but target user hasn't
            for course in similar_user_interactions.index:
                if course not in user_courses and similar_user_interactions[course] > 0:
                    if course not in recommendations:
                        recommendations[course] = 0
                    
                    # Weight by similarity
                    similarity = self.user_similarity_matrix.loc[user_id, similar_user]
                    recommendations[course] += similarity * similar_user_interactions[course]
        
        # Sort recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n recommendations
        top_recommendations = [course_id for course_id, _ in sorted_recommendations[:n_recommendations]]
        
        # Get course details
        if not self.courses_df.empty:
            recommended_courses = self.courses_df[self.courses_df['_id'].isin(top_recommendations)]
            return recommended_courses.to_dict('records')
        
        return top_recommendations
    
    def recommend_item_based(self, user_id, n_recommendations=5):
        """Generate item-based recommendations"""
        if self.item_similarity_matrix is None:
            self.train_item_based()
        
        if user_id not in self.interaction_matrix.index:
            return []
        
        # Courses that the user has already interacted with
        user_interactions = self.interaction_matrix.loc[user_id]
        user_courses = user_interactions[user_interactions > 0].index
        
        if len(user_courses) == 0:
            return []
        
        # Initialize recommendation scores
        recommendations = {}
        
        # For each course the user has interacted with
        for course in user_courses:
            # Get similar courses
            similar_courses = self.item_similarity_matrix[course].sort_values(ascending=False).index[1:11]  # Top 10 similar courses
            
            for similar_course in similar_courses:
                if similar_course not in user_courses:
                    if similar_course not in recommendations:
                        recommendations[similar_course] = 0
                    
                    # Weight by similarity and user's interaction score
                    similarity = self.item_similarity_matrix.loc[course, similar_course]
                    interaction_score = user_interactions[course]
                    recommendations[similar_course] += similarity * interaction_score
        
        # Sort recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Return top n recommendations
        top_recommendations = [course_id for course_id, _ in sorted_recommendations[:n_recommendations]]
        
        # Get course details
        if not self.courses_df.empty:
            recommended_courses = self.courses_df[self.courses_df['_id'].isin(top_recommendations)]
            return recommended_courses.to_dict('records')
        
        return top_recommendations
    
    def close(self):
        """Close data loader connection"""
        self.data_loader.close() 