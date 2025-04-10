import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import DataLoader

class ContentBasedRecommender:
    def __init__(self):
        self.data_loader = DataLoader()
        self.courses_df = None
        self.tfidf_matrix = None
        self.course_indices = None
        self.similarity_matrix = None
    
    def preprocess_data(self):
        """Load and preprocess data"""
        self.courses_df = self.data_loader.load_courses()
        
        if self.courses_df.empty:
            return False
        
        # Create content features for each course
        self.courses_df['content_features'] = ''
        
        # Combine categories, tags, level, description for feature extraction
        for idx, row in self.courses_df.iterrows():
            content = []
            if 'categories' in row and row['categories']:
                content.append(str(row['categories']))
            if 'tags' in row and row['tags']:
                content.append(str(row['tags']))
            if 'level' in row and row['level']:
                content.append(str(row['level']))
            if 'description' in row and row['description']:
                content.append(str(row['description']))
                
            self.courses_df.at[idx, 'content_features'] = ' '.join(content)
        
        return True
    
    def train(self):
        """Train the content-based recommender"""
        if self.courses_df is None:
            if not self.preprocess_data():
                return False
        
        # Create TF-IDF matrix for course features
        tfidf = TfidfVectorizer(stop_words='english')
        
        try:
            self.tfidf_matrix = tfidf.fit_transform(self.courses_df['content_features'])
        except:
            return False
        
        # Calculate cosine similarity between courses
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        # Create course indices mapping for faster lookup
        self.course_indices = pd.Series(self.courses_df.index, index=self.courses_df['_id']).drop_duplicates()
        
        return True
    
    def recommend_similar_courses(self, course_id, n_recommendations=5):
        """Recommend courses similar to a given course"""
        if self.similarity_matrix is None:
            self.train()
            
        if course_id not in self.course_indices.index:
            return []
            
        # Get course index
        idx = self.course_indices[course_id]
        
        # Get similarity scores for the course
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Sort courses by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar courses (excluding itself)
        similar_course_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]
        
        # Return recommended course details
        return self.courses_df.iloc[similar_course_indices].to_dict('records')
    
    def recommend_for_user(self, user_id, n_recommendations=5):
        """Recommend courses for a user based on their previous purchases"""
        if self.similarity_matrix is None:
            self.train()
        
        # Get user data
        users_df = self.data_loader.load_users()
        
        if users_df.empty:
            return []
        
        user_data = users_df[users_df['_id'] == user_id]
        
        if user_data.empty:
            return []
        
        # Get user's purchased courses
        purchased_courses = []
        user_courses = user_data.iloc[0].get('courses', [])
        
        if user_courses:
            for course in user_courses:
                if 'courseId' in course:
                    purchased_courses.append(str(course['courseId']))
        
        if not purchased_courses:
            return []
        
        # Calculate recommendation scores
        course_scores = {}
        
        for course_id in purchased_courses:
            if course_id in self.course_indices.index:
                similar_courses = self.recommend_similar_courses(course_id, 10)  # Get more recommendations per course
                
                for similar_course in similar_courses:
                    similar_id = similar_course['_id']
                    if similar_id not in purchased_courses:
                        if similar_id not in course_scores:
                            course_scores[similar_id] = 0
                        
                        # Add similarity score
                        idx = self.course_indices[course_id]
                        similar_idx = self.course_indices[similar_id]
                        similarity = self.similarity_matrix[idx][similar_idx]
                        course_scores[similar_id] += similarity
        
        # Sort courses by score
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top n recommendations
        top_recommendations = [course_id for course_id, _ in sorted_courses[:n_recommendations]]
        
        # Get course details
        recommended_courses = self.courses_df[self.courses_df['_id'].isin(top_recommendations)]
        return recommended_courses.to_dict('records')
    
    def close(self):
        """Close data loader connection"""
        self.data_loader.close() 