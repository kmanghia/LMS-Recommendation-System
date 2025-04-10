import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataLoader:
    def __init__(self):
        # Connect to MongoDB
        mongo_uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/trannghia')
        self.client = MongoClient(mongo_uri)
        self.db = self.client.get_database()
    
    def check_connection(self):
        """Check if the connection to MongoDB is successful"""
        try:
            # Try to get server info
            self.client.server_info()
            return True, "Successfully connected to MongoDB"
        except Exception as e:
            return False, f"Failed to connect to MongoDB: {str(e)}"
    
    def load_courses(self):
        """Load courses from the database"""
        courses = list(self.db.courses.find({}))
        courses_df = pd.DataFrame(courses)
        
        # Extract relevant features
        if not courses_df.empty:
            courses_df = courses_df[['_id', 'name', 'description', 'categories', 'tags', 'level', 'ratings', 'purchased']]
            courses_df['_id'] = courses_df['_id'].astype(str)
        
        return courses_df
    
    def load_users(self):
        """Load users from the database"""
        users = list(self.db.users.find({}))
        users_df = pd.DataFrame(users)
        
        # Extract relevant features
        if not users_df.empty:
            users_df = users_df[['_id', 'name', 'email', 'courses', 'progress']]
            users_df['_id'] = users_df['_id'].astype(str)
        
        return users_df
    
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        users_df = self.load_users()
        courses_df = self.load_courses()
        
        # Initialize the matrix
        interactions = []
        
        if not users_df.empty and not courses_df.empty:
            for _, user in users_df.iterrows():
                user_id = str(user['_id'])
                
                # Extract purchased courses
                purchased_courses = []
                if 'courses' in user and user['courses'] is not None:
                    for course in user['courses']:
                        if 'courseId' in course:
                            purchased_courses.append(str(course['courseId']))
                
                # Extract progress data
                progress_data = {}
                if 'progress' in user and user['progress'] is not None:
                    for progress in user['progress']:
                        if 'courseId' in progress and 'chapters' in progress:
                            course_id = str(progress['courseId'])
                            completed_chapters = sum(1 for chapter in progress['chapters'] if chapter.get('isCompleted', False))
                            total_chapters = len(progress['chapters'])
                            
                            if total_chapters > 0:
                                progress_data[course_id] = completed_chapters / total_chapters
                
                # Create interaction records
                for course_id in courses_df['_id']:
                    interaction = {
                        'user_id': user_id,
                        'course_id': course_id,
                        'purchased': 1 if course_id in purchased_courses else 0,
                        'progress': progress_data.get(course_id, 0)
                    }
                    interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client:
            self.client.close() 