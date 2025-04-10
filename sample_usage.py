#!/usr/bin/env python3
"""
Sample script to demonstrate the usage of the recommender system
"""

import os
import json
from dotenv import load_dotenv
from hybrid_recommender import HybridRecommender
from data_loader import DataLoader

# Load environment variables
load_dotenv()

def print_course_info(course):
    """Print formatted course information"""
    print(f"ID: {course['_id']}")
    print(f"Name: {course['name']}")
    print(f"Level: {course.get('level', 'N/A')}")
    print(f"Categories: {course.get('categories', 'N/A')}")
    print(f"Tags: {course.get('tags', 'N/A')}")
    print(f"Ratings: {course.get('ratings', 'N/A')}")
    print(f"Purchased: {course.get('purchased', 'N/A')}")
    print("-" * 50)

def main():
    """Main function to demonstrate the recommender system"""
    print("\n===== LMS RECOMMENDER SYSTEM DEMO =====\n")
    
    # Initialize data loader to get sample data
    data_loader = DataLoader()
    
    # Get a sample user and course
    users_df = data_loader.load_users()
    courses_df = data_loader.load_courses()
    
    if users_df.empty or courses_df.empty:
        print("Error: No users or courses found in the database.")
        return
    
    # Get a sample user and course
    sample_user_id = users_df.iloc[0]['_id']
    sample_course_id = courses_df.iloc[0]['_id']
    
    print(f"Sample User ID: {sample_user_id}")
    print(f"Sample Course ID: {sample_course_id}")
    print("\n")
    
    # Initialize the hybrid recommender
    recommender = HybridRecommender()
    
    try:
        # 1. Get personalized recommendations for the user
        print("===== PERSONALIZED RECOMMENDATIONS =====")
        user_recommendations = recommender.recommend(sample_user_id, 3)
        
        if user_recommendations:
            print(f"\nTop 3 recommended courses for user {sample_user_id}:\n")
            for course in user_recommendations:
                print_course_info(course)
        else:
            print("No personalized recommendations found.")
        
        # 2. Get similar courses recommendation
        print("\n===== SIMILAR COURSES =====")
        similar_courses = recommender.recommend_similar_to_course(sample_course_id, 3)
        
        if similar_courses:
            print(f"\nTop 3 courses similar to {courses_df[courses_df['_id'] == sample_course_id].iloc[0]['name']}:\n")
            for course in similar_courses:
                print_course_info(course)
        else:
            print("No similar courses found.")
        
        # 3. Get popular courses
        print("\n===== POPULAR COURSES =====")
        popular_courses = recommender.recommend_popular_courses(3)
        
        if popular_courses:
            print("\nTop 3 popular courses:\n")
            for course in popular_courses:
                print_course_info(course)
        else:
            print("No popular courses found.")
            
    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
    finally:
        # Close connections
        recommender.close()
        data_loader.close()
    
    print("\n===== DEMO COMPLETED =====\n")

if __name__ == "__main__":
    main() 