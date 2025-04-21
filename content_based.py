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
        
        # Define related technology mapping for better recommendations
        self.tech_relationships = {
            'java': ['spring', 'hibernate', 'j2ee', 'servlet', 'jsp', 'jdbc', 'jpa', 'maven', 'gradle', 'junit', 'jvm', 'backend', 'enterprise','microservices','webflux'],
            'python': ['django', 'flask', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch', 'data science', 'machine learning', 'ai'],
            'javascript': ['typescript', 'react', 'angular', 'vue', 'node', 'express', 'frontend', 'web development', 'dom', 'npm'],
            'c#': ['asp.net', '.net', 'dotnet', 'unity', 'xamarin', 'windows', 'microsoft'],
            'php': ['laravel', 'symfony', 'wordpress', 'codeigniter', 'web development'],
            'ruby': ['rails', 'sinatra', 'web development'],
            'frontend': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'responsive', 'web design'],
            'backend': ['api', 'server', 'database', 'rest', 'microservices', 'authentication'],
            'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'oracle', 'data modeling'],
            'devops': ['docker', 'kubernetes', 'jenkins', 'ci/cd', 'aws', 'azure', 'gcp', 'cloud'],
            'mobile': ['android', 'ios', 'swift', 'kotlin', 'react native', 'flutter'],
        }
        
        # Topic weights for boosting relevance
        self.topic_weights = {
            'programming language': 2.0,
            'framework': 1.5,
            'tool': 1.3,
            'concept': 1.2
        }
    
    def preprocess_data(self):
        """Load and preprocess data"""
        self.courses_df = self.data_loader.load_courses()
        
        if self.courses_df.empty:
            return False
        
        # Create content features for each course
        self.courses_df['content_features'] = ''
        self.courses_df['main_topics'] = ''
        
        # Combine name, categories, tags, level, description, benefits, prerequisites, and course content for feature extraction
        for idx, row in self.courses_df.iterrows():
            content = []
            main_topics = set()
            
            # Process course name with extra weight
            if 'name' in row and row['name']:
                course_name = str(row['name']).lower()
                content.append(str(row['name']) + ' ' + str(row['name']))  # Add twice for more weight
                
                # Identify main technology topics in the course name
                for tech, related in self.tech_relationships.items():
                    if tech in course_name:
                        main_topics.add(tech)
                    for rel_tech in related:
                        if rel_tech in course_name:
                            main_topics.add(tech)
                            main_topics.add(rel_tech)
            
            if 'categories' in row and row['categories']:
                categories = str(row['categories']).lower()
                content.append(str(row['categories']))
                
                # Extract main topics from categories
                for tech in self.tech_relationships:
                    if tech in categories:
                        main_topics.add(tech)
            
            if 'tags' in row and row['tags']:
                tags = str(row['tags']).lower()
                content.append(str(row['tags']))
                
                # Extract main topics from tags
                for tech in self.tech_relationships:
                    if tech in tags:
                        main_topics.add(tech)
                        
                    # Check for related technologies
                    for rel_tech in self.tech_relationships.get(tech, []):
                        if rel_tech in tags:
                            main_topics.add(tech)
                            main_topics.add(rel_tech)
            
            if 'level' in row and row['level']:
                content.append(str(row['level']))
                
            if 'description' in row and row['description']:
                description = str(row['description']).lower()
                content.append(str(row['description']))
                
                # Extract main topics from description
                for tech in self.tech_relationships:
                    if tech in description:
                        main_topics.add(tech)
            
            # Add benefits
            if 'benefits' in row and isinstance(row['benefits'], list):
                benefits_text = ' '.join([str(benefit.get('title', '')) for benefit in row['benefits'] if isinstance(benefit, dict) and 'title' in benefit])
                if benefits_text:
                    content.append(benefits_text)
            
            # Add prerequisites
            if 'prerequisites' in row and isinstance(row['prerequisites'], list):
                prereq_text = ' '.join([str(prereq.get('title', '')) for prereq in row['prerequisites'] if isinstance(prereq, dict) and 'title' in prereq])
                if prereq_text:
                    content.append(prereq_text)
                    
                    # Extract main topics from prerequisites
                    prereq_lower = prereq_text.lower()
                    for tech in self.tech_relationships:
                        if tech in prereq_lower:
                            main_topics.add(tech)
            
            # Add course data details
            if 'courseData' in row and isinstance(row['courseData'], list):
                lesson_text = []
                for lesson in row['courseData']:
                    if isinstance(lesson, dict):
                        if 'title' in lesson and lesson['title']:
                            lesson_text.append(str(lesson['title']))
                        if 'description' in lesson and lesson['description']:
                            lesson_text.append(str(lesson['description']))
                        if 'videoSection' in lesson and lesson['videoSection']:
                            lesson_text.append(str(lesson['videoSection']))
                        if 'suggestion' in lesson and lesson['suggestion']:
                            lesson_text.append(str(lesson['suggestion']))
                
                full_lesson_text = ' '.join(lesson_text).lower()
                content.append(' '.join(lesson_text))
                
                # Extract main topics from lesson content
                for tech in self.tech_relationships:
                    if tech in full_lesson_text:
                        main_topics.add(tech)
                        
                    # Check for related technologies in lessons
                    for rel_tech in self.tech_relationships.get(tech, []):
                        if rel_tech in full_lesson_text:
                            main_topics.add(tech)
                            main_topics.add(rel_tech)
            
            # Enhance content features with main topics
            if main_topics:
                # Add main topics with boosted weight
                for topic in main_topics:
                    # Add the topic multiple times to increase its weight
                    content.append(topic + ' ' + topic + ' ' + topic)
                    
                    # Add related technologies
                    for related in self.tech_relationships.get(topic, []):
                        content.append(related)
            
            self.courses_df.at[idx, 'content_features'] = ' '.join(content)
            self.courses_df.at[idx, 'main_topics'] = ','.join(main_topics)
        
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
        
        # Get the source course main topics
        source_course = self.courses_df.iloc[idx]
        source_topics = source_course.get('main_topics', '').split(',') if 'main_topics' in source_course else []
        
        # Get similarity scores for the course
        similarity_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Apply topic-based boosting to similarity scores
        if source_topics:
            for i, (course_idx, score) in enumerate(similarity_scores):
                if course_idx != idx:  # Skip the source course
                    target_course = self.courses_df.iloc[course_idx]
                    target_topics = target_course.get('main_topics', '').split(',') if 'main_topics' in target_course else []
                    
                    # Boost score for courses with shared topics
                    topic_boost = 1.0
                    for topic in source_topics:
                        if topic and topic in target_topics:
                            topic_boost *= 1.3  # 30% boost for each shared main topic
                    
                    # Apply the boost
                    similarity_scores[i] = (course_idx, score * topic_boost)
        
        # Sort courses by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar courses (excluding itself)
        similar_course_indices = [i[0] for i in similarity_scores[1:n_recommendations+1]]
        
        # Return recommended course details
        recommended_courses = self.courses_df.iloc[similar_course_indices].to_dict('records')
        
        # Add similarity score and topic match info to recommendations
        for i, idx in enumerate(similar_course_indices):
            recommended_courses[i]['similarity_score'] = similarity_scores[i+1][1]
            
            # Add information about matching topics
            source_topics_set = set(source_topics)
            target_topics = self.courses_df.iloc[idx].get('main_topics', '').split(',')
            matching_topics = source_topics_set.intersection(set(target_topics))
            recommended_courses[i]['matching_topics'] = list(matching_topics)
            
        return recommended_courses
    
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
        course_topics = {}
        
        for course_id in purchased_courses:
            if course_id in self.course_indices.index:
                similar_courses = self.recommend_similar_courses(course_id, 10)  # Get more recommendations per course
                
                # Get source course main topics
                source_idx = self.course_indices[course_id]
                source_course = self.courses_df.iloc[source_idx]
                source_topics = source_course.get('main_topics', '').split(',') if 'main_topics' in source_course else []
                
                for similar_course in similar_courses:
                    similar_id = similar_course['_id']
                    if similar_id not in purchased_courses:
                        if similar_id not in course_scores:
                            course_scores[similar_id] = 0
                            course_topics[similar_id] = set()
                        
                        # Add similarity score
                        similarity = similar_course['similarity_score']
                        
                        # Apply rating as a weight if available
                        rating_weight = 1.0
                        course_row = self.courses_df[self.courses_df['_id'] == similar_id]
                        if not course_row.empty and 'ratings' in course_row.iloc[0] and course_row.iloc[0]['ratings']:
                            rating = float(course_row.iloc[0]['ratings'])
                            rating_weight = 1.0 + (rating / 5.0) * 0.5  # Ratings boost up to 50%
                        
                        # Topic match boost
                        topic_weight = 1.0
                        if 'matching_topics' in similar_course and similar_course['matching_topics']:
                            course_topics[similar_id].update(similar_course['matching_topics'])
                            topic_weight = 1.0 + (len(similar_course['matching_topics']) * 0.2)  # 20% boost per matching topic
                        
                        course_scores[similar_id] += similarity * rating_weight * topic_weight
        
        # Sort courses by score
        sorted_courses = sorted(course_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get top n recommendations
        top_recommendations = [course_id for course_id, _ in sorted_courses[:n_recommendations]]
        
        # Get course details
        recommended_courses = self.courses_df[self.courses_df['_id'].isin(top_recommendations)].copy()
        
        # Add matching topics information to recommendations
        for i, course_id in enumerate(top_recommendations):
            if i < len(recommended_courses) and course_id in course_topics:
                idx = recommended_courses[recommended_courses['_id'] == course_id].index[0]
                recommended_courses.at[idx, 'matching_topics'] = list(course_topics[course_id])
                recommended_courses.at[idx, 'recommendation_score'] = course_scores[course_id]
                
        return recommended_courses.to_dict('records')
    
    def close(self):
        """Close data loader connection"""
        self.data_loader.close() 