from data_loader import DataLoader

def main():
    print("Checking database connection and data...")
    data_loader = DataLoader()
    
    # Check connection
    is_connected, message = data_loader.check_connection()
    print(f"\n1. Connection Status: {message}")
    
    if not is_connected:
        print("Please make sure MongoDB is running and the connection string is correct")
        return
    
    # Check courses collection
    try:
        courses = data_loader.load_courses()
        print(f"\n2. Courses Collection:")
        print(f"Number of courses: {len(courses)}")
        if not courses.empty:
            print("Sample course data:")
            print(courses.head())
        else:
            print("No courses found in the database")
    except Exception as e:
        print(f"Error loading courses: {str(e)}")
    
    # Check users collection
    try:
        users = data_loader.load_users()
        print(f"\n3. Users Collection:")
        print(f"Number of users: {len(users)}")
        if not users.empty:
            print("Sample user data:")
            print(users.head())
        else:
            print("No users found in the database")
    except Exception as e:
        print(f"Error loading users: {str(e)}")
    
    data_loader.close()

if __name__ == "__main__":
    main() 