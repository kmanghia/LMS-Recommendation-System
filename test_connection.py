from data_loader import DataLoader

def main():
    print("Testing database connection...")
    data_loader = DataLoader()
    
    is_connected, message = data_loader.check_connection()
    print(message)
    
    if is_connected:
        print("Database connection is working properly!")
    else:
        print("Failed to connect to database. Please check your connection settings.")

if __name__ == "__main__":
    main() 