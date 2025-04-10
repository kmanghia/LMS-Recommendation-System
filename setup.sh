#!/bin/bash

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source venv/Scripts/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "Creating .env file from example..."
  cp .env.example .env
  echo "Please update the .env file with your MongoDB connection string."
fi

# Install Node.js dependency for the Express server
echo "Installing axios in the Node.js project..."
cd ..
npm install axios --save

echo "Setup completed successfully!"
echo "To start the recommender system API, run:"
echo "cd recommender_system"
echo "source venv/bin/activate" 
echo "python api.py"