# PowerShell script for Windows users

# Create virtual environment
Write-Host "Creating Python virtual environment..." -ForegroundColor Green
python -m venv venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
. .\venv\Scripts\Activate.ps1

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Create .env file if it doesn't exist
if (-not (Test-Path -Path ".env" -PathType Leaf)) {
    Write-Host "Creating .env file from example..." -ForegroundColor Green
    Copy-Item -Path ".env.example" -Destination ".env"
    Write-Host "Please update the .env file with your MongoDB connection string." -ForegroundColor Yellow
}

# Install Node.js dependency for the Express server
Write-Host "Installing axios in the Node.js project..." -ForegroundColor Green
cd ..
npm install axios --save
cd recommender_system

Write-Host "Setup completed successfully!" -ForegroundColor Green
Write-Host "To start the recommender system API, run:" -ForegroundColor Cyan
Write-Host "cd recommender_system" -ForegroundColor White
Write-Host ". .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "python api.py" -ForegroundColor White