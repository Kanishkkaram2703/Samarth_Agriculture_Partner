# Use official lightweight Python image
FROM python:3.11.9-slim

# Set working directory inside container
WORKDIR /app

# Prevent Python from writing .pyc files and use unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy all files into container
COPY . .

# Install system dependencies (optional but useful for pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy environment variables if using locally (Render injects automatically)
COPY .env .env

# Expose port (Render/Railway use PORT environment variable)
EXPOSE 5000

# Command to run your app using Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
