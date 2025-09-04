# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create results directory
RUN mkdir -p results/visualizations

# Expose ports
EXPOSE 5000 8050

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=production

# Create startup script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    python -m src.api.app\n\
elif [ "$1" = "dashboard" ]; then\n\
    python -m src.visualization.dashboard\n\
else\n\
    python example_usage.py\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh", "example"]