# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with home directory
RUN groupadd -r agent && useradd -r -g agent -m -d /home/agent agent

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R agent:agent /app && \
    chown -R agent:agent /home/agent

# Switch to non-root user
USER agent

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; response = requests.get('http://localhost:8001/health', timeout=5); exit(0 if response.status_code == 200 else 1)"

# Run the application
CMD ["uvicorn", "agent_api:app", "--host", "0.0.0.0", "--port", "8001"]