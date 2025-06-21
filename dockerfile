# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Streamlit (add to requirements.txt in production)
RUN pip install streamlit==1.28.0

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/app /app/utils /app/pipelines /app/UI

# Copy application code
COPY . .

# Create supervisor configuration
RUN mkdir -p /etc/supervisor/conf.d
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
user=root
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

[program:fastapi]
command=python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/fastapi.err.log
stdout_logfile=/var/log/supervisor/fastapi.out.log
environment=PYTHONPATH="/app"

[program:streamlit]
command=streamlit run UI/streamlit_ui.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/streamlit.err.log
stdout_logfile=/var/log/supervisor/streamlit.out.log
environment=PYTHONPATH="/app"
EOF

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Change ownership of the app directory
RUN chown -R appuser:appuser /app

# Create logs directory and set permissions
RUN mkdir -p /var/log/supervisor && \
    chmod 755 /var/log/supervisor && \
    mkdir -p /app/logs && \
    chmod 755 /app/logs && \
    chown -R appuser:appuser /var/log/supervisor

# Expose ports
EXPOSE 8000 8501

# Health check for both services
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8501/_stcore/health || exit 1

# Create startup script
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

echo "Starting Payment Info Assistant..."
echo "FastAPI will be available at: http://localhost:8000"
echo "Streamlit UI will be available at: http://localhost:8501"
echo "API Documentation will be available at: http://localhost:8000/docs"

# Start supervisor
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
EOF

RUN chmod +x /app/start.sh

# Command to run both applications
CMD ["/app/start.sh"]