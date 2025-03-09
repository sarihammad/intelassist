FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN python setup.py

# Expose ports
EXPOSE 8000 8501

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000
ENV DEBUG=False

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "api" ]; then\n\
    exec uvicorn app.main:app --host $HOST --port $PORT\n\
elif [ "$1" = "ui" ]; then\n\
    exec streamlit run app/ui/streamlit_app.py\n\
else\n\
    echo "Usage: docker run [options] <image> [api|ui]"\n\
    exit 1\n\
fi' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"] 