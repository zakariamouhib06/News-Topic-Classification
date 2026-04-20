# Use an official Python concise image
# Using slim to keep the image size small, which is a best practice
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Set environment variables:
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures stdout/stderr is logged immediately without buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies. Clean up after to reduce final image size.
# Also create a non-root user to run the app securely
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

# Copy just the requirements.txt first to take advantage of Docker cache layers
COPY requirements.txt .

# Install dependencies, including gunicorn for production server handling
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt gunicorn

# Copy the rest of the application files (ignoring files in .dockerignore)
COPY . .

# Change ownership of the working directory so appuser can access models/ etc. if needed
RUN chown -R appuser:appuser /app

# Switch to non-root user for security best practices
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Run the app with gunicorn and uvicorn workers
# Gunicorn handles process management, uvicorn handles the ASGI async magic
CMD ["gunicorn", "api.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
