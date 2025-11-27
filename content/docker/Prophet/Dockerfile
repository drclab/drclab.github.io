# Use Python 3.12 as base image (matching the notebook environment)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for Prophet
# Prophet requires cmdstan which needs C++ compiler and related tools
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Prophet and its dependencies
# Using the exact versions from the notebook output
RUN pip install --no-cache-dir \
    pandas==2.3.3 \
    prophet==1.2.1 \
    matplotlib==3.10.7 \
    jupyter \
    notebook

# Expose Jupyter notebook port
EXPOSE 8888

# Set environment variable to prevent Python from buffering output
ENV PYTHONUNBUFFERED=1

# Create a directory for notebooks
RUN mkdir -p /app/notebooks

# Default command to run Jupyter notebook
# Start in /app directory to show mcmc folder with demo notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''", "--notebook-dir=/app"]
