FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Install system build deps for numerical libs
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    git \
    ca-certificates \
    wget \
    libopenblas-dev \
    liblapack-dev \
    libffi-dev \
    libssl-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python deps
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -r /app/requirements.txt

# Copy the example test
COPY example.py /app/example.py
COPY pymc_example.ipynb /app/pymc_example.ipynb

# Expose Jupyter Lab port
EXPOSE 8888

# Default command: start Jupyter Lab (override with CMD if needed)
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
