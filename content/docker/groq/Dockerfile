FROM python:3.11-slim

# Create user cjduan
RUN useradd -m -u 1000 cjduan

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Change ownership
RUN chown -R cjduan:cjduan /app

# Switch to user
USER cjduan

EXPOSE 8888

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--IdentityProvider.token=''"]
