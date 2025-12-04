# Groq Docker Environment

This directory contains a Docker setup for running the Groq Python SDK.

## Prerequisites

- Docker installed
- A Groq API Key

## Setup

1.  **Build the Docker image:**

    ```bash
    docker build -t groq-app .
    ```

2.  **Run the container:**

    You need to provide your `GROQ_API_KEY`. You can do this in two ways:

    **Option A: Using an environment variable directly**

    ```bash
    docker run --rm -e GROQ_API_KEY=your_api_key_here groq-app
    ```

    **Option B: Using a .env file**

    Create a `.env` file in this directory with the following content:

    ```env
    GROQ_API_KEY=your_api_key_here
    ```

    Then run:

    ```bash
    docker run --rm --env-file .env groq-app
    ```

    **Option C: Using Docker Compose (Recommended)**

    Make sure you have a `.env` file with your API key.

    ```bash
    docker compose up --build
    ```

    Access Jupyter Lab at `http://localhost:8888`.

## Files

-   `docker-compose.yml`: Docker Compose configuration.
-   `Dockerfile`: Defines the Docker image with Jupyter AI.
-   `requirements.txt`: Python dependencies (`groq`, `python-dotenv`, `jupyter-ai`).
-   `main.py`: A simple script to verify the connection.
