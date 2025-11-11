# Custom PyMC v5 Docker Image

This directory contains everything needed to build a Docker image with the latest PyMC v5 for probabilistic programming.

## Files

- **Dockerfile** - Multi-layer image based on Python 3.11-slim with scientific computing dependencies
- **requirements.txt** - Python packages including PyMC ≥5.0, ArviZ, NumPy, etc.
- **example.py** - Test script that runs a simple Bayesian model to verify the installation

## Prerequisites

1. **Start Docker Desktop** (or Docker daemon)
   - On macOS: Open Docker Desktop application
   - Verify with: `docker ps` (should not show connection error)

## Build the Image

```bash
cd /Users/09344682/GitHub/Docker
docker build -t pymc5-latest:cpu .
```

Build time: ~5-10 minutes (downloads base image + installs scientific packages + Jupyter Lab)

## Run Jupyter Lab (Interactive Notebooks)

Start Jupyter Lab server with your local directory mounted:

```bash
docker run -d --name pymc-jupyter -p 8888:8888 -v $(pwd):/app/notebooks pymc5-latest:cpu
```

**Access Jupyter Lab:**
- Open your browser to: **http://localhost:8888**
- No password required (configured for local development)
- Your current directory is mounted at `/app/notebooks` inside the container
- Edit `.ipynb` files directly—changes sync to your local filesystem

**Manage the container:**
```bash
# View logs
docker logs pymc-jupyter

# Stop the container
docker stop pymc-jupyter

# Start it again
docker start pymc-jupyter

# Remove the container
docker rm -f pymc-jupyter
```

## Run the Test (Python Script)

To test the installation without Jupyter:

After building, run the example to verify PyMC works:

```bash
docker run --rm pymc5-latest:cpu
```

Expected output:
- PyMC version printed
- Sampling progress bars
- ArviZ summary table
- "Docker image test PASSED ✓"

## Interactive Shell

To explore the container interactively:

```bash
docker run --rm -it pymc5-latest:cpu bash
```

Inside the container, try:
```bash
python -c "import pymc as pm; print(pm.__version__)"
python example.py
```

## Run Your Own Scripts

Mount a local directory with your PyMC models:

```bash
docker run --rm -v $(pwd)/my_models:/app/models pymc5-latest:cpu python /app/models/my_model.py
```

## Image Details

- **Base**: `python:3.11-slim`
- **PyMC version**: Latest 5.x (as of build date)
- **Size**: ~1.5-2GB (includes NumPy, SciPy, JAX, etc.)
- **Acceleration**: CPU-only (for GPU variant, see below)

## Installed Packages

Core packages from `requirements.txt`:
- pymc ≥5.0
- arviz (diagnostics & visualization)
- xarray, numpy, scipy, pandas
- matplotlib (plotting)
- **jupyterlab** (interactive notebooks)
- **ipywidgets** (interactive widgets)
- **notebook** (classic Jupyter notebook)

## GPU Variant (Optional)

For GPU-accelerated sampling via JAX:

1. Use an NVIDIA CUDA base image (e.g., `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`)
2. Install matching jaxlib wheel for your CUDA version
3. Run with `--gpus all` flag:
   ```bash
   docker run --rm --gpus all pymc5-latest:gpu python example.py
   ```

See [JAX installation guide](https://github.com/google/jax#installation) for CUDA-specific wheels.

## Customization

### Add More Packages

Edit `requirements.txt` and rebuild:
```txt
pymc>=5.0
your-package==1.2.3
```

### Change Python Version

In `Dockerfile`, change first line:
```dockerfile
FROM python:3.10-slim  # or 3.12-slim
```

### Reduce Image Size

Use multi-stage build or conda-based approach (micromamba). Current image prioritizes simplicity.

## Troubleshooting

**Build fails with compilation errors:**
- Ensure Docker has enough memory (≥4GB recommended)
- Check Docker Desktop settings → Resources

**"Cannot connect to Docker daemon":**
- Start Docker Desktop application
- Run `docker ps` to verify connection

**Sampling is slow:**
- This is a CPU-only image; for faster sampling use GPU variant
- Reduce `draws` and `tune` in your models for testing

**Import errors in VS Code:**
- Normal! Packages are only installed inside the container, not on host

## Push to Registry (Optional)

Tag and push to Docker Hub or other registry:

```bash
docker tag pymc5-latest:cpu yourusername/pymc5:latest
docker push yourusername/pymc5:latest
```

## License

Match the license of your project. PyMC itself is Apache 2.0.

---

**Built on:** November 11, 2025  
**PyMC docs:** https://www.pymc.io/  
**Maintainer:** Add your contact info
