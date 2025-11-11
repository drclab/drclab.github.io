# Dockerfile for Jupyter Notebook with RStan and Bayesplot
# Based on rocker/verse which includes R, RStudio, tidyverse, and publishing tools

FROM rocker/verse:latest

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Update system and install system dependencies for RStan
RUN apt-get update && apt-get install -y \
    build-essential \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libv8-dev \
    libnode-dev \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Jupyter and IRkernel for R notebooks
RUN pip3 install --no-cache-dir --break-system-packages \
    jupyter \
    notebook \
    jupyterlab \
    matplotlib \
    numpy \
    pandas

# Install R packages for Bayesian modeling
RUN R -e "install.packages(c( \
    'rstan', \
    'bayesplot', \
    'loo', \
    'posterior', \
    'cmdstanr', \
    'rstanarm', \
    'brms', \
    'tidybayes', \
    'ggplot2', \
    'dplyr', \
    'tidyr', \
    'purrr', \
    'coda', \
    'mcmcplots' \
    ), repos='https://cloud.r-project.org/', Ncpus=4)"

# Install CmdStan (optional but recommended for faster sampling)
# Only install if cmdstanr package was successfully installed
RUN R -e "if (requireNamespace('cmdstanr', quietly = TRUE)) { cmdstanr::install_cmdstan(cores = 4) } else { message('Skipping CmdStan installation') }"

# Configure RStan options for better performance
RUN mkdir -p ~/.R && \
    echo "CXXFLAGS=-O3 -march=native -mtune=native" > ~/.R/Makevars && \
    echo "CXX14FLAGS=-O3 -march=native -mtune=native" >> ~/.R/Makevars

# Install IRkernel for Jupyter
RUN R -e "install.packages('IRkernel', repos='https://cloud.r-project.org/')" && \
    R -e "IRkernel::installspec(user = FALSE)"

# Create working directory
WORKDIR /workspace

# Expose Jupyter notebook port
EXPOSE 8888

# Set up Jupyter configuration
RUN mkdir -p /root/.jupyter && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py

# Default command to start Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
