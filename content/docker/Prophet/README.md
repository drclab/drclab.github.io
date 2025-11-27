# Prophet Docker Environment

This Docker environment is configured to run Facebook Prophet for time series forecasting, matching the setup from the `ts-201-prophet-demo.ipynb` notebook.

## Contents

- **Dockerfile**: Builds a Python 3.12 environment with Prophet and all dependencies
- **docker-compose.yml**: Convenient orchestration file for easy startup
- **.dockerignore**: Excludes unnecessary files from the Docker build

## Quick Start

### Using Docker Compose (Recommended)

```bash
cd content/docker/Prophet
docker-compose up
```

Then open your browser to `http://localhost:8888`

The Prophet demo notebook will be available at `mcmc/ts-201-prophet-demo.ipynb`

Or access it directly at: `http://localhost:8888/notebooks/mcmc/ts-201-prophet-demo.ipynb`

### Using Docker Directly

Build the image:
```bash
docker build -t prophet-env .
```

Run the container:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks prophet-env
```

## Installed Packages

- Python 3.12
- pandas 2.3.3
- prophet 1.2.1
- matplotlib 3.10.7
- jupyter
- cmdstan (Prophet dependency)

## Usage

Once the container is running, you can:

1. Access Jupyter Notebook at `http://localhost:8888`
2. Create new notebooks or upload the demo notebook
3. Run Prophet forecasting models

## Example Code

```python
import pandas as pd
from prophet import Prophet

# Load data
url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
df = pd.read_csv(url)

# Fit model
m = Prophet()
m.fit(df)

# Make forecast
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)

# Plot
m.plot(forecast)
```

## Notes

- The notebook server runs without a token/password by default (for development only)
- Notebooks are saved to the `notebooks` directory which is mounted as a volume
- Prophet uses CmdStan in the background for Bayesian inference
- Container has restart policy `unless-stopped` for automatic recovery

## Container Management

To stop the container **without removing it**:
```bash
docker-compose stop
```

To start it again:
```bash
docker-compose start
```

For more details on container lifecycle management, see [CONTAINER_MANAGEMENT.md](CONTAINER_MANAGEMENT.md)
