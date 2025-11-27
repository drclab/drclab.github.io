# Quick Start Guide - Prophet Docker Environment

## Start the Container

```bash
cd content/docker/Prophet
docker-compose up
```

## Access the Prophet Demo Notebook

Once the container is running, open your browser and go to:

**Direct link to the demo notebook:**
```
http://localhost:8888/notebooks/mcmc/ts-201-prophet-demo.ipynb
```

**Or browse from the Jupyter home:**
```
http://localhost:8888
```
Then navigate to `mcmc/ts-201-prophet-demo.ipynb`

## Stop the Container

```bash
docker-compose down
```

## What's Included

✅ Python 3.12  
✅ pandas 2.3.3  
✅ prophet 1.2.1  
✅ matplotlib 3.10.7  
✅ Jupyter Notebook  
✅ Prophet demo notebook pre-loaded

## Troubleshooting

**Port 8888 already in use?**
- Stop other Jupyter instances: `docker-compose down`
- Or change the port in `docker-compose.yml`

**Notebook not showing up?**
- Check the container is running: `docker ps`
- Check the logs: `docker logs prophet-jupyter`
