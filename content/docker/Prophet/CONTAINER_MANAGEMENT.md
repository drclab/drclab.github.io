# Container Lifecycle Management

## Understanding Docker Commands

### Starting the Container

**Fresh start (builds if needed):**
```bash
docker-compose up -d
```

**Start existing stopped container:**
```bash
docker-compose start
```

### Stopping the Container

**Stop WITHOUT removing the container:**
```bash
docker-compose stop
```

**Stop AND remove the container:**
```bash
docker-compose down
```

## Key Differences

| Command | Action | Container State After | Data Preserved? |
|---------|--------|----------------------|-----------------|
| `docker-compose stop` | Stops running container | Stopped (still exists) | ✅ Yes |
| `docker-compose down` | Stops and removes container | Removed | ✅ Yes (volumes persist) |
| `docker-compose start` | Starts stopped container | Running | ✅ Yes |
| `docker-compose restart` | Stops then starts | Running | ✅ Yes |

## Restart Policy

The docker-compose.yml is now configured with:
```yaml
restart: unless-stopped
```

This means:
- ✅ Container automatically restarts if it crashes
- ✅ Container restarts when Docker daemon starts (e.g., after reboot)
- ✅ Container does NOT restart if you manually stop it with `docker-compose stop`
- ✅ Container persists when stopped (not removed)

## Recommended Workflow

**Daily use:**
```bash
# Start
docker-compose start

# Stop (keeps container)
docker-compose stop
```

**When you need to rebuild:**
```bash
# Stop and remove
docker-compose down

# Rebuild and start
docker-compose up -d --build
```

## Data Persistence

Your work is safe! Even if the container is removed (`docker-compose down`), your notebooks are preserved because they're stored in:
- `./notebooks/` - Mounted volume (your local machine)
- `../../posts/mcmc/` - Read-only mount (your local machine)

## Current Status

With the updated configuration:
- Container will **not** be automatically removed when stopped
- Container will automatically restart after system reboots
- Use `docker-compose stop` to stop without removing
- Use `docker-compose down` only when you want to clean up
