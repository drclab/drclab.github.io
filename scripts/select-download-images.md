# Select Download Images Script

This helper copies chosen images from your Windows `Downloads` folder (mounted under `/mnt/c`) into WSL.

## Setup
- Ensure the script has execute permission: `chmod +x scripts/select-download-images.sh`.
- Optionally add an alias in `~/.bashrc`: `alias sync-download-images='/home/dulunche/drclab/scripts/select-download-images.sh'`.

## Usage
- Run the script from the repo root: `./scripts/select-download-images.sh`.
- Provide an optional destination path to copy somewhere else: `./scripts/select-download-images.sh static/img/uploads`.
- When prompted, enter the image numbers separated by spaces or commas (e.g. `1 4 9`).
- Existing files with the same name in the destination are skipped to avoid overwriting.

## Notes
- The script scans `/mnt/c/Users/*/Downloads`. If more than one user directory exists, you'll be prompted to choose one.
- Override detection by exporting `WINDOWS_USER=<win-username>` or `WINDOWS_DOWNLOADS_DIR=/mnt/c/Users/<win-username>/Downloads` before running the script.
- Supported extensions include JPG, PNG, GIF, WEBP, BMP, and TIFF.
- Run `hugo server --buildDrafts --buildFuture` to confirm the images render correctly on the site after copying.
