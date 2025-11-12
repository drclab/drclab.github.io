#!/usr/bin/env python3
import json
import base64
from pathlib import Path

# Read the notebook
notebook_path = Path('content/ipynb/pymc/pymc_202.ipynb')
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Create output directory
output_dir = Path('static/img/pymc-202')
output_dir.mkdir(parents=True, exist_ok=True)

# Function to find cell by partial ID and extract image
def extract_image(partial_id, filename):
    for cell in nb['cells']:
        cell_id = cell.get('id', '')
        if partial_id in cell_id:
            outputs = cell.get('outputs', [])
            for output in outputs:
                if 'data' in output and 'image/png' in output['data']:
                    img_data = output['data']['image/png']
                    img_bytes = base64.b64decode(img_data)
                    output_path = output_dir / filename
                    with open(output_path, 'wb') as f:
                        f.write(img_bytes)
                    print(f'✓ Saved {filename}')
                    return True
    print(f'✗ Could not find image in cell {partial_id}')
    return False

# Extract all 4 plots (Cell 5, 7, 8, 14 have images)
plots = [
    ('cb412fc9', 'pooled-trace.png'),          # Cell 5: pooled trace
    ('b22cc6c8', 'hierarchical-trace-mu.png'), # Cell 7: hierarchical mu trace
    ('cf4b45f5', 'hierarchical-forest-theta.png'), # Cell 8: forest plot
    ('254677e2', 'model-comparison.png')       # Cell 14: compare plot
]

print('Extracting images from notebook...')
success_count = 0
for partial_id, filename in plots:
    if extract_image(partial_id, filename):
        success_count += 1

print(f'\nExtracted {success_count}/{len(plots)} images successfully!')
