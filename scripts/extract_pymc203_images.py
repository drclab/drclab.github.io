#!/usr/bin/env python3
import json
import base64
from pathlib import Path

# Read the notebook
notebook_path = Path('content/ipynb/pymc/pymc_203.ipynb')
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Create output directory
output_dir = Path('static/img/pymc-203')
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

# Extract all plots (cells with images: 6, 8, 9, 12, 14, 18)
plots = [
    ('12b32d67', 'prior-flat.png'),                 # Cell 6: flat priors
    ('08e4adc4', 'prior-weakly-regularizing.png'),  # Cell 8: weakly regularizing priors
    ('ba482a87', 'trace-plot.png'),                 # Cell 9: trace plot
    ('27547e0b', 'ppc-plot.png'),                   # Cell 12: posterior predictive check
    ('cbfa5e70', 'posterior-fit.png'),              # Cell 14: posterior fit with HDI
    ('e4191e9a', 'out-of-sample-predictions.png')   # Cell 18: out of sample predictions
]

print('Extracting images from notebook...')
success_count = 0
for partial_id, filename in plots:
    if extract_image(partial_id, filename):
        success_count += 1

print(f'\nExtracted {success_count}/{len(plots)} images successfully!')
