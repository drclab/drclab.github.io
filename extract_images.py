
import json
import base64
import os

notebook_path = '/home/cjduan/drclab.github.io/content/ipynb/pymc/pymc_403.ipynb'
output_dir = '/home/cjduan/drclab.github.io/static/images/causal-inference/ci_201'

os.makedirs(output_dir, exist_ok=True)

with open(notebook_path, 'r') as f:
    nb = json.load(f)

image_count = 0
for cell_index, cell in enumerate(nb['cells']):
    if 'outputs' in cell:
        for output in cell['outputs']:
            if 'data' in output:
                # Handle SVG
                if 'image/svg+xml' in output['data']:
                    svg_data = output['data']['image/svg+xml']
                    if isinstance(svg_data, list):
                        svg_data = ''.join(svg_data)
                    
                    # We expect the DAG to be in the early cells. Let's name it specifically if possible, 
                    # otherwise generic names.
                    # Cell 3 and 4 have DAGs. Cell 4 is the detailed one.
                    if cell_index == 11: # Based on view_file, cell 4 is index 3? No, let's check execution_count
                         # view_file shows cell_type code, execution_count 3 for simple DAG, 4 for detailed DAG.
                         # Let's just save all SVGs with index.
                         pass
                    
                    filename = f"output_{cell_index}_{image_count}.svg"
                    if cell.get('execution_count') == 3:
                        filename = "dag_simple.svg"
                    elif cell.get('execution_count') == 4:
                        filename = "dag_detailed.svg"
                        
                    with open(os.path.join(output_dir, filename), 'w') as f_out:
                        f_out.write(svg_data)
                    print(f"Saved {filename}")
                    image_count += 1

                # Handle PNG
                if 'image/png' in output['data']:
                    png_data = output['data']['image/png']
                    # PNG data is base64 encoded
                    img_bytes = base64.b64decode(png_data)
                    
                    filename = f"output_{cell_index}_{image_count}.png"
                    # Cell 18 has the PPC plot
                    if cell.get('execution_count') == 18:
                        filename = "ppc_plot.png"
                    
                    with open(os.path.join(output_dir, filename), 'wb') as f_out:
                        f_out.write(img_bytes)
                    print(f"Saved {filename}")
                    image_count += 1
