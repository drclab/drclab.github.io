import json
import os
import base64

notebook_path = 'content/ipynb/pymc/pymc_403.ipynb'
output_md_path = 'content/posts/causal-inference/ci_201.md'
image_dir = 'static/images/ci_201'

# Ensure directories exist
os.makedirs(os.path.dirname(output_md_path), exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading notebook: {e}")
    exit(1)

md_content = []
# Frontmatter
md_content.append('---')
md_content.append('title: "PyMC 403: Introduction to Causal Inference with PPLs"')
md_content.append('date: 2025-11-29')
md_content.append('categories: ["Causal Inference"]')
md_content.append('tags: ["PyMC", "Causal Inference"]')
md_content.append('---')
md_content.append('')

cells = nb.get('cells', [])

for i, cell in enumerate(cells):
    cell_type = cell.get('cell_type')
    source = cell.get('source', [])
    if isinstance(source, list):
        source = ''.join(source)
    
    if cell_type == 'markdown':
        md_content.append(source)
        md_content.append('')
    elif cell_type == 'code':
        md_content.append('```python')
        md_content.append(source)
        md_content.append('```')
        md_content.append('')
        
        outputs = cell.get('outputs', [])
        for j, output in enumerate(outputs):
            output_type = output.get('output_type')
            if output_type in ['display_data', 'execute_result']:
                data = output.get('data', {})
                # Prioritize rich media
                if 'image/png' in data:
                    img_data_b64 = data['image/png']
                    if isinstance(img_data_b64, list):
                        img_data_b64 = ''.join(img_data_b64)
                    # Fix newlines in base64 string if any
                    img_data_b64 = img_data_b64.replace('\n', '')
                    
                    img_data = base64.b64decode(img_data_b64)
                    img_filename = f'output_{i}_{j}.png'
                    img_path = os.path.join(image_dir, img_filename)
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    # Use absolute path from site root for Hugo
                    md_content.append(f'![Output {i}_{j}](/images/ci_201/{img_filename})')
                    md_content.append('')
                elif 'image/svg+xml' in data:
                    svg_data = data['image/svg+xml']
                    if isinstance(svg_data, list):
                        svg_data = ''.join(svg_data)
                    img_filename = f'output_{i}_{j}.svg'
                    img_path = os.path.join(image_dir, img_filename)
                    with open(img_path, 'w') as f:
                        f.write(svg_data)
                    md_content.append(f'![Output {i}_{j}](/images/ci_201/{img_filename})')
                    md_content.append('')
                elif 'text/html' in data:
                    html_data = data['text/html']
                    if isinstance(html_data, list):
                        html_data = ''.join(html_data)
                    md_content.append(html_data)
                    md_content.append('')
                elif 'text/plain' in data:
                    # Fallback if no other rich media
                    pass

            elif output_type == 'stream':
                text = output.get('text', [])
                if isinstance(text, list):
                    text = ''.join(text)
                md_content.append('```')
                md_content.append(text)
                md_content.append('```')
                md_content.append('')

with open(output_md_path, 'w') as f:
    f.write('\n'.join(md_content))

print(f"Converted {notebook_path} to {output_md_path}")
