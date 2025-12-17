
import json
import os
import base64
from datetime import datetime

# Configuration
NOTEBOOK_PATH = '/Users/09344682/GitHub/drclab.github.io/content/ipynb/C2_W3_Lab_3_Optimization_Using_Newtons_Method.ipynb'
POST_SLUG = 'optimization-using-newtons-method'
POST_DIR = '/Users/09344682/GitHub/drclab.github.io/content/posts/stats'
IMAGE_DIR = f'/Users/09344682/GitHub/drclab.github.io/static/images/{POST_SLUG}'
POST_PATH = os.path.join(POST_DIR, f'{POST_SLUG}.md')

# Ensure image directory exists
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(POST_DIR, exist_ok=True)

def extract_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    content = []
    
    # Front matter
    date_str = datetime.now().strftime('%Y-%m-%d')
    front_matter = f"""+++
title = "Optimization Using Newton's Method"
date = "{date_str}"
type = "post"
draft = false
math = true
tags = ["optimization", "newtons-method", "python", "calculus"]
categories = ["posts", "stats"]
description = "Implementing Newton's method for optimization in one and two variables, and comparing it with Gradient Descent."
+++

"""
    content.append(front_matter)

    image_counter = 0

    for cell in nb['cells']:
        cell_type = cell['cell_type']
        source = ''.join(cell['source'])

        if cell_type == 'markdown':
            content.append(source + "\n\n")
        
        elif cell_type == 'code':
            content.append("```python\n" + source + "\n```\n\n")
            
            # Helper to process outputs
            for output in cell.get('outputs', []):
                output_type = output.get('output_type')
                
                if output_type == 'stream':
                    text = ''.join(output['text'])
                    content.append("```\n" + text + "```\n\n")
                
                elif output_type in ['display_data', 'execute_result']:
                    data = output.get('data', {})
                    
                    # Handle images
                    if 'image/png' in data:
                        image_data = data['image/png']
                        image_filename = f'image_{image_counter}.png'
                        image_path = os.path.join(IMAGE_DIR, image_filename)
                        
                        with open(image_path, 'wb') as img_f:
                            img_f.write(base64.b64decode(image_data))
                        
                        # Relative path for the blog post
                        rel_image_path = f'/images/{POST_SLUG}/{image_filename}'
                        content.append(f"![Generated Plot]({rel_image_path})\n\n")
                        image_counter += 1
                        
                    # Handle text/plain fallback if needed (usually for simple results)
                    elif 'text/plain' in data:
                        text = ''.join(data['text/plain'])
                        # Don't print text/plain if it's just the object repr of a figure we just displayed
                        if not text.startswith('<Figure'):
                             content.append("```\n" + text + "\n```\n\n")

    with open(POST_PATH, 'w', encoding='utf-8') as f:
        f.write(''.join(content))

    print(f"Post generated at {POST_PATH}")
    print(f"Images extracted to {IMAGE_DIR}")

if __name__ == "__main__":
    extract_notebook()
