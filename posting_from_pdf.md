---
description: Extract and Generate Hugo Posts from a PDF Chapter
---

// turbo-all

This workflow automates the process of extracting a specific chapter from a source PDF, segmenting it into subsections, applying targeted text cleanup, and generating Hugo-compatible Markdown posts.

### 1. Define Variables
Before running the steps, define your inputs: the path to your source PDF, the starting date for the series, and the chapter number you are extracting.

```bash
SOURCE_PDF="content/pdf/\$\$_Causal_Inference_Mkting.pdf"
START_DATE="2026-05-01"  # Format: YYYY-MM-DD
CHAPTER_NUMBER=7
```

### 2. Extract Raw Document Text
Convert the entire source PDF to a plain text file.
```bash
pdftotext "$SOURCE_PDF" tmp_full_book.txt
```

### 3. Isolate Target Chapter
Manually identify the starting and ending line numbers for the target chapter from `tmp_full_book.txt` and slice it into a raw chapter file. Replace `START_LINE` and `END_LINE` in your execution below.
```bash
# Example: sed -n '14605,16957p' tmp_full_book.txt > tmp_chapter_raw.txt
sed -n '{START_LINE},{END_LINE}p' tmp_full_book.txt > tmp_chapter_raw.txt
```

### 4. Generate Posts via Python
Create and run a python script to parse the subsections, clean text artifacts (like inline PDF hyphen breaks), and output properly formatted Hugo markdown files. It will accept the date and chapter number as arguments.

```python
# Save this to generate_posts.py
import re
import os
import sys
import datetime

# Inputs from args
start_date_str = sys.argv[1]
chapter_number = int(sys.argv[2])

raw_path = "tmp_chapter_raw.txt"
out_dir = "content/posts/mmm"

# Parse the user-provided date
start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")

with open(raw_path, "r", encoding="utf-8") as f:
    text = f.read()

# Segment the chapter by subheadings like "7.1 Motivation", "7.2 ASCM", etc.
pattern = re.compile(rf"^({chapter_number}\.(\d+))\s+(.*?)$", re.MULTILINE)
matches = list(pattern.finditer(text))

for i in range(len(matches)):
    match = matches[i]
    subsec_num = int(match.group(2))
    title = match.group(3).strip()
    
    start_pos = match.end()
    end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
    
    content = text[start_pos:end_pos].strip()
    
    # Text Cleanup (Remove End-Of-Line hyphens and connect paragraphs)
    content = re.sub(r'-\n\s*', '', content)
    paragraphs = re.split(r'\n\s*\n', content)
    clean_paragraphs = [re.sub(r'(?<!\n)\n(?!\n)', ' ', p).strip() for p in paragraphs]
    content = "\n\n".join(clean_paragraphs)
    
    # Formatting
    post_date = (start_date + datetime.timedelta(days=subsec_num - 1)).strftime("%Y-%m-%d")
    post_num = (chapter_number * 100) + subsec_num
    md_filepath = os.path.join(out_dir, f"mmm-{post_num}.md")
    
    frontmatter = f"""+++
title = "MMM {post_num}: {title}"
date = "{post_date}"
type = "post"
draft = false
categories = ["posts", "stats"]
tags = ["marketing", "causal-inference", "mmm", "panel-data"]
description = "Section {chapter_number}.{subsec_num}: {title}."
math = true
+++

{content}

## References
- Shaw, C. (2025). *Causal Inference in Marketing: Panel Data and Machine Learning Methods* (Community Review Edition), Section {chapter_number}.{subsec_num}.
"""
    with open(md_filepath, "w", encoding="utf-8") as out_f:
        out_f.write(frontmatter)

print(f"Generated {len(matches)} files.")
```

Run the payload with your inputs:
```bash
python3 generate_posts.py "$START_DATE" "$CHAPTER_NUMBER"
```

### 5. Cleanup Artifacts
Remove the temporary files.
```bash
rm tmp_full_book.txt tmp_chapter_raw.txt generate_posts.py
```

### 6. Final Post-Processing
Run a cleanup on the generated `.md` files to remove stray PDF header artifacts (e.g., chapter titles looping into the margin) or page numbers matching the pattern. Be sure to replace `CHAPTER_TITLE_STRING` and verify the regex scope for your specific PDF.
```bash
sed -i '/CHAPTER_TITLE_STRING/d' content/posts/mmm/mmm-*.md
sed -i '/^[0-9]\{3\}$/d' content/posts/mmm/mmm-*.md
```
