import re
import os
import datetime

raw_path = "content/pdf/chapter_7_raw.txt"
out_dir = "content/posts/mmm"

with open(raw_path, "r", encoding="utf-8") as f:
    text = f.read()

# Pattern to find sections like "7.1 Motivation and Setup"
# The text from one section to the next
pattern = re.compile(r"^(7\.(\d+))\s+(.*?)$", re.MULTILINE)
matches = list(pattern.finditer(text))

sections = []
for i in range(len(matches)):
    match = matches[i]
    sec_num = match.group(1)
    subsec = int(match.group(2))
    title_text = match.group(3).strip()
    
    start_pos = match.end()
    end_pos = matches[i+1].start() if i + 1 < len(matches) else len(text)
    
    sec_text = text[start_pos:end_pos].strip()
    
    # Text cleaning
    # 1. fix hyphens at end of lines
    sec_text = re.sub(r'-\n\s*', '', sec_text)
    # 2. replace single newlines with space (but preserve double newlines as paragraphs)
    # Actually, we split by double newline, clean single newlines, and rejoin
    paragraphs = re.split(r'\n\s*\n', sec_text)
    clean_paragraphs = []
    for p in paragraphs:
        # replace single newlines with space
        cleaned_p = re.sub(r'(?<!\n)\n(?!\n)', ' ', p)
        clean_paragraphs.append(cleaned_p.strip())
    sec_text = "\n\n".join(clean_paragraphs)
    
    sections.append({
        "num": subsec,
        "title": title_text,
        "content": sec_text
    })

for sec in sections:
    num = sec['num']
    # Format date: 2026-05-01 starting
    dt = datetime.datetime(2026, 5, num)
    date_str = dt.strftime("%Y-%m-%d")
    
    # 701, 702
    post_num = 700 + num
    
    md_filename = f"mmm-{post_num}.md"
    md_filepath = os.path.join(out_dir, md_filename)
    
    frontmatter = f"""+++
title = "MMM {post_num}: {sec['title']}"
date = "{date_str}"
type = "post"
draft = false
categories = ["posts", "stats"]
tags = ["marketing", "causal-inference", "mmm", "panel-data", "SDID", "synthetic-control"]
description = "Chapter 7 section on {sec['title']}."
math = true
+++

{sec['content']}

## References
- Shaw, C. (2025). *Causal Inference in Marketing: Panel Data and Machine Learning Methods* (Community Review Edition), Section 7.{num}.
"""
    with open(md_filepath, "w", encoding="utf-8") as out_f:
        out_f.write(frontmatter)

print(f"Generated {len(sections)} files.")
