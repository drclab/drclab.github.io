import os
import glob
import re

files = sorted(glob.glob('/home/cjduan/drclab.github.io/content/posts/mmm/mmm-7*.md'))

# Simple replacements
replacements = {
    'ŵj': r'\hat{w}_j',
    'm̂1t': r'\hat{m}_{1t}',
    'm̂jt': r'\hat{m}_{jt}',
    'm̂it': r'\hat{m}_{it}',
    'Ŷ1tASCM (0)': r'\hat{Y}_{1t}^{ASCM}(0)',
    'Yit (0)': r'Y_{it}(0)',
    'Yjt (0)': r'Y_{jt}(0)',
    'Y1t (0)': r'Y_{1t}(0)',
    'Yit (1)': r'Y_{it}(1)',
    'Yjt (1)': r'Y_{jt}(1)',
    'Y1t (1)': r'Y_{1t}(1)',
    'τ1t': r'\tau_{1t}',
    'wj∗': r'w_j^*',
    'T0': r'T_0',
    'X j∈J': r'\sum_{j \in J}',
    'P j∈J': r'\sum_{j \in J}',
    'j∈J': r'j \in J',
}

# Regex replacements for variables that need word boundaries and math wrapping
# A list of tuples (pattern, replacement)
regex_replacements = [
    # Replace variable text that is not surrounded by $, wait, just wrap them in $
    # (negative lookbehind/ahead for $)
    (r'(?<!\$)\bYit\b(?!\$)', r'$Y_{it}$'),
    (r'(?<!\$)\bYjt\b(?!\$)', r'$Y_{jt}$'),
    (r'(?<!\$)\bY1t\b(?!\$)', r'$Y_{1t}$'),
    (r'(?<!\$)\bXit\b(?!\$)', r'$X_{it}$'),
    (r'(?<!\$)\buit\b(?!\$)', r'$u_{it}$'),
    (r'(?<!\$)\bmit\b(?!\$)', r'$m_{it}$'),
    (r'(?<!\$)\bujt\b(?!\$)', r'$u_{jt}$'),
    (r'(?<!\$)\bτ\b(?!\$)', r'$\tau$'),
    (r'(?<!\$)\bθ\b(?!\$)', r'$\theta$'),
]

for file_path in files:
    if "mmm-701.md" in file_path:
        continue # skip as already fixed
        
    with open(file_path, 'r') as f:
        text = f.read()

    # Clean up empty lines that break paragraphs 
    # (replace multiple \n with space if they appear in the middle of a sentence)
    # Actually just replacing ' \n\n ' where it splits lines without punctuation
    text = re.sub(r'(?<=[a-z])\n\n(?=[a-z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z,])\n+(?=[a-z])', ' ', text)

    # Some equations are split over multiple lines like:
    # X j∈J
    # 
    #     X ŵj ...
    text = re.sub(r'X\s*j∈J\n+\s*X', r'\\sum_{j \\in J}', text)
    
    # Generic replacements
    for k, v in replacements.items():
        text = text.replace(k, f"${v}$" if '\\' in v or '_' in v else v)
    
    # Let's fix the double $$ created if we just wrapped strings
    text = text.replace('$$', '$')
    
    for pattern, repl in regex_replacements:
        text = re.sub(pattern, repl, text)
        
    # Extra paragraph cleanup: just reducing huge newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Also wrap Greek letters if floating space around them
    text = re.sub(r'(?<!\$) (τ|θ|λ|σ|β|α) (?!\$)', r' $\1$ ', text)

    with open(file_path, 'w') as f:
        f.write(text)

