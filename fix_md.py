import re

with open('/home/cjduan/drclab.github.io/content/posts/mmm/mmm-701.md', 'r') as f:
    text = f.read()

# Fix the section title and the broken word
text = text.replace('pe7.1 Motivation and Setup\n\n\nnalise', 'penalise')
# Insert a newline or let's find a place to put the `# 7.1 Motivation and Setup`.
# It's currently at line 12. Let's put `# 7.1 Motivation and Setup` right after the front matter.

# Fix math formulas
text = text.replace('P σ 2 (1 + j (wj∗ )2 )', r'$\sigma^2 (1 + \sum_j (w_j^*)^2)$')
text = text.replace('τ (g, t)', r'$\tau(g, t)$')
text = text.replace('θk', r'$\theta_k$')
text = text.replace('differencein-differences', 'difference-in-differences')

# Add the heading if it is not already there
if '# 7.1 Motivation and Setup' not in text:
    text = text.replace('+++\n\n', '+++\n\n# 7.1 Motivation and Setup\n\n', 1)

with open('/home/cjduan/drclab.github.io/content/posts/mmm/mmm-701.md', 'w') as f:
    f.write(text)
