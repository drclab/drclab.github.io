import re

with open('/home/cjduan/drclab.github.io/content/posts/mmm/mmm-701.md', 'r') as f:
    text = f.read()

# Fix the broken sentence across empty lines
text = text.replace('staggered analysis\n\n\n\nis to define', 'staggered analysis is to define')

# Fix a missing space/hyphen
text = text.replace('paralleltrends-style', 'parallel-trends-style')

# Add header for 7.2 if not already formatted
text = text.replace('\n7.2 Augmented Synthetic Control (ASCM)\n', '\n## 7.2 Augmented Synthetic Control (ASCM)\n')

# The `7.1 Motivation and Setup` might be better as `## 7.1 Motivation and Setup`
text = text.replace('# 7.1 Motivation and Setup', '## 7.1 Motivation and Setup')

# Let me break line 14 into smaller paragraphs. 
# It currently has no newlines within it, let's split it at logical points.
# Since it's a markdown post, typically double newline is used for paragraphs.
text = text.replace('This chapter confronts that failure and offers a resolution.', 'This chapter confronts that failure and offers a resolution.\n\n')
text = text.replace('Consider what happens when a retailer pilots a loyalty programme', '\n\nConsider what happens when a retailer pilots a loyalty programme')
text = text.replace('Hybrid methods attack this problem from three directions.', '\n\nHybrid methods attack this problem from three directions.')
text = text.replace('Each approach relaxes a constraint that pure synthetic control imposes.', '\n\nEach approach relaxes a constraint that pure synthetic control imposes.')
text = text.replace('Added flexibility, however, has costs.', '\n\nAdded flexibility, however, has costs.') # Wait, maybe not there
text = text.replace('You gain flexibility, but flexibility has costs.', '\n\nYou gain flexibility, but flexibility has costs.')
text = text.replace('The factor model foundation developed in Chapter 6 remains central.', '\n\nThe factor model foundation developed in Chapter 6 remains central.')
text = text.replace('Return to the loyalty programme.', '\n\nReturn to the loyalty programme.')
text = text.replace('Augmented synthetic control (ASCM) offers a partial fix.', '\n\nAugmented synthetic control (ASCM) offers a partial fix.')
text = text.replace('Regularisation addresses a different pathology.', '\n\nRegularisation addresses a different pathology.')
text = text.replace('Synthetic difference-in-differences (SDID)', '\n\nSynthetic difference-in-differences (SDID)')
text = text.replace('Now consider a brand launching campaigns', '\n\nNow consider a brand launching campaigns')
text = text.replace('The design-first philosophy remains central,', '\n\nThe design-first philosophy remains central,')
text = text.replace('What follows develops these ideas systematically.', '\n\nWhat follows develops these ideas systematically.')
text = text.replace('The core message is pragmatic.', '\n\nThe core message is pragmatic.')

# Remove multiple newlines created by replacements (limit to max \n\n)
text = re.sub(r'\n{3,}', '\n\n', text)


with open('/home/cjduan/drclab.github.io/content/posts/mmm/mmm-701.md', 'w') as f:
    f.write(text)
