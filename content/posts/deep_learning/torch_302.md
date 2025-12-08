+++
title = "Torch 302: Embeddings"
date = "2025-12-08T01:00:00Z"
type = "post"
draft = false
tags = ["deep learning", "torch", "nlp", "embeddings", "glove", "bert"]
categories = ["deep_learning"]
description = "From static GloVe vectors to training your own and understanding contextual BERT embeddings."
+++

You've learned how to process text by tokenizing it and converting it into tensors. However, the integer IDs assigned to words don't inherently capture their meaning. The number for "cat" is no more related to the number for "dog" than it is to the one for "banana". This is where **word embeddings** come in. Embeddings are dense vector representations that map words into a multi-dimensional space where semantic relationships can be measured mathematically.

There are two main categories of word embeddings:

* **Static Embeddings**: Each word in the vocabulary is mapped to a single, fixed vector. Models like **Word2Vec** and **GloVe** use this method. The vector for "Apple" is the same, regardless of context.
* **Dynamic (or Contextual) Embeddings**: The vector for a word changes based on the sentence it's in. Models like **BERT** excel at this.

In this post, we'll explore both types, starting with GloVe, then building a simple model from scratch, and finally looking at BERT.

## GloVe: Static Embeddings

[GloVe](https://nlp.stanford.edu/projects/glove/) (Global Vectors for Word Representation) is a popular pre-trained static embedding model. It learns word vectors by analyzing a massive corpus of text and computing aggregated global word-word co-occurrence statistics.

We'll use the **glove.6B** model, trained on Wikipedia and Gigaword.

### Semantic Similarity

One of the most powerful features of word embeddings is **semantic similarity**. Words with similar meanings have vector representations that are close to each other in vector space. We can measure this using **cosine similarity**.

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_closest_words(embedding, embeddings_dict, exclude_words=[], top_n=5):
    """Finds the N most semantically similar words to a given vector."""
    filtered_words = [word for word in embeddings_dict.keys() if word not in exclude_words]
    if not filtered_words: return None
        
    embedding_matrix = np.array([embeddings_dict[word] for word in filtered_words])
    target_embedding = embedding.reshape(1, -1)
    similarity_scores = cosine_similarity(target_embedding, embedding_matrix)
    closest_word_indices = np.argsort(similarity_scores[0])[::-1][:top_n]
    
    return [(filtered_words[i], similarity_scores[0][i]) for i in closest_word_indices]
```

### Analogies

Embeddings can capture complex relationships. The classic example is `king - man + woman ≈ queen`.

```python
# Assuming glove_embeddings is loaded
king = glove_embeddings['king']
man = glove_embeddings['man']
woman = glove_embeddings['woman']

result_embedding = king - man + woman

# Find closest words
closest = find_closest_words(result_embedding, glove_embeddings, exclude_words=['king', 'man', 'woman'])
print(f"king - man + woman ≈ {closest[0][0]}")
```

### Visualization

We can use **Principal Component Analysis (PCA)** to reduce the 100-dimensional GloVe vectors to 2 dimensions for visualization. This reveals that related concepts (like vehicles, pets, fruits) form distinct clusters.

## Building Your Own Embeddings From Scratch

Sometimes you need to train embeddings for a specialized vocabulary. Let's build a simple model in PyTorch.

### Defining the Vocabulary

```python
vocabulary = ['car', 'bike', 'plane', 'cat', 'dog', 'bird', 'orange', 'apple', 'grape']
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size = len(vocabulary)
embedding_dim = 3
```

### The Model

We'll use a simple neural network with an `nn.Embedding` layer (the lookup table) and a linear layer.

```python
import torch
import torch.nn as nn

class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.linear(embedded)
        return output, embedded

model = SimpleEmbeddingModel(vocab_size, embedding_dim)
```

### Training

We train the model to predict related words (e.g., given 'car', predict 'bike'). After training, the learned embeddings will naturally group similar words together, just like GloVe.

## Beyond Static Embeddings: BERT

Static models have a "Bat" problem: they give the same vector for "bat" (animal) and "bat" (baseball). **BERT** (Bidirectional Encoder Representations from Transformers) solves this by being **contextual**.

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

sentence1 = "A bat flew out of the cave."
sentence2 = "He swung the baseball bat."

# Process sentence 1
inputs1 = tokenizer(sentence1, return_tensors='pt')
outputs1 = model(**inputs1)
bat_vector1 = outputs1.last_hidden_state[0][2] # Index of 'bat'

# Process sentence 2
inputs2 = tokenizer(sentence2, return_tensors='pt')
outputs2 = model(**inputs2)
bat_vector2 = outputs2.last_hidden_state[0][5] # Index of 'bat'

# Compare
are_identical = torch.equal(bat_vector1, bat_vector2)
print(f"Are vectors identical? {are_identical}") # False
```

BERT generates a unique vector for "bat" in each sentence, capturing the specific meaning based on context.

## Conclusion

*   **Use Static Embeddings (GloVe)** for straightforward tasks, speed, and efficiency.
*   **Use Contextual Embeddings (BERT)** when deep understanding, ambiguity handling, and high accuracy are critical.

You now have the foundation to work with both types of embeddings, a crucial skill for modern NLP.
