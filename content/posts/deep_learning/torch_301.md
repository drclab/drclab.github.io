+++
title = "Torch 301: Basic Tokenization"
date = "2025-12-07T00:00:00Z"
type = "post"
draft = false
tags = ["deep learning", "torch", "nlp", "tokenization"]
categories = ["deep_learning"]
description = "Mastering the first step of NLP: From manual vocabulary building to subword tokenization with Hugging Face's AutoTokenizer."
+++

Welcome to your first hands-on lab for Natural Language Processing (NLP)! Unlike images, which are naturally numerical arrays, text is a sequence of symbols that machines don't inherently understand. Before you can perform tasks like sentiment analysis or translation, you must first convert your text into a format a model can process.

**Tokenization** is an important first step in any NLP workflow, converting raw text into meaningful units called **tokens**. These tokens are building blocks used by models, such as BERT, to generate word embeddings - dense vector representations capturing semantic meaning.

This post provides a practical look into this fundamental process. You will explore tokenization by comparing a manual, from-scratch approach with the use of a modern, pre-trained tool.

Specifically, you'll learn to:
* Build a simple tokenizer from scratch to understand the core mechanics.
* Use a powerful, pre-trained BERT tokenizer from the popular Hugging Face library.
* Understand why matching tokenizers to models is critical and use `AutoTokenizer` as a best practice.
* Observe how this advanced tool automatically handles challenges like out-of-vocabulary (OOV) words.

## Manual Tokenization: Building a Vocabulary

Let's start by building a simple tokenizer from scratch. We'll define a few sample sentences, implement a function to split text into words, and build a vocabulary where each unique word maps to a numerical ID.

```python
sentences = [
    'I love my dog',
    'I love my cat'
]

# Tokenization function
def tokenize(text):
  # Lowercase the text and split by whitespace
  return text.lower().split()

# Build the vocabulary
def build_vocab(sentences):
    vocab = {}
    # Iterate through each sentence.
    for sentence in sentences:
        # Tokenize the current sentence
        tokens = tokenize(sentence)
        # Iterate through each token in the sentence
        for token in tokens:
            # If the token is not already in the vocabulary
            if token not in vocab:
                # Add the token to the vocabulary and assign it a unique integer ID
                # IDs start from 1; 0 can be reserved for padding.
                vocab[token] = len(vocab) + 1
    return vocab

# Create the vocabulary index
vocab = build_vocab(sentences)

print("Vocabulary Index:", vocab)
```

## Using a Pre-trained BERT Tokenizer

While manual tokenization helps us understand the concept, in practice, we use robust, pre-trained tokenizers. Let's use the `BertTokenizerFast` from the Hugging Face `transformers` library, loading the `bert-base-uncased` model.

```python
import torch
from transformers import BertTokenizerFast

# Initialize the tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

sentences = [
    'I love my dog',
    'I love my cat'
]

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input (helpful for understanding the output)
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

# Get the model's vocabulary (mapping from tokens to IDs)
word_index = tokenizer.get_vocab()

# Print the human-readable `tokens` for each sentence
print("Tokens:", tokens)

print("\nToken IDs:", encoded_inputs['input_ids'])
```

### Remark on Model Compatibility
It is worth emphasizing that in NLP, tokenizers are not one-size-fits-all tools. Each tokenizer is specifically designed to work with a particular model. The `bert-base-uncased` tokenizer is designed to format text in the exact way the BERT model was trained to understand it. This includes its specific vocabulary, rules for splitting words, and the use of special tokens like `[CLS]` and `[SEP]`.

**Using the tokenizer that matches your model ensures the input format is exactly what the model expects.** Mismatching a model and tokenizer can lead to poor performance or errors.

## Using `AutoTokenizer`

While using a specific class like `BertTokenizerFast` works perfectly, the Hugging Face `transformers` library offers a convenient and robust solution: `AutoTokenizer`.

The `AutoTokenizer` class is a smart wrapper that automatically detects and loads the correct tokenizer class for any given model checkpoint. Instead of needing to remember whether a model requires `BertTokenizerFast`, `GPT2Tokenizer`, or another specific class, `AutoTokenizer.from_pretrained()` handles it for you.

```python
from transformers import AutoTokenizer

# Initialize the tokenizer using the AutoTokenizer class
# This automatically loads the correct tokenizer (BertTokenizerFast in this case)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

sentences = [
    'I love my dog',
    'I love my cat'
]

# Tokenize the sentences and encode them
encoded_inputs = tokenizer(sentences, padding=True, 
                           truncation=True, return_tensors='pt')

# To see the tokens for each input
tokens = [tokenizer.convert_ids_to_tokens(ids)
          for ids in encoded_inputs["input_ids"]]

print("Tokens:", tokens)
print("\nToken IDs:", encoded_inputs['input_ids'])
```

## "Out-of-Vocabulary" (OOV) Words

You might encounter words that are not in the tokenizer's built-in dictionary (e.g., proper names or rare terms). These are called Out-of-Vocabulary (OOV) words.

Modern tokenizers handle this by breaking OOV words into smaller, known sub-word parts. A sub-word starting with `##` (like `##bs`) attaches to the previous piece to form the original word.

**Example**:
If a name like `"Mubsi"` is OOV, it might become `['mu', '##bs', '##i']`. This means "mu" + "bs" + "i" are combined to represent "Mubsi".

This "subword tokenization" allows the tokenizer to handle any word, ensuring no word is truly "unknown."

```python
# A list of words that are likely "Out-of-Vocabulary" (OOV)
oov_words = ["Tokenization", "HuggingFace", "unintelligible"]

print("--- Subword Tokenization Example ---")

# Iterate through the words and show how they are tokenized
for word in oov_words:
    # The .tokenize() method is a direct way to see the subword breakdown
    subwords = tokenizer.tokenize(word)
    
    # Print the results
    print(f"Original word: '{word}'")
    print(f"Subword tokens: {subwords}\n")
```

## Conclusion

Congratulations! You have successfully transformed raw text into structured, numerical tensors that a deep learning model can understand.

You started by building a vocabulary manually, highlighting the core challenge: every unique word needs a numerical representation. Then, you saw the modern approach using a pre-trained tokenizer, which handles the entire preprocessing pipeline—from splitting words and adding special tokens like `[CLS]` and `[SEP]` to padding and truncation. You also saw how **subword tokenization** elegantly solves the out-of-vocabulary problem.

The key takeaway is that the tokenizer and its corresponding model are tightly coupled. Using tools like `AutoTokenizer` simplifies this process, guaranteeing you always use the correct tokenizer for your chosen model.
