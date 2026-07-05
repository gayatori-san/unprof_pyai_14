Embeddings & Sentence Similarity Analyzer

## What Are Embeddings?

Embeddings are dense numerical vector representations of text that capture **semantic meaning**. Unlike one-hot encoding (which treats every word as equally unrelated), embeddings place similar words/sentences close together in a high-dimensional vector space.

## Key Concepts

### Word2Vec
Google's 2013 neural approach that learns word relationships by predicting context words. Uses Skip-gram or CBOW architecture.

### Sentence Embeddings
Extension of word embeddings to entire sentences. Uses transformer architectures (like BERT) to create a single vector representing sentence meaning.

### Vector Space
An N-dimensional space where:
- Each dimension captures some latent semantic feature
- Distance between vectors = semantic dissimilarity
- Direction between vectors = semantic relationships

### Cosine Similarity
Measures the cosine of the angle between two vectors:
- `1.0` = identical direction (same meaning)
- `0.0` = orthogonal (unrelated)
- `-1.0` = opposite (rare in NLP)

## How This App Works

1. **Dataset**: 10 sentences across 5 semantic themes (animals, ML, food, finance, running)
2. **Embedding**: `all-MiniLM-L6-v2` model converts each sentence to a 384-dim vector
3. **Similarity**: Cosine similarity computed between all sentence pairs
4. **Output**: Ranked list of most/least similar pairs + heatmap visualization

## Run It

```bash
pip install -r requirements.txt
```

## Requirements:
```bash
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
```
```bash
git clone https://github.com/gayatori-san/unprof_pyai_14
```
```bash
python task14.py
```
