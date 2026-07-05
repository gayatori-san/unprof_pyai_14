from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# ============================================================
# 1. CREATE DATASET (10 sentences with varying themes)
# ============================================================

sentences = [
    "The cat sat on the warm windowsill watching birds.",
    "A kitten perched on the sunny ledge observing sparrows.",
    "Machine learning models require large amounts of training data.",
    "Neural networks learn patterns from massive datasets.",
    "The pizza was delivered hot and covered in melted cheese.",
    "I ordered a steaming pepperoni pie with extra toppings.",
    "The stock market crashed after the unexpected economic report.",
    "Investors panicked when the financial news broke yesterday.",
    "She went for a morning jog through the misty forest trail.",
    "He ran an early run along the foggy woodland path."
]

print("=" * 60)
print("📚 SENTENCE SIMILARITY ANALYZER")
print("=" * 60)
print(f"\nDataset: {len(sentences)} sentences\n")
for i, s in enumerate(sentences, 1):
    print(f"  {i}. {s}")

# ============================================================
# 2. GENERATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 60)
print("🔢 Generating Embeddings...")
print("=" * 60)

# Load pre-trained model (384-dimensional vectors)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences into vectors
embeddings = model.encode(sentences)

print(f"\n✅ Embeddings generated!")
print(f"   • Model: all-MiniLM-L6-v2")
print(f"   • Vector dimension: {embeddings.shape[1]}")
print(f"   • Total embeddings: {embeddings.shape[0]}")

# ============================================================
# 3. COMPUTE SIMILARITY MATRIX
# ============================================================

print("\n" + "=" * 60)
print("📊 Computing Cosine Similarity Matrix...")
print("=" * 60)

similarity_matrix = cosine_similarity(embeddings)

# Display as DataFrame
df_sim = pd.DataFrame(
    similarity_matrix,
    index=[f"S{i}" for i in range(1, 11)],
    columns=[f"S{i}" for i in range(1, 11)]
)

print("\n🔥 Similarity Matrix (Cosine Similarity):")
print(df_sim.round(3).to_string())

# ============================================================
# 4. FIND MOST SIMILAR PAIRS
# ============================================================

print("\n" + "=" * 60)
print("🔍 Top Similar Sentence Pairs (Semantic Vibes)")
print("=" * 60)

# Get upper triangle indices (avoid duplicates & self-similarity)
pairs = []
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        pairs.append({
            'sentence_a': sentences[i],
            'sentence_b': sentences[j],
            'similarity': similarity_matrix[i][j],
            'pair': (i + 1, j + 1)
        })

# Sort by similarity (descending)
pairs_sorted = sorted(pairs, key=lambda x: x['similarity'], reverse=True)

print("\n🏆 TOP 5 MOST SIMILAR PAIRS:\n")
for rank, pair in enumerate(pairs_sorted[:5], 1):
    sim_pct = pair['similarity'] * 100
    bar = "█" * int(sim_pct / 5)  # Visual bar
    print(f"  #{rank}  [{pair['pair'][0]} ↔ {pair['pair'][1]}]  {sim_pct:.1f}% {bar}")
    print(f"       🅰️  {pair['sentence_a']}")
    print(f"       🅱️  {pair['sentence_b']}")
    print()

print("\n📉 BOTTOM 3 LEAST SIMILAR PAIRS:\n")
for rank, pair in enumerate(pairs_sorted[-3:], 1):
    sim_pct = pair['similarity'] * 100
    bar = "░" * int(sim_pct / 5) if sim_pct > 0 else "·"
    print(f"  #{rank}  [{pair['pair'][0]} ↔ {pair['pair'][1]}]  {sim_pct:.1f}% {bar}")
    print(f"       🅰️  {pair['sentence_a']}")
    print(f"       🅱️  {pair['sentence_b']}")
    print()

# ============================================================
# 5. BONUS: VISUALIZE WITH A HEATMAP
# ============================================================

print("\n" + "=" * 60)
print("🎨 Generating Heatmap Visualization...")
print("=" * 60)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(similarity_matrix, dtype=bool))  # Mask upper triangle
sns.heatmap(
    similarity_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='YlOrRd',
    xticklabels=[f"S{i}" for i in range(1, 11)],
    yticklabels=[f"S{i}" for i in range(1, 11)],
    vmin=0, vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Cosine Similarity"}
)
plt.title('Sentence Similarity Heatmap\n(Lower Triangle = Unique Pairs)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('similarity_heatmap.png', dpi=150, bbox_inches='tight')
print("\n✅ Heatmap saved as 'similarity_heatmap.png'")