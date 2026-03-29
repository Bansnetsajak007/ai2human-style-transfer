#visualize length distributions, check quality of samples, and compare human vs AI text characteristics.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# LOAD DATA
# ============================================
df = pd.read_csv("paired_dataset.csv")
print(f"Total pairs: {len(df)}")
print(df.head(2))

# ============================================
# BASIC STATS
# ============================================
df['human_word_count'] = df['human_text'].apply(lambda x: len(str(x).split()))
df['ai_word_count'] = df['ai_text'].apply(lambda x: len(str(x).split()))
df['human_char_count'] = df['human_text'].apply(len)
df['ai_char_count'] = df['ai_text'].apply(len)
df['human_sentence_count'] = df['human_text'].apply(lambda x: len([s for s in str(x).split('.') if s.strip()]))
df['ai_sentence_count'] = df['ai_text'].apply(lambda x: len([s for s in str(x).split('.') if s.strip()]))
df['human_avg_word_length'] = df['human_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
df['ai_avg_word_length'] = df['ai_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

print("\n--- WORD COUNT ---")
print(f"Human avg words: {df['human_word_count'].mean():.1f}")
print(f"AI avg words:    {df['ai_word_count'].mean():.1f}")

print("\n--- SENTENCE COUNT ---")
print(f"Human avg sentences: {df['human_sentence_count'].mean():.1f}")
print(f"AI avg sentences:    {df['ai_sentence_count'].mean():.1f}")

print("\n--- AVG WORD LENGTH ---")
print(f"Human avg word length: {df['human_avg_word_length'].mean():.2f}")
print(f"AI avg word length:    {df['ai_avg_word_length'].mean():.2f}")

# ============================================
# VISUALIZATIONS
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Human vs AI Text Analysis', fontsize=16, fontweight='bold')

# Plot 1 - Word count distribution
axes[0,0].hist(df['human_word_count'], bins=30, alpha=0.6, color='steelblue', label='Human')
axes[0,0].hist(df['ai_word_count'], bins=30, alpha=0.6, color='coral', label='AI')
axes[0,0].set_title('Word Count Distribution')
axes[0,0].set_xlabel('Word Count')
axes[0,0].legend()

# Plot 2 - Sentence count
axes[0,1].hist(df['human_sentence_count'], bins=30, alpha=0.6, color='steelblue', label='Human')
axes[0,1].hist(df['ai_sentence_count'], bins=30, alpha=0.6, color='coral', label='AI')
axes[0,1].set_title('Sentence Count Distribution')
axes[0,1].set_xlabel('Sentence Count')
axes[0,1].legend()

# Plot 3 - Avg word length
axes[1,0].hist(df['human_avg_word_length'], bins=30, alpha=0.6, color='steelblue', label='Human')
axes[1,0].hist(df['ai_avg_word_length'], bins=30, alpha=0.6, color='coral', label='AI')
axes[1,0].set_title('Average Word Length')
axes[1,0].set_xlabel('Avg Word Length (chars)')
axes[1,0].legend()

# Plot 4 - Boxplot comparison
data_to_plot = [df['human_word_count'], df['ai_word_count']]
axes[1,1].boxplot(data_to_plot, labels=['Human', 'AI'], patch_artist=True,
                  boxprops=dict(facecolor='steelblue', alpha=0.6))
axes[1,1].set_title('Word Count Boxplot')
axes[1,1].set_ylabel('Word Count')

plt.tight_layout()
plt.savefig('text_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to text_analysis.png ✅")

# ============================================
# LEXICAL DIVERSITY
# ============================================
def lexical_diversity(text):
    words = str(text).lower().split()
    return len(set(words)) / len(words) if words else 0

df['human_lexical_diversity'] = df['human_text'].apply(lexical_diversity)
df['ai_lexical_diversity'] = df['ai_text'].apply(lexical_diversity)

print("\n--- LEXICAL DIVERSITY (unique words ratio) ---")
print(f"Human: {df['human_lexical_diversity'].mean():.3f}")
print(f"AI:    {df['ai_lexical_diversity'].mean():.3f}")

# ============================================
# CONTRACTION ANALYSIS
# ============================================
contractions = ["don't", "can't", "won't", "it's", "i'm", "you're", 
                "they're", "we're", "isn't", "aren't", "didn't"]

def count_contractions(text):
    text_lower = str(text).lower()
    return sum(text_lower.count(c) for c in contractions)

df['human_contractions'] = df['human_text'].apply(count_contractions)
df['ai_contractions'] = df['ai_text'].apply(count_contractions)

print("\n--- CONTRACTIONS (casual language indicator) ---")
print(f"Human avg contractions: {df['human_contractions'].mean():.2f}")
print(f"AI avg contractions:    {df['ai_contractions'].mean():.2f}")