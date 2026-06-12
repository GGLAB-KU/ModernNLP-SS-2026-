import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1. Toy Corpus with distinct semantic clusters
corpus = [
    "cat and dog are friends",
    "the cute cat purrs",
    "a loyal dog barks",
    "i love hot coffee in the morning",
    "espresso and latte are types of coffee",
    "cappuccino is a hot drink",
    "write clean python code for machine learning",
    "deploy the model on the cluster",
    "pytorch is a machine learning framework"
]

# 2. Preprocessing & Vocabulary Construction
tokenized_corpus = [sentence.lower().split() for sentence in corpus]
vocab = list(set([word for sentence in tokenized_corpus for word in sentence]))
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)

# 3. Generate CBOW Training Data (Context window = 2)
# Context: [word_t-2, word_t-1, word_t+1, word_t+2] -> Target: word_t
CONTEXT_SIZE = 2
data = []
for sentence in tokenized_corpus:
    for i in range(CONTEXT_SIZE, len(sentence) - CONTEXT_SIZE):
        context = (
            [sentence[i - 2], sentence[i - 1]] +
            [sentence[i + 1], sentence[i + 2]]
        )
        target = sentence[i]
        data.append((context, target))

# 4. Define the CBOW Model Architecture
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        # The rows of this embedding matrix are our word vectors!
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        # inputs shape: [batch_size, context_len]
        embeds = self.embeddings(inputs)  # [batch_size, context_len, embedding_dim]
        # Average the context word vectors
        context_mean = torch.mean(embeds, dim=1)  # [batch_size, embedding_dim]
        out = self.linear(context_mean)  # [batch_size, vocab_size]
        return out

# 5. Model Training
EMBEDDING_DIM = 16
model = CBOW(vocab_size, EMBEDDING_DIM)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(150):
    total_loss = 0
    for context, target in data:
        # Prepare inputs
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long).unsqueeze(0)
        target_idx = torch.tensor([word_to_ix[target]], dtype=torch.long)

        # Forward pass
        model.zero_grad()
        log_probs = model(context_idxs)
        loss = loss_function(log_probs, target_idx)

        # Backward pass & update
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

# 6. Extract Embeddings and Reduce Dimensions via PCA to 2D
# We pull the weight matrix directly from the embedding layer
word_vectors = model.embeddings.weight.detach().numpy()

pca = PCA(n_components=2)
coords = pca.fit_transform(word_vectors)

# 7. Plotting the results
plt.figure(figsize=(10, 8))
plt.scatter(coords[:, 0], coords[:, 1], edgecolors='k', c='skyblue')

# Annotate specific words to showcase clustering during the session
highlight_words = [
    'cat', 'dog', 'cute', # Animal cluster
    'coffee', 'espresso', 'latte', 'cappuccino', 'hot', # Drink cluster
    'python', 'code', 'pytorch', 'model', 'machine' # Tech cluster
]

for word in highlight_words:
    if word in word_to_ix:
        ix = word_to_ix[word]
        plt.scatter(coords[ix, 0], coords[ix, 1], color='red', s=50)
        plt.annotate(word, xy=(coords[ix, 0], coords[ix, 1]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', fontsize=12, weight='bold')

plt.title('Custom CBOW Word Embeddings 2D PCA Projection', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()