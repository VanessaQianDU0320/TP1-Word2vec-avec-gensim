import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import numpy as np

def plot_word2vec(model, words, save_path="rapport/word2vec.png"):

    vectors = []
    labels = []

    for word in words:
        if word in model.wv:
            vectors.append(model.wv[word])
            labels.append(word)

    if len(vectors) < 2:
        print("Not enough words to visualize.")
        return

    vectors = np.array(vectors)

    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, label)

    plt.title("Word2Vec Embedding")
    plt.tight_layout()

    os.makedirs("rapport", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"Word2Vec plot saved to {save_path}")


def plot_glove(embeddings, vocab, words, save_path="rapport/glove.png"):

    vectors = []
    labels = []

    for word in words:
        if word in vocab:
            idx = vocab.index(word)
            vectors.append(embeddings[idx])
            labels.append(word)

    if len(vectors) < 2:
        print("Not enough words to visualize.")
        return
    vectors = np.array(vectors)

    tsne = TSNE(n_components=2,
                random_state=42,
                perplexity=5)

    reduced = tsne.fit_transform(vectors)

    plt.figure(figsize=(8, 6))
    for i, label in enumerate(labels):
        plt.scatter(reduced[i, 0], reduced[i, 1])
        plt.text(reduced[i, 0] + 0.01, reduced[i, 1] + 0.01, label)

    plt.title("GloVe Embedding")
    plt.tight_layout()

    os.makedirs("rapport", exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    print(f"GloVe plot saved to {save_path}")