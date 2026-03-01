import numpy as np
import tensorflow as tf
import pandas as pd
import re
from gensim.utils import simple_preprocess
from collections import Counter

from collections import defaultdict


def read_data(path="data/movies_metadata.csv"):
    df = pd.read_csv(path, low_memory=False)
    df = df[['overview']].dropna()
    return df

def prepare_data(df):
    def preprocess(text):
        text = re.sub(r'\W+', ' ', str(text))
        return simple_preprocess(text)

    sentences = df['overview'].apply(preprocess)
    return sentences

def build_vocab(sentences, min_count=50):
    all_words = [word for sentence in sentences for word in sentence]
    word_counts = Counter(all_words)
    
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    vocab_to_ix = {word: i for i, word in enumerate(vocab)}
    
    return vocab, vocab_to_ix

def build_cooccurrence(sentences, vocab_to_ix, window_size=5):
    cooccurrence = defaultdict(float)

    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in vocab_to_ix:
                continue

            center_id = vocab_to_ix[word]

            left = max(i - window_size, 0)
            right = min(i + window_size + 1, len(sentence))

            for j in range(left, right):
                if i == j:
                    continue

                context_word = sentence[j]
                if context_word not in vocab_to_ix:
                    continue

                context_id = vocab_to_ix[context_word]
                cooccurrence[(center_id, context_id)] += 1.0

    return cooccurrence

def train_glove(cooccurrence,
                vocab_size,
                embedding_size=100,
                iterations=3,
                learning_rate=0.001):

    center_embeddings = tf.Variable(
        tf.random.normal([vocab_size, embedding_size])
    )
    context_embeddings = tf.Variable(
        tf.random.normal([vocab_size, embedding_size])
    )

    bias_center = tf.Variable(tf.zeros([vocab_size]))
    bias_context = tf.Variable(tf.zeros([vocab_size]))

    optimizer = tf.keras.optimizers.Adam(learning_rate)

    pairs = list(cooccurrence.items())
    print("Number of cooccurrence pairs:", len(pairs))

    max_pairs = 30000   

    def weighting(x, x_max=100, alpha=0.75):
        return min(1.0, (x / x_max) ** alpha)

    for epoch in range(iterations):
        total_loss = 0

        for idx, ((i, j), x_ij) in enumerate(pairs):

            if idx >= max_pairs:
                break

            weight = weighting(x_ij)

            with tf.GradientTape() as tape:

                dot = tf.reduce_sum(center_embeddings[i] * context_embeddings[j])

                log_x = tf.math.log(1 + x_ij)

                loss = weight * tf.square(
                    dot + bias_center[i] + bias_context[j] - log_x
                )

            grads = tape.gradient(
                loss,
                [center_embeddings, context_embeddings,
                 bias_center, bias_context]
            )

            optimizer.apply_gradients(
                zip(grads,
                    [center_embeddings, context_embeddings,
                     bias_center, bias_context])
            )

            total_loss += loss.numpy()

            if idx % 2000 == 0:
                print(f"Epoch {epoch}, processed {idx} pairs")

        print(f"Epoch {epoch} finished, Loss {total_loss}")

    final_embeddings = (center_embeddings + context_embeddings).numpy()
    return final_embeddings

def save_glove_embeddings(embeddings, vocab):
    np.save("models/movies_glove.npy", embeddings)
    np.save("models/movies_glove_vocab.npy", np.array(vocab))

def run_glove_pipeline():
    df = read_data()
    sentences = prepare_data(df)
    sentences = sentences[:4000]
    vocab, vocab_to_ix = build_vocab(sentences)
    print("Vocab size:", len(vocab))
    cooccurrence = build_cooccurrence(sentences, vocab_to_ix)

    embeddings = train_glove(cooccurrence, len(vocab))

    save_glove_embeddings(embeddings, vocab)

    return embeddings, vocab