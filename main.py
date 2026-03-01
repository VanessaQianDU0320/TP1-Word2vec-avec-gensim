from movies_word2vec import run_movies_pipeline
from cellphone_word2vec import run_cellphone_pipeline
from movies_GloVe import run_glove_pipeline
from gensim.models import Word2Vec
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Visualisation import plot_word2vec, plot_glove


#Avoid weste time in training each time
def load_or_train(model_path, train_function):

    if os.path.exists(model_path):
        print(f"Use existe model: {model_path}")
        model = Word2Vec.load(model_path)
    else:
        print(f"Training new model: {model_path}")
        model = train_function()

    return model

#Train or load GloVe Model
def load_or_train_glove(model_path, train_function):

    if os.path.exists(model_path):
        print(f"Use existing GloVe model: {model_path}")
        embeddings = np.load("models/movies_glove.npy")
        vocab = np.load("models/movies_glove_vocab.npy", allow_pickle=True).tolist()
    else:
        print(f"Training new GloVe model: {model_path}")
        embeddings, vocab = train_function()

    return embeddings, vocab


def test_keywords(model, words, model_name="Model",
                  save_path="rapport/word2vec_results.txt"):

    os.makedirs("rapport", exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:

        header = f"\nTest : {model_name}\n"
        print(header)
        f.write(header)

        for word in words:

            if word in model.wv:

                line = f"\nMost similar to '{word}':\n"
                print(line)
                f.write(line)

                results = model.wv.most_similar(word)

                for item in results:
                    result_line = f"{item[0]} {item[1]}\n"
                    print(result_line.strip())
                    f.write(result_line)

            else:
                line = f"\n'{word}' not in vocabulary.\n"
                print(line)
                f.write(line)

    print(f"Word2Vec results saved to {save_path}")

#Glove TEST
def most_similar_glove(word, embeddings, vocab,
                       topn=5,
                       save_path="rapport/glove_results.txt"):

    os.makedirs("rapport", exist_ok=True)

    with open(save_path, "a", encoding="utf-8") as f:

        header = f"\nMost similar to '{word}' (GloVe):\n"
        print(header)
        f.write(header)

        if word not in vocab:
            line = f"'{word}' not in vocabulary.\n"
            print(line)
            f.write(line)
            return

        word_index = vocab.index(word)
        word_vector = embeddings[word_index].reshape(1, -1)

        similarities = cosine_similarity(word_vector, embeddings)[0]
        top_indices = similarities.argsort()[-topn-1:-1][::-1]

        for idx in top_indices:
            result_line = f"{vocab[idx]} {similarities[idx]}\n"
            print(result_line.strip())
            f.write(result_line)

if __name__ == "__main__":
    model_movies = load_or_train(
        "models/movies_w2v.model",
        run_movies_pipeline
    )

    model_cellphone = load_or_train(
        "models/cellphone_w2v.model",
        run_cellphone_pipeline
    )

    movie_test_words = ["love", "war", "family", "death", "king"]
    cellphone_test_words = ["phone", "battery", "good", "screen", "price"]

    test_keywords(model_movies, movie_test_words,
              "Movies Word2Vec",
              save_path="rapport/word2vec_results_Movies.txt")
    test_keywords(model_cellphone, cellphone_test_words, "CellPhone ord2Vec",
              save_path="rapport/word2vec_results_Cellphone.txt")

    glove_embeddings, glove_vocab = load_or_train_glove(
    "models/movies_glove.npy",
    run_glove_pipeline
    )
    print("\n Test : Movies GloVe")


    open("rapport/glove_results.txt", "w").close()

    for word in movie_test_words:
        most_similar_glove(word,
                        glove_embeddings,
                        glove_vocab,
                        save_path="rapport/glove_results.txt")

    words_to_plot = [
    "love", "romance", "marriage",
    "war", "battle", "soldier",
    "king", "queen", "prince",
    "family", "father", "mother",
    "death", "murder", "crime"
    ]

    plot_word2vec(model_movies, words_to_plot, "Movies Word2Vec")
    plot_glove(glove_embeddings, glove_vocab, words_to_plot)

