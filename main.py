from movies_word2vec import run_movies_pipeline
from cellphone_word2vec import run_cellphone_pipeline
from gensim.models import Word2Vec
import os

#Avoid weste time in training each time
def load_or_train(model_path, train_function):

    if os.path.exists(model_path):
        print(f"Use existe model: {model_path}")
        model = Word2Vec.load(model_path)
    else:
        print(f"Training new model: {model_path}")
        model = train_function()

    return model

def test_keywords(model, words, model_name="Model"):

    print(f"\n Test : {model_name}")

    for word in words:
        if word in model.wv:
            print(f"\nMost similar to '{word}':")
            print(model.wv.most_similar(word))
        else:
            print(f"\n'{word}' not in vocabulary.")



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

    test_keywords(model_movies, movie_test_words, "Movies")
    test_keywords(model_cellphone, cellphone_test_words, "CellPhone")


