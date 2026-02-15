import json
import re
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec


# Read JSON 
def read_data(path="data/Cell_Phones_and_Accessories_5.json"):

    reviews = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if "reviewText" in data:
                reviews.append(data["reviewText"])

    return reviews

#Preprocess text
# Preprocess text
def prepare_data(reviews):

    def preprocess(text):
        text = re.sub(r'\W+', ' ', str(text))
        return simple_preprocess(text)

    sentences = [preprocess(review) for review in reviews]

    return sentences



#Train model
def train_model(sentences,
                vector_size=100,
                window=5,
                min_count=5,
                sg=1,
                epochs=20):

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs
    )

    return model


#run
def run_cellphone_pipeline():
    reviews = read_data()
    sentences = prepare_data(reviews)
    model = train_model(sentences)

    model.save("models/cellphone_w2v.model")
    print("\nMost similar to 'battery':")
    print(model.wv.most_similar("battery"))

    return model