import pandas as pd
import re
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec


#Read dataset
def read_data(path="data/movies_metadata.csv"):
    df = pd.read_csv(path, low_memory=False)
    df = df[['overview']].dropna()
    return df


#Preprocess text
def prepare_data(df):
    def preprocess(text):
        text = re.sub(r'\W+', ' ', str(text))
        return simple_preprocess(text)

    sentences = df['overview'].apply(preprocess)
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
def run_movies_pipeline():
    df = read_data()
    sentences = prepare_data(df)
    model = train_model(sentences)

    model.save("models/movies_w2v.model")

    print("\nMost similar to 'love':")
    print(model.wv.most_similar("love"))

    #print("\nMost similar to 'war':")
    #print(model.wv.most_similar("war"))

    return model