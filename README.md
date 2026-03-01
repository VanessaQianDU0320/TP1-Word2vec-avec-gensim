# README — Word Embeddings Project (Fouille de données textuelles) 

**Note:** Due to GitHub file size limitations, datasets and trained models are not included in this repository.  
Please download the datasets from:  
> - Movies Dataset: https://www.kaggle.com/rounakbanik/the-movies-dataset  
> - Cellphone Reviews Dataset: http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Cell_Phones_and_Accessories_5.json.gz


## Project Description

This project explores two major word embedding models:

Word2Vec (Gensim implementation)
GloVe (custom implementation based on the teacher’s GitHub notebook)

The objective is to train and compare word embeddings on the Movies dataset (used in TP1), and analyse the semantic structure produced by both approaches.
This project is developed as part of the course Fouille de données textuelles.

## Datasets

The two datasets used are:

- Movies Overview dataset
- Cell Phone Reviews dataset

The Cell Phone Reviews dataset from TP1 is kept for Word2Vec experiments but the main comparison in TP2 focuses on the Movies dataset.
--- 

The project is part of the course Fouille de données textuell


## Architecture

```
TP1/
│
├── data/
│   ├── movies_metadata.csv
│   └── Cell_Phones_and_Accessories_5.json
│
├── models/
│   ├── movies_w2v.model
│   └── cellphone_w2v.model
│   ├── movies_glove.npy
│   └── movies_glove_vocab.npy
│
├── movies_word2vec.py
├── cellphone_word2vec.py
├── movies_GloVe.py
├── Visualisation.py
├── main.py
│
├── rapport/
│   ├── word2vec_results.txt
│   ├── glove_results.txt
│   ├── word2vec.png
│   ├── glove.png
│   └── Rapport_TP Word2vec vs GloVe.pdf
│   └── Rapport TP Word2vec avec gensim.pdf
│
└── venv/

```
`movies_word2vec.py `
Contains the full pipeline for the Movies dataset:
- Reading data
- Preprocessing (cleaning and tokenization)
- Training Word2Vec model
- Saving the trained model
The trained model is saved as: `models/movies_w2v.model`

Same for `cellphone_word2vec.py`

`main.py`
Call 2 model if model exist in models, otherwise train new model. 
Test with the list of vocabulaire, print the similar word with percentage. 

## How to run 

Activate the virtual environment (if required):
`venv\Scripts\activate`
or
`.\venv\Scripts\python`
Then run:
`python main.py`

## Report 
The full analysis and theoretical explanation are provided in:
`rapport/Rapport TP Word2vec avec gensim.pdf`s
