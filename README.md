# README — Word2Vec Project (Fouille de données textuelles)

## Project Description

This project implements the Word2Vec model using the Gensim library in Python.
The objective is to train word embeddings on two different corpora and analyse the semantic quality of the learned representations.

## Datasets

The two datasets used are:

- Movies Overview dataset
- Cell Phone Reviews dataset

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
│
├── movies_word2vec.py
├── cellphone_word2vec.py
├── main.py
│
├── rapport/
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
Then run:
`python main.py`

## Report 
The full analysis and theoretical explanation are provided in:
`rapport/Rapport TP Word2vec avec gensim.pdf`
